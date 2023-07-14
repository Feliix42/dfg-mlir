/// Implements the dfg dialect ops.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#define DEBUG_TYPE "dfg-ops"

using namespace mlir;
using namespace mlir::dfg;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "dfg-mlir/Dialect/dfg/IR/Ops.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// OperatorOp
//===----------------------------------------------------------------------===//

void OperatorOp::build(
    OpBuilder &builder,
    OperationState &state,
    StringRef name,
    FunctionType type)
{

    state.addAttribute(
        SymbolTable::getSymbolAttrName(),
        builder.getStringAttr(name));
    state.addAttribute(
        getFunctionTypeAttrName(state.name),
        TypeAttr::get(type));
    state.addRegion();

    // FIXME(feliix42): addArgAndResultAttrs to register the function arguments
    // as block arguments (?) -> maybe also not needed, we'll see
}

// temporary workaround
bool OperatorOp::isExternal()
{
    Region &body = getRegion();
    return body.empty();
}

static ParseResult parseFunctionArgumentList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::Argument> &arguments)
{
    return parser.parseCommaSeparatedList(
        OpAsmParser::Delimiter::Paren,
        [&]() -> ParseResult {
            // // Handle ellipsis as a special case.
            // if (succeeded(parser.parseOptionalEllipsis())) {
            //   // This is a variadic designator.
            //   return failure(); // Stop parsing arguments.
            // }

            // Parse argument name if present.
            OpAsmParser::Argument argument;
            auto argPresent = parser.parseOptionalArgument(
                argument,
                /*allowType=*/true,
                /*allowAttrs=*/false);
            if (argPresent.has_value()) {
                if (failed(argPresent.value()))
                    return failure(); // Present but malformed.

                // Reject this if the preceding argument was missing a name.
                if (!arguments.empty() && arguments.back().ssaName.name.empty())
                    return parser.emitError(
                        argument.ssaName.location,
                        "expected type instead of SSA identifier");

            } else {
                argument.ssaName.location = parser.getCurrentLocation();
                // Otherwise we just have a type list without SSA names.  Reject
                // this if the preceding argument had a name.
                if (!arguments.empty()
                    && !arguments.back().ssaName.name.empty())
                    return parser.emitError(
                        argument.ssaName.location,
                        "expected SSA identifier");

                NamedAttrList attrs;
                if (parser.parseType(argument.type)
                    || parser.parseOptionalAttrDict(attrs)
                    || parser.parseOptionalLocationSpecifier(
                        argument.sourceLoc))
                    return failure();
                argument.attrs = attrs.getDictionary(parser.getContext());
            }
            arguments.push_back(argument);
            return success();
        });
}

ParseResult OperatorOp::parse(OpAsmParser &parser, OperationState &result)
{
    auto &builder = parser.getBuilder();

    // parse the operator name
    StringAttr nameAttr;
    if (parser.parseSymbolName(
            nameAttr,
            SymbolTable::getSymbolAttrName(),
            result.attributes))
        return failure();

    // parse the signature of the operator
    SmallVector<OpAsmParser::Argument> arguments, outputArgs;
    SMLoc signatureLocation = parser.getCurrentLocation();

    // parse inputs/outputs separately for later distinction
    if (succeeded(parser.parseOptionalKeyword("inputs"))) {
        if (parseFunctionArgumentList(parser, arguments)) return failure();
    }

    if (succeeded(parser.parseOptionalKeyword("outputs"))) {
        if (parseFunctionArgumentList(parser, outputArgs)) return failure();
    }

    SmallVector<Type> argTypes;
    argTypes.reserve(arguments.size());
    SmallVector<Type> resultTypes;
    resultTypes.reserve(outputArgs.size());

    for (auto &arg : arguments) argTypes.push_back(arg.type);
    for (auto &arg : outputArgs) resultTypes.push_back(arg.type);
    Type type = builder.getFunctionType(argTypes, resultTypes);

    if (!type) {
        return parser.emitError(signatureLocation)
               << "Failed to construct operator type";
    }

    result.addAttribute(
        getFunctionTypeAttrName(result.name),
        TypeAttr::get(type));

    // merge both argument lists for the block arguments
    arguments.append(outputArgs);

    if (parser.parseOptionalAttrDict(result.attributes)) return failure();

    auto* body = result.addRegion();
    SMLoc loc = parser.getCurrentLocation();
    OptionalParseResult parseResult = parser.parseOptionalRegion(
        *body,
        arguments,
        /*enableNameShadowing=*/false);

    if (parseResult.has_value()) {
        if (failed(*parseResult)) return failure();
        if (body->empty())
            return parser.emitError(loc, "expected non-empty operator body");
    }
    return success();
}

void OperatorOp::print(OpAsmPrinter &p)
{
    Operation* op = getOperation();
    Region &body = op->getRegion(0);

    // print the operation and function name
    auto funcName =
        op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
            .getValue();

    p << ' ';
    p.printSymbolName(funcName);
    p << ' ';

    ArrayRef<Type> inputTypes = getFunctionType().getInputs();
    ArrayRef<Type> outputTypes = getFunctionType().getResults();

    // // NOTE(feliix42): Is this even necessary? There's no attributes
    // supported afaik FunctionOpInterface fu =
    // llvm::cast<FunctionOpInterface>(op); ArrayAttr argAttrs =
    // fu.getArgAttrsAttr();

    if (!inputTypes.empty()) {
        p << " inputs (";
        for (unsigned i = 0; i < inputTypes.size(); i++) {
            if (i > 0) p << ", ";

            ArrayRef<NamedAttribute> attrs;
            // if(argAttrs)
            //     attrs = argAttrs[i].cast<DictionaryAttr>().getValue();

            p.printRegionArgument(body.getArgument(i), attrs);
        }
        p << ')';
    }

    if (!outputTypes.empty()) {
        p << " outputs (";
        unsigned inpSize = inputTypes.size();
        for (unsigned i = inpSize; i < outputTypes.size() + inpSize; i++) {
            if (i > inpSize) p << ", ";

            ArrayRef<NamedAttribute> attrs;
            // if(argAttrs)
            //     attrs = argAttrs[i].cast<DictionaryAttr>().getValue();

            p.printRegionArgument(body.getArgument(i), attrs);
        }
        p << ')';
    }

    if (!op->getAttrs().empty()) {
        // NOTE(feliix42): Might needs a list of elided attrs -> inputs/...
        p.printOptionalAttrDict(op->getAttrs());
    }

    // Print the region
    if (!body.empty()) {
        p << ' ';
        p.printRegion(
            body,
            /*printEntryBlockArgs =*/false,
            /*printBlockTerminators =*/true);
    }
}

LogicalResult OperatorOp::verify()
{
    auto inputsType = getInputTypes();
    for (const auto inTy : inputsType)
        if (!inTy.isa<OutputType>())
            return emitOpError("requires OutputType for input ports");

    auto outputsType = getOutputTypes();
    for (const auto outTy : outputsType)
        if (!outTy.isa<InputType>())
            return emitOpError("requires InputType for output ports");

    return success();
}

//===----------------------------------------------------------------------===//
// ChannelOp
//===----------------------------------------------------------------------===//

ParseResult ChannelOp::parse(OpAsmParser &parser, OperationState &result)
{
    if (failed(parser.parseLess())) return failure();

    Type ty;
    if (parser.parseType(ty)) return failure();

    result.addAttribute(
        getEncapsulatedTypeAttrName(result.name),
        TypeAttr::get(ty));

    SmallVector<Type> results;
    InputType inp = InputType::get(ty.getContext(), ty);
    results.push_back(inp);
    OutputType out = OutputType::get(ty.getContext(), ty);
    results.push_back(out);

    // TODO(feliix42): Verify correctness: Is `getInChan` and `getOutChan`
    // yielding the expected results??
    result.addTypes(results);

    if (succeeded(parser.parseOptionalComma())) {
        int size = 0;
        if (parser.parseInteger(size)) return failure();
        MLIRContext context;
        result.addAttribute(
            getBufferSizeAttrName(result.name),
            parser.getBuilder().getI32IntegerAttr(size));
    }

    return parser.parseGreater();
}

void ChannelOp::print(OpAsmPrinter &p)
{
    p << '<';
    p.printType(getEncapsulatedType());
    if (const auto size = getBufferSize()) p << ',' << size;
    p << '>';
}

//===----------------------------------------------------------------------===//
// InstantiateOp
//===----------------------------------------------------------------------===//

constexpr char kOperandSegmentSizesAttr[] = "operand_segment_sizes";

ParseResult InstantiateOp::parse(OpAsmParser &parser, OperationState &result)
{
    // parse operator name
    StringAttr calleeAttr;
    if (parser.parseSymbolName(
            calleeAttr,
            getCalleeAttrName(result.name),
            result.attributes))
        return failure();

    // parse the operator inputs and outpus
    SmallVector<OpAsmParser::UnresolvedOperand, 4> inputs;
    if (succeeded(parser.parseOptionalKeyword("inputs"))) {
        if (parser.parseLParen() || parser.parseOperandList(inputs) ||
            // parser.resolveOperands(inputs, opTy, result.operands)) {
            parser.parseRParen())
            return failure();
    }

    SmallVector<OpAsmParser::UnresolvedOperand, 4> outputs;
    if (succeeded(parser.parseOptionalKeyword("outputs"))) {
        if (parser.parseLParen() || parser.parseOperandList(outputs)
            || parser.parseRParen())
            return failure();
    }

    if (parser.parseColon()) return failure();

    // parse the signature & resolve the input/output types
    FunctionType signature;
    if (parser.parseType(signature)) return failure();

    ArrayRef<Type> inpTypes = signature.getInputs();
    ArrayRef<Type> outTypes = signature.getResults();
    SMLoc location = parser.getCurrentLocation();

    int32_t numInputs = inpTypes.size();
    int32_t numOutputs = outTypes.size();

    if (inpTypes.size() != inputs.size() || outTypes.size() != outputs.size()) {
        parser.emitError(
            location,
            "Call signature does not match operand count");
    }

    if (parser.resolveOperands(inputs, inpTypes, location, result.operands))
        return failure();

    if (parser.resolveOperands(outputs, outTypes, location, result.operands))
        return failure();

    // Add derived `operand_segment_sizes` attribute based on parsed operands.
    auto operandSegmentSizes =
        parser.getBuilder().getDenseI32ArrayAttr({numInputs, numOutputs});
    result.addAttribute(kOperandSegmentSizesAttr, operandSegmentSizes);

    return success();
}

void InstantiateOp::print(OpAsmPrinter &p)
{
    // callee
    p.printAttributeWithoutType(getCalleeAttr());

    // print `inputs (...)` if existent
    if (!getInputs().empty()) p << "inputs (" << getInputs() << ")";

    // print `outputs (...)` if existent
    if (!getOutputs().empty()) p << "outputs (" << getOutputs() << ")";

    // signature
    p << ":";
    p.printFunctionalType(getInputs().getTypes(), getOutputs().getTypes());
}

//===----------------------------------------------------------------------===//
// KernelOp
//===----------------------------------------------------------------------===//

ParseResult KernelOp::parse(OpAsmParser &parser, OperationState &result)
{
    // parse operator name
    StringAttr calleeAttr;
    if (parser.parseSymbolName(
            calleeAttr,
            getCalleeAttrName(result.name),
            result.attributes))
        return failure();

    // parse the operator inputs and outpus
    SmallVector<OpAsmParser::UnresolvedOperand, 4> inputs;
    if (succeeded(parser.parseOptionalKeyword("inputs"))) {
        if (parser.parseLParen() || parser.parseOperandList(inputs) ||
            // parser.resolveOperands(inputs, opTy, result.operands)) {
            parser.parseRParen())
            return failure();
    }

    SmallVector<OpAsmParser::UnresolvedOperand, 4> outputs;
    if (succeeded(parser.parseOptionalKeyword("outputs"))) {
        if (parser.parseLParen() || parser.parseOperandList(outputs)
            || parser.parseRParen())
            return failure();
    }

    if (parser.parseColon()) return failure();

    // parse the signature & resolve the input/output types
    FunctionType signature;
    if (parser.parseType(signature)) return failure();

    ArrayRef<Type> inpTypes = signature.getInputs();
    ArrayRef<Type> outTypes = signature.getResults();
    SMLoc location = parser.getCurrentLocation();

    int32_t numInputs = inpTypes.size();
    int32_t numOutputs = outTypes.size();

    if (inpTypes.size() != inputs.size() || outTypes.size() != outputs.size()) {
        parser.emitError(
            location,
            "Call signature does not match operand count");
    }

    if (parser.resolveOperands(inputs, inpTypes, location, result.operands))
        return failure();

    if (parser.resolveOperands(outputs, outTypes, location, result.operands))
        return failure();

    // Add derived `operand_segment_sizes` attribute based on parsed operands.
    auto operandSegmentSizes =
        parser.getBuilder().getDenseI32ArrayAttr({numInputs, numOutputs});
    result.addAttribute(kOperandSegmentSizesAttr, operandSegmentSizes);

    return success();
}

void KernelOp::print(OpAsmPrinter &p)
{
    // callee
    p.printAttributeWithoutType(getCalleeAttr());

    // print `inputs (...)` if existent
    if (!getInputs().empty()) p << "inputs(" << getInputs() << ")";

    // print `outputs (...)` if existent
    if (!getOutputs().empty()) p << "outputs(" << getOutputs() << ")";

    // signature
    p << ":";
    p.printFunctionalType(getInputs().getTypes(), getOutputs().getTypes());
}

//===----------------------------------------------------------------------===//
// DfgDialect
//===----------------------------------------------------------------------===//

void DfgDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "dfg-mlir/Dialect/dfg/IR/Ops.cpp.inc"
        >();
}
