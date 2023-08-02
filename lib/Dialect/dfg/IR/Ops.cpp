/// Implements the dfg dialect ops.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "dfg-mlir/Dialect/dfg/IR/Ops.h"

#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
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

// -> for multiple variadic attributes
constexpr char kOperandSegmentSizesAttr[] = "operand_segment_sizes";

//===----------------------------------------------------------------------===//
// OperatorOp
//===----------------------------------------------------------------------===//

// void OperatorOp::build(
//     OpBuilder &builder,
//     OperationState &state,
//     StringRef name,
//     ValueRange inputs,
//     ValueRange outputs)
// {
//     state.addAttribute(
//         SymbolTable::getSymbolAttrName(),
//         builder.getStringAttr(name));
//     state.addOperands(inputs);
//     state.addOperands(outputs);

//     int32_t numInputs = inputs.size();
//     int32_t numOutputs = outputs.size();

//     // set sizes of the inputs and outputs
//     auto operandSegmentSizes =
//         builder.getDenseI32ArrayAttr({numInputs, numOutputs});
//     state.addAttribute(kOperandSegmentSizesAttr, operandSegmentSizes);

//     state.addRegion();
// }

/// @brief  Returns whether the operator is externally defined, i.e., has no
/// body.
/// @return `true` if there is no body attached, `false` the operator has a
/// body.
bool OperatorOp::isExternal()
{
    Region &body = getRegion();
    return body.empty();
}

/// @brief Parses a function argument list for inputs or outputs
/// @tparam T The class of the argument. Must be either InputType or OutputType
/// @param parser The currently used parser
/// @param arguments A list of arguments to parse
/// @return A parse result indicating success or failure to parse.
template<typename T>
static ParseResult parseChannelArgumentList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::Argument> &arguments)
{
    return parser.parseCommaSeparatedList(
        OpAsmParser::Delimiter::Paren,
        [&]() -> ParseResult {
            // TODO(feliix42): I think this whole thing needs a redesign. What I
            // need at the end are `ValueRange`s for both inputs and outputs.
            // These are then going to be fed into the builer in the parser
            // (rewrite this as well, remove manual construction in favor of
            // calling the Builder with all necessary information!).
            // So what I could see myself (painfully) doing for each argument
            // is:
            // 1. Parse the operand (just the name)
            // 2. parse the colon
            // 3. Parse the type
            //    ----------------- Push both into a list
            // 4. resolve all operands at the end with the list and some bs.
            // location info

            // OpAsmParser::UnresolvedOperand opname;
            // mlir::Type ty;
            // if (parser.parseOperand(opname) || parser.parseColon()
            //     || parser.parseType(ty))
            //     return failure();

            // ty = T::get(ty.getContext(), ty);

            OpAsmParser::Argument argument;
            if (parser.parseArgument(
                    argument,
                    /*allowType=*/true,
                    /*allowAttrs=*/false))
                return failure();

            argument.type = T::get(argument.type.getContext(), argument.type);
            arguments.push_back(argument);

            // additionally store the result as Value
            // return parser.resolveOperand(
            //     argument.ssaName,
            //     argument.type,
            //     values);

            // auto argPresent = parser.parseOptionalArgument(
            //     argument,
            //     /*allowType=*/true,
            //     /*allowAttrs=*/false);
            // if (argPresent.has_value()) {
            //     if (failed(argPresent.value()))
            //         return failure(); // Present but malformed.

            //     // Reject this if the preceding argument was missing a name.
            //     if (!arguments.empty() &&
            //     arguments.back().ssaName.name.empty())
            //         return parser.emitError(
            //             argument.ssaName.location,
            //             "expected type instead of SSA identifier");
            // } else {
            //     argument.ssaName.location = parser.getCurrentLocation();
            //     // Otherwise we just have a type list without SSA names.
            //     Reject
            //     // this if the preceding argument had a name.
            //     if (!arguments.empty()
            //         && !arguments.back().ssaName.name.empty())
            //         return parser.emitError(
            //             argument.ssaName.location,
            //             "expected SSA identifier");

            //     NamedAttrList attrs;
            //     if (parser.parseType(argument.type)
            //         || parser.parseOptionalAttrDict(attrs)
            //         || parser.parseOptionalLocationSpecifier(
            //             argument.sourceLoc))
            //         return failure();
            //     argument.attrs = attrs.getDictionary(parser.getContext());
            // }

            // argument.type = T::get(argument.type.getContext(),
            // argument.type);

            // arguments.push_back(argument);
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
            getSymNameAttrName(result.name),
            result.attributes))
        return failure();

    // parse the signature of the operator
    SmallVector<OpAsmParser::Argument> inVals, outVals;

    SMLoc signatureLocation = parser.getCurrentLocation();

    // parse inputs/outputs separately for later distinction
    if (succeeded(parser.parseOptionalKeyword("inputs"))) {
        if (parseChannelArgumentList<OutputType>(parser, inVals))
            return failure();
    }

    if (succeeded(parser.parseOptionalKeyword("outputs"))) {
        if (parseChannelArgumentList<InputType>(parser, outVals))
            return failure();
    }

    // int32_t numInputs = inVals.size();
    // int32_t numOutputs = outVals.size();

    SmallVector<Type> argTypes;
    argTypes.reserve(inVals.size());
    SmallVector<Type> resultTypes;
    resultTypes.reserve(outVals.size());

    // set sizes of the inputs and outputs
    // auto operandSegmentSizes =
    //     builder.getDenseI32ArrayAttr({numInputs, numOutputs});
    // result.addAttribute(kOperandSegmentSizesAttr, operandSegmentSizes);

    for (auto &arg : inVals) argTypes.push_back(arg.type);
    for (auto &arg : outVals) resultTypes.push_back(arg.type);
    Type type = builder.getFunctionType(argTypes, resultTypes);

    if (!type) {
        return parser.emitError(signatureLocation)
               << "Failed to construct operator type";
    }

    result.addAttribute(
        getFunctionTypeAttrName(result.name),
        TypeAttr::get(type));

    // if (parser.addTypesToList(argTypes, result.types)
    //     || parser.addTypesToList(resultTypes, result.types))
    //     return failure();

    // merge both argument lists for the block arguments
    inVals.append(outVals);

    // if (parser.parseOptionalAttrDict(result.attributes)) return failure();

    // OpBuilder opbuilder(parser.getContext());
    // build(opbuilder, result, nameAttr, TypeAttr::get(type));

    OptionalParseResult attrResult =
        parser.parseOptionalAttrDictWithKeyword(result.attributes);
    if (attrResult.has_value() && failed(*attrResult)) return failure();

    // OptionalParseResult parseResult =
    //     parser.parseOptionalRegion(result.regions[0]);
    // SMLoc loc = parser.getCurrentLocation();
    // if (parseResult.has_value()) {
    //     if (failed(*parseResult)) return failure();
    //     if (result.regions[0]->empty())
    //         return parser.emitError(loc, "expected non-empty operator body");
    // }

    // for (auto &arg : arguments)
    //     result.regions[0]->addArgument(
    //         arg.type,
    //         parser.getEncodedSourceLoc(arg.ssaName.location));
    auto* body = result.addRegion();
    SMLoc loc = parser.getCurrentLocation();
    OptionalParseResult parseResult = parser.parseOptionalRegion(
        *body,
        inVals,
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

    if (!inputTypes.empty()) {
        p << " inputs (";
        for (unsigned i = 0; i < inputTypes.size(); i++) {
            if (i > 0) p << ", ";

            ArrayRef<NamedAttribute> attrs;
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
            p.printRegionArgument(body.getArgument(i), attrs);
        }
        p << ')';
    }

    // ValueRange inputTypes = getInputs();
    // ValueRange outputTypes = getOutputs();

    // // // NOTE(feliix42): Is this even necessary? There's no attributes
    // // supported afaik FunctionOpInterface fu =
    // // llvm::cast<FunctionOpInterface>(op); ArrayAttr argAttrs =
    // // fu.getArgAttrsAttr();

    // if (!inputTypes.empty()) {
    //     p << " inputs (" << inputTypes << ")";
    //     // for (unsigned i = 0; i < inputTypes.size(); i++) {
    //     //     if (i > 0) p << ", ";

    //     //     BlockArgument arg = body.getArgument(i);
    //     //     p.printOperand(arg);
    //     //     p << ": ";
    //     // p.printType(arg.getType().cast<OutputType>().getElementType());
    //     // }
    //     // p << ')';
    // }

    // if (!outputTypes.empty()) {
    //     p << " outputs (" << outputTypes << ")";
    //     // unsigned inpSize = inputTypes.size();
    //     // for (unsigned i = inpSize; i < outputTypes.size() + inpSize; i++)
    //     {
    //     //     if (i > inpSize) p << ", ";

    //     //     BlockArgument arg = body.getArgument(i);
    //     //     p.printOperand(arg);
    //     //     p << ": ";
    //     //     p.printType(arg.getType().cast<InputType>().getElementType());
    //     // }
    //     // p << ')';
    // }

    if (!op->getAttrs().empty())
        p.printOptionalAttrDictWithKeyword(op->getAttrs());

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
    // NOTE(feliix42): This check failed to compile but it should be enforced by
    // type bounds now.

    // // Check if all the input are OutputType
    // // and all output are InputType
    // auto inputsType = getInputs();
    // for (const auto inTy : inputsType)
    //     if (!inTy.isa<OutputType>())
    //         return emitOpError("requires OutputType for input ports");

    // auto outputsType = getOutputs();
    // for (const auto outTy : outputsType)
    //     if (!outTy.isa<InputType>())
    //         return emitOpError("requires InputType for output ports");

    // If there is a LoopOp, it must be the first op in the body
    if (!getBody().empty()) {
        auto ops = getBody().getOps();
        bool isFirstLoop, hasLoop = false;
        for (const auto &op : ops)
            if (auto loopOp = dyn_cast<LoopOp>(op)) hasLoop = true;
        if (auto loopOp = dyn_cast<LoopOp>(&getBody().front().front()))
            isFirstLoop = true;
        if (hasLoop && !isFirstLoop)
            return emitError("The LoopOp must be the first op of Operator");
    }

    return success();
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

ParseResult LoopOp::parse(OpAsmParser &parser, OperationState &result)
{
    if (parser.parseLParen()) return failure();

    SmallVector<OpAsmParser::UnresolvedOperand> operands;
    SmallVector<Type> types;
    if (parser.parseOperandList(operands)) return failure();
    if (parser.parseColonTypeList(types)) return failure();

    assert(operands.size() == types.size());

    for (size_t i = 0; i < operands.size(); i++)
        if (parser.resolveOperand(operands[i], types[i], result.operands))
            return failure();

    if (parser.parseRParen()) return failure();

    Region* body = result.addRegion();
    if (parser.parseRegion(*body, {}, {})) return failure();

    return success();
}

void LoopOp::print(OpAsmPrinter &p)
{
    assert(!getOperands().empty());

    p << '(';
    p << getOperands();
    p << ')';

    Region &body = getBody();
    if (!body.empty()) {
        p << ' ';
        p.printRegion(
            body,
            /*printEntryBlockArgs =*/false,
            /*printBlockTerminators =*/true);
    }
}

//===----------------------------------------------------------------------===//
// ChannelOp
//===----------------------------------------------------------------------===//

ParseResult ChannelOp::parse(OpAsmParser &parser, OperationState &result)
{
    if (failed(parser.parseLParen())) return failure();

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

    result.addTypes(results);

    if (succeeded(parser.parseOptionalComma())) {
        int size = 0;
        if (parser.parseInteger(size)) return failure();
        MLIRContext context;
        result.addAttribute(
            getBufferSizeAttrName(result.name),
            parser.getBuilder().getI32IntegerAttr(size));
    }

    return parser.parseRParen();
}

void ChannelOp::print(OpAsmPrinter &p)
{
    p << '<';
    p.printType(getEncapsulatedType());
    if (const auto size = getBufferSize()) p << ", " << size;
    p << '>';
}

//===----------------------------------------------------------------------===//
// InstantiateOp
//===----------------------------------------------------------------------===//

ParseResult InstantiateOp::parse(OpAsmParser &parser, OperationState &result)
{
    // optionally mark as `offloaded`
    if (succeeded(parser.parseOptionalKeyword("offloaded")))
        result.addAttribute(
            getOffloadedAttrName(result.name),
            BoolAttr::get(parser.getContext(), true));

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
    SMLoc location = parser.getCurrentLocation();
    FunctionType signature;
    if (parser.parseType(signature)) return failure();

    ArrayRef<Type> inpTypes = signature.getInputs();
    ArrayRef<Type> outTypes = signature.getResults();
    int32_t numInputs = inpTypes.size();
    int32_t numOutputs = outTypes.size();

    SmallVector<Type> inChanTypes;
    SmallVector<Type> outChanTypes;
    for (auto &inp : inpTypes)
        inChanTypes.push_back(OutputType::get(inp.getContext(), inp));
    for (auto &out : outTypes)
        outChanTypes.push_back(InputType::get(out.getContext(), out));

    if (inChanTypes.size() != inputs.size()
        || outChanTypes.size() != outputs.size()) {
        parser.emitError(
            location,
            "Call signature does not match operand count");
    }

    if (parser.resolveOperands(inputs, inChanTypes, location, result.operands))
        return failure();

    if (parser
            .resolveOperands(outputs, outChanTypes, location, result.operands))
        return failure();

    // Add derived `operand_segment_sizes` attribute based on parsed
    // operands.
    auto operandSegmentSizes =
        parser.getBuilder().getDenseI32ArrayAttr({numInputs, numOutputs});
    result.addAttribute(kOperandSegmentSizesAttr, operandSegmentSizes);

    return success();
}

void InstantiateOp::print(OpAsmPrinter &p)
{
    // offloaded?
    if (getOffloaded()) p << " offloaded";

    // callee
    p << ' ';
    p.printAttributeWithoutType(getCalleeAttr());

    // print `inputs (...)` if existent
    if (!getInputs().empty()) p << " inputs (" << getInputs() << ")";

    // print `outputs (...)` if existent
    if (!getOutputs().empty()) p << " outputs (" << getOutputs() << ")";

    // signature
    SmallVector<Type> inpChans(getInputs().size());
    SmallVector<Type> outChans(getOutputs().size());

    for (auto in : getInputs().getTypes())
        inpChans.push_back(in.cast<OutputType>().getElementType());
    for (auto out : getOutputs().getTypes())
        outChans.push_back(out.cast<InputType>().getElementType());

    p << " : ";
    p.printFunctionalType(inpChans, outChans);
}

//===----------------------------------------------------------------------===//
// PullOp
//===----------------------------------------------------------------------===//

ParseResult PullOp::parse(OpAsmParser &parser, OperationState &result)
{
    OpAsmParser::UnresolvedOperand inputChan;
    Type dataTy;

    if (parser.parseOperand(inputChan) || parser.parseColon()
        || parser.parseType(dataTy))
        return failure();

    result.addTypes(dataTy);

    Type channelTy = OutputType::get(dataTy.getContext(), dataTy);
    if (parser.resolveOperand(inputChan, channelTy, result.operands))
        return failure();

    return success();
}

void PullOp::print(OpAsmPrinter &p)
{
    p << " " << getChan() << " : " << getType();
}

//===----------------------------------------------------------------------===//
// PushOp
//===----------------------------------------------------------------------===//

ParseResult PushOp::parse(OpAsmParser &parser, OperationState &result)
{
    OpAsmParser::UnresolvedOperand inp;
    OpAsmParser::UnresolvedOperand outputChan;
    Type dataTy;

    if (parser.parseLParen() || parser.parseOperand(inp) || parser.parseRParen()
        || parser.parseOperand(outputChan) || parser.parseColon()
        || parser.parseType(dataTy))
        return failure();

    Type channelTy = InputType::get(dataTy.getContext(), dataTy);
    if (parser.resolveOperand(inp, dataTy, result.operands)
        || parser.resolveOperand(outputChan, channelTy, result.operands))
        return failure();

    return success();
}

void PushOp::print(OpAsmPrinter &p)
{
    p << " (" << getInp() << ") " << getChan() << " : " << getInp().getType();
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
