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
            OpAsmParser::Argument argument;
            if (parser.parseArgument(
                    argument,
                    /*allowType=*/true,
                    /*allowAttrs=*/false))
                return failure();

            argument.type = T::get(argument.type.getContext(), argument.type);
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

    SmallVector<Type> argTypes, resultTypes;
    argTypes.reserve(inVals.size());
    resultTypes.reserve(outVals.size());

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

    // merge both argument lists for the block arguments
    inVals.append(outVals);

    OptionalParseResult attrResult =
        parser.parseOptionalAttrDictWithKeyword(result.attributes);
    if (attrResult.has_value() && failed(*attrResult)) return failure();

    // parse the attached region, if any
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
        p << "inputs (";
        for (unsigned i = 0; i < inputTypes.size(); i++) {
            if (i > 0) p << ", ";

            p.printOperand(body.getArgument(i));
            p << " : " << inputTypes[i].cast<OutputType>().getElementType();
        }
        p << ") ";
    }

    if (!outputTypes.empty()) {
        p << "outputs (";
        unsigned inpSize = inputTypes.size();
        for (unsigned i = inpSize; i < outputTypes.size() + inpSize; i++) {
            if (i > inpSize) p << ", ";

            p.printOperand(body.getArgument(i));
            p << " : "
              << outputTypes[i - inpSize].cast<InputType>().getElementType();
        }
        p << ") ";
    }

    // print any attributes in the attribute list into the dict
    if (!op->getAttrs().empty())
        p.printOptionalAttrDictWithKeyword(
            op->getAttrs(),
            /*elidedAttrs=*/{getFunctionTypeAttrName(), getSymNameAttrName()});

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
    // parse inputs/outputs
    SmallVector<OpAsmParser::Argument> inVals, outVals;

    // parse inputs/outputs separately for later distinction
    SMLoc inputLocation = parser.getCurrentLocation();
    if (succeeded(parser.parseOptionalKeyword("inputs"))) {
        if (parseChannelArgumentList<OutputType>(parser, inVals))
            return failure();
    }

    SMLoc outputLocation = parser.getCurrentLocation();
    if (succeeded(parser.parseOptionalKeyword("outputs"))) {
        if (parseChannelArgumentList<InputType>(parser, outVals))
            return failure();
    }

    SmallVector<OpAsmParser::UnresolvedOperand, 4> inputs, outputs;
    SmallVector<Type> argTypes, resultTypes;
    int32_t numInputs = inVals.size();
    int32_t numOutputs = outVals.size();
    argTypes.reserve(numInputs);
    inputs.reserve(numInputs);
    resultTypes.reserve(numOutputs);
    outputs.reserve(numOutputs);

    for (auto &arg : inVals) {
        argTypes.push_back(arg.type);
        inputs.push_back(arg.ssaName);
    }
    for (auto &arg : outVals) {
        resultTypes.push_back(arg.type);
        outputs.push_back(arg.ssaName);
    }

    if (parser
            .resolveOperands(inputs, argTypes, inputLocation, result.operands))
        return failure();

    if (parser.resolveOperands(
            outputs,
            resultTypes,
            outputLocation,
            result.operands))
        return failure();

    // Add derived `operand_segment_sizes` attribute based on parsed
    // operands.
    auto operandSegmentSizes =
        parser.getBuilder().getDenseI32ArrayAttr({numInputs, numOutputs});
    result.addAttribute(kOperandSegmentSizesAttr, operandSegmentSizes);

    Region* body = result.addRegion();
    if (parser.parseRegion(*body, {}, {})) return failure();

    return success();
}

void LoopOp::print(OpAsmPrinter &p)
{
    assert(!getOperands().empty());

    // print `inputs (...)` if existent
    Operation::operand_range inputs = getInChans();
    if (!inputs.empty()) {
        p << " inputs (";
        for (unsigned i = 0; i < inputs.size(); i++) {
            if (i > 0) p << ", ";
            p.printOperand(inputs[i]);
            p << " : ";
            p.printType(inputs[i]
                            .getImpl()
                            ->getType()
                            .cast<OutputType>()
                            .getElementType());
        }
        p << ")";
    }

    // print `outputs (...)` if existent
    Operation::operand_range outputs = getOutChans();
    if (!outputs.empty()) {
        p << " outputs (";
        for (unsigned i = 0; i < outputs.size(); i++) {
            if (i > 0) p << ", ";
            p.printOperand(outputs[i]);
            p << " : ";
            p.printType(outputs[i]
                            .getImpl()
                            ->getType()
                            .cast<InputType>()
                            .getElementType());
        }
        p << ")";
    }

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

    // parse an optional channel size
    int size = 0;
    OptionalParseResult sizeResult = parser.parseOptionalInteger(size);
    if (sizeResult.has_value()) {
        if (succeeded(*sizeResult)) {
            MLIRContext context;
            result.addAttribute(
                getBufferSizeAttrName(result.name),
                parser.getBuilder().getI32IntegerAttr(size));
        } else {
            return failure();
        }
    }

    if (parser.parseRParen() || parser.parseColon()) return failure();

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

    return success();
}

void ChannelOp::print(OpAsmPrinter &p)
{
    p << '(';
    if (const auto size = getBufferSize()) p << size;
    p << ") : ";

    p.printType(getEncapsulatedType());
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
    if (parser.parseSymbolName(calleeAttr)) return failure();

    result.addAttribute(
        getCalleeAttrName(result.name),
        SymbolRefAttr::get(calleeAttr));

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
    SmallVector<Type> inpChans, outChans;

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
