/// Implements the dfg dialect ops.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "dfg-mlir/Dialect/dfg/IR/Ops.h"

#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"

#define DEBUG_TYPE "dfg-ops"

using namespace mlir;
using namespace mlir::dfg;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "dfg-mlir/Dialect/dfg/IR/Ops.cpp.inc"

//===----------------------------------------------------------------------===//

// -> for multiple variadic attributes
constexpr char kOperandSegmentSizesAttr[] = "operandSegmentSizes";

/// @brief Parses a function argument list for inputs or outputs
/// @tparam T The class of the argument. Must be either InputType or OutputType
/// @param parser The currently used parser
/// @param arguments A list of arguments to parse
/// @return A parse result indicating success or failure to parse.
struct TypeId {
    static Type get(MLIRContext*, Type t) { return t; }
};
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

//===----------------------------------------------------------------------===//
// RegionOp
//===----------------------------------------------------------------------===//

void RegionOp::build(
    OpBuilder &builder,
    OperationState &state,
    StringRef name,
    FunctionType function_type)
{
    state.addAttribute(
        SymbolTable::getSymbolAttrName(),
        builder.getStringAttr(name));
    state.addAttribute(
        RegionOp::getFunctionTypeAttrName(state.name),
        TypeAttr::get(function_type));

    Region* region = state.addRegion();
    Block* body = new Block();
    region->push_back(body);

    SmallVector<Type> blockArgTypes;
    blockArgTypes.append(
        function_type.getInputs().begin(),
        function_type.getInputs().end());
    blockArgTypes.append(
        function_type.getResults().begin(),
        function_type.getResults().end());
    body->addArguments(
        blockArgTypes,
        SmallVector<Location>(blockArgTypes.size(), builder.getUnknownLoc()));
}

ParseResult RegionOp::parse(OpAsmParser &parser, OperationState &result)
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
    OptionalParseResult parseResult = parser.parseRegion(
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

void RegionOp::print(OpAsmPrinter &p)
{
    Operation* op = getOperation();
    Region &body = op->getRegion(0);
    bool isExternal = body.empty();

    // print the operation and function name
    auto funcName =
        op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
            .getValue();

    p << ' ';
    p.printSymbolName(funcName);

    ArrayRef<Type> inputTypes = getFunctionType().getInputs();
    ArrayRef<Type> outputTypes = getFunctionType().getResults();

    if (!inputTypes.empty()) {
        p << " inputs(";
        for (unsigned i = 0; i < inputTypes.size(); i++) {
            if (i > 0) p << ", ";

            if (isExternal)
                p << "\%arg" << i;
            else
                p.printOperand(body.getArgument(i));
            p << " : " << cast<OutputType>(inputTypes[i]).getElementType();
        }
        p << ") ";
    }

    if (!outputTypes.empty()) {
        p << " outputs(";
        unsigned inpSize = inputTypes.size();
        for (unsigned i = inpSize; i < outputTypes.size() + inpSize; i++) {
            if (i > inpSize) p << ", ";

            if (isExternal)
                p << "\%arg" << i;
            else
                p.printOperand(body.getArgument(i));
            p << " : "
              << cast<InputType>(outputTypes[i - inpSize]).getElementType();
        }
        p << ") ";
    }

    // print any attributes in the attribute list into the dict
    if (!op->getAttrs().empty())
        p.printOptionalAttrDictWithKeyword(
            op->getAttrs(),
            /*elidedAttrs=*/{getFunctionTypeAttrName(), getSymNameAttrName()});

    // Print the region
    if (!isExternal) {
        p << ' ';
        p.printRegion(
            body,
            /*printEntryBlockArgs =*/false,
            /*printBlockTerminators =*/true);
    }
}

LogicalResult RegionOp::verify()
{
    auto thisRegion = dyn_cast<RegionOp>(getOperation());
    auto moduleOp = thisRegion->getParentOfType<ModuleOp>();
    auto ops = thisRegion.getBody().getOps();
    auto args = thisRegion.getBody().getArguments();

    RegionOp topRegion;
    moduleOp->walk([&](RegionOp regionOp) { topRegion = regionOp; });
    bool isTopRegion = thisRegion == topRegion;

    // Check if the contents in a region is correct
    bool isContentCorrect = true;
    if (isTopRegion) {
        for (auto &op : ops)
            if (!isa<ChannelOp>(op) && !isa<InstantiateOp>(op)
                && !isa<EmbedOp>(op) && !isa<ConnectInputOp>(op)
                && !isa<ConnectOutputOp>(op))
                isContentCorrect = false;
    } else {
        for (auto &op : ops)
            if (!isa<ChannelOp>(op) && !isa<InstantiateOp>(op)
                && !isa<EmbedOp>(op))
                isContentCorrect = false;
    }

    if (!isContentCorrect) {
        if (isTopRegion)
            return ::emitError(getLoc(), "Unsupported ops used in top region.");
        else
            return ::emitError(
                getLoc(),
                "Only channels and instances are allowed in non-top region.");
    }

    // TODO: Check if all ports are connected and only once
    for (auto arg : args) {
        auto argUsers = arg.getUsers();
        auto sizeUsers = std::distance(argUsers.begin(), argUsers.end());
        if (sizeUsers == 0)
            return ::emitError(getLoc(), "Detecting dangling port.");
        else if (sizeUsers > 1)
            return ::emitError(getLoc(), "Detecting port used more than once.");
    }

    thisRegion.walk([&](ChannelOp channelOp) {
        auto inputUser = channelOp.getInChan().getUses().begin().getUser();
        if (auto instantiateOp = dyn_cast<InstantiateOp>(inputUser)) {
            auto calleeName =
                instantiateOp.getCallee().getRootReference().str();
        }
    });

    return success();
}

//===----------------------------------------------------------------------===//
// EmbedOp
//===----------------------------------------------------------------------===//

ParseResult EmbedOp::parse(OpAsmParser &parser, OperationState &result)
{
    // parse operator name
    StringAttr calleeAttr;
    if (parser.parseSymbolName(calleeAttr)) return failure();

    result.addAttribute(
        getCalleeAttrName(result.name),
        SymbolRefAttr::get(calleeAttr));

    // parse the operator inputs and outputs
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

void EmbedOp::print(OpAsmPrinter &p)
{
    // callee
    p << ' ';
    p.printAttributeWithoutType(getCalleeAttr());

    // print `inputs (...)` if existent
    if (!getInputs().empty()) p << " inputs(" << getInputs() << ")";

    // print `outputs (...)` if existent
    if (!getOutputs().empty()) p << " outputs(" << getOutputs() << ")";

    // signature
    SmallVector<Type> inpChans, outChans;

    for (auto in : getInputs().getTypes())
        inpChans.push_back(cast<OutputType>(in).getElementType());
    for (auto out : getOutputs().getTypes())
        outChans.push_back(mlir::cast<InputType>(out).getElementType());

    p << " : ";
    p.printFunctionalType(inpChans, outChans);
}

LogicalResult EmbedOp::verify()
{
    auto moduleOp = getOperation()->getParentOfType<ModuleOp>();
    auto embedRegionName = getCallee().getRootReference().str();
    auto hasEmbeddedRegion = false;
    auto embeddingProcessOrOperator = false;
    RegionOp embeddedRegionOp;

    // Check if there is a region which has the name of the one to be
    // embedded
    moduleOp->walk([&](Operation* op) {
        if (auto processOp = dyn_cast<ProcessOp>(op)) {
            if (processOp.getSymName().str() == embedRegionName)
                embeddingProcessOrOperator = true;
        } else if (auto operatorOp = dyn_cast<OperatorOp>(op)) {
            if (operatorOp.getSymName().str() == embedRegionName)
                embeddingProcessOrOperator = true;
        } else if (auto regionOp = dyn_cast<RegionOp>(op)) {
            if (regionOp.getSymName().str() == embedRegionName) {
                hasEmbeddedRegion = true;
                embeddedRegionOp = regionOp;
            }
        } else
            WalkResult::advance();
    });

    // If there is the embedded region, check if the function type mathces
    if (embeddingProcessOrOperator || !hasEmbeddedRegion)
        return ::emitError(getLoc(), "Cannot find the embedded region op.");

    auto inputTy = getInputs().getTypes();
    auto outputTy = getOutputs().getTypes();
    auto embedFuncTy = embeddedRegionOp.getFunctionType();
    if (embedFuncTy != FunctionType::get(getContext(), inputTy, outputTy))
        return ::emitError(
            getLoc(),
            "Fcuntion type mismatches the embedded region");

    return success();
}

//===----------------------------------------------------------------------===//
// ConnectInputOp
//===----------------------------------------------------------------------===//

ParseResult ConnectInputOp::parse(OpAsmParser &parser, OperationState &result)
{
    OpAsmParser::UnresolvedOperand regionPort;
    OpAsmParser::UnresolvedOperand channelPort;
    Type dataTy;

    if (parser.parseOperand(regionPort) || parser.parseComma()
        || parser.parseOperand(channelPort) || parser.parseColon()
        || parser.parseType(dataTy))
        return failure();

    Type regionTy = OutputType::get(dataTy.getContext(), dataTy);
    Type channelTy = InputType::get(dataTy.getContext(), dataTy);
    if (parser.resolveOperand(regionPort, regionTy, result.operands)
        || parser.resolveOperand(channelPort, channelTy, result.operands))
        return failure();

    return success();
}

void ConnectInputOp::print(OpAsmPrinter &p)
{
    p << " " << getRegionPort() << ", " << getChannelPort() << " : "
      << getRegionPort().getType().getElementType();
}

//===----------------------------------------------------------------------===//
// ConnectOutputOp
//===----------------------------------------------------------------------===//

ParseResult ConnectOutputOp::parse(OpAsmParser &parser, OperationState &result)
{
    OpAsmParser::UnresolvedOperand regionPort;
    OpAsmParser::UnresolvedOperand channelPort;
    Type dataTy;

    if (parser.parseOperand(regionPort) || parser.parseComma()
        || parser.parseOperand(channelPort) || parser.parseColon()
        || parser.parseType(dataTy))
        return failure();

    Type regionTy = InputType::get(dataTy.getContext(), dataTy);
    Type channelTy = OutputType::get(dataTy.getContext(), dataTy);
    if (parser.resolveOperand(regionPort, regionTy, result.operands)
        || parser.resolveOperand(channelPort, channelTy, result.operands))
        return failure();

    return success();
}

void ConnectOutputOp::print(OpAsmPrinter &p)
{
    p << " " << getRegionPort() << ", " << getChannelPort() << " : "
      << getRegionPort().getType().getElementType();
}

//===----------------------------------------------------------------------===//
// ProcessOp
//===----------------------------------------------------------------------===//

void ProcessOp::build(
    OpBuilder &builder,
    OperationState &state,
    StringRef name,
    FunctionType function_type,
    ArrayRef<int64_t> multiplicity)
{
    state.addAttribute(
        SymbolTable::getSymbolAttrName(),
        builder.getStringAttr(name));
    state.addAttribute(
        ProcessOp::getFunctionTypeAttrName(state.name),
        TypeAttr::get(function_type));
    if (!multiplicity.empty())
        state.addAttribute(
            "multiplicity",
            DenseI64ArrayAttr::get(state.getContext(), multiplicity));

    Region* region = state.addRegion();
    Block* body = new Block();
    region->push_back(body);

    SmallVector<Type> blockArgTypes;
    blockArgTypes.append(
        function_type.getInputs().begin(),
        function_type.getInputs().end());
    blockArgTypes.append(
        function_type.getResults().begin(),
        function_type.getResults().end());
    body->addArguments(
        blockArgTypes,
        SmallVector<Location>(blockArgTypes.size(), builder.getUnknownLoc()));
}

/// @brief  Returns whether the operator is externally defined, i.e., has no
/// body.
/// @return `true` if there is no body attached, `false` the operator has a
/// body.
bool ProcessOp::isExternal()
{
    Region &body = getRegion();
    return body.empty();
}

ParseResult ProcessOp::parse(OpAsmParser &parser, OperationState &result)
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

void ProcessOp::print(OpAsmPrinter &p)
{
    Operation* op = getOperation();
    Region &body = op->getRegion(0);
    bool isExternal = body.empty();

    // print the operation and function name
    auto funcName =
        op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
            .getValue();

    p << ' ';
    p.printSymbolName(funcName);

    ArrayRef<Type> inputTypes = getFunctionType().getInputs();
    ArrayRef<Type> outputTypes = getFunctionType().getResults();

    if (!inputTypes.empty()) {
        p << " inputs(";
        for (unsigned i = 0; i < inputTypes.size(); i++) {
            if (i > 0) p << ", ";

            if (isExternal)
                p << "\%arg" << i;
            else
                p.printOperand(body.getArgument(i));
            p << " : " << cast<OutputType>(inputTypes[i]).getElementType();
        }
        p << ")";
    }

    if (!outputTypes.empty()) {
        p << " outputs(";
        unsigned inpSize = inputTypes.size();
        for (unsigned i = inpSize; i < outputTypes.size() + inpSize; i++) {
            if (i > inpSize) p << ", ";

            if (isExternal)
                p << "\%arg" << i;
            else
                p.printOperand(body.getArgument(i));
            p << " : "
              << cast<InputType>(outputTypes[i - inpSize]).getElementType();
        }
        p << ")";
    }

    // print any attributes in the attribute list into the dict
    if (!op->getAttrs().empty())
        p.printOptionalAttrDictWithKeyword(
            op->getAttrs(),
            /*elidedAttrs=*/{getFunctionTypeAttrName(), getSymNameAttrName()});

    // Print the region
    if (!isExternal) {
        p << ' ';
        p.printRegion(
            body,
            /*printEntryBlockArgs =*/false,
            /*printBlockTerminators =*/true);
    }
}

// TODO: ProcessOp should allow ops outside of loop
LogicalResult ProcessOp::verify()
{
    // If there is a LoopOp, it must be the first op in the body
    // if (!getBody().empty()) {
    //     auto ops = getBody().getOps();
    //     bool isFirstLoop, hasLoop = false;
    //     for (const auto &op : ops)
    //         if (auto loopOp = dyn_cast<LoopOp>(op)) hasLoop = true;
    //     if (auto loopOp = dyn_cast<LoopOp>(&getBody().front().front()))
    //         isFirstLoop = true;
    //     if (hasLoop && !isFirstLoop)
    //         return emitError("The LoopOp must be the first op of ProcessOp");
    // }

    // Ensure that all inputs are of type OutputType and all outputs of type
    // InputType
    FunctionType fnSig = getFunctionType();
    for (Type in : fnSig.getInputs())
        if (!llvm::isa<OutputType>(in))
            return ::emitError(
                getLoc(),
                "LoopOp inputs must be of type OutputType");

    for (Type out : fnSig.getResults())
        if (!llvm::isa<InputType>(out))
            return ::emitError(
                getLoc(),
                "LoopOp outputs must be of type InputType");

    // if a multiplicity is defined, it must cover all arguments!
    ArrayRef<int64_t> multiplicity = getMultiplicity();
    size_t fnSigArgCount = fnSig.getNumInputs() + fnSig.getNumResults();
    if (!multiplicity.empty() && multiplicity.size() != fnSigArgCount)
        return emitError(
            "ProcessOp multiplicity must have a multiplicity for each "
            "channel");

    return success();
}

//===----------------------------------------------------------------------===//
// OperatorOp
//===----------------------------------------------------------------------===//

void OperatorOp::build(
    OpBuilder &builder,
    OperationState &state,
    StringRef name,
    FunctionType function_type,
    ArrayRef<Type> iter_args)
{
    state.addAttribute(
        SymbolTable::getSymbolAttrName(),
        builder.getStringAttr(name));
    state.addAttribute(
        OperatorOp::getFunctionTypeAttrName(state.name),
        TypeAttr::get(function_type));

    // add init body (no arguments)
    state.addRegion();
    // add actual body region
    Region* bodyRegion = state.addRegion();
    Block* body = new Block();
    bodyRegion->push_back(body);

    SmallVector<Type> blockArgTypes;
    blockArgTypes.append(
        function_type.getInputs().begin(),
        function_type.getInputs().end());
    blockArgTypes.append(iter_args.begin(), iter_args.end());
    body->addArguments(
        blockArgTypes,
        SmallVector<Location>(blockArgTypes.size(), builder.getUnknownLoc()));

    SmallVector<Attribute> iterArgsTypesAttr;
    for (Type ty : iter_args) iterArgsTypesAttr.push_back(TypeAttr::get(ty));
    state.addAttribute(
        "iter_args_types",
        ArrayAttr::get(builder.getContext(), iterArgsTypesAttr));
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
    SmallVector<OpAsmParser::Argument> inVals, outVals, iterVals;
    SMLoc signatureLocation = parser.getCurrentLocation();

    // parse inputs/outputs separately for later distinction
    if (succeeded(parser.parseOptionalKeyword("inputs"))) {
        if (parseChannelArgumentList<TypeId>(parser, inVals)) return failure();
    }

    if (succeeded(parser.parseOptionalKeyword("outputs"))) {
        if (parseChannelArgumentList<TypeId>(parser, outVals)) return failure();
    }

    if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
        if (parseChannelArgumentList<TypeId>(parser, iterVals))
            return failure();
    }

    SmallVector<Type> argTypes, resultTypes, iterArgsTypes;
    argTypes.reserve(inVals.size());
    resultTypes.reserve(outVals.size());
    iterArgsTypes.reserve(iterVals.size());

    for (auto &arg : inVals) argTypes.push_back(arg.type);
    for (auto &arg : outVals) resultTypes.push_back(arg.type);
    for (auto &arg : iterVals) iterArgsTypes.push_back(arg.type);
    Type type = builder.getFunctionType(argTypes, resultTypes);

    if (!type) {
        return parser.emitError(signatureLocation)
               << "Failed to construct operator type";
    }

    result.addAttribute(
        getFunctionTypeAttrName(result.name),
        TypeAttr::get(type));
    SmallVector<Attribute> iterArgsTypeAttr;
    for (auto type : iterArgsTypes)
        iterArgsTypeAttr.push_back(TypeAttr::get(type));
    result.addAttribute(
        "iter_args_types",
        ArrayAttr::get(builder.getContext(), iterArgsTypeAttr));

    // merge both argument lists for the block arguments
    inVals.append(iterVals);

    OptionalParseResult attrResult =
        parser.parseOptionalAttrDictWithKeyword(result.attributes);
    if (attrResult.has_value() && failed(*attrResult)) return failure();

    // parse the initialization region if iter_args exist
    auto* initBody = result.addRegion();
    if (!iterVals.empty()) {
        if (parser.parseKeyword("initialize")) return failure();
        // auto* initBody = result.addRegion();
        SMLoc loc = parser.getCurrentLocation();
        if (failed(parser.parseRegion(*initBody)))
            return parser.emitError(loc, "expected an initialize region");
        if (initBody->empty())
            return parser.emitError(
                loc,
                "expected non-empty initialization body");
    }

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
    Region &initBody = op->getRegion(0);
    Region &body = op->getRegion(1);
    bool isExternal = body.empty();

    // print the operation and function name
    auto funcName =
        op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
            .getValue();

    p << ' ';
    p.printSymbolName(funcName);

    ArrayRef<Type> inputTypes = getFunctionType().getInputs();
    ArrayRef<Type> outputTypes = getFunctionType().getResults();
    auto numArgs = body.getArguments().size();
    auto bias = inputTypes.size();
    auto hasIterArgs =
        !op->getAttrOfType<ArrayAttr>("iter_args_types").getValue().empty();

    if (!inputTypes.empty()) {
        p << " inputs(";
        for (unsigned i = 0; i < inputTypes.size(); i++) {
            if (i > 0) p << ", ";

            if (isExternal)
                p << "\%arg" << i;
            else
                p.printOperand(body.getArgument(i));
            p << " : " << inputTypes[i];
        }
        p << ")";
    }

    if (!outputTypes.empty()) {
        p << " outputs(";
        unsigned inpSize = inputTypes.size();
        for (unsigned i = inpSize; i < outputTypes.size() + inpSize; i++) {
            if (i > inpSize) p << ", ";

            p << "\%arg" << i;
            // if (isExternal)
            //     p << "\%arg" << i;
            // else
            //     p.printOperand(body.getArgument(i));
            p << " : " << outputTypes[i - inpSize];
        }
        p << ")";
    }

    if (hasIterArgs) {
        p << " iter_args(";
        // unsigned bias = inputTypes.size() + outputTypes.size();
        for (unsigned i = bias; i < numArgs; i++) {
            if (i > bias)
                p << ", ";
            else
                p.printOperand(body.getArgument(i));
            p << " : " << body.getArgument(i).getType();
        }
        p << ") ";
    }

    // print any attributes in the attribute list into the dict
    if (!op->getAttrs().empty())
        p.printOptionalAttrDictWithKeyword(
            op->getAttrs(),
            /*elidedAttrs=*/{
                getFunctionTypeAttrName(),
                getSymNameAttrName(),
                StringAttr::get(getContext(), "iter_args_types")});

    // Print the regions
    if (hasIterArgs) {
        p << "initialize ";
        p.printRegion(
            initBody,
            /*printEntryBlockArgs =*/false,
            /*printBlockTerminators =*/true);
    }
    if (!isExternal) {
        p << ' ';
        p.printRegion(
            body,
            /*printEntryBlockArgs =*/false,
            /*printBlockTerminators =*/true);
    }
}

LogicalResult OperatorOp::verify()
{
    // In OperatorOp, YieldOp or OutputOp is the only legal dfg operation
    for (auto &op : getInitBody().getOps()) {
        if (op.getDialect()->getNamespace() == "dfg") {
            if (!isa<YieldOp>(op))
                return ::emitError(getLoc(), "Only yield op is allowed here.");
        } else if (!isa<arith::ConstantOp>(op)) {
            return ::emitError(
                getLoc(),
                "Only constants are allowed in initialize region.");
        }
    }
    for (auto &op : getBody().getOps()) {
        if (op.getDialect()->getNamespace() == "dfg") {
            if (!isa<YieldOp>(op) && !isa<OutputOp>(op))
                return ::emitError(
                    getLoc(),
                    "Only yield or output op is allowed here.");
        }
    }

    // // Verify that all output ports are only used in OutputOp
    // // and all iter args are only used in YieldOp
    // auto numInputs = getFunctionType().getNumInputs();
    // auto numOutputs = getFunctionType().getNumResults();
    // if (!getBody().empty())
    //     for (size_t i = numInputs; i < numInputs + numOutputs; i++) {
    //         for (Operation* user : getBody().getArgument(i).getUsers())
    //             if (!isa<OutputOp>(user))
    //                 return ::emitError(
    //                     user->getLoc(),
    //                     "Output ports can only be used to output data into "
    //                     "them.");
    //     }

    return success();
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

LogicalResult OutputOp::verify()
{
    // TODO
    auto op = getOperation();
    auto operatorOp = getParentOp();
    if (!isa<OperatorOp>(operatorOp))
        return ::emitError(getLoc(), "OutputOp must be in an operator.");
    auto* body = op->getParentRegion();
    if (body != &operatorOp.getBody())
        return ::emitError(
            getLoc(),
            "OutputOp must be in the body of an operator.");
    return success();
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

LogicalResult YieldOp::verify()
{
    auto op = getOperation();
    if (auto operatorOp = op->getParentOfType<OperatorOp>()) {
        auto* region = op->getParentRegion();
        auto yieldValueTypes = getOperandTypes();
        SmallVector<Type> iterArgsTypes;
        auto iterArgsTypesAttr =
            operatorOp.getOperation()
                ->getAttrOfType<ArrayAttr>("iter_args_types")
                .getValue();
        for (auto typeAttr : iterArgsTypesAttr)
            iterArgsTypes.push_back(cast<TypeAttr>(typeAttr).getValue());
        if (yieldValueTypes.size() != iterArgsTypes.size())
            return ::emitError(
                getLoc(),
                "Yield value size mismatches iter args size.");
        if (region == &operatorOp.getInitBody()) {
            for (auto type : llvm::zip(yieldValueTypes, iterArgsTypes))
                if (std::get<0>(type) != std::get<1>(type)) {
                    return ::emitError(
                        getLoc(),
                        "The type of operand must match the iterate arguments' "
                        "type in initialize region");
                }
        } else if (region == &operatorOp.getBody()) {
            for (auto type : llvm::zip(yieldValueTypes, iterArgsTypes))
                if (std::get<0>(type) != std::get<1>(type)) {
                    return ::emitError(
                        getLoc(),
                        "The type of operand must match the iterate arguments' "
                        "type in OperatorOp region");
                }
        }
    } else if (auto loopOp = op->getParentOfType<LoopOp>()) {
        auto iterArgs = loopOp.getIterArgs();
        if (getOperands().size() != iterArgs.size())
            return ::emitError(
                getLoc(),
                "The size of yielded values must match the size of iter args.");
        for (size_t i = 0; i < iterArgs.size(); i++)
            if (iterArgs[i].getType() != getOperand(i).getType())
                return ::emitError(
                    getLoc(),
                    "The type of operand must match the iterate arguments' "
                    "type in LoopOp region");
    }

    return success();
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

void LoopOp::build(
    OpBuilder &builder,
    OperationState &state,
    ValueRange inChans,
    ValueRange outChans,
    ValueRange iterArgs)
{
    state.addOperands(inChans);
    state.addOperands(outChans);
    llvm::copy(
        ArrayRef<int32_t>(
            {static_cast<int32_t>(inChans.size()),
             static_cast<int32_t>(outChans.size()),
             static_cast<int32_t>(iterArgs.size())}),
        state.getOrAddProperties<Properties>().operandSegmentSizes.begin());
    Region* region = state.addRegion();
    Block* block = new Block();
    region->push_back(block);

    if (!iterArgs.empty()) {
        state.addOperands(iterArgs);
        for (Value v : iterArgs) block->addArgument(v.getType(), v.getLoc());
    }
    // builder.createBlock(state.addRegion());
}

ParseResult LoopOp::parse(OpAsmParser &parser, OperationState &result)
{
    // parse inputs/outputs
    SmallVector<OpAsmParser::Argument> inVals, outVals, iterVals;

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

    SMLoc iterArgsLocation = parser.getCurrentLocation();
    if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
        if (parseChannelArgumentList<TypeId>(parser, iterVals))
            return failure();
    }

    SmallVector<OpAsmParser::UnresolvedOperand, 4> inputs, outputs, iterArgs;
    SmallVector<Type> argTypes, resultTypes, iterArgsTypes;
    int32_t numInputs = inVals.size();
    int32_t numOutputs = outVals.size();
    int32_t numIterArgs = iterVals.size();
    argTypes.reserve(numInputs);
    inputs.reserve(numInputs);
    resultTypes.reserve(numOutputs);
    outputs.reserve(numOutputs);
    iterArgs.reserve(numIterArgs);
    iterArgsTypes.reserve(numIterArgs);

    for (auto &arg : inVals) {
        argTypes.push_back(arg.type);
        inputs.push_back(arg.ssaName);
    }
    for (auto &arg : outVals) {
        resultTypes.push_back(arg.type);
        outputs.push_back(arg.ssaName);
    }
    for (auto &arg : iterVals) {
        iterArgsTypes.push_back(arg.type);
        iterArgs.push_back(arg.ssaName);
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

    if (parser.resolveOperands(
            iterArgs,
            iterArgsTypes,
            iterArgsLocation,
            result.operands))
        return failure();

    // Add derived `operand_segment_sizes` attribute based on parsed
    // operands.
    auto operandSegmentSizes = parser.getBuilder().getDenseI32ArrayAttr(
        {numInputs, numOutputs, numIterArgs});
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
            p.printType(cast<OutputType>(inputs[i].getImpl()->getType())
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
            p.printType(cast<InputType>(outputs[i].getImpl()->getType())
                            .getElementType());
        }
        p << ")";
    }

    Operation::operand_range iterArgs = getIterArgs();
    if (!iterArgs.empty()) {
        p << " iter_args (";
        for (unsigned i = 0; i < iterArgs.size(); i++) {
            if (i > 0) p << ", ";
            p.printOperand(iterArgs[i]);
            p << " : ";
            p.printType(iterArgs[i].getImpl()->getType());
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

LogicalResult LoopOp::verify()
{
    auto loopOp = dyn_cast<LoopOp>(getOperation());
    auto processOp = loopOp.getParentOp();
    auto processFuncTy = processOp.getFunctionType();

    auto loopInputSize =
        std::distance(loopOp.getInChans().begin(), loopOp.getInChans().end());
    auto loopOutputSize =
        std::distance(loopOp.getOutChans().begin(), loopOp.getOutChans().end());

    if (loopInputSize) {
        if (loopInputSize != processFuncTy.getNumInputs())
            return ::emitError(
                getLoc(),
                "LoopOp should monitor all input ports or none.");
    }
    if (loopOutputSize) {
        if (loopOutputSize != processFuncTy.getNumResults())
            return ::emitError(
                getLoc(),
                "LoopOp should monitor all output ports or none.");
    }

    return success();
}

//===----------------------------------------------------------------------===//
// ChannelOp
//===----------------------------------------------------------------------===//

void ChannelOp::build(
    OpBuilder &builder,
    OperationState &state,
    Type encapsulatedType,
    int bufferSize)
{
    build(
        builder,
        state,
        InputType::get(builder.getContext(), encapsulatedType),
        OutputType::get(builder.getContext(), encapsulatedType),
        encapsulatedType,
        IntegerAttr::get(builder.getI32Type(), bufferSize));
}

ParseResult ChannelOp::parse(OpAsmParser &parser, OperationState &result)
{
    if (failed(parser.parseLParen())) return failure();

    // parse an optional channel size
    int size = 0;
    Attribute sizeAttr;
    OptionalParseResult sizeResult = parser.parseOptionalInteger(size);
    OptionalParseResult sizeAliasResult =
        parser.parseOptionalAttribute(sizeAttr);
    // try parseCustomAttributeWithFallback
    if (sizeResult.has_value()) {
        if (failed(sizeResult.value())) return failure();
        sizeAttr = parser.getBuilder().getI32IntegerAttr(size);
        result.addAttribute(getBufferSizeAttrName(result.name), sizeAttr);
    } else if (sizeAliasResult.has_value()) {
        if (failed(sizeAliasResult.value())) return failure();
        if (!isa<IntegerAttr>(sizeAttr))
            return parser.emitError(
                parser.getCurrentLocation(),
                "The buffer size must be an i32 signless integer "
                "attribute!");
        result.addAttribute(getBufferSizeAttrName(result.name), sizeAttr);
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
    if (const auto size = getBufferSize()) p << size.value();
    p << ") : ";

    p.printType(getEncapsulatedType());
}

LogicalResult ChannelOp::verify()
{
    Type encapsulated = getEncapsulatedType();
    Type out = getOutChan().getType().getElementType();
    Type in = getInChan().getType().getElementType();

    if (encapsulated != out || encapsulated != in)
        return ::emitError(
            getLoc(),
            "The element types of both results and the encapsulated type "
            "of "
            "the function itself must match");

    return success();
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

    // parse the operator inputs and outputs
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
    if (!getInputs().empty()) p << " inputs(" << getInputs() << ")";

    // print `outputs (...)` if existent
    if (!getOutputs().empty()) p << " outputs(" << getOutputs() << ")";

    // signature
    SmallVector<Type> inpChans, outChans;

    for (auto in : getInputs().getTypes())
        inpChans.push_back(cast<OutputType>(in).getElementType());
    for (auto out : getOutputs().getTypes())
        outChans.push_back(cast<InputType>(out).getElementType());

    p << " : ";
    p.printFunctionalType(inpChans, outChans);
}

LogicalResult InstantiateOp::verify()
{
    auto moduleOp = getOperation()->getParentOfType<ModuleOp>();
    auto calleeName = getCallee().getRootReference().str();
    auto hasCalledProcessOrOperator = false;
    auto callingRegion = false;
    ProcessOp calledProcessOp;
    OperatorOp calledOperatorOp;

    // Check if there is a process or an operator which has the name of the one
    // to be instantiated
    moduleOp->walk([&](Operation* op) {
        if (auto processOp = dyn_cast<ProcessOp>(op)) {
            if (processOp.getSymName().str() == calleeName) {
                hasCalledProcessOrOperator = true;
                calledProcessOp = processOp;
            }
        } else if (auto operatorOp = dyn_cast<OperatorOp>(op)) {
            if (operatorOp.getSymName().str() == calleeName) {
                hasCalledProcessOrOperator = true;
                calledOperatorOp = operatorOp;
            }
        } else if (auto regionOp = dyn_cast<RegionOp>(op)) {
            if (regionOp.getSymName().str() == calleeName) callingRegion = true;
        } else
            WalkResult::advance();
    });

    if (callingRegion || !hasCalledProcessOrOperator)
        return ::emitError(
            getLoc(),
            "Cannot find the called process or operator op.");

    // If there is the called process or opereator, check if the function type
    // mathces
    auto inputTy = getInputs().getTypes();
    auto outputTy = getOutputs().getTypes();
    FunctionType calledFuncTy;
    if (calledProcessOp != nullptr)
        calledFuncTy = calledProcessOp.getFunctionType();
    else {
        auto operatorFunc = calledOperatorOp.getFunctionType();
        SmallVector<Type> inTy, outTy;
        // SmallVector<Type> outTy;
        for (auto type : operatorFunc.getInputs())
            inTy.push_back(OutputType::get(getContext(), type));
        for (auto type : operatorFunc.getResults())
            outTy.push_back(InputType::get(getContext(), type));
        calledFuncTy = FunctionType::get(getContext(), inTy, outTy);
    }
    if (calledFuncTy != FunctionType::get(getContext(), inputTy, outputTy))
        return ::emitError(
            getLoc(),
            "Fcuntion type mismatches the called process or operator");

    return success();
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
// PullNOp
//===----------------------------------------------------------------------===//

ParseResult PullNOp::parse(OpAsmParser &parser, OperationState &result)
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

void PullNOp::print(OpAsmPrinter &p)
{
    p << " " << getChan() << " : " << getType();
}

LogicalResult PullNOp::verify()
{
    Type chanType = getChan().getType().getElementType();
    Type memrefType = getType().getElementType();

    if (chanType != memrefType)
        return ::emitError(
            getLoc(),
            "The element types of the result memref and the channel "
            "must match");

    return success();
}

//===----------------------------------------------------------------------===//
// PushOp
//===----------------------------------------------------------------------===//

ParseResult PushNOp::parse(OpAsmParser &parser, OperationState &result)
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

void PushNOp::print(OpAsmPrinter &p)
{
    p << " (" << getInp() << ") " << getChan() << " : " << getInp().getType();
}

LogicalResult PushNOp::verify()
{
    Type chanType = getChan().getType().getElementType();
    Type memrefType = getInp().getType().getElementType();

    if (chanType != memrefType)
        return ::emitError(
            getLoc(),
            "The element types of the input memref and the channel "
            "must match");

    return success();
}

//===----------------------------------------------------------------------===//
// Intermediate HW operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// HWConnectOp
//===----------------------------------------------------------------------===//

ParseResult HWConnectOp::parse(OpAsmParser &parser, OperationState &result)
{
    OpAsmParser::UnresolvedOperand portArgument;
    OpAsmParser::UnresolvedOperand portQueue;
    Type dataTy;

    if (parser.parseOperand(portArgument) || parser.parseComma()
        || parser.parseOperand(portQueue) || parser.parseColon()
        || parser.parseType(dataTy))
        return failure();

    Type channelTy = InputType::get(dataTy.getContext(), dataTy);
    if (parser.resolveOperand(portArgument, channelTy, result.operands)
        || parser.resolveOperand(portQueue, channelTy, result.operands))
        return failure();

    return success();
}

void HWConnectOp::print(OpAsmPrinter &p)
{
    p << " " << getPortArgument() << ", " << getPortQueue() << " : "
      << getPortArgument().getType().getElementType();
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
