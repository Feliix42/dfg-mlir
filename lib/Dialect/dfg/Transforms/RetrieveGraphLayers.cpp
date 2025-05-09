/// Implementation of OperatorToProcess transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <optional>
#include <regex>
#include <string>
#include <vector>

#define DEBUG_TYPE "retrieve-layers"

namespace mlir {
namespace func {
#define GEN_PASS_DEF_FUNCRETRIEVEGRAPHLAYERS
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"
} // namespace func
} // namespace mlir

using namespace mlir;

struct TensorInfo {
    std::vector<int64_t> shape;
    std::string type;
    bool isConst;
};

struct LayerInfo {
    unsigned index;
    std::string name;
    std::string start_op;
    std::vector<TensorInfo> inputs;
    TensorInfo output;
};

namespace {
LogicalResult parseTensorInfo(const llvm::json::Object &obj, TensorInfo &info)
{
    // Get tensor's shape
    auto shapeVal = obj.getArray("shape");
    if (!shapeVal) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to parse tensor shape\n");
        return failure();
    }

    for (const auto &dim : *shapeVal) {
        auto dimVal = dim.getAsInteger();
        if (!dimVal) {
            LLVM_DEBUG(
                llvm::dbgs() << "Failed to parse tensor shape's dimension\n");
            return failure();
        }
        info.shape.push_back(*dimVal);
    }

    // Get tensor's element type
    auto typeVal = obj.getString("type");
    if (!typeVal) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to parse tensor element type\n");
        return failure();
    }
    info.type = *typeVal;

    // If this is a constant
    auto isConstVal = obj.getBoolean("is_const");
    if (!isConstVal) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to parse if tensor is constant\n");
        return failure();
    }
    info.isConst = *isConstVal;

    return success();
}
LogicalResult parseLayerInfo(const llvm::json::Object &obj, LayerInfo &info)
{
    // Get layer index
    auto indexVal = obj.getInteger("index");
    if (!indexVal) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to parse layer index\n");
        return failure();
    }
    info.index = *indexVal;
    // Get layer name
    auto nameVal = obj.getString("name");
    if (!nameVal) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to parse layer name\n");
        return failure();
    }
    info.name = *nameVal;
    // Get the starting operation in this layer
    auto startOpVal = obj.getString("start_op");
    if (!startOpVal) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to parse layer start operation\n");
        return failure();
    }
    info.start_op = *startOpVal;
    // Get input tensors information
    auto inputsVal = obj.getArray("inputs");
    if (!inputsVal) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to parse layer input array\n");
        return failure();
    }

    for (const auto &input : *inputsVal) {
        auto inputObj = input.getAsObject();
        if (!inputObj) {
            LLVM_DEBUG(llvm::dbgs() << "Failed to parse layer input\n");
            return failure();
        }

        TensorInfo inputInfo;
        if (failed(parseTensorInfo(*inputObj, inputInfo))) {
            LLVM_DEBUG(llvm::dbgs() << "Failed to parse input tensor\n");
            return failure();
        }

        info.inputs.push_back(inputInfo);
    }
    // Get output tensors information
    auto outputsVal = obj.getArray("output");
    if (!outputsVal) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to parse output array\n");
        return failure();
    }

    for (const auto &output : *outputsVal) {
        auto outputObj = output.getAsObject();
        if (!outputObj) {
            LLVM_DEBUG(llvm::dbgs() << "Failed to parse output\n");
            return failure();
        }

        TensorInfo outputInfo;
        if (failed(parseTensorInfo(*outputObj, outputInfo))) {
            LLVM_DEBUG(llvm::dbgs() << "Failed to parse output tensor\n");
            return failure();
        }

        info.output = outputInfo;
    }
    return success();
}
SmallVector<LayerInfo> parseModelStructure(StringRef filePath)
{
    SmallVector<LayerInfo> layers;
    // Read json file
    auto memBufOrErr = llvm::MemoryBuffer::getFile(filePath);
    if (!memBufOrErr) {
        LLVM_DEBUG(llvm::dbgs() << "Cannot open file: " << filePath << "\n");
        return layers;
    }
    // Parse json file
    auto jsonOrErr = llvm::json::parse(memBufOrErr.get()->getBuffer());
    if (!jsonOrErr) {
        LLVM_DEBUG(
            llvm::dbgs() << "Failed to parse file: "
                         << toString(jsonOrErr.takeError()) << "\n");
        return layers;
    }
    // Check if it's array
    auto rootArray = jsonOrErr->getAsArray();
    if (!rootArray) {
        LLVM_DEBUG(llvm::dbgs() << "Expect root to be array\n");
        return layers;
    }
    // Parse each layer
    for (const auto &layer : *rootArray) {
        auto layerObj = layer.getAsObject();
        if (!layerObj) {
            LLVM_DEBUG(llvm::dbgs() << "Each layer much be object\n");
            continue;
        }

        LayerInfo info;
        if (succeeded(parseLayerInfo(*layerObj, info)))
            layers.push_back(info);
        else
            LLVM_DEBUG(llvm::dbgs() << "Failed to parse layer\n");
    }
    return layers;
}
} // namespace

namespace {
struct FuncRetrieveGraphLayersPass
        : public func::impl::FuncRetrieveGraphLayersBase<
              FuncRetrieveGraphLayersPass> {
public:
    void runOnOperation() override;

    LogicalResult getStartOps(func::FuncOp op);
    LogicalResult createLayerFunc(func::FuncOp op, OpBuilder &builder);

    Type getTypeFromStr(std::string typeStr, OpBuilder &builder);
    SmallVector<Operation*>
    getLayerOps(Operation* start, SmallVector<Type> argTypes, Type resultType);

private:
    SmallVector<LayerInfo> layers;
    SmallVector<Operation*> startOps;
    SmallVector<Operation*> processedOps;
};
} // namespace

Type FuncRetrieveGraphLayersPass::getTypeFromStr(
    std::string typeStr,
    OpBuilder &builder)
{
    // get type prefix and bitwidth
    std::regex pattern("([a-z]+)(\\d+)");
    std::smatch matches;

    if (std::regex_match(typeStr, matches, pattern) && matches.size() > 2) {
        std::string typePrefix = matches[1].str();
        int bitwidth = std::stoi(matches[2].str());

        if (typePrefix == "f") {
            // floating point numbers
            if (bitwidth == 16) return builder.getF16Type();
            if (bitwidth == 32) return builder.getF32Type();
        } else if (typePrefix == "i") {
            // signed integer
            return builder.getIntegerType(bitwidth);
        } else if (typePrefix == "ui") {
            // unsigned integer
            return builder.getIntegerType(bitwidth, /*isSigned=*/false);
        }
    }

    // Cannot get type
    LLVM_DEBUG(llvm::dbgs() << "=== unknown type: " << typeStr << "\n");
    return Type();
}

SmallVector<Operation*> FuncRetrieveGraphLayersPass::getLayerOps(
    Operation* start,
    SmallVector<Type> argTypes,
    Type resultType)
{
    // TODO: rewrite  this part, logic is broken
    SmallVector<Operation*> layerOps;
    auto isProcessed = [&](Operation* op) {
        return isInSmallVector<Operation*>(op, startOps)
               || isInSmallVector<Operation*>(op, processedOps);
    };

    std::function<void(Value, Type)> getDefiningOps = [&](Value operand,
                                                          Type inType) {
        // If the operand is a block argument, skip
        if (isa<BlockArgument>(operand)) {
            LLVM_DEBUG(
                llvm::dbgs()
                << "===DEBUG=== found block argument as operand, skip "
                   "===DEBUG===\n");
            return;
        }

        LLVM_DEBUG(
            llvm::dbgs() << "===DEBUG=== looking for definition ===DEBUG===\n");
        auto defOp = operand.getDefiningOp();
        // If defined by a const op, the type must be the same as argType
        if (isa<tosa::ConstOp>(defOp)) {
            layerOps.push_back(defOp);
            processedOps.push_back(defOp);
            return;
        }
        // Else, looking for defining op upto the type is correct
        for (auto iOperand : defOp->getOperands()) {
            auto defDefOp = iOperand.getDefiningOp();
            if (!isInSmallVector(defDefOp, processedOps)
                && iOperand.getType() != inType) {
                getDefiningOps(iOperand, inType);
            }
        }
        // Once it's found, can store it in the container
        if (!isInSmallVector(defOp, processedOps)) {
            layerOps.push_back(defOp);
            processedOps.push_back(defOp);
        }
        LLVM_DEBUG(
            llvm::dbgs() << "===DEBUG=== found definition from "
                         << defOp->getName() << " ===DEBUG===\n");
    };

    // Get all the ops before the start
    for (auto [operand, inType] : llvm::zip(start->getOperands(), argTypes))
        getDefiningOps(operand, inType);

    // Start op itself is in this layer as well
    layerOps.push_back(start);
    processedOps.push_back(start);

    // Operations following start until the other start is in this layer
    std::function<void(Operation*)> getOpsUntilNext = [&](Operation* op) {
        for (auto result : op->getResults()) {
            for (auto user : result.getUsers()) {
                if (!isProcessed(user)) {
                    // The operands of user must be already processed or a
                    // blkarg/constant
                    for (auto iOperand : user->getOperands()) {
                        LLVM_DEBUG(
                            llvm::dbgs()
                            << "===DEBUG=== looking for definition of user "
                            << user->getName() << " ===DEBUG===\n");
                        getDefiningOps(iOperand, Type{});
                    }
                    layerOps.push_back(user);
                    processedOps.push_back(user);
                    if (user->getNumResults() == 1
                        && user->getResult(0).getType() == resultType) {
                        // If already get to the result type but still have a
                        // user clamp
                        // TODO: this only supports clamp after you get result
                        // type value
                        auto userUsers = user->getResult(0).getUsers();
                        for (auto userUser : userUsers)
                            if (isa<tosa::ClampOp>(userUser)) {
                                layerOps.push_back(userUser);
                                processedOps.push_back(userUser);
                            }
                        return;
                    }
                    getOpsUntilNext(user);
                }
            }
        }
    };
    getOpsUntilNext(start);

    return layerOps;
}

LogicalResult FuncRetrieveGraphLayersPass::getStartOps(func::FuncOp op)
{
    SmallVector<Operation*> allOps;
    for (Operation &opi : op.getBody().front()) allOps.push_back(&opi);
    unsigned cur = 0;
    for (auto layer : layers) {
        bool found = false;
        auto startOpName = layer.start_op;
        for (auto [i, opi] : llvm::enumerate(allOps)) {
            if (i < cur) continue;
            auto opName = opi->getName().stripDialect().str();
            if (opName == startOpName) {
                startOps.push_back(opi);
                cur = ++i;
                found = true;
                break;
            }
        }
        if (!found) {
            op->emitError()
                << "=== Cannot find start op " << startOpName << "\n";
            return failure();
        }
    }
    // Also push back the return op
    startOps.push_back(allOps.back());

    LLVM_DEBUG(llvm::dbgs() << "===DEBUG=== found start ops ===DEBUG===\n");
    for (auto opi : startOps) {
        LLVM_DEBUG(
            llvm::dbgs() << "===DEBUG=== " << opi->getName() << " at "
                         << opi->getLoc() << " ===DEBUG===\n");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");

    return success();
}

LogicalResult FuncRetrieveGraphLayersPass::createLayerFunc(
    func::FuncOp op,
    OpBuilder &builder)
{
    auto loc = op.getLoc();
    auto topNameAttr = op.getSymNameAttr();
    auto topFuncTy = op.getFunctionType();
    SmallVector<func::FuncOp> layerFuncOps;

    // Get start ops of each layer
    if (failed(getStartOps(op))) return failure();

    DenseMap<unsigned, IRMapping> mappers;
    SmallVector<SmallVector<Operation*>> layerOps;
    SmallVector<SmallVector<Value>> layerArguments;
    DenseMap<unsigned, SmallVector<unsigned>> callValueIndexMap;

    // Create func operation for each layer
    for (unsigned i = 0; i < layers.size(); ++i) {
        auto layer = layers[i];
        auto startOp = startOps[i];
        SmallVector<Type> inputTypes, checkInputTypes;
        // Create FuncOp based on the info
        auto layerName =
            "layer" + std::to_string(layer.index) + "_" + layer.name;
        for (auto input : layer.inputs) {
            auto inputType = RankedTensorType::get(
                input.shape,
                getTypeFromStr(input.type, builder));
            if (!input.isConst) inputTypes.push_back(inputType);
            checkInputTypes.push_back(inputType);
        }
        auto outputType = RankedTensorType::get(
            layer.output.shape,
            getTypeFromStr(layer.output.type, builder));
        builder.setInsertionPoint(op);
        auto layerFunc = builder.create<func::FuncOp>(
            loc,
            layerName,
            builder.getFunctionType(inputTypes, {outputType}));
        LLVM_DEBUG(
            llvm::dbgs()
            << "===INFO=== Creating " << layerName << " with signature "
            << layerFunc.getFunctionType() << " ===INFO===\n");
        Block* entryBlock = layerFunc.addEntryBlock();
        SmallVector<Value> layerArgs;
        for (auto blkArg : entryBlock->getArguments())
            layerArgs.push_back(blkArg);
        layerArguments.push_back(layerArgs);
        LLVM_DEBUG(
            llvm::dbgs()
            << "===INFO=== Looking for operations for this layer ===INFO===\n");
        layerOps.push_back(getLayerOps(startOp, checkInputTypes, outputType));
        layerFuncOps.push_back(layerFunc);
        LLVM_DEBUG(
            llvm::dbgs()
            << "===INFO=== Created " << layerName << " ===INFO===\n");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");

    // Create mapping from the original value to new layer's blkarg
    auto findAndMap = [&](Value value, unsigned layerIdx) {
        for (auto user : value.getUsers()) {
            unsigned i = 0;
            for (auto [idx, zipped] :
                 llvm::enumerate(llvm::zip(layerOps, layerArguments))) {
                auto [opsVec, argsVec] = zipped;
                if (isInSmallVector(user, opsVec)) {
                    // Get the mappings
                    if (mappers.find(idx) == mappers.end()) {
                        IRMapping mapper;
                        mapper.map(value, argsVec[i++]);
                        mappers[idx] = mapper;
                    } else {
                        i = mappers[idx].getValueMap().size();
                        mappers[idx].map(value, argsVec[i++]);
                    }
                    // Store the indices
                    if (callValueIndexMap.find(idx)
                        == callValueIndexMap.end()) {
                        SmallVector<unsigned> indices;
                        indices.push_back(layerIdx);
                        callValueIndexMap[idx] = indices;
                    } else {
                        callValueIndexMap[idx].push_back(layerIdx);
                    }
                }
            }
        }
    };
    // Clone the ops into each layer
    for (auto [idx, funcOp] : llvm::enumerate(layerFuncOps)) {
        auto opsVec = layerOps[idx];
        auto argsVec = layerArguments[idx];
        // If it's the first layer, we only need to map the input argument of
        // the original func to this layer's block argument
        if (idx == 0) {
            IRMapping mapper;
            mapper.map(op.getArgument(0), argsVec.front());
            mappers[idx] = mapper;
        }
        // Otherwise, the value must be defined by a previous op's return value
        // The last op's result of this layer will be used by other layers
        // Skip the last layer
        if (idx < layerOps.size() - 1) {
            auto lastOp = opsVec.back();
            Value returnValue = lastOp->getResult(0);
            findAndMap(returnValue, idx);
        }
        LLVM_DEBUG(
            llvm::dbgs()
            << "===INFO=== In " << funcOp.getSymName() << " ===INFO===\n");
        builder.setInsertionPointToEnd(&funcOp.getBody().front());
        Value returnValue;
        for (auto [i, opi] : llvm::enumerate(opsVec)) {
            auto opCloned = builder.clone(*opi, mappers[idx]);
            LLVM_DEBUG(
                llvm::dbgs()
                << "===INFO=== Cloned " << opCloned->getName()
                << " at location: " << opi->getLoc() << " ===INFO===\n");
            if (i == opsVec.size() - 1) {
                LLVM_DEBUG(
                    llvm::dbgs() << "===INFO=== Found return value from "
                                 << opCloned->getName() << " at "
                                 << opCloned->getLoc() << " ===INFO===\n");
                returnValue = opCloned->getResult(0);
            }
        }
        // Create return with the value
        auto returnLoc = returnValue.getDefiningOp()->getLoc();
        LLVM_DEBUG(
            llvm::dbgs() << "===INFO=== Creating return operation at "
                         << returnLoc << " ===INFO===\n");
        builder.create<func::ReturnOp>(returnLoc, returnValue);
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");

    // Create the top function with the same name as original func
    LLVM_DEBUG(llvm::dbgs() << "===INFO=== Creating top function ===INFO===\n");
    SmallVector<Value> calledValues;
    builder.setInsertionPoint(op);
    auto topFunc = builder.create<func::FuncOp>(loc, topNameAttr, topFuncTy);
    auto topLoc = topFunc.getLoc();
    Block* entryBlock = topFunc.addEntryBlock();
    builder.setInsertionPointToEnd(entryBlock);
    // Value calledValue = topFunc.getArgument(0);
    for (auto [idx, layer] : llvm::enumerate(layerFuncOps)) {
        func::CallOp callLayer;
        if (idx == 0) {
            callLayer = builder.create<func::CallOp>(
                topLoc,
                layer,
                topFunc.getArgument(0));
        } else {
            SmallVector<Value> callValues;
            for (auto valueIdx : callValueIndexMap[idx])
                callValues.push_back(calledValues[valueIdx]);
            callLayer = builder.create<func::CallOp>(topLoc, layer, callValues);
        }
        calledValues.push_back(callLayer.getResult(0));
    }
    builder.create<func::ReturnOp>(topLoc, calledValues.back());
    LLVM_DEBUG(
        llvm::dbgs() << "===INFO=== Successfully created graph ===INFO===\n");

    return success();
}

void FuncRetrieveGraphLayersPass::runOnOperation()
{
    auto &context = getContext();
    auto builder = OpBuilder(&context);
    auto module = dyn_cast<ModuleOp>(getOperation());
    // Must include the information llvm::json file to retrieve the graph
    if (infoPath.empty()) {
        module.emitError() << "info-path option cannot be empty";
        signalPassFailure();
    }
    // Parse layers info from file
    LLVM_DEBUG(llvm::dbgs() << "=== Parsing file: " << infoPath << " ===\n");
    layers = parseModelStructure(infoPath);
    if (layers.empty()) {
        module.emitError() << "=== Failed to parse layers information ===\n";
        signalPassFailure();
    }
    LLVM_DEBUG(llvm::dbgs() << "=== Read layer info succeed ===\n");
    // If there is only one layer, no need to do anything
    if (layers.size() == 1) {
        LLVM_DEBUG(llvm::dbgs() << "=== Only one layer, no extra work ===\n");
        return;
    }

    // Get the graph func, which is the only operation in module
    auto graphFunc = dyn_cast<func::FuncOp>(*module.getOps().begin());
    if (!graphFunc) {
        module.emitError()
            << "there should be only one func operation in module";
        signalPassFailure();
    }
    // Set insertion point to be above original func
    if (failed(createLayerFunc(graphFunc, builder))) {
        module.emitError() << "failed to create func for layers";
        signalPassFailure();
    }
    // Erase original func
    graphFunc->erase();
}

std::unique_ptr<Pass> mlir::func::createFuncRetrieveGraphLayersPass()
{
    return std::make_unique<FuncRetrieveGraphLayersPass>();
}
