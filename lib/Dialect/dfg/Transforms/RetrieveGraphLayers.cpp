/// Implementation of OperatorToProcess transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
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

namespace mlir {
namespace dfg {
#define GEN_PASS_DEF_DFGRETRIEVEGRAPHLAYERS
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"
} // namespace dfg
} // namespace mlir

using namespace mlir;
using namespace dfg;

struct TensorInfo {
    std::vector<int64_t> shape;
    std::string type;
};

struct LayerInfo {
    unsigned index;
    std::string start_op;
    TensorInfo input;
    TensorInfo output;
};

namespace {
LogicalResult parseTensorInfo(const llvm::json::Object &obj, TensorInfo &info)
{
    // Get tensor's shape
    auto shapeVal = obj.getArray("shape");
    if (!shapeVal) return failure();

    for (const auto &dim : *shapeVal) {
        auto dimVal = dim.getAsInteger();
        if (!dimVal) return failure();
        info.shape.push_back(*dimVal);
    }

    // Get tensor's element type
    auto typeVal = obj.getString("type");
    if (!typeVal) return failure();
    info.type = *typeVal;

    return success();
}
LogicalResult parseLayerInfo(const llvm::json::Object &obj, LayerInfo &info)
{
    // Get layer index
    auto indexVal = obj.getInteger("index");
    if (!indexVal) return failure();
    info.index = *indexVal;
    // Get the starting operation in this layer
    auto startOpVal = obj.getString("start_op");
    if (!startOpVal) return failure();
    info.start_op = *startOpVal;
    // Get input tensors information
    auto inputsVal = obj.getArray("input");
    if (!inputsVal) return failure();

    for (const auto &input : *inputsVal) {
        auto inputObj = input.getAsObject();
        if (!inputObj) return failure();

        TensorInfo inputInfo;
        if (failed(parseTensorInfo(*inputObj, inputInfo))) return failure();

        info.input = inputInfo;
    }
    // Get output tensors information
    auto outputsVal = obj.getArray("output");
    if (!outputsVal) return failure();

    for (const auto &output : *outputsVal) {
        auto outputObj = output.getAsObject();
        if (!outputObj) return failure();

        TensorInfo outputInfo;
        if (failed(parseTensorInfo(*outputObj, outputInfo))) return failure();

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
        llvm::errs() << "Cannot open file: " << filePath << "\n";
        return layers;
    }
    // Parse json file
    auto jsonOrErr = llvm::json::parse(memBufOrErr.get()->getBuffer());
    if (!jsonOrErr) {
        llvm::errs()
            << "Failed to parse file: " << toString(jsonOrErr.takeError())
            << "\n";
        return layers;
    }
    // Check if it's array
    auto rootArray = jsonOrErr->getAsArray();
    if (!rootArray) {
        llvm::errs() << "Expect root to be array\n";
        return layers;
    }
    // Parse each layer
    for (const auto &layer : *rootArray) {
        auto layerObj = layer.getAsObject();
        if (!layerObj) {
            llvm::errs() << "Each layer much be object\n";
            continue;
        }

        LayerInfo info;
        if (succeeded(parseLayerInfo(*layerObj, info)))
            layers.push_back(info);
        else
            llvm::errs() << "Failed to parse layer\n";
    }
    return layers;
}
} // namespace

namespace {
struct DfgRetrieveGraphLayersPass
        : public dfg::impl::DfgRetrieveGraphLayersBase<
              DfgRetrieveGraphLayersPass> {
public:
    void runOnOperation() override;

    LogicalResult getStartOps(func::FuncOp op);
    LogicalResult createLayerFunc(func::FuncOp op, OpBuilder &builder);

    Type getTypeFromStr(std::string typeStr, OpBuilder &builder);
    SmallVector<Operation*> getLayerOps(Operation* start);

private:
    SmallVector<LayerInfo> layers;
    SmallVector<Operation*> startOps;
};
} // namespace

Type DfgRetrieveGraphLayersPass::getTypeFromStr(
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
    llvm::errs() << "=== unknown type: " << typeStr << ", 使用默认f32类型\n";
    return Type();
}

SmallVector<Operation*>
DfgRetrieveGraphLayersPass::getLayerOps(Operation* start)
{
    SmallVector<Operation*> ops;
    auto isProcessed = [&](Operation* op) {
        return !isInSmallVector<Operation*>(op, startOps)
               && !isInSmallVector<Operation*>(op, ops);
    };
    auto getDefiningOps = [&](Operation* op) {
        for (auto operand : op->getOperands()) {
            Operation* defOp = operand.getDefiningOp();
            // Don't save original function or next start op
            // And if the ops is not stored yet
            if (isProcessed(defOp) && !isa<BlockArgument>(operand))
                ops.push_back(defOp);
        }
    };
    std::function<void(Operation*)> getOpsUntilNext = [&](Operation* op) {
        for (auto result : op->getResults()) {
            for (auto user : result.getUsers()) {
                if (isProcessed(user)) {
                    getDefiningOps(user);
                    ops.push_back(user);
                    getOpsUntilNext(user);
                }
            }
        }
    };

    // First, get all the operands' defining ops that used in this op
    getDefiningOps(start);
    // Then, store this op itself
    ops.push_back(start);
    // Last, store all the ops in the chain until the next op
    getOpsUntilNext(start);
    return ops;
}

LogicalResult DfgRetrieveGraphLayersPass::getStartOps(func::FuncOp op)
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

    return success();
}

LogicalResult
DfgRetrieveGraphLayersPass::createLayerFunc(func::FuncOp op, OpBuilder &builder)
{
    auto loc = op.getLoc();
    auto topNameAttr = op.getSymNameAttr();
    auto topFuncTy = op.getFunctionType();
    SmallVector<func::FuncOp> layerFuncOps;

    // Get start ops of each layer
    if (failed(getStartOps(op))) return failure();

    IRMapping mapper;
    Value lastLayerResult;
    Operation* lastLayerTailOp;
    // Create func operation for each layer
    for (unsigned i = 0; i < layers.size(); ++i) {
        auto layer = layers[i];
        auto startOp = startOps[i];
        // Create FuncOp based on the info
        auto layerName = "layer" + std::to_string(layer.index);
        auto inputType = RankedTensorType::get(
            layer.input.shape,
            getTypeFromStr(layer.input.type, builder));
        auto outputType = RankedTensorType::get(
            layer.output.shape,
            getTypeFromStr(layer.output.type, builder));
        builder.setInsertionPoint(op);
        auto layerFunc = builder.create<func::FuncOp>(
            loc,
            layerName,
            builder.getFunctionType({inputType}, {outputType}));
        Block* entryBlock = layerFunc.addEntryBlock();
        // If it's the first layer, create the mapping for original argument
        if (i == 0) mapper.map(op.getArgument(0), layerFunc.getArgument(0));
        // Else it's from last layer, map to this layer's argument
        else
            mapper.map(lastLayerResult, layerFunc.getArgument(0));
        // Insert layer ops
        builder.setInsertionPointToEnd(entryBlock);
        auto layerOps = getLayerOps(startOp);
        Value returnValue;
        for (auto [idx, opi] : llvm::enumerate(layerOps)) {
            if (opi != lastLayerTailOp) {
                auto opCloned = builder.clone(*opi, mapper);
                llvm::errs()
                    << "=== Cloned " << opCloned->getName() << " ===\n";
                if (idx == layerOps.size() - 1) {
                    lastLayerResult = opi->getResult(0);
                    returnValue = opCloned->getResult(0);
                    lastLayerTailOp = opi;
                }
            }
        }
        builder.create<func::ReturnOp>(loc, returnValue);
        layerFuncOps.push_back(layerFunc);
        llvm::errs() << "=== Created layer " << layerName << " ===\n";
    }

    // Create the top function with the same name as original func
    builder.setInsertionPoint(op);
    auto topFunc = builder.create<func::FuncOp>(loc, topNameAttr, topFuncTy);
    Block* entryBlock = topFunc.addEntryBlock();
    builder.setInsertionPointToEnd(entryBlock);
    Value calledValue = topFunc.getArgument(0);
    for (auto layer : layerFuncOps) {
        auto callLayer = builder.create<func::CallOp>(loc, layer, calledValue);
        calledValue = callLayer.getResult(0);
    }
    builder.create<func::ReturnOp>(loc, calledValue);

    return success();
}

void DfgRetrieveGraphLayersPass::runOnOperation()
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
    llvm::errs() << "=== Parsing file: " << infoPath << " ===\n";
    layers = parseModelStructure(infoPath);
    if (layers.empty()) {
        module.emitError() << "=== Failed to parse layers information ===\n";
        signalPassFailure();
    }
    llvm::errs() << "=== Parse file succeed ===\n";
    // If there is only one layer, no need to do anything
    if (layers.size() == 1) {
        llvm::errs() << "=== Only one layer, no extra work ===\n";
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

std::unique_ptr<Pass> mlir::dfg::createDfgRetrieveGraphLayersPass()
{
    return std::make_unique<DfgRetrieveGraphLayersPass>();
}
