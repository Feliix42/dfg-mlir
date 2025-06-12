#include "dfg-mlir/Dialect/dfg/Transforms/mergingregions.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;
using namespace mlir::dfg;

namespace {
struct MergingRegionsPass : public PassWrapper<MergingRegionsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergingRegionsPass)

  StringRef getArgument() const override { return "dfg-merge-regions"; }
  StringRef getDescription() const override { return "Merge multiple DFG regions into a single region"; }

  static void printValueInfo(Value val, bool isInput) {
    if (auto arg = mlir::dyn_cast<BlockArgument>(val)) {
      unsigned idx = arg.getArgNumber();
      std::string portName = (isInput ? "in" : "out") + std::to_string(idx);
      if (auto nameLoc = mlir::dyn_cast<NameLoc>(arg.getLoc())) {
        portName = nameLoc.getName().str();
      }
      llvm::outs() << "    " << portName << " : " << val.getType() << "\n";
      return;
    }

    if (auto definingOp = val.getDefiningOp()) {
      if (auto chanOp = dyn_cast<dfg::ChannelOp>(definingOp)) {
        auto results = chanOp->getResults();
        auto attr = chanOp->getAttrOfType<IntegerAttr>("depth");
        int64_t chanId = attr ? attr.getInt() : 0;
        for (unsigned i = 0; i < results.size(); ++i) {
          if (results[i] == val) {
            std::string chanName = (i == 0 ? "in_chan_" : "out_chan_") + std::to_string(chanId);
            llvm::outs() << "    " << chanName << ", dfg.channel(" 
                        << chanId << ") : " << val.getType() << "\n";
            return;
          }
        }
      }
      // Default fallback
      llvm::outs() << "    (defined by: ";
      definingOp->print(llvm::outs(), OpPrintingFlags().useLocalScope());
      llvm::outs() << ") : " << val.getType() << "\n";
      return;
    }

    llvm::outs() << "    <unknown> : " << val.getType() << "\n";
  }

void runOnOperation() override {
  ModuleOp module = getOperation();
  OpBuilder builder(module.getContext());
  SmallVector<RegionOp> regions;
  // Collect all regionOps and print info (optional)

  module.walk([&](RegionOp reg) {
    regions.push_back(reg);
  });
 
  if (regions.empty()) {
    module.emitError("No dfg.region found.");
    return;
  }

  RegionOp firstRegion = regions.front();
  FunctionType combinedType = firstRegion.getFunctionType();
  SmallVector<Type> allInputs(combinedType.getInputs().begin(), combinedType.getInputs().end());
  SmallVector<Type> allOutputs(combinedType.getResults().begin(), combinedType.getResults().end());

  // Collect input/output types from remaining regions
  for (size_t i = 1; i < regions.size(); ++i) {
    auto ft = regions[i].getFunctionType();
    allInputs.append(ft.getInputs().begin(), ft.getInputs().end());
    allOutputs.append(ft.getResults().begin(), ft.getResults().end());
  }
  // Create new region @top
  builder.setInsertionPointToEnd(module.getBody());
  auto mergedRegion = builder.create<RegionOp>(
      builder.getUnknownLoc(),
      builder.getStringAttr("top"),
      builder.getFunctionType(allInputs, allOutputs));


  // Remove default block and move first region's block into merged region
  if (!mergedRegion.getBody().empty())
    mergedRegion.getBody().front().erase();

  Block &mergedBlock = mergedRegion.getBody().emplaceBlock();
  auto locsIn = SmallVector<Location>(allInputs.size(), builder.getUnknownLoc());
  auto locsOut = SmallVector<Location>(allOutputs.size(), builder.getUnknownLoc());

  mergedBlock.addArguments(allInputs, locsIn);
  mergedBlock.addArguments(allOutputs, locsOut);
  // Helper to clone body from a region to merged block
  auto cloneRegionOps = [&](RegionOp fromRegion, unsigned &inIdx, unsigned &outIdx) {
  Block &src = fromRegion.getBody().front();
  mlir::IRMapping mapping;
  // Map block arguments (input/output ports)
  auto ft = fromRegion.getFunctionType();
  for (unsigned i = 0; i < ft.getNumInputs(); ++i)
    mapping.map(src.getArgument(i), mergedBlock.getArgument(inIdx + i));
  
  for (unsigned i = 0; i < ft.getNumResults(); ++i)
    mapping.map(src.getArgument(ft.getNumInputs() + i),
                mergedBlock.getArgument(allInputs.size() + outIdx + i));
                
  for (Operation &op : src.getOperations()) {
      builder.setInsertionPointToEnd(&mergedBlock);
      Operation *newOp = builder.clone(op, mapping);
      // Map original block arguments to new ones
      for (unsigned i = 0; i < op.getNumResults(); ++i) {
      mapping.map(op.getResult(i), newOp->getResult(i));
        }
      }
    
  inIdx += ft.getNumInputs();
  outIdx += ft.getNumResults();
  };

  // Clone first region ops
  builder.setInsertionPointToEnd(&mergedBlock);
  unsigned inOffset = 0, outOffset = 0;
  cloneRegionOps(firstRegion, inOffset, outOffset);

  // Clone rest of the regions
  for (size_t i = 1; i < regions.size(); ++i) {
    builder.setInsertionPointToEnd(&mergedBlock);
    cloneRegionOps(regions[i], inOffset, outOffset);
    regions[i].erase(); // Remove old region
  }

  firstRegion.erase(); // Remove the original too

  llvm::outs() << "[INFO] Regions merged into @top.\n";  

}


};
} // namespace

std::unique_ptr<Pass> mlir::dfg::createMergingRegionsPass() {
  return std::make_unique<MergingRegionsPass>();
}
