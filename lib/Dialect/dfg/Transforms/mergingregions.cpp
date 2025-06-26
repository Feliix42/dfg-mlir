//mergingregions.cpp
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

  
  void runOnOperation() override {
    ModuleOp module = getOperation();
  OpBuilder builder(module.getContext());
    //insertSBoxIfMissing(module, builder);
    
 
  SmallVector<dfg::RegionOp> regions;

  module.walk([&](dfg::RegionOp reg) {
   // llvm::errs() << "[DBG] Found region @" << reg.getSymName() << " with type: "<< reg.getFunctionType() << "\n";
    regions.push_back(reg);
  });

  if (regions.size() < 2) {
    llvm::errs() << "[DBG] Less than 2 regions; skipping merge.\n";
    return;
  }

 dfg::RegionOp *maxInputRegion = nullptr;
  unsigned maxInputs = 0;
// Step 1: Find region with max number of inputs
  for (auto &reg : regions) {
    unsigned numInputs = reg.getFunctionType().getNumInputs();
    if (numInputs > maxInputs) {
      maxInputs = numInputs;
      maxInputRegion = &reg;
    }
  }
unsigned minInputs = std::numeric_limits<unsigned>::max();
for (auto &reg : regions) {
  unsigned count = reg.getFunctionType().getNumInputs();
  if (count < minInputs)
    minInputs = count;
}


  SmallVector<Type> sharedInputs;
  SmallVector<Type> allInputs;
for (unsigned i = 0; i < maxInputs; ++i) {
  // Get the type at index i from the first region that has enough inputs
  std::optional<Type> refType;
  for (auto &reg : regions) {
    auto types = reg.getFunctionType().getInputs();
    if (i < types.size()) {
      refType = types[i];
      break;
    }
  }
    bool isShared = true;
  for (auto &reg : regions) {
    auto inputs = reg.getFunctionType().getInputs();
    if (i >= inputs.size() || inputs[i] != *refType) {
      isShared = false;
      break;
    }
  }
   if (isShared) {
    sharedInputs.push_back(*refType);
    allInputs.push_back(*refType); // only one shared instance
  } else {
    // not shared: push one input per region that has this index
    for (auto &reg : regions) {
      auto inputs = reg.getFunctionType().getInputs();
      if (i < inputs.size())
        allInputs.push_back(inputs[i]);
    }
  }
} 
unsigned unsharedInputs = minInputs -sharedInputs.size() ;
  SmallVector<Type> allOutputs;
  //llvm::errs() << "[DBG] shared inputs: " << sharedInputs << " all inputs: "<<allInputs << "\n";
  ///llvm::errs() << "[DBG] shared inputs: " << sharedInputs.size() << " all inputs: "<<allInputs.size() << " unshared inputs: "<<unsharedInputs << "\n";

  for (auto &r : regions) {
    for (auto t : r.getFunctionType().getResults())
      allOutputs.push_back(t);
  }

  builder.setInsertionPointToEnd(module.getBody());
  auto mergedRegion = builder.create<dfg::RegionOp>(
      builder.getUnknownLoc(),
      builder.getStringAttr("top"),
      builder.getFunctionType(allInputs, allOutputs));
  llvm::errs() << "[DBG] Created merged region @top\n";

  if (!mergedRegion.getBody().empty())
    mergedRegion.getBody().front().erase();

  Block &block = mergedRegion.getBody().emplaceBlock();
  block.addArguments(allInputs, SmallVector<Location>(allInputs.size(), builder.getUnknownLoc()));
  block.addArguments(allOutputs, SmallVector<Location>(allOutputs.size(), builder.getUnknownLoc()));
    
  /*llvm::errs() << "[DBG] Merged block args:\n";
  for (unsigned i = 0; i < block.getNumArguments(); ++i) {
      llvm::errs() << "  Arg #" << i << ": " << block.getArgument(i) << " of type " << block.getArgument(i).getType() << "\n";
  }*/

  builder.setInsertionPointToStart(&block);
  SmallVector<SmallVector<Value>> inputChannels(sharedInputs.size());

  // Create channels for shared inputs
  for (size_t i = 0; i < sharedInputs.size(); ++i) {
    Type elementType = mlir::cast<dfg::OutputType>(sharedInputs[i]).getElementType();

    Value topArg = block.getArgument(i);
    //llvm::errs() << "[DBG] Creating channel for input #" << i << " with type " << elementType << "\n";

    SmallVector<Value> recvPorts;
    SmallVector<Value> sendPorts;                 

  for (size_t r = 0; r < regions.size(); ++r) {
    auto ch = builder.create<dfg::ChannelOp>(
        builder.getUnknownLoc(), elementType, /*depth=*/1);

    Value recv = ch.getResult(0); // to be connected to region
    Value send = ch.getResult(1); // from sbox

    recvPorts.push_back(recv);
    sendPorts.push_back(send);

    //llvm::errs() << "  Region " << r << " Channel: recv=" << recv << ", send=" << send << "\n";
  }    
  auto outType = topArg.getType().dyn_cast<dfg::OutputType>();
  auto elType = outType.getElementType();
  auto intType = elType.dyn_cast<mlir::IntegerType>();
  unsigned width = intType.getWidth();
    // Instantiate sbox12int32
    //llvm::errs() << "[DBG]  topArg: " << topArg.getType() << " recvPorts=" << recvPorts[0].getType()<< " , "<<recvPorts[1] << "\n";
    builder.create<dfg::InstantiateOp>(
            builder.getUnknownLoc(),
            "sbox1x2int" + std::to_string(width),
            ValueRange{topArg},
            recvPorts,
            /*offloaded=*/false);
    inputChannels[i] = sendPorts; // Each region will use inputChannels[i][r]
    //llvm::errs() << "[DBG] Instantiated sbox12int32 for input #" << i << "\n";

  }

  // Clone region bodies
  unsigned outOffset = unsharedInputs;
  for (size_t r = 0; r < regions.size(); ++r) {
    auto &srcBB = regions[r].getBody().front();
    mlir::IRMapping mapping;

    auto ft = regions[r].getFunctionType();
    //llvm::errs() << "[DBG] Cloning region " << r << " (@" << regions[r].getSymName() << ") input numbers: "<< ft.getNumInputs()<<"\n";

    // Map inputs - all regions now use channel outputs
    //shared inputs
    for (unsigned i = 0; i < sharedInputs.size(); ++i) {
      Value target = inputChannels[i][r];
      mapping.map(srcBB.getArgument(i), target);
      //llvm::errs() << "  Mapped shared input arg " << i << " (" << srcBB.getArgument(i) << ") -> " << target << " (type: " << target.getType() << ")\n";
    }
 //unshared inputs
    for (unsigned i =  sharedInputs.size(); i < minInputs; ++i) {
      
      Value target = block.getArgument(i+unsharedInputs*r);
      mapping.map(srcBB.getArgument(i), target);
      //llvm::errs() << "  Mapped independ input arg " << i << " *(" << srcBB.getArgument(i) << ")* -> " << target << " (type: " << target.getType() << ")\n";

    }
    //indepen inputs
   for (unsigned i =  minInputs; i < ft.getNumInputs(); ++i) {
      
      Value target = block.getArgument(i+unsharedInputs);
      mapping.map(srcBB.getArgument(i), target);
      //llvm::errs() << "  Mapped independ input arg " << i << " *(" << srcBB.getArgument(i) << ")* -> " << target << " (type: " << target.getType() << ")\n";

    }
    // Map outputs
    for (unsigned o = 0; o < ft.getNumResults(); ++o) {
      unsigned outputArgIdx = maxInputs + outOffset + o;      
      mapping.map(srcBB.getArgument(ft.getNumInputs() + o),
                  block.getArgument(outputArgIdx));
      //llvm::errs() << "  Mapped output arg " << o << " (" << srcBB.getArgument(ft.getNumInputs() + o) << ") -> block arg #" << outputArgIdx << " ("<< block.getArgument(outputArgIdx) << ")\n";
    }

    // Clone operations
    for (auto &op : srcBB.getOperations()) {
      builder.setInsertionPointToEnd(&block);
      Operation *cloned = builder.clone(op, mapping);
     // llvm::errs() << "    Cloned op: " << op.getName() << "\n";
      for (unsigned i = 0; i < op.getNumResults(); ++i) {
        //llvm::errs() << "    Mapping result " << i << ": " << op.getResult(i)<< " -> " << cloned->getResult(i) << "\n";        
        mapping.map(op.getResult(i), cloned->getResult(i));
      }
    }

    outOffset += ft.getNumResults();
  }

  for (auto &r : regions) r.erase();
  llvm::errs() << "[INFO] Completed merging regions\n";
}

};


} // namespace


std::unique_ptr<Pass> mlir::dfg::createMergingRegionsPass() {
  return std::make_unique<MergingRegionsPass>();
}


