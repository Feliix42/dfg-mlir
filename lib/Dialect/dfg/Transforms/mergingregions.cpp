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
    std::vector<unsigned int> indexinput(minInputs, 0);
    for (unsigned i = 0; i < maxInputs; ++i) {
      // Get the type at index i from the first region that has enough inputs
      std::optional<Type> refType;
      for (auto &reg : regions) {
        auto types = reg.getFunctionType().getInputs();
        //llvm::errs() << "[DBG] Found region @" <<reg.getSymName() << " number of input : "<< i << " with type: "<< types <<"\n";
        if (i < types.size()) {
          refType = types[i];
           //llvm::errs() << "[DBG] refType" << *refType << "\n";
          break;
        }
      }
      bool isShared = true;
      for (auto &reg : regions) {
        auto inputs = reg.getFunctionType().getInputs();
        //llvm::errs() << "[DBG] Checking region @" << reg.getSymName() << " number of input : "<< i ;
        //if (i < inputs.size()) llvm::errs() <<" with type: "<< inputs[i] <<"\n";
        if (i >= inputs.size()) {
          isShared = false;
          //llvm::errs() << "[DBG] Not shared in region @" << reg.getSymName() << "\n";
          break;
        }
        if (inputs[i] != *refType) {
          isShared = false;
          //llvm::errs() << "[DBG] Not shared in region @" << reg.getSymName() << "\n";
          indexinput[i] = 1;

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
     //for (unsigned i = 0; i < minInputs; ++i) 
      //llvm::errs() << "[DBG] indexinput[" << i << "] = " << indexinput[i] << "\n";
    

    unsigned unsharedInputs = minInputs -sharedInputs.size();
    dfg::RegionOp *maxoutputRegion = nullptr;
    unsigned maxoutputs = 0;
    // Step 1: Find region with max number of outputs
    for (auto &reg : regions) {
      unsigned numoutputs = reg.getFunctionType().getNumResults();
      if (numoutputs > maxoutputs) {
        maxoutputs = numoutputs;
        maxoutputRegion = &reg;
      }
    }
    unsigned minoutputs = std::numeric_limits<unsigned>::max();
    for (auto &reg : regions) {
      unsigned count = reg.getFunctionType().getNumResults();
      if (count < minoutputs)
        minoutputs = count;
    }

    SmallVector<Type> sharedoutputs;
    SmallVector<Type> alloutputs;
    std::vector<unsigned int> indexoutput(minoutputs, 0);
    for (unsigned i = 0; i < maxoutputs; ++i) {
      // Get the type at index i from the first region that has enough outputs
      std::optional<Type> refType;
      for (auto &reg : regions) {
        auto types = reg.getFunctionType().getResults();
        if (i < types.size()) {
          refType = types[i];
          break;
        }
      }
        bool isShared = true;
      for (auto &reg : regions) {
        auto outputs = reg.getFunctionType().getResults();
        if (i >= outputs.size() ) {
          isShared = false;
          break;
        }
        if (outputs[i] != *refType) {
          isShared = false;
          indexoutput[i] = 1;
          break;
        }
      }
      if (isShared) {
        sharedoutputs.push_back(*refType);
        alloutputs.push_back(*refType); // only one shared instance
      } else {
        // not shared: push one output per region that has this index
        for (auto &reg : regions) {
          auto outputs = reg.getFunctionType().getResults();
          if (i < outputs.size())
            alloutputs.push_back(outputs[i]);
        }
      }
    }
    unsigned unsharedoutputs = minoutputs - sharedoutputs.size();

    //llvm::errs() << "[DBG] shared inputs: " << sharedInputs << " all inputs: "<<allInputs << "\n";
    //llvm::errs() << "[DBG] shared inputs: " << sharedInputs.size() << " all inputs: "<<allInputs.size() << " unshared inputs: "<<unsharedInputs << "\n";

 

    builder.setInsertionPointToEnd(module.getBody());
    auto mergedRegion = builder.create<dfg::RegionOp>(
        builder.getUnknownLoc(),
        builder.getStringAttr("top"),
        builder.getFunctionType(allInputs, alloutputs));
    llvm::errs() << "[DBG] Created merged region @top\n";

    if (!mergedRegion.getBody().empty())
        mergedRegion.getBody().front().erase();

    Block &block = mergedRegion.getBody().emplaceBlock();
    block.addArguments(allInputs, SmallVector<Location>(allInputs.size(), builder.getUnknownLoc()));
    block.addArguments(alloutputs, SmallVector<Location>(alloutputs.size(), builder.getUnknownLoc()));

    /*llvm::errs() << "[DBG] Merged block args:\n";
    for (unsigned i = 0; i < block.getNumArguments(); ++i) {
        llvm::errs() << "  Arg #" << i << ": " << block.getArgument(i) << " of type " << block.getArgument(i).getType() << "\n";
    }*/

    builder.setInsertionPointToStart(&block);
    SmallVector<SmallVector<Value>> inputChannels(sharedInputs.size());

    // Create channels for shared inputs
    for (size_t i = 0; i < sharedInputs.size(); ++i) {
      Type elementType = mlir::cast<dfg::OutputType>(sharedInputs[i]).getElementType();
          unsigned int num1= 0,num0=0;
          for (unsigned j = 0; j < minInputs; ++j) {
            if (indexinput[j] == 1) {
              num1+=2;// count number of 1 in indexinput
            }
            else if (indexinput[j] == 0) {
              if (num0 == i) break;
              num0++;
            }
          }
      Value topArg = block.getArgument(num1+i);
      //llvm::errs() << "[DBG] Creating channel for input #" << num1+i << " with type " << elementType << "\n";

      SmallVector<Value> recvPorts;
      SmallVector<Value> sendPorts;                 

    for (size_t r = 0; r < regions.size(); ++r) {
      auto ch = builder.create<dfg::ChannelOp>(builder.getUnknownLoc(), elementType, /*depth=*/1);
      Value recv = ch.getResult(0); // to be connected to region
      Value send = ch.getResult(1); // from sbox
      recvPorts.push_back(recv);
      sendPorts.push_back(send);
      //llvm::errs() << "  Region " << r << " Channel: recv=" << recv << ", send=" << send << "\n";
    }    
    auto outType = mlir::dyn_cast<dfg::OutputType>(topArg.getType());
    auto intType =mlir::dyn_cast<mlir::IntegerType>(outType.getElementType());
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
    //llvm::errs() << "[DBG] Instantiated sbox12int32 for input #" <<num1+i << "\n";

    }

    builder.setInsertionPointToStart(&block);
    SmallVector<SmallVector<Value>> outputChannels(sharedoutputs.size());

    // Create channels for shared outputs
    for (size_t i = 0; i < sharedoutputs.size(); ++i) {
      Type elementType = mlir::cast<dfg::InputType>(sharedoutputs[i]).getElementType();
      unsigned int num1= 0,num0=0;
      for (unsigned j = 0; j < minoutputs; ++j) {
            if (indexoutput[j] == 1) {
              num1+=2;// count number of 1 in indexoutput
            }
            else if (indexoutput[j] == 0) {
              if (num0 == i) break;
              num0++;
            }
      }
      Value topArg = block.getArgument(allInputs.size() + i + num1);
      //llvm::errs() << "[DBG] Creating channel for output #" << i << " arg # " << allInputs.size() + i + num1 << "\n";
      //llvm::errs() << "[DBG] Creating channel for output #" << i << " with type " << elementType << "\n";

      SmallVector<Value> recvPorts;
      SmallVector<Value> sendPorts;                 

    for (size_t r = 0; r < regions.size(); ++r) {
      auto ch = builder.create<dfg::ChannelOp>(builder.getUnknownLoc(), elementType, /*depth=*/1);

      Value recv = ch.getResult(0); // to be connected to region
      Value send = ch.getResult(1); // from sbox

      recvPorts.push_back(recv);
      sendPorts.push_back(send);

      //llvm::errs() << "  Region " << r << " Channel: recv=" << recv << ", send=" << send << "\n";
    }  

    auto outType = mlir::dyn_cast<dfg::InputType>(topArg.getType());
    auto intType =mlir::dyn_cast<mlir::IntegerType>(outType.getElementType());
    unsigned width = intType.getWidth();

    // Instantiate sbox12int32
    //llvm::errs() << "[DBG]  topArg: " << topArg.getType() << " recvPorts=" << recvPorts[0].getType()<< " , "<<recvPorts[1] << "\n";
    //llvm::errs() << "[DBG]  topArg: " << topArg.getType() << " sendPorts=" << sendPorts[0].getType()<< " , "<<sendPorts[1] << "\n";

    builder.create<dfg::InstantiateOp>(
            builder.getUnknownLoc(),
            "sbox2x1int" + std::to_string(width),
            sendPorts, // multiple input ports from regions
            ValueRange{topArg}, // one output to top-level            
            /*offloaded=*/false);
    outputChannels[i] = recvPorts; // Each region will use outputChannels[i][r]
    //llvm::errs() << "[DBG] Instantiated sbox2x1int32 for output #" << i << "\n";

    }

    // Clone region bodies
    for (size_t r = 0; r < regions.size(); ++r) {
      auto &srcBB = regions[r].getBody().front();
      mlir::IRMapping mapping;
      auto ft = regions[r].getFunctionType();
      //llvm::errs() << "[DBG] Cloning region " << r << " (@" << regions[r].getSymName() << ") input numbers: "<< ft.getNumInputs()<<"\n";
      // Map inputs - all regions now use channel outputs

      //shared and unshared inputs
      unsigned int num1= 0,num0=0;
      for (unsigned i = 0; i < minInputs; ++i) {
        if (indexinput[i] == 0){
          num0++;
          Value target = inputChannels[i-num1][r];
          mapping.map(srcBB.getArgument(i), target); ////
          //llvm::errs() << "[DBG] Map sh input arg " << i << " (" << srcBB.getArgument(i) << ") -> " << target << " (type: " << target.getType() << ")\n";
        } 
        else {
          
          Value target = block.getArgument(num1*2+num0+r);
          mapping.map(srcBB.getArgument(i), target);
          //llvm::errs() << "[DBG] Mapped unshared input arg " << i << " *(" << srcBB.getArgument(i) << ")* -> " << target << " (type: " << target.getType() << ")\n";
          num1++;

        }
    }
      //indepen inputs
      for (unsigned i =  minInputs; i < ft.getNumInputs(); ++i) {         
          Value target = block.getArgument(i+unsharedInputs);
          mapping.map(srcBB.getArgument(i), target);
          //llvm::errs() << "[DBG] Mapped independ input arg " << i << " *(" << srcBB.getArgument(i) << ")* -> " << target << " (type: " << target.getType() << ")\n";
        }
      // shared and unshared Map outputs
      num1= 0;
      num0=0;
      for (unsigned o = 0; o < minoutputs; ++o) {
        if (indexoutput[o] == 0){
          num0++;
          Value target = outputChannels[o-num1][r]; // region-specific 
          Value src = srcBB.getArgument(ft.getNumInputs() + o);     
          mapping.map(src, target);
          //llvm::errs() << "[DBG] Mapped shared output arg " << ft.getNumInputs() + o << " (" << src << ") -> block arg #" <<  " (" << target << ")\n";
        }
        else {
          Value target = block.getArgument(allInputs.size() + num0 + r + num1*2);
          Value src = srcBB.getArgument(ft.getNumInputs() + o);      
          mapping.map(src, target);
          //llvm::errs() << "[DBG] Mapped unshared output arg " << ft.getNumInputs() + o << " (" << src << ") -> block arg #" << target << " (type: " << target.getType() << ")\n";
          num1++;
        }
      }
      // independ Map outputs
      for (unsigned o = minoutputs; o < ft.getNumResults(); ++o) {
        unsigned outputArgIdx = maxInputs + unsharedoutputs + o;      
        mapping.map(srcBB.getArgument(ft.getNumInputs() + o),
                    block.getArgument(outputArgIdx+unsharedoutputs));
        //llvm::errs() << "[DBG] Mapped independ output arg " <<ft.getNumInputs() +  o << " (" << srcBB.getArgument(ft.getNumInputs() + o) << ") -> block arg #" << outputArgIdx +unsharedoutputs<< " ("<< block.getArgument(outputArgIdx+unsharedoutputs) << ")\n";
      }

      // Clone operations
      for (auto &op : srcBB.getOperations()) {
        builder.setInsertionPointToEnd(&block);
        Operation *cloned = builder.clone(op, mapping);
       //llvm::errs() << "[DBG] Cloned op: " << op.getName() << "\n";
        for (unsigned i = 0; i < op.getNumResults(); ++i) {
          //llvm::errs() << "[DBG] Mapping result " << i << ": " << op.getResult(i)<< " -> " << cloned->getResult(i) << "\n";        
          mapping.map(op.getResult(i), cloned->getResult(i));
        }
      }

    }

    for (auto &r : regions) r.erase();
    llvm::errs() << "[INFO] Completed merging regions\n";
    // Print the final MLIR after merging
    //llvm::outs() << "[INFO] Final MLIR after merging regions:\n";
    //module.print(llvm::outs());
    //llvm::outs() << "\n";

}

};
} // namespace


std::unique_ptr<Pass> mlir::dfg::createMergingRegionsPass() {
  return std::make_unique<MergingRegionsPass>();
}


