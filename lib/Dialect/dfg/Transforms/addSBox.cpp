#include "dfg-mlir/Dialect/dfg/Transforms/addSBox.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "dfg-mlir/Dialect/dfg/Transforms/mergingregions.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Block.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
using namespace mlir;
//MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(mlir::dfg::InsertSBoxPass)
namespace {
struct InsertSBoxPass : public PassWrapper<InsertSBoxPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertSBoxPass) 
  
    void runOnOperation() override {
      ModuleOp module = getOperation();
      OpBuilder builder(module.getContext());
      SymbolTable sym(module);
      SmallVector<dfg::RegionOp> regions;
      bool foundSBox2x1 = false;
      bool foundSBox1x2 = false;

      module.walk([&](dfg::RegionOp reg) {
      // llvm::errs() << "[DBG] Found region @" << reg.getSymName() << " with type: "<< reg.getFunctionType() << "\n";
        regions.push_back(reg);
      });

      if (regions.size() < 2) {
        llvm::errs() << "[DBG] Less than 2 regions; skipping adding sbox.\n";
        return;
      }

      if (sym.lookup("sbox1x2int32")) {
        llvm::errs() << "[DEBUG] Operator already defined: @sbox12int32\n";
        foundSBox1x2 = true;
      }

      if (sym.lookup("sbox2x1int32")) {
        llvm::errs() << "[DEBUG] Operator already defined: @sbox2x1int32\n";
        foundSBox2x1 = true; 
      }
      if(!foundSBox1x2){
        //dfg::RegionOp *maxInputRegion = nullptr;

        unsigned minInputs = std::numeric_limits<unsigned>::max();
        for (auto &reg : regions) {
          unsigned count = reg.getFunctionType().getNumInputs();
          if (count < minInputs)
            minInputs = count;
        }
        std::vector<unsigned int> indexinput(minInputs, 0);
        SmallVector<Type> sharedInputs;
        for (unsigned i = 0; i < minInputs; ++i) {
        // Get the type at index i from the first region that has enough inputs
        std::optional<Type> refType;
        for (auto &reg : regions) {
          auto types = reg.getFunctionType().getInputs();
          //llvm::errs() << "[DBG] Found region @" <<reg.getSymName() << " number of input : "<< i << " with type: "<< types <<"\n";
          if (i < types.size()) {
            refType = types[i];
            //llvm::errs() << "[DBG] refType : " << *refType << "\n";
            break;
          }
        } 
        for (auto &reg : regions) {
          auto inputs = reg.getFunctionType().getInputs();

          if (inputs[i] != *refType) {
            //llvm::errs() << "[DBG] Not shared in region @" << reg.getSymName() << "\n";
            indexinput[i] = 1;
            break;
          }
        }
        sharedInputs.push_back(*refType);
        } 
        //auto maxInputTypes = maxInputRegion->getFunctionType();
        //SmallVector<Type> sharedInputs(maxInputTypes.getInputs().begin(),maxInputTypes.getInputs().begin()+ minInputs);
        //for(unsigned i=0 ; i < minInputs; i++)
          //llvm::errs() << "[DBG] indexinput["<<i<<"] : "<<indexinput[i]<<"\n";
        //for(unsigned i=0 ; i < sharedInputs.size(); i++)
          //llvm::errs() << "[DBG] sharedInputs["<<i<<"] : "<<sharedInputs[i]<<"\n";
        for (unsigned i=0 ; i < minInputs; i++) {
          if (indexinput[i] == 0) {
            // Only do this once:
            Type elementType = mlir::cast<dfg::OutputType>(sharedInputs[i]).getElementType();
            std::string typeStr;
            llvm::raw_string_ostream os(typeStr);
            elementType.print(os);
            os.flush();
            if (typeStr.starts_with("i") && isdigit(typeStr[1]))
                  typeStr.insert(1, "nt"); // i32 -> int32, i16 -> int16
            // Now concatenate
            std::string nameSbox = "sbox1x2" + typeStr;
            //llvm::errs() << "[DBG] sbox name: "<<nameSbox<<"\n";
            SymbolTable sym2(module);
            if (!sym2.lookup<dfg::ProcessOp>(nameSbox)) {
              // --- 1) Build the ProcessOp ---
              builder.setInsertionPointToEnd(module.getBody());
              auto inTy  = dfg::OutputType::get(module.getContext(), elementType);
              auto outTy = dfg::InputType::get(module.getContext(), elementType);
              auto fnTy  = builder.getFunctionType(/*inputs=*/inTy,/*outputs=*/ArrayRef<Type>{outTy, outTy});
              auto proc = builder.create<dfg::ProcessOp>(builder.getUnknownLoc(),
                  builder.getStringAttr(nameSbox),fnTy);

              // --- 2) Entry block for the ProcessOp ---
              Region &pr = proc.getRegion();
              pr.getBlocks().clear();                   // ditch any existing blocks
              auto *entry = new Block;                  // allocate an empty block
              pr.push_back(entry);                      // append it

              Location loc = builder.getUnknownLoc();
              // now add exactly 3 arguments *with* locations
              Value pIn   = entry->addArgument(inTy,  loc);
              Value pOut0 = entry->addArgument(outTy, loc);
              Value pOut1 = entry->addArgument(outTy, loc);

              builder.setInsertionPointToStart(entry);
              // constant 1
              auto c1 = builder.create<arith::ConstantOp>(loc, mlir::IntegerAttr::get(elementType, 1));
            //llvm::errs() << "[DEBUG]" <<mlir::IntegerAttr::get(elementType, 1)<<"\n";
              // --- 3) Build the LoopOp inside that block ---
              auto loop = builder.create<dfg::LoopOp>(
                  loc,
                  ValueRange{pIn},                 // feed in pIn
                  ValueRange{pOut0, pOut1});       // two loop-carried outputs

              // carve out its single body block
              Region &lr = loop.getRegion();
              lr.getBlocks().clear();
              auto *body = new Block;
              lr.push_back(body);

              // Use in0, out0, out1 for clarity
              Value pulled = builder.create<dfg::PullOp>(loc, elementType, pIn);
              Value added  = builder.create<arith::AddIOp>(loc, pulled, c1);
              builder.create<dfg::PushOp>(loc, added, pOut0);
              builder.create<dfg::PushOp>(loc, added, pOut1);
              //builder.create<dfg::YieldOp>(loc, ValueRange{});


        }}}
        /*
          SymbolTable symTable(module);
          if (auto sboxOp = symTable.lookup<dfg::ProcessOp>("sbox1x2int16")) {
          llvm::errs() << "[DEBUG] Operator @sbox12int32 found. Dumping contents:\n";

          // Dump the entire operation
          sboxOp.dump();

          // Optional: walk through its operations
          sboxOp.getBody().walk([&](Operation *op) {
              llvm::errs() << "  - " << op->getName() << "\n";
              llvm::errs() << "    Operands:\n";
              for (auto operand : op->getOperands())
              llvm::errs() << "      - " << operand << " (type: " << operand.getType() << ")\n";
              llvm::errs() << "    Results:\n";
              for (auto result : op->getResults())
              llvm::errs() << "      - " << result << " (type: " << result.getType() << ")\n";
          });
          }*/
      }
      if(!foundSBox2x1){

        unsigned minoutputs = std::numeric_limits<unsigned>::max();
        for (auto &reg : regions) {
          unsigned count = reg.getFunctionType().getNumResults();
          if (count < minoutputs)
            minoutputs = count;
        }
        std::vector<unsigned int> indexoutput(minoutputs, 0);
        SmallVector<Type> sharedoutputs;
        for (unsigned i = 0; i < minoutputs; ++i) {
        // Get the type at index i from the first region that has enough inputs
        std::optional<Type> refType;
        for (auto &reg : regions) {
          auto types = reg.getFunctionType().getResults();
          //llvm::errs() << "[DBG] Found region @" <<reg.getSymName() << " number of input : "<< i << " with type: "<< types <<"\n";
          if (i < types.size()) {
            refType = types[i];
            //llvm::errs() << "[DBG] refType : " << *refType << "\n";
            break;
          }
        } 
        for (auto &reg : regions) {
          auto outputs = reg.getFunctionType().getResults();

          if (outputs[i] != *refType) {
            //llvm::errs() << "[DBG] Not shared in region @" << reg.getSymName() << "\n";
            indexoutput[i] = 1;
            break;
          }
        }
        sharedoutputs.push_back(*refType);
        }
        //for(unsigned i=0 ; i < minoutputs; i++)
          //llvm::errs() << "[DBG] sharedOutputs["<<i<<"] : "<<sharedoutputs[i]<<"\n";

        for (unsigned i=0 ; i < minoutputs; i++) {
          if (indexoutput[i] == 0) {
            // Only do this once:
            Type elementType = mlir::cast<dfg::InputType>(sharedoutputs[i]).getElementType();
            std::string typeStr;
            llvm::raw_string_ostream os(typeStr);

            elementType.print(os);
            os.flush();
            if (typeStr.starts_with("i") && isdigit(typeStr[1]))
                  typeStr.insert(1, "nt"); // i32 -> int32, i16 -> int16
            // Now concatenate
            std::string nameSbox = "sbox2x1" + typeStr;
            //llvm::errs() << "[DBG] sbox name: "<<nameSbox<<"\n";
            SymbolTable sym2(module);
            if (!sym2.lookup<dfg::ProcessOp>(nameSbox)) {
              //llvm::errs() << "[DBG] in sbox name: "<<nameSbox<<"\n";
              // --- 1) Build the ProcessOp ---
              builder.setInsertionPointToEnd(module.getBody());
              auto inTy  = dfg::OutputType::get(module.getContext(), elementType);
              auto outTy = dfg::InputType::get(module.getContext(), elementType);
              auto fnTy  = builder.getFunctionType(
                /*inputs=*/ArrayRef<Type>{inTy, inTy},
                /*outputs=*/outTy);

              auto proc = builder.create<dfg::ProcessOp>(
                  builder.getUnknownLoc(),
                  builder.getStringAttr(nameSbox),
                  fnTy);

              // --- 2) Entry block for the ProcessOp ---
              Region &pr = proc.getRegion();
              pr.getBlocks().clear();                   // ditch any existing blocks
              auto *entry = new Block;                  // allocate an empty block
              pr.push_back(entry);                      // append it

              Location loc = builder.getUnknownLoc();
              // now add exactly 3 arguments *with* locations
              Value pIn0   = entry->addArgument(inTy,  loc);
              Value pIn1   = entry->addArgument(inTy,  loc);
              Value pOut = entry->addArgument(outTy, loc);

              builder.setInsertionPointToStart(entry);
              // constant 1

              // --- 3) Build the LoopOp inside that block ---
              auto loop = builder.create<dfg::LoopOp>(
                  loc,
                  ValueRange{pIn0,pIn1},                 // feed in pIn
                  ValueRange{pOut});       // two loop-carried outputs

              // carve out its single body block
              Region &lr = loop.getRegion();
              lr.getBlocks().clear();
              auto *body = new Block;
              lr.push_back(body);

              // Use in0, out0, out1 for clarity
              Value pulled0 = builder.create<dfg::PullOp>(loc, elementType, pIn0);
              Value pulled1 = builder.create<dfg::PullOp>(loc, elementType, pIn1);

              Value added  = builder.create<arith::AddIOp>(loc, pulled0, pulled1);
              builder.create<dfg::PushOp>(loc, added, pOut);
              //builder.create<dfg::YieldOp>(loc, ValueRange{});


          }}}
        /*
          SymbolTable symTable(module);
          if (auto sboxOp = symTable.lookup<dfg::ProcessOp>("sbox1x2int16")) {
          llvm::errs() << "[DEBUG] Operator @sbox12int32 found. Dumping contents:\n";

          // Dump the entire operation
          sboxOp.dump();

          // Optional: walk through its operations
          sboxOp.getBody().walk([&](Operation *op) {
              llvm::errs() << "  - " << op->getName() << "\n";
              llvm::errs() << "    Operands:\n";
              for (auto operand : op->getOperands())
              llvm::errs() << "      - " << operand << " (type: " << operand.getType() << ")\n";
              llvm::errs() << "    Results:\n";
              for (auto result : op->getResults())
              llvm::errs() << "      - " << result << " (type: " << result.getType() << ")\n";
          });
          }*/
      }






}

  StringRef getArgument() const final { return "insert-sbox"; }
  StringRef getDescription() const final { return "Ensure sbox12int32 operator+process exist"; }
};
} // namespace

std::unique_ptr<Pass> mlir::dfg::createInsertSBoxPass() {
  return std::make_unique<InsertSBoxPass>();
}
