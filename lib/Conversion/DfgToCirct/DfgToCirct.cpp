/// Implementation of DfgToCirct pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@mailbox.tu-dresden.de)

#include "dfg-mlir/Conversion/DfgToCirct/DfgToCirct.h"

#include "../PassDetails.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FSM/FSMDialect.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/APInt.h"

using namespace mlir;
using namespace mlir::dfg;
using namespace circt;

namespace {

// Helper to determine which new argument to use
SmallVector<std::pair<int, Value>> oldArgsIndex;
SmallVector<std::pair<Value, Value>> newArguments;
SmallVector<std::pair<Operation*, StringAttr>> newOperators;

fsm::MachineOp insertController(
    ModuleOp module,
    std::string name,
    int numPullChan,
    FunctionType funcTy,
    SmallVector<Operation*> ops)
{
    auto context = module.getContext();
    OpBuilder builder(context);
    builder.setInsertionPointToStart(module.getBody());
    auto loc = builder.getUnknownLoc();

    // Create a new fsm.machine
    auto machine = builder.create<fsm::MachineOp>(loc, name, "INIT", funcTy);
    builder.setInsertionPointToStart(&machine.getBody().front());

    // Create constants and variables
    auto i1Ty = builder.getI1Type();
    auto c_false = builder.create<hw::ConstantOp>(loc, i1Ty, 0);
    size_t numPull = 0;
    SmallVector<Value> pullVars;
    SmallVector<Value> calcVars;
    SmallVector<std::pair<Value, int>> zeroWidth;
    int numCalculation = 0;
    // Now assume that every computation takes only one cycle
    for (size_t i = 0; i < ops.size(); i++) {
        auto op = ops[i];
        if (auto pullOp = dyn_cast<PullOp>(op)) {
            assert(i <= numPull && "must pull every data at beginning");
            auto valueTy = pullOp.getChan().getType().getElementType();
            auto varOp = builder.create<fsm::VariableOp>(
                loc,
                valueTy,
                builder.getIntegerAttr(valueTy, 0),
                builder.getStringAttr("data" + std::to_string(numPull++)));
            pullVars.push_back(varOp.getResult());
            newArguments.push_back(
                std::make_pair(varOp.getResult(), pullOp.getOutp()));
        } else if (auto pushOp = dyn_cast<PushOp>(op)) {
            continue;
        } else {
            auto types = op->getResultTypes();
            auto numVars = types.size();
            for (size_t i = 0; i < numVars; i++) {
                auto type = types[i];
                auto varName =
                    "result" + std::to_string(numCalculation++)
                    + ((numVars == 1) ? "" : "_" + std::to_string(i));
                auto varOp = builder.create<fsm::VariableOp>(
                    loc,
                    type,
                    builder.getIntegerAttr(type, 0),
                    builder.getStringAttr(varName));
                calcVars.push_back(varOp.getResult());
                newArguments.push_back(
                    std::make_pair(varOp.getResult(), op->getResult(i)));
            }
        }
    }

    // All zero vector
    std::vector<Value> outputAllZero;
    for (const auto out : machine.getFunctionType().getResults()) {
        auto outWidth = dyn_cast<IntegerType>(out).getWidth();
        if (outWidth == 1)
            outputAllZero.push_back(c_false.getResult());
        else if (
            const auto zeroExist =
                getNewIndexOrArg<Value, int>(outWidth, zeroWidth)) {
            outputAllZero.push_back(zeroExist.value());
        } else {
            auto zero = builder.create<hw::ConstantOp>(
                loc,
                builder.getIntegerType(outWidth),
                0);
            zeroWidth.push_back(std::make_pair(zero.getResult(), outWidth));
            outputAllZero.push_back(zero.getResult());
        }
    }

    // Create states and transitions
    // INIT
    auto stateInit = builder.create<fsm::StateOp>(loc, "INIT");
    builder.setInsertionPointToEnd(&stateInit.getOutput().back());
    stateInit.getOutput().front().front().erase();
    builder.create<fsm::OutputOp>(loc, ArrayRef<Value>(outputAllZero));
    builder.setInsertionPointToEnd(&stateInit.getTransitions().back());
    builder.create<fsm::TransitionOp>(loc, "READ0");
    builder.setInsertionPointToEnd(&machine.getBody().back());

    // All pulls are at beginning of an operator
    // For every PullOp there are two states: READ and WAIT
    auto isNextPush = isa<PushOp>(ops[numPull]);
    for (size_t i = 0; i < pullVars.size(); i++) {
        auto pullOp = dyn_cast<PullOp>(ops[i]);
        auto argIndex =
            getNewIndexOrArg<int, Value>(pullOp.getChan(), oldArgsIndex)
                .value();

        auto stateRead =
            builder.create<fsm::StateOp>(loc, "READ" + std::to_string(i));
        builder.setInsertionPointToEnd(&stateRead.getOutput().back());
        stateRead.getOutput().front().front().erase();
        builder.create<fsm::OutputOp>(loc, ArrayRef<Value>(outputAllZero));
        builder.setInsertionPointToEnd(&stateRead.getTransitions().back());
        auto transRead = builder.create<fsm::TransitionOp>(
            loc,
            "WAIT_IN" + std::to_string(i));
        builder.setInsertionPointToEnd(transRead.ensureGuard(builder));
        transRead.getGuard().front().front().erase();
        builder.create<fsm::ReturnOp>(loc, machine.getArgument(2 * argIndex));
        builder.setInsertionPointToEnd(transRead.ensureAction(builder));
        builder.create<fsm::UpdateOp>(
            loc,
            pullVars[i],
            machine.getArgument(2 * argIndex + 1));
        builder.setInsertionPointToEnd(&machine.getBody().back());

        auto stateWait =
            builder.create<fsm::StateOp>(loc, "WAIT_IN" + std::to_string(i));
        builder.setInsertionPointToEnd(&stateWait.getOutput().back());
        stateWait.getOutput().front().front().erase();
        std::vector<Value> newOutputs = outputAllZero;
        newOutputs[argIndex] = machine.getArgument(2 * argIndex);
        builder.create<fsm::OutputOp>(loc, ArrayRef<Value>(newOutputs));
        builder.setInsertionPointToEnd(&stateWait.getTransitions().back());
        auto transWait = builder.create<fsm::TransitionOp>(
            loc,
            (i == pullVars.size() - 1) ? (isNextPush ? "WRITE0" : "CALC0")
                                       : "READ" + std::to_string(i + 1));
        builder.setInsertionPointToEnd(transWait.ensureGuard(builder));
        transWait.getGuard().front().front().erase();
        builder.create<fsm::ReturnOp>(loc, machine.getArgument(2 * argIndex));
        builder.setInsertionPointToEnd(&machine.getBody().back());
    }

    // TODO: Dependency Graph, merge states
    // TODO: multiple cycles computation
    int idxCalc = 0;
    int idxPush = 0;
    isNextPush = false;
    for (size_t i = numPull; i < ops.size(); i++) {
        auto op = ops[i];
        auto isThisPush = dyn_cast<PushOp>(op);
        if (i < ops.size() - 1) isNextPush = isa<PushOp>(ops[i + 1]);

        // For every push create one STATE: WRITE
        if (isThisPush) {
            auto argIndex =
                getNewIndexOrArg<int, Value>(isThisPush.getChan(), oldArgsIndex)
                    .value();

            auto stateWrite = builder.create<fsm::StateOp>(
                loc,
                "WRITE" + std::to_string(idxPush));
            builder.setInsertionPointToEnd(&stateWrite.getOutput().back());
            stateWrite.getOutput().front().front().erase();
            std::vector<Value> newOutputs = outputAllZero;
            newOutputs[2 * argIndex - numPullChan] =
                machine.getArgument(argIndex + numPullChan);
            newOutputs[2 * argIndex - numPullChan + 1] =
                getNewIndexOrArg<Value, Value>(
                    isThisPush.getInp(),
                    newArguments)
                    .value();
            builder.create<fsm::OutputOp>(loc, ArrayRef<Value>(newOutputs));
            builder.setInsertionPointToEnd(&stateWrite.getTransitions().back());
            auto transWrite = builder.create<fsm::TransitionOp>(
                loc,
                (i == ops.size() - 1)
                    ? "INIT"
                    : (isNextPush ? "WRITE" + std::to_string(idxPush + 1)
                                  : "CALC" + std::to_string(idxCalc)));
            builder.setInsertionPointToEnd(transWrite.ensureGuard(builder));
            transWrite.getGuard().front().front().erase();
            builder.create<fsm::ReturnOp>(
                loc,
                machine.getArgument(argIndex + numPullChan));
            builder.setInsertionPointToEnd(&machine.getBody().back());
            idxPush++;
        }
        // For every computation create one STATE: CALC
        else {
            auto stateCal = builder.create<fsm::StateOp>(
                loc,
                "CALC" + std::to_string(idxCalc));
            builder.setInsertionPointToEnd(&stateCal.getOutput().back());
            stateCal.getOutput().front().front().erase();
            builder.create<fsm::OutputOp>(loc, ArrayRef<Value>(outputAllZero));
            builder.setInsertionPointToEnd(&stateCal.getTransitions().back());
            auto transCal = builder.create<fsm::TransitionOp>(
                loc,
                (i == ops.size() - 1)
                    ? "INIT"
                    : (isNextPush ? "WRITE" + std::to_string(idxPush)
                                  : "CALC" + std::to_string(idxCalc + 1)));
            builder.setInsertionPointToEnd(transCal.ensureAction(builder));
            auto newCalcOp = op->clone();
            for (size_t j = 0; j < newCalcOp->getNumOperands(); j++) {
                auto newOperand = getNewIndexOrArg<Value, Value>(
                                      newCalcOp->getOperand(j),
                                      newArguments)
                                      .value();
                newCalcOp->setOperand(j, newOperand);
            }
            builder.insert(newCalcOp);
            builder.create<fsm::UpdateOp>(
                loc,
                calcVars[idxCalc],
                newCalcOp->getResult(0));
            builder.setInsertionPointToEnd(&machine.getBody().back());
            idxCalc++;
        }
    }

    return machine;
}

struct ConvertOperator : OpConversionPattern<OperatorOp> {
public:
    using OpConversionPattern<OperatorOp>::OpConversionPattern;

    ConvertOperator(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<OperatorOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        OperatorOp op,
        OperatorOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto funcTy = op.getFunctionType();
        auto numInputs = funcTy.getNumInputs();
        auto numOutputs = funcTy.getNumResults();

        auto args = op.getBody().getArguments();
        auto types = op.getBody().getArgumentTypes();
        size_t size = types.size();

        SmallVector<hw::PortInfo> ports;
        int in_num = 1;
        int out_num = 1;

        // Add clock port.
        hw::PortInfo clock;
        clock.name = rewriter.getStringAttr("clock");
        clock.dir = hw::ModulePort::Direction::Input;
        clock.type = rewriter.getI1Type();
        clock.argNum = 0;
        ports.push_back(clock);

        // Add reset port.
        hw::PortInfo reset;
        reset.name = rewriter.getStringAttr("reset");
        reset.dir = hw::ModulePort::Direction::Input;
        reset.type = rewriter.getI1Type();
        reset.argNum = 1;
        ports.push_back(reset);

        for (size_t i = 0; i < size; i++) {
            const auto type = types[i];
            std::string name;

            if (const auto inTy = dyn_cast<OutputType>(type)) {
                auto elemTy = dyn_cast<IntegerType>(inTy.getElementType());
                assert(elemTy && "only integers are supported on hardware");
                // Add ready for input port
                hw::PortInfo in_ready;
                name = ((numInputs == 1) ? "in" : "in" + std::to_string(in_num))
                       + "_ready";
                in_ready.name = rewriter.getStringAttr(name);
                in_ready.dir = hw::ModulePort::Direction::Output;
                in_ready.type = rewriter.getI1Type();
                in_ready.argNum = in_num + 1;
                ports.push_back(in_ready);
                // Add valid for input port
                hw::PortInfo in_valid;
                name = ((numInputs == 1) ? "in" : "in" + std::to_string(in_num))
                       + "_valid";
                in_valid.name = rewriter.getStringAttr(name);
                in_valid.dir = hw::ModulePort::Direction::Input;
                in_valid.type = rewriter.getI1Type();
                in_valid.argNum = in_num + 1;
                ports.push_back(in_valid);
                // Add bits for input port
                hw::PortInfo in_bits;
                name = ((numInputs == 1) ? "in" : "in" + std::to_string(in_num))
                       + "_bits";
                in_bits.name = rewriter.getStringAttr(name);
                in_bits.dir = hw::ModulePort::Direction::Input;
                in_bits.type = rewriter.getIntegerType(elemTy.getWidth());
                in_bits.argNum = in_num + 1;
                ports.push_back(in_bits);
                // Index increment
                in_num++;
            } else if (const auto outTy = dyn_cast<InputType>(type)) {
                auto elemTy = dyn_cast<IntegerType>(outTy.getElementType());
                assert(elemTy && "only integers are supported on hardware");
                // Add ready for output port
                hw::PortInfo out_ready;
                name = ((numOutputs == 1) ? "out"
                                          : "out" + std::to_string(out_num))
                       + "_ready";
                out_ready.name = rewriter.getStringAttr(name);
                out_ready.dir = hw::ModulePort::Direction::Input;
                out_ready.type = rewriter.getI1Type();
                out_ready.argNum = out_num;
                ports.push_back(out_ready);
                // Add valid for output port
                hw::PortInfo out_valid;
                name = ((numOutputs == 1) ? "out"
                                          : "out" + std::to_string(out_num))
                       + "_valid";
                out_valid.name = rewriter.getStringAttr(name);
                out_valid.dir = hw::ModulePort::Direction::Output;
                out_valid.type = rewriter.getI1Type();
                out_valid.argNum = out_num;
                ports.push_back(out_valid);
                // Add bits for output port
                hw::PortInfo out_bits;
                name = ((numOutputs == 1) ? "out"
                                          : "out" + std::to_string(out_num))
                       + "_bits";
                out_bits.name = rewriter.getStringAttr(name);
                out_bits.dir = hw::ModulePort::Direction::Output;
                out_bits.type = rewriter.getIntegerType(elemTy.getWidth());
                out_bits.argNum = out_num;
                ports.push_back(out_bits);
                // Index increment
                out_num++;
            }
        }

        // Create new HWModule
        auto hwModule = rewriter.create<hw::HWModuleOp>(
            op.getLoc(),
            op.getSymNameAttr(),
            hw::ModulePortInfo(ports),
            ArrayAttr{},
            ArrayRef<NamedAttribute>{},
            StringAttr{},
            false);
        newOperators.push_back(
            std::make_pair(hwModule.getOperation(), op.getSymNameAttr()));

        // Store the pair of old and new argument(s) in vector
        oldArgsIndex.clear();
        for (size_t i = 0; i < size; i++) {
            const auto arg = args[i];
            oldArgsIndex.push_back(std::make_pair(i, arg));
        }

        // Insert controller machine at top of module
        auto module = op->getParentOfType<ModuleOp>();
        SmallVector<Operation*> ops;
        for (auto &opi : op.getBody().getOps()) ops.push_back(&opi);
        auto hwFuncTy = hwModule.getFunctionType();
        auto hwInTypes = hwFuncTy.getInputs();
        auto hwOutTypes = hwFuncTy.getResults();
        // The machine inputs don't contain clock and reset
        ArrayRef<Type> fsmInTypes(hwInTypes.data() + 2, hwInTypes.size() - 2);

        auto newMachine = insertController(
            module,
            op.getSymName().str() + "_controller",
            numInputs,
            FunctionType::get(hwFuncTy.getContext(), fsmInTypes, hwOutTypes),
            ops);

        // Create a fsm.hw_instance op and take the results to output
        rewriter.setInsertionPointToStart(&hwModule.getBody().front());
        SmallVector<Value> fsmInputs;
        for (size_t i = 2; i < hwInTypes.size(); i++)
            fsmInputs.push_back(hwModule.getArgument(i));
        auto outputs = rewriter.create<fsm::HWInstanceOp>(
            rewriter.getUnknownLoc(),
            newMachine.getFunctionType().getResults(),
            rewriter.getStringAttr("controller"),
            newMachine.getSymNameAttr(),
            fsmInputs,
            hwModule.getArgument(0),
            hwModule.getArgument(1));

        rewriter.create<hw::OutputOp>(
            rewriter.getUnknownLoc(),
            outputs.getResults());

        rewriter.eraseOp(op);

        return success();
    }
};

std::map<std::string, hw::HWModuleOp> queueList;
hw::HWModuleOp
insertQueue(OpBuilder &builder, Location loc, unsigned portBit, unsigned size)
{
    auto suffix = std::to_string(size) + "x" + "i" + std::to_string(portBit);
    auto name = "queue_" + suffix;
    auto isExist = queueList.find(name);
    if (isExist != queueList.end()) return isExist->second;
    auto i1Ty = builder.getI1Type();

    SmallVector<hw::PortInfo> ports;
    // Add clock port.
    hw::PortInfo clock;
    clock.name = builder.getStringAttr("clock");
    clock.dir = hw::ModulePort::Direction::Input;
    clock.type = i1Ty;
    // clock.argNum = 0;
    ports.push_back(clock);
    // Add reset port.
    hw::PortInfo reset;
    reset.name = builder.getStringAttr("reset");
    reset.dir = hw::ModulePort::Direction::Input;
    reset.type = i1Ty;
    // reset.argNum = 1;
    ports.push_back(reset);
    // Add io_enq_valid
    hw::PortInfo io_enq_valid;
    io_enq_valid.name = builder.getStringAttr("io_enq_valid");
    io_enq_valid.dir = hw::ModulePort::Direction::Input;
    io_enq_valid.type = i1Ty;
    // io_enq_valid.argNum = 2;
    ports.push_back(io_enq_valid);
    // Add io_enq_bits
    hw::PortInfo io_enq_bits;
    io_enq_bits.name = builder.getStringAttr("io_enq_bits");
    io_enq_bits.dir = hw::ModulePort::Direction::Input;
    io_enq_bits.type = builder.getIntegerType(portBit);
    // io_enq_bits.argNum = 3;
    ports.push_back(io_enq_bits);
    // Add io_deq_ready
    hw::PortInfo io_deq_ready;
    io_deq_ready.name = builder.getStringAttr("io_deq_ready");
    io_deq_ready.dir = hw::ModulePort::Direction::Input;
    io_deq_ready.type = i1Ty;
    // io_deq_ready.argNum = 4;
    ports.push_back(io_deq_ready);
    // Add io_enq_ready
    hw::PortInfo io_enq_ready;
    io_enq_ready.name = builder.getStringAttr("io_enq_ready");
    io_enq_ready.dir = hw::ModulePort::Direction::Output;
    io_enq_ready.type = i1Ty;
    // io_enq_ready.argNum = 5;
    ports.push_back(io_enq_ready);
    // Add io_deq_valid
    hw::PortInfo io_deq_valid;
    io_deq_valid.name = builder.getStringAttr("io_deq_valid");
    io_deq_valid.dir = hw::ModulePort::Direction::Output;
    io_deq_valid.type = i1Ty;
    // io_deq_valid.argNum = 5;
    ports.push_back(io_deq_valid);
    // Add io_deq_bits
    hw::PortInfo io_deq_bits;
    io_deq_bits.name = builder.getStringAttr("io_deq_bits");
    io_deq_bits.dir = hw::ModulePort::Direction::Output;
    io_deq_bits.type = builder.getIntegerType(portBit);
    // io_deq_bits.argNum = 3;
    ports.push_back(io_deq_bits);
    // Create new HWModule
    auto hwModule = builder.create<hw::HWModuleOp>(
        loc,
        builder.getStringAttr(name),
        hw::ModulePortInfo(ports),
        ArrayAttr{},
        ArrayRef<NamedAttribute>{},
        StringAttr{},
        false);
    auto clk = hwModule.getArgument(0);
    auto rst = hwModule.getArgument(1);
    auto in_valid = hwModule.getArgument(2);
    auto in_bits = hwModule.getArgument(3);
    auto out_ready = hwModule.getArgument(4);
    builder.setInsertionPointToStart(&hwModule.getBodyRegion().front());

    // Constants
    auto c_true = builder.create<hw::ConstantOp>(loc, i1Ty, 1);
    auto c_false = builder.create<hw::ConstantOp>(loc, i1Ty, 0);
    unsigned bitDepth = std::ceil(std::log2(size));
    auto ptrTy = builder.getIntegerType(bitDepth);
    auto isTwoPower = size > 0 && (size & (size - 1)) == 0;
    auto c_widthZero = builder.create<hw::ConstantOp>(loc, ptrTy, 0);
    auto c_widthOne = builder.create<hw::ConstantOp>(loc, ptrTy, 1);
    hw::ConstantOp c_sizeMax;
    if (!isTwoPower)
        c_sizeMax = builder.create<hw::ConstantOp>(loc, ptrTy, size - 1);

    // RAM
    auto ram = builder.create<sv::RegOp>(
        loc,
        hw::UnpackedArrayType::get(builder.getIntegerType(portBit), size));
    auto placeholderReadIndex = builder.create<hw::ConstantOp>(loc, ptrTy, 0);
    auto ram_read = builder.create<sv::ArrayIndexInOutOp>(
        loc,
        ram.getResult(),
        placeholderReadIndex.getResult());
    auto ram_read_data =
        builder.create<sv::ReadInOutOp>(loc, ram_read.getResult());

    // Pointers
    auto ptr_write = builder.create<sv::RegOp>(loc, ptrTy);
    auto ptr_write_value =
        builder.create<sv::ReadInOutOp>(loc, ptr_write.getResult());
    auto ptr_read = builder.create<sv::RegOp>(loc, ptrTy);
    auto ptr_read_value =
        builder.create<sv::ReadInOutOp>(loc, ptr_read.getResult());
    placeholderReadIndex.replaceAllUsesWith(ptr_read_value.getResult());
    auto maybe_full = builder.create<sv::RegOp>(loc, i1Ty);
    auto maybe_full_value =
        builder.create<sv::ReadInOutOp>(loc, maybe_full.getResult());

    // Signals
    auto ptr_match = builder.create<comb::ICmpOp>(
        loc,
        comb::ICmpPredicate::eq,
        ptr_write_value.getResult(),
        ptr_read_value.getResult());
    auto _empty_T = builder.create<comb::XorOp>(
        loc,
        maybe_full_value.getResult(),
        c_true.getResult());
    auto empty = builder.create<comb::AndOp>(
        loc,
        ptr_match.getResult(),
        _empty_T.getResult());
    auto full = builder.create<comb::AndOp>(
        loc,
        ptr_match.getResult(),
        maybe_full_value.getResult());
    auto placeholderNotFull = builder.create<hw::ConstantOp>(loc, i1Ty, 0);
    auto do_enq = builder.create<comb::AndOp>(
        loc,
        placeholderNotFull.getResult(),
        in_valid);
    auto placeholderNotEmpty = builder.create<hw::ConstantOp>(loc, i1Ty, 0);
    auto do_deq = builder.create<comb::AndOp>(
        loc,
        out_ready,
        placeholderNotEmpty.getResult());
    auto next_write = builder.create<comb::AddOp>(
        loc,
        ptr_write_value.getResult(),
        c_widthOne.getResult());
    auto next_read = builder.create<comb::AddOp>(
        loc,
        ptr_read_value.getResult(),
        c_widthOne.getResult());
    auto notSameEnqDeq = builder.create<comb::ICmpOp>(
        loc,
        comb::ICmpPredicate::ne,
        do_enq.getResult(),
        do_deq.getResult());
    auto not_empty =
        builder.create<comb::XorOp>(loc, empty.getResult(), c_true.getResult());
    placeholderNotEmpty.replaceAllUsesWith(not_empty.getResult());
    auto not_full =
        builder.create<comb::XorOp>(loc, full.getResult(), c_true.getResult());
    placeholderNotFull.replaceAllUsesWith(not_full.getResult());

    // Clocked logic
    builder.create<sv::AlwaysOp>(loc, sv::EventControl::AtPosEdge, clk, [&] {
        builder.create<sv::IfOp>(
            loc,
            rst,
            [&] {
                builder.create<sv::PAssignOp>(
                    loc,
                    ptr_write.getResult(),
                    c_widthZero.getResult());
                builder.create<sv::PAssignOp>(
                    loc,
                    ptr_read.getResult(),
                    c_widthZero.getResult());
                builder.create<sv::PAssignOp>(
                    loc,
                    maybe_full.getResult(),
                    c_false.getResult());
            },
            [&] {
                builder.create<sv::IfOp>(loc, do_enq.getResult(), [&] {
                    auto ram_write = builder.create<sv::ArrayIndexInOutOp>(
                        loc,
                        ram.getResult(),
                        ptr_write_value.getResult());
                    builder.create<sv::PAssignOp>(
                        loc,
                        ram_write.getResult(),
                        in_bits);
                    if (isTwoPower) {
                        builder.create<sv::PAssignOp>(
                            loc,
                            ptr_write.getResult(),
                            next_write.getResult());
                    } else {
                        auto isSizeMax = builder.create<comb::ICmpOp>(
                            loc,
                            comb::ICmpPredicate::eq,
                            ptr_write_value.getResult(),
                            c_sizeMax.getResult());
                        builder.create<sv::IfOp>(
                            loc,
                            isSizeMax.getResult(),
                            [&] {
                                builder.create<sv::PAssignOp>(
                                    loc,
                                    ptr_write.getResult(),
                                    c_widthZero.getResult());
                            },
                            [&] {
                                builder.create<sv::PAssignOp>(
                                    loc,
                                    ptr_write.getResult(),
                                    next_write.getResult());
                            });
                    }
                });
                builder.create<sv::IfOp>(loc, do_deq.getResult(), [&] {
                    if (isTwoPower) {
                        builder.create<sv::PAssignOp>(
                            loc,
                            ptr_read.getResult(),
                            next_read.getResult());
                    } else {
                        auto isSizeMax = builder.create<comb::ICmpOp>(
                            loc,
                            comb::ICmpPredicate::eq,
                            ptr_read_value.getResult(),
                            c_sizeMax.getResult());
                        builder.create<sv::IfOp>(
                            loc,
                            isSizeMax.getResult(),
                            [&] {
                                builder.create<sv::PAssignOp>(
                                    loc,
                                    ptr_read.getResult(),
                                    c_widthZero.getResult());
                            },
                            [&] {
                                builder.create<sv::PAssignOp>(
                                    loc,
                                    ptr_read.getResult(),
                                    next_read.getResult());
                            });
                    }
                });
                builder.create<sv::IfOp>(loc, notSameEnqDeq.getResult(), [&] {
                    builder.create<sv::PAssignOp>(
                        loc,
                        maybe_full.getResult(),
                        do_enq.getResult());
                });
            });
    });

    // Clean up placeholders
    placeholderReadIndex.erase();
    placeholderNotEmpty.erase();
    placeholderNotFull.erase();

    // Output
    SmallVector<Value> outputs;
    outputs.push_back(not_full.getResult());
    outputs.push_back(not_empty.getResult());
    outputs.push_back(ram_read_data.getResult());
    builder.create<hw::OutputOp>(loc, outputs);
    queueList.emplace(name, hwModule);
    return hwModule;
}

SmallVector<std::pair<Value, Value>> newConnections;
SmallVector<std::pair<int, Value>> oldOutputs;
SmallVector<Value> newOutputs;
SmallVector<std::pair<SmallVector<Value>, Value>> newOutputBundles;
SmallVector<Value> willBeConnected;
SmallVector<std::pair<Value, Value>> placeholderInputs;
SmallVector<std::pair<SmallVector<Value>, Value>> placeholderOutputs;
SmallVector<std::pair<SmallVector<Value>, Value>> instanceInputs;
SmallVector<std::pair<Value, Value>> instanceOutputs;
std::map<std::string, int> instantiatedOperators;
int getNameOfInstance(std::string to_find)
{
    auto it = instantiatedOperators.find(to_find);
    if (it != instantiatedOperators.end()) {
        auto num = it->second;
        it->second++;
        return num;
    } else {
        instantiatedOperators.emplace(to_find, 1);
        return 0;
    }
}

struct LegalizeHWModule : OpConversionPattern<hw::HWModuleOp> {
    using OpConversionPattern<hw::HWModuleOp>::OpConversionPattern;

    LegalizeHWModule(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<hw::HWModuleOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        hw::HWModuleOp op,
        hw::HWModuleOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto context = rewriter.getContext();
        // Get connection info
        op.walk([&](HWConnectOp connect) {
            newConnections.push_back(std::make_pair(
                connect.getPortArgument(),
                connect.getPortQueue()));
        });
        // Get output port index
        op.walk([&](hw::OutputOp outputs) {
            int i = 0;
            for (auto operand : outputs.getOperands())
                oldOutputs.push_back(std::make_pair(i, operand));
        });
        // unsigned numChannels = 0;
        // op.walk([&](ChannelOp channel) { numChannels++; });

        auto funcTy = op.getFunctionType();
        auto numInputs = funcTy.getNumInputs();
        auto numOutputs = funcTy.getNumResults();
        // assert(
        //     (numChannels >= numInputs + numOutputs)
        //     && "potentially not enough channels for input/output ports");

        SmallVector<hw::PortInfo> ports;
        int in_num = 1;
        int out_num = 1;

        // Add clock port.
        hw::PortInfo clock;
        clock.name = rewriter.getStringAttr("clock");
        clock.dir = hw::ModulePort::Direction::Input;
        clock.type = rewriter.getI1Type();
        clock.argNum = 0;
        ports.push_back(clock);

        // Add reset port.
        hw::PortInfo reset;
        reset.name = rewriter.getStringAttr("reset");
        reset.dir = hw::ModulePort::Direction::Input;
        reset.type = rewriter.getI1Type();
        reset.argNum = 1;
        ports.push_back(reset);

        for (size_t i = 0; i < numInputs; i++) {
            auto type = funcTy.getInput(i);
            std::string name;
            auto inTy = dyn_cast<InputType>(type);
            auto elemTy = dyn_cast<IntegerType>(inTy.getElementType());
            assert(elemTy && "only integers are supported on hardware");
            // Add ready for input port
            hw::PortInfo in_ready;
            name = ((numInputs == 1) ? "in" : "in" + std::to_string(in_num))
                   + "_ready";
            in_ready.name = rewriter.getStringAttr(name);
            in_ready.dir = hw::ModulePort::Direction::Output;
            in_ready.type = rewriter.getI1Type();
            in_ready.argNum = in_num + 1;
            ports.push_back(in_ready);
            // Add valid for input port
            hw::PortInfo in_valid;
            name = ((numInputs == 1) ? "in" : "in" + std::to_string(in_num))
                   + "_valid";
            in_valid.name = rewriter.getStringAttr(name);
            in_valid.dir = hw::ModulePort::Direction::Input;
            in_valid.type = rewriter.getI1Type();
            in_valid.argNum = in_num + 1;
            ports.push_back(in_valid);
            // Add bits for input port
            hw::PortInfo in_bits;
            name = ((numInputs == 1) ? "in" : "in" + std::to_string(in_num))
                   + "_bits";
            in_bits.name = rewriter.getStringAttr(name);
            in_bits.dir = hw::ModulePort::Direction::Input;
            in_bits.type = rewriter.getIntegerType(elemTy.getWidth());
            in_bits.argNum = in_num + 1;
            ports.push_back(in_bits);
            // Index increment
            in_num++;
        }
        for (size_t i = 0; i < numOutputs; i++) {
            auto type = funcTy.getResult(i);
            std::string name;
            auto outTy = dyn_cast<OutputType>(type);
            auto elemTy = dyn_cast<IntegerType>(outTy.getElementType());
            assert(elemTy && "only integers are supported on hardware");
            // Add ready for output port
            hw::PortInfo out_ready;
            name = ((numOutputs == 1) ? "out" : "out" + std::to_string(out_num))
                   + "_ready";
            out_ready.name = rewriter.getStringAttr(name);
            out_ready.dir = hw::ModulePort::Direction::Input;
            out_ready.type = rewriter.getI1Type();
            out_ready.argNum = out_num;
            ports.push_back(out_ready);
            // Add valid for output port
            hw::PortInfo out_valid;
            name = ((numOutputs == 1) ? "out" : "out" + std::to_string(out_num))
                   + "_valid";
            out_valid.name = rewriter.getStringAttr(name);
            out_valid.dir = hw::ModulePort::Direction::Output;
            out_valid.type = rewriter.getI1Type();
            out_valid.argNum = out_num;
            ports.push_back(out_valid);
            // Add bits for output port
            hw::PortInfo out_bits;
            name = ((numOutputs == 1) ? "out" : "out" + std::to_string(out_num))
                   + "_bits";
            out_bits.name = rewriter.getStringAttr(name);
            out_bits.dir = hw::ModulePort::Direction::Output;
            out_bits.type = rewriter.getIntegerType(elemTy.getWidth());
            out_bits.argNum = out_num;
            ports.push_back(out_bits);
            // Index increment
            out_num++;
        }

        // Create new HWModule
        auto hwModule = rewriter.create<hw::HWModuleOp>(
            loc,
            op.getNameAttr(),
            hw::ModulePortInfo(ports),
            ArrayAttr{},
            ArrayRef<NamedAttribute>{},
            StringAttr{},
            false);
        auto moduleOp = hwModule.getParentOp<ModuleOp>();
        rewriter.setInsertionPointToStart(&hwModule.getBody().front());
        // Create placeholders for the port that may connected later
        op.walk([&](InstantiateOp instances) {
            for (auto inport : instances.getInputs()) {
                auto placeholder = rewriter.create<hw::ConstantOp>(
                    loc,
                    rewriter.getI1Type(),
                    0);
                placeholderInputs.push_back(
                    std::make_pair(placeholder, inport));
            }
            for (auto outport : instances.getOutputs()) {
                if (find(willBeConnected, outport) == willBeConnected.end())
                    willBeConnected.push_back(outport);
                auto placeholderValid = rewriter.create<hw::ConstantOp>(
                    loc,
                    rewriter.getI1Type(),
                    0);
                auto placeholderBits = rewriter.create<hw::ConstantOp>(
                    loc,
                    rewriter.getIntegerType(
                        dyn_cast<InputType>(outport.getType())
                            .getElementType()
                            .getIntOrFloatBitWidth()),
                    0);
                SmallVector<Value> placeholders;
                placeholders.push_back(placeholderValid.getResult());
                placeholders.push_back(placeholderBits.getResult());
                placeholderOutputs.push_back(
                    std::make_pair(placeholders, outport));
            }
        });

        // Store the pair of old and new argument(s) in vector
        oldArgsIndex.clear();
        for (size_t i = 0; i < numInputs; i++)
            oldArgsIndex.push_back(std::make_pair(i, op.getArgument(i)));

        // Convert ops in the old HWModule into new one
        int queueSuffixNum = 0;
        for (auto &opInside :
             llvm::make_early_inc_range(op.getBody().getOps())) {
            if (auto channelOp = dyn_cast<ChannelOp>(opInside)) {
                auto portBit =
                    channelOp.getEncapsulatedType().getIntOrFloatBitWidth();
                auto size = channelOp.getBufferSize().value();

                OpBuilder builder(context);
                builder.setInsertionPointToStart(
                    &moduleOp.getBodyRegion().front());
                auto queueModule = insertQueue(
                    builder,
                    builder.getUnknownLoc(),
                    portBit,
                    size);
                auto newModuleArg = getNewIndexOrArg<Value, Value>(
                    channelOp.getInChan(),
                    newConnections);
                auto isConnectedLater =
                    find(willBeConnected, channelOp.getInChan())
                    != willBeConnected.end();
                assert(
                    (newModuleArg || isConnectedLater)
                    && "all ports must be connected");
                SmallVector<Value> inputs;
                inputs.push_back(hwModule.getArgument(0));
                inputs.push_back(hwModule.getArgument(1));
                if (!isConnectedLater) { // If connected to arguments
                    auto newArgIndex = getNewIndexOrArg<int, Value>(
                        newModuleArg.value(),
                        oldArgsIndex);
                    auto idx_input = newArgIndex.value();
                    inputs.push_back(hwModule.getArgument(2 * idx_input + 2));
                    inputs.push_back(hwModule.getArgument(2 * idx_input + 3));
                } else { // If connected later
                    auto placeholders =
                        getNewIndexOrArg<SmallVector<Value>, Value>(
                            channelOp.getInChan(),
                            placeholderOutputs);
                    inputs.push_back(placeholders.value()[0]);
                    inputs.push_back(placeholders.value()[1]);
                }
                auto oldOutputArg = getNewIndexOrArg<int, Value>(
                    channelOp.getOutChan(),
                    oldOutputs);
                if (oldOutputArg) // If connected to output
                    inputs.push_back(hwModule.getArgument(
                        2 * numInputs + 2 + oldOutputArg.value()));
                else { // If connected later
                    auto placeholder = getNewIndexOrArg<Value, Value>(
                        channelOp.getOutChan(),
                        placeholderInputs);
                    inputs.push_back(placeholder.value());
                }
                auto queueInstance = rewriter.create<hw::InstanceOp>(
                    loc,
                    queueModule,
                    "queue" + std::to_string(queueSuffixNum++),
                    inputs);
                auto in_ready = queueInstance.getResult(0);
                auto in_valid = queueInstance.getResult(1);
                auto in_bits = queueInstance.getResult(2);
                if (newModuleArg) newOutputs.push_back(in_ready);
                SmallVector<Value, 2> instancePorts;
                instancePorts.push_back(in_valid);
                instancePorts.push_back(in_bits);
                instanceInputs.push_back(
                    std::make_pair(instancePorts, channelOp.getOutChan()));
                // }
                if (oldOutputArg) {
                    SmallVector<Value, 2> newBundle;
                    newBundle.push_back(in_valid);
                    newBundle.push_back(in_bits);
                    newOutputBundles.push_back(
                        std::make_pair(newBundle, channelOp.getOutChan()));
                }
                instanceOutputs.push_back(
                    std::make_pair(in_ready, channelOp.getInChan()));
            } else if (auto instantiateOp = dyn_cast<InstantiateOp>(opInside)) {
                // TODO: lower here
                SmallVector<Value> inputs;
                inputs.push_back(hwModule.getArgument(0));
                inputs.push_back(hwModule.getArgument(1));
                for (auto input : instantiateOp.getInputs()) {
                    auto queuePorts = getNewIndexOrArg(input, instanceInputs);
                    inputs.append(queuePorts.value());
                }
                for (auto output : instantiateOp.getOutputs()) {
                    auto queuePort = getNewIndexOrArg(output, instanceOutputs);
                    inputs.push_back(queuePort.value());
                }
                auto calleeName =
                    instantiateOp.getCalleeAttr().getRootReference();
                auto operatorToCall = getNewIndexOrArg<Operation*, StringAttr>(
                    calleeName,
                    newOperators);
                auto calleeStr = calleeName.str();
                auto nameSuffix = getNameOfInstance(calleeStr);
                auto instanceName =
                    nameSuffix == 0 ? calleeStr
                                    : calleeStr + std::to_string(nameSuffix);
                auto instanceOp = rewriter.create<hw::InstanceOp>(
                    loc,
                    operatorToCall.value(),
                    instanceName,
                    inputs);
                int i = 0;
                for (auto input : instantiateOp.getInputs()) {
                    auto placeholder =
                        getNewIndexOrArg(input, placeholderInputs);
                    placeholder.value().replaceAllUsesWith(
                        instanceOp.getResult(i++));
                }
                int j = 0;
                for (auto output : instantiateOp.getOutputs()) {
                    auto placeholders =
                        getNewIndexOrArg(output, placeholderOutputs);
                    placeholders.value()[0].replaceAllUsesWith(
                        instanceOp.getResult(i + j++));
                    placeholders.value()[1].replaceAllUsesWith(
                        instanceOp.getResult(i + j++));
                }
            } else if (auto connectOp = dyn_cast<HWConnectOp>(opInside)) {
                continue;
            } else if (auto outputOp = dyn_cast<hw::OutputOp>(opInside)) {
                for (auto operand : outputOp.getOperands())
                    if (auto bundle =
                            getNewIndexOrArg(operand, newOutputBundles))
                        newOutputs.append(bundle.value());
                rewriter.create<hw::OutputOp>(loc, newOutputs);
            } else {
                op.emitError() << "unsupported ops in top module";
            }
        }

        // Clean up the placeholders
        hwModule.walk([&](hw::ConstantOp constOp) {
            if (constOp.getResult().use_empty()) rewriter.eraseOp(constOp);
        });

        rewriter.eraseOp(op);

        return success();
    }
};

} // namespace

void mlir::populateDfgToCirctConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<ConvertOperator>(typeConverter, patterns.getContext());
    patterns.add<LegalizeHWModule>(typeConverter, patterns.getContext());
}

namespace {
struct ConvertDfgToCirctPass
        : public ConvertDfgToCirctBase<ConvertDfgToCirctPass> {
    void runOnOperation() final;
};
} // namespace

void ConvertDfgToCirctPass::runOnOperation()
{
    TypeConverter converter;

    converter.addConversion([&](Type type) { return type; });
    // converter.addConversion([&](InputType type) -> Type {
    //     return converter.convertType(type.getElementType());
    // });
    // converter.addConversion([&](OutputType type) -> Type {
    //     return converter.convertType(type.getElementType());
    // });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateDfgToCirctConversionPatterns(converter, patterns);

    target.addLegalDialect<
        comb::CombDialect,
        fsm::FSMDialect,
        hw::HWDialect,
        sv::SVDialect>();
    target.addIllegalDialect<dfg::DfgDialect>();
    target.addDynamicallyLegalOp<hw::HWModuleOp>([&](hw::HWModuleOp op) {
        auto funcTy = op.getFunctionType();
        for (const auto inTy : funcTy.getInputs())
            if (dyn_cast<InputType>(inTy)) return false;
        for (const auto outTy : funcTy.getResults())
            if (dyn_cast<OutputType>(outTy)) return false;
        return true;
    });
    target.addDynamicallyLegalOp<hw::OutputOp>([&](hw::OutputOp op) {
        for (auto resultTy : op.getOperandTypes())
            if (dyn_cast<OutputType>(resultTy)) return false;
        return true;
    });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertDfgToCirctPass()
{
    return std::make_unique<ConvertDfgToCirctPass>();
}
