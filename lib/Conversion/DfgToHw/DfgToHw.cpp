/// Implementation of DfgToHw pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Conversion/DfgToHw/DfgToHw.h"

#include "circt/Conversion/HandshakeToHW.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/APInt.h"

#include <iostream>

namespace mlir {
#define GEN_PASS_DEF_CONVERTDFGTOHW
#include "dfg-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;
using namespace circt;

namespace {} // namespace

void mlir::populateDfgToHWConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{}

namespace {
struct ConvertDfgToHWPass
        : public impl::ConvertDfgToHWBase<ConvertDfgToHWPass> {
    void insertPullPushChannelModule(OpBuilder builder, bool pullOrPush);
    void insertQueueModule(OpBuilder builder);
    void runOnOperation() final;
};
} // namespace

void ConvertDfgToHWPass::insertPullPushChannelModule(
    OpBuilder builder,
    bool pullOrPush)
{
    auto name = pullOrPush ? "push_channel" : "pull_channel";
    auto ctx = builder.getContext();
    auto loc = builder.getUnknownLoc();
    auto i1Ty = builder.getI1Type();
    auto i32Ty = builder.getI32Type();
    auto inDir = hw::ModulePort::Direction::Input;
    auto outDir = hw::ModulePort::Direction::Output;
    SmallVector<Attribute> params;
    params.push_back(hw::ParamDeclAttr::get("bitwidth", i32Ty));
    auto dataParamAttr =
        hw::ParamDeclRefAttr::get(builder.getStringAttr("bitwidth"), i32Ty);
    auto dataParamTy = hw::IntType::get(dataParamAttr);

    SmallVector<hw::PortInfo> ports;
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("clock"), i1Ty, inDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("reset"), i1Ty, inDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("valid"), i1Ty, inDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("in"), dataParamTy, inDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("finish"), i1Ty, outDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("ready"), i1Ty, outDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("out"), dataParamTy, outDir}
    });
    auto hwModule = builder.create<hw::HWModuleOp>(
        loc,
        builder.getStringAttr(name),
        hw::ModulePortInfo(ports),
        ArrayAttr::get(builder.getContext(), params),
        ArrayRef<NamedAttribute>{},
        StringAttr{},
        false);
    auto clk = hwModule.getBody().getArgument(0);
    auto rst = hwModule.getBody().getArgument(1);
    auto inValid = hwModule.getBody().getArgument(2);
    auto inData = hwModule.getBody().getArgument(3);
    builder.setInsertionPointToStart(&hwModule.getBodyRegion().front());

    auto cstFalse = builder.create<hw::ConstantOp>(loc, i1Ty, 0);
    auto cstTrue = builder.create<hw::ConstantOp>(loc, i1Ty, 1);
    auto validReg =
        builder.create<sv::RegOp>(loc, i1Ty, StringAttr::get(ctx, "valid_reg"));
    auto validRegValue =
        builder.create<sv::ReadInOutOp>(loc, validReg.getResult());
    auto dataReg = builder.create<sv::RegOp>(
        loc,
        dataParamTy,
        StringAttr::get(ctx, "data_reg"));
    auto dataRegValue =
        builder.create<sv::ReadInOutOp>(loc, dataReg.getResult());
    auto validRegComp = builder.create<comb::XorOp>(
        loc,
        validRegValue.getResult(),
        cstTrue.getResult());
    auto readyOutput = builder.create<comb::MuxOp>(
        loc,
        validRegValue.getResult(),
        cstFalse.getResult(),
        inValid);
    Value dataOutput = dataRegValue.getResult();
    if (pullOrPush)
        dataOutput = builder
                         .create<comb::MuxOp>(
                             loc,
                             validRegValue.getResult(),
                             dataRegValue.getResult(),
                             inData)
                         .getResult();

    builder.create<sv::AlwaysOp>(loc, sv::EventControl::AtPosEdge, clk, [&] {
        builder.create<sv::IfOp>(
            loc,
            rst,
            [&] {
                builder.create<sv::PAssignOp>(
                    loc,
                    validReg.getResult(),
                    cstFalse.getResult());
            },
            [&] {
                builder.create<sv::IfOp>(loc, inValid, [&] {
                    builder.create<sv::PAssignOp>(
                        loc,
                        validReg.getResult(),
                        cstTrue.getResult());
                    builder.create<sv::IfOp>(
                        loc,
                        validRegComp.getResult(),
                        [&] {
                            builder.create<sv::PAssignOp>(
                                loc,
                                dataReg.getResult(),
                                inData);
                        });
                });
            });
    });

    builder.create<hw::OutputOp>(
        loc,
        ValueRange{
            validRegValue.getResult(),
            readyOutput.getResult(),
            dataOutput});
}

void ConvertDfgToHWPass::insertQueueModule(OpBuilder builder)
{
    auto loc = builder.getUnknownLoc();
    auto ctx = builder.getContext();
    auto i1Ty = builder.getI1Type();
    auto i32Ty = builder.getI32Type();
    auto inDir = hw::ModulePort::Direction::Input;
    auto outDir = hw::ModulePort::Direction::Output;
    SmallVector<Attribute> params;
    params.push_back(hw::ParamDeclAttr::get("bitwidth", builder.getI32Type()));
    params.push_back(hw::ParamDeclAttr::get("size", builder.getI32Type()));
    auto dataParamAttr = hw::ParamDeclRefAttr::get(
        builder.getStringAttr("bitwidth"),
        builder.getI32Type());
    auto dataParamTy = hw::IntType::get(dataParamAttr);
    auto sizeParamAttr = hw::ParamDeclRefAttr::get(
        builder.getStringAttr("size"),
        builder.getI32Type());

    SmallVector<hw::PortInfo> ports;
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("clock"), i1Ty, inDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("reset"), i1Ty, inDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("close"), i1Ty, inDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("in_valid"), i1Ty, inDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("in"), dataParamTy, inDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("out_ready"), i1Ty, inDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("in_ready"), i1Ty, outDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("out_valid"), i1Ty, outDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("out"), dataParamTy, outDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("done"), i1Ty, outDir}
    });

    auto hwModule = builder.create<hw::HWModuleOp>(
        loc,
        builder.getStringAttr("queue"),
        hw::ModulePortInfo(ports),
        ArrayAttr::get(builder.getContext(), params),
        ArrayRef<NamedAttribute>{},
        StringAttr{},
        false);
    auto clk = hwModule.getBody().getArgument(0);
    auto rst = hwModule.getBody().getArgument(1);
    auto close = hwModule.getBody().getArgument(2);
    auto inValid = hwModule.getBody().getArgument(3);
    auto inData = hwModule.getBody().getArgument(4);
    auto outReady = hwModule.getBody().getArgument(5);
    builder.setInsertionPointToStart(&hwModule.getBodyRegion().front());

    auto cstTrue = builder.create<hw::ConstantOp>(loc, i1Ty, 1);
    auto cstFalse = builder.create<hw::ConstantOp>(loc, i1Ty, 0);
    auto cstZero = builder.create<hw::ConstantOp>(loc, i32Ty, 0);
    auto cstOne = builder.create<hw::ConstantOp>(loc, i32Ty, 1);
    auto sizeValue =
        builder.create<hw::ParamValueOp>(loc, i32Ty, sizeParamAttr);
    auto sizeMaxIdx = builder.create<comb::SubOp>(
        loc,
        sizeValue.getResult(),
        cstOne.getResult());

    // Close behavior
    auto wantClose = builder.create<sv::RegOp>(
        loc,
        i1Ty,
        StringAttr::get(ctx, "want_close"));
    auto wantCloseValue =
        builder.create<sv::ReadInOutOp>(loc, wantClose.getResult());
    // Memory
    auto ram = builder.create<sv::RegOp>(
        loc,
        hw::UnpackedArrayType::get(
            builder.getContext(),
            dataParamTy,
            sizeParamAttr),
        StringAttr::get(ctx, "ram"));
    auto placeholderReadIndex = builder.create<hw::ConstantOp>(loc, i32Ty, 0);
    auto ramRead = builder.create<sv::ArrayIndexInOutOp>(
        loc,
        ram.getResult(),
        placeholderReadIndex.getResult());
    auto ramReadValue =
        builder.create<sv::ReadInOutOp>(loc, ramRead.getResult());

    // Pointers
    auto ptrWrite = builder.create<sv::RegOp>(
        loc,
        i32Ty,
        StringAttr::get(ctx, "ptr_write"));
    auto ptrWriteValue =
        builder.create<sv::ReadInOutOp>(loc, ptrWrite.getResult());
    auto ptrRead =
        builder.create<sv::RegOp>(loc, i32Ty, StringAttr::get(ctx, "ptr_read"));
    auto ptrReadValue =
        builder.create<sv::ReadInOutOp>(loc, ptrRead.getResult());
    placeholderReadIndex.replaceAllUsesWith(ptrReadValue.getResult());
    auto maybeFull = builder.create<sv::RegOp>(
        loc,
        i1Ty,
        StringAttr::get(ctx, "maybe_full"));
    auto maybeFullValue =
        builder.create<sv::ReadInOutOp>(loc, maybeFull.getResult());
    auto ptrLast =
        builder.create<sv::RegOp>(loc, i32Ty, StringAttr::get(ctx, "ptr_last"));
    auto ptrLastValue =
        builder.create<sv::ReadInOutOp>(loc, ptrLast.getResult());
    auto isDone =
        builder.create<sv::RegOp>(loc, i1Ty, StringAttr::get(ctx, "is_done"));
    auto isDoneValue = builder.create<sv::ReadInOutOp>(loc, isDone.getResult());

    // Signals
    auto ptrMatch = builder.create<comb::ICmpOp>(
        loc,
        comb::ICmpPredicate::eq,
        ptrWriteValue.getResult(),
        ptrReadValue.getResult());
    auto emptyT = builder.create<comb::XorOp>(
        loc,
        maybeFullValue.getResult(),
        cstTrue.getResult());
    auto empty = builder.create<comb::AndOp>(
        loc,
        ptrMatch.getResult(),
        emptyT.getResult());
    auto full = builder.create<comb::AndOp>(
        loc,
        ptrMatch.getResult(),
        maybeFullValue.getResult());
    auto placeholderNotFull = builder.create<hw::ConstantOp>(loc, i1Ty, 0);
    auto doEnq = builder.create<comb::AndOp>(
        loc,
        placeholderNotFull.getResult(),
        inValid);
    auto placeholderNotEmpty = builder.create<hw::ConstantOp>(loc, i1Ty, 0);
    auto doDeq = builder.create<comb::AndOp>(
        loc,
        outReady,
        placeholderNotEmpty.getResult());
    auto nextWrite = builder.create<comb::AddOp>(
        loc,
        ptrWriteValue.getResult(),
        cstOne.getResult());
    auto nextRead = builder.create<comb::AddOp>(
        loc,
        ptrReadValue.getResult(),
        cstOne.getResult());
    auto notSameEnqDeq = builder.create<comb::ICmpOp>(
        loc,
        comb::ICmpPredicate::ne,
        doEnq.getResult(),
        doDeq.getResult());
    auto notEmpty = builder.create<comb::XorOp>(
        loc,
        empty.getResult(),
        cstTrue.getResult());
    placeholderNotEmpty.replaceAllUsesWith(notEmpty.getResult());
    auto not_full =
        builder.create<comb::XorOp>(loc, full.getResult(), cstTrue.getResult());
    auto notWantClose = builder.create<comb::XorOp>(
        loc,
        wantCloseValue.getResult(),
        cstTrue.getResult());
    auto inReadyOutput = builder.create<comb::AndOp>(
        loc,
        not_full.getResult(),
        notWantClose.getResult());
    placeholderNotFull.replaceAllUsesWith(inReadyOutput.getResult());
    auto last_read = builder.create<comb::ICmpOp>(
        loc,
        comb::ICmpPredicate::eq,
        ptrLastValue.getResult(),
        ptrReadValue.getResult());
    auto lastReadWantClose = builder.create<comb::AndOp>(
        loc,
        wantCloseValue.getResult(),
        last_read.getResult());
    auto lastReadWantCloseDeq = builder.create<comb::AndOp>(
        loc,
        lastReadWantClose.getResult(),
        doDeq.getResult());
    auto updateClose =
        builder.create<comb::AndOp>(loc, close, notWantClose.getResult());
    auto doneOutput = builder.create<comb::OrOp>(
        loc,
        lastReadWantCloseDeq.getResult(),
        isDoneValue.getResult());

    // Clocked logic
    builder.create<sv::AlwaysOp>(loc, sv::EventControl::AtPosEdge, clk, [&] {
        builder.create<sv::IfOp>(
            loc,
            rst,
            [&] {
                builder.create<sv::PAssignOp>(
                    loc,
                    ptrWrite.getResult(),
                    cstZero.getResult());
                builder.create<sv::PAssignOp>(
                    loc,
                    ptrRead.getResult(),
                    cstZero.getResult());
                builder.create<sv::PAssignOp>(
                    loc,
                    maybeFull.getResult(),
                    cstFalse.getResult());
                builder.create<sv::PAssignOp>(
                    loc,
                    wantClose.getResult(),
                    cstFalse.getResult());
                builder.create<sv::PAssignOp>(
                    loc,
                    ptrLast.getResult(),
                    cstZero.getResult());
                builder.create<sv::PAssignOp>(
                    loc,
                    isDone.getResult(),
                    cstFalse.getResult());
            },
            [&] {
                builder.create<sv::IfOp>(loc, doEnq.getResult(), [&] {
                    auto ramWrite = builder.create<sv::ArrayIndexInOutOp>(
                        loc,
                        ram.getResult(),
                        ptrWriteValue.getResult());
                    builder.create<sv::PAssignOp>(
                        loc,
                        ramWrite.getResult(),
                        inData);
                    auto isSizeMax = builder.create<comb::ICmpOp>(
                        loc,
                        comb::ICmpPredicate::eq,
                        ptrWriteValue.getResult(),
                        sizeMaxIdx.getResult());
                    builder.create<sv::IfOp>(
                        loc,
                        isSizeMax.getResult(),
                        [&] {
                            builder.create<sv::PAssignOp>(
                                loc,
                                ptrWrite.getResult(),
                                cstZero.getResult());
                        },
                        [&] {
                            builder.create<sv::PAssignOp>(
                                loc,
                                ptrWrite.getResult(),
                                nextWrite.getResult());
                        });
                });
                builder.create<sv::IfOp>(loc, doDeq.getResult(), [&] {
                    auto isSizeMax = builder.create<comb::ICmpOp>(
                        loc,
                        comb::ICmpPredicate::eq,
                        ptrReadValue.getResult(),
                        sizeMaxIdx.getResult());
                    builder.create<sv::IfOp>(
                        loc,
                        isSizeMax.getResult(),
                        [&] {
                            builder.create<sv::PAssignOp>(
                                loc,
                                ptrRead.getResult(),
                                cstZero.getResult());
                        },
                        [&] {
                            builder.create<sv::PAssignOp>(
                                loc,
                                ptrRead.getResult(),
                                nextRead.getResult());
                        });
                });
                builder.create<sv::IfOp>(loc, notSameEnqDeq.getResult(), [&] {
                    builder.create<sv::PAssignOp>(
                        loc,
                        maybeFull.getResult(),
                        doEnq.getResult());
                });
                builder.create<sv::IfOp>(loc, updateClose.getResult(), [&] {
                    builder.create<sv::PAssignOp>(
                        loc,
                        wantClose.getResult(),
                        cstTrue.getResult());
                    builder.create<sv::PAssignOp>(
                        loc,
                        ptrLast.getResult(),
                        ptrWriteValue.getResult());
                });
                builder.create<sv::IfOp>(
                    loc,
                    lastReadWantCloseDeq.getResult(),
                    [&] {
                        builder.create<sv::PAssignOp>(
                            loc,
                            isDone.getResult(),
                            cstTrue.getResult());
                    });
            });
    });

    // Clean up placeholders
    placeholderReadIndex.erase();
    placeholderNotEmpty.erase();
    placeholderNotFull.erase();

    // Outputs
    builder.create<hw::OutputOp>(
        loc,
        ValueRange{
            inReadyOutput.getResult(),
            notEmpty.getResult(),
            ramReadValue.getResult(),
            doneOutput.getResult()});
}

void ConvertDfgToHWPass::runOnOperation()
{
    TypeConverter converter;

    converter.addConversion([&](Type type) { return type; });

    auto module = dyn_cast<ModuleOp>(getOperation());
    OpBuilder builder(&getContext());
    auto insertionPoint = &module.getBodyRegion().front();
    module.walk([&](ProcessOp processOp) -> WalkResult {
        builder.setInsertionPointToStart(insertionPoint);
        insertPullPushChannelModule(builder, 1);
        builder.setInsertionPointToStart(insertionPoint);
        insertPullPushChannelModule(builder, 0);
        return WalkResult::interrupt();
    });
    module.walk([&](RegionOp regionOp) {
        regionOp.walk([&](ChannelOp channelOp) -> WalkResult {
            builder.setInsertionPointToStart(insertionPoint);
            insertQueueModule(builder);
            return WalkResult::interrupt();
        });
    });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateDfgToHWConversionPatterns(converter, patterns);

    target.addLegalDialect<
        comb::CombDialect,
        hw::HWDialect,
        seq::SeqDialect,
        sv::SVDialect>();
    target.addIllegalDialect<dfg::DfgDialect, arith::ArithDialect>();

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertDfgToHWPass()
{
    return std::make_unique<ConvertDfgToHWPass>();
}
