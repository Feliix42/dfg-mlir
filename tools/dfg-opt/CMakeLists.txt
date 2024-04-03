# ###############################################################################
# dfg-opt
#
# The dfg-mlir optimizer driver.
# ###############################################################################

project(dfg-opt)

add_executable(${PROJECT_NAME}
    dfg-opt.cpp
)

# Link all standard MLIR dialect and conversion libs.
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${dialect_libs}
    ${conversion_libs}

    MLIROptLib
    MLIRTransforms

    CIRCTAffineToLoopSchedule
    CIRCTAnalysisTestPasses
    CIRCTArc
    CIRCTArcToLLVM
    CIRCTArcTransforms
    CIRCTCalyx
    CIRCTCalyxToHW
    CIRCTCalyxNative
    CIRCTCalyxToFSM
    CIRCTCalyxTransforms
    CIRCTComb
    CIRCTCombToSMT
    CIRCTCombTransforms
    CIRCTConvertToArcs
    CIRCTDC
    CIRCTDCToHW
    CIRCTDCTransforms
    CIRCTDebug
    CIRCTEmit
    CIRCTESI
    CIRCTExportChiselInterface
    CIRCTExportVerilog
    CIRCTFIRRTL
    CIRCTFIRRTLToHW
    CIRCTFIRRTLTransforms
    CIRCTFSM
    CIRCTFSMTransforms
    CIRCTFSMToSV
    CIRCTHandshake
    CIRCTHandshakeToDC
    CIRCTHandshakeToHW
    CIRCTHandshakeTransforms
    CIRCTLECTransforms
    CIRCTLLHD
    CIRCTLLHDToLLVM
    CIRCTHWToLLVM
    CIRCTCombToArith
    CIRCTCombToLLVM
    CIRCTLLHDTransforms
    CIRCTMoore
    CIRCTMooreToCore
    CIRCTMSFT
    CIRCTMSFTTransforms
    CIRCTHW
    CIRCTHWArith
    CIRCTHWArithToHW
    CIRCTHWToBTOR2
    CIRCTIbis
    CIRCTIbisTransforms
    CIRCTInteropDialect
    CIRCTHWToLLHD
    CIRCTHWToSMT
    CIRCTHWToSystemC
    CIRCTHWToSV
    CIRCTHWTransforms
    CIRCTLoopSchedule
    CIRCTLoopScheduleToCalyx
    CIRCTLTL
    CIRCTSCFToCalyx
    CIRCTScheduling
    CIRCTSeq
    CIRCTSeqToSV
    CIRCTSeqTransforms
    CIRCTSimToSV
    CIRCTSSP
    CIRCTSSPTransforms
    CIRCTCFToHandshake
    CIRCTOM
    CIRCTOMTransforms
    CIRCTPipelineOps
    CIRCTPipelineToHW
    CIRCTPipelineTransforms
    CIRCTSMT
    CIRCTSV
    CIRCTSVTransforms
    CIRCTHWArith
    CIRCTSystemC
    CIRCTSystemCTransforms
    CIRCTTransforms
    CIRCTVerif
    CIRCTVerifToSMT
    CIRCTVerifToSV
)
