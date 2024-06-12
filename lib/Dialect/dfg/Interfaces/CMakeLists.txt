add_mlir_library(DfgInterfaces
    DataflowGraphOp.cpp

    ENABLE_AGGREGATION
    DEPENDS
        DfgIncGen
)