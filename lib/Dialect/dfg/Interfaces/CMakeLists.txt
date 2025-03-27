add_mlir_library(DFGMLIRDfgInterfaces
    DataflowGraphOp.cpp

    ENABLE_AGGREGATION
    DEPENDS
        DFGMLIRDfgIncGen
)