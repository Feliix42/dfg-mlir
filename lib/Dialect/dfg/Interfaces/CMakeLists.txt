add_mlir_library(DFGMLIRDfgInterfaces
    EdgeOp.cpp
    GraphOp.cpp
    NodeOp.cpp

    ENABLE_AGGREGATION
    DEPENDS
        DFGMLIRDfgIncGen
)