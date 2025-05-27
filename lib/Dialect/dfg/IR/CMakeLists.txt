add_mlir_dialect_library(DFGMLIRDfgDialect
        Dialect.cpp
        Ops.cpp
        Types.cpp

    DEPENDS
        DFGMLIRDfgIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        DFGMLIRDfgEnums
        DFGMLIRDfgInterfaces
)
