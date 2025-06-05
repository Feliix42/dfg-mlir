add_mlir_dialect_library(DFGMLIREmitHLSDialect
        Dialect.cpp
        Ops.cpp
        Types.cpp

    DEPENDS
        DFGMLIREmitHLSIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        DFGMLIREmitHLSEnums
)
