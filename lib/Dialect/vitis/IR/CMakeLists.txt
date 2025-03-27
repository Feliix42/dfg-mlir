add_mlir_dialect_library(VitisDialect
        Dialect.cpp
        Ops.cpp
        Types.cpp

    DEPENDS
        DFGMLIRVitisIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        DFGMLIRVitisEnums
)
