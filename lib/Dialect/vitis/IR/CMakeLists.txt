add_mlir_dialect_library(VitisDialect
        Dialect.cpp
        Ops.cpp
        Types.cpp

    DEPENDS
        VitisIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        VitisEnums
)
