add_mlir_dialect_library(DfgDialect
        Base.cpp

    DEPENDS
        DfgIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
)
