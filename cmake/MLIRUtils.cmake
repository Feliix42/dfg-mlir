#[========================================================================[.rst:
MLIRUtils
---------

Some utility functions for reliably declaring MLIR TableGen targets and
enforcing a common naming scheme.

#]========================================================================]

function(mlir_gen_enums prefix)
    set(full_prefix DFGMLIR${prefix})
    set(LLVM_TARGET_DEFINITIONS Enums.td)

    mlir_tablegen(Enums.h.inc -gen-enum-decls)
    mlir_tablegen(Enums.cpp.inc -gen-enum-defs)

    add_public_tablegen_target(${full_prefix}EnumsIncGen)
    add_dependencies(${full_prefix}IncGen ${full_prefix}EnumsIncGen)
endfunction()

function(mlir_gen_iface prefix iface kind)
    set(full_prefix DFGMLIR${prefix})
    set(LLVM_TARGET_DEFINITIONS ${iface}.td)

    mlir_tablegen(${iface}.h.inc -gen-${kind}-interface-decls)
    mlir_tablegen(${iface}.cpp.inc -gen-${kind}-interface-defs)

    add_public_tablegen_target(${full_prefix}${iface}InterfaceIncGen)
    add_dependencies(${full_prefix}IncGen ${full_prefix}${iface}InterfaceIncGen)
endfunction()

function(mlir_gen_ir prefix)
    set(full_prefix DFGMLIR${prefix})
    # string(TOLOWER ${prefix} filter)
    string(SUBSTRING ${prefix} 0 1 first_char)
    string(SUBSTRING ${prefix} 1 -1 rest_chars)
    string(TOLOWER ${first_char} first_char_lower)
    set(filter "${first_char_lower}${rest_chars}")

    set(LLVM_TARGET_DEFINITIONS Ops.td)

    mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=${filter})
    mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=${filter})
    mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=${filter})
    mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=${filter})
    # mlir_tablegen(Attributes.h.inc -gen-attrdef-decls -attrdefs-dialect=${filter})
    # mlir_tablegen(Attributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=${filter})
    mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=${filter})
    mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=${filter})

    add_public_tablegen_target(${full_prefix}IRIncGen)
    add_dependencies(${full_prefix}IncGen ${full_prefix}IRIncGen)

    add_mlir_doc(Ops ${full_prefix}Ops Dialects/ -gen-dialect-doc -dialect=${filter})
endfunction()

function(mlir_gen_passes prefix)
    set(full_prefix DFGMLIR${prefix})
    set(LLVM_TARGET_DEFINITIONS Passes.td)

    mlir_tablegen(Passes.h.inc -gen-pass-decls -name ${full_prefix})
    mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix ${full_prefix})
    mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix ${full_prefix})

    add_public_tablegen_target(${full_prefix}PassesIncGen)
    add_dependencies(${full_prefix}IncGen ${full_prefix}PassesIncGen)
endfunction()

function(mlir_gen_transforms prefix)
    set(full_prefix DFGMLIR${prefix})
    set(LLVM_TARGET_DEFINITIONS Passes.td)

    mlir_tablegen(Passes.h.inc -gen-pass-decls -name ${full_prefix})
    mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix ${full_prefix})
    mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix ${full_prefix})

    add_public_tablegen_target(${full_prefix}TransformsIncGen)
    add_dependencies(${full_prefix}IncGen ${full_prefix}TransformsIncGen)
endfunction()