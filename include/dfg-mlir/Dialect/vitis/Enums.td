//===- Enums.td --------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Vitis enumerations.
//
//===----------------------------------------------------------------------===//

#ifndef VITIS_ENUMS
#define VITIS_ENUMS

include "mlir/IR/EnumAttr.td"

def Vitis_CmpPredicateAttr : I64EnumAttr<
    "CmpPredicate", "",
    [
      I64EnumAttrCase<"eq", 0>,
      I64EnumAttrCase<"ne", 1>,
      I64EnumAttrCase<"lt", 2>,
      I64EnumAttrCase<"le", 3>,
      I64EnumAttrCase<"gt", 4>,
      I64EnumAttrCase<"ge", 5>,
      I64EnumAttrCase<"three_way", 6>,
    ]> {
  let cppNamespace = "::mlir::vitis";
}

def Vitis_PragmaStorageTypeAttr : I64EnumAttr<
    "PragmaStorageType", "",
    [
      I64EnumAttrCase<"fifo", 0>,
      I64EnumAttrCase<"ram_1p", 1>,
      I64EnumAttrCase<"ram_1wnr", 2>,
      I64EnumAttrCase<"ram_2p", 3>,
      I64EnumAttrCase<"ram_s2p", 4>,
      I64EnumAttrCase<"ram_t2p", 5>,
      I64EnumAttrCase<"rom_1p", 6>,
      I64EnumAttrCase<"rom_2p", 7>,
      I64EnumAttrCase<"rom_np", 8>,
    ]> {
  let cppNamespace = "::mlir::vitis";
}

def Vitis_PragmaStorageImplAttr : I64EnumAttr<
    "PragmaStorageImpl", "",
    [
      I64EnumAttrCase<"bram", 0>,
      I64EnumAttrCase<"bram_ecc", 1>,
      I64EnumAttrCase<"lutram", 2>,
      I64EnumAttrCase<"uram", 3>,
      I64EnumAttrCase<"uram_ecc", 4>,
      I64EnumAttrCase<"srl", 5>,
      I64EnumAttrCase<"memory", 6>,
      I64EnumAttrCase<"automatic", 7>,
    ]> {
  let cppNamespace = "::mlir::vitis";
}

def Vitis_PragmaInterfaceModeAttr : I64EnumAttr<
    "PragmaInterfaceMode", "",
    [
      // Extend possible
      I64EnumAttrCase<"m_axi", 0>,
      I64EnumAttrCase<"s_axilite", 1>
    ]> {
  let cppNamespace = "::mlir::vitis";
}

def Vitis_PragmaInterfaceMasterAxiOffsetAttr : I64EnumAttr<
    "PragmaInterfaceMasterAxiOffset", "",
    [
      // Extend possible
      I64EnumAttrCase<"off", 0>,
      I64EnumAttrCase<"direct", 1>,
      I64EnumAttrCase<"slave", 2>
    ]> {
  let cppNamespace = "::mlir::vitis";
}

def Vitis_PragmaPipelineStyleAttr : I64EnumAttr<
    "PragmaPipelineStyle", "",
    [
      I64EnumAttrCase<"stp", 0>,
      I64EnumAttrCase<"flp", 1>,
      I64EnumAttrCase<"frp", 2>,
    ]> {
  let cppNamespace = "::mlir::vitis";
}

#endif // VITIS_ENUMS