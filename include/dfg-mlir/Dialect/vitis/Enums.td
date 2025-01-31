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

#endif // VITIS_ENUMS