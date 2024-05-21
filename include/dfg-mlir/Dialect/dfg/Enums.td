//===- Enums.td --------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Base2 enumerations.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_ENUMS
#define DFG_ENUMS

include "mlir/IR/EnumAttr.td"

def Dfg_ChannelTypeAttr : I64EnumAttr<
    "ChannelType",
    "Type of the channel to implement",
    [
        I64EnumAttrCase<"Dfg", 0b00, "dfg">,
        I64EnumAttrCase<"Xilinx", 0b01, "xilinx">
    ]>
{
    let cppNamespace = "::mlir::dfg";
}

def Dfg_ChannelTypeDefaultAttr
    : DefaultValuedAttr<Dfg_ChannelTypeAttr, "ChannelType::dfg">;

#endif // DFG_ENUMS
