//===- Enums.td --------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Base2 enumerations.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_ENUMS
#define DFG_ENUMS

include "mlir/IR/EnumAttr.td"

def Dfg_ChannelStyleAttr : I64EnumAttr<
    "ChannelStyle",
    "Style of the channel to implement",
    [
        I64EnumAttrCase<"Dfg", 0b00, "dfg">,
        I64EnumAttrCase<"Xilinx", 0b01, "xilinx">
    ]>
{
    let cppNamespace = "::mlir::dfg";
}

def Dfg_ChannelStyleDefaultAttr
    : DefaultValuedAttr<Dfg_ChannelStyleAttr, "ChannelStyle::Dfg">;

#endif // DFG_ENUMS