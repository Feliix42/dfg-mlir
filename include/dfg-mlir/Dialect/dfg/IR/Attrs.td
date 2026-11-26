//===- Attrs.td ---------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the dfg dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_ATTRS
#define DFG_ATTRS

include "dfg-mlir/Dialect/dfg/IR/Dialect.td"
include "mlir/IR/BuiltinAttributes.td"

def FusionAttr : Dfg_Attr<"Fusion"> {
    let mnemonic = "fusion";
    let summary = "Fusion group membership marker";
    let description = [{
        Marks an operation as a member of a named fusion group. Ops carrying
        the same group name (attached as the discardable attribute `fusion`)
        form one fusion candidate; a fusion pass (e.g. `dfg-fuse-groups`)
        collects them and fuses them into a single node.

        Example: `!dfg.fusion<"k1">`
    }];
    let parameters = (ins "StringAttr":$group);
    let assemblyFormat = "`<` $group `>`";
}

#endif
