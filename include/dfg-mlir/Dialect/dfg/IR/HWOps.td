//===- HWOps.td --------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the dfg intermediate hw ops.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_HWOPS
#define DFG_HWOPS

// Template for hw ops.
class Dfg_HWOpBase<string mnemonic, list<Trait> traits = []>
        : Dfg_Op< "hw." # mnemonic, traits> {}

def Dfg_HWConnectOp
        : Dfg_HWOpBase<"connect", []>
{
    let summary = "intermediate connect between ports";

    let description = [{
        This op will replace the dfg.push during conversion
        of StdToCirct. It represents the actual connection
        between top module ports and head/tail fifo buffer
        ports.
    }];

    let arguments = (ins
        Dfg_InputType: $portArgument,
        Dfg_InputType: $portQueue
    );

    let hasCustomAssemblyFormat = 1;
}

def Dfg_HWInstanceOp
        : Dfg_HWOpBase<"instance", []>
{
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SymbolRefAttr:$module,
        Variadic<AnyType>:$operands
    );
    let results = (outs Variadic<AnyType>:$results);

    let assemblyFormat = [{
        $module `(` $operands `)` attr-dict `:` functional-type($operands, $results)
    }];
}

#endif // DFG_HWOPS