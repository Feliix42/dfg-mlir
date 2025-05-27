//===- PragmaOps.td -----------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the vitis ops related to arrays.
//
//===----------------------------------------------------------------------===//

#ifndef VITIS_PRAGMAOPS
#define VITIS_PRAGMAOPS

class Vitis_PragmaOpBase<string mnemonic, list<Trait> traits = []>
        : Vitis_Op< "pragma." # mnemonic, traits> {}

def Vitis_PragmaBindStorageOp
                : Vitis_PragmaOpBase<"bind_storage", []>
{
    let summary = "HLS pragma BIND_STORAGE";

    let arguments = (ins AnyTypeOf<[Vitis_ArrayType, Vitis_StreamType]>:$variable,
                         Vitis_PragmaStorageTypeAttr:$type,
                         Vitis_PragmaStorageImplAttr:$impl);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Value":$variable,
                       "vitis::PragmaStorageType":$type,
                       "vitis::PragmaStorageImpl":$impl)>
    ];
    
    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        Type getVariableType() {
            return getVariable().getType();
        }
    }];
}

def Vitis_PragmaDataflowOp
                : Vitis_PragmaOpBase<"dataflow", [NoTerminator]>
{
    let summary = "HLS pragma DATAFLOW";

    let regions = (region SizedRegion<1>:$dataflow_region);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins CArg<"llvm::function_ref<void()>", "{}">:$dataflowRegionCtor)>
    ];

    let assemblyFormat = "$dataflow_region attr-dict";
}

def Vitis_PragmaInlineOp
                : Vitis_PragmaOpBase<"inline", []>
{
    let summary = "HLS pragma INLINE";

    let arguments = (ins OptionalAttr<UnitAttr>:$off);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins CArg<"bool", "false">:$off)>
    ];
    
    let hasCustomAssemblyFormat = 1;
    let hasCanonicalizer = 1;
}

def Vitis_PragmaInterfaceOp
                : Vitis_PragmaOpBase<"interface", []>
{
    let summary = "HLS pragma INTERFACE";

    let arguments = (ins Vitis_PragmaInterfaceModeAttr:$mode,
                         Vitis_PointerType:$port,
                         OptionalAttr<Vitis_PragmaInterfaceMasterAxiOffsetAttr>:$offset,
                         StrAttr:$bundle);
    
    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "vitis::PragmaInterfaceMode":$mode,
                       "Value":$port,
                       "StringRef":$bundle,
                       CArg<"std::optional<vitis::PragmaInterfaceMasterAxiOffset>", "{}">:$offset)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        Type getPortType() {
            return getPort().getType();
        }
    }];
}

def Vitis_PragmaReturnInterfaceOp
                : Vitis_PragmaOpBase<"return_interface", []>
{
    let summary = "HLS pragma INTERFACE port=return";
    
    let assemblyFormat = "attr-dict";
}

def Vitis_PragmaPipelineOp
                : Vitis_PragmaOpBase<"pipeline", []>
{
    let summary = "HLS pragma PIPELINE";

    let arguments = (ins I64Attr:$interval,
                         Vitis_PragmaPipelineStyleAttr:$style);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins CArg<"int64_t", "1">:$interval,
                       CArg<"vitis::PragmaPipelineStyle",
                            "vitis::PragmaPipelineStyle::flp">:$style)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def Vitis_PragmaStreamOp
                : Vitis_PragmaOpBase<"stream", []>
{
    let summary = "HLS pragma STREAM";

    let arguments = (ins Vitis_StreamType:$variable,
                         I64Attr:$depth);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Value":$variable,
                       CArg<"int64_t", "1">:$interval)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        Type getVariableType() {
            return getVariable().getType();
        }
    }];
}

#endif // VITIS_PRAGMAOPS