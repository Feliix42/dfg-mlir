//===- PragmaOps.td -----------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the emitHLS ops related to arrays.
//
//===----------------------------------------------------------------------===//

#ifndef EMITHLS_PRAGMAOPS
#define EMITHLS_PRAGMAOPS

class emitHLS_PragmaOpBase<string mnemonic, list<Trait> traits = []>
        : emitHLS_Op< "pragma." # mnemonic, traits> {}

def emitHLS_PragmaBindStorageOp
                : emitHLS_PragmaOpBase<"bind_storage", []>
{
    let summary = "HLS pragma BIND_STORAGE";

    let arguments = (ins AnyTypeOf<[emitHLS_ArrayType, emitHLS_StreamType]>:$variable,
                         emitHLS_PragmaStorageTypeAttr:$type,
                         emitHLS_PragmaStorageImplAttr:$impl);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Value":$variable,
                       "emitHLS::PragmaStorageType":$type,
                       "emitHLS::PragmaStorageImpl":$impl)>
    ];
    
    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        Type getVariableType() {
            return getVariable().getType();
        }
    }];
}

def emitHLS_PragmaDataflowOp
                : emitHLS_PragmaOpBase<"dataflow", [NoTerminator]>
{
    let summary = "HLS pragma DATAFLOW";

    let regions = (region SizedRegion<1>:$dataflow_region);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins CArg<"llvm::function_ref<void()>", "{}">:$dataflowRegionCtor)>
    ];

    let assemblyFormat = "$dataflow_region attr-dict";
}

def emitHLS_PragmaInlineOp
                : emitHLS_PragmaOpBase<"inline", []>
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

def emitHLS_PragmaInterfaceOp
                : emitHLS_PragmaOpBase<"interface", []>
{
    let summary = "HLS pragma INTERFACE";

    let arguments = (ins emitHLS_PragmaInterfaceModeAttr:$mode,
                         emitHLS_PointerType:$port,
                         OptionalAttr<emitHLS_PragmaInterfaceMasterAxiOffsetAttr>:$offset,
                         StrAttr:$bundle);
    
    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "emitHLS::PragmaInterfaceMode":$mode,
                       "Value":$port,
                       "StringRef":$bundle,
                       CArg<"std::optional<emitHLS::PragmaInterfaceMasterAxiOffset>", "{}">:$offset)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        Type getPortType() {
            return getPort().getType();
        }
    }];
}

def emitHLS_PragmaReturnInterfaceOp
                : emitHLS_PragmaOpBase<"return_interface", []>
{
    let summary = "HLS pragma INTERFACE port=return";
    
    let assemblyFormat = "attr-dict";
}

def emitHLS_PragmaPipelineOp
                : emitHLS_PragmaOpBase<"pipeline", []>
{
    let summary = "HLS pragma PIPELINE";

    let arguments = (ins I64Attr:$interval,
                         emitHLS_PragmaPipelineStyleAttr:$style);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins CArg<"int64_t", "1">:$interval,
                       CArg<"emitHLS::PragmaPipelineStyle",
                            "emitHLS::PragmaPipelineStyle::flp">:$style)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def emitHLS_PragmaStreamOp
                : emitHLS_PragmaOpBase<"stream", []>
{
    let summary = "HLS pragma STREAM";

    let arguments = (ins emitHLS_StreamType:$variable,
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

#endif // EMITHLS_PRAGMAOPS