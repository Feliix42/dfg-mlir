//===- StreamOps.td -----------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the emitHLS ops related to hls::stream.
//
//===----------------------------------------------------------------------===//

#ifndef EMITHLS_STREAMOPS
#define EMITHLS_STREAMOPS

class emitHLS_StreamOpBase<string mnemonic, list<Trait> traits = []>
        : emitHLS_Op< "stream." # mnemonic, traits> {}

def emitHLS_StreamReadOp : emitHLS_StreamOpBase<"read", [MemoryEffectsOpInterface]>
{
    let summary = "read datas from hls::stream";
    let arguments = (ins AnyType:$stream);
    let results = (outs AnyType:$data_pkt);
    let assemblyFormat = "$stream attr-dict `:` type($stream) `->` type($data_pkt)";
    let extraClassDeclaration = [{
        void getEffects(
                SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                &effects)
        {
            effects.emplace_back(MemoryEffects::Read::get(), &getStreamMutable(),
                                 SideEffects::DefaultResource::get());
        }
    }];
}

def emitHLS_StreamWriteOp : emitHLS_StreamOpBase<"write", [MemoryEffectsOpInterface]>
{
    let summary = "read datas from hls::stream";
    let arguments = (ins AnyType:$data_pkt,
                         AnyType:$stream);
    let assemblyFormat = "`(` $data_pkt `)` $stream attr-dict `:` type($data_pkt) `->` type($stream)";
    let extraClassDeclaration = [{
        void getEffects(
                SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                &effects)
        {
            effects.emplace_back(MemoryEffects::Write::get(), &getStreamMutable(),
                                 SideEffects::DefaultResource::get());
        }
    }];
}

#endif // EMITHLS_STREAMOPS