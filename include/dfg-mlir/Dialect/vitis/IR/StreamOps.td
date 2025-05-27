//===- StreamOps.td -----------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the vitis ops related to hls::stream.
//
//===----------------------------------------------------------------------===//

#ifndef VITIS_STREAMOPS
#define VITIS_STREAMOPS

class Vitis_StreamOpBase<string mnemonic, list<Trait> traits = []>
        : Vitis_Op< "stream." # mnemonic, traits> {}

def Vitis_StreamReadOp : Vitis_StreamOpBase<"read", [MemoryEffectsOpInterface]>
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

def Vitis_StreamWriteOp : Vitis_StreamOpBase<"write", [MemoryEffectsOpInterface]>
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

#endif // VITIS_STREAMOPS