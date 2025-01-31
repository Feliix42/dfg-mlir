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

def Vitis_StreamGetLastOp : Vitis_StreamOpBase<"get_last", []>
{
    let summary = "read last signal from data packet";
    let arguments = (ins AnyType:$data_pkt);
    let results = (outs I1:$last);
    let assemblyFormat = "$data_pkt attr-dict `:` type($data_pkt) `->` type($last)";
}

def Vitis_StreamSetLastOp : Vitis_StreamOpBase<"set_last",[]>
{
    let summary = "set last signal to data packet";
    let arguments = (ins I1:$last,
                         AnyType:$data_pkt);
    let assemblyFormat = "`(` $last `)` $data_pkt attr-dict `:` type($last) `->` type($data_pkt)";
}

def Vitis_StreamGetDataOp : Vitis_StreamOpBase<"get_data", []>
{
    let summary = "read data from data packet";
    let arguments = (ins AnyType:$data_pkt);
    let results = (outs AnyType:$data);
    let assemblyFormat = "$data_pkt attr-dict `:` type($data_pkt) `->` type($data)";
}

def Vitis_StreamSetDataOp : Vitis_StreamOpBase<"set_data", []>
{
    let summary = "set data to data packet";
    let arguments = (ins AnyType:$data,
                         AnyType:$data_pkt);
    let assemblyFormat = "`(` $data `)` $data_pkt attr-dict `:` type($data) `->` type($data_pkt)";
}

#endif // VITIS_STREAMOPS