//===- ArrayOps.td -----------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the vitis ops related to arrays.
//
//===----------------------------------------------------------------------===//

#ifndef VITIS_ARRAYOPS
#define VITIS_ARRAYOPS

class Vitis_ArrayOpBase<string mnemonic, list<Trait> traits = []>
        : Vitis_Op< "array." # mnemonic, traits> {}

def Vitis_ArrayReadOp : Vitis_ArrayOpBase<"read",
                        [TypesMatchWith<"result type must mathces array type",
                                        "array", "result",
                                        "::llvm::cast<vitis::ArrayType>($_self).getElemType()">,
                         MemoryEffectsOpInterface]>
{
    let arguments = (ins Arg<Vitis_ArrayType>:$array, Index:$index);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$array `[` $index `]` attr-dict `:` type($array)";
    let extraClassDeclaration = [{
        void getEffects(
                SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                &effects)
        {
            effects.emplace_back(MemoryEffects::Read::get(), &getArrayMutable(),
                                 SideEffects::DefaultResource::get());
        }
    }];
}

def Vitis_ArrayWriteOp : Vitis_ArrayOpBase<"write",
                         [TypesMatchWith<"result type must mathces array type",
                                         "array", "value",
                                         "::llvm::cast<vitis::ArrayType>($_self).getElemType()">,
                         MemoryEffectsOpInterface]>
{
    let arguments = (ins AnyType:$value,
                         Arg<Vitis_ArrayType>:$array,
                         Index:$index);
    let assemblyFormat = "$value `,` $array `[` $index `]` attr-dict `:` type($array)";
    let extraClassDeclaration = [{
        void getEffects(
                SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                &effects)
        {
            effects.emplace_back(MemoryEffects::Write::get(), &getArrayMutable(),
                                 SideEffects::DefaultResource::get());
        }
    }];
}

#endif // VITIS_ARRAYOPS