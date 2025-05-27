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
                                        "::llvm::cast<vitis::ArrayType>($_self).getElementType()">,
                         MemoryEffectsOpInterface]>
{
    let summary = "Read a value from an array";

    let arguments = (ins Arg<Vitis_ArrayType>:$array, Variadic<Index>:$indices);
    let results = (outs AnyType:$result);

    let assemblyFormat = "$array `[` $indices `]` attr-dict `:` qualified(type($array)) `->` type($result)";
    let hasVerifier = 1;
    let hasCanonicalizer = 1;
    
    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // MemoryEffectsOpInterface Methods
        //===------------------------------------------------------------------===//
        void getEffects(
                SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                &effects)
        {
            effects.emplace_back(MemoryEffects::Read::get(), &getArrayMutable(),
                                 SideEffects::DefaultResource::get());
        }

        ArrayType getArrayType() { return getArray().getType(); }
    }];
}

def Vitis_ArrayWriteOp : Vitis_ArrayOpBase<"write",
                         [TypesMatchWith<"result type must mathces array type",
                                         "array", "value",
                                         "::llvm::cast<vitis::ArrayType>($_self).getElementType()">,
                         MemoryEffectsOpInterface]>
{
    let summary = "Write a value into an array";

    let arguments = (ins AnyType:$value,
                         Arg<Vitis_ArrayType>:$array,
                         Variadic<Index>:$indices);

    let assemblyFormat = "$value `,` $array `[` $indices `]` attr-dict `:` type($value) `->` qualified(type($array))";
    let hasVerifier = 1;
    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // MemoryEffectsOpInterface Methods
        //===------------------------------------------------------------------===//
        void getEffects(
                SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                &effects)
        {
            effects.emplace_back(MemoryEffects::Write::get(), &getArrayMutable(),
                                 SideEffects::DefaultResource::get());
        }

        ArrayType getArrayType() { return getArray().getType(); }
    }];
}

def Vitis_ArrayPointerReadOp : Vitis_ArrayOpBase<"ptr_read",
                        [TypesMatchWith<"result type must mathces array type",
                                        "array", "result",
                                        "::llvm::cast<vitis::PointerType>($_self).getPointerType()">,
                         MemoryEffectsOpInterface]>
{
    let summary = "Read a value from an array pointer";

    let arguments = (ins Arg<Vitis_PointerType>:$array, Index:$index);
    let results = (outs AnyType:$result);

    let assemblyFormat = "$array `[` $index `]` attr-dict `:` qualified(type($array)) `->` type($result)";

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // MemoryEffectsOpInterface Methods
        //===------------------------------------------------------------------===//
        void getEffects(
                SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                &effects)
        {
            effects.emplace_back(MemoryEffects::Read::get(), &getArrayMutable(),
                                 SideEffects::DefaultResource::get());
        }
    }];
}

def Vitis_ArrayPointerWriteOp : Vitis_ArrayOpBase<"ptr_write",
                         [TypesMatchWith<"result type must mathces array type",
                                         "array", "value",
                                         "::llvm::cast<vitis::PointerType>($_self).getPointerType()">,
                         MemoryEffectsOpInterface]>
{
    let summary = "Write a value into an array pointer";

    let arguments = (ins AnyType:$value,
                         Arg<Vitis_PointerType>:$array,
                         Index:$index);

    let assemblyFormat = "$value `,` $array `[` $index `]` attr-dict `:` type($value) `->` qualified(type($array))";

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // MemoryEffectsOpInterface Methods
        //===------------------------------------------------------------------===//
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