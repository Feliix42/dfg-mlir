//===- ArrayOps.td -----------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the emitHLS ops related to arrays.
//
//===----------------------------------------------------------------------===//

#ifndef EMITHLS_ARRAYOPS
#define EMITHLS_ARRAYOPS

class emitHLS_ArrayOpBase<string mnemonic, list<Trait> traits = []>
        : emitHLS_Op< "array." # mnemonic, traits> {}

def emitHLS_ArrayReadOp : emitHLS_ArrayOpBase<"read",
                        [TypesMatchWith<"result type must mathces array type",
                                        "array", "result",
                                        "::llvm::cast<emitHLS::ArrayType>($_self).getElementType()">,
                         MemoryEffectsOpInterface]>
{
    let summary = "Read a value from an array";

    let arguments = (ins Arg<emitHLS_ArrayType>:$array, Variadic<Index>:$indices);
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

def emitHLS_ArrayWriteOp : emitHLS_ArrayOpBase<"write",
                         [TypesMatchWith<"result type must mathces array type",
                                         "array", "value",
                                         "::llvm::cast<emitHLS::ArrayType>($_self).getElementType()">,
                         MemoryEffectsOpInterface]>
{
    let summary = "Write a value into an array";

    let arguments = (ins AnyType:$value,
                         Arg<emitHLS_ArrayType>:$array,
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

def emitHLS_ArrayPointerReadOp : emitHLS_ArrayOpBase<"ptr_read",
                        [TypesMatchWith<"result type must mathces array type",
                                        "array", "result",
                                        "::llvm::cast<emitHLS::PointerType>($_self).getPointerType()">,
                         MemoryEffectsOpInterface]>
{
    let summary = "Read a value from an array pointer";

    let arguments = (ins Arg<emitHLS_PointerType>:$array, Index:$index);
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

def emitHLS_ArrayPointerWriteOp : emitHLS_ArrayOpBase<"ptr_write",
                         [TypesMatchWith<"result type must mathces array type",
                                         "array", "value",
                                         "::llvm::cast<emitHLS::PointerType>($_self).getPointerType()">,
                         MemoryEffectsOpInterface]>
{
    let summary = "Write a value into an array pointer";

    let arguments = (ins AnyType:$value,
                         Arg<emitHLS_PointerType>:$array,
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

#endif // EMITHLS_ARRAYOPS