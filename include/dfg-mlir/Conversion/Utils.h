/// Declares some shared utility functions for conversions.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)

#pragma once

#include "mlir/IR/Value.h"

#include "llvm/ADT/SmallVector.h"

using namespace llvm;
using namespace mlir;

namespace {

class LowerHelper {
public:
    static SmallVector<std::pair<Value, Value>> newConnections;

    static SmallVector<std::pair<Value, Value>> getConnections()
    {
        return newConnections;
    }

    template<typename T1, typename T2>
    static std::optional<T1>
    getNewIndexOrArg(T2 find, SmallVector<std::pair<T1, T2>> args)
    {
        for (const auto &kv : args)
            if (kv.second == find) return kv.first;
        return std::nullopt;
    }
};

} // namespace
