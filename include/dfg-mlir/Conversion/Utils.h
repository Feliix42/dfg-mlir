/// Declares some shared utility functions for conversions.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@mailbox.tu-dresden.de)

#pragma once

#include "mlir/IR/Value.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

// Searching for the mapped value of provided one
template<typename T1, typename T2>
std::optional<T1> getNewIndexOrArg(T2 find, SmallVector<std::pair<T1, T2>> args)
{
    for (const auto &kv : args)
        if (kv.second == find) return kv.first;
    return std::nullopt;
}
