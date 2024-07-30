/// Declares some shared utility functions for conversions.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

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

template<typename T>
bool isInSmallVector(T find, SmallVector<T> vec)
{
    for (const auto &elem : vec)
        if (elem == find) return true;
    return false;
}

template<typename T>
std::optional<int> getVectorIdx(T find, SmallVector<T> vec)
{
    if (vec.empty()) return std::nullopt;
    for (size_t i = 0; i < vec.size(); i++)
        if (vec[i] == find) return (int)i;
    return std::nullopt;
}
