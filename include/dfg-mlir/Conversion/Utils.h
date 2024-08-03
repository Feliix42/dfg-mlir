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


template<typename ContainerA, typename ContainerB>
auto combine(ContainerA &&A, ContainerB &&B){
    using NonRefContainerA = std::remove_reference_t<ContainerA>;
    using NonRefContainerB = std::remove_reference_t<ContainerA>;
    static_assert(std::is_same_v<typename NonRefContainerA::value_type,typename NonRefContainerB::value_type>, "Containers must have the same element type");
    using ElementT = typename NonRefContainerA::value_type;
    std::vector<ElementT> result;
    result.reserve(A.size() + B.size());
    result.insert(result.end(), A.begin(), A.end());
    result.insert(result.end(), B.begin(), B.end());
    return result;
}

template<typename ContainerT, typename LambdaT>
auto map(ContainerT &&A, LambdaT &&lambda){
    using NonRefContainer = std::remove_reference_t<ContainerT>;
    using ReturnT = typename std::invoke_result_t<LambdaT, typename NonRefContainer::value_type>;
    std::vector<ReturnT> result;
    result.reserve(A.size());
    for(const auto& element : A){
        result.push_back(lambda(element));
    }
    return result;
}
