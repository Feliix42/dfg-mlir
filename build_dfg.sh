#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset

DFG_DIR="$(pwd)"
LLVM_DIR=$1
LLVM_BUILD_DIR=$LLVM_DIR/build

# Configure CMake for LLVM

cd $LLVM_DIR
echo ""
echo "--- Configuring LLVM ---"
echo ""

cmake -S llvm -B build -G Ninja \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_STANDARD=17 \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_USE_LINKER=lld \
    -DBUILD_SHARED_LIBS=1 \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Building LLVM

echo ""
echo "--- Building LLVM ---"
echo ""

cmake --build build

echo "--- Done building LLVM ---"

# Configure CMake for dfg-mlir

cd $DFG_DIR
echo ""
echo "--- Configuring dfg-mlir ---"
echo ""

cmake -B build -G Ninja \
    -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
    -DLLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_CXX_STANDARD=20 \
    -DLLVM_USE_LINKER=lld \
    -DLLVM_EXTERNAL_LIT=$LLVM_BUILD_DIR/build/bin/llvm-lit

# Building dfg-mlir

echo ""
echo "--- Building dfg-mlir ---"
echo ""

cmake --build build

echo "--- Done building dfg-mlir ---"

cd build && ninja check-dfg-mlir

echo "--- Done checking dfg-mlir ---"
echo "--- !!!Have fun!!! ---"
