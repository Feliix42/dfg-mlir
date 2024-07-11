#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset

# Make sure polygeist submodule is up-to-date.
git submodule sync
git submodule update --init --recursive

DFG_DIR="$(pwd)"
LLVM_DIR=$DFG_DIR/circt/llvm
CIRCT_DIR=$DFG_DIR/circt
LLVM_BUILD_DIR=$LLVM_DIR/build
CIRCT_BUILD_DIR=$CIRCT_DIR/build

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
    -DLLVM_USE_LINKER=mold \
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

# Configure CMake for CIRCT

cd $CIRCT_DIR
echo ""
echo "--- Configuring CIRCT ---"
echo ""

cmake -B build -G Ninja \
    -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
    -DLLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_USE_LINKER=mold \
    -DBUILD_SHARED_LIBS=1 \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Building CIRCT

echo ""
echo "--- Building CIRCT ---"
echo ""

cmake --build build

echo "--- Done building CIRCT ---"

# Configure CMake for dfg-mlir

cd $DFG_DIR
echo ""
echo "--- Configuring dfg-mlir ---"
echo ""

cmake -B build -G Ninja \
    -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
    -DLLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm \
    -DCIRCT_DIR=$CIRCT_BUILD_DIR/lib/cmake/circt \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_CXX_STANDARD=20 \
    -DLLVM_USE_LINKER=mold

# Building dfg-mlir

echo ""
echo "--- Building dfg-mlir ---"
echo ""

cmake --build build

echo "--- Done building dfg-mlir ---"
echo "--- !!!Have fun!!! ---"
