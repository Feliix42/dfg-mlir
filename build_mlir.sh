#!/usr/bin/env bash

set -euo pipefail

# LLVM hash to build
LLVMHASH=45e874e39030bc622ea43fbcfc4fcdd1dd404353

LIBDIR=$PWD/libs
LLVMDIR=$LIBDIR/llvm
LLVMOUT=$LIBDIR/llvm_build

if [ -x "$(nproc)" ]
then
    CPUCOUNT=$(nproc)
else
    # MacOS
    CPUCOUNT=$(sysctl -n hw.ncpu)
fi

mkdir -p $LIBDIR

wget -nc --output-document $LIBDIR/llvm.tar.gz http://github.com/llvm/llvm-project/archive/$LLVMHASH.tar.gz

tar xf $LIBDIR/llvm.tar.gz --directory $LIBDIR

mv $LIBDIR/llvm-project-$LLVMHASH $LLVMDIR


cd $LLVMDIR/llvm
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DLLVM_LINK_LLVM_DYLIB=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_ENABLE_PROJECTS="clang;lldb;lld;openmp;mlir" \
    -DLLVM_ENABLE_RUNTIMES="compiler-rt" \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_INSTALL_PREFIX=$LLVMOUT
cmake --build build -j$CPUCOUNT
cmake --build build --target install -j$CPUCOUNT

rm -rf $LLVMDIR/llvm/build

echo "###############################################################################"
echo ""
echo "  [build completed]"
echo "    Build outputs have been produced and installed to the following paths:"
echo "      - LLVM:  $LLVMOUT"
echo ""
echo "###############################################################################"

