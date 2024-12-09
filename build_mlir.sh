#!/usr/bin/env bash

set -euo pipefail

# LLVM and CIRCT hashes to build
# NOTE: The LLVM hash *must* match the hash required by CIRCT
LLVMHASH=2ee2b6aa7a3d9ba6ba13f6881b25e26d7d12c823
CIRCTHASH=464f177ddcd3eb737fe8a592d20a24b15f25aef6

LIBDIR=$PWD/libs
LLVMDIR=$LIBDIR/llvm
CIRCTDIR=$LIBDIR/circt
LLVMOUT=$LIBDIR/llvm_build
CIRCTOUT=$LIBDIR/circt_build

if [ -x "$(nproc)" ]
then
    CPUCOUNT=$(nproc)
else
    # MacOS
    CPUCOUNT=$(sysctl -n hw.ncpu)
fi

mkdir -p $LIBDIR

wget -nc --output-document $LIBDIR/llvm.tar.gz http://github.com/llvm/llvm-project/archive/$LLVMHASH.tar.gz
wget -nc --output-document $LIBDIR/circt.tar.gz http://github.com/llvm/circt/archive/$CIRCTHASH.tar.gz 

tar xf $LIBDIR/llvm.tar.gz --directory $LIBDIR
tar xf $LIBDIR/circt.tar.gz --directory $LIBDIR

mv $LIBDIR/llvm-project-$LLVMHASH $LLVMDIR
mv $LIBDIR/circt-$CIRCTHASH $CIRCTDIR


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

cd $CIRCTDIR

cmake -S . -B build \
    -GNinja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DLLVM_LINK_LLVM_DYLIB=ON \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=$LLVMOUT/bin/clang \
    -DCMAKE_CXX_COMPILER=$LLVMOUT/bin/clang++ \
    -DLLVM_ENABLE_LLD=ON \
    -DVERILATOR_DISABLE=ON \
    -DMLIR_DIR=$LLVMOUT/lib/cmake/mlir \
    -DLLVM_DIR=$LLVMOUT/lib/cmake/llvm \
    -DCMAKE_INSTALL_PREFIX=$CIRCTOUT

cmake --build build -j$CPUCOUNT
cmake --build build --target install -j$CPUCOUNT

rm -rf $CIRCTDIR/build


echo "###############################################################################"
echo ""
echo "  [build completed]"
echo "    Build outputs have been produced and installed to the following paths:"
echo "      - LLVM:  $LLVMOUT"
echo "      - CIRCT: $CIRCTOUT"
echo ""
echo "###############################################################################"

