# `dfg-mlir`

This repository implements an MLIR dialect for representing dataflow graphs.

## Building

The `dfg-mlir` project is built using **CMake** (version `3.20` or newer). Make sure to provide all dependencies required by the project, either by installing them to system-default locations, or by setting the appropriate search location hints!

### Using `nix`

To build the dialect on a Nix-based system, you can use the provided `flake.nix` file to get a development shell up and running.
Make sure that you have [enabled Flakes](https://nixos.wiki/wiki/Flakes#Enable_flakes) for your Nix installation (verifiable by calling `nix flake` from the command line).

Then, just run `nix develop` to get a shell that provides LLVM and MLIR versions working for this project.
Build the dialect by running

```bash
# Configure.
cmake -S . -B build $cmakeFlags
# Build.
cmake --build build
```

### Manually

Make sure, you have MLIR and LLVM built and available on your system. Then point CMake to the include directories as shown below:

```sh
# Configure.
cmake -S . -B build \
    -G Ninja \
    -DLLVM_DIR=$LLVM_PREFIX/lib/cmake/llvm \
    -DMLIR_DIR=$MLIR_PREFIX/lib/cmake/mlir \
    -DCIRCT_DIR=$CIRCT_PREFIX/lib/cmake/circt
    (Optional)
    -DCMAKE_C_COMPILER=<clang> \
    -DCMAKE_CXX_COMPILER=<clang++> \
    -DLLVM_USE_LINKER=<lld>

# Build.
cmake --build build
```

The following CMake variables can be configured:

|       Name  | Type     | Description |
| ---------:  | :------- | --- |
| `LLVM_DIR`  | `STRING` | Path to the CMake directory of an **LLVM** installation. <br/> *e.g. `~/tools/llvm-17/lib/cmake/llvm`* |
| `MLIR_DIR`  | `STRING` | Path to the CMake directory of an **MLIR** installation. <br/> *e.g. `~/tools/llvm-17/lib/cmake/mlir`* |
| `CIRCT_DIR` | `STRING` | Path to the CMake directory of an **CIRCT** installation. <br/> *e.g. `~/tools/circt/lib/cmake/circt`* |

Notice that the llvm version should be the same as the `CIRCT` is using.

## Acknowledgements

`dfg-mlir` has been supported throughout its history by the following projects.

European Union projects:

- Grant agreement ID 957269 **EVEREST** – dEsign enVironmEnt foR Extreme-Scale big data analytics on heterogeneous platforms

