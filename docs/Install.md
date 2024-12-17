# Installation

In this repository there are two methods to build `dfg-mlir`, which are using `ninja` in `build_dfg.sh` script and `nix` build-system.

The `dfg-mlir` project is built using **CMake** (version `3.20` or newer). Make sure to provide all dependencies required by the project, either by installing them to system-default locations, or by setting the appropriate search location hints!

The same as most [MLIR](mlir.llvm.org)-based project, we recommand to use `Ninja` as the build system along with `clang` as the main compiler, as well as using `lld` to link the programs. Last but not least, `dfg-mlir` is based on llvm with this git hash `2ee2b6aa7a3d9ba6ba13f6881b25e26d7d12c823`.

## dummy version for most OS

[build_dfg.sh](../build_dfg.sh) script is provided in the root directory of this repository. It requires you to provide the absolute path to your clone of the llvm project.
```
If in the root dir:
./build_dfg.sh /path/to/you/llvm/dir
```
This script will automatically configure and build both `llvm` and our `dfg-mlir` for you. You'll find the executables inside `dfg-mlir/build` directory, i.e.
- `dfg-opt`: same concept as `opt` or `mlir-opt` in LLVM.
- `dfg-lsp-server`: this contain the Language Server Protocol(LSP), which you can use to detect error and etc. It's fully usable with [MLIR extension](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-mlir) in VS Code.
- `dfg-translate`: this contains several translations, which will transpile the `dfg-mlir` code into something else.

## Using Nix
If you're one of the `nix` enthusiastics, we kindly provide the possibility to build this project using it.

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

## Manually
Well, well, well! You succesfully drew my attention by reaching here. So huh, you are one of those "I don't trust you and I master CMake" people. But here are some rules you still need to follow if you want to build `dfg-mlir` with your own configuration. (Big brother is watching you all the time!)

The following CMake variables must be configured:

|       Name  | Type     | Description |
| ---------:  | :------- | --- |
| `LLVM_DIR`  | `STRING` | Path to the CMake directory of an **LLVM** installation. <br/> *e.g. `/your/path/to/llvm/lib/cmake/llvm`* |
| `MLIR_DIR`  | `STRING` | Path to the CMake directory of an **MLIR** installation. <br/> *e.g. `/your/path/to/llvm/lib/cmake/mlir`* |
