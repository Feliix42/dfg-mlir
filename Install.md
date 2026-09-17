# Installation

`dfg-mlir` builds with CMake (3.20+) and requires LLVM 22.1.7.

## Quick Start

We recommend Ninja as the build system with clang and lld. Two build methods are available:

1. **Script-based**: Use `build_dfg.sh` for a simple, automated build
2. **Nix**: Use the flake for a reproducible development environment

## Using the Build Script

The [build_dfg.sh](build_dfg.sh) script automates the process. Provide the path to your LLVM directory:

```bash
chmod +x build_dfg.sh
./build_dfg.sh /path/to/llvm
```

This builds LLVM and dfg-mlir. Outputs are in `build/`:
- `dfg-opt` — MLIR optimization and transformation tool
- `dfg-lsp-server` — Language Server Protocol implementation for VS Code integration with the [MLIR extension](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-mlir)

## Using Nix

For Nix users: enable [Flakes](https://nixos.wiki/wiki/Flakes#Enable_flakes), then run:

```bash
nix develop
cmake -S . -B build
cmake --build build
```

To build documentation:

```bash
cmake --build build --target build-dfg-doc
```

The flake provides all dependencies: LLVM 22.1.7, MLIR, clang, and build tools.

## Manual Build

Set these CMake variables to point to your LLVM 22.1.7 installation:

| Variable | Type | Description |
|----------|------|-------------|
| `LLVM_DIR` | STRING | Path to LLVM cmake config (e.g., `/path/to/llvm/lib/cmake/llvm`) |
| `MLIR_DIR` | STRING | Path to MLIR cmake config (e.g., `/path/to/llvm/lib/cmake/mlir`) |

Then configure and build:

```bash
cmake -B build -G Ninja \
    -DLLVM_DIR=$LLVM_DIR \
    -DMLIR_DIR=$MLIR_DIR
cmake --build build
```
