{
  description = "An over-engineered Hello World in C";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    mlir = {
      url = "github:Feliix42/mlir.nix/main";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      mlir,
    }:
    let

      # to work with older version of flakes
      lastModifiedDate = self.lastModifiedDate or self.lastModified or "19700101";

      # Generate a user-friendly version number.
      version = builtins.substring 0 8 lastModifiedDate;

      # System types to support.
      supportedSystems = [ "x86_64-linux" ]; # "x86_64-darwin" "aarch64-linux" "aarch64-darwin" ];

      # Helper function to generate an attrset '{ x86_64-linux = f "x86_64-linux"; ... }'.
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;

      # Nixpkgs instantiated for supported system types.
      nixpkgsFor = forAllSystems (
        system:
        import nixpkgs {
          inherit system;
          overlays = [ self.overlay ];
        }
      );


    in

    {

      # A Nixpkgs overlay.
      overlay = final: prev: {

        dfg_dialect =
          with final;
          final.callPackage (
            {
              inShell ? false,
            }:
            llvmPackages_22.stdenv.mkDerivation rec {
              pname = "dfg-mlir";
              inherit version;

              # need no copy of the source tree in the Nix Store when using `nix develop`
              src = if inShell then null else ./.;

              python = python312.override {
                packageOverrides = pfinal: pprev: {
                  numpy = pprev.numpy.overridePythonAttrs (old: rec {
                    version = "2.1.2";
                    src = fetchPypi {
                      inherit (old) pname;
                      inherit version;
                      hash = "sha256-E1MqCIIX+mJMmbhD7rVGQN4js0FLFKpm0COAXrcxBmw=";
                    };
                  });
                  exhale = pfinal.buildPythonPackage rec {
                    pname = "exhale";
                    version = "0.3.7";
                    pyproject = true;
                    src = fetchPypi {
                      inherit pname version;
                      hash = "sha256-dSqW0KWUVlEdkzMR1KgfZCzWaClurNJWGQVyfV7WsNg=";
                    };
                    build-system = [ pfinal.setuptools ];
                    dependencies = [
                      pfinal.breathe
                      pfinal.beautifulsoup4
                      pfinal.lxml
                      pfinal.six
                    ];
                    doCheck = false;
                  };
                };
              };

              pythonEnv = python.withPackages (
                ps: with ps; [
                  nanobind
                  pyyaml
                  typing-extensions
                  numpy
                  ml-dtypes
                  breathe
                  myst-parser
                  scikit-build-core
                  sphinx
                  sphinx-rtd-theme
                  exhale
                  lit
                ]
              );

              nativeBuildInputs = [
                pythonEnv
                ninja
                cmake
                llvmPackages_22.clang
                llvmPackages_22.bintools
                llvmPackages_22.openmp
                llvmPackages_22.clang-tools
                mlir.packages.x86_64-linux.mlir
                lit
                mold
                zlib
                doxygen
              ];

              # buildInputs = (if inShell then [
              #   # in the `nix develop` shell, we also want:
              # ]);

              cmakeFlags = [
                "-GNinja"
                "-DMLIR_DIR=${mlir}/lib/cmake/mlir"
                "-DLLVM_DIR=${mlir}/lib/cmake/llvm"

                # Debug for debug builds
                #"-DCMAKE_BUILD_TYPE=RelWithDebInfo"
                # this makes llvm only to produce code for the current platform, this saves CPU time, change it to what you need
                #"-DLLVM_TARGETS_TO_BUILD=X86"
                # NOTE(feliix42): THIS IS ABI BREAKING!!
                #"-DLLVM_ENABLE_ASSERTIONS=ON"
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
                # Using clang and lld speeds up the build, we recomment adding:
                "-DCMAKE_C_COMPILER=clang"
                "-DCMAKE_CXX_COMPILER=clang++"
                "-DCMAKE_CXX_FLAGS='-fuse-ld=mold'"
                "-DLLVM_EXTERNAL_LIT=${lit}/bin/lit"
              ];

              # Generate a .clangd config pointing at the Nix-provided toolchain
              # whenever we enter the dev shell.
              # shellHook = lib.optionalString inShell ''
              #   cat > .clangd <<EOF
              #   CompileFlags:
              #     Compiler: ${llvmPackages_22.clang}/bin/clang++
              #     CompilationDatabase: build
              #     Add:
              #       - -I$PWD/include
              #       - -I$PWD/build/include
              #       - -isystem${mlir}/include
              #       - -isystem${llvmPackages_22.clang-tools}/lib/clang/22/include
              #   EOF
              #   echo "Generated .clangd for this dev shell."
              # '';
            }
          ) { };
      };

      # Provide some binary packages for selected system types.
      packages = forAllSystems (system: {
        inherit (nixpkgsFor.${system}) dfg_dialect;
      });

      # The default package for 'nix build'. This makes sense if the
      # flake provides only one package or there is a clear "main"
      # package.
      defaultPackage = forAllSystems (system: self.packages.${system}.dfg_dialect);

      # Provide a 'nix develop' environment for interactive hacking.
      devShell = forAllSystems (system: self.packages.${system}.dfg_dialect.override { inShell = true; });

      # A NixOS module, if applicable (e.g. if the package provides a system service).
      nixosModules.dfg_dialect =
        { pkgs, ... }:
        {
          nixpkgs.overlays = [ self.overlay ];

          environment.systemPackages = [ pkgs.dfg_dialect ];
        };

    };
}
