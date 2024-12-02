{
  description = "An over-engineered Hello World in C";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-24.05";
  };

  outputs = { self, nixpkgs }:
    let

      # to work with older version of flakes
      lastModifiedDate = self.lastModifiedDate or self.lastModified or "19700101";

      # Generate a user-friendly version number.
      version = builtins.substring 0 8 lastModifiedDate;

      # System types to support.
      supportedSystems = [ "x86_64-linux" "aarch64-linux" ]; # "x86_64-darwin" "aarch64-darwin" ];

      # Helper function to generate an attrset '{ x86_64-linux = f "x86_64-linux"; ... }'.
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;

      # Nixpkgs instantiated for supported system types.
      nixpkgsFor = forAllSystems (system: import nixpkgs { inherit system; overlays = [ self.overlay ]; });

    in

    {

      # A Nixpkgs overlay.
      overlay = final: prev: {

        # a specific version of llvm is needed for CIRCT
        llvm-custom = with final; llvmPackages_18.stdenv.mkDerivation rec {
          name = "llvm-custom";
          src = fetchFromGitHub {
            owner = "llvm";
            repo = "llvm-project";
            rev = "2ee2b6aa7a3d9ba6ba13f6881b25e26d7d12c823";
            sha256 = "sha256-piXv4YUf8q5Lf36my0bHtVkSJTrc+NvyZpLEwLPfV7U="; #lib.fakeSha256;
          };
          sourceRoot = "source/llvm";
          nativeBuildInputs = [
            python3
            ninja
            cmake
            llvmPackages_18.bintools
          ];
          buildInputs = [ libxml2 ];
          cmakeFlags = [
            "-GNinja"
            "-DCMAKE_BUILD_TYPE=Release"
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
            # from the original LLVM expr
            "-DLLVM_LINK_LLVM_DYLIB=ON"
            "-DLLVM_INSTALL_UTILS=ON"
            # change this to enable the projects you need
            "-DLLVM_ENABLE_PROJECTS=mlir"
            "-DLLVM_TARGETS_TO_BUILD=X86;AArch64"
            # NOTE(feliix42): THIS IS ABI BREAKING!!
            "-DLLVM_ENABLE_ASSERTIONS=ON"
            # Using clang and lld speeds up the build, we recomment adding:
            "-DCMAKE_C_COMPILER=clang"
            "-DCMAKE_CXX_COMPILER=clang++"
            "-DLLVM_ENABLE_LLD=ON"
          ];
        };


        # there are some new commits in the CIRCT repository that are needed for this project
        circt-custom = with final; llvmPackages_18.stdenv.mkDerivation rec {
          name = "circt-custom";
          src = fetchFromGitHub {
            owner = "llvm";
            repo = "circt";
            rev = "464f177ddcd3eb737fe8a592d20a24b15f25aef6";
            sha256 = "sha256-gl/TGZe30GAGnVrZl+iSBLTujddz7Qs6jUOq2gZ4xVI="; #lib.fakeSha256;
          };
          sourceRoot = "source/";
          nativeBuildInputs = [
            cmake
            ninja
            python3
            llvmPackages_18.bintools
            zlib
          ];
          cmakeFlags = [
            "-GNinja"
            "-DMLIR_DIR=${llvm-custom}/lib/cmake/mlir"
            "-DLLVM_DIR=${llvm-custom}/lib/cmake/llvm"
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
            "-DLLVM_LINK_LLVM_DYLIB=ON"
            "-DLLVM_TARGETS_TO_BUILD=X86;AArch64"
            "-DLLVM_ENABLE_ASSERTIONS=ON"
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
            "-DCMAKE_C_COMPILER=clang"
            "-DCMAKE_CXX_COMPILER=clang++"
            "-DLLVM_ENABLE_LLD=ON"
            "-DLLVM_EXTERNAL_LIT=${lit}/bin/lit"
            "-DCIRCT_LLHD_SIM_ENABLED=OFF"
          ];
        };

        dfg_dialect = with final; final.callPackage ({ inShell ? false }: llvmPackages_18.stdenv.mkDerivation rec {
          pname = "dfg-mlir";
          inherit version;

          # need no copy of the source tree in the Nix Store when using `nix develop`
          src = if inShell then null else ./.;

          nativeBuildInputs = [
            cmake
            ninja
            llvm-custom
            circt-custom
            llvmPackages_18.bintools
            zlib
          ];

          cmakeFlags = [
            "-GNinja"
            "-DMLIR_DIR=${llvm-custom}/lib/cmake/mlir"
            "-DLLVM_DIR=${llvm-custom}/lib/cmake/llvm"
            "-DCIRCT_DIR=${circt-custom}/lib/cmake/circt"
            "-DMLIR_TABLEGEN_EXE=${llvm-custom}/bin/mlir-tblgen"
            # Debug for debug builds
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
            "-DCMAKE_C_COMPILER=clang"
            "-DCMAKE_CXX_COMPILER=clang++"
            "-DLLVM_ENABLE_LLD=ON"
          ];

          installPhase = ''
            cmake --install . --verbose
            mkdir -p $out/bin
            cp bin/dfg-opt $out/bin
          '';

        }) {};
      };

      # Provide some binary packages for selected system types. -> packages."X".dfg_dialect = nixpkgsFor."X".dfg_dialect;
      packages = forAllSystems (system: {
        inherit (nixpkgsFor.${system}) dfg_dialect;
        # The default package for 'nix build'. This makes sense if the
        # flake provides only one package or there is a clear "main"
        # package.
        default = self.packages.${system}.dfg_dialect;
      });

      devShells = forAllSystems (system: let pkgs = nixpkgsFor.${system}; in {
         # default shell for dfg development
         default = pkgs.dfg_dialect.override { inShell = true; };
         # shell for dpm_connection
         dpm_connect = pkgs.mkShell {
           buildInputs = with pkgs; [
             (dfg_dialect.override { inShell = false; })
             (callPackage ./external/dppm/default.nix {})
           ];
         };
      });

      # A NixOS module, if applicable (e.g. if the package provides a system service).
      nixosModules.dfg_dialect =
        { pkgs, ... }:
        {
          nixpkgs.overlays = [ self.overlay ];

          environment.systemPackages = [ pkgs.dfg_dialect ];
        };

    };
}
