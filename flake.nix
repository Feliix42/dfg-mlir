{
  description = "An over-engineered Hello World in C";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-23.05";
    mlir = {
      #url = "github:Feliix42/mlir.nix/main";
      url = "github:Feliix42/mlir.nix/circt";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, mlir }:
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
      nixpkgsFor = forAllSystems (system: import nixpkgs { inherit system; overlays = [ self.overlay ]; });

    in

    {

      # A Nixpkgs overlay.
      overlay = final: prev: {

        dfg_dialect = with final; final.callPackage ({ inShell ? false }: llvmPackages_16.stdenv.mkDerivation rec {
          pname = "dfg-mlir";
          inherit version;

          # need no copy of the source tree in the Nix Store when using `nix develop`
          src = if inShell then null else ./.;

          nativeBuildInputs = [
            python3
            ninja
            cmake
            llvmPackages_16.clang
            llvmPackages_16.bintools
            llvmPackages_16.openmp
            clang-tools_16
            mlir.packages.x86_64-linux.mlir
            mlir.packages.x86_64-linux.circt
            lit
          ];

          # buildInputs = (if inShell then [
          #   # in the `nix develop` shell, we also want:
          # ]);

          cmakeFlags = [
            "-GNinja"
            "-DMLIR_DIR=${mlir}/lib/cmake/mlir"
            "-DLLVM_DIR=${mlir}/lib/cmake/llvm"
            "-DCIRCT_DIR=${circt}/lib/cmake/circt"

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
            "-DLLVM_ENABLE_LLD=ON"
            "-DLLVM_EXTERNAL_LIT=${lit}/bin/lit"
          ];
        }) {};
      };

      # Provide some binary packages for selected system types.
      packages = forAllSystems (system:
        {
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
