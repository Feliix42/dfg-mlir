let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {


  nativeBuildInputs = with pkgs; [
    yaml-cpp
    protobuf
    (callPackage ../../../tetris/default.nix {})
    (callPackage ../../../dppm/default.nix {})
    python3
    gdb
    openssl
  ];

}
