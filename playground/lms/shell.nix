let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {


  nativeBuildInputs = with pkgs; [
    pkgsStatic.yaml-cpp
    (callPackage ../../../dppm/default.nix {})
    (pkgs.python3.withPackages (python-pkgs: [
      python-pkgs.matplotlib
    ]))
    gdb
    pkgsStatic.openssl
    glibc.static
  ];

}
