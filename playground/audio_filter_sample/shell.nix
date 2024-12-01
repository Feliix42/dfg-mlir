let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {


  nativeBuildInputs = with pkgs; [
	yaml-cpp
    pkgsStatic.yaml-cpp
    (callPackage ../../../tetris/default.nix {})
    (callPackage ../../../dppm/default.nix {})
    (pkgs.python3.withPackages (python-pkgs: [
      python-pkgs.matplotlib
    ]))
    gdb
	(kissfftFloat.override { enableStatic = true; })
	kissfftFloat
	glibc.static
  ];

}
