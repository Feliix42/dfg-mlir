module attributes {llvm.data_layout = ""} {
  llvm.func @source() -> !llvm.struct<(i64, i64, i64)> attributes {sym_visibility = "private"}
  llvm.func @sum(i64, i64) -> i64 attributes {sym_visibility = "private"}
  llvm.func @mul(i64, i64) -> i64 attributes {sym_visibility = "private"}
  llvm.func @sink(i64) attributes {sym_visibility = "private"}
  llvm.func @source_wrap(%arg0: !llvm.ptr<struct<"RTChannelUntypedPtrsWrapper", (i32)>>, %arg1: !llvm.ptr<struct<"RTChannelUntypedPtrsWrapper", (i32)>>) {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.call @RTChannelUntypedPtrsWrapperGet(%arg1, %0) : (!llvm.ptr<struct<"RTChannelUntypedPtrsWrapper", (i32)>>, i64) -> !llvm.ptr<struct<"RTChannelUntyped", (i32)>>
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.call @RTChannelUntypedPtrsWrapperGet(%arg1, %2) : (!llvm.ptr<struct<"RTChannelUntypedPtrsWrapper", (i32)>>, i64) -> !llvm.ptr<struct<"RTChannelUntyped", (i32)>>
    %4 = llvm.mlir.constant(2 : i64) : i64
    %5 = llvm.call @RTChannelUntypedPtrsWrapperGet(%arg1, %4) : (!llvm.ptr<struct<"RTChannelUntypedPtrsWrapper", (i32)>>, i64) -> !llvm.ptr<struct<"RTChannelUntyped", (i32)>>
    llvm.br ^bb1(%1, %3, %5 : !llvm.ptr<struct<"RTChannelUntyped", (i32)>>, !llvm.ptr<struct<"RTChannelUntyped", (i32)>>, !llvm.ptr<struct<"RTChannelUntyped", (i32)>>)
  ^bb1(%6: !llvm.ptr<struct<"RTChannelUntyped", (i32)>>, %7: !llvm.ptr<struct<"RTChannelUntyped", (i32)>>, %8: !llvm.ptr<struct<"RTChannelUntyped", (i32)>>):  // pred: ^bb0
    %9 = llvm.call @source() : () -> !llvm.struct<(i64, i64, i64)>
    %10 = llvm.extractvalue %9[0] : !llvm.struct<(i64, i64, i64)> 
    %11 = llvm.extractvalue %9[1] : !llvm.struct<(i64, i64, i64)> 
    %12 = llvm.extractvalue %9[2] : !llvm.struct<(i64, i64, i64)> 
    llvm.call @C_INTERFACE_PushBytes(%6, %10) : (!llvm.ptr<struct<"RTChannelUntyped", (i32)>>, i64) -> ()
    llvm.call @C_INTERFACE_PushBytes(%7, %11) : (!llvm.ptr<struct<"RTChannelUntyped", (i32)>>, i64) -> ()
    llvm.call @C_INTERFACE_PushBytes(%8, %12) : (!llvm.ptr<struct<"RTChannelUntyped", (i32)>>, i64) -> ()
    llvm.return
  }
  llvm.func @sum_wrap(%arg0: !llvm.ptr<struct<"RTChannelUntypedPtrsWrapper", (i32)>>, %arg1: !llvm.ptr<struct<"RTChannelUntypedPtrsWrapper", (i32)>>) {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.call @RTChannelUntypedPtrsWrapperGet(%arg0, %0) : (!llvm.ptr<struct<"RTChannelUntypedPtrsWrapper", (i32)>>, i64) -> !llvm.ptr<struct<"RTChannelUntyped", (i32)>>
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.call @RTChannelUntypedPtrsWrapperGet(%arg0, %2) : (!llvm.ptr<struct<"RTChannelUntypedPtrsWrapper", (i32)>>, i64) -> !llvm.ptr<struct<"RTChannelUntyped", (i32)>>
    %4 = llvm.mlir.constant(0 : i64) : i64
    %5 = llvm.call @RTChannelUntypedPtrsWrapperGet(%arg1, %4) : (!llvm.ptr<struct<"RTChannelUntypedPtrsWrapper", (i32)>>, i64) -> !llvm.ptr<struct<"RTChannelUntyped", (i32)>>
    llvm.br ^bb1(%1, %3, %5 : !llvm.ptr<struct<"RTChannelUntyped", (i32)>>, !llvm.ptr<struct<"RTChannelUntyped", (i32)>>, !llvm.ptr<struct<"RTChannelUntyped", (i32)>>)
  ^bb1(%6: !llvm.ptr<struct<"RTChannelUntyped", (i32)>>, %7: !llvm.ptr<struct<"RTChannelUntyped", (i32)>>, %8: !llvm.ptr<struct<"RTChannelUntyped", (i32)>>):  // pred: ^bb0
    %9 = llvm.call @C_INTERFACE_PopBytes(%6) : (!llvm.ptr<struct<"RTChannelUntyped", (i32)>>) -> i64
    %10 = llvm.call @C_INTERFACE_PopBytes(%7) : (!llvm.ptr<struct<"RTChannelUntyped", (i32)>>) -> i64
    %11 = llvm.call @sum(%9, %10) : (i64, i64) -> i64
    llvm.call @C_INTERFACE_PushBytes(%8, %11) : (!llvm.ptr<struct<"RTChannelUntyped", (i32)>>, i64) -> ()
    llvm.return
  }
  llvm.func @mul_wrap(%arg0: !llvm.ptr<struct<"RTChannelUntypedPtrsWrapper", (i32)>>, %arg1: !llvm.ptr<struct<"RTChannelUntypedPtrsWrapper", (i32)>>) {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.call @RTChannelUntypedPtrsWrapperGet(%arg0, %0) : (!llvm.ptr<struct<"RTChannelUntypedPtrsWrapper", (i32)>>, i64) -> !llvm.ptr<struct<"RTChannelUntyped", (i32)>>
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.call @RTChannelUntypedPtrsWrapperGet(%arg0, %2) : (!llvm.ptr<struct<"RTChannelUntypedPtrsWrapper", (i32)>>, i64) -> !llvm.ptr<struct<"RTChannelUntyped", (i32)>>
    %4 = llvm.mlir.constant(0 : i64) : i64
    %5 = llvm.call @RTChannelUntypedPtrsWrapperGet(%arg1, %4) : (!llvm.ptr<struct<"RTChannelUntypedPtrsWrapper", (i32)>>, i64) -> !llvm.ptr<struct<"RTChannelUntyped", (i32)>>
    llvm.br ^bb1(%1, %3, %5 : !llvm.ptr<struct<"RTChannelUntyped", (i32)>>, !llvm.ptr<struct<"RTChannelUntyped", (i32)>>, !llvm.ptr<struct<"RTChannelUntyped", (i32)>>)
  ^bb1(%6: !llvm.ptr<struct<"RTChannelUntyped", (i32)>>, %7: !llvm.ptr<struct<"RTChannelUntyped", (i32)>>, %8: !llvm.ptr<struct<"RTChannelUntyped", (i32)>>):  // pred: ^bb0
    %9 = llvm.call @C_INTERFACE_PopBytes(%6) : (!llvm.ptr<struct<"RTChannelUntyped", (i32)>>) -> i64
    %10 = llvm.call @C_INTERFACE_PopBytes(%7) : (!llvm.ptr<struct<"RTChannelUntyped", (i32)>>) -> i64
    %11 = llvm.call @mul(%9, %10) : (i64, i64) -> i64
    llvm.call @C_INTERFACE_PushBytes(%8, %11) : (!llvm.ptr<struct<"RTChannelUntyped", (i32)>>, i64) -> ()
    llvm.return
  }
  llvm.func @sink_wrap(%arg0: !llvm.ptr<struct<"RTChannelUntypedPtrsWrapper", (i32)>>, %arg1: !llvm.ptr<struct<"RTChannelUntypedPtrsWrapper", (i32)>>) {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.call @RTChannelUntypedPtrsWrapperGet(%arg0, %0) : (!llvm.ptr<struct<"RTChannelUntypedPtrsWrapper", (i32)>>, i64) -> !llvm.ptr<struct<"RTChannelUntyped", (i32)>>
    llvm.br ^bb1(%1 : !llvm.ptr<struct<"RTChannelUntyped", (i32)>>)
  ^bb1(%2: !llvm.ptr<struct<"RTChannelUntyped", (i32)>>):  // pred: ^bb0
    %3 = llvm.call @C_INTERFACE_PopBytes(%2) : (!llvm.ptr<struct<"RTChannelUntyped", (i32)>>) -> i64
    llvm.call @sink(%3) : (i64) -> ()
    llvm.return
  }
  llvm.func @C_INTERFACE_AddChannel(i32) -> !llvm.ptr<struct<"ChannelUntyped", (i32)>>
  llvm.func @ChannelUntypedPtrsWrapperSet(!llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>, !llvm.ptr<struct<"ChannelUntyped", (i32)>>, i64)
  llvm.func @RTChannelUntypedPtrsWrapperGet(!llvm.ptr<struct<"RTChannelUntypedPtrsWrapper", (i32)>>, i64) -> !llvm.ptr<struct<"RTChannelUntyped", (i32)>>
  llvm.func @C_INTERFACE_PopBytes(!llvm.ptr<struct<"RTChannelUntyped", (i32)>>) -> i64
  llvm.func @C_INTERFACE_PushBytes(!llvm.ptr<struct<"RTChannelUntyped", (i32)>>, i64)
  llvm.func @C_INTERFACE_GetMainRegion(!llvm.ptr<struct<"class.Dppm::Application", (i32)>>) -> !llvm.ptr<struct<"class.Dppm::RegionUntyped", (i32)>>
  llvm.func @C_INTERFACE_CreateApplication(!llvm.ptr<struct<"class.Dppm::Manager", (i32)>>) -> !llvm.ptr<struct<"class.Dppm::Application", (i32)>>
  llvm.func @C_INTERFACE_ManagerGetInstance() -> !llvm.ptr<struct<"class.Dppm::Manager", (i32)>>
  llvm.func @ChannelUntypedPtrsWrapperInit(!llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>, i64)
  llvm.func @C_INTERFACE_AddKpnProcess(!llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>, !llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>) -> !llvm.ptr
  llvm.func @main() {
    %0 = llvm.call @C_INTERFACE_ManagerGetInstance() : () -> !llvm.ptr<struct<"class.Dppm::Manager", (i32)>>
    %1 = llvm.call @C_INTERFACE_CreateApplication(%0) : (!llvm.ptr<struct<"class.Dppm::Manager", (i32)>>) -> !llvm.ptr<struct<"class.Dppm::Application", (i32)>>
    %2 = llvm.call @C_INTERFACE_GetMainRegion(%1) : (!llvm.ptr<struct<"class.Dppm::Application", (i32)>>) -> !llvm.ptr<struct<"class.Dppm::RegionUntyped", (i32)>>
    %3 = llvm.mlir.constant(4 : i32) : i32
    %4 = llvm.call @C_INTERFACE_AddChannel(%3) : (i32) -> !llvm.ptr<struct<"ChannelUntyped", (i32)>>
    %5 = llvm.mlir.constant(4 : i32) : i32
    %6 = llvm.call @C_INTERFACE_AddChannel(%5) : (i32) -> !llvm.ptr<struct<"ChannelUntyped", (i32)>>
    %7 = llvm.mlir.constant(4 : i32) : i32
    %8 = llvm.call @C_INTERFACE_AddChannel(%7) : (i32) -> !llvm.ptr<struct<"ChannelUntyped", (i32)>>
    %9 = llvm.mlir.constant(4 : i32) : i32
    %10 = llvm.call @C_INTERFACE_AddChannel(%9) : (i32) -> !llvm.ptr<struct<"ChannelUntyped", (i32)>>
    %11 = llvm.mlir.constant(4 : i32) : i32
    %12 = llvm.call @C_INTERFACE_AddChannel(%11) : (i32) -> !llvm.ptr<struct<"ChannelUntyped", (i32)>>
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.alloca %13 x !llvm.struct<"ChannelUntypedPtrsWrapper", (i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>
    %15 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @ChannelUntypedPtrsWrapperInit(%14, %15) : (!llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>, i64) -> ()
    %16 = llvm.alloca %13 x !llvm.struct<"ChannelUntypedPtrsWrapper", (i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>
    %17 = llvm.mlir.constant(3 : i64) : i64
    llvm.call @ChannelUntypedPtrsWrapperInit(%16, %17) : (!llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>, i64) -> ()
    %18 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @ChannelUntypedPtrsWrapperSet(%16, %4, %18) : (!llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>, !llvm.ptr<struct<"ChannelUntyped", (i32)>>, i64) -> ()
    %19 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @ChannelUntypedPtrsWrapperSet(%16, %6, %19) : (!llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>, !llvm.ptr<struct<"ChannelUntyped", (i32)>>, i64) -> ()
    %20 = llvm.mlir.constant(2 : i64) : i64
    llvm.call @ChannelUntypedPtrsWrapperSet(%16, %10, %20) : (!llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>, !llvm.ptr<struct<"ChannelUntyped", (i32)>>, i64) -> ()
    %21 = llvm.call @C_INTERFACE_AddKpnProcess(%14, %16) : (!llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>, !llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>) -> !llvm.ptr
    %22 = llvm.mlir.constant(1 : i32) : i32
    %23 = llvm.alloca %22 x !llvm.struct<"ChannelUntypedPtrsWrapper", (i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>
    %24 = llvm.mlir.constant(2 : i64) : i64
    llvm.call @ChannelUntypedPtrsWrapperInit(%23, %24) : (!llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>, i64) -> ()
    %25 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @ChannelUntypedPtrsWrapperSet(%23, %4, %25) : (!llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>, !llvm.ptr<struct<"ChannelUntyped", (i32)>>, i64) -> ()
    %26 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @ChannelUntypedPtrsWrapperSet(%23, %6, %26) : (!llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>, !llvm.ptr<struct<"ChannelUntyped", (i32)>>, i64) -> ()
    %27 = llvm.alloca %22 x !llvm.struct<"ChannelUntypedPtrsWrapper", (i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>
    %28 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @ChannelUntypedPtrsWrapperInit(%27, %28) : (!llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>, i64) -> ()
    %29 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @ChannelUntypedPtrsWrapperSet(%27, %8, %29) : (!llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>, !llvm.ptr<struct<"ChannelUntyped", (i32)>>, i64) -> ()
    %30 = llvm.call @C_INTERFACE_AddKpnProcess(%23, %27) : (!llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>, !llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>) -> !llvm.ptr
    %31 = llvm.mlir.constant(1 : i32) : i32
    %32 = llvm.alloca %31 x !llvm.struct<"ChannelUntypedPtrsWrapper", (i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>
    %33 = llvm.mlir.constant(2 : i64) : i64
    llvm.call @ChannelUntypedPtrsWrapperInit(%32, %33) : (!llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>, i64) -> ()
    %34 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @ChannelUntypedPtrsWrapperSet(%32, %8, %34) : (!llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>, !llvm.ptr<struct<"ChannelUntyped", (i32)>>, i64) -> ()
    %35 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @ChannelUntypedPtrsWrapperSet(%32, %10, %35) : (!llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>, !llvm.ptr<struct<"ChannelUntyped", (i32)>>, i64) -> ()
    %36 = llvm.alloca %31 x !llvm.struct<"ChannelUntypedPtrsWrapper", (i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>
    %37 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @ChannelUntypedPtrsWrapperInit(%36, %37) : (!llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>, i64) -> ()
    %38 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @ChannelUntypedPtrsWrapperSet(%36, %12, %38) : (!llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>, !llvm.ptr<struct<"ChannelUntyped", (i32)>>, i64) -> ()
    %39 = llvm.call @C_INTERFACE_AddKpnProcess(%32, %36) : (!llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>, !llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>) -> !llvm.ptr
    %40 = llvm.mlir.constant(1 : i32) : i32
    %41 = llvm.alloca %40 x !llvm.struct<"ChannelUntypedPtrsWrapper", (i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>
    %42 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @ChannelUntypedPtrsWrapperInit(%41, %42) : (!llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>, i64) -> ()
    %43 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @ChannelUntypedPtrsWrapperSet(%41, %12, %43) : (!llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>, !llvm.ptr<struct<"ChannelUntyped", (i32)>>, i64) -> ()
    %44 = llvm.alloca %40 x !llvm.struct<"ChannelUntypedPtrsWrapper", (i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>
    %45 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @ChannelUntypedPtrsWrapperInit(%44, %45) : (!llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>, i64) -> ()
    %46 = llvm.call @C_INTERFACE_AddKpnProcess(%41, %44) : (!llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>, !llvm.ptr<struct<"ChannelUntypedPtrsWrapper", (i32)>>) -> !llvm.ptr
    llvm.return
  }
}

