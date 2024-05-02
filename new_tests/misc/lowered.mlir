module attributes {llvm.data_layout = ""} {
  llvm.func @channel(i64) -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func @pull(!llvm.ptr, !llvm.ptr) -> i1 attributes {sym_visibility = "private"}
  llvm.func @push(!llvm.ptr, !llvm.ptr) -> i1 attributes {sym_visibility = "private"}
  llvm.func @close_channel(!llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @printf(!llvm.ptr<i8>, ...) -> i32
  llvm.mlir.global internal constant @str0("num recvd: %d\0A\00") {addr_space = 0 : i32}
  llvm.func @sum(i64, i64) -> i64 attributes {sym_visibility = "private"}
  llvm.func @mul(i64, i64) -> i64 attributes {sym_visibility = "private"}
  llvm.func @sum_wrap(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i64 : (i64) -> !llvm.ptr<i64>
    %2 = llvm.bitcast %1 : !llvm.ptr<i64> to !llvm.ptr
    %3 = llvm.call @pull(%arg0, %2) : (!llvm.ptr, !llvm.ptr) -> i1
    %4 = llvm.load %1 : !llvm.ptr<i64>
    llvm.cond_br %3, ^bb1, ^bb4
  ^bb1:  // pred: ^bb0
    %5 = llvm.alloca %0 x i64 : (i64) -> !llvm.ptr<i64>
    %6 = llvm.bitcast %5 : !llvm.ptr<i64> to !llvm.ptr
    %7 = llvm.call @pull(%arg1, %6) : (!llvm.ptr, !llvm.ptr) -> i1
    %8 = llvm.load %5 : !llvm.ptr<i64>
    llvm.cond_br %7, ^bb2, ^bb4
  ^bb2:  // pred: ^bb1
    %9 = llvm.call @sum(%4, %8) : (i64, i64) -> i64
    %10 = llvm.alloca %0 x i64 : (i64) -> !llvm.ptr<i64>
    llvm.store %9, %10 : !llvm.ptr<i64>
    %11 = llvm.bitcast %10 : !llvm.ptr<i64> to !llvm.ptr
    %12 = llvm.call @push(%arg2, %11) : (!llvm.ptr, !llvm.ptr) -> i1
    llvm.cond_br %12, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    llvm.br ^bb4
  ^bb4:  // 4 preds: ^bb0, ^bb1, ^bb2, ^bb3
    llvm.call @close_channel(%arg2) : (!llvm.ptr) -> ()
    llvm.call @close_channel(%arg0) : (!llvm.ptr) -> ()
    llvm.call @close_channel(%arg1) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @mul_wrap(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i64 : (i64) -> !llvm.ptr<i64>
    %2 = llvm.bitcast %1 : !llvm.ptr<i64> to !llvm.ptr
    %3 = llvm.call @pull(%arg0, %2) : (!llvm.ptr, !llvm.ptr) -> i1
    %4 = llvm.load %1 : !llvm.ptr<i64>
    llvm.cond_br %3, ^bb1, ^bb4
  ^bb1:  // pred: ^bb0
    %5 = llvm.alloca %0 x i64 : (i64) -> !llvm.ptr<i64>
    %6 = llvm.bitcast %5 : !llvm.ptr<i64> to !llvm.ptr
    %7 = llvm.call @pull(%arg1, %6) : (!llvm.ptr, !llvm.ptr) -> i1
    %8 = llvm.load %5 : !llvm.ptr<i64>
    llvm.cond_br %7, ^bb2, ^bb4
  ^bb2:  // pred: ^bb1
    %9 = llvm.call @mul(%4, %8) : (i64, i64) -> i64
    %10 = llvm.alloca %0 x i64 : (i64) -> !llvm.ptr<i64>
    llvm.store %9, %10 : !llvm.ptr<i64>
    %11 = llvm.bitcast %10 : !llvm.ptr<i64> to !llvm.ptr
    %12 = llvm.call @push(%arg2, %11) : (!llvm.ptr, !llvm.ptr) -> i1
    llvm.cond_br %12, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    llvm.br ^bb4
  ^bb4:  // 4 preds: ^bb0, ^bb1, ^bb2, ^bb3
    llvm.call @close_channel(%arg2) : (!llvm.ptr) -> ()
    llvm.call @close_channel(%arg0) : (!llvm.ptr) -> ()
    llvm.call @close_channel(%arg1) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant(2 : i32) : i32
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(2 : i64) : i64
    %3 = llvm.mlir.constant(3 : i64) : i64
    %4 = llvm.mlir.constant(4 : i64) : i64
    %5 = llvm.mlir.null : !llvm.ptr<i64>
    %6 = llvm.getelementptr %5[1] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %7 = llvm.ptrtoint %6 : !llvm.ptr<i64> to i64
    %8 = llvm.call @channel(%7) : (i64) -> !llvm.ptr
    %9 = llvm.mlir.null : !llvm.ptr<i64>
    %10 = llvm.getelementptr %9[1] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %11 = llvm.ptrtoint %10 : !llvm.ptr<i64> to i64
    %12 = llvm.call @channel(%11) : (i64) -> !llvm.ptr
    %13 = llvm.mlir.null : !llvm.ptr<i64>
    %14 = llvm.getelementptr %13[1] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %15 = llvm.ptrtoint %14 : !llvm.ptr<i64> to i64
    %16 = llvm.call @channel(%15) : (i64) -> !llvm.ptr
    %17 = llvm.mlir.null : !llvm.ptr<i64>
    %18 = llvm.getelementptr %17[1] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %19 = llvm.ptrtoint %18 : !llvm.ptr<i64> to i64
    %20 = llvm.call @channel(%19) : (i64) -> !llvm.ptr
    %21 = llvm.mlir.null : !llvm.ptr<i64>
    %22 = llvm.getelementptr %21[1] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %23 = llvm.ptrtoint %22 : !llvm.ptr<i64> to i64
    %24 = llvm.call @channel(%23) : (i64) -> !llvm.ptr
    %25 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr<i64>
    llvm.store %2, %25 : !llvm.ptr<i64>
    %26 = llvm.bitcast %25 : !llvm.ptr<i64> to !llvm.ptr
    %27 = llvm.call @push(%8, %26) : (!llvm.ptr, !llvm.ptr) -> i1
    llvm.cond_br %27, ^bb1, ^bb5
  ^bb1:  // pred: ^bb0
    %28 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr<i64>
    llvm.store %3, %28 : !llvm.ptr<i64>
    %29 = llvm.bitcast %28 : !llvm.ptr<i64> to !llvm.ptr
    %30 = llvm.call @push(%12, %29) : (!llvm.ptr, !llvm.ptr) -> i1
    llvm.cond_br %30, ^bb2, ^bb5
  ^bb2:  // pred: ^bb1
    %31 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr<i64>
    llvm.store %4, %31 : !llvm.ptr<i64>
    %32 = llvm.bitcast %31 : !llvm.ptr<i64> to !llvm.ptr
    %33 = llvm.call @push(%20, %32) : (!llvm.ptr, !llvm.ptr) -> i1
    llvm.cond_br %33, ^bb3, ^bb5
  ^bb3:  // pred: ^bb2
    omp.parallel   num_threads(%0 : i32) {
      omp.sections   nowait {
        omp.section {
          llvm.call @sum_wrap(%8, %12, %16) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
          omp.terminator
        }
        omp.section {
          llvm.call @mul_wrap(%16, %20, %24) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }
    %34 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr<i64>
    %35 = llvm.bitcast %34 : !llvm.ptr<i64> to !llvm.ptr
    %36 = llvm.call @pull(%24, %35) : (!llvm.ptr, !llvm.ptr) -> i1
    %37 = llvm.load %34 : !llvm.ptr<i64>
    llvm.cond_br %36, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %38 = llvm.mlir.addressof @str0 : !llvm.ptr<array<15 x i8>>
    %39 = llvm.getelementptr %38[0, 0] : (!llvm.ptr<array<15 x i8>>) -> !llvm.ptr<i8>
    %40 = llvm.call @printf(%39, %37) : (!llvm.ptr<i8>, i64) -> i32
    llvm.br ^bb5
  ^bb5:  // 5 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4
    llvm.call @close_channel(%8) : (!llvm.ptr) -> ()
    llvm.call @close_channel(%12) : (!llvm.ptr) -> ()
    llvm.call @close_channel(%20) : (!llvm.ptr) -> ()
    llvm.call @close_channel(%24) : (!llvm.ptr) -> ()
    llvm.return
  }
}

