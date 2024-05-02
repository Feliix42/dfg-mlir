module attributes {llvm.data_layout = ""} {
  llvm.func @channel(i64) -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func @pull(!llvm.ptr, !llvm.ptr) -> i1 attributes {sym_visibility = "private"}
  llvm.func @push(!llvm.ptr, !llvm.ptr) -> i1 attributes {sym_visibility = "private"}
  llvm.func @close_channel(!llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @source() -> !llvm.struct<(i64, i64, i64)> attributes {sym_visibility = "private"}
  llvm.func @sum(i64, i64) -> i64 attributes {sym_visibility = "private"}
  llvm.func @mul(i64, i64) -> i64 attributes {sym_visibility = "private"}
  llvm.func @sink(i64) attributes {sym_visibility = "private"}
  llvm.func @source_wrap(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.call @source() : () -> !llvm.struct<(i64, i64, i64)>
    %2 = llvm.extractvalue %1[0] : !llvm.struct<(i64, i64, i64)> 
    %3 = llvm.extractvalue %1[1] : !llvm.struct<(i64, i64, i64)> 
    %4 = llvm.extractvalue %1[2] : !llvm.struct<(i64, i64, i64)> 
    %5 = llvm.alloca %0 x i64 : (i64) -> !llvm.ptr<i64>
    llvm.store %2, %5 : !llvm.ptr<i64>
    %6 = llvm.bitcast %5 : !llvm.ptr<i64> to !llvm.ptr
    %7 = llvm.call @push(%arg0, %6) : (!llvm.ptr, !llvm.ptr) -> i1
    llvm.cond_br %7, ^bb1, ^bb4
  ^bb1:  // pred: ^bb0
    %8 = llvm.alloca %0 x i64 : (i64) -> !llvm.ptr<i64>
    llvm.store %3, %8 : !llvm.ptr<i64>
    %9 = llvm.bitcast %8 : !llvm.ptr<i64> to !llvm.ptr
    %10 = llvm.call @push(%arg1, %9) : (!llvm.ptr, !llvm.ptr) -> i1
    llvm.cond_br %10, ^bb2, ^bb4
  ^bb2:  // pred: ^bb1
    %11 = llvm.alloca %0 x i64 : (i64) -> !llvm.ptr<i64>
    llvm.store %4, %11 : !llvm.ptr<i64>
    %12 = llvm.bitcast %11 : !llvm.ptr<i64> to !llvm.ptr
    %13 = llvm.call @push(%arg2, %12) : (!llvm.ptr, !llvm.ptr) -> i1
    llvm.cond_br %13, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    llvm.br ^bb4
  ^bb4:  // 4 preds: ^bb0, ^bb1, ^bb2, ^bb3
    llvm.call @close_channel(%arg0) : (!llvm.ptr) -> ()
    llvm.call @close_channel(%arg1) : (!llvm.ptr) -> ()
    llvm.call @close_channel(%arg2) : (!llvm.ptr) -> ()
    llvm.return
  }
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
  llvm.func @sink_wrap(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i64 : (i64) -> !llvm.ptr<i64>
    %2 = llvm.bitcast %1 : !llvm.ptr<i64> to !llvm.ptr
    %3 = llvm.call @pull(%arg0, %2) : (!llvm.ptr, !llvm.ptr) -> i1
    %4 = llvm.load %1 : !llvm.ptr<i64>
    llvm.cond_br %3, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.call @sink(%4) : (i64) -> ()
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    llvm.call @close_channel(%arg0) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant(4 : i32) : i32
    %1 = llvm.mlir.null : !llvm.ptr<i64>
    %2 = llvm.getelementptr %1[1] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %3 = llvm.ptrtoint %2 : !llvm.ptr<i64> to i64
    %4 = llvm.call @channel(%3) : (i64) -> !llvm.ptr
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
    omp.parallel   num_threads(%0 : i32) {
      omp.sections   nowait {
        omp.section {
          llvm.call @source_wrap(%4, %8, %16) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
          omp.terminator
        }
        omp.section {
          llvm.call @sum_wrap(%4, %8, %12) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
          omp.terminator
        }
        omp.section {
          llvm.call @mul_wrap(%12, %16, %20) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
          omp.terminator
        }
        omp.section {
          llvm.call @sink_wrap(%20) : (!llvm.ptr) -> ()
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
}

