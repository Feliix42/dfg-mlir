; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

%struct.ident_t = type { i32, i32, i32, i32, ptr }
%elem = type { i32, ptr }

@str0 = internal constant [15 x i8] c"num recvd: %d\0A\00"
@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8

declare ptr @malloc(i64)

declare void @free(ptr)

declare i32 @printf(ptr, ...)

define ptr @make_channel() {
  %1 = call ptr @malloc(i64 ptrtoint (ptr getelementptr ({ ptr, ptr, i64, i64, i1 }, ptr null, i32 1) to i64))
  store { ptr, ptr, i64, i64, i1 } { ptr null, ptr null, i64 -1, i64 0, i1 true }, ptr %1, align 8
  ret ptr %1
}

define void @send_data(ptr %0, i32 %1) {
  %3 = load { ptr, ptr, i64, i64, i1 }, ptr %0, align 8
  %4 = extractvalue { ptr, ptr, i64, i64, i1 } %3, 4
  br i1 %4, label %5, label %28

5:                                                ; preds = %2
  %6 = phi { ptr, ptr, i64, i64, i1 } [ %3, %2 ]
  %7 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%elem, ptr null, i32 1) to i64))
  %8 = insertvalue %elem undef, i32 %1, 0
  store %elem { i32 undef, ptr null }, ptr %7, align 8
  %9 = extractvalue { ptr, ptr, i64, i64, i1 } %6, 3
  %10 = icmp ne i64 %9, 0
  br i1 %10, label %16, label %11

11:                                               ; preds = %5
  %12 = phi { ptr, ptr, i64, i64, i1 } [ %6, %5 ]
  %13 = phi ptr [ %7, %5 ]
  %14 = insertvalue { ptr, ptr, i64, i64, i1 } %12, ptr %13, 0
  %15 = insertvalue { ptr, ptr, i64, i64, i1 } %14, ptr %13, 1
  br label %23

16:                                               ; preds = %5
  %17 = phi { ptr, ptr, i64, i64, i1 } [ %6, %5 ]
  %18 = phi ptr [ %7, %5 ]
  %19 = extractvalue { ptr, ptr, i64, i64, i1 } %6, 1
  %20 = load %elem, ptr %19, align 8
  %21 = insertvalue %elem %20, ptr %18, 1
  store %elem %21, ptr %19, align 8
  %22 = insertvalue { ptr, ptr, i64, i64, i1 } %17, ptr %18, 1
  br label %23

23:                                               ; preds = %16, %11
  %24 = phi { ptr, ptr, i64, i64, i1 } [ %22, %16 ], [ %15, %11 ]
  %25 = extractvalue { ptr, ptr, i64, i64, i1 } %24, 3
  %26 = add i64 %25, 1
  %27 = insertvalue { ptr, ptr, i64, i64, i1 } %24, i64 %26, 3
  store { ptr, ptr, i64, i64, i1 } %27, ptr %0, align 8
  br label %28

28:                                               ; preds = %23, %2
  ret void
}

define i32 @recv_data(ptr %0) {
  %2 = load { ptr, ptr, i64, i64, i1 }, ptr %0, align 8
  %3 = extractvalue { ptr, ptr, i64, i64, i1 } %2, 3
  %4 = icmp eq i64 %3, 0
  br i1 %4, label %5, label %8

5:                                                ; preds = %1
  %6 = phi { ptr, ptr, i64, i64, i1 } [ %2, %1 ]
  %7 = extractvalue { ptr, ptr, i64, i64, i1 } %6, 4
  br label %19

8:                                                ; preds = %1
  %9 = phi { ptr, ptr, i64, i64, i1 } [ %2, %1 ]
  %10 = phi i64 [ %3, %1 ]
  %11 = extractvalue { ptr, ptr, i64, i64, i1 } %9, 0
  %12 = load %elem, ptr %11, align 8
  %13 = extractvalue %elem %12, 1
  %14 = insertvalue { ptr, ptr, i64, i64, i1 } %9, ptr %13, 0
  %15 = sub i64 %10, 1
  %16 = getelementptr inbounds { ptr, ptr, i64, i64, i1 }, ptr %0, i32 0, i32 1
  %17 = cmpxchg ptr %16, ptr %13, ptr null acq_rel monotonic, align 8
  %18 = extractvalue %elem %12, 0
  call void @free(ptr %11)
  br label %19

19:                                               ; preds = %5, %8
  %20 = phi i32 [ %18, %8 ], [ 0, %5 ]
  ret i32 %20
}

define void @close(ptr %0) {
  %2 = load { ptr, ptr, i64, i64, i1 }, ptr %0, align 8
  %3 = insertvalue { ptr, ptr, i64, i64, i1 } %2, i1 false, 4
  store { ptr, ptr, i64, i64, i1 } %3, ptr %0, align 8
  ret void
}

define void @produce_value(ptr %0) {
  call void @send_data(ptr %0, i32 323729)
  call void @close(ptr %0)
  ret void
}

define void @consume(ptr %0) {
  %2 = call i32 @recv_data(ptr %0)
  %3 = call i32 (ptr, ...) @printf(ptr @str0, i32 %2)
  call void @close(ptr %0)
  ret void
}

define void @algo() {
  %structArg = alloca { ptr }, align 8
  %1 = call ptr @make_channel()
  br label %entry

entry:                                            ; preds = %0
  %omp_global_thread_num = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_push_num_threads(ptr @1, i32 %omp_global_thread_num, i32 4)
  br label %omp_parallel

omp_parallel:                                     ; preds = %entry
  %gep_ = getelementptr { ptr }, ptr %structArg, i32 0, i32 0
  store ptr %1, ptr %gep_, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @algo..omp_par, ptr %structArg)
  br label %omp.par.outlined.exit

omp.par.outlined.exit:                            ; preds = %omp_parallel
  br label %omp.par.exit.split

omp.par.exit.split:                               ; preds = %omp.par.outlined.exit
  ret void
}

; Function Attrs: norecurse nounwind
define internal void @algo..omp_par(ptr noalias %tid.addr, ptr noalias %zero.addr, ptr %0) #0 {
omp.par.entry:
  %gep_ = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %loadgep_ = load ptr, ptr %gep_, align 8
  %tid.addr.local = alloca i32, align 4
  %1 = load i32, ptr %tid.addr, align 4
  store i32 %1, ptr %tid.addr.local, align 4
  %tid = load i32, ptr %tid.addr.local, align 4
  br label %omp.par.region

omp.par.region:                                   ; preds = %omp.par.entry
  br label %omp.par.region1

omp.par.region1:                                  ; preds = %omp.par.region
  call void @produce_value(ptr %loadgep_)
  call void @consume(ptr %loadgep_)
  br label %omp.region.cont

omp.region.cont:                                  ; preds = %omp.par.region1
  br label %omp.par.pre_finalize

omp.par.pre_finalize:                             ; preds = %omp.region.cont
  br label %omp.par.outlined.exit.exitStub

omp.par.outlined.exit.exitStub:                   ; preds = %omp.par.pre_finalize
  ret void
}

; Function Attrs: nounwind
declare i32 @__kmpc_global_thread_num(ptr) #1

; Function Attrs: nounwind
declare void @__kmpc_push_num_threads(ptr, i32, i32) #1

; Function Attrs: nounwind
declare !callback !1 void @__kmpc_fork_call(ptr, i32, ptr, ...) #1

attributes #0 = { norecurse nounwind }
attributes #1 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{!2}
!2 = !{i64 2, i64 -1, i64 -1, i1 true}
