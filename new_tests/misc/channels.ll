; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8

declare ptr @malloc(i64)

declare void @free(ptr)

define ptr @channel_i64() {
  %1 = call ptr @malloc(i64 ptrtoint (ptr getelementptr ({ ptr, i64, i64, i64, i64, i8 }, ptr null, i64 1) to i64))
  %2 = call ptr @malloc(i64 ptrtoint (ptr getelementptr ({ ptr, i64, i64, i64, i64, i8 }, ptr null, i64 64) to i64))
  %3 = insertvalue { ptr, i64, i64, i64, i64, i8 } undef, ptr %2, 0
  %4 = insertvalue { ptr, i64, i64, i64, i64, i8 } %3, i64 0, 1
  %5 = insertvalue { ptr, i64, i64, i64, i64, i8 } %4, i64 0, 2
  %6 = insertvalue { ptr, i64, i64, i64, i64, i8 } %5, i64 64, 3
  %7 = insertvalue { ptr, i64, i64, i64, i64, i8 } %6, i64 0, 4
  %8 = insertvalue { ptr, i64, i64, i64, i64, i8 } %7, i8 1, 5
  store { ptr, i64, i64, i64, i64, i8 } %8, ptr %1, align 8
  ret ptr %1
}

define i1 @push_i64(ptr %0, i64 %1) {
  br label %3

3:                                                ; preds = %24, %2
  %4 = getelementptr inbounds { ptr, i64, i64, i64, i64, i8 }, ptr %0, i32 0, i32 5
  %5 = load i8, ptr %4, align 1
  %6 = icmp ne i8 %5, 0
  br i1 %6, label %7, label %26

7:                                                ; preds = %3
  %8 = getelementptr inbounds { ptr, i64, i64, i64, i64, i8 }, ptr %0, i32 0, i32 4
  %9 = getelementptr inbounds { ptr, i64, i64, i64, i64, i8 }, ptr %0, i32 0, i32 3
  %10 = load i64, ptr %8, align 4
  %11 = load i64, ptr %9, align 4
  %12 = icmp ult i64 %10, %11
  br i1 %12, label %13, label %24

13:                                               ; preds = %7
  %14 = phi ptr [ %8, %7 ]
  %15 = phi i64 [ %11, %7 ]
  %16 = getelementptr inbounds { ptr, i64, i64, i64, i64, i8 }, ptr %0, i32 0, i32 2
  %17 = load i64, ptr %16, align 4
  %18 = getelementptr inbounds { ptr, i64, i64, i64, i64, i8 }, ptr %0, i32 0, i32 0
  %19 = load ptr, ptr %18, align 8
  %20 = getelementptr i64, ptr %19, i64 %17
  store i64 %1, ptr %20, align 4
  %21 = add i64 %17, 1
  %22 = urem i64 %21, %15
  store i64 %22, ptr %16, align 4
  %23 = atomicrmw add ptr %14, i64 1 monotonic, align 8
  br label %26

24:                                               ; preds = %7
  %omp_global_thread_num = call i32 @__kmpc_global_thread_num(ptr @1)
  %25 = call i32 @__kmpc_omp_taskyield(ptr @1, i32 %omp_global_thread_num, i32 0)
  br label %3

26:                                               ; preds = %13, %3
  %27 = phi i1 [ true, %13 ], [ false, %3 ]
  ret i1 %27
}

define { i64, i1 } @pull_i64(ptr %0) {
  br label %2

2:                                                ; preds = %10, %1
  %3 = getelementptr inbounds { ptr, i64, i64, i64, i64, i8 }, ptr %0, i32 0, i32 4
  %4 = load i64, ptr %3, align 4
  %5 = icmp eq i64 %4, 0
  br i1 %5, label %6, label %13

6:                                                ; preds = %2
  %7 = getelementptr inbounds { ptr, i64, i64, i64, i64, i8 }, ptr %0, i32 0, i32 5
  %8 = load i8, ptr %7, align 1
  %9 = icmp ne i8 %8, 0
  br i1 %9, label %10, label %12

10:                                               ; preds = %6
  %omp_global_thread_num = call i32 @__kmpc_global_thread_num(ptr @1)
  %11 = call i32 @__kmpc_omp_taskyield(ptr @1, i32 %omp_global_thread_num, i32 0)
  br label %2

12:                                               ; preds = %6
  br label %26

13:                                               ; preds = %2
  %14 = phi ptr [ %3, %2 ]
  %15 = getelementptr inbounds { ptr, i64, i64, i64, i64, i8 }, ptr %0, i32 0, i32 3
  %16 = load i64, ptr %15, align 4
  %17 = getelementptr inbounds { ptr, i64, i64, i64, i64, i8 }, ptr %0, i32 0, i32 1
  %18 = load i64, ptr %17, align 4
  %19 = getelementptr inbounds { ptr, i64, i64, i64, i64, i8 }, ptr %0, i32 0, i32 0
  %20 = load ptr, ptr %19, align 8
  %21 = getelementptr i64, ptr %20, i64 %18
  %22 = load i64, ptr %21, align 4
  %23 = add i64 %18, 1
  %24 = urem i64 %23, %16
  store i64 %24, ptr %17, align 4
  %25 = atomicrmw sub ptr %14, i64 1 monotonic, align 8
  br label %26

26:                                               ; preds = %12, %13
  %27 = phi i64 [ %22, %13 ], [ poison, %12 ]
  %28 = phi i1 [ true, %13 ], [ false, %12 ]
  %29 = insertvalue { i64, i1 } undef, i64 %27, 0
  %30 = insertvalue { i64, i1 } %29, i1 %28, 1
  ret { i64, i1 } %30
}

define void @close_i64(ptr %0) {
  %2 = getelementptr inbounds { ptr, i64, i64, i64, i64, i8 }, ptr %0, i32 0, i32 4
  %3 = cmpxchg ptr %2, i8 1, i8 0 acq_rel monotonic, align 1
  %4 = extractvalue { i8, i1 } %3, 1
  br i1 %4, label %8, label %5

5:                                                ; preds = %1
  %6 = getelementptr inbounds { ptr, i64, i64, i64, i64, i8 }, ptr %0, i32 0, i32 0
  %7 = load ptr, ptr %6, align 8
  call void @free(ptr %7)
  call void @free(ptr %0)
  br label %8

8:                                                ; preds = %5, %1
  ret void
}

; Function Attrs: nounwind
declare i32 @__kmpc_global_thread_num(ptr) #0

; Function Attrs: nounwind
declare i32 @__kmpc_omp_taskyield(ptr, i32, i32) #0

attributes #0 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
