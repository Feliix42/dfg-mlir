; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@str0 = internal constant [15 x i8] c"num recvd: %d\0A\00"
@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8

declare ptr @malloc(i64)

declare void @free(ptr)

declare ptr @channel(i64)

declare i1 @pull(ptr, ptr)

declare i1 @push(ptr, ptr)

declare void @close_channel(ptr)

declare i32 @printf(ptr, ...)

declare i64 @sum(i64, i64)

declare i64 @mul(i64, i64)

define void @sum_wrap(ptr %0, ptr %1, ptr %2) {
  %4 = alloca i64, i64 1, align 8
  %5 = call i1 @pull(ptr %0, ptr %4)
  %6 = load i64, ptr %4, align 4
  br i1 %5, label %7, label %16

7:                                                ; preds = %3
  %8 = alloca i64, i64 1, align 8
  %9 = call i1 @pull(ptr %1, ptr %8)
  %10 = load i64, ptr %8, align 4
  br i1 %9, label %11, label %16

11:                                               ; preds = %7
  %12 = call i64 @sum(i64 %6, i64 %10)
  %13 = alloca i64, i64 1, align 8
  store i64 %12, ptr %13, align 4
  %14 = call i1 @push(ptr %2, ptr %13)
  br i1 %14, label %15, label %16

15:                                               ; preds = %11
  br label %16

16:                                               ; preds = %15, %11, %7, %3
  call void @close_channel(ptr %2)
  call void @close_channel(ptr %0)
  call void @close_channel(ptr %1)
  ret void
}

define void @mul_wrap(ptr %0, ptr %1, ptr %2) {
  %4 = alloca i64, i64 1, align 8
  %5 = call i1 @pull(ptr %0, ptr %4)
  %6 = load i64, ptr %4, align 4
  br i1 %5, label %7, label %16

7:                                                ; preds = %3
  %8 = alloca i64, i64 1, align 8
  %9 = call i1 @pull(ptr %1, ptr %8)
  %10 = load i64, ptr %8, align 4
  br i1 %9, label %11, label %16

11:                                               ; preds = %7
  %12 = call i64 @mul(i64 %6, i64 %10)
  %13 = alloca i64, i64 1, align 8
  store i64 %12, ptr %13, align 4
  %14 = call i1 @push(ptr %2, ptr %13)
  br i1 %14, label %15, label %16

15:                                               ; preds = %11
  br label %16

16:                                               ; preds = %15, %11, %7, %3
  call void @close_channel(ptr %2)
  call void @close_channel(ptr %0)
  call void @close_channel(ptr %1)
  ret void
}

define void @main() {
  %structArg = alloca { ptr, ptr, ptr, ptr, ptr }, align 8
  %1 = call ptr @channel(i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64))
  %2 = call ptr @channel(i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64))
  %3 = call ptr @channel(i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64))
  %4 = call ptr @channel(i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64))
  %5 = call ptr @channel(i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64))
  %6 = alloca i64, i64 1, align 8
  store i64 2, ptr %6, align 4
  %7 = call i1 @push(ptr %1, ptr %6)
  br i1 %7, label %8, label %20

8:                                                ; preds = %0
  %9 = alloca i64, i64 1, align 8
  store i64 3, ptr %9, align 4
  %10 = call i1 @push(ptr %2, ptr %9)
  br i1 %10, label %11, label %20

11:                                               ; preds = %8
  %12 = alloca i64, i64 1, align 8
  store i64 4, ptr %12, align 4
  %13 = call i1 @push(ptr %4, ptr %12)
  br i1 %13, label %14, label %20

14:                                               ; preds = %11
  %omp_global_thread_num = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_push_num_threads(ptr @1, i32 %omp_global_thread_num, i32 2)
  br label %omp_parallel

omp_parallel:                                     ; preds = %14
  %gep_ = getelementptr { ptr, ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 0
  store ptr %3, ptr %gep_, align 8
  %gep_7 = getelementptr { ptr, ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 1
  store ptr %4, ptr %gep_7, align 8
  %gep_8 = getelementptr { ptr, ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 2
  store ptr %5, ptr %gep_8, align 8
  %gep_9 = getelementptr { ptr, ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 3
  store ptr %1, ptr %gep_9, align 8
  %gep_10 = getelementptr { ptr, ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 4
  store ptr %2, ptr %gep_10, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @main..omp_par, ptr %structArg)
  br label %omp.par.outlined.exit

omp.par.outlined.exit:                            ; preds = %omp_parallel
  br label %omp.par.exit.split

omp.par.exit.split:                               ; preds = %omp.par.outlined.exit
  %15 = alloca i64, i64 1, align 8
  %16 = call i1 @pull(ptr %5, ptr %15)
  %17 = load i64, ptr %15, align 4
  br i1 %16, label %18, label %20

18:                                               ; preds = %omp.par.exit.split
  %19 = call i32 (ptr, ...) @printf(ptr @str0, i64 %17)
  br label %20

20:                                               ; preds = %18, %omp.par.exit.split, %11, %8, %0
  call void @close_channel(ptr %1)
  call void @close_channel(ptr %2)
  call void @close_channel(ptr %4)
  call void @close_channel(ptr %5)
  ret void
}

; Function Attrs: norecurse nounwind
define internal void @main..omp_par(ptr noalias %tid.addr, ptr noalias %zero.addr, ptr %0) #0 {
omp.par.entry:
  %gep_ = getelementptr { ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_ = load ptr, ptr %gep_, align 8
  %gep_1 = getelementptr { ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_2 = load ptr, ptr %gep_1, align 8
  %gep_3 = getelementptr { ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 2
  %loadgep_4 = load ptr, ptr %gep_3, align 8
  %gep_5 = getelementptr { ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 3
  %loadgep_6 = load ptr, ptr %gep_5, align 8
  %gep_7 = getelementptr { ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 4
  %loadgep_8 = load ptr, ptr %gep_7, align 8
  %tid.addr.local = alloca i32, align 4
  %1 = load i32, ptr %tid.addr, align 4
  store i32 %1, ptr %tid.addr.local, align 4
  %tid = load i32, ptr %tid.addr.local, align 4
  %p.lastiter = alloca i32, align 4
  %p.lowerbound = alloca i32, align 4
  %p.upperbound = alloca i32, align 4
  %p.stride = alloca i32, align 4
  br label %omp.par.region

omp.par.region:                                   ; preds = %omp.par.entry
  br label %omp.par.region1

omp.par.region1:                                  ; preds = %omp.par.region
  br label %omp_section_loop.preheader

omp_section_loop.preheader:                       ; preds = %omp.par.region1
  store i32 0, ptr %p.lowerbound, align 4
  store i32 1, ptr %p.upperbound, align 4
  store i32 1, ptr %p.stride, align 4
  %omp_global_thread_num6 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_4u(ptr @1, i32 %omp_global_thread_num6, i32 34, ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride, i32 1, i32 0)
  %2 = load i32, ptr %p.lowerbound, align 4
  %3 = load i32, ptr %p.upperbound, align 4
  %4 = sub i32 %3, %2
  %5 = add i32 %4, 1
  br label %omp_section_loop.header

omp_section_loop.header:                          ; preds = %omp_section_loop.inc, %omp_section_loop.preheader
  %omp_section_loop.iv = phi i32 [ 0, %omp_section_loop.preheader ], [ %omp_section_loop.next, %omp_section_loop.inc ]
  br label %omp_section_loop.cond

omp_section_loop.cond:                            ; preds = %omp_section_loop.header
  %omp_section_loop.cmp = icmp ult i32 %omp_section_loop.iv, %5
  br i1 %omp_section_loop.cmp, label %omp_section_loop.body, label %omp_section_loop.exit

omp_section_loop.exit:                            ; preds = %omp_section_loop.cond
  call void @__kmpc_for_static_fini(ptr @1, i32 %omp_global_thread_num6)
  br label %omp_section_loop.after

omp_section_loop.after:                           ; preds = %omp_section_loop.exit
  br label %omp_section_loop.aftersections.fini

omp_section_loop.aftersections.fini:              ; preds = %omp_section_loop.after
  br label %omp.region.cont

omp.region.cont:                                  ; preds = %omp_section_loop.aftersections.fini
  br label %omp.par.pre_finalize

omp.par.pre_finalize:                             ; preds = %omp.region.cont
  br label %omp.par.outlined.exit.exitStub

omp_section_loop.body:                            ; preds = %omp_section_loop.cond
  %6 = add i32 %omp_section_loop.iv, %2
  %7 = mul i32 %6, 1
  %8 = add i32 %7, 0
  switch i32 %8, label %omp_section_loop.body.sections.after [
    i32 0, label %omp_section_loop.body.case
    i32 1, label %omp_section_loop.body.case3
  ]

omp_section_loop.body.case3:                      ; preds = %omp_section_loop.body
  br label %omp.section.region5

omp.section.region5:                              ; preds = %omp_section_loop.body.case3
  call void @mul_wrap(ptr %loadgep_, ptr %loadgep_2, ptr %loadgep_4)
  br label %omp.region.cont4

omp.region.cont4:                                 ; preds = %omp.section.region5
  br label %omp_section_loop.body.sections.after

omp_section_loop.body.case:                       ; preds = %omp_section_loop.body
  br label %omp.section.region

omp.section.region:                               ; preds = %omp_section_loop.body.case
  call void @sum_wrap(ptr %loadgep_6, ptr %loadgep_8, ptr %loadgep_)
  br label %omp.region.cont2

omp.region.cont2:                                 ; preds = %omp.section.region
  br label %omp_section_loop.body.sections.after

omp_section_loop.body.sections.after:             ; preds = %omp.region.cont4, %omp.region.cont2, %omp_section_loop.body
  br label %omp_section_loop.inc

omp_section_loop.inc:                             ; preds = %omp_section_loop.body.sections.after
  %omp_section_loop.next = add nuw i32 %omp_section_loop.iv, 1
  br label %omp_section_loop.header

omp.par.outlined.exit.exitStub:                   ; preds = %omp.par.pre_finalize
  ret void
}

; Function Attrs: nounwind
declare i32 @__kmpc_global_thread_num(ptr) #1

; Function Attrs: nounwind
declare void @__kmpc_push_num_threads(ptr, i32, i32) #1

; Function Attrs: nounwind
declare void @__kmpc_for_static_init_4u(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32) #1

; Function Attrs: nounwind
declare void @__kmpc_for_static_fini(ptr, i32) #1

; Function Attrs: nounwind
declare !callback !1 void @__kmpc_fork_call(ptr, i32, ptr, ...) #1

attributes #0 = { norecurse nounwind }
attributes #1 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{!2}
!2 = !{i64 2, i64 -1, i64 -1, i1 true}
