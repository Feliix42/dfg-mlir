; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

%ChannelUntypedPtrsWrapper = type { i64, ptr }

@process_sink_wrap_global_string = private constant [18 x i8] c"process_sink_wrap\00"
@process_mul_wrap_global_string = private constant [17 x i8] c"process_mul_wrap\00"
@process_sum_wrap_global_string = private constant [17 x i8] c"process_sum_wrap\00"
@process_source_wrap_global_string = private constant [20 x i8] c"process_source_wrap\00"

declare { i64, i64, i64 } @source()

declare i64 @sum(i64, i64)

declare i64 @mul(i64, i64)

declare void @sink(i64)

define void @source_wrap(ptr %0, ptr %1, ptr %2) {
  %4 = call ptr @RTChannelUntypedPtrsWrapperGet(ptr %2, i64 0)
  %5 = call ptr @RTChannelUntypedPtrsWrapperGet(ptr %2, i64 1)
  %6 = call ptr @RTChannelUntypedPtrsWrapperGet(ptr %2, i64 2)
  br label %7

7:                                                ; preds = %3
  %8 = phi ptr [ %4, %3 ]
  %9 = phi ptr [ %5, %3 ]
  %10 = phi ptr [ %6, %3 ]
  %11 = call { i64, i64, i64 } @source()
  %12 = extractvalue { i64, i64, i64 } %11, 0
  %13 = extractvalue { i64, i64, i64 } %11, 1
  %14 = extractvalue { i64, i64, i64 } %11, 2
  %15 = alloca i64, align 4
  store i64 %12, ptr %15, align 4
  call void @C_INTERFACE_PushBytes(ptr %8, ptr %15)
  %16 = alloca i64, align 4
  store i64 %13, ptr %16, align 4
  call void @C_INTERFACE_PushBytes(ptr %9, ptr %16)
  %17 = alloca i64, align 4
  store i64 %14, ptr %17, align 4
  call void @C_INTERFACE_PushBytes(ptr %10, ptr %17)
  ret void
}

define void @sum_wrap(ptr %0, ptr %1, ptr %2) {
  %4 = call ptr @RTChannelUntypedPtrsWrapperGet(ptr %1, i64 0)
  %5 = call ptr @RTChannelUntypedPtrsWrapperGet(ptr %1, i64 1)
  %6 = call ptr @RTChannelUntypedPtrsWrapperGet(ptr %2, i64 0)
  br label %7

7:                                                ; preds = %3
  %8 = phi ptr [ %4, %3 ]
  %9 = phi ptr [ %5, %3 ]
  %10 = phi ptr [ %6, %3 ]
  %11 = call ptr @C_INTERFACE_PopBytes(ptr %8)
  %12 = load i64, ptr %11, align 4
  %13 = call ptr @C_INTERFACE_PopBytes(ptr %9)
  %14 = load i64, ptr %13, align 4
  %15 = call i64 @sum(i64 %12, i64 %14)
  %16 = alloca i64, align 4
  store i64 %15, ptr %16, align 4
  call void @C_INTERFACE_PushBytes(ptr %10, ptr %16)
  ret void
}

define void @mul_wrap(ptr %0, ptr %1, ptr %2) {
  %4 = call ptr @RTChannelUntypedPtrsWrapperGet(ptr %1, i64 0)
  %5 = call ptr @RTChannelUntypedPtrsWrapperGet(ptr %1, i64 1)
  %6 = call ptr @RTChannelUntypedPtrsWrapperGet(ptr %2, i64 0)
  br label %7

7:                                                ; preds = %3
  %8 = phi ptr [ %4, %3 ]
  %9 = phi ptr [ %5, %3 ]
  %10 = phi ptr [ %6, %3 ]
  %11 = call ptr @C_INTERFACE_PopBytes(ptr %8)
  %12 = load i64, ptr %11, align 4
  %13 = call ptr @C_INTERFACE_PopBytes(ptr %9)
  %14 = load i64, ptr %13, align 4
  %15 = call i64 @mul(i64 %12, i64 %14)
  %16 = alloca i64, align 4
  store i64 %15, ptr %16, align 4
  call void @C_INTERFACE_PushBytes(ptr %10, ptr %16)
  ret void
}

define void @sink_wrap(ptr %0, ptr %1, ptr %2) {
  %4 = call ptr @RTChannelUntypedPtrsWrapperGet(ptr %1, i64 0)
  br label %5

5:                                                ; preds = %3
  %6 = phi ptr [ %4, %3 ]
  %7 = call ptr @C_INTERFACE_PopBytes(ptr %6)
  %8 = load i64, ptr %7, align 4
  call void @sink(i64 %8)
  ret void
}

declare ptr @C_INTERFACE_AddChannel(ptr, i32)

declare void @ChannelUntypedPtrsWrapperSet(ptr, ptr, i64)

declare ptr @RTChannelUntypedPtrsWrapperGet(ptr, i64)

declare ptr @C_INTERFACE_PopBytes(ptr)

declare void @C_INTERFACE_PushBytes(ptr, ptr)

declare ptr @C_INTERFACE_GetMainRegion(ptr)

declare ptr @C_INTERFACE_CreateApplication(ptr)

declare void @C_INTERFACE_RunAndWaitForApplication(ptr, ptr)

declare ptr @C_INTERFACE_ManagerGetInstance()

declare void @ChannelUntypedPtrsWrapperInit(ptr, i64)

declare ptr @C_INTERFACE_AddKpnProcess(ptr, ptr, ptr, ptr, ptr)

define void @main() {
  %1 = call ptr @C_INTERFACE_ManagerGetInstance()
  %2 = call ptr @C_INTERFACE_CreateApplication(ptr %1)
  %3 = call ptr @C_INTERFACE_GetMainRegion(ptr %2)
  %4 = call ptr @C_INTERFACE_AddChannel(ptr %3, i32 8)
  %5 = call ptr @C_INTERFACE_AddChannel(ptr %3, i32 8)
  %6 = call ptr @C_INTERFACE_AddChannel(ptr %3, i32 8)
  %7 = call ptr @C_INTERFACE_AddChannel(ptr %3, i32 8)
  %8 = call ptr @C_INTERFACE_AddChannel(ptr %3, i32 8)
  %9 = alloca %ChannelUntypedPtrsWrapper, align 8
  call void @ChannelUntypedPtrsWrapperInit(ptr %9, i64 0)
  %10 = alloca %ChannelUntypedPtrsWrapper, i64 3, align 8
  call void @ChannelUntypedPtrsWrapperInit(ptr %10, i64 3)
  call void @ChannelUntypedPtrsWrapperSet(ptr %10, ptr %4, i64 0)
  call void @ChannelUntypedPtrsWrapperSet(ptr %10, ptr %5, i64 1)
  call void @ChannelUntypedPtrsWrapperSet(ptr %10, ptr %7, i64 2)
  %11 = call ptr @C_INTERFACE_AddKpnProcess(ptr %3, ptr @process_source_wrap_global_string, ptr @source_wrap, ptr %9, ptr %10)
  %12 = alloca %ChannelUntypedPtrsWrapper, align 8
  call void @ChannelUntypedPtrsWrapperInit(ptr %12, i64 2)
  call void @ChannelUntypedPtrsWrapperSet(ptr %12, ptr %4, i64 0)
  call void @ChannelUntypedPtrsWrapperSet(ptr %12, ptr %5, i64 1)
  %13 = alloca %ChannelUntypedPtrsWrapper, i64 1, align 8
  call void @ChannelUntypedPtrsWrapperInit(ptr %13, i64 1)
  call void @ChannelUntypedPtrsWrapperSet(ptr %13, ptr %6, i64 0)
  %14 = call ptr @C_INTERFACE_AddKpnProcess(ptr %3, ptr @process_sum_wrap_global_string, ptr @sum_wrap, ptr %12, ptr %13)
  %15 = alloca %ChannelUntypedPtrsWrapper, align 8
  call void @ChannelUntypedPtrsWrapperInit(ptr %15, i64 2)
  call void @ChannelUntypedPtrsWrapperSet(ptr %15, ptr %6, i64 0)
  call void @ChannelUntypedPtrsWrapperSet(ptr %15, ptr %7, i64 1)
  %16 = alloca %ChannelUntypedPtrsWrapper, i64 1, align 8
  call void @ChannelUntypedPtrsWrapperInit(ptr %16, i64 1)
  call void @ChannelUntypedPtrsWrapperSet(ptr %16, ptr %8, i64 0)
  %17 = call ptr @C_INTERFACE_AddKpnProcess(ptr %3, ptr @process_mul_wrap_global_string, ptr @mul_wrap, ptr %15, ptr %16)
  %18 = alloca %ChannelUntypedPtrsWrapper, align 8
  call void @ChannelUntypedPtrsWrapperInit(ptr %18, i64 1)
  call void @ChannelUntypedPtrsWrapperSet(ptr %18, ptr %8, i64 0)
  %19 = alloca %ChannelUntypedPtrsWrapper, i64 0, align 8
  call void @ChannelUntypedPtrsWrapperInit(ptr %19, i64 0)
  %20 = call ptr @C_INTERFACE_AddKpnProcess(ptr %3, ptr @process_sink_wrap_global_string, ptr @sink_wrap, ptr %18, ptr %19)
  call void @C_INTERFACE_RunAndWaitForApplication(ptr %1, ptr %2)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
