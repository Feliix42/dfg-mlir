; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

%ChannelUntypedPtrsWrapper = type { i32 }

declare ptr @malloc(i64)

declare void @free(ptr)

declare { i64, i64, i64 } @source()

declare i64 @sum(i64, i64)

declare i64 @mul(i64, i64)

declare void @sink(i64)

define void @source_wrap(ptr %0, ptr %1) {
  %3 = call ptr @RTChannelUntypedPtrsWrapperGet(ptr %1, i64 0)
  %4 = call ptr @RTChannelUntypedPtrsWrapperGet(ptr %1, i64 1)
  %5 = call ptr @RTChannelUntypedPtrsWrapperGet(ptr %1, i64 2)
  br label %6

6:                                                ; preds = %2
  %7 = phi ptr [ %3, %2 ]
  %8 = phi ptr [ %4, %2 ]
  %9 = phi ptr [ %5, %2 ]
  %10 = call { i64, i64, i64 } @source()
  %11 = extractvalue { i64, i64, i64 } %10, 0
  %12 = extractvalue { i64, i64, i64 } %10, 1
  %13 = extractvalue { i64, i64, i64 } %10, 2
  call void @C_INTERFACE_PushBytes(ptr %7, i64 %11)
  call void @C_INTERFACE_PushBytes(ptr %8, i64 %12)
  call void @C_INTERFACE_PushBytes(ptr %9, i64 %13)
  ret void
}

define void @sum_wrap(ptr %0, ptr %1) {
  %3 = call ptr @RTChannelUntypedPtrsWrapperGet(ptr %0, i64 0)
  %4 = call ptr @RTChannelUntypedPtrsWrapperGet(ptr %0, i64 1)
  %5 = call ptr @RTChannelUntypedPtrsWrapperGet(ptr %1, i64 0)
  br label %6

6:                                                ; preds = %2
  %7 = phi ptr [ %3, %2 ]
  %8 = phi ptr [ %4, %2 ]
  %9 = phi ptr [ %5, %2 ]
  %10 = call i64 @C_INTERFACE_PopBytes(ptr %7)
  %11 = call i64 @C_INTERFACE_PopBytes(ptr %8)
  %12 = call i64 @sum(i64 %10, i64 %11)
  call void @C_INTERFACE_PushBytes(ptr %9, i64 %12)
  ret void
}

define void @mul_wrap(ptr %0, ptr %1) {
  %3 = call ptr @RTChannelUntypedPtrsWrapperGet(ptr %0, i64 0)
  %4 = call ptr @RTChannelUntypedPtrsWrapperGet(ptr %0, i64 1)
  %5 = call ptr @RTChannelUntypedPtrsWrapperGet(ptr %1, i64 0)
  br label %6

6:                                                ; preds = %2
  %7 = phi ptr [ %3, %2 ]
  %8 = phi ptr [ %4, %2 ]
  %9 = phi ptr [ %5, %2 ]
  %10 = call i64 @C_INTERFACE_PopBytes(ptr %7)
  %11 = call i64 @C_INTERFACE_PopBytes(ptr %8)
  %12 = call i64 @mul(i64 %10, i64 %11)
  call void @C_INTERFACE_PushBytes(ptr %9, i64 %12)
  ret void
}

define void @sink_wrap(ptr %0, ptr %1) {
  %3 = call ptr @RTChannelUntypedPtrsWrapperGet(ptr %0, i64 0)
  br label %4

4:                                                ; preds = %2
  %5 = phi ptr [ %3, %2 ]
  %6 = call i64 @C_INTERFACE_PopBytes(ptr %5)
  call void @sink(i64 %6)
  ret void
}

declare ptr @C_INTERFACE_AddChannel(i32)

declare void @ChannelUntypedPtrsWrapperSet(ptr, ptr, i64)

declare ptr @RTChannelUntypedPtrsWrapperGet(ptr, i64)

declare i64 @C_INTERFACE_PopBytes(ptr)

declare void @C_INTERFACE_PushBytes(ptr, i64)

declare ptr @C_INTERFACE_GetMainRegion(ptr)

declare ptr @C_INTERFACE_CreateApplication(ptr)

declare ptr @C_INTERFACE_ManagerGetInstance()

declare void @ChannelUntypedPtrsWrapperInit(ptr, i64)

declare ptr @C_INTERFACE_AddKpnProcess(ptr, ptr)

define void @main() {
  %1 = call ptr @C_INTERFACE_ManagerGetInstance()
  %2 = call ptr @C_INTERFACE_CreateApplication(ptr %1)
  %3 = call ptr @C_INTERFACE_GetMainRegion(ptr %2)
  %4 = call ptr @C_INTERFACE_AddChannel(i32 4)
  %5 = call ptr @C_INTERFACE_AddChannel(i32 4)
  %6 = call ptr @C_INTERFACE_AddChannel(i32 4)
  %7 = call ptr @C_INTERFACE_AddChannel(i32 4)
  %8 = call ptr @C_INTERFACE_AddChannel(i32 4)
  %9 = alloca %ChannelUntypedPtrsWrapper, align 8
  call void @ChannelUntypedPtrsWrapperInit(ptr %9, i64 0)
  %10 = alloca %ChannelUntypedPtrsWrapper, align 8
  call void @ChannelUntypedPtrsWrapperInit(ptr %10, i64 3)
  call void @ChannelUntypedPtrsWrapperSet(ptr %10, ptr %4, i64 0)
  call void @ChannelUntypedPtrsWrapperSet(ptr %10, ptr %5, i64 1)
  call void @ChannelUntypedPtrsWrapperSet(ptr %10, ptr %7, i64 2)
  %11 = call ptr @C_INTERFACE_AddKpnProcess(ptr %9, ptr %10)
  %12 = alloca %ChannelUntypedPtrsWrapper, align 8
  call void @ChannelUntypedPtrsWrapperInit(ptr %12, i64 2)
  call void @ChannelUntypedPtrsWrapperSet(ptr %12, ptr %4, i64 0)
  call void @ChannelUntypedPtrsWrapperSet(ptr %12, ptr %5, i64 1)
  %13 = alloca %ChannelUntypedPtrsWrapper, align 8
  call void @ChannelUntypedPtrsWrapperInit(ptr %13, i64 1)
  call void @ChannelUntypedPtrsWrapperSet(ptr %13, ptr %6, i64 0)
  %14 = call ptr @C_INTERFACE_AddKpnProcess(ptr %12, ptr %13)
  %15 = alloca %ChannelUntypedPtrsWrapper, align 8
  call void @ChannelUntypedPtrsWrapperInit(ptr %15, i64 2)
  call void @ChannelUntypedPtrsWrapperSet(ptr %15, ptr %6, i64 0)
  call void @ChannelUntypedPtrsWrapperSet(ptr %15, ptr %7, i64 1)
  %16 = alloca %ChannelUntypedPtrsWrapper, align 8
  call void @ChannelUntypedPtrsWrapperInit(ptr %16, i64 1)
  call void @ChannelUntypedPtrsWrapperSet(ptr %16, ptr %8, i64 0)
  %17 = call ptr @C_INTERFACE_AddKpnProcess(ptr %15, ptr %16)
  %18 = alloca %ChannelUntypedPtrsWrapper, align 8
  call void @ChannelUntypedPtrsWrapperInit(ptr %18, i64 1)
  call void @ChannelUntypedPtrsWrapperSet(ptr %18, ptr %8, i64 0)
  %19 = alloca %ChannelUntypedPtrsWrapper, align 8
  call void @ChannelUntypedPtrsWrapperInit(ptr %19, i64 0)
  %20 = call ptr @C_INTERFACE_AddKpnProcess(ptr %18, ptr %19)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
