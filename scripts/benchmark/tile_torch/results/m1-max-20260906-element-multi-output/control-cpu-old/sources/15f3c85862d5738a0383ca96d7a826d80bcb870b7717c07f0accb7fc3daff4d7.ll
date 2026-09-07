; ModuleID = 'TVMMod'
source_filename = "TVMMod"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-darwin25.6.0"

%closure_loop_parallel_parallel_0 = type { i32, ptr, ptr, ptr }

@__TVMBackendParallelLaunch = linkonce dllexport local_unnamed_addr global ptr null, align 8
@.str = private constant [10 x i8] c"TypeError\00", align 1
@.str.1 = private constant [10 x i8] c"Expected \00", align 1
@.str.2 = private constant [2 x i8] c"3\00", align 1
@.str.3 = private constant [11 x i8] c" arguments\00", align 1
@.str.4 = private constant [19 x i8] c" when calling:\0A  `\00", align 1
@.str.5 = private constant [185 x i8] c"benchmark_activation_pair(arg0: Tensor([T.int64(37), T.int64(1537)], float32), arg1: Tensor([T.int64(37), T.int64(1537)], float32), arg2: Tensor([T.int64(37), T.int64(1537)], float32))\00", align 1
@.str.6 = private constant [2 x i8] c"`\00", align 1
@.str.7 = private constant [21 x i8] c"args pointer is NULL\00", align 1
@.str.8 = private constant [30 x i8] c"Mismatched type on argument #\00", align 1
@.str.9 = private constant [2 x i8] c"0\00", align 1
@.str.10 = private constant [15 x i8] c"`,\0A  expected \00", align 1
@.str.11 = private constant [7 x i8] c"Tensor\00", align 1
@.str.12 = private constant [2 x i8] c"1\00", align 1
@.str.13 = private constant [2 x i8] c"2\00", align 1
@.str.14 = private constant [11 x i8] c"ValueError\00", align 1
@.str.15 = private constant [12 x i8] c"Mismatched \00", align 1
@.str.16 = private constant [5 x i8] c"arg0\00", align 1
@.str.17 = private constant [20 x i8] c".ndim on argument #\00", align 1
@.str.18 = private constant [21 x i8] c".dtype on argument #\00", align 1
@.str.19 = private constant [8 x i8] c"float32\00", align 1
@.str.20 = private constant [27 x i8] c".device_type on argument #\00", align 1
@.str.21 = private constant [4 x i8] c"cpu\00", align 1
@.str.22 = private constant [5 x i8] c"arg1\00", align 1
@.str.23 = private constant [5 x i8] c"arg2\00", align 1
@.str.24 = private constant [23 x i8] c".strides on argument #\00", align 1
@.str.25 = private constant [34 x i8] c"`,\0A  expected to be compact array\00", align 1
@.str.26 = private constant [36 x i8] c" data pointer is NULL on argument #\00", align 1
@.str.27 = private constant [36 x i8] c"`,\0A  expected non-NULL data pointer\00", align 1
@.str.28 = private constant [15 x i8] c"arg1.device_id\00", align 1
@.str.29 = private constant [15 x i8] c" on argument #\00", align 1
@.str.30 = private constant [24 x i8] c"`,\0A  expected to match \00", align 1
@.str.31 = private constant [15 x i8] c"arg0.device_id\00", align 1
@.str.32 = private constant [15 x i8] c"arg2.device_id\00", align 1
@.str.33 = private constant [9 x i8] c"Invalid \00", align 1
@.str.34 = private constant [14 x i8] c"arg0.shape[0]\00", align 1
@.str.35 = private constant [3 x i8] c"37\00", align 1
@.str.36 = private constant [14 x i8] c"arg0.shape[1]\00", align 1
@.str.37 = private constant [5 x i8] c"1537\00", align 1
@.str.38 = private constant [17 x i8] c"arg0.byte_offset\00", align 1
@.str.39 = private constant [14 x i8] c"arg1.shape[0]\00", align 1
@.str.40 = private constant [14 x i8] c"arg1.shape[1]\00", align 1
@.str.41 = private constant [17 x i8] c"arg1.byte_offset\00", align 1
@.str.42 = private constant [14 x i8] c"arg2.shape[0]\00", align 1
@.str.43 = private constant [14 x i8] c"arg2.shape[1]\00", align 1
@.str.44 = private constant [17 x i8] c"arg2.byte_offset\00", align 1
@__TVMBackendAllocWorkspace = linkonce dllexport local_unnamed_addr global ptr null, align 8
@__TVMBackendFreeWorkspace = linkonce dllexport local_unnamed_addr global ptr null, align 8
@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

define dllexport i32 @__tvm_ffi_benchmark_activation_pair(ptr noalias readnone captures(none) %self_handle, ptr noalias readonly captures(address_is_null) %args, i32 %num_args, ptr noalias readnone captures(none) %result) local_unnamed_addr #0 !dbg !5 {
entry:
    #dbg_value(ptr poison, !11, !DIExpression(), !15)
    #dbg_value(ptr %args, !12, !DIExpression(), !15)
    #dbg_value(i32 %num_args, !13, !DIExpression(), !15)
    #dbg_value(ptr poison, !14, !DIExpression(), !15)
  %0 = icmp eq i32 %num_args, 3, !dbg !15
  br i1 %0, label %assert_end, label %assert_fail, !dbg !15, !prof !16

common.ret:                                       ; preds = %assert_end78, %assert_fail77, %assert_fail75, %assert_fail73, %assert_fail71, %assert_fail69, %assert_fail67, %assert_fail65, %assert_fail63, %assert_fail61, %assert_fail59, %assert_fail57, %assert_fail55, %assert_fail51, %assert_fail49, %assert_fail47, %assert_fail43, %assert_fail41, %assert_fail37, %assert_fail35, %assert_fail33, %assert_fail31, %assert_fail29, %assert_fail27, %assert_fail25, %assert_fail23, %assert_fail21, %assert_fail19, %assert_fail17, %assert_fail15, %assert_fail10, %assert_fail5, %assert_fail3, %assert_fail1, %assert_fail
  %common.ret.op = phi i32 [ -1, %assert_fail ], [ -1, %assert_fail1 ], [ -1, %assert_fail3 ], [ -1, %assert_fail5 ], [ -1, %assert_fail10 ], [ -1, %assert_fail15 ], [ -1, %assert_fail17 ], [ -1, %assert_fail19 ], [ -1, %assert_fail21 ], [ -1, %assert_fail23 ], [ -1, %assert_fail25 ], [ -1, %assert_fail27 ], [ -1, %assert_fail29 ], [ -1, %assert_fail31 ], [ -1, %assert_fail33 ], [ -1, %assert_fail35 ], [ -1, %assert_fail37 ], [ -1, %assert_fail41 ], [ -1, %assert_fail43 ], [ -1, %assert_fail47 ], [ -1, %assert_fail49 ], [ -1, %assert_fail51 ], [ -1, %assert_fail55 ], [ -1, %assert_fail57 ], [ -1, %assert_fail59 ], [ -1, %assert_fail61 ], [ -1, %assert_fail63 ], [ -1, %assert_fail65 ], [ -1, %assert_fail67 ], [ -1, %assert_fail69 ], [ -1, %assert_fail71 ], [ -1, %assert_fail73 ], [ -1, %assert_fail75 ], [ -1, %assert_fail77 ], [ %133, %assert_end78 ]
  ret i32 %common.ret.op, !dbg !15

assert_fail:                                      ; preds = %entry
  tail call fastcc void @__tvm_set_raised_6(ptr nonnull @.str, i32 6, ptr nonnull @.str.1, ptr nonnull @.str.2, ptr nonnull @.str.3, ptr nonnull @.str.4, ptr nonnull @.str.5, ptr nonnull @.str.6), !dbg !15
  br label %common.ret, !dbg !15

assert_end:                                       ; preds = %entry
  %.not = icmp eq ptr %args, null, !dbg !15
  br i1 %.not, label %assert_fail1, label %assert_end2, !dbg !15, !prof !17

assert_fail1:                                     ; preds = %assert_end
  tail call fastcc void @__tvm_set_raised_6(ptr nonnull @.str, i32 4, ptr nonnull @.str.7, ptr nonnull @.str.4, ptr nonnull @.str.5, ptr nonnull @.str.6, ptr null, ptr null), !dbg !15
  br label %common.ret, !dbg !15

assert_end2:                                      ; preds = %assert_end
  %arg0.type_index = load i32, ptr %args, align 4, !dbg !15
    #dbg_value(i32 %arg0.type_index, !18, !DIExpression(), !15)
    #dbg_value(i32 %arg0.type_index, !18, !DIExpression(), !15)
  %arg0.type_index.fr = freeze i32 %arg0.type_index, !dbg !15
  %1 = icmp sgt i32 %arg0.type_index.fr, 63, !dbg !15
  br i1 %1, label %assert_end4, label %switch.early.test, !dbg !15

switch.early.test:                                ; preds = %assert_end2
  switch i32 %arg0.type_index.fr, label %assert_fail3 [
    i32 0, label %if_else
    i32 4, label %if_else
    i32 7, label %if_else
  ], !dbg !15

assert_fail3:                                     ; preds = %switch.early.test
  tail call fastcc void @__tvm_set_raised_6(ptr nonnull @.str, i32 6, ptr nonnull @.str.8, ptr nonnull @.str.9, ptr nonnull @.str.4, ptr nonnull @.str.5, ptr nonnull @.str.10, ptr nonnull @.str.11), !dbg !15
  br label %common.ret, !dbg !15

assert_end4:                                      ; preds = %assert_end2
  %2 = icmp eq i32 %arg0.type_index.fr, 70, !dbg !15
  br i1 %2, label %if_then, label %if_else, !dbg !15

if_then:                                          ; preds = %assert_end4
  %3 = getelementptr inbounds nuw i8, ptr %args, i64 8, !dbg !15
  %4 = load ptr, ptr %3, align 8, !dbg !15
  %5 = getelementptr inbounds nuw i8, ptr %4, i64 24, !dbg !15
  br label %if_end, !dbg !15

if_else:                                          ; preds = %switch.early.test, %switch.early.test, %switch.early.test, %assert_end4
  %6 = getelementptr inbounds nuw i8, ptr %args, i64 8, !dbg !15
  %7 = load ptr, ptr %6, align 8, !dbg !15
  br label %if_end, !dbg !15

if_end:                                           ; preds = %if_else, %if_then
  %arg0.handle = phi ptr [ %5, %if_then ], [ %7, %if_else ], !dbg !15
    #dbg_declare(ptr %arg0.handle, !19, !DIExpression(), !15)
    #dbg_declare(ptr %arg0.handle, !19, !DIExpression(), !15)
  %8 = getelementptr inbounds nuw i8, ptr %args, i64 16, !dbg !15
  %arg1.type_index = load i32, ptr %8, align 4, !dbg !15
    #dbg_value(i32 %arg1.type_index, !20, !DIExpression(), !15)
    #dbg_value(i32 %arg1.type_index, !20, !DIExpression(), !15)
  %arg1.type_index.fr = freeze i32 %arg1.type_index, !dbg !15
  %9 = icmp sgt i32 %arg1.type_index.fr, 63, !dbg !15
  br i1 %9, label %assert_end6, label %switch.early.test79, !dbg !15

switch.early.test79:                              ; preds = %if_end
  switch i32 %arg1.type_index.fr, label %assert_fail5 [
    i32 0, label %if_else8
    i32 4, label %if_else8
    i32 7, label %if_else8
  ], !dbg !15

assert_fail5:                                     ; preds = %switch.early.test79
  tail call fastcc void @__tvm_set_raised_6(ptr nonnull @.str, i32 6, ptr nonnull @.str.8, ptr nonnull @.str.12, ptr nonnull @.str.4, ptr nonnull @.str.5, ptr nonnull @.str.10, ptr nonnull @.str.11), !dbg !15
  br label %common.ret, !dbg !15

assert_end6:                                      ; preds = %if_end
  %10 = icmp eq i32 %arg1.type_index.fr, 70, !dbg !15
  br i1 %10, label %if_then7, label %if_else8, !dbg !15

if_then7:                                         ; preds = %assert_end6
  %11 = getelementptr inbounds nuw i8, ptr %args, i64 24, !dbg !15
  %12 = load ptr, ptr %11, align 8, !dbg !15
  %13 = getelementptr inbounds nuw i8, ptr %12, i64 24, !dbg !15
  br label %if_end9, !dbg !15

if_else8:                                         ; preds = %switch.early.test79, %switch.early.test79, %switch.early.test79, %assert_end6
  %14 = getelementptr inbounds nuw i8, ptr %args, i64 24, !dbg !15
  %15 = load ptr, ptr %14, align 8, !dbg !15
  br label %if_end9, !dbg !15

if_end9:                                          ; preds = %if_else8, %if_then7
  %arg1.handle = phi ptr [ %13, %if_then7 ], [ %15, %if_else8 ], !dbg !15
    #dbg_declare(ptr %arg1.handle, !21, !DIExpression(), !15)
    #dbg_declare(ptr %arg1.handle, !21, !DIExpression(), !15)
  %16 = getelementptr inbounds nuw i8, ptr %args, i64 32, !dbg !15
  %arg2.type_index = load i32, ptr %16, align 4, !dbg !15
    #dbg_value(i32 %arg2.type_index, !22, !DIExpression(), !15)
    #dbg_value(i32 %arg2.type_index, !22, !DIExpression(), !15)
  %arg2.type_index.fr = freeze i32 %arg2.type_index, !dbg !15
  %17 = icmp sgt i32 %arg2.type_index.fr, 63, !dbg !15
  br i1 %17, label %assert_end11, label %switch.early.test80, !dbg !15

switch.early.test80:                              ; preds = %if_end9
  switch i32 %arg2.type_index.fr, label %assert_fail10 [
    i32 0, label %if_else13
    i32 4, label %if_else13
    i32 7, label %if_else13
  ], !dbg !15

assert_fail10:                                    ; preds = %switch.early.test80
  tail call fastcc void @__tvm_set_raised_6(ptr nonnull @.str, i32 6, ptr nonnull @.str.8, ptr nonnull @.str.13, ptr nonnull @.str.4, ptr nonnull @.str.5, ptr nonnull @.str.10, ptr nonnull @.str.11), !dbg !15
  br label %common.ret, !dbg !15

assert_end11:                                     ; preds = %if_end9
  %18 = icmp eq i32 %arg2.type_index.fr, 70, !dbg !15
  br i1 %18, label %if_then12, label %if_else13, !dbg !15

if_then12:                                        ; preds = %assert_end11
  %19 = getelementptr inbounds nuw i8, ptr %args, i64 40, !dbg !15
  %20 = load ptr, ptr %19, align 8, !dbg !15
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 24, !dbg !15
  br label %if_end14, !dbg !15

if_else13:                                        ; preds = %switch.early.test80, %switch.early.test80, %switch.early.test80, %assert_end11
  %22 = getelementptr inbounds nuw i8, ptr %args, i64 40, !dbg !15
  %23 = load ptr, ptr %22, align 8, !dbg !15
  br label %if_end14, !dbg !15

if_end14:                                         ; preds = %if_else13, %if_then12
  %arg2.handle = phi ptr [ %21, %if_then12 ], [ %23, %if_else13 ], !dbg !15
    #dbg_declare(ptr %arg2.handle, !23, !DIExpression(), !15)
    #dbg_declare(ptr %arg2.handle, !23, !DIExpression(), !15)
  %.not81 = icmp eq ptr %arg0.handle, null, !dbg !15
  br i1 %.not81, label %assert_fail15, label %assert_end16, !dbg !15, !prof !17

assert_fail15:                                    ; preds = %if_end14
  tail call fastcc void @__tvm_set_raised_6(ptr nonnull @.str, i32 6, ptr nonnull @.str.8, ptr nonnull @.str.9, ptr nonnull @.str.4, ptr nonnull @.str.5, ptr nonnull @.str.10, ptr nonnull @.str.11), !dbg !15
  br label %common.ret, !dbg !15

assert_end16:                                     ; preds = %if_end14
  %24 = getelementptr inbounds nuw i8, ptr %arg0.handle, i64 16, !dbg !15
  %25 = load i32, ptr %24, align 4, !dbg !15
  %26 = icmp eq i32 %25, 2, !dbg !15
  br i1 %26, label %assert_end18, label %assert_fail17, !dbg !15, !prof !16

assert_fail17:                                    ; preds = %assert_end16
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.14, i32 8, ptr nonnull @.str.15, ptr nonnull @.str.16, ptr nonnull @.str.17, ptr nonnull @.str.9, ptr nonnull @.str.10, ptr nonnull @.str.13), !dbg !15
  br label %common.ret, !dbg !15

assert_end18:                                     ; preds = %assert_end16
  %27 = getelementptr inbounds nuw i8, ptr %arg0.handle, i64 20, !dbg !15
  %28 = load i8, ptr %27, align 1, !dbg !15
  %29 = icmp eq i8 %28, 2, !dbg !15
  %30 = getelementptr inbounds nuw i8, ptr %arg0.handle, i64 21, !dbg !15
  %31 = load i8, ptr %30, align 1, !dbg !15
  %32 = icmp eq i8 %31, 32, !dbg !15
  %33 = and i1 %29, %32, !dbg !15
  %34 = getelementptr inbounds nuw i8, ptr %arg0.handle, i64 22, !dbg !15
  %35 = load i16, ptr %34, align 2, !dbg !15
  %36 = icmp eq i16 %35, 1, !dbg !15
  %37 = and i1 %33, %36, !dbg !15
  br i1 %37, label %assert_end20, label %assert_fail19, !dbg !15, !prof !16

assert_fail19:                                    ; preds = %assert_end18
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str, i32 8, ptr nonnull @.str.15, ptr nonnull @.str.16, ptr nonnull @.str.18, ptr nonnull @.str.9, ptr nonnull @.str.10, ptr nonnull @.str.19), !dbg !15
  br label %common.ret, !dbg !15

assert_end20:                                     ; preds = %assert_end18
  %38 = getelementptr inbounds nuw i8, ptr %arg0.handle, i64 24, !dbg !15
  %benchmark_activation_pair.arg0_shape = load ptr, ptr %38, align 8, !dbg !15
    #dbg_declare(ptr %benchmark_activation_pair.arg0_shape, !24, !DIExpression(), !15)
    #dbg_declare(ptr %benchmark_activation_pair.arg0_shape, !24, !DIExpression(), !15)
  %39 = getelementptr inbounds nuw i8, ptr %arg0.handle, i64 32, !dbg !15
  %benchmark_activation_pair.arg0_strides = load ptr, ptr %39, align 8, !dbg !15
    #dbg_declare(ptr %benchmark_activation_pair.arg0_strides, !27, !DIExpression(), !15)
    #dbg_declare(ptr %benchmark_activation_pair.arg0_strides, !27, !DIExpression(), !15)
  %40 = getelementptr inbounds nuw i8, ptr %arg0.handle, i64 8, !dbg !15
  %41 = load i32, ptr %40, align 4, !dbg !15
  %42 = icmp eq i32 %41, 1, !dbg !15
  br i1 %42, label %assert_end22, label %assert_fail21, !dbg !15, !prof !16

assert_fail21:                                    ; preds = %assert_end20
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.14, i32 8, ptr nonnull @.str.15, ptr nonnull @.str.16, ptr nonnull @.str.20, ptr nonnull @.str.9, ptr nonnull @.str.10, ptr nonnull @.str.21), !dbg !15
  br label %common.ret, !dbg !15

assert_end22:                                     ; preds = %assert_end20
  %43 = getelementptr inbounds nuw i8, ptr %arg0.handle, i64 12, !dbg !15
  %dev_id = load i32, ptr %43, align 4, !dbg !15
    #dbg_value(i32 %dev_id, !28, !DIExpression(), !15)
    #dbg_value(i32 %dev_id, !28, !DIExpression(), !15)
  %.not82 = icmp eq ptr %arg1.handle, null, !dbg !15
  br i1 %.not82, label %assert_fail23, label %assert_end24, !dbg !15, !prof !17

assert_fail23:                                    ; preds = %assert_end22
  tail call fastcc void @__tvm_set_raised_6(ptr nonnull @.str, i32 6, ptr nonnull @.str.8, ptr nonnull @.str.12, ptr nonnull @.str.4, ptr nonnull @.str.5, ptr nonnull @.str.10, ptr nonnull @.str.11), !dbg !15
  br label %common.ret, !dbg !15

assert_end24:                                     ; preds = %assert_end22
  %44 = getelementptr inbounds nuw i8, ptr %arg1.handle, i64 16, !dbg !15
  %45 = load i32, ptr %44, align 4, !dbg !15
  %46 = icmp eq i32 %45, 2, !dbg !15
  br i1 %46, label %assert_end26, label %assert_fail25, !dbg !15, !prof !16

assert_fail25:                                    ; preds = %assert_end24
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.14, i32 8, ptr nonnull @.str.15, ptr nonnull @.str.22, ptr nonnull @.str.17, ptr nonnull @.str.12, ptr nonnull @.str.10, ptr nonnull @.str.13), !dbg !15
  br label %common.ret, !dbg !15

assert_end26:                                     ; preds = %assert_end24
  %47 = getelementptr inbounds nuw i8, ptr %arg1.handle, i64 20, !dbg !15
  %48 = load i8, ptr %47, align 1, !dbg !15
  %49 = icmp eq i8 %48, 2, !dbg !15
  %50 = getelementptr inbounds nuw i8, ptr %arg1.handle, i64 21, !dbg !15
  %51 = load i8, ptr %50, align 1, !dbg !15
  %52 = icmp eq i8 %51, 32, !dbg !15
  %53 = and i1 %49, %52, !dbg !15
  %54 = getelementptr inbounds nuw i8, ptr %arg1.handle, i64 22, !dbg !15
  %55 = load i16, ptr %54, align 2, !dbg !15
  %56 = icmp eq i16 %55, 1, !dbg !15
  %57 = and i1 %53, %56, !dbg !15
  br i1 %57, label %assert_end28, label %assert_fail27, !dbg !15, !prof !16

assert_fail27:                                    ; preds = %assert_end26
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str, i32 8, ptr nonnull @.str.15, ptr nonnull @.str.22, ptr nonnull @.str.18, ptr nonnull @.str.12, ptr nonnull @.str.10, ptr nonnull @.str.19), !dbg !15
  br label %common.ret, !dbg !15

assert_end28:                                     ; preds = %assert_end26
  %58 = getelementptr inbounds nuw i8, ptr %arg1.handle, i64 24, !dbg !15
  %benchmark_activation_pair.arg1_shape = load ptr, ptr %58, align 8, !dbg !15
    #dbg_declare(ptr %benchmark_activation_pair.arg1_shape, !29, !DIExpression(), !15)
    #dbg_declare(ptr %benchmark_activation_pair.arg1_shape, !29, !DIExpression(), !15)
  %59 = getelementptr inbounds nuw i8, ptr %arg1.handle, i64 32, !dbg !15
  %benchmark_activation_pair.arg1_strides = load ptr, ptr %59, align 8, !dbg !15
    #dbg_declare(ptr %benchmark_activation_pair.arg1_strides, !30, !DIExpression(), !15)
    #dbg_declare(ptr %benchmark_activation_pair.arg1_strides, !30, !DIExpression(), !15)
  %60 = getelementptr inbounds nuw i8, ptr %arg1.handle, i64 8, !dbg !15
  %61 = load i32, ptr %60, align 4, !dbg !15
  %62 = icmp eq i32 %61, 1, !dbg !15
  br i1 %62, label %assert_end30, label %assert_fail29, !dbg !15, !prof !16

assert_fail29:                                    ; preds = %assert_end28
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.14, i32 8, ptr nonnull @.str.15, ptr nonnull @.str.22, ptr nonnull @.str.20, ptr nonnull @.str.12, ptr nonnull @.str.10, ptr nonnull @.str.21), !dbg !15
  br label %common.ret, !dbg !15

assert_end30:                                     ; preds = %assert_end28
  %.not83 = icmp eq ptr %arg2.handle, null, !dbg !15
  br i1 %.not83, label %assert_fail31, label %assert_end32, !dbg !15, !prof !17

assert_fail31:                                    ; preds = %assert_end30
  tail call fastcc void @__tvm_set_raised_6(ptr nonnull @.str, i32 6, ptr nonnull @.str.8, ptr nonnull @.str.13, ptr nonnull @.str.4, ptr nonnull @.str.5, ptr nonnull @.str.10, ptr nonnull @.str.11), !dbg !15
  br label %common.ret, !dbg !15

assert_end32:                                     ; preds = %assert_end30
  %63 = getelementptr inbounds nuw i8, ptr %arg2.handle, i64 16, !dbg !15
  %64 = load i32, ptr %63, align 4, !dbg !15
  %65 = icmp eq i32 %64, 2, !dbg !15
  br i1 %65, label %assert_end34, label %assert_fail33, !dbg !15, !prof !16

assert_fail33:                                    ; preds = %assert_end32
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.14, i32 8, ptr nonnull @.str.15, ptr nonnull @.str.23, ptr nonnull @.str.17, ptr nonnull @.str.13, ptr nonnull @.str.10, ptr nonnull @.str.13), !dbg !15
  br label %common.ret, !dbg !15

assert_end34:                                     ; preds = %assert_end32
  %66 = getelementptr inbounds nuw i8, ptr %arg2.handle, i64 20, !dbg !15
  %67 = load i8, ptr %66, align 1, !dbg !15
  %68 = icmp eq i8 %67, 2, !dbg !15
  %69 = getelementptr inbounds nuw i8, ptr %arg2.handle, i64 21, !dbg !15
  %70 = load i8, ptr %69, align 1, !dbg !15
  %71 = icmp eq i8 %70, 32, !dbg !15
  %72 = and i1 %68, %71, !dbg !15
  %73 = getelementptr inbounds nuw i8, ptr %arg2.handle, i64 22, !dbg !15
  %74 = load i16, ptr %73, align 2, !dbg !15
  %75 = icmp eq i16 %74, 1, !dbg !15
  %76 = and i1 %72, %75, !dbg !15
  br i1 %76, label %assert_end36, label %assert_fail35, !dbg !15, !prof !16

assert_fail35:                                    ; preds = %assert_end34
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str, i32 8, ptr nonnull @.str.15, ptr nonnull @.str.23, ptr nonnull @.str.18, ptr nonnull @.str.13, ptr nonnull @.str.10, ptr nonnull @.str.19), !dbg !15
  br label %common.ret, !dbg !15

assert_end36:                                     ; preds = %assert_end34
  %77 = getelementptr inbounds nuw i8, ptr %arg2.handle, i64 24, !dbg !15
  %benchmark_activation_pair.arg2_shape = load ptr, ptr %77, align 8, !dbg !15
    #dbg_declare(ptr %benchmark_activation_pair.arg2_shape, !31, !DIExpression(), !15)
    #dbg_declare(ptr %benchmark_activation_pair.arg2_shape, !31, !DIExpression(), !15)
  %78 = getelementptr inbounds nuw i8, ptr %arg2.handle, i64 32, !dbg !15
  %benchmark_activation_pair.arg2_strides = load ptr, ptr %78, align 8, !dbg !15
    #dbg_declare(ptr %benchmark_activation_pair.arg2_strides, !32, !DIExpression(), !15)
    #dbg_declare(ptr %benchmark_activation_pair.arg2_strides, !32, !DIExpression(), !15)
  %79 = getelementptr inbounds nuw i8, ptr %arg2.handle, i64 8, !dbg !15
  %80 = load i32, ptr %79, align 4, !dbg !15
  %81 = icmp eq i32 %80, 1, !dbg !15
  br i1 %81, label %assert_end38, label %assert_fail37, !dbg !15, !prof !16

assert_fail37:                                    ; preds = %assert_end36
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.14, i32 8, ptr nonnull @.str.15, ptr nonnull @.str.23, ptr nonnull @.str.20, ptr nonnull @.str.13, ptr nonnull @.str.10, ptr nonnull @.str.21), !dbg !15
  br label %common.ret, !dbg !15

assert_end38:                                     ; preds = %assert_end36
  %.not84 = icmp eq ptr %benchmark_activation_pair.arg0_strides, null, !dbg !15
  br i1 %.not84, label %if_end40, label %if_then39, !dbg !15, !prof !17

if_then39:                                        ; preds = %assert_end38
  %82 = getelementptr inbounds nuw i8, ptr %benchmark_activation_pair.arg0_strides, i64 8, !dbg !15
  %83 = load i64, ptr %82, align 8, !dbg !15
  %84 = icmp eq i64 %83, 1, !dbg !15
  %85 = load i64, ptr %benchmark_activation_pair.arg0_strides, align 8, !dbg !15
  %86 = icmp eq i64 %85, 1537, !dbg !15
  %87 = and i1 %84, %86, !dbg !15
  br i1 %87, label %if_end40, label %assert_fail41, !dbg !15, !prof !16

if_end40:                                         ; preds = %if_then39, %assert_end38
  %88 = load ptr, ptr %arg0.handle, align 8, !dbg !15
  %.not85 = icmp eq ptr %88, null, !dbg !15
  br i1 %.not85, label %assert_fail43, label %assert_end44, !dbg !15, !prof !17

assert_fail41:                                    ; preds = %if_then39
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.14, i32 7, ptr nonnull @.str.15, ptr nonnull @.str.16, ptr nonnull @.str.24, ptr nonnull @.str.9, ptr nonnull @.str.25, ptr null), !dbg !15
  br label %common.ret, !dbg !15

assert_fail43:                                    ; preds = %if_end40
  tail call fastcc void @__tvm_set_raised_6(ptr nonnull @.str.14, i32 6, ptr nonnull @.str.16, ptr nonnull @.str.26, ptr nonnull @.str.9, ptr nonnull @.str.4, ptr nonnull @.str.5, ptr nonnull @.str.27), !dbg !15
  br label %common.ret, !dbg !15

assert_end44:                                     ; preds = %if_end40
  %.not86 = icmp eq ptr %benchmark_activation_pair.arg1_strides, null, !dbg !15
  br i1 %.not86, label %if_end46, label %if_then45, !dbg !15, !prof !17

if_then45:                                        ; preds = %assert_end44
  %89 = getelementptr inbounds nuw i8, ptr %benchmark_activation_pair.arg1_strides, i64 8, !dbg !15
  %90 = load i64, ptr %89, align 8, !dbg !15
  %91 = icmp eq i64 %90, 1, !dbg !15
  %92 = load i64, ptr %benchmark_activation_pair.arg1_strides, align 8, !dbg !15
  %93 = icmp eq i64 %92, 1537, !dbg !15
  %94 = and i1 %91, %93, !dbg !15
  br i1 %94, label %if_end46, label %assert_fail47, !dbg !15, !prof !16

if_end46:                                         ; preds = %if_then45, %assert_end44
  %95 = getelementptr inbounds nuw i8, ptr %arg1.handle, i64 12, !dbg !15
  %96 = load i32, ptr %95, align 4, !dbg !15
  %97 = icmp eq i32 %dev_id, %96, !dbg !15
  br i1 %97, label %assert_end50, label %assert_fail49, !dbg !15, !prof !16

assert_fail47:                                    ; preds = %if_then45
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.14, i32 7, ptr nonnull @.str.15, ptr nonnull @.str.22, ptr nonnull @.str.24, ptr nonnull @.str.12, ptr nonnull @.str.25, ptr null), !dbg !15
  br label %common.ret, !dbg !15

assert_fail49:                                    ; preds = %if_end46
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.14, i32 8, ptr nonnull @.str.15, ptr nonnull @.str.28, ptr nonnull @.str.29, ptr nonnull @.str.12, ptr nonnull @.str.30, ptr nonnull @.str.31), !dbg !15
  br label %common.ret, !dbg !15

assert_end50:                                     ; preds = %if_end46
  %98 = load ptr, ptr %arg1.handle, align 8, !dbg !15
  %.not87 = icmp eq ptr %98, null, !dbg !15
  br i1 %.not87, label %assert_fail51, label %assert_end52, !dbg !15, !prof !17

assert_fail51:                                    ; preds = %assert_end50
  tail call fastcc void @__tvm_set_raised_6(ptr nonnull @.str.14, i32 6, ptr nonnull @.str.22, ptr nonnull @.str.26, ptr nonnull @.str.12, ptr nonnull @.str.4, ptr nonnull @.str.5, ptr nonnull @.str.27), !dbg !15
  br label %common.ret, !dbg !15

assert_end52:                                     ; preds = %assert_end50
  %.not88 = icmp eq ptr %benchmark_activation_pair.arg2_strides, null, !dbg !15
  br i1 %.not88, label %if_end54, label %if_then53, !dbg !15, !prof !17

if_then53:                                        ; preds = %assert_end52
  %99 = getelementptr inbounds nuw i8, ptr %benchmark_activation_pair.arg2_strides, i64 8, !dbg !15
  %100 = load i64, ptr %99, align 8, !dbg !15
  %101 = icmp eq i64 %100, 1, !dbg !15
  %102 = load i64, ptr %benchmark_activation_pair.arg2_strides, align 8, !dbg !15
  %103 = icmp eq i64 %102, 1537, !dbg !15
  %104 = and i1 %101, %103, !dbg !15
  br i1 %104, label %if_end54, label %assert_fail55, !dbg !15, !prof !16

if_end54:                                         ; preds = %if_then53, %assert_end52
  %105 = getelementptr inbounds nuw i8, ptr %arg2.handle, i64 12, !dbg !15
  %106 = load i32, ptr %105, align 4, !dbg !15
  %107 = icmp eq i32 %dev_id, %106, !dbg !15
  br i1 %107, label %assert_end58, label %assert_fail57, !dbg !15, !prof !16

assert_fail55:                                    ; preds = %if_then53
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.14, i32 7, ptr nonnull @.str.15, ptr nonnull @.str.23, ptr nonnull @.str.24, ptr nonnull @.str.13, ptr nonnull @.str.25, ptr null), !dbg !15
  br label %common.ret, !dbg !15

assert_fail57:                                    ; preds = %if_end54
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.14, i32 8, ptr nonnull @.str.15, ptr nonnull @.str.32, ptr nonnull @.str.29, ptr nonnull @.str.13, ptr nonnull @.str.30, ptr nonnull @.str.31), !dbg !15
  br label %common.ret, !dbg !15

assert_end58:                                     ; preds = %if_end54
  %108 = load ptr, ptr %arg2.handle, align 8, !dbg !15
  %.not89 = icmp eq ptr %108, null, !dbg !15
  br i1 %.not89, label %assert_fail59, label %assert_end60, !dbg !15, !prof !17

assert_fail59:                                    ; preds = %assert_end58
  tail call fastcc void @__tvm_set_raised_6(ptr nonnull @.str.14, i32 6, ptr nonnull @.str.23, ptr nonnull @.str.26, ptr nonnull @.str.13, ptr nonnull @.str.4, ptr nonnull @.str.5, ptr nonnull @.str.27), !dbg !15
  br label %common.ret, !dbg !15

assert_end60:                                     ; preds = %assert_end58
  %109 = load i64, ptr %benchmark_activation_pair.arg0_shape, align 8, !dbg !15
  %110 = icmp eq i64 %109, 37, !dbg !15
  br i1 %110, label %assert_end62, label %assert_fail61, !dbg !15, !prof !16

assert_fail61:                                    ; preds = %assert_end60
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.14, i32 8, ptr nonnull @.str.33, ptr nonnull @.str.34, ptr nonnull @.str.29, ptr nonnull @.str.9, ptr nonnull @.str.10, ptr nonnull @.str.35), !dbg !15
  br label %common.ret, !dbg !15

assert_end62:                                     ; preds = %assert_end60
  %111 = getelementptr inbounds nuw i8, ptr %benchmark_activation_pair.arg0_shape, i64 8, !dbg !15
  %112 = load i64, ptr %111, align 8, !dbg !15
  %113 = icmp eq i64 %112, 1537, !dbg !15
  br i1 %113, label %assert_end64, label %assert_fail63, !dbg !15, !prof !16

assert_fail63:                                    ; preds = %assert_end62
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.14, i32 8, ptr nonnull @.str.33, ptr nonnull @.str.36, ptr nonnull @.str.29, ptr nonnull @.str.9, ptr nonnull @.str.10, ptr nonnull @.str.37), !dbg !15
  br label %common.ret, !dbg !15

assert_end64:                                     ; preds = %assert_end62
  %114 = getelementptr inbounds nuw i8, ptr %arg0.handle, i64 40, !dbg !15
  %115 = load i64, ptr %114, align 8, !dbg !15
  %116 = icmp eq i64 %115, 0, !dbg !15
  br i1 %116, label %assert_end66, label %assert_fail65, !dbg !15, !prof !16

assert_fail65:                                    ; preds = %assert_end64
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.14, i32 8, ptr nonnull @.str.33, ptr nonnull @.str.38, ptr nonnull @.str.29, ptr nonnull @.str.9, ptr nonnull @.str.10, ptr nonnull @.str.9), !dbg !15
  br label %common.ret, !dbg !15

assert_end66:                                     ; preds = %assert_end64
  %117 = load i64, ptr %benchmark_activation_pair.arg1_shape, align 8, !dbg !15
  %118 = icmp eq i64 %117, 37, !dbg !15
  br i1 %118, label %assert_end68, label %assert_fail67, !dbg !15, !prof !16

assert_fail67:                                    ; preds = %assert_end66
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.14, i32 8, ptr nonnull @.str.33, ptr nonnull @.str.39, ptr nonnull @.str.29, ptr nonnull @.str.12, ptr nonnull @.str.10, ptr nonnull @.str.35), !dbg !15
  br label %common.ret, !dbg !15

assert_end68:                                     ; preds = %assert_end66
  %119 = getelementptr inbounds nuw i8, ptr %benchmark_activation_pair.arg1_shape, i64 8, !dbg !15
  %120 = load i64, ptr %119, align 8, !dbg !15
  %121 = icmp eq i64 %120, 1537, !dbg !15
  br i1 %121, label %assert_end70, label %assert_fail69, !dbg !15, !prof !16

assert_fail69:                                    ; preds = %assert_end68
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.14, i32 8, ptr nonnull @.str.33, ptr nonnull @.str.40, ptr nonnull @.str.29, ptr nonnull @.str.12, ptr nonnull @.str.10, ptr nonnull @.str.37), !dbg !15
  br label %common.ret, !dbg !15

assert_end70:                                     ; preds = %assert_end68
  %122 = getelementptr inbounds nuw i8, ptr %arg1.handle, i64 40, !dbg !15
  %123 = load i64, ptr %122, align 8, !dbg !15
  %124 = icmp eq i64 %123, 0, !dbg !15
  br i1 %124, label %assert_end72, label %assert_fail71, !dbg !15, !prof !16

assert_fail71:                                    ; preds = %assert_end70
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.14, i32 8, ptr nonnull @.str.33, ptr nonnull @.str.41, ptr nonnull @.str.29, ptr nonnull @.str.12, ptr nonnull @.str.10, ptr nonnull @.str.9), !dbg !15
  br label %common.ret, !dbg !15

assert_end72:                                     ; preds = %assert_end70
  %125 = load i64, ptr %benchmark_activation_pair.arg2_shape, align 8, !dbg !15
  %126 = icmp eq i64 %125, 37, !dbg !15
  br i1 %126, label %assert_end74, label %assert_fail73, !dbg !15, !prof !16

assert_fail73:                                    ; preds = %assert_end72
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.14, i32 8, ptr nonnull @.str.33, ptr nonnull @.str.42, ptr nonnull @.str.29, ptr nonnull @.str.13, ptr nonnull @.str.10, ptr nonnull @.str.35), !dbg !15
  br label %common.ret, !dbg !15

assert_end74:                                     ; preds = %assert_end72
  %127 = getelementptr inbounds nuw i8, ptr %benchmark_activation_pair.arg2_shape, i64 8, !dbg !15
  %128 = load i64, ptr %127, align 8, !dbg !15
  %129 = icmp eq i64 %128, 1537, !dbg !15
  br i1 %129, label %assert_end76, label %assert_fail75, !dbg !15, !prof !16

assert_fail75:                                    ; preds = %assert_end74
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.14, i32 8, ptr nonnull @.str.33, ptr nonnull @.str.43, ptr nonnull @.str.29, ptr nonnull @.str.13, ptr nonnull @.str.10, ptr nonnull @.str.37), !dbg !15
  br label %common.ret, !dbg !15

assert_end76:                                     ; preds = %assert_end74
  %130 = getelementptr inbounds nuw i8, ptr %arg2.handle, i64 40, !dbg !15
  %131 = load i64, ptr %130, align 8, !dbg !15
  %132 = icmp eq i64 %131, 0, !dbg !15
  br i1 %132, label %assert_end78, label %assert_fail77, !dbg !15, !prof !16

assert_fail77:                                    ; preds = %assert_end76
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.14, i32 8, ptr nonnull @.str.33, ptr nonnull @.str.44, ptr nonnull @.str.29, ptr nonnull @.str.13, ptr nonnull @.str.10, ptr nonnull @.str.9), !dbg !15
  br label %common.ret, !dbg !15

assert_end78:                                     ; preds = %assert_end76
    #dbg_declare(ptr %88, !33, !DIExpression(), !15)
  call void @llvm.assume(i1 true) [ "align"(ptr %88, i64 64) ], !dbg !15
    #dbg_declare(ptr %98, !36, !DIExpression(), !15)
  call void @llvm.assume(i1 true) [ "align"(ptr %98, i64 64) ], !dbg !15
    #dbg_declare(ptr %108, !37, !DIExpression(), !15)
  call void @llvm.assume(i1 true) [ "align"(ptr %108, i64 64) ], !dbg !15
  %133 = tail call fastcc i32 @benchmark_activation_pair_compute_(ptr nonnull %88, ptr nonnull %98, ptr nonnull %108, i32 %dev_id), !dbg !15
  br label %common.ret, !dbg !15
}

; Function Attrs: noinline
define internal fastcc void @__tvm_set_raised_6(ptr %0, i32 range(i32 4, 7) %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7) unnamed_addr #1 {
entry:
  %8 = alloca [6 x ptr], align 8
  store ptr %2, ptr %8, align 8
  %9 = getelementptr inbounds nuw i8, ptr %8, i64 8
  store ptr %3, ptr %9, align 8
  %10 = getelementptr inbounds nuw i8, ptr %8, i64 16
  store ptr %4, ptr %10, align 8
  %11 = getelementptr inbounds nuw i8, ptr %8, i64 24
  store ptr %5, ptr %11, align 8
  %12 = getelementptr inbounds nuw i8, ptr %8, i64 32
  store ptr %6, ptr %12, align 8
  %13 = getelementptr inbounds nuw i8, ptr %8, i64 40
  store ptr %7, ptr %13, align 8
  call void @TVMFFIErrorSetRaisedFromCStrParts(ptr %0, ptr nonnull %8, i32 %1)
  ret void
}

declare void @TVMFFIErrorSetRaisedFromCStrParts(ptr, ptr, i32) local_unnamed_addr

; Function Attrs: noinline
define internal fastcc void @__tvm_set_raised_12(ptr %0, i32 range(i32 7, 9) %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7) unnamed_addr #1 {
entry:
  %8 = alloca [12 x ptr], align 8
  store ptr %2, ptr %8, align 8
  %9 = getelementptr inbounds nuw i8, ptr %8, i64 8
  store ptr %3, ptr %9, align 8
  %10 = getelementptr inbounds nuw i8, ptr %8, i64 16
  store ptr %4, ptr %10, align 8
  %11 = getelementptr inbounds nuw i8, ptr %8, i64 24
  store ptr %5, ptr %11, align 8
  %12 = getelementptr inbounds nuw i8, ptr %8, i64 32
  store ptr @.str.4, ptr %12, align 8
  %13 = getelementptr inbounds nuw i8, ptr %8, i64 40
  store ptr @.str.5, ptr %13, align 8
  %14 = getelementptr inbounds nuw i8, ptr %8, i64 48
  store ptr %6, ptr %14, align 8
  %15 = getelementptr inbounds nuw i8, ptr %8, i64 56
  store ptr %7, ptr %15, align 8
  %16 = getelementptr inbounds nuw i8, ptr %8, i64 64
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %16, i8 0, i64 32, i1 false)
  call void @TVMFFIErrorSetRaisedFromCStrParts(ptr %0, ptr nonnull %8, i32 %1)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #2

; Function Attrs: noinline
define internal fastcc i32 @benchmark_activation_pair_compute_(ptr noalias align 64 %arg0, ptr noalias align 64 %arg1, ptr noalias align 64 %arg2, i32 %dev_id) unnamed_addr #3 !dbg !38 {
entry:
    #dbg_value(ptr %arg0, !42, !DIExpression(), !46)
    #dbg_value(ptr %arg1, !43, !DIExpression(), !46)
    #dbg_value(ptr %arg2, !44, !DIExpression(), !46)
    #dbg_value(i32 %dev_id, !45, !DIExpression(), !46)
  %0 = alloca %closure_loop_parallel_parallel_0, align 8, !dbg !46
    #dbg_declare(ptr %arg0, !47, !DIExpression(), !46)
  call void @llvm.assume(i1 true) [ "align"(ptr %arg0, i64 64) ], !dbg !46
    #dbg_declare(ptr %arg1, !48, !DIExpression(), !46)
  call void @llvm.assume(i1 true) [ "align"(ptr %arg1, i64 64) ], !dbg !46
    #dbg_declare(ptr %arg2, !49, !DIExpression(), !46)
  call void @llvm.assume(i1 true) [ "align"(ptr %arg2, i64 64) ], !dbg !46
  store i32 %dev_id, ptr %0, align 8, !dbg !46
  %1 = getelementptr inbounds nuw i8, ptr %0, i64 8, !dbg !46
  store ptr %arg0, ptr %1, align 8, !dbg !46
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 16, !dbg !46
  store ptr %arg1, ptr %2, align 8, !dbg !46
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 24, !dbg !46
  store ptr %arg2, ptr %3, align 8, !dbg !46
  %4 = load ptr, ptr @__TVMBackendParallelLaunch, align 8, !dbg !46, !tbaa !50
  %5 = call i32 %4(ptr nonnull @__tvm_parallel_lambda, ptr nonnull %0, i32 0), !dbg !46
  ret i32 %5, !dbg !46
}

define private range(i32 -1, 1) i32 @__tvm_parallel_lambda(i32 %task_id, ptr readonly captures(none) %0, ptr readonly captures(none) %1) #0 {
parallel_closure_entry:
  %dev_id = load i32, ptr %1, align 4, !dbg !46
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 8, !dbg !46
  %arg0 = load ptr, ptr %2, align 8, !dbg !46
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 16, !dbg !46
  %arg1 = load ptr, ptr %3, align 8, !dbg !46
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 24, !dbg !46
  %arg2 = load ptr, ptr %4, align 8, !dbg !46
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 8, !dbg !46
  %num_task = load i32, ptr %5, align 4, !dbg !46
  %6 = add nsw i32 %num_task, 258, !dbg !46
  %7 = sdiv i32 %6, %num_task, !dbg !46
  %8 = mul nsw i32 %7, %task_id, !dbg !46
  %9 = add nsw i32 %task_id, 1, !dbg !46
  %10 = mul nsw i32 %7, %9, !dbg !46
  %11 = tail call i32 @llvm.smin.i32(i32 %10, i32 259), !dbg !46
    #dbg_value(i32 poison, !53, !DIExpression(), !46)
  %12 = icmp slt i32 %8, %11, !dbg !46
  br i1 %12, label %for_body_parallel_0, label %common.ret, !dbg !46, !prof !54

for_begin_parallel_0:                             ; preds = %if_end11
  %13 = add nsw i32 %parallel_021, 1, !dbg !46
    #dbg_value(i32 %13, !53, !DIExpression(), !46)
  %exitcond39.not = icmp eq i32 %13, %11, !dbg !46
  br i1 %exitcond39.not, label %common.ret, label %for_body_parallel_0, !dbg !46, !prof !55

for_body_parallel_0:                              ; preds = %parallel_closure_entry, %for_begin_parallel_0
  %parallel_021 = phi i32 [ %13, %for_begin_parallel_0 ], [ %8, %parallel_closure_entry ]
    #dbg_value(i32 %parallel_021, !53, !DIExpression(), !46)
  %14 = load ptr, ptr @__TVMBackendAllocWorkspace, align 8, !dbg !46, !tbaa !50
  %tile_storage_0 = tail call ptr %14(i32 1, i32 %dev_id, i64 1024, i32 2, i32 32), !dbg !46
    #dbg_declare(ptr %tile_storage_0, !56, !DIExpression(), !46)
  %15 = icmp eq ptr %tile_storage_0, null, !dbg !46
  br i1 %15, label %common.ret, label %if_end, !dbg !46, !prof !16

common.ret:                                       ; preds = %for_begin_parallel_0, %for_body_parallel_0, %for_end_tile_i_2, %for_end_tile_i_9, %if_end11, %parallel_closure_entry
  %common.ret.op = phi i32 [ 0, %parallel_closure_entry ], [ -1, %if_end11 ], [ -1, %for_end_tile_i_9 ], [ -1, %for_end_tile_i_2 ], [ -1, %for_body_parallel_0 ], [ 0, %for_begin_parallel_0 ]
  ret i32 %common.ret.op, !dbg !46

if_end:                                           ; preds = %for_body_parallel_0
  %parallel_021.frozen = freeze i32 %parallel_021, !dbg !46
  %16 = sdiv i32 %parallel_021.frozen, 7, !dbg !46
  %17 = mul i32 %16, 7, !dbg !46
  %.decomposed = sub i32 %parallel_021.frozen, %17, !dbg !46
  %cse_v1 = shl nsw i32 %.decomposed, 8, !dbg !46
    #dbg_value(i32 %cse_v1, !57, !DIExpression(), !46)
  %18 = mul nsw i32 %16, 1537, !dbg !46
  %cse_v2 = add nsw i32 %18, %cse_v1, !dbg !46
    #dbg_value(i32 %cse_v2, !58, !DIExpression(), !46)
    #dbg_value(i32 0, !59, !DIExpression(), !46)
  %19 = sext i32 %cse_v1 to i64, !dbg !46
  %20 = sext i32 %cse_v2 to i64, !dbg !46
  %invariant.gep = getelementptr float, ptr %arg0, i64 %20, !dbg !46
  br label %for_body_tile_i_2, !dbg !46

for_body_tile_i_2:                                ; preds = %if_end, %if_end2
  %indvars.iv = phi i64 [ 0, %if_end ], [ %indvars.iv.next, %if_end2 ]
    #dbg_value(i64 %indvars.iv, !59, !DIExpression(), !46)
  %21 = or disjoint i64 %indvars.iv, %19, !dbg !46
  %22 = icmp slt i64 %21, 1537, !dbg !46
  br i1 %22, label %if_then1, label %if_end2, !dbg !46

for_end_tile_i_2:                                 ; preds = %if_end2
  %23 = load ptr, ptr @__TVMBackendAllocWorkspace, align 8, !dbg !46, !tbaa !50
  %tile_storage_3 = tail call ptr %23(i32 1, i32 %dev_id, i64 1024, i32 2, i32 32), !dbg !46
    #dbg_declare(ptr %tile_storage_3, !60, !DIExpression(), !46)
  %24 = icmp eq ptr %tile_storage_3, null, !dbg !46
  br i1 %24, label %common.ret, label %vector.body, !dbg !46, !prof !16

vector.body:                                      ; preds = %for_end_tile_i_2, %vector.body
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %for_end_tile_i_2 ], !dbg !46
  %25 = getelementptr inbounds nuw float, ptr %tile_storage_0, i64 %index, !dbg !46
  %wide.load = load <4 x float>, ptr %25, align 4, !dbg !46, !tbaa !61
  %26 = fsub <4 x float> zeroinitializer, %wide.load, !dbg !46
  %27 = tail call <4 x float> @llvm.exp.v4f32(<4 x float> %26), !dbg !46
  %28 = fadd <4 x float> %27, splat (float 1.000000e+00), !dbg !46
  %29 = fdiv <4 x float> splat (float 1.000000e+00), %28, !dbg !46
  %30 = getelementptr inbounds nuw float, ptr %tile_storage_3, i64 %index, !dbg !46
  store <4 x float> %29, ptr %30, align 4, !dbg !46, !tbaa !63
  %index.next = add nuw i64 %index, 4, !dbg !46
  %31 = icmp eq i64 %index.next, 256, !dbg !46
  br i1 %31, label %for_body_tile_i_7.preheader, label %vector.body, !dbg !46, !prof !65, !llvm.loop !66

if_then1:                                         ; preds = %for_body_tile_i_2
  %gep = getelementptr float, ptr %invariant.gep, i64 %indvars.iv, !dbg !46
  %32 = load float, ptr %gep, align 4, !dbg !46, !tbaa !69
  br label %if_end2, !dbg !46

if_end2:                                          ; preds = %for_body_tile_i_2, %if_then1
  %33 = phi float [ %32, %if_then1 ], [ 0.000000e+00, %for_body_tile_i_2 ], !dbg !46
  %34 = getelementptr inbounds nuw float, ptr %tile_storage_0, i64 %indvars.iv, !dbg !46
  store float %33, ptr %34, align 4, !dbg !46, !tbaa !61
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !46
    #dbg_value(i64 %indvars.iv.next, !59, !DIExpression(), !46)
  %exitcond.not = icmp eq i64 %indvars.iv.next, 256, !dbg !46
  br i1 %exitcond.not, label %for_end_tile_i_2, label %for_body_tile_i_2, !dbg !46, !prof !71

for_body_tile_i_7.preheader:                      ; preds = %vector.body
  %invariant.gep40 = getelementptr float, ptr %arg1, i64 %20, !dbg !46
  br label %for_body_tile_i_7, !dbg !46

for_body_tile_i_7:                                ; preds = %for_body_tile_i_7.preheader, %if_end7
  %indvars.iv31 = phi i64 [ 0, %for_body_tile_i_7.preheader ], [ %indvars.iv.next32, %if_end7 ]
    #dbg_value(i64 %indvars.iv31, !72, !DIExpression(), !46)
  %35 = or disjoint i64 %indvars.iv31, %19, !dbg !46
  %36 = icmp slt i64 %35, 1537, !dbg !46
  br i1 %36, label %if_then6, label %if_end7, !dbg !46, !prof !16

if_then6:                                         ; preds = %for_body_tile_i_7
  %37 = getelementptr inbounds nuw float, ptr %tile_storage_3, i64 %indvars.iv31, !dbg !46
  %38 = load float, ptr %37, align 4, !dbg !46, !tbaa !63
  %gep41 = getelementptr float, ptr %invariant.gep40, i64 %indvars.iv31, !dbg !46
  store float %38, ptr %gep41, align 4, !dbg !46, !tbaa !73
  br label %if_end7, !dbg !46

if_end7:                                          ; preds = %if_then6, %for_body_tile_i_7
  %indvars.iv.next32 = add nuw nsw i64 %indvars.iv31, 1, !dbg !46
    #dbg_value(i64 %indvars.iv.next32, !72, !DIExpression(), !46)
  %exitcond34.not = icmp eq i64 %indvars.iv.next32, 256, !dbg !46
  br i1 %exitcond34.not, label %for_body_tile_i_9.preheader, label %for_body_tile_i_7, !dbg !46, !prof !71

for_body_tile_i_9.preheader:                      ; preds = %if_end7
  %invariant.gep42 = getelementptr float, ptr %arg2, i64 %20, !dbg !46
  br label %for_body_tile_i_9, !dbg !46

for_body_tile_i_9:                                ; preds = %for_body_tile_i_9.preheader, %if_end9
  %indvars.iv35 = phi i64 [ 0, %for_body_tile_i_9.preheader ], [ %indvars.iv.next36, %if_end9 ]
    #dbg_value(i64 %indvars.iv35, !75, !DIExpression(), !46)
  %39 = or disjoint i64 %indvars.iv35, %19, !dbg !46
  %40 = icmp slt i64 %39, 1537, !dbg !46
  br i1 %40, label %if_then8, label %if_end9, !dbg !46, !prof !16

for_end_tile_i_9:                                 ; preds = %if_end9
  %41 = load ptr, ptr @__TVMBackendFreeWorkspace, align 8, !dbg !46, !tbaa !50
  %42 = tail call i32 %41(i32 1, i32 %dev_id, ptr nonnull %tile_storage_3), !dbg !46
  %.not = icmp eq i32 %42, 0, !dbg !46
  br i1 %.not, label %if_end11, label %common.ret, !dbg !46, !prof !17

if_then8:                                         ; preds = %for_body_tile_i_9
  %43 = getelementptr inbounds nuw float, ptr %tile_storage_3, i64 %indvars.iv35, !dbg !46
  %44 = load float, ptr %43, align 4, !dbg !46, !tbaa !63
  %45 = fsub float 1.000000e+00, %44, !dbg !46
  %46 = fmul float %44, %45, !dbg !46
  %gep43 = getelementptr float, ptr %invariant.gep42, i64 %indvars.iv35, !dbg !46
  store float %46, ptr %gep43, align 4, !dbg !46, !tbaa !76
  br label %if_end9, !dbg !46

if_end9:                                          ; preds = %if_then8, %for_body_tile_i_9
  %indvars.iv.next36 = add nuw nsw i64 %indvars.iv35, 1, !dbg !46
    #dbg_value(i64 %indvars.iv.next36, !75, !DIExpression(), !46)
  %exitcond38.not = icmp eq i64 %indvars.iv.next36, 256, !dbg !46
  br i1 %exitcond38.not, label %for_end_tile_i_9, label %for_body_tile_i_9, !dbg !46, !prof !71

if_end11:                                         ; preds = %for_end_tile_i_9
  %47 = load ptr, ptr @__TVMBackendFreeWorkspace, align 8, !dbg !46, !tbaa !50
  %48 = tail call i32 %47(i32 1, i32 %dev_id, ptr nonnull %tile_storage_0), !dbg !46
  %.not16 = icmp eq i32 %48, 0, !dbg !46
  br i1 %.not16, label %for_begin_parallel_0, label %common.ret, !dbg !46, !prof !17
}

define weak dllexport i32 @__tvm_ffi_main(ptr %0, ptr %1, i32 %2, ptr %3) local_unnamed_addr {
entry:
  %4 = tail call i32 @__tvm_ffi_benchmark_activation_pair(ptr poison, ptr %1, i32 %2, ptr poison), !dbg !15
  ret i32 %4, !dbg !15
}

; Function Attrs: nofree nosync nounwind memory(none)
define weak dso_local i16 @__truncsfhf2(float %a0) local_unnamed_addr #4 {
b0:
  %v0 = bitcast float %a0 to i32
  %0 = tail call float @llvm.fabs.f32(float %a0)
  %v1 = bitcast float %0 to i32
  %v2 = add nsw i32 %v1, -947912704
  %v3 = add nsw i32 %v1, -1199570944
  %v4 = icmp ult i32 %v2, %v3
  br i1 %v4, label %b1, label %b5

b1:                                               ; preds = %b0
  %v5 = lshr i32 %v0, 13
  %v7 = add nsw i32 %v5, -114688
  %v8 = and i32 %v0, 8191
  %v9 = icmp samesign ugt i32 %v8, 4096
  br i1 %v9, label %b2, label %b3

b2:                                               ; preds = %b1
  %v10 = add nsw i32 %v5, -114687
  br label %b13

b3:                                               ; preds = %b1
  %v11 = icmp eq i32 %v8, 4096
  br i1 %v11, label %b4, label %b13

b4:                                               ; preds = %b3
  %v13 = and i32 %v5, 1
  %v14 = add nsw i32 %v7, %v13
  br label %b13

b5:                                               ; preds = %b0
  %v15 = icmp samesign ugt i32 %v1, 2139095040
  br i1 %v15, label %b6, label %b7

b6:                                               ; preds = %b5
  %v16 = lshr i32 %v0, 13
  %v17 = and i32 %v16, 511
  %v18 = or disjoint i32 %v17, 32256
  br label %b13

b7:                                               ; preds = %b5
  %v19 = icmp samesign ugt i32 %v1, 1199570943
  br i1 %v19, label %b13, label %b8

b8:                                               ; preds = %b7
  %v20 = icmp samesign ult i32 %v1, 754974720
  br i1 %v20, label %b13, label %b9

b9:                                               ; preds = %b8
  %v21 = lshr i32 %v1, 23
  %v22 = sub nsw i32 113, %v21
  %v23 = and i32 %v0, 8388607
  %v24 = or disjoint i32 %v23, 8388608
  %v25 = add nsw i32 %v21, -81
  %v26 = shl i32 %v24, %v25
  %v27 = icmp ne i32 %v26, 0
  %v28 = lshr i32 %v24, %v22
  %v29 = zext i1 %v27 to i32
  %v30 = lshr i32 %v28, 13
  %v31 = and i32 %v28, 8191
  %v32 = or i32 %v31, %v29
  %v33 = icmp samesign ugt i32 %v32, 4096
  br i1 %v33, label %b10, label %b11

b10:                                              ; preds = %b9
  %v34 = add nuw nsw i32 %v30, 1
  br label %b13

b11:                                              ; preds = %b9
  %v35 = icmp eq i32 %v32, 4096
  br i1 %v35, label %b12, label %b13

b12:                                              ; preds = %b11
  %v36 = and i32 %v30, 1
  %v37 = add nuw nsw i32 %v36, %v30
  br label %b13

b13:                                              ; preds = %b12, %b11, %b10, %b8, %b7, %b6, %b4, %b3, %b2
  %v38 = phi i32 [ %v18, %b6 ], [ %v10, %b2 ], [ %v14, %b4 ], [ %v7, %b3 ], [ 31744, %b7 ], [ 0, %b8 ], [ %v34, %b10 ], [ %v37, %b12 ], [ %v30, %b11 ]
  %v39 = lshr i32 %v0, 16
  %v40 = and i32 %v39, 32768
  %v41 = or i32 %v38, %v40
  %vlast = trunc i32 %v41 to i16
  ret i16 %vlast
}

; Function Attrs: nofree nosync nounwind memory(none)
define weak dso_local float @__extendhfsf2(i16 %a0) local_unnamed_addr #4 {
b0:
  %v1 = and i16 %a0, 32767
  %v2 = zext nneg i16 %v1 to i32
  %v3 = add nsw i16 %v1, -1024
  %v4 = icmp ult i16 %v3, 30720
  br i1 %v4, label %b1, label %b2

b1:                                               ; preds = %b0
  %v5 = shl nuw nsw i32 %v2, 13
  %v6 = add nuw nsw i32 %v5, 939524096
  br label %b6

b2:                                               ; preds = %b0
  %v7 = icmp samesign ugt i16 %v1, 31743
  br i1 %v7, label %b3, label %b4

b3:                                               ; preds = %b2
  %v8 = shl nuw nsw i32 %v2, 13
  %v9 = or i32 %v8, 2139095040
  br label %b6

b4:                                               ; preds = %b2
  %v10 = icmp eq i16 %v1, 0
  br i1 %v10, label %b6, label %b5

b5:                                               ; preds = %b4
  %v11 = icmp samesign ult i16 %v1, 256
  %v12 = lshr i32 %v2, 8
  %v13 = select i1 %v11, i32 %v2, i32 %v12
  %v14 = select i1 %v11, i32 32, i32 24
  %v15 = icmp samesign ult i32 %v13, 16
  %v16 = lshr i32 %v13, 4
  %v17 = add nsw i32 %v14, -4
  %v18 = select i1 %v15, i32 %v13, i32 %v16
  %v19 = select i1 %v15, i32 %v14, i32 %v17
  %v20 = icmp samesign ult i32 %v18, 4
  %v21 = lshr i32 %v18, 2
  %v22 = add nsw i32 %v19, -2
  %v23 = select i1 %v20, i32 %v18, i32 %v21
  %v24 = select i1 %v20, i32 %v19, i32 %v22
  %v25 = icmp samesign ult i32 %v23, 2
  %v26 = sub nsw i32 0, %v23
  %v27 = select i1 %v25, i32 %v26, i32 -2
  %v28 = add nsw i32 %v27, %v24
  %v29 = add nsw i32 %v28, -8
  %v30 = shl i32 %v2, %v29
  %v31 = xor i32 %v30, 8388608
  %v32 = shl i32 %v28, 23
  %v33 = sub i32 1124073472, %v32
  %v34 = or i32 %v31, %v33
  br label %b6

b6:                                               ; preds = %b5, %b4, %b3, %b1
  %v35 = phi i32 [ %v6, %b1 ], [ %v9, %b3 ], [ %v34, %b5 ], [ 0, %b4 ]
  %v36 = and i16 %a0, -32768
  %v37 = zext i16 %v36 to i32
  %v38 = shl nuw i32 %v37, 16
  %v39 = or i32 %v35, %v38
  %v40 = bitcast i32 %v39 to float
  ret float %v40
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #5

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #5

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #6

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.exp.v4f32(<4 x float>) #5

attributes #0 = { "target-cpu"="generic" }
attributes #1 = { noinline }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #3 = { noinline "target-cpu"="generic" }
attributes #4 = { nofree nosync nounwind memory(none) "target-cpu"="generic" "target-features" }
attributes #5 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { nocallback nofree nounwind willreturn memory(argmem: write) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "TVM", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "IRModule.CodeGenLLVM", directory: ".")
!2 = !{i32 2, !"tvm_target", !"{\22kind\22:\22llvm\22,\22mtriple\22:\22arm64-apple-darwin25.6.0\22}"}
!3 = !{i32 4, !"Debug Info Version", i32 3}
!4 = !{i32 4, !"Dwarf Version", i32 2}
!5 = distinct !DISubprogram(name: "__tvm_ffi_benchmark_activation_pair", scope: !1, file: !1, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9, !9, !8, !9}
!8 = !DIBasicType(name: "int32", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null)
!10 = !{!11, !12, !13, !14}
!11 = !DILocalVariable(name: "self_handle", arg: 1, scope: !5, file: !1, type: !9)
!12 = !DILocalVariable(name: "args", arg: 2, scope: !5, file: !1, type: !9)
!13 = !DILocalVariable(name: "num_args", arg: 3, scope: !5, file: !1, type: !8)
!14 = !DILocalVariable(name: "result", arg: 4, scope: !5, file: !1, type: !9)
!15 = !DILocation(line: 0, scope: !5)
!16 = !{!"branch_weights", i32 1048576, i32 1}
!17 = !{!"branch_weights", i32 1, i32 1048576}
!18 = !DILocalVariable(name: "arg0.type_index", scope: !5, file: !1, type: !8)
!19 = !DILocalVariable(name: "arg0.handle", scope: !5, file: !1, type: !9)
!20 = !DILocalVariable(name: "arg1.type_index", scope: !5, file: !1, type: !8)
!21 = !DILocalVariable(name: "arg1.handle", scope: !5, file: !1, type: !9)
!22 = !DILocalVariable(name: "arg2.type_index", scope: !5, file: !1, type: !8)
!23 = !DILocalVariable(name: "arg2.handle", scope: !5, file: !1, type: !9)
!24 = !DILocalVariable(name: "benchmark_activation_pair.arg0_shape", scope: !5, file: !1, type: !25)
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !26)
!26 = !DIBasicType(name: "int64", size: 64, encoding: DW_ATE_signed)
!27 = !DILocalVariable(name: "benchmark_activation_pair.arg0_strides", scope: !5, file: !1, type: !25)
!28 = !DILocalVariable(name: "dev_id", scope: !5, file: !1, type: !8)
!29 = !DILocalVariable(name: "benchmark_activation_pair.arg1_shape", scope: !5, file: !1, type: !25)
!30 = !DILocalVariable(name: "benchmark_activation_pair.arg1_strides", scope: !5, file: !1, type: !25)
!31 = !DILocalVariable(name: "benchmark_activation_pair.arg2_shape", scope: !5, file: !1, type: !25)
!32 = !DILocalVariable(name: "benchmark_activation_pair.arg2_strides", scope: !5, file: !1, type: !25)
!33 = !DILocalVariable(name: "arg0", scope: !5, file: !1, type: !34)
!34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !35)
!35 = !DIBasicType(name: "float32", size: 32, encoding: DW_ATE_float)
!36 = !DILocalVariable(name: "arg1", scope: !5, file: !1, type: !34)
!37 = !DILocalVariable(name: "arg2", scope: !5, file: !1, type: !34)
!38 = distinct !DISubprogram(name: "benchmark_activation_pair_compute_", scope: !1, file: !1, type: !39, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !41)
!39 = !DISubroutineType(types: !40)
!40 = !{!8, !34, !34, !34, !8}
!41 = !{!42, !43, !44, !45}
!42 = !DILocalVariable(name: "arg0", arg: 1, scope: !38, file: !1, type: !34)
!43 = !DILocalVariable(name: "arg1", arg: 2, scope: !38, file: !1, type: !34)
!44 = !DILocalVariable(name: "arg2", arg: 3, scope: !38, file: !1, type: !34)
!45 = !DILocalVariable(name: "dev_id", arg: 4, scope: !38, file: !1, type: !8)
!46 = !DILocation(line: 0, scope: !38)
!47 = !DILocalVariable(name: "arg0", scope: !38, file: !1, type: !34)
!48 = !DILocalVariable(name: "arg1", scope: !38, file: !1, type: !34)
!49 = !DILocalVariable(name: "arg2", scope: !38, file: !1, type: !34)
!50 = !{!51, !51, i64 0}
!51 = !{!"ctx_ptr", !52, i64 0}
!52 = !{!"tvm-tbaa"}
!53 = !DILocalVariable(name: "parallel_0", scope: !38, file: !1, type: !8)
!54 = !{!"branch_weights", i32 127, i32 1}
!55 = !{!"branch_weights", i32 127, i32 134217601}
!56 = !DILocalVariable(name: "tile_storage_0", scope: !38, file: !1, type: !34)
!57 = !DILocalVariable(name: "cse_v1", scope: !38, file: !1, type: !8)
!58 = !DILocalVariable(name: "cse_v2", scope: !38, file: !1, type: !8)
!59 = !DILocalVariable(name: "tile_i_2", scope: !38, file: !1, type: !8)
!60 = !DILocalVariable(name: "tile_storage_3", scope: !38, file: !1, type: !34)
!61 = !{!62, !62, i64 0}
!62 = !{!"0x737274280", !52, i64 0}
!63 = !{!64, !64, i64 0}
!64 = !{!"0x737454340", !52, i64 0}
!65 = !{!"branch_weights", i32 1, i32 262143}
!66 = distinct !{!66, !67, !68}
!67 = !{!"llvm.loop.isvectorized", i32 1}
!68 = !{!"llvm.loop.unroll.runtime.disable"}
!69 = !{!70, !70, i64 0}
!70 = !{!"0x73722d940", !52, i64 0}
!71 = !{!"branch_weights", i32 1, i32 1048575}
!72 = !DILocalVariable(name: "tile_i_7", scope: !38, file: !1, type: !8)
!73 = !{!74, !74, i64 0}
!74 = !{!"0x73722d980", !52, i64 0}
!75 = !DILocalVariable(name: "tile_i_9", scope: !38, file: !1, type: !8)
!76 = !{!77, !77, i64 0}
!77 = !{!"0x73722d9c0", !52, i64 0}
