; ModuleID = 'TVMMod'
source_filename = "TVMMod"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-darwin25.6.0"

%closure_loop_parallel_parallel_0 = type { i32, ptr, ptr }

@__TVMBackendParallelLaunch = linkonce dllexport local_unnamed_addr global ptr null, align 8
@.str = private constant [10 x i8] c"TypeError\00", align 1
@.str.1 = private constant [10 x i8] c"Expected \00", align 1
@.str.2 = private constant [2 x i8] c"2\00", align 1
@.str.3 = private constant [11 x i8] c" arguments\00", align 1
@.str.4 = private constant [19 x i8] c" when calling:\0A  `\00", align 1
@.str.5 = private constant [122 x i8] c"benchmark_softmax(arg0: Tensor([T.int64(17), T.int64(257)], float32), arg1: Tensor([T.int64(17), T.int64(257)], float32))\00", align 1
@.str.6 = private constant [2 x i8] c"`\00", align 1
@.str.7 = private constant [21 x i8] c"args pointer is NULL\00", align 1
@.str.8 = private constant [30 x i8] c"Mismatched type on argument #\00", align 1
@.str.9 = private constant [2 x i8] c"0\00", align 1
@.str.10 = private constant [15 x i8] c"`,\0A  expected \00", align 1
@.str.11 = private constant [7 x i8] c"Tensor\00", align 1
@.str.12 = private constant [2 x i8] c"1\00", align 1
@.str.13 = private constant [11 x i8] c"ValueError\00", align 1
@.str.14 = private constant [12 x i8] c"Mismatched \00", align 1
@.str.15 = private constant [5 x i8] c"arg0\00", align 1
@.str.16 = private constant [20 x i8] c".ndim on argument #\00", align 1
@.str.17 = private constant [21 x i8] c".dtype on argument #\00", align 1
@.str.18 = private constant [8 x i8] c"float32\00", align 1
@.str.19 = private constant [27 x i8] c".device_type on argument #\00", align 1
@.str.20 = private constant [4 x i8] c"cpu\00", align 1
@.str.21 = private constant [5 x i8] c"arg1\00", align 1
@.str.22 = private constant [23 x i8] c".strides on argument #\00", align 1
@.str.23 = private constant [34 x i8] c"`,\0A  expected to be compact array\00", align 1
@.str.24 = private constant [36 x i8] c" data pointer is NULL on argument #\00", align 1
@.str.25 = private constant [36 x i8] c"`,\0A  expected non-NULL data pointer\00", align 1
@.str.26 = private constant [15 x i8] c"arg1.device_id\00", align 1
@.str.27 = private constant [15 x i8] c" on argument #\00", align 1
@.str.28 = private constant [24 x i8] c"`,\0A  expected to match \00", align 1
@.str.29 = private constant [15 x i8] c"arg0.device_id\00", align 1
@.str.30 = private constant [9 x i8] c"Invalid \00", align 1
@.str.31 = private constant [14 x i8] c"arg0.shape[0]\00", align 1
@.str.32 = private constant [3 x i8] c"17\00", align 1
@.str.33 = private constant [14 x i8] c"arg0.shape[1]\00", align 1
@.str.34 = private constant [4 x i8] c"257\00", align 1
@.str.35 = private constant [17 x i8] c"arg0.byte_offset\00", align 1
@.str.36 = private constant [14 x i8] c"arg1.shape[0]\00", align 1
@.str.37 = private constant [14 x i8] c"arg1.shape[1]\00", align 1
@.str.38 = private constant [17 x i8] c"arg1.byte_offset\00", align 1
@__TVMBackendAllocWorkspace = linkonce dllexport local_unnamed_addr global ptr null, align 8
@__TVMBackendFreeWorkspace = linkonce dllexport local_unnamed_addr global ptr null, align 8
@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

define dllexport i32 @__tvm_ffi_benchmark_softmax(ptr noalias readnone captures(none) %self_handle, ptr noalias readonly captures(address_is_null) %args, i32 %num_args, ptr noalias readnone captures(none) %result) local_unnamed_addr #0 !dbg !5 {
entry:
    #dbg_value(ptr poison, !11, !DIExpression(), !15)
    #dbg_value(ptr %args, !12, !DIExpression(), !15)
    #dbg_value(i32 %num_args, !13, !DIExpression(), !15)
    #dbg_value(ptr poison, !14, !DIExpression(), !15)
  %0 = icmp eq i32 %num_args, 2, !dbg !15
  br i1 %0, label %assert_end, label %assert_fail, !dbg !15, !prof !16

common.ret:                                       ; preds = %assert_end51, %assert_fail50, %assert_fail48, %assert_fail46, %assert_fail44, %assert_fail42, %assert_fail40, %assert_fail38, %assert_fail36, %assert_fail34, %assert_fail30, %assert_fail28, %assert_fail24, %assert_fail22, %assert_fail20, %assert_fail18, %assert_fail16, %assert_fail14, %assert_fail12, %assert_fail10, %assert_fail5, %assert_fail3, %assert_fail1, %assert_fail
  %common.ret.op = phi i32 [ -1, %assert_fail ], [ -1, %assert_fail1 ], [ -1, %assert_fail3 ], [ -1, %assert_fail5 ], [ -1, %assert_fail10 ], [ -1, %assert_fail12 ], [ -1, %assert_fail14 ], [ -1, %assert_fail16 ], [ -1, %assert_fail18 ], [ -1, %assert_fail20 ], [ -1, %assert_fail22 ], [ -1, %assert_fail24 ], [ -1, %assert_fail28 ], [ -1, %assert_fail30 ], [ -1, %assert_fail34 ], [ -1, %assert_fail36 ], [ -1, %assert_fail38 ], [ -1, %assert_fail40 ], [ -1, %assert_fail42 ], [ -1, %assert_fail44 ], [ -1, %assert_fail46 ], [ -1, %assert_fail48 ], [ -1, %assert_fail50 ], [ %88, %assert_end51 ]
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
  br i1 %9, label %assert_end6, label %switch.early.test52, !dbg !15

switch.early.test52:                              ; preds = %if_end
  switch i32 %arg1.type_index.fr, label %assert_fail5 [
    i32 0, label %if_else8
    i32 4, label %if_else8
    i32 7, label %if_else8
  ], !dbg !15

assert_fail5:                                     ; preds = %switch.early.test52
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

if_else8:                                         ; preds = %switch.early.test52, %switch.early.test52, %switch.early.test52, %assert_end6
  %14 = getelementptr inbounds nuw i8, ptr %args, i64 24, !dbg !15
  %15 = load ptr, ptr %14, align 8, !dbg !15
  br label %if_end9, !dbg !15

if_end9:                                          ; preds = %if_else8, %if_then7
  %arg1.handle = phi ptr [ %13, %if_then7 ], [ %15, %if_else8 ], !dbg !15
    #dbg_declare(ptr %arg1.handle, !21, !DIExpression(), !15)
    #dbg_declare(ptr %arg1.handle, !21, !DIExpression(), !15)
  %.not53 = icmp eq ptr %arg0.handle, null, !dbg !15
  br i1 %.not53, label %assert_fail10, label %assert_end11, !dbg !15, !prof !17

assert_fail10:                                    ; preds = %if_end9
  tail call fastcc void @__tvm_set_raised_6(ptr nonnull @.str, i32 6, ptr nonnull @.str.8, ptr nonnull @.str.9, ptr nonnull @.str.4, ptr nonnull @.str.5, ptr nonnull @.str.10, ptr nonnull @.str.11), !dbg !15
  br label %common.ret, !dbg !15

assert_end11:                                     ; preds = %if_end9
  %16 = getelementptr inbounds nuw i8, ptr %arg0.handle, i64 16, !dbg !15
  %17 = load i32, ptr %16, align 4, !dbg !15
  %18 = icmp eq i32 %17, 2, !dbg !15
  br i1 %18, label %assert_end13, label %assert_fail12, !dbg !15, !prof !16

assert_fail12:                                    ; preds = %assert_end11
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.13, i32 8, ptr nonnull @.str.14, ptr nonnull @.str.15, ptr nonnull @.str.16, ptr nonnull @.str.9, ptr nonnull @.str.10, ptr nonnull @.str.2), !dbg !15
  br label %common.ret, !dbg !15

assert_end13:                                     ; preds = %assert_end11
  %19 = getelementptr inbounds nuw i8, ptr %arg0.handle, i64 20, !dbg !15
  %20 = load i8, ptr %19, align 1, !dbg !15
  %21 = icmp eq i8 %20, 2, !dbg !15
  %22 = getelementptr inbounds nuw i8, ptr %arg0.handle, i64 21, !dbg !15
  %23 = load i8, ptr %22, align 1, !dbg !15
  %24 = icmp eq i8 %23, 32, !dbg !15
  %25 = and i1 %21, %24, !dbg !15
  %26 = getelementptr inbounds nuw i8, ptr %arg0.handle, i64 22, !dbg !15
  %27 = load i16, ptr %26, align 2, !dbg !15
  %28 = icmp eq i16 %27, 1, !dbg !15
  %29 = and i1 %25, %28, !dbg !15
  br i1 %29, label %assert_end15, label %assert_fail14, !dbg !15, !prof !16

assert_fail14:                                    ; preds = %assert_end13
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str, i32 8, ptr nonnull @.str.14, ptr nonnull @.str.15, ptr nonnull @.str.17, ptr nonnull @.str.9, ptr nonnull @.str.10, ptr nonnull @.str.18), !dbg !15
  br label %common.ret, !dbg !15

assert_end15:                                     ; preds = %assert_end13
  %30 = getelementptr inbounds nuw i8, ptr %arg0.handle, i64 24, !dbg !15
  %benchmark_softmax.arg0_shape = load ptr, ptr %30, align 8, !dbg !15
    #dbg_declare(ptr %benchmark_softmax.arg0_shape, !22, !DIExpression(), !15)
    #dbg_declare(ptr %benchmark_softmax.arg0_shape, !22, !DIExpression(), !15)
  %31 = getelementptr inbounds nuw i8, ptr %arg0.handle, i64 32, !dbg !15
  %benchmark_softmax.arg0_strides = load ptr, ptr %31, align 8, !dbg !15
    #dbg_declare(ptr %benchmark_softmax.arg0_strides, !25, !DIExpression(), !15)
    #dbg_declare(ptr %benchmark_softmax.arg0_strides, !25, !DIExpression(), !15)
  %32 = getelementptr inbounds nuw i8, ptr %arg0.handle, i64 8, !dbg !15
  %33 = load i32, ptr %32, align 4, !dbg !15
  %34 = icmp eq i32 %33, 1, !dbg !15
  br i1 %34, label %assert_end17, label %assert_fail16, !dbg !15, !prof !16

assert_fail16:                                    ; preds = %assert_end15
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.13, i32 8, ptr nonnull @.str.14, ptr nonnull @.str.15, ptr nonnull @.str.19, ptr nonnull @.str.9, ptr nonnull @.str.10, ptr nonnull @.str.20), !dbg !15
  br label %common.ret, !dbg !15

assert_end17:                                     ; preds = %assert_end15
  %35 = getelementptr inbounds nuw i8, ptr %arg0.handle, i64 12, !dbg !15
  %dev_id = load i32, ptr %35, align 4, !dbg !15
    #dbg_value(i32 %dev_id, !26, !DIExpression(), !15)
    #dbg_value(i32 %dev_id, !26, !DIExpression(), !15)
  %.not54 = icmp eq ptr %arg1.handle, null, !dbg !15
  br i1 %.not54, label %assert_fail18, label %assert_end19, !dbg !15, !prof !17

assert_fail18:                                    ; preds = %assert_end17
  tail call fastcc void @__tvm_set_raised_6(ptr nonnull @.str, i32 6, ptr nonnull @.str.8, ptr nonnull @.str.12, ptr nonnull @.str.4, ptr nonnull @.str.5, ptr nonnull @.str.10, ptr nonnull @.str.11), !dbg !15
  br label %common.ret, !dbg !15

assert_end19:                                     ; preds = %assert_end17
  %36 = getelementptr inbounds nuw i8, ptr %arg1.handle, i64 16, !dbg !15
  %37 = load i32, ptr %36, align 4, !dbg !15
  %38 = icmp eq i32 %37, 2, !dbg !15
  br i1 %38, label %assert_end21, label %assert_fail20, !dbg !15, !prof !16

assert_fail20:                                    ; preds = %assert_end19
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.13, i32 8, ptr nonnull @.str.14, ptr nonnull @.str.21, ptr nonnull @.str.16, ptr nonnull @.str.12, ptr nonnull @.str.10, ptr nonnull @.str.2), !dbg !15
  br label %common.ret, !dbg !15

assert_end21:                                     ; preds = %assert_end19
  %39 = getelementptr inbounds nuw i8, ptr %arg1.handle, i64 20, !dbg !15
  %40 = load i8, ptr %39, align 1, !dbg !15
  %41 = icmp eq i8 %40, 2, !dbg !15
  %42 = getelementptr inbounds nuw i8, ptr %arg1.handle, i64 21, !dbg !15
  %43 = load i8, ptr %42, align 1, !dbg !15
  %44 = icmp eq i8 %43, 32, !dbg !15
  %45 = and i1 %41, %44, !dbg !15
  %46 = getelementptr inbounds nuw i8, ptr %arg1.handle, i64 22, !dbg !15
  %47 = load i16, ptr %46, align 2, !dbg !15
  %48 = icmp eq i16 %47, 1, !dbg !15
  %49 = and i1 %45, %48, !dbg !15
  br i1 %49, label %assert_end23, label %assert_fail22, !dbg !15, !prof !16

assert_fail22:                                    ; preds = %assert_end21
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str, i32 8, ptr nonnull @.str.14, ptr nonnull @.str.21, ptr nonnull @.str.17, ptr nonnull @.str.12, ptr nonnull @.str.10, ptr nonnull @.str.18), !dbg !15
  br label %common.ret, !dbg !15

assert_end23:                                     ; preds = %assert_end21
  %50 = getelementptr inbounds nuw i8, ptr %arg1.handle, i64 24, !dbg !15
  %benchmark_softmax.arg1_shape = load ptr, ptr %50, align 8, !dbg !15
    #dbg_declare(ptr %benchmark_softmax.arg1_shape, !27, !DIExpression(), !15)
    #dbg_declare(ptr %benchmark_softmax.arg1_shape, !27, !DIExpression(), !15)
  %51 = getelementptr inbounds nuw i8, ptr %arg1.handle, i64 32, !dbg !15
  %benchmark_softmax.arg1_strides = load ptr, ptr %51, align 8, !dbg !15
    #dbg_declare(ptr %benchmark_softmax.arg1_strides, !28, !DIExpression(), !15)
    #dbg_declare(ptr %benchmark_softmax.arg1_strides, !28, !DIExpression(), !15)
  %52 = getelementptr inbounds nuw i8, ptr %arg1.handle, i64 8, !dbg !15
  %53 = load i32, ptr %52, align 4, !dbg !15
  %54 = icmp eq i32 %53, 1, !dbg !15
  br i1 %54, label %assert_end25, label %assert_fail24, !dbg !15, !prof !16

assert_fail24:                                    ; preds = %assert_end23
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.13, i32 8, ptr nonnull @.str.14, ptr nonnull @.str.21, ptr nonnull @.str.19, ptr nonnull @.str.12, ptr nonnull @.str.10, ptr nonnull @.str.20), !dbg !15
  br label %common.ret, !dbg !15

assert_end25:                                     ; preds = %assert_end23
  %.not55 = icmp eq ptr %benchmark_softmax.arg0_strides, null, !dbg !15
  br i1 %.not55, label %if_end27, label %if_then26, !dbg !15, !prof !17

if_then26:                                        ; preds = %assert_end25
  %55 = getelementptr inbounds nuw i8, ptr %benchmark_softmax.arg0_strides, i64 8, !dbg !15
  %56 = load i64, ptr %55, align 8, !dbg !15
  %57 = icmp eq i64 %56, 1, !dbg !15
  %58 = load i64, ptr %benchmark_softmax.arg0_strides, align 8, !dbg !15
  %59 = icmp eq i64 %58, 257, !dbg !15
  %60 = and i1 %57, %59, !dbg !15
  br i1 %60, label %if_end27, label %assert_fail28, !dbg !15, !prof !16

if_end27:                                         ; preds = %if_then26, %assert_end25
  %61 = load ptr, ptr %arg0.handle, align 8, !dbg !15
  %.not56 = icmp eq ptr %61, null, !dbg !15
  br i1 %.not56, label %assert_fail30, label %assert_end31, !dbg !15, !prof !17

assert_fail28:                                    ; preds = %if_then26
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.13, i32 7, ptr nonnull @.str.14, ptr nonnull @.str.15, ptr nonnull @.str.22, ptr nonnull @.str.9, ptr nonnull @.str.23, ptr null), !dbg !15
  br label %common.ret, !dbg !15

assert_fail30:                                    ; preds = %if_end27
  tail call fastcc void @__tvm_set_raised_6(ptr nonnull @.str.13, i32 6, ptr nonnull @.str.15, ptr nonnull @.str.24, ptr nonnull @.str.9, ptr nonnull @.str.4, ptr nonnull @.str.5, ptr nonnull @.str.25), !dbg !15
  br label %common.ret, !dbg !15

assert_end31:                                     ; preds = %if_end27
  %.not57 = icmp eq ptr %benchmark_softmax.arg1_strides, null, !dbg !15
  br i1 %.not57, label %if_end33, label %if_then32, !dbg !15, !prof !17

if_then32:                                        ; preds = %assert_end31
  %62 = getelementptr inbounds nuw i8, ptr %benchmark_softmax.arg1_strides, i64 8, !dbg !15
  %63 = load i64, ptr %62, align 8, !dbg !15
  %64 = icmp eq i64 %63, 1, !dbg !15
  %65 = load i64, ptr %benchmark_softmax.arg1_strides, align 8, !dbg !15
  %66 = icmp eq i64 %65, 257, !dbg !15
  %67 = and i1 %64, %66, !dbg !15
  br i1 %67, label %if_end33, label %assert_fail34, !dbg !15, !prof !16

if_end33:                                         ; preds = %if_then32, %assert_end31
  %68 = getelementptr inbounds nuw i8, ptr %arg1.handle, i64 12, !dbg !15
  %69 = load i32, ptr %68, align 4, !dbg !15
  %70 = icmp eq i32 %dev_id, %69, !dbg !15
  br i1 %70, label %assert_end37, label %assert_fail36, !dbg !15, !prof !16

assert_fail34:                                    ; preds = %if_then32
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.13, i32 7, ptr nonnull @.str.14, ptr nonnull @.str.21, ptr nonnull @.str.22, ptr nonnull @.str.12, ptr nonnull @.str.23, ptr null), !dbg !15
  br label %common.ret, !dbg !15

assert_fail36:                                    ; preds = %if_end33
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.13, i32 8, ptr nonnull @.str.14, ptr nonnull @.str.26, ptr nonnull @.str.27, ptr nonnull @.str.12, ptr nonnull @.str.28, ptr nonnull @.str.29), !dbg !15
  br label %common.ret, !dbg !15

assert_end37:                                     ; preds = %if_end33
  %71 = load ptr, ptr %arg1.handle, align 8, !dbg !15
  %.not58 = icmp eq ptr %71, null, !dbg !15
  br i1 %.not58, label %assert_fail38, label %assert_end39, !dbg !15, !prof !17

assert_fail38:                                    ; preds = %assert_end37
  tail call fastcc void @__tvm_set_raised_6(ptr nonnull @.str.13, i32 6, ptr nonnull @.str.21, ptr nonnull @.str.24, ptr nonnull @.str.12, ptr nonnull @.str.4, ptr nonnull @.str.5, ptr nonnull @.str.25), !dbg !15
  br label %common.ret, !dbg !15

assert_end39:                                     ; preds = %assert_end37
  %72 = load i64, ptr %benchmark_softmax.arg0_shape, align 8, !dbg !15
  %73 = icmp eq i64 %72, 17, !dbg !15
  br i1 %73, label %assert_end41, label %assert_fail40, !dbg !15, !prof !16

assert_fail40:                                    ; preds = %assert_end39
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.13, i32 8, ptr nonnull @.str.30, ptr nonnull @.str.31, ptr nonnull @.str.27, ptr nonnull @.str.9, ptr nonnull @.str.10, ptr nonnull @.str.32), !dbg !15
  br label %common.ret, !dbg !15

assert_end41:                                     ; preds = %assert_end39
  %74 = getelementptr inbounds nuw i8, ptr %benchmark_softmax.arg0_shape, i64 8, !dbg !15
  %75 = load i64, ptr %74, align 8, !dbg !15
  %76 = icmp eq i64 %75, 257, !dbg !15
  br i1 %76, label %assert_end43, label %assert_fail42, !dbg !15, !prof !16

assert_fail42:                                    ; preds = %assert_end41
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.13, i32 8, ptr nonnull @.str.30, ptr nonnull @.str.33, ptr nonnull @.str.27, ptr nonnull @.str.9, ptr nonnull @.str.10, ptr nonnull @.str.34), !dbg !15
  br label %common.ret, !dbg !15

assert_end43:                                     ; preds = %assert_end41
  %77 = getelementptr inbounds nuw i8, ptr %arg0.handle, i64 40, !dbg !15
  %78 = load i64, ptr %77, align 8, !dbg !15
  %79 = icmp eq i64 %78, 0, !dbg !15
  br i1 %79, label %assert_end45, label %assert_fail44, !dbg !15, !prof !16

assert_fail44:                                    ; preds = %assert_end43
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.13, i32 8, ptr nonnull @.str.30, ptr nonnull @.str.35, ptr nonnull @.str.27, ptr nonnull @.str.9, ptr nonnull @.str.10, ptr nonnull @.str.9), !dbg !15
  br label %common.ret, !dbg !15

assert_end45:                                     ; preds = %assert_end43
  %80 = load i64, ptr %benchmark_softmax.arg1_shape, align 8, !dbg !15
  %81 = icmp eq i64 %80, 17, !dbg !15
  br i1 %81, label %assert_end47, label %assert_fail46, !dbg !15, !prof !16

assert_fail46:                                    ; preds = %assert_end45
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.13, i32 8, ptr nonnull @.str.30, ptr nonnull @.str.36, ptr nonnull @.str.27, ptr nonnull @.str.12, ptr nonnull @.str.10, ptr nonnull @.str.32), !dbg !15
  br label %common.ret, !dbg !15

assert_end47:                                     ; preds = %assert_end45
  %82 = getelementptr inbounds nuw i8, ptr %benchmark_softmax.arg1_shape, i64 8, !dbg !15
  %83 = load i64, ptr %82, align 8, !dbg !15
  %84 = icmp eq i64 %83, 257, !dbg !15
  br i1 %84, label %assert_end49, label %assert_fail48, !dbg !15, !prof !16

assert_fail48:                                    ; preds = %assert_end47
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.13, i32 8, ptr nonnull @.str.30, ptr nonnull @.str.37, ptr nonnull @.str.27, ptr nonnull @.str.12, ptr nonnull @.str.10, ptr nonnull @.str.34), !dbg !15
  br label %common.ret, !dbg !15

assert_end49:                                     ; preds = %assert_end47
  %85 = getelementptr inbounds nuw i8, ptr %arg1.handle, i64 40, !dbg !15
  %86 = load i64, ptr %85, align 8, !dbg !15
  %87 = icmp eq i64 %86, 0, !dbg !15
  br i1 %87, label %assert_end51, label %assert_fail50, !dbg !15, !prof !16

assert_fail50:                                    ; preds = %assert_end49
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.13, i32 8, ptr nonnull @.str.30, ptr nonnull @.str.38, ptr nonnull @.str.27, ptr nonnull @.str.12, ptr nonnull @.str.10, ptr nonnull @.str.9), !dbg !15
  br label %common.ret, !dbg !15

assert_end51:                                     ; preds = %assert_end49
    #dbg_declare(ptr %61, !29, !DIExpression(), !15)
  call void @llvm.assume(i1 true) [ "align"(ptr %61, i64 64) ], !dbg !15
    #dbg_declare(ptr %71, !32, !DIExpression(), !15)
  call void @llvm.assume(i1 true) [ "align"(ptr %71, i64 64) ], !dbg !15
  %88 = tail call fastcc i32 @benchmark_softmax_compute_(ptr nonnull %61, ptr nonnull %71, i32 %dev_id), !dbg !15
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
define internal fastcc i32 @benchmark_softmax_compute_(ptr noalias align 64 %arg0, ptr noalias align 64 %arg1, i32 %dev_id) unnamed_addr #3 !dbg !33 {
entry:
    #dbg_value(ptr %arg0, !37, !DIExpression(), !40)
    #dbg_value(ptr %arg1, !38, !DIExpression(), !40)
    #dbg_value(i32 %dev_id, !39, !DIExpression(), !40)
  %0 = alloca %closure_loop_parallel_parallel_0, align 8, !dbg !40
    #dbg_declare(ptr %arg0, !41, !DIExpression(), !40)
  call void @llvm.assume(i1 true) [ "align"(ptr %arg0, i64 64) ], !dbg !40
    #dbg_declare(ptr %arg1, !42, !DIExpression(), !40)
  call void @llvm.assume(i1 true) [ "align"(ptr %arg1, i64 64) ], !dbg !40
  store i32 %dev_id, ptr %0, align 8, !dbg !40
  %1 = getelementptr inbounds nuw i8, ptr %0, i64 8, !dbg !40
  store ptr %arg0, ptr %1, align 8, !dbg !40
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 16, !dbg !40
  store ptr %arg1, ptr %2, align 8, !dbg !40
  %3 = load ptr, ptr @__TVMBackendParallelLaunch, align 8, !dbg !40, !tbaa !43
  %4 = call i32 %3(ptr nonnull @__tvm_parallel_lambda, ptr nonnull %0, i32 0), !dbg !40
  ret i32 %4, !dbg !40
}

define private range(i32 -1, 1) i32 @__tvm_parallel_lambda(i32 %task_id, ptr readonly captures(none) %0, ptr readonly captures(none) %1) #0 {
parallel_closure_entry:
  %dev_id = load i32, ptr %1, align 4, !dbg !40
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 8, !dbg !40
  %arg0 = load ptr, ptr %2, align 8, !dbg !40
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 16, !dbg !40
  %arg1 = load ptr, ptr %3, align 8, !dbg !40
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 8, !dbg !40
  %num_task = load i32, ptr %4, align 4, !dbg !40
  %5 = add nsw i32 %num_task, 16, !dbg !40
  %6 = sdiv i32 %5, %num_task, !dbg !40
  %7 = mul nsw i32 %6, %task_id, !dbg !40
  %8 = add nsw i32 %task_id, 1, !dbg !40
  %9 = mul nsw i32 %6, %8, !dbg !40
  %10 = tail call i32 @llvm.smin.i32(i32 %9, i32 17), !dbg !40
    #dbg_value(i32 poison, !46, !DIExpression(), !40)
  %11 = icmp slt i32 %7, %10, !dbg !40
  br i1 %11, label %for_body_parallel_0.preheader, label %common.ret, !dbg !40, !prof !47

for_body_parallel_0.preheader:                    ; preds = %parallel_closure_entry
    #dbg_value(i32 %7, !46, !DIExpression(), !40)
  %12 = sext i32 %7 to i64, !dbg !40
  br label %for_body_parallel_0, !dbg !40

for_begin_parallel_0:                             ; preds = %if_end41
  %indvars.iv.next93 = add nsw i64 %indvars.iv92, 1, !dbg !40
    #dbg_value(i64 %indvars.iv.next93, !46, !DIExpression(), !40)
  %lftr.wideiv = trunc i64 %indvars.iv.next93 to i32, !dbg !40
  %exitcond96.not = icmp eq i32 %10, %lftr.wideiv, !dbg !40
  br i1 %exitcond96.not, label %common.ret, label %for_body_parallel_0, !dbg !40, !prof !48

for_body_parallel_0:                              ; preds = %for_body_parallel_0.preheader, %for_begin_parallel_0
  %indvars.iv92 = phi i64 [ %12, %for_body_parallel_0.preheader ], [ %indvars.iv.next93, %for_begin_parallel_0 ]
    #dbg_value(i64 %indvars.iv92, !46, !DIExpression(), !40)
  %13 = load ptr, ptr @__TVMBackendAllocWorkspace, align 8, !dbg !40, !tbaa !43
  %tile_storage_0 = tail call ptr %13(i32 1, i32 %dev_id, i64 1028, i32 2, i32 32), !dbg !40
    #dbg_declare(ptr %tile_storage_0, !49, !DIExpression(), !40)
  %14 = icmp eq ptr %tile_storage_0, null, !dbg !40
  br i1 %14, label %common.ret, label %if_end, !dbg !40, !prof !16

common.ret:                                       ; preds = %for_begin_parallel_0, %for_body_parallel_0, %if_end, %if_end2, %for_end_n_5_0, %for_end_tile_i_9_pack, %if_end17, %for_end_n_18_0, %if_end29, %if_end32, %if_end35, %if_end38, %if_end41, %if_end8, %for_body_n_5_0, %if_end23, %for_body_n_18_0, %parallel_closure_entry
  %common.ret.op = phi i32 [ 0, %parallel_closure_entry ], [ -1, %for_body_n_18_0 ], [ -1, %if_end23 ], [ -1, %for_body_n_5_0 ], [ -1, %if_end8 ], [ 0, %for_begin_parallel_0 ], [ -1, %for_body_parallel_0 ], [ -1, %if_end ], [ -1, %if_end2 ], [ -1, %for_end_n_5_0 ], [ -1, %for_end_tile_i_9_pack ], [ -1, %if_end17 ], [ -1, %for_end_n_18_0 ], [ -1, %if_end29 ], [ -1, %if_end32 ], [ -1, %if_end35 ], [ -1, %if_end38 ], [ -1, %if_end41 ]
  ret i32 %common.ret.op, !dbg !40

if_end:                                           ; preds = %for_body_parallel_0
  %15 = mul nsw i64 %indvars.iv92, 257, !dbg !40
    #dbg_value(i64 %15, !50, !DIExpression(), !40)
    #dbg_value(i64 0, !51, !DIExpression(), !40)
    #dbg_value(i64 0, !52, !DIExpression(), !40)
  %16 = getelementptr inbounds float, ptr %arg0, i64 %15, !dbg !40
  %17 = load <16 x float>, ptr %16, align 4, !dbg !40, !tbaa !53
  store <16 x float> %17, ptr %tile_storage_0, align 4, !dbg !40, !tbaa !55
    #dbg_value(i64 1, !51, !DIExpression(), !40)
    #dbg_value(i64 16, !52, !DIExpression(), !40)
  %18 = add nsw i64 %15, 16, !dbg !40
  %19 = getelementptr inbounds float, ptr %arg0, i64 %18, !dbg !40
  %20 = load <16 x float>, ptr %19, align 4, !dbg !40, !tbaa !53
  %21 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 64, !dbg !40
  store <16 x float> %20, ptr %21, align 4, !dbg !40, !tbaa !55
    #dbg_value(i64 2, !51, !DIExpression(), !40)
    #dbg_value(i64 32, !52, !DIExpression(), !40)
  %22 = add nsw i64 %15, 32, !dbg !40
  %23 = getelementptr inbounds float, ptr %arg0, i64 %22, !dbg !40
  %24 = load <16 x float>, ptr %23, align 4, !dbg !40, !tbaa !53
  %25 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 128, !dbg !40
  store <16 x float> %24, ptr %25, align 4, !dbg !40, !tbaa !55
    #dbg_value(i64 3, !51, !DIExpression(), !40)
    #dbg_value(i64 48, !52, !DIExpression(), !40)
  %26 = add nsw i64 %15, 48, !dbg !40
  %27 = getelementptr inbounds float, ptr %arg0, i64 %26, !dbg !40
  %28 = load <16 x float>, ptr %27, align 4, !dbg !40, !tbaa !53
  %29 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 192, !dbg !40
  store <16 x float> %28, ptr %29, align 4, !dbg !40, !tbaa !55
    #dbg_value(i64 4, !51, !DIExpression(), !40)
    #dbg_value(i64 64, !52, !DIExpression(), !40)
  %30 = add nsw i64 %15, 64, !dbg !40
  %31 = getelementptr inbounds float, ptr %arg0, i64 %30, !dbg !40
  %32 = load <16 x float>, ptr %31, align 4, !dbg !40, !tbaa !53
  %33 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 256, !dbg !40
  store <16 x float> %32, ptr %33, align 4, !dbg !40, !tbaa !55
    #dbg_value(i64 5, !51, !DIExpression(), !40)
    #dbg_value(i64 80, !52, !DIExpression(), !40)
  %34 = add nsw i64 %15, 80, !dbg !40
  %35 = getelementptr inbounds float, ptr %arg0, i64 %34, !dbg !40
  %36 = load <16 x float>, ptr %35, align 4, !dbg !40, !tbaa !53
  %37 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 320, !dbg !40
  store <16 x float> %36, ptr %37, align 4, !dbg !40, !tbaa !55
    #dbg_value(i64 6, !51, !DIExpression(), !40)
    #dbg_value(i64 96, !52, !DIExpression(), !40)
  %38 = add nsw i64 %15, 96, !dbg !40
  %39 = getelementptr inbounds float, ptr %arg0, i64 %38, !dbg !40
  %40 = load <16 x float>, ptr %39, align 4, !dbg !40, !tbaa !53
  %41 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 384, !dbg !40
  store <16 x float> %40, ptr %41, align 4, !dbg !40, !tbaa !55
    #dbg_value(i64 7, !51, !DIExpression(), !40)
    #dbg_value(i64 112, !52, !DIExpression(), !40)
  %42 = add nsw i64 %15, 112, !dbg !40
  %43 = getelementptr inbounds float, ptr %arg0, i64 %42, !dbg !40
  %44 = load <16 x float>, ptr %43, align 4, !dbg !40, !tbaa !53
  %45 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 448, !dbg !40
  store <16 x float> %44, ptr %45, align 4, !dbg !40, !tbaa !55
    #dbg_value(i64 8, !51, !DIExpression(), !40)
    #dbg_value(i64 128, !52, !DIExpression(), !40)
  %46 = add nsw i64 %15, 128, !dbg !40
  %47 = getelementptr inbounds float, ptr %arg0, i64 %46, !dbg !40
  %48 = load <16 x float>, ptr %47, align 4, !dbg !40, !tbaa !53
  %49 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 512, !dbg !40
  store <16 x float> %48, ptr %49, align 4, !dbg !40, !tbaa !55
    #dbg_value(i64 9, !51, !DIExpression(), !40)
    #dbg_value(i64 144, !52, !DIExpression(), !40)
  %50 = add nsw i64 %15, 144, !dbg !40
  %51 = getelementptr inbounds float, ptr %arg0, i64 %50, !dbg !40
  %52 = load <16 x float>, ptr %51, align 4, !dbg !40, !tbaa !53
  %53 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 576, !dbg !40
  store <16 x float> %52, ptr %53, align 4, !dbg !40, !tbaa !55
    #dbg_value(i64 10, !51, !DIExpression(), !40)
    #dbg_value(i64 160, !52, !DIExpression(), !40)
  %54 = add nsw i64 %15, 160, !dbg !40
  %55 = getelementptr inbounds float, ptr %arg0, i64 %54, !dbg !40
  %56 = load <16 x float>, ptr %55, align 4, !dbg !40, !tbaa !53
  %57 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 640, !dbg !40
  store <16 x float> %56, ptr %57, align 4, !dbg !40, !tbaa !55
    #dbg_value(i64 11, !51, !DIExpression(), !40)
    #dbg_value(i64 176, !52, !DIExpression(), !40)
  %58 = add nsw i64 %15, 176, !dbg !40
  %59 = getelementptr inbounds float, ptr %arg0, i64 %58, !dbg !40
  %60 = load <16 x float>, ptr %59, align 4, !dbg !40, !tbaa !53
  %61 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 704, !dbg !40
  store <16 x float> %60, ptr %61, align 4, !dbg !40, !tbaa !55
    #dbg_value(i64 12, !51, !DIExpression(), !40)
    #dbg_value(i64 192, !52, !DIExpression(), !40)
  %62 = add nsw i64 %15, 192, !dbg !40
  %63 = getelementptr inbounds float, ptr %arg0, i64 %62, !dbg !40
  %64 = load <16 x float>, ptr %63, align 4, !dbg !40, !tbaa !53
  %65 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 768, !dbg !40
  store <16 x float> %64, ptr %65, align 4, !dbg !40, !tbaa !55
    #dbg_value(i64 13, !51, !DIExpression(), !40)
    #dbg_value(i64 208, !52, !DIExpression(), !40)
  %66 = add nsw i64 %15, 208, !dbg !40
  %67 = getelementptr inbounds float, ptr %arg0, i64 %66, !dbg !40
  %68 = load <16 x float>, ptr %67, align 4, !dbg !40, !tbaa !53
  %69 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 832, !dbg !40
  store <16 x float> %68, ptr %69, align 4, !dbg !40, !tbaa !55
    #dbg_value(i64 14, !51, !DIExpression(), !40)
    #dbg_value(i64 224, !52, !DIExpression(), !40)
  %70 = add nsw i64 %15, 224, !dbg !40
  %71 = getelementptr inbounds float, ptr %arg0, i64 %70, !dbg !40
  %72 = load <16 x float>, ptr %71, align 4, !dbg !40, !tbaa !53
  %73 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 896, !dbg !40
  store <16 x float> %72, ptr %73, align 4, !dbg !40, !tbaa !55
    #dbg_value(i64 15, !51, !DIExpression(), !40)
    #dbg_value(i64 240, !52, !DIExpression(), !40)
  %74 = add nsw i64 %15, 240, !dbg !40
  %75 = getelementptr inbounds float, ptr %arg0, i64 %74, !dbg !40
  %76 = load <16 x float>, ptr %75, align 4, !dbg !40, !tbaa !53
  %77 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 960, !dbg !40
  store <16 x float> %76, ptr %77, align 4, !dbg !40, !tbaa !55
    #dbg_value(i64 16, !51, !DIExpression(), !40)
  %78 = add nsw i64 %15, 256, !dbg !40
    #dbg_value(i64 %78, !57, !DIExpression(), !40)
  %79 = getelementptr inbounds float, ptr %arg0, i64 %78, !dbg !40
  %80 = load float, ptr %79, align 4, !dbg !40, !tbaa !53
  %81 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 1024, !dbg !40
  store float %80, ptr %81, align 4, !dbg !40, !tbaa !58
  %82 = load ptr, ptr @__TVMBackendAllocWorkspace, align 8, !dbg !40, !tbaa !43
  %tile_storage_3 = tail call ptr %82(i32 1, i32 %dev_id, i64 4, i32 2, i32 32), !dbg !40
    #dbg_declare(ptr %tile_storage_3, !68, !DIExpression(), !40)
  %83 = icmp eq ptr %tile_storage_3, null, !dbg !40
  br i1 %83, label %common.ret, label %if_end2, !dbg !40, !prof !16

if_end2:                                          ; preds = %if_end
  %84 = load ptr, ptr @__TVMBackendAllocWorkspace, align 8, !dbg !40, !tbaa !43
  %tile_storage_5 = tail call ptr %84(i32 1, i32 %dev_id, i64 4, i32 2, i32 32), !dbg !40
    #dbg_declare(ptr %tile_storage_5, !69, !DIExpression(), !40)
  %85 = icmp eq ptr %tile_storage_5, null, !dbg !40
  br i1 %85, label %common.ret, label %if_end5, !dbg !40, !prof !16

if_end5:                                          ; preds = %if_end2
  store float 0xFFF0000000000000, ptr %tile_storage_5, align 4, !dbg !40, !tbaa !70
    #dbg_value(i32 0, !81, !DIExpression(), !40)
  br label %for_body_n_5_0, !dbg !40

for_begin_n_5_0:                                  ; preds = %if_end8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !40
    #dbg_value(i32 poison, !81, !DIExpression(), !40)
  %exitcond.not = icmp eq i64 %indvars.iv.next, 257, !dbg !40
  br i1 %exitcond.not, label %for_end_n_5_0, label %for_body_n_5_0, !dbg !40, !prof !82

for_body_n_5_0:                                   ; preds = %if_end5, %for_begin_n_5_0
  %indvars.iv = phi i64 [ 0, %if_end5 ], [ %indvars.iv.next, %for_begin_n_5_0 ]
    #dbg_value(i64 %indvars.iv, !81, !DIExpression(), !40)
  %86 = load ptr, ptr @__TVMBackendAllocWorkspace, align 8, !dbg !40, !tbaa !43
  %tile_storage_6 = tail call ptr %86(i32 1, i32 %dev_id, i64 4, i32 2, i32 32), !dbg !40
    #dbg_declare(ptr %tile_storage_6, !83, !DIExpression(), !40)
  %87 = icmp eq ptr %tile_storage_6, null, !dbg !40
  br i1 %87, label %common.ret, label %if_end8, !dbg !40, !prof !16

for_end_n_5_0:                                    ; preds = %for_begin_n_5_0
  %88 = load float, ptr %tile_storage_5, align 4, !dbg !40, !tbaa !70
  store float %88, ptr %tile_storage_3, align 4, !dbg !40, !tbaa !84
  %89 = load ptr, ptr @__TVMBackendAllocWorkspace, align 8, !dbg !40, !tbaa !43
  %tile_storage_7 = tail call ptr %89(i32 1, i32 %dev_id, i64 1028, i32 2, i32 32), !dbg !40
    #dbg_declare(ptr %tile_storage_7, !95, !DIExpression(), !40)
  %90 = icmp eq ptr %tile_storage_7, null, !dbg !40
  br i1 %90, label %common.ret, label %for_begin_tile_i_9_pack.preheader, !dbg !40, !prof !16

for_begin_tile_i_9_pack.preheader:                ; preds = %for_end_n_5_0
  %91 = load float, ptr %tile_storage_3, align 4, !tbaa !84
  %92 = insertelement <16 x float> poison, float %91, i64 0
  %93 = shufflevector <16 x float> %92, <16 x float> poison, <16 x i32> zeroinitializer
    #dbg_value(i32 0, !96, !DIExpression(), !40)
  br label %for_body_tile_i_9_pack, !dbg !40

if_end8:                                          ; preds = %for_body_n_5_0
  %94 = load float, ptr %tile_storage_5, align 4, !dbg !40, !tbaa !70
  %95 = getelementptr inbounds nuw float, ptr %tile_storage_0, i64 %indvars.iv, !dbg !40
  %96 = load float, ptr %95, align 4, !dbg !40, !tbaa !55
  %97 = fcmp ogt float %94, %96, !dbg !40
  %98 = select i1 %97, float %94, float %96, !dbg !40
  store float %98, ptr %tile_storage_6, align 4, !dbg !40, !tbaa !97
  store float %98, ptr %tile_storage_5, align 4, !dbg !40, !tbaa !70
  %99 = load ptr, ptr @__TVMBackendFreeWorkspace, align 8, !dbg !40, !tbaa !43
  %100 = tail call i32 %99(i32 1, i32 %dev_id, ptr nonnull %tile_storage_6), !dbg !40
  %.not52 = icmp eq i32 %100, 0, !dbg !40
  br i1 %.not52, label %for_begin_n_5_0, label %common.ret, !dbg !40, !prof !17

for_body_tile_i_9_pack:                           ; preds = %for_begin_tile_i_9_pack.preheader, %for_body_tile_i_9_pack
  %indvars.iv78 = phi i64 [ 0, %for_begin_tile_i_9_pack.preheader ], [ %indvars.iv.next79, %for_body_tile_i_9_pack ]
    #dbg_value(i64 %indvars.iv78, !96, !DIExpression(), !40)
  %101 = shl nuw nsw i64 %indvars.iv78, 4, !dbg !40
    #dbg_value(i64 %101, !108, !DIExpression(), !40)
  %102 = getelementptr inbounds nuw float, ptr %tile_storage_0, i64 %101, !dbg !40
  %103 = load <16 x float>, ptr %102, align 4, !dbg !40, !tbaa !55
  %104 = fsub <16 x float> %103, %93, !dbg !40
  %105 = tail call <16 x float> @llvm.exp.v16f32(<16 x float> %104), !dbg !40
  %106 = getelementptr inbounds nuw float, ptr %tile_storage_7, i64 %101, !dbg !40
  store <16 x float> %105, ptr %106, align 4, !dbg !40, !tbaa !109
  %indvars.iv.next79 = add nuw nsw i64 %indvars.iv78, 1, !dbg !40
    #dbg_value(i64 %indvars.iv.next79, !96, !DIExpression(), !40)
  %exitcond82.not = icmp eq i64 %indvars.iv.next79, 16, !dbg !40
  br i1 %exitcond82.not, label %for_end_tile_i_9_pack, label %for_body_tile_i_9_pack, !dbg !40, !prof !82

for_end_tile_i_9_pack:                            ; preds = %for_body_tile_i_9_pack
  %107 = load float, ptr %81, align 4, !dbg !40, !tbaa !58
  %108 = fsub float %107, %91, !dbg !40
  %109 = tail call float @llvm.exp.f32(float %108), !dbg !40
  %110 = getelementptr inbounds nuw i8, ptr %tile_storage_7, i64 1024, !dbg !40
  store float %109, ptr %110, align 4, !dbg !40, !tbaa !111
  %111 = load ptr, ptr @__TVMBackendAllocWorkspace, align 8, !dbg !40, !tbaa !43
  %tile_storage_10 = tail call ptr %111(i32 1, i32 %dev_id, i64 4, i32 2, i32 32), !dbg !40
    #dbg_declare(ptr %tile_storage_10, !121, !DIExpression(), !40)
  %112 = icmp eq ptr %tile_storage_10, null, !dbg !40
  br i1 %112, label %common.ret, label %if_end17, !dbg !40, !prof !16

if_end17:                                         ; preds = %for_end_tile_i_9_pack
  %113 = load ptr, ptr @__TVMBackendAllocWorkspace, align 8, !dbg !40, !tbaa !43
  %tile_storage_12 = tail call ptr %113(i32 1, i32 %dev_id, i64 4, i32 2, i32 32), !dbg !40
    #dbg_declare(ptr %tile_storage_12, !122, !DIExpression(), !40)
  %114 = icmp eq ptr %tile_storage_12, null, !dbg !40
  br i1 %114, label %common.ret, label %if_end20, !dbg !40, !prof !16

if_end20:                                         ; preds = %if_end17
  store float 0.000000e+00, ptr %tile_storage_12, align 4, !dbg !40, !tbaa !123
    #dbg_value(i32 0, !134, !DIExpression(), !40)
  br label %for_body_n_18_0, !dbg !40

for_begin_n_18_0:                                 ; preds = %if_end23
  %indvars.iv.next84 = add nuw nsw i64 %indvars.iv83, 1, !dbg !40
    #dbg_value(i32 poison, !134, !DIExpression(), !40)
  %exitcond86.not = icmp eq i64 %indvars.iv.next84, 257, !dbg !40
  br i1 %exitcond86.not, label %for_end_n_18_0, label %for_body_n_18_0, !dbg !40, !prof !82

for_body_n_18_0:                                  ; preds = %if_end20, %for_begin_n_18_0
  %indvars.iv83 = phi i64 [ 0, %if_end20 ], [ %indvars.iv.next84, %for_begin_n_18_0 ]
    #dbg_value(i64 %indvars.iv83, !134, !DIExpression(), !40)
  %115 = load ptr, ptr @__TVMBackendAllocWorkspace, align 8, !dbg !40, !tbaa !43
  %tile_storage_13 = tail call ptr %115(i32 1, i32 %dev_id, i64 4, i32 2, i32 32), !dbg !40
    #dbg_declare(ptr %tile_storage_13, !135, !DIExpression(), !40)
  %116 = icmp eq ptr %tile_storage_13, null, !dbg !40
  br i1 %116, label %common.ret, label %if_end23, !dbg !40, !prof !16

for_end_n_18_0:                                   ; preds = %for_begin_n_18_0
  %117 = load float, ptr %tile_storage_12, align 4, !dbg !40, !tbaa !123
  store float %117, ptr %tile_storage_10, align 4, !dbg !40, !tbaa !136
  %118 = insertelement <16 x float> poison, float %117, i64 0
  %119 = shufflevector <16 x float> %118, <16 x float> poison, <16 x i32> zeroinitializer
    #dbg_value(i64 0, !147, !DIExpression(), !40)
    #dbg_value(i64 0, !148, !DIExpression(), !40)
  %120 = load <16 x float>, ptr %tile_storage_7, align 4, !dbg !40, !tbaa !109
  %121 = fdiv <16 x float> %120, %119, !dbg !40
  %122 = getelementptr inbounds float, ptr %arg1, i64 %15, !dbg !40
  store <16 x float> %121, ptr %122, align 4, !dbg !40, !tbaa !149
    #dbg_value(i64 1, !147, !DIExpression(), !40)
    #dbg_value(i64 16, !148, !DIExpression(), !40)
  %123 = getelementptr inbounds nuw i8, ptr %tile_storage_7, i64 64, !dbg !40
  %124 = load <16 x float>, ptr %123, align 4, !dbg !40, !tbaa !109
  %125 = fdiv <16 x float> %124, %119, !dbg !40
  %126 = getelementptr inbounds float, ptr %arg1, i64 %18, !dbg !40
  store <16 x float> %125, ptr %126, align 4, !dbg !40, !tbaa !149
    #dbg_value(i64 2, !147, !DIExpression(), !40)
    #dbg_value(i64 32, !148, !DIExpression(), !40)
  %127 = getelementptr inbounds nuw i8, ptr %tile_storage_7, i64 128, !dbg !40
  %128 = load <16 x float>, ptr %127, align 4, !dbg !40, !tbaa !109
  %129 = fdiv <16 x float> %128, %119, !dbg !40
  %130 = getelementptr inbounds float, ptr %arg1, i64 %22, !dbg !40
  store <16 x float> %129, ptr %130, align 4, !dbg !40, !tbaa !149
    #dbg_value(i64 3, !147, !DIExpression(), !40)
    #dbg_value(i64 48, !148, !DIExpression(), !40)
  %131 = getelementptr inbounds nuw i8, ptr %tile_storage_7, i64 192, !dbg !40
  %132 = load <16 x float>, ptr %131, align 4, !dbg !40, !tbaa !109
  %133 = fdiv <16 x float> %132, %119, !dbg !40
  %134 = getelementptr inbounds float, ptr %arg1, i64 %26, !dbg !40
  store <16 x float> %133, ptr %134, align 4, !dbg !40, !tbaa !149
    #dbg_value(i64 4, !147, !DIExpression(), !40)
    #dbg_value(i64 64, !148, !DIExpression(), !40)
  %135 = getelementptr inbounds nuw i8, ptr %tile_storage_7, i64 256, !dbg !40
  %136 = load <16 x float>, ptr %135, align 4, !dbg !40, !tbaa !109
  %137 = fdiv <16 x float> %136, %119, !dbg !40
  %138 = getelementptr inbounds float, ptr %arg1, i64 %30, !dbg !40
  store <16 x float> %137, ptr %138, align 4, !dbg !40, !tbaa !149
    #dbg_value(i64 5, !147, !DIExpression(), !40)
    #dbg_value(i64 80, !148, !DIExpression(), !40)
  %139 = getelementptr inbounds nuw i8, ptr %tile_storage_7, i64 320, !dbg !40
  %140 = load <16 x float>, ptr %139, align 4, !dbg !40, !tbaa !109
  %141 = fdiv <16 x float> %140, %119, !dbg !40
  %142 = getelementptr inbounds float, ptr %arg1, i64 %34, !dbg !40
  store <16 x float> %141, ptr %142, align 4, !dbg !40, !tbaa !149
    #dbg_value(i64 6, !147, !DIExpression(), !40)
    #dbg_value(i64 96, !148, !DIExpression(), !40)
  %143 = getelementptr inbounds nuw i8, ptr %tile_storage_7, i64 384, !dbg !40
  %144 = load <16 x float>, ptr %143, align 4, !dbg !40, !tbaa !109
  %145 = fdiv <16 x float> %144, %119, !dbg !40
  %146 = getelementptr inbounds float, ptr %arg1, i64 %38, !dbg !40
  store <16 x float> %145, ptr %146, align 4, !dbg !40, !tbaa !149
    #dbg_value(i64 7, !147, !DIExpression(), !40)
    #dbg_value(i64 112, !148, !DIExpression(), !40)
  %147 = getelementptr inbounds nuw i8, ptr %tile_storage_7, i64 448, !dbg !40
  %148 = load <16 x float>, ptr %147, align 4, !dbg !40, !tbaa !109
  %149 = fdiv <16 x float> %148, %119, !dbg !40
  %150 = getelementptr inbounds float, ptr %arg1, i64 %42, !dbg !40
  store <16 x float> %149, ptr %150, align 4, !dbg !40, !tbaa !149
    #dbg_value(i64 8, !147, !DIExpression(), !40)
    #dbg_value(i64 128, !148, !DIExpression(), !40)
  %151 = getelementptr inbounds nuw i8, ptr %tile_storage_7, i64 512, !dbg !40
  %152 = load <16 x float>, ptr %151, align 4, !dbg !40, !tbaa !109
  %153 = fdiv <16 x float> %152, %119, !dbg !40
  %154 = getelementptr inbounds float, ptr %arg1, i64 %46, !dbg !40
  store <16 x float> %153, ptr %154, align 4, !dbg !40, !tbaa !149
    #dbg_value(i64 9, !147, !DIExpression(), !40)
    #dbg_value(i64 144, !148, !DIExpression(), !40)
  %155 = getelementptr inbounds nuw i8, ptr %tile_storage_7, i64 576, !dbg !40
  %156 = load <16 x float>, ptr %155, align 4, !dbg !40, !tbaa !109
  %157 = fdiv <16 x float> %156, %119, !dbg !40
  %158 = getelementptr inbounds float, ptr %arg1, i64 %50, !dbg !40
  store <16 x float> %157, ptr %158, align 4, !dbg !40, !tbaa !149
    #dbg_value(i64 10, !147, !DIExpression(), !40)
    #dbg_value(i64 160, !148, !DIExpression(), !40)
  %159 = getelementptr inbounds nuw i8, ptr %tile_storage_7, i64 640, !dbg !40
  %160 = load <16 x float>, ptr %159, align 4, !dbg !40, !tbaa !109
  %161 = fdiv <16 x float> %160, %119, !dbg !40
  %162 = getelementptr inbounds float, ptr %arg1, i64 %54, !dbg !40
  store <16 x float> %161, ptr %162, align 4, !dbg !40, !tbaa !149
    #dbg_value(i64 11, !147, !DIExpression(), !40)
    #dbg_value(i64 176, !148, !DIExpression(), !40)
  %163 = getelementptr inbounds nuw i8, ptr %tile_storage_7, i64 704, !dbg !40
  %164 = load <16 x float>, ptr %163, align 4, !dbg !40, !tbaa !109
  %165 = fdiv <16 x float> %164, %119, !dbg !40
  %166 = getelementptr inbounds float, ptr %arg1, i64 %58, !dbg !40
  store <16 x float> %165, ptr %166, align 4, !dbg !40, !tbaa !149
    #dbg_value(i64 12, !147, !DIExpression(), !40)
    #dbg_value(i64 192, !148, !DIExpression(), !40)
  %167 = getelementptr inbounds nuw i8, ptr %tile_storage_7, i64 768, !dbg !40
  %168 = load <16 x float>, ptr %167, align 4, !dbg !40, !tbaa !109
  %169 = fdiv <16 x float> %168, %119, !dbg !40
  %170 = getelementptr inbounds float, ptr %arg1, i64 %62, !dbg !40
  store <16 x float> %169, ptr %170, align 4, !dbg !40, !tbaa !149
    #dbg_value(i64 13, !147, !DIExpression(), !40)
    #dbg_value(i64 208, !148, !DIExpression(), !40)
  %171 = getelementptr inbounds nuw i8, ptr %tile_storage_7, i64 832, !dbg !40
  %172 = load <16 x float>, ptr %171, align 4, !dbg !40, !tbaa !109
  %173 = fdiv <16 x float> %172, %119, !dbg !40
  %174 = getelementptr inbounds float, ptr %arg1, i64 %66, !dbg !40
  store <16 x float> %173, ptr %174, align 4, !dbg !40, !tbaa !149
    #dbg_value(i64 14, !147, !DIExpression(), !40)
    #dbg_value(i64 224, !148, !DIExpression(), !40)
  %175 = getelementptr inbounds nuw i8, ptr %tile_storage_7, i64 896, !dbg !40
  %176 = load <16 x float>, ptr %175, align 4, !dbg !40, !tbaa !109
  %177 = fdiv <16 x float> %176, %119, !dbg !40
  %178 = getelementptr inbounds float, ptr %arg1, i64 %70, !dbg !40
  store <16 x float> %177, ptr %178, align 4, !dbg !40, !tbaa !149
    #dbg_value(i64 15, !147, !DIExpression(), !40)
    #dbg_value(i64 240, !148, !DIExpression(), !40)
  %179 = getelementptr inbounds nuw i8, ptr %tile_storage_7, i64 960, !dbg !40
  %180 = load <16 x float>, ptr %179, align 4, !dbg !40, !tbaa !109
  %181 = fdiv <16 x float> %180, %119, !dbg !40
  %182 = getelementptr inbounds float, ptr %arg1, i64 %74, !dbg !40
  store <16 x float> %181, ptr %182, align 4, !dbg !40, !tbaa !149
    #dbg_value(i64 16, !147, !DIExpression(), !40)
  %183 = load float, ptr %110, align 4, !dbg !40, !tbaa !111
  %184 = fdiv float %183, %117, !dbg !40
  %185 = getelementptr inbounds float, ptr %arg1, i64 %78, !dbg !40
  store float %184, ptr %185, align 4, !dbg !40, !tbaa !149
  %186 = load ptr, ptr @__TVMBackendFreeWorkspace, align 8, !dbg !40, !tbaa !43
  %187 = tail call i32 %186(i32 1, i32 %dev_id, ptr nonnull %tile_storage_12), !dbg !40
  %.not = icmp eq i32 %187, 0, !dbg !40
  br i1 %.not, label %if_end29, label %common.ret, !dbg !40, !prof !17

if_end23:                                         ; preds = %for_body_n_18_0
  %188 = load float, ptr %tile_storage_12, align 4, !dbg !40, !tbaa !123
  %189 = getelementptr inbounds nuw float, ptr %tile_storage_7, i64 %indvars.iv83, !dbg !40
  %190 = load float, ptr %189, align 4, !dbg !40, !tbaa !109
  %191 = fadd float %188, %190, !dbg !40
  store float %191, ptr %tile_storage_13, align 4, !dbg !40, !tbaa !151
  store float %191, ptr %tile_storage_12, align 4, !dbg !40, !tbaa !123
  %192 = load ptr, ptr @__TVMBackendFreeWorkspace, align 8, !dbg !40, !tbaa !43
  %193 = tail call i32 %192(i32 1, i32 %dev_id, ptr nonnull %tile_storage_13), !dbg !40
  %.not51 = icmp eq i32 %193, 0, !dbg !40
  br i1 %.not51, label %for_begin_n_18_0, label %common.ret, !dbg !40, !prof !17

if_end29:                                         ; preds = %for_end_n_18_0
  %194 = load ptr, ptr @__TVMBackendFreeWorkspace, align 8, !dbg !40, !tbaa !43
  %195 = tail call i32 %194(i32 1, i32 %dev_id, ptr nonnull %tile_storage_10), !dbg !40
  %.not46 = icmp eq i32 %195, 0, !dbg !40
  br i1 %.not46, label %if_end32, label %common.ret, !dbg !40, !prof !17

if_end32:                                         ; preds = %if_end29
  %196 = load ptr, ptr @__TVMBackendFreeWorkspace, align 8, !dbg !40, !tbaa !43
  %197 = tail call i32 %196(i32 1, i32 %dev_id, ptr nonnull %tile_storage_7), !dbg !40
  %.not47 = icmp eq i32 %197, 0, !dbg !40
  br i1 %.not47, label %if_end35, label %common.ret, !dbg !40, !prof !17

if_end35:                                         ; preds = %if_end32
  %198 = load ptr, ptr @__TVMBackendFreeWorkspace, align 8, !dbg !40, !tbaa !43
  %199 = tail call i32 %198(i32 1, i32 %dev_id, ptr nonnull %tile_storage_5), !dbg !40
  %.not48 = icmp eq i32 %199, 0, !dbg !40
  br i1 %.not48, label %if_end38, label %common.ret, !dbg !40, !prof !17

if_end38:                                         ; preds = %if_end35
  %200 = load ptr, ptr @__TVMBackendFreeWorkspace, align 8, !dbg !40, !tbaa !43
  %201 = tail call i32 %200(i32 1, i32 %dev_id, ptr nonnull %tile_storage_3), !dbg !40
  %.not49 = icmp eq i32 %201, 0, !dbg !40
  br i1 %.not49, label %if_end41, label %common.ret, !dbg !40, !prof !17

if_end41:                                         ; preds = %if_end38
  %202 = load ptr, ptr @__TVMBackendFreeWorkspace, align 8, !dbg !40, !tbaa !43
  %203 = tail call i32 %202(i32 1, i32 %dev_id, ptr nonnull %tile_storage_0), !dbg !40
  %.not50 = icmp eq i32 %203, 0, !dbg !40
  br i1 %.not50, label %for_begin_parallel_0, label %common.ret, !dbg !40, !prof !17
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <16 x float> @llvm.exp.v16f32(<16 x float>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.exp.f32(float) #4

define weak dllexport i32 @__tvm_ffi_main(ptr %0, ptr %1, i32 %2, ptr %3) local_unnamed_addr {
entry:
  %4 = tail call i32 @__tvm_ffi_benchmark_softmax(ptr poison, ptr %1, i32 %2, ptr poison), !dbg !15
  ret i32 %4, !dbg !15
}

; Function Attrs: nofree nosync nounwind memory(none)
define weak dso_local i16 @__truncsfhf2(float %a0) local_unnamed_addr #5 {
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
define weak dso_local float @__extendhfsf2(i16 %a0) local_unnamed_addr #5 {
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
declare i32 @llvm.smin.i32(i32, i32) #6

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #6

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #7

attributes #0 = { "target-cpu"="generic" }
attributes #1 = { noinline }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #3 = { noinline "target-cpu"="generic" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { nofree nosync nounwind memory(none) "target-cpu"="generic" "target-features" }
attributes #6 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #7 = { nocallback nofree nounwind willreturn memory(argmem: write) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "TVM", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "IRModule.CodeGenLLVM", directory: ".")
!2 = !{i32 2, !"tvm_target", !"{\22kind\22:\22llvm\22,\22mtriple\22:\22arm64-apple-darwin25.6.0\22}"}
!3 = !{i32 4, !"Debug Info Version", i32 3}
!4 = !{i32 4, !"Dwarf Version", i32 2}
!5 = distinct !DISubprogram(name: "__tvm_ffi_benchmark_softmax", scope: !1, file: !1, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
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
!22 = !DILocalVariable(name: "benchmark_softmax.arg0_shape", scope: !5, file: !1, type: !23)
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !24)
!24 = !DIBasicType(name: "int64", size: 64, encoding: DW_ATE_signed)
!25 = !DILocalVariable(name: "benchmark_softmax.arg0_strides", scope: !5, file: !1, type: !23)
!26 = !DILocalVariable(name: "dev_id", scope: !5, file: !1, type: !8)
!27 = !DILocalVariable(name: "benchmark_softmax.arg1_shape", scope: !5, file: !1, type: !23)
!28 = !DILocalVariable(name: "benchmark_softmax.arg1_strides", scope: !5, file: !1, type: !23)
!29 = !DILocalVariable(name: "arg0", scope: !5, file: !1, type: !30)
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !31)
!31 = !DIBasicType(name: "float32", size: 32, encoding: DW_ATE_float)
!32 = !DILocalVariable(name: "arg1", scope: !5, file: !1, type: !30)
!33 = distinct !DISubprogram(name: "benchmark_softmax_compute_", scope: !1, file: !1, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !36)
!34 = !DISubroutineType(types: !35)
!35 = !{!8, !30, !30, !8}
!36 = !{!37, !38, !39}
!37 = !DILocalVariable(name: "arg0", arg: 1, scope: !33, file: !1, type: !30)
!38 = !DILocalVariable(name: "arg1", arg: 2, scope: !33, file: !1, type: !30)
!39 = !DILocalVariable(name: "dev_id", arg: 3, scope: !33, file: !1, type: !8)
!40 = !DILocation(line: 0, scope: !33)
!41 = !DILocalVariable(name: "arg0", scope: !33, file: !1, type: !30)
!42 = !DILocalVariable(name: "arg1", scope: !33, file: !1, type: !30)
!43 = !{!44, !44, i64 0}
!44 = !{!"ctx_ptr", !45, i64 0}
!45 = !{!"tvm-tbaa"}
!46 = !DILocalVariable(name: "parallel_0", scope: !33, file: !1, type: !8)
!47 = !{!"branch_weights", i32 127, i32 1}
!48 = !{!"branch_weights", i32 127, i32 134217601}
!49 = !DILocalVariable(name: "tile_storage_0", scope: !33, file: !1, type: !30)
!50 = !DILocalVariable(name: "cse_v1", scope: !33, file: !1, type: !8)
!51 = !DILocalVariable(name: "tile_i_2_pack", scope: !33, file: !1, type: !8)
!52 = !DILocalVariable(name: "cse_v2", scope: !33, file: !1, type: !8)
!53 = !{!54, !54, i64 0}
!54 = !{!"0x8baeb9400", !45, i64 0}
!55 = !{!56, !56, i64 0}
!56 = !{!"0x8ba80ad00", !45, i64 0}
!57 = !DILocalVariable(name: "cse_v5", scope: !33, file: !1, type: !8)
!58 = !{!59, !59, i64 0}
!59 = !{!"0x8ba80ad00.w4.b1024", !60, i64 0}
!60 = !{!"0x8ba80ad00.w8.b1024", !61, i64 0}
!61 = !{!"0x8ba80ad00.w16.b1024", !62, i64 0}
!62 = !{!"0x8ba80ad00.w32.b1024", !63, i64 0}
!63 = !{!"0x8ba80ad00.w64.b1024", !64, i64 0}
!64 = !{!"0x8ba80ad00.w128.b1024", !65, i64 0}
!65 = !{!"0x8ba80ad00.w256.b1024", !66, i64 0}
!66 = !{!"0x8ba80ad00.w512.b1024", !67, i64 0}
!67 = !{!"0x8ba80ad00.w1024.b1024", !56, i64 0}
!68 = !DILocalVariable(name: "tile_storage_3", scope: !33, file: !1, type: !30)
!69 = !DILocalVariable(name: "tile_storage_5", scope: !33, file: !1, type: !30)
!70 = !{!71, !71, i64 0}
!71 = !{!"0x8ba80adc0.w4.b0", !72, i64 0}
!72 = !{!"0x8ba80adc0.w8.b0", !73, i64 0}
!73 = !{!"0x8ba80adc0.w16.b0", !74, i64 0}
!74 = !{!"0x8ba80adc0.w32.b0", !75, i64 0}
!75 = !{!"0x8ba80adc0.w64.b0", !76, i64 0}
!76 = !{!"0x8ba80adc0.w128.b0", !77, i64 0}
!77 = !{!"0x8ba80adc0.w256.b0", !78, i64 0}
!78 = !{!"0x8ba80adc0.w512.b0", !79, i64 0}
!79 = !{!"0x8ba80adc0.w1024.b0", !80, i64 0}
!80 = !{!"0x8ba80adc0", !45, i64 0}
!81 = !DILocalVariable(name: "n_5_0", scope: !33, file: !1, type: !8)
!82 = !{!"branch_weights", i32 1, i32 1048575}
!83 = !DILocalVariable(name: "tile_storage_6", scope: !33, file: !1, type: !30)
!84 = !{!85, !85, i64 0}
!85 = !{!"0x8ba80ae40.w4.b0", !86, i64 0}
!86 = !{!"0x8ba80ae40.w8.b0", !87, i64 0}
!87 = !{!"0x8ba80ae40.w16.b0", !88, i64 0}
!88 = !{!"0x8ba80ae40.w32.b0", !89, i64 0}
!89 = !{!"0x8ba80ae40.w64.b0", !90, i64 0}
!90 = !{!"0x8ba80ae40.w128.b0", !91, i64 0}
!91 = !{!"0x8ba80ae40.w256.b0", !92, i64 0}
!92 = !{!"0x8ba80ae40.w512.b0", !93, i64 0}
!93 = !{!"0x8ba80ae40.w1024.b0", !94, i64 0}
!94 = !{!"0x8ba80ae40", !45, i64 0}
!95 = !DILocalVariable(name: "tile_storage_7", scope: !33, file: !1, type: !30)
!96 = !DILocalVariable(name: "tile_i_9_pack", scope: !33, file: !1, type: !8)
!97 = !{!98, !98, i64 0}
!98 = !{!"0x8ba80b000.w4.b0", !99, i64 0}
!99 = !{!"0x8ba80b000.w8.b0", !100, i64 0}
!100 = !{!"0x8ba80b000.w16.b0", !101, i64 0}
!101 = !{!"0x8ba80b000.w32.b0", !102, i64 0}
!102 = !{!"0x8ba80b000.w64.b0", !103, i64 0}
!103 = !{!"0x8ba80b000.w128.b0", !104, i64 0}
!104 = !{!"0x8ba80b000.w256.b0", !105, i64 0}
!105 = !{!"0x8ba80b000.w512.b0", !106, i64 0}
!106 = !{!"0x8ba80b000.w1024.b0", !107, i64 0}
!107 = !{!"0x8ba80b000", !45, i64 0}
!108 = !DILocalVariable(name: "cse_v3", scope: !33, file: !1, type: !8)
!109 = !{!110, !110, i64 0}
!110 = !{!"0x8ba80b0c0", !45, i64 0}
!111 = !{!112, !112, i64 0}
!112 = !{!"0x8ba80b0c0.w4.b1024", !113, i64 0}
!113 = !{!"0x8ba80b0c0.w8.b1024", !114, i64 0}
!114 = !{!"0x8ba80b0c0.w16.b1024", !115, i64 0}
!115 = !{!"0x8ba80b0c0.w32.b1024", !116, i64 0}
!116 = !{!"0x8ba80b0c0.w64.b1024", !117, i64 0}
!117 = !{!"0x8ba80b0c0.w128.b1024", !118, i64 0}
!118 = !{!"0x8ba80b0c0.w256.b1024", !119, i64 0}
!119 = !{!"0x8ba80b0c0.w512.b1024", !120, i64 0}
!120 = !{!"0x8ba80b0c0.w1024.b1024", !110, i64 0}
!121 = !DILocalVariable(name: "tile_storage_10", scope: !33, file: !1, type: !30)
!122 = !DILocalVariable(name: "tile_storage_12", scope: !33, file: !1, type: !30)
!123 = !{!124, !124, i64 0}
!124 = !{!"0x8ba80b1c0.w4.b0", !125, i64 0}
!125 = !{!"0x8ba80b1c0.w8.b0", !126, i64 0}
!126 = !{!"0x8ba80b1c0.w16.b0", !127, i64 0}
!127 = !{!"0x8ba80b1c0.w32.b0", !128, i64 0}
!128 = !{!"0x8ba80b1c0.w64.b0", !129, i64 0}
!129 = !{!"0x8ba80b1c0.w128.b0", !130, i64 0}
!130 = !{!"0x8ba80b1c0.w256.b0", !131, i64 0}
!131 = !{!"0x8ba80b1c0.w512.b0", !132, i64 0}
!132 = !{!"0x8ba80b1c0.w1024.b0", !133, i64 0}
!133 = !{!"0x8ba80b1c0", !45, i64 0}
!134 = !DILocalVariable(name: "n_18_0", scope: !33, file: !1, type: !8)
!135 = !DILocalVariable(name: "tile_storage_13", scope: !33, file: !1, type: !30)
!136 = !{!137, !137, i64 0}
!137 = !{!"0x8ba80b180.w4.b0", !138, i64 0}
!138 = !{!"0x8ba80b180.w8.b0", !139, i64 0}
!139 = !{!"0x8ba80b180.w16.b0", !140, i64 0}
!140 = !{!"0x8ba80b180.w32.b0", !141, i64 0}
!141 = !{!"0x8ba80b180.w64.b0", !142, i64 0}
!142 = !{!"0x8ba80b180.w128.b0", !143, i64 0}
!143 = !{!"0x8ba80b180.w256.b0", !144, i64 0}
!144 = !{!"0x8ba80b180.w512.b0", !145, i64 0}
!145 = !{!"0x8ba80b180.w1024.b0", !146, i64 0}
!146 = !{!"0x8ba80b180", !45, i64 0}
!147 = !DILocalVariable(name: "tile_i_15_pack", scope: !33, file: !1, type: !8)
!148 = !DILocalVariable(name: "cse_v4", scope: !33, file: !1, type: !8)
!149 = !{!150, !150, i64 0}
!150 = !{!"0x8baeb9440", !45, i64 0}
!151 = !{!152, !152, i64 0}
!152 = !{!"0x8ba80b240.w4.b0", !153, i64 0}
!153 = !{!"0x8ba80b240.w8.b0", !154, i64 0}
!154 = !{!"0x8ba80b240.w16.b0", !155, i64 0}
!155 = !{!"0x8ba80b240.w32.b0", !156, i64 0}
!156 = !{!"0x8ba80b240.w64.b0", !157, i64 0}
!157 = !{!"0x8ba80b240.w128.b0", !158, i64 0}
!158 = !{!"0x8ba80b240.w256.b0", !159, i64 0}
!159 = !{!"0x8ba80b240.w512.b0", !160, i64 0}
!160 = !{!"0x8ba80b240.w1024.b0", !161, i64 0}
!161 = !{!"0x8ba80b240", !45, i64 0}
