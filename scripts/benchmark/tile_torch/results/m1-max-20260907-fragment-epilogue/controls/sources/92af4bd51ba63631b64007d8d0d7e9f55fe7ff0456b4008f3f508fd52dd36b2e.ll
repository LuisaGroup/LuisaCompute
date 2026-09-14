; ModuleID = 'TVMMod'
source_filename = "TVMMod"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-darwin25.6.0"

@.str = private constant [10 x i8] c"TypeError\00", align 1
@.str.1 = private constant [10 x i8] c"Expected \00", align 1
@.str.2 = private constant [2 x i8] c"2\00", align 1
@.str.3 = private constant [11 x i8] c" arguments\00", align 1
@.str.4 = private constant [19 x i8] c" when calling:\0A  `\00", align 1
@.str.5 = private constant [104 x i8] c"benchmark_sum(arg0: Tensor([T.int64(17), T.int64(257)], float32), arg1: Tensor([T.int64(17)], float32))\00", align 1
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
@.str.37 = private constant [17 x i8] c"arg1.byte_offset\00", align 1
@__TVMBackendAllocWorkspace = linkonce dllexport local_unnamed_addr global ptr null, align 8
@__TVMBackendFreeWorkspace = linkonce dllexport local_unnamed_addr global ptr null, align 8
@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

define dllexport range(i32 -1, 1) i32 @__tvm_ffi_benchmark_sum(ptr noalias readnone captures(none) %self_handle, ptr noalias readonly captures(address_is_null) %args, i32 %num_args, ptr noalias readnone captures(none) %result) local_unnamed_addr #0 !dbg !5 {
entry:
    #dbg_value(ptr poison, !11, !DIExpression(), !15)
    #dbg_value(ptr %args, !12, !DIExpression(), !15)
    #dbg_value(i32 %num_args, !13, !DIExpression(), !15)
    #dbg_value(ptr poison, !14, !DIExpression(), !15)
  %0 = icmp eq i32 %num_args, 2, !dbg !15
  br i1 %0, label %assert_end, label %assert_fail, !dbg !15, !prof !16

common.ret:                                       ; preds = %assert_end49, %assert_fail48, %assert_fail46, %assert_fail44, %assert_fail42, %assert_fail40, %assert_fail38, %assert_fail36, %assert_fail34, %assert_fail30, %assert_fail28, %assert_fail24, %assert_fail22, %assert_fail20, %assert_fail18, %assert_fail16, %assert_fail14, %assert_fail12, %assert_fail10, %assert_fail5, %assert_fail3, %assert_fail1, %assert_fail
  %common.ret.op = phi i32 [ -1, %assert_fail ], [ -1, %assert_fail1 ], [ -1, %assert_fail3 ], [ -1, %assert_fail5 ], [ -1, %assert_fail10 ], [ -1, %assert_fail12 ], [ -1, %assert_fail14 ], [ -1, %assert_fail16 ], [ -1, %assert_fail18 ], [ -1, %assert_fail20 ], [ -1, %assert_fail22 ], [ -1, %assert_fail24 ], [ -1, %assert_fail28 ], [ -1, %assert_fail30 ], [ -1, %assert_fail34 ], [ -1, %assert_fail36 ], [ -1, %assert_fail38 ], [ -1, %assert_fail40 ], [ -1, %assert_fail42 ], [ -1, %assert_fail44 ], [ -1, %assert_fail46 ], [ -1, %assert_fail48 ], [ %81, %assert_end49 ]
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
  br i1 %9, label %assert_end6, label %switch.early.test50, !dbg !15

switch.early.test50:                              ; preds = %if_end
  switch i32 %arg1.type_index.fr, label %assert_fail5 [
    i32 0, label %if_else8
    i32 4, label %if_else8
    i32 7, label %if_else8
  ], !dbg !15

assert_fail5:                                     ; preds = %switch.early.test50
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

if_else8:                                         ; preds = %switch.early.test50, %switch.early.test50, %switch.early.test50, %assert_end6
  %14 = getelementptr inbounds nuw i8, ptr %args, i64 24, !dbg !15
  %15 = load ptr, ptr %14, align 8, !dbg !15
  br label %if_end9, !dbg !15

if_end9:                                          ; preds = %if_else8, %if_then7
  %arg1.handle = phi ptr [ %13, %if_then7 ], [ %15, %if_else8 ], !dbg !15
    #dbg_declare(ptr %arg1.handle, !21, !DIExpression(), !15)
    #dbg_declare(ptr %arg1.handle, !21, !DIExpression(), !15)
  %.not51 = icmp eq ptr %arg0.handle, null, !dbg !15
  br i1 %.not51, label %assert_fail10, label %assert_end11, !dbg !15, !prof !17

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
  %benchmark_sum.arg0_shape = load ptr, ptr %30, align 8, !dbg !15
    #dbg_declare(ptr %benchmark_sum.arg0_shape, !22, !DIExpression(), !15)
    #dbg_declare(ptr %benchmark_sum.arg0_shape, !22, !DIExpression(), !15)
  %31 = getelementptr inbounds nuw i8, ptr %arg0.handle, i64 32, !dbg !15
  %benchmark_sum.arg0_strides = load ptr, ptr %31, align 8, !dbg !15
    #dbg_declare(ptr %benchmark_sum.arg0_strides, !25, !DIExpression(), !15)
    #dbg_declare(ptr %benchmark_sum.arg0_strides, !25, !DIExpression(), !15)
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
  %.not52 = icmp eq ptr %arg1.handle, null, !dbg !15
  br i1 %.not52, label %assert_fail18, label %assert_end19, !dbg !15, !prof !17

assert_fail18:                                    ; preds = %assert_end17
  tail call fastcc void @__tvm_set_raised_6(ptr nonnull @.str, i32 6, ptr nonnull @.str.8, ptr nonnull @.str.12, ptr nonnull @.str.4, ptr nonnull @.str.5, ptr nonnull @.str.10, ptr nonnull @.str.11), !dbg !15
  br label %common.ret, !dbg !15

assert_end19:                                     ; preds = %assert_end17
  %36 = getelementptr inbounds nuw i8, ptr %arg1.handle, i64 16, !dbg !15
  %37 = load i32, ptr %36, align 4, !dbg !15
  %38 = icmp eq i32 %37, 1, !dbg !15
  br i1 %38, label %assert_end21, label %assert_fail20, !dbg !15, !prof !16

assert_fail20:                                    ; preds = %assert_end19
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.13, i32 8, ptr nonnull @.str.14, ptr nonnull @.str.21, ptr nonnull @.str.16, ptr nonnull @.str.12, ptr nonnull @.str.10, ptr nonnull @.str.12), !dbg !15
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
  %benchmark_sum.arg1_shape = load ptr, ptr %50, align 8, !dbg !15
    #dbg_declare(ptr %benchmark_sum.arg1_shape, !27, !DIExpression(), !15)
    #dbg_declare(ptr %benchmark_sum.arg1_shape, !27, !DIExpression(), !15)
  %51 = getelementptr inbounds nuw i8, ptr %arg1.handle, i64 32, !dbg !15
  %benchmark_sum.arg1_strides = load ptr, ptr %51, align 8, !dbg !15
    #dbg_declare(ptr %benchmark_sum.arg1_strides, !28, !DIExpression(), !15)
    #dbg_declare(ptr %benchmark_sum.arg1_strides, !28, !DIExpression(), !15)
  %52 = getelementptr inbounds nuw i8, ptr %arg1.handle, i64 8, !dbg !15
  %53 = load i32, ptr %52, align 4, !dbg !15
  %54 = icmp eq i32 %53, 1, !dbg !15
  br i1 %54, label %assert_end25, label %assert_fail24, !dbg !15, !prof !16

assert_fail24:                                    ; preds = %assert_end23
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.13, i32 8, ptr nonnull @.str.14, ptr nonnull @.str.21, ptr nonnull @.str.19, ptr nonnull @.str.12, ptr nonnull @.str.10, ptr nonnull @.str.20), !dbg !15
  br label %common.ret, !dbg !15

assert_end25:                                     ; preds = %assert_end23
  %.not53 = icmp eq ptr %benchmark_sum.arg0_strides, null, !dbg !15
  br i1 %.not53, label %if_end27, label %if_then26, !dbg !15, !prof !17

if_then26:                                        ; preds = %assert_end25
  %55 = getelementptr inbounds nuw i8, ptr %benchmark_sum.arg0_strides, i64 8, !dbg !15
  %56 = load i64, ptr %55, align 8, !dbg !15
  %57 = icmp eq i64 %56, 1, !dbg !15
  %58 = load i64, ptr %benchmark_sum.arg0_strides, align 8, !dbg !15
  %59 = icmp eq i64 %58, 257, !dbg !15
  %60 = and i1 %57, %59, !dbg !15
  br i1 %60, label %if_end27, label %assert_fail28, !dbg !15, !prof !16

if_end27:                                         ; preds = %if_then26, %assert_end25
  %61 = load ptr, ptr %arg0.handle, align 8, !dbg !15
  %.not54 = icmp eq ptr %61, null, !dbg !15
  br i1 %.not54, label %assert_fail30, label %assert_end31, !dbg !15, !prof !17

assert_fail28:                                    ; preds = %if_then26
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.13, i32 7, ptr nonnull @.str.14, ptr nonnull @.str.15, ptr nonnull @.str.22, ptr nonnull @.str.9, ptr nonnull @.str.23, ptr null), !dbg !15
  br label %common.ret, !dbg !15

assert_fail30:                                    ; preds = %if_end27
  tail call fastcc void @__tvm_set_raised_6(ptr nonnull @.str.13, i32 6, ptr nonnull @.str.15, ptr nonnull @.str.24, ptr nonnull @.str.9, ptr nonnull @.str.4, ptr nonnull @.str.5, ptr nonnull @.str.25), !dbg !15
  br label %common.ret, !dbg !15

assert_end31:                                     ; preds = %if_end27
  %.not55 = icmp eq ptr %benchmark_sum.arg1_strides, null, !dbg !15
  br i1 %.not55, label %if_end33, label %if_then32, !dbg !15, !prof !17

if_then32:                                        ; preds = %assert_end31
  %62 = load i64, ptr %benchmark_sum.arg1_strides, align 8, !dbg !15
  %63 = icmp eq i64 %62, 1, !dbg !15
  br i1 %63, label %if_end33, label %assert_fail34, !dbg !15, !prof !16

if_end33:                                         ; preds = %if_then32, %assert_end31
  %64 = getelementptr inbounds nuw i8, ptr %arg1.handle, i64 12, !dbg !15
  %65 = load i32, ptr %64, align 4, !dbg !15
  %66 = icmp eq i32 %dev_id, %65, !dbg !15
  br i1 %66, label %assert_end37, label %assert_fail36, !dbg !15, !prof !16

assert_fail34:                                    ; preds = %if_then32
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.13, i32 7, ptr nonnull @.str.14, ptr nonnull @.str.21, ptr nonnull @.str.22, ptr nonnull @.str.12, ptr nonnull @.str.23, ptr null), !dbg !15
  br label %common.ret, !dbg !15

assert_fail36:                                    ; preds = %if_end33
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.13, i32 8, ptr nonnull @.str.14, ptr nonnull @.str.26, ptr nonnull @.str.27, ptr nonnull @.str.12, ptr nonnull @.str.28, ptr nonnull @.str.29), !dbg !15
  br label %common.ret, !dbg !15

assert_end37:                                     ; preds = %if_end33
  %67 = load ptr, ptr %arg1.handle, align 8, !dbg !15
  %.not56 = icmp eq ptr %67, null, !dbg !15
  br i1 %.not56, label %assert_fail38, label %assert_end39, !dbg !15, !prof !17

assert_fail38:                                    ; preds = %assert_end37
  tail call fastcc void @__tvm_set_raised_6(ptr nonnull @.str.13, i32 6, ptr nonnull @.str.21, ptr nonnull @.str.24, ptr nonnull @.str.12, ptr nonnull @.str.4, ptr nonnull @.str.5, ptr nonnull @.str.25), !dbg !15
  br label %common.ret, !dbg !15

assert_end39:                                     ; preds = %assert_end37
  %68 = load i64, ptr %benchmark_sum.arg0_shape, align 8, !dbg !15
  %69 = icmp eq i64 %68, 17, !dbg !15
  br i1 %69, label %assert_end41, label %assert_fail40, !dbg !15, !prof !16

assert_fail40:                                    ; preds = %assert_end39
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.13, i32 8, ptr nonnull @.str.30, ptr nonnull @.str.31, ptr nonnull @.str.27, ptr nonnull @.str.9, ptr nonnull @.str.10, ptr nonnull @.str.32), !dbg !15
  br label %common.ret, !dbg !15

assert_end41:                                     ; preds = %assert_end39
  %70 = getelementptr inbounds nuw i8, ptr %benchmark_sum.arg0_shape, i64 8, !dbg !15
  %71 = load i64, ptr %70, align 8, !dbg !15
  %72 = icmp eq i64 %71, 257, !dbg !15
  br i1 %72, label %assert_end43, label %assert_fail42, !dbg !15, !prof !16

assert_fail42:                                    ; preds = %assert_end41
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.13, i32 8, ptr nonnull @.str.30, ptr nonnull @.str.33, ptr nonnull @.str.27, ptr nonnull @.str.9, ptr nonnull @.str.10, ptr nonnull @.str.34), !dbg !15
  br label %common.ret, !dbg !15

assert_end43:                                     ; preds = %assert_end41
  %73 = getelementptr inbounds nuw i8, ptr %arg0.handle, i64 40, !dbg !15
  %74 = load i64, ptr %73, align 8, !dbg !15
  %75 = icmp eq i64 %74, 0, !dbg !15
  br i1 %75, label %assert_end45, label %assert_fail44, !dbg !15, !prof !16

assert_fail44:                                    ; preds = %assert_end43
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.13, i32 8, ptr nonnull @.str.30, ptr nonnull @.str.35, ptr nonnull @.str.27, ptr nonnull @.str.9, ptr nonnull @.str.10, ptr nonnull @.str.9), !dbg !15
  br label %common.ret, !dbg !15

assert_end45:                                     ; preds = %assert_end43
  %76 = load i64, ptr %benchmark_sum.arg1_shape, align 8, !dbg !15
  %77 = icmp eq i64 %76, 17, !dbg !15
  br i1 %77, label %assert_end47, label %assert_fail46, !dbg !15, !prof !16

assert_fail46:                                    ; preds = %assert_end45
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.13, i32 8, ptr nonnull @.str.30, ptr nonnull @.str.36, ptr nonnull @.str.27, ptr nonnull @.str.12, ptr nonnull @.str.10, ptr nonnull @.str.32), !dbg !15
  br label %common.ret, !dbg !15

assert_end47:                                     ; preds = %assert_end45
  %78 = getelementptr inbounds nuw i8, ptr %arg1.handle, i64 40, !dbg !15
  %79 = load i64, ptr %78, align 8, !dbg !15
  %80 = icmp eq i64 %79, 0, !dbg !15
  br i1 %80, label %assert_end49, label %assert_fail48, !dbg !15, !prof !16

assert_fail48:                                    ; preds = %assert_end47
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.13, i32 8, ptr nonnull @.str.30, ptr nonnull @.str.37, ptr nonnull @.str.27, ptr nonnull @.str.12, ptr nonnull @.str.10, ptr nonnull @.str.9), !dbg !15
  br label %common.ret, !dbg !15

assert_end49:                                     ; preds = %assert_end47
    #dbg_declare(ptr %61, !29, !DIExpression(), !15)
  call void @llvm.assume(i1 true) [ "align"(ptr %61, i64 64) ], !dbg !15
    #dbg_declare(ptr %67, !32, !DIExpression(), !15)
  call void @llvm.assume(i1 true) [ "align"(ptr %67, i64 64) ], !dbg !15
  %81 = tail call fastcc i32 @benchmark_sum_compute_(ptr nonnull %61, ptr nonnull %67, i32 %dev_id), !dbg !15
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
define internal fastcc range(i32 -1, 1) i32 @benchmark_sum_compute_(ptr noalias align 64 %arg0, ptr noalias align 64 %arg1, i32 %dev_id) unnamed_addr #3 !dbg !33 {
entry:
    #dbg_declare(ptr %arg0, !40, !DIExpression(), !41)
    #dbg_value(ptr %arg0, !37, !DIExpression(), !41)
    #dbg_value(ptr %arg1, !38, !DIExpression(), !41)
    #dbg_value(i32 %dev_id, !39, !DIExpression(), !41)
  call void @llvm.assume(i1 true) [ "align"(ptr %arg0, i64 64) ], !dbg !41
    #dbg_declare(ptr %arg1, !42, !DIExpression(), !41)
  call void @llvm.assume(i1 true) [ "align"(ptr %arg1, i64 64) ], !dbg !41
  %invariant.gep = getelementptr inbounds nuw i8, ptr %arg0, i64 1024, !dbg !41
    #dbg_value(i32 0, !43, !DIExpression(), !41)
  %invariant.gep39 = getelementptr inbounds nuw i8, ptr %arg0, i64 64, !dbg !41
  %invariant.gep41 = getelementptr inbounds nuw i8, ptr %arg0, i64 128, !dbg !41
  %invariant.gep43 = getelementptr inbounds nuw i8, ptr %arg0, i64 192, !dbg !41
  %invariant.gep45 = getelementptr inbounds nuw i8, ptr %arg0, i64 256, !dbg !41
  %invariant.gep47 = getelementptr inbounds nuw i8, ptr %arg0, i64 320, !dbg !41
  %invariant.gep49 = getelementptr inbounds nuw i8, ptr %arg0, i64 384, !dbg !41
  %invariant.gep51 = getelementptr inbounds nuw i8, ptr %arg0, i64 448, !dbg !41
  %invariant.gep53 = getelementptr inbounds nuw i8, ptr %arg0, i64 512, !dbg !41
  %invariant.gep55 = getelementptr inbounds nuw i8, ptr %arg0, i64 576, !dbg !41
  %invariant.gep57 = getelementptr inbounds nuw i8, ptr %arg0, i64 640, !dbg !41
  %invariant.gep59 = getelementptr inbounds nuw i8, ptr %arg0, i64 704, !dbg !41
  %invariant.gep61 = getelementptr inbounds nuw i8, ptr %arg0, i64 768, !dbg !41
  %invariant.gep63 = getelementptr inbounds nuw i8, ptr %arg0, i64 832, !dbg !41
  %invariant.gep65 = getelementptr inbounds nuw i8, ptr %arg0, i64 896, !dbg !41
  %invariant.gep67 = getelementptr inbounds nuw i8, ptr %arg0, i64 960, !dbg !41
  br label %for_body_parallel_0, !dbg !41

for_begin_parallel_0:                             ; preds = %if_end17
  %indvars.iv.next34 = add nuw nsw i64 %indvars.iv33, 1, !dbg !41
    #dbg_value(i64 %indvars.iv.next34, !43, !DIExpression(), !41)
  %exitcond37.not = icmp eq i64 %indvars.iv.next34, 17, !dbg !41
  br i1 %exitcond37.not, label %common.ret, label %for_body_parallel_0, !dbg !41, !prof !44

for_body_parallel_0:                              ; preds = %entry, %for_begin_parallel_0
  %indvars.iv33 = phi i64 [ 0, %entry ], [ %indvars.iv.next34, %for_begin_parallel_0 ]
    #dbg_value(i64 %indvars.iv33, !43, !DIExpression(), !41)
  %0 = load ptr, ptr @__TVMBackendAllocWorkspace, align 8, !dbg !41, !tbaa !45
  %tile_storage_0 = tail call ptr %0(i32 1, i32 %dev_id, i64 1028, i32 2, i32 32), !dbg !41
    #dbg_declare(ptr %tile_storage_0, !48, !DIExpression(), !41)
  %1 = icmp eq ptr %tile_storage_0, null, !dbg !41
  br i1 %1, label %common.ret, label %if_end, !dbg !41, !prof !16

common.ret:                                       ; preds = %if_end17, %if_end14, %for_end_n_5_0, %if_end2, %if_end, %for_body_parallel_0, %for_begin_parallel_0, %if_end8, %for_body_n_5_0
  %common.ret.op = phi i32 [ -1, %for_body_n_5_0 ], [ -1, %if_end8 ], [ -1, %if_end17 ], [ -1, %if_end14 ], [ -1, %for_end_n_5_0 ], [ -1, %if_end2 ], [ -1, %if_end ], [ -1, %for_body_parallel_0 ], [ 0, %for_begin_parallel_0 ]
  ret i32 %common.ret.op, !dbg !41

if_end:                                           ; preds = %for_body_parallel_0
  %2 = mul nuw nsw i64 %indvars.iv33, 257, !dbg !41
    #dbg_value(i64 %2, !49, !DIExpression(), !41)
    #dbg_value(i64 0, !50, !DIExpression(), !41)
    #dbg_value(i64 0, !51, !DIExpression(), !41)
  %3 = getelementptr inbounds nuw float, ptr %arg0, i64 %2, !dbg !41
  %4 = load <16 x float>, ptr %3, align 4, !dbg !41, !tbaa !52
  store <16 x float> %4, ptr %tile_storage_0, align 4, !dbg !41, !tbaa !54
    #dbg_value(i64 1, !50, !DIExpression(), !41)
    #dbg_value(i64 16, !51, !DIExpression(), !41)
  %gep40 = getelementptr inbounds nuw float, ptr %invariant.gep39, i64 %2, !dbg !41
  %5 = load <16 x float>, ptr %gep40, align 4, !dbg !41, !tbaa !52
  %6 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 64, !dbg !41
  store <16 x float> %5, ptr %6, align 4, !dbg !41, !tbaa !54
    #dbg_value(i64 2, !50, !DIExpression(), !41)
    #dbg_value(i64 32, !51, !DIExpression(), !41)
  %gep42 = getelementptr inbounds nuw float, ptr %invariant.gep41, i64 %2, !dbg !41
  %7 = load <16 x float>, ptr %gep42, align 4, !dbg !41, !tbaa !52
  %8 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 128, !dbg !41
  store <16 x float> %7, ptr %8, align 4, !dbg !41, !tbaa !54
    #dbg_value(i64 3, !50, !DIExpression(), !41)
    #dbg_value(i64 48, !51, !DIExpression(), !41)
  %gep44 = getelementptr inbounds nuw float, ptr %invariant.gep43, i64 %2, !dbg !41
  %9 = load <16 x float>, ptr %gep44, align 4, !dbg !41, !tbaa !52
  %10 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 192, !dbg !41
  store <16 x float> %9, ptr %10, align 4, !dbg !41, !tbaa !54
    #dbg_value(i64 4, !50, !DIExpression(), !41)
    #dbg_value(i64 64, !51, !DIExpression(), !41)
  %gep46 = getelementptr inbounds nuw float, ptr %invariant.gep45, i64 %2, !dbg !41
  %11 = load <16 x float>, ptr %gep46, align 4, !dbg !41, !tbaa !52
  %12 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 256, !dbg !41
  store <16 x float> %11, ptr %12, align 4, !dbg !41, !tbaa !54
    #dbg_value(i64 5, !50, !DIExpression(), !41)
    #dbg_value(i64 80, !51, !DIExpression(), !41)
  %gep48 = getelementptr inbounds nuw float, ptr %invariant.gep47, i64 %2, !dbg !41
  %13 = load <16 x float>, ptr %gep48, align 4, !dbg !41, !tbaa !52
  %14 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 320, !dbg !41
  store <16 x float> %13, ptr %14, align 4, !dbg !41, !tbaa !54
    #dbg_value(i64 6, !50, !DIExpression(), !41)
    #dbg_value(i64 96, !51, !DIExpression(), !41)
  %gep50 = getelementptr inbounds nuw float, ptr %invariant.gep49, i64 %2, !dbg !41
  %15 = load <16 x float>, ptr %gep50, align 4, !dbg !41, !tbaa !52
  %16 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 384, !dbg !41
  store <16 x float> %15, ptr %16, align 4, !dbg !41, !tbaa !54
    #dbg_value(i64 7, !50, !DIExpression(), !41)
    #dbg_value(i64 112, !51, !DIExpression(), !41)
  %gep52 = getelementptr inbounds nuw float, ptr %invariant.gep51, i64 %2, !dbg !41
  %17 = load <16 x float>, ptr %gep52, align 4, !dbg !41, !tbaa !52
  %18 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 448, !dbg !41
  store <16 x float> %17, ptr %18, align 4, !dbg !41, !tbaa !54
    #dbg_value(i64 8, !50, !DIExpression(), !41)
    #dbg_value(i64 128, !51, !DIExpression(), !41)
  %gep54 = getelementptr inbounds nuw float, ptr %invariant.gep53, i64 %2, !dbg !41
  %19 = load <16 x float>, ptr %gep54, align 4, !dbg !41, !tbaa !52
  %20 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 512, !dbg !41
  store <16 x float> %19, ptr %20, align 4, !dbg !41, !tbaa !54
    #dbg_value(i64 9, !50, !DIExpression(), !41)
    #dbg_value(i64 144, !51, !DIExpression(), !41)
  %gep56 = getelementptr inbounds nuw float, ptr %invariant.gep55, i64 %2, !dbg !41
  %21 = load <16 x float>, ptr %gep56, align 4, !dbg !41, !tbaa !52
  %22 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 576, !dbg !41
  store <16 x float> %21, ptr %22, align 4, !dbg !41, !tbaa !54
    #dbg_value(i64 10, !50, !DIExpression(), !41)
    #dbg_value(i64 160, !51, !DIExpression(), !41)
  %gep58 = getelementptr inbounds nuw float, ptr %invariant.gep57, i64 %2, !dbg !41
  %23 = load <16 x float>, ptr %gep58, align 4, !dbg !41, !tbaa !52
  %24 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 640, !dbg !41
  store <16 x float> %23, ptr %24, align 4, !dbg !41, !tbaa !54
    #dbg_value(i64 11, !50, !DIExpression(), !41)
    #dbg_value(i64 176, !51, !DIExpression(), !41)
  %gep60 = getelementptr inbounds nuw float, ptr %invariant.gep59, i64 %2, !dbg !41
  %25 = load <16 x float>, ptr %gep60, align 4, !dbg !41, !tbaa !52
  %26 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 704, !dbg !41
  store <16 x float> %25, ptr %26, align 4, !dbg !41, !tbaa !54
    #dbg_value(i64 12, !50, !DIExpression(), !41)
    #dbg_value(i64 192, !51, !DIExpression(), !41)
  %gep62 = getelementptr inbounds nuw float, ptr %invariant.gep61, i64 %2, !dbg !41
  %27 = load <16 x float>, ptr %gep62, align 4, !dbg !41, !tbaa !52
  %28 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 768, !dbg !41
  store <16 x float> %27, ptr %28, align 4, !dbg !41, !tbaa !54
    #dbg_value(i64 13, !50, !DIExpression(), !41)
    #dbg_value(i64 208, !51, !DIExpression(), !41)
  %gep64 = getelementptr inbounds nuw float, ptr %invariant.gep63, i64 %2, !dbg !41
  %29 = load <16 x float>, ptr %gep64, align 4, !dbg !41, !tbaa !52
  %30 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 832, !dbg !41
  store <16 x float> %29, ptr %30, align 4, !dbg !41, !tbaa !54
    #dbg_value(i64 14, !50, !DIExpression(), !41)
    #dbg_value(i64 224, !51, !DIExpression(), !41)
  %gep66 = getelementptr inbounds nuw float, ptr %invariant.gep65, i64 %2, !dbg !41
  %31 = load <16 x float>, ptr %gep66, align 4, !dbg !41, !tbaa !52
  %32 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 896, !dbg !41
  store <16 x float> %31, ptr %32, align 4, !dbg !41, !tbaa !54
    #dbg_value(i64 15, !50, !DIExpression(), !41)
    #dbg_value(i64 240, !51, !DIExpression(), !41)
  %gep68 = getelementptr inbounds nuw float, ptr %invariant.gep67, i64 %2, !dbg !41
  %33 = load <16 x float>, ptr %gep68, align 4, !dbg !41, !tbaa !52
  %34 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 960, !dbg !41
  store <16 x float> %33, ptr %34, align 4, !dbg !41, !tbaa !54
    #dbg_value(i64 16, !50, !DIExpression(), !41)
  %gep = getelementptr inbounds nuw float, ptr %invariant.gep, i64 %2, !dbg !41
  %35 = load float, ptr %gep, align 4, !dbg !41, !tbaa !52
  %36 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 1024, !dbg !41
  store float %35, ptr %36, align 4, !dbg !41, !tbaa !56
  %37 = load ptr, ptr @__TVMBackendAllocWorkspace, align 8, !dbg !41, !tbaa !45
  %tile_storage_3 = tail call ptr %37(i32 1, i32 %dev_id, i64 4, i32 2, i32 32), !dbg !41
    #dbg_declare(ptr %tile_storage_3, !66, !DIExpression(), !41)
  %38 = icmp eq ptr %tile_storage_3, null, !dbg !41
  br i1 %38, label %common.ret, label %if_end2, !dbg !41, !prof !16

if_end2:                                          ; preds = %if_end
  %39 = load ptr, ptr @__TVMBackendAllocWorkspace, align 8, !dbg !41, !tbaa !45
  %tile_storage_5 = tail call ptr %39(i32 1, i32 %dev_id, i64 4, i32 2, i32 32), !dbg !41
    #dbg_declare(ptr %tile_storage_5, !67, !DIExpression(), !41)
  %40 = icmp eq ptr %tile_storage_5, null, !dbg !41
  br i1 %40, label %common.ret, label %if_end5, !dbg !41, !prof !16

if_end5:                                          ; preds = %if_end2
  store float 0.000000e+00, ptr %tile_storage_5, align 4, !dbg !41, !tbaa !68
    #dbg_value(i32 0, !79, !DIExpression(), !41)
  br label %for_body_n_5_0, !dbg !41

for_begin_n_5_0:                                  ; preds = %if_end8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !41
    #dbg_value(i32 poison, !79, !DIExpression(), !41)
  %exitcond.not = icmp eq i64 %indvars.iv.next, 257, !dbg !41
  br i1 %exitcond.not, label %for_end_n_5_0, label %for_body_n_5_0, !dbg !41, !prof !44

for_body_n_5_0:                                   ; preds = %if_end5, %for_begin_n_5_0
  %indvars.iv = phi i64 [ 0, %if_end5 ], [ %indvars.iv.next, %for_begin_n_5_0 ]
    #dbg_value(i64 %indvars.iv, !79, !DIExpression(), !41)
  %41 = load ptr, ptr @__TVMBackendAllocWorkspace, align 8, !dbg !41, !tbaa !45
  %tile_storage_6 = tail call ptr %41(i32 1, i32 %dev_id, i64 4, i32 2, i32 32), !dbg !41
    #dbg_declare(ptr %tile_storage_6, !80, !DIExpression(), !41)
  %42 = icmp eq ptr %tile_storage_6, null, !dbg !41
  br i1 %42, label %common.ret, label %if_end8, !dbg !41, !prof !16

for_end_n_5_0:                                    ; preds = %for_begin_n_5_0
  %43 = load float, ptr %tile_storage_5, align 4, !dbg !41, !tbaa !68
  store float %43, ptr %tile_storage_3, align 4, !dbg !41, !tbaa !81
  %44 = getelementptr inbounds nuw float, ptr %arg1, i64 %indvars.iv33, !dbg !41
  store float %43, ptr %44, align 4, !dbg !41, !tbaa !92
  %45 = load ptr, ptr @__TVMBackendFreeWorkspace, align 8, !dbg !41, !tbaa !45
  %46 = tail call i32 %45(i32 1, i32 %dev_id, ptr nonnull %tile_storage_5), !dbg !41
  %.not = icmp eq i32 %46, 0, !dbg !41
  br i1 %.not, label %if_end14, label %common.ret, !dbg !41, !prof !17

if_end8:                                          ; preds = %for_body_n_5_0
  %47 = load float, ptr %tile_storage_5, align 4, !dbg !41, !tbaa !68
  %48 = getelementptr inbounds nuw float, ptr %tile_storage_0, i64 %indvars.iv, !dbg !41
  %49 = load float, ptr %48, align 4, !dbg !41, !tbaa !54
  %50 = fadd float %47, %49, !dbg !41
  store float %50, ptr %tile_storage_6, align 4, !dbg !41, !tbaa !94
  store float %50, ptr %tile_storage_5, align 4, !dbg !41, !tbaa !68
  %51 = load ptr, ptr @__TVMBackendFreeWorkspace, align 8, !dbg !41, !tbaa !45
  %52 = tail call i32 %51(i32 1, i32 %dev_id, ptr nonnull %tile_storage_6), !dbg !41
  %.not24 = icmp eq i32 %52, 0, !dbg !41
  br i1 %.not24, label %for_begin_n_5_0, label %common.ret, !dbg !41, !prof !17

if_end14:                                         ; preds = %for_end_n_5_0
  %53 = load ptr, ptr @__TVMBackendFreeWorkspace, align 8, !dbg !41, !tbaa !45
  %54 = tail call i32 %53(i32 1, i32 %dev_id, ptr nonnull %tile_storage_3), !dbg !41
  %.not22 = icmp eq i32 %54, 0, !dbg !41
  br i1 %.not22, label %if_end17, label %common.ret, !dbg !41, !prof !17

if_end17:                                         ; preds = %if_end14
  %55 = load ptr, ptr @__TVMBackendFreeWorkspace, align 8, !dbg !41, !tbaa !45
  %56 = tail call i32 %55(i32 1, i32 %dev_id, ptr nonnull %tile_storage_0), !dbg !41
  %.not23 = icmp eq i32 %56, 0, !dbg !41
  br i1 %.not23, label %for_begin_parallel_0, label %common.ret, !dbg !41, !prof !17
}

define weak dllexport i32 @__tvm_ffi_main(ptr %0, ptr %1, i32 %2, ptr %3) local_unnamed_addr {
entry:
  %4 = tail call i32 @__tvm_ffi_benchmark_sum(ptr poison, ptr %1, i32 %2, ptr poison), !dbg !15
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
declare float @llvm.fabs.f32(float) #5

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #6

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
!5 = distinct !DISubprogram(name: "__tvm_ffi_benchmark_sum", scope: !1, file: !1, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
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
!22 = !DILocalVariable(name: "benchmark_sum.arg0_shape", scope: !5, file: !1, type: !23)
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !24)
!24 = !DIBasicType(name: "int64", size: 64, encoding: DW_ATE_signed)
!25 = !DILocalVariable(name: "benchmark_sum.arg0_strides", scope: !5, file: !1, type: !23)
!26 = !DILocalVariable(name: "dev_id", scope: !5, file: !1, type: !8)
!27 = !DILocalVariable(name: "benchmark_sum.arg1_shape", scope: !5, file: !1, type: !23)
!28 = !DILocalVariable(name: "benchmark_sum.arg1_strides", scope: !5, file: !1, type: !23)
!29 = !DILocalVariable(name: "arg0", scope: !5, file: !1, type: !30)
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !31)
!31 = !DIBasicType(name: "float32", size: 32, encoding: DW_ATE_float)
!32 = !DILocalVariable(name: "arg1", scope: !5, file: !1, type: !30)
!33 = distinct !DISubprogram(name: "benchmark_sum_compute_", scope: !1, file: !1, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !36)
!34 = !DISubroutineType(types: !35)
!35 = !{!8, !30, !30, !8}
!36 = !{!37, !38, !39}
!37 = !DILocalVariable(name: "arg0", arg: 1, scope: !33, file: !1, type: !30)
!38 = !DILocalVariable(name: "arg1", arg: 2, scope: !33, file: !1, type: !30)
!39 = !DILocalVariable(name: "dev_id", arg: 3, scope: !33, file: !1, type: !8)
!40 = !DILocalVariable(name: "arg0", scope: !33, file: !1, type: !30)
!41 = !DILocation(line: 0, scope: !33)
!42 = !DILocalVariable(name: "arg1", scope: !33, file: !1, type: !30)
!43 = !DILocalVariable(name: "parallel_0", scope: !33, file: !1, type: !8)
!44 = !{!"branch_weights", i32 1, i32 1048575}
!45 = !{!46, !46, i64 0}
!46 = !{!"ctx_ptr", !47, i64 0}
!47 = !{!"tvm-tbaa"}
!48 = !DILocalVariable(name: "tile_storage_0", scope: !33, file: !1, type: !30)
!49 = !DILocalVariable(name: "cse_v1", scope: !33, file: !1, type: !8)
!50 = !DILocalVariable(name: "tile_i_2_pack", scope: !33, file: !1, type: !8)
!51 = !DILocalVariable(name: "cse_v2", scope: !33, file: !1, type: !8)
!52 = !{!53, !53, i64 0}
!53 = !{!"0x78f356280", !47, i64 0}
!54 = !{!55, !55, i64 0}
!55 = !{!"0x78f370900", !47, i64 0}
!56 = !{!57, !57, i64 0}
!57 = !{!"0x78f370900.w4.b1024", !58, i64 0}
!58 = !{!"0x78f370900.w8.b1024", !59, i64 0}
!59 = !{!"0x78f370900.w16.b1024", !60, i64 0}
!60 = !{!"0x78f370900.w32.b1024", !61, i64 0}
!61 = !{!"0x78f370900.w64.b1024", !62, i64 0}
!62 = !{!"0x78f370900.w128.b1024", !63, i64 0}
!63 = !{!"0x78f370900.w256.b1024", !64, i64 0}
!64 = !{!"0x78f370900.w512.b1024", !65, i64 0}
!65 = !{!"0x78f370900.w1024.b1024", !55, i64 0}
!66 = !DILocalVariable(name: "tile_storage_3", scope: !33, file: !1, type: !30)
!67 = !DILocalVariable(name: "tile_storage_5", scope: !33, file: !1, type: !30)
!68 = !{!69, !69, i64 0}
!69 = !{!"0x78f370700.w4.b0", !70, i64 0}
!70 = !{!"0x78f370700.w8.b0", !71, i64 0}
!71 = !{!"0x78f370700.w16.b0", !72, i64 0}
!72 = !{!"0x78f370700.w32.b0", !73, i64 0}
!73 = !{!"0x78f370700.w64.b0", !74, i64 0}
!74 = !{!"0x78f370700.w128.b0", !75, i64 0}
!75 = !{!"0x78f370700.w256.b0", !76, i64 0}
!76 = !{!"0x78f370700.w512.b0", !77, i64 0}
!77 = !{!"0x78f370700.w1024.b0", !78, i64 0}
!78 = !{!"0x78f370700", !47, i64 0}
!79 = !DILocalVariable(name: "n_5_0", scope: !33, file: !1, type: !8)
!80 = !DILocalVariable(name: "tile_storage_6", scope: !33, file: !1, type: !30)
!81 = !{!82, !82, i64 0}
!82 = !{!"0x78f370740.w4.b0", !83, i64 0}
!83 = !{!"0x78f370740.w8.b0", !84, i64 0}
!84 = !{!"0x78f370740.w16.b0", !85, i64 0}
!85 = !{!"0x78f370740.w32.b0", !86, i64 0}
!86 = !{!"0x78f370740.w64.b0", !87, i64 0}
!87 = !{!"0x78f370740.w128.b0", !88, i64 0}
!88 = !{!"0x78f370740.w256.b0", !89, i64 0}
!89 = !{!"0x78f370740.w512.b0", !90, i64 0}
!90 = !{!"0x78f370740.w1024.b0", !91, i64 0}
!91 = !{!"0x78f370740", !47, i64 0}
!92 = !{!93, !93, i64 0}
!93 = !{!"0x78f356300", !47, i64 0}
!94 = !{!95, !95, i64 0}
!95 = !{!"0x78f370680.w4.b0", !96, i64 0}
!96 = !{!"0x78f370680.w8.b0", !97, i64 0}
!97 = !{!"0x78f370680.w16.b0", !98, i64 0}
!98 = !{!"0x78f370680.w32.b0", !99, i64 0}
!99 = !{!"0x78f370680.w64.b0", !100, i64 0}
!100 = !{!"0x78f370680.w128.b0", !101, i64 0}
!101 = !{!"0x78f370680.w256.b0", !102, i64 0}
!102 = !{!"0x78f370680.w512.b0", !103, i64 0}
!103 = !{!"0x78f370680.w1024.b0", !104, i64 0}
!104 = !{!"0x78f370680", !47, i64 0}
