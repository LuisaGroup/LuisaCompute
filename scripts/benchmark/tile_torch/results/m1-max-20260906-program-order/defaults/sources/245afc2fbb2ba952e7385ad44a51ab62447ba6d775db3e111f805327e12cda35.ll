; ModuleID = 'TVMMod'
source_filename = "TVMMod"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-darwin25.6.0"

@.str = private constant [10 x i8] c"TypeError\00", align 1
@.str.1 = private constant [10 x i8] c"Expected \00", align 1
@.str.2 = private constant [2 x i8] c"3\00", align 1
@.str.3 = private constant [11 x i8] c" arguments\00", align 1
@.str.4 = private constant [19 x i8] c" when calling:\0A  `\00", align 1
@.str.5 = private constant [170 x i8] c"benchmark_add(arg0: Tensor([T.int64(17), T.int64(257)], float32), arg1: Tensor([T.int64(17), T.int64(257)], float32), arg2: Tensor([T.int64(17), T.int64(257)], float32))\00", align 1
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
@.str.35 = private constant [3 x i8] c"17\00", align 1
@.str.36 = private constant [14 x i8] c"arg0.shape[1]\00", align 1
@.str.37 = private constant [4 x i8] c"257\00", align 1
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

define dllexport range(i32 -1, 1) i32 @__tvm_ffi_benchmark_add(ptr noalias readnone captures(none) %self_handle, ptr noalias readonly captures(address_is_null) %args, i32 %num_args, ptr noalias readnone captures(none) %result) local_unnamed_addr #0 !dbg !5 {
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
  %benchmark_add.arg0_shape = load ptr, ptr %38, align 8, !dbg !15
    #dbg_declare(ptr %benchmark_add.arg0_shape, !24, !DIExpression(), !15)
    #dbg_declare(ptr %benchmark_add.arg0_shape, !24, !DIExpression(), !15)
  %39 = getelementptr inbounds nuw i8, ptr %arg0.handle, i64 32, !dbg !15
  %benchmark_add.arg0_strides = load ptr, ptr %39, align 8, !dbg !15
    #dbg_declare(ptr %benchmark_add.arg0_strides, !27, !DIExpression(), !15)
    #dbg_declare(ptr %benchmark_add.arg0_strides, !27, !DIExpression(), !15)
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
  %benchmark_add.arg1_shape = load ptr, ptr %58, align 8, !dbg !15
    #dbg_declare(ptr %benchmark_add.arg1_shape, !29, !DIExpression(), !15)
    #dbg_declare(ptr %benchmark_add.arg1_shape, !29, !DIExpression(), !15)
  %59 = getelementptr inbounds nuw i8, ptr %arg1.handle, i64 32, !dbg !15
  %benchmark_add.arg1_strides = load ptr, ptr %59, align 8, !dbg !15
    #dbg_declare(ptr %benchmark_add.arg1_strides, !30, !DIExpression(), !15)
    #dbg_declare(ptr %benchmark_add.arg1_strides, !30, !DIExpression(), !15)
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
  %benchmark_add.arg2_shape = load ptr, ptr %77, align 8, !dbg !15
    #dbg_declare(ptr %benchmark_add.arg2_shape, !31, !DIExpression(), !15)
    #dbg_declare(ptr %benchmark_add.arg2_shape, !31, !DIExpression(), !15)
  %78 = getelementptr inbounds nuw i8, ptr %arg2.handle, i64 32, !dbg !15
  %benchmark_add.arg2_strides = load ptr, ptr %78, align 8, !dbg !15
    #dbg_declare(ptr %benchmark_add.arg2_strides, !32, !DIExpression(), !15)
    #dbg_declare(ptr %benchmark_add.arg2_strides, !32, !DIExpression(), !15)
  %79 = getelementptr inbounds nuw i8, ptr %arg2.handle, i64 8, !dbg !15
  %80 = load i32, ptr %79, align 4, !dbg !15
  %81 = icmp eq i32 %80, 1, !dbg !15
  br i1 %81, label %assert_end38, label %assert_fail37, !dbg !15, !prof !16

assert_fail37:                                    ; preds = %assert_end36
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.14, i32 8, ptr nonnull @.str.15, ptr nonnull @.str.23, ptr nonnull @.str.20, ptr nonnull @.str.13, ptr nonnull @.str.10, ptr nonnull @.str.21), !dbg !15
  br label %common.ret, !dbg !15

assert_end38:                                     ; preds = %assert_end36
  %.not84 = icmp eq ptr %benchmark_add.arg0_strides, null, !dbg !15
  br i1 %.not84, label %if_end40, label %if_then39, !dbg !15, !prof !17

if_then39:                                        ; preds = %assert_end38
  %82 = getelementptr inbounds nuw i8, ptr %benchmark_add.arg0_strides, i64 8, !dbg !15
  %83 = load i64, ptr %82, align 8, !dbg !15
  %84 = icmp eq i64 %83, 1, !dbg !15
  %85 = load i64, ptr %benchmark_add.arg0_strides, align 8, !dbg !15
  %86 = icmp eq i64 %85, 257, !dbg !15
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
  %.not86 = icmp eq ptr %benchmark_add.arg1_strides, null, !dbg !15
  br i1 %.not86, label %if_end46, label %if_then45, !dbg !15, !prof !17

if_then45:                                        ; preds = %assert_end44
  %89 = getelementptr inbounds nuw i8, ptr %benchmark_add.arg1_strides, i64 8, !dbg !15
  %90 = load i64, ptr %89, align 8, !dbg !15
  %91 = icmp eq i64 %90, 1, !dbg !15
  %92 = load i64, ptr %benchmark_add.arg1_strides, align 8, !dbg !15
  %93 = icmp eq i64 %92, 257, !dbg !15
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
  %.not88 = icmp eq ptr %benchmark_add.arg2_strides, null, !dbg !15
  br i1 %.not88, label %if_end54, label %if_then53, !dbg !15, !prof !17

if_then53:                                        ; preds = %assert_end52
  %99 = getelementptr inbounds nuw i8, ptr %benchmark_add.arg2_strides, i64 8, !dbg !15
  %100 = load i64, ptr %99, align 8, !dbg !15
  %101 = icmp eq i64 %100, 1, !dbg !15
  %102 = load i64, ptr %benchmark_add.arg2_strides, align 8, !dbg !15
  %103 = icmp eq i64 %102, 257, !dbg !15
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
  %109 = load i64, ptr %benchmark_add.arg0_shape, align 8, !dbg !15
  %110 = icmp eq i64 %109, 17, !dbg !15
  br i1 %110, label %assert_end62, label %assert_fail61, !dbg !15, !prof !16

assert_fail61:                                    ; preds = %assert_end60
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.14, i32 8, ptr nonnull @.str.33, ptr nonnull @.str.34, ptr nonnull @.str.29, ptr nonnull @.str.9, ptr nonnull @.str.10, ptr nonnull @.str.35), !dbg !15
  br label %common.ret, !dbg !15

assert_end62:                                     ; preds = %assert_end60
  %111 = getelementptr inbounds nuw i8, ptr %benchmark_add.arg0_shape, i64 8, !dbg !15
  %112 = load i64, ptr %111, align 8, !dbg !15
  %113 = icmp eq i64 %112, 257, !dbg !15
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
  %117 = load i64, ptr %benchmark_add.arg1_shape, align 8, !dbg !15
  %118 = icmp eq i64 %117, 17, !dbg !15
  br i1 %118, label %assert_end68, label %assert_fail67, !dbg !15, !prof !16

assert_fail67:                                    ; preds = %assert_end66
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.14, i32 8, ptr nonnull @.str.33, ptr nonnull @.str.39, ptr nonnull @.str.29, ptr nonnull @.str.12, ptr nonnull @.str.10, ptr nonnull @.str.35), !dbg !15
  br label %common.ret, !dbg !15

assert_end68:                                     ; preds = %assert_end66
  %119 = getelementptr inbounds nuw i8, ptr %benchmark_add.arg1_shape, i64 8, !dbg !15
  %120 = load i64, ptr %119, align 8, !dbg !15
  %121 = icmp eq i64 %120, 257, !dbg !15
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
  %125 = load i64, ptr %benchmark_add.arg2_shape, align 8, !dbg !15
  %126 = icmp eq i64 %125, 17, !dbg !15
  br i1 %126, label %assert_end74, label %assert_fail73, !dbg !15, !prof !16

assert_fail73:                                    ; preds = %assert_end72
  tail call fastcc void @__tvm_set_raised_12(ptr nonnull @.str.14, i32 8, ptr nonnull @.str.33, ptr nonnull @.str.42, ptr nonnull @.str.29, ptr nonnull @.str.13, ptr nonnull @.str.10, ptr nonnull @.str.35), !dbg !15
  br label %common.ret, !dbg !15

assert_end74:                                     ; preds = %assert_end72
  %127 = getelementptr inbounds nuw i8, ptr %benchmark_add.arg2_shape, i64 8, !dbg !15
  %128 = load i64, ptr %127, align 8, !dbg !15
  %129 = icmp eq i64 %128, 257, !dbg !15
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
  %133 = tail call fastcc i32 @benchmark_add_compute_(ptr nonnull %88, ptr nonnull %98, ptr nonnull %108, i32 %dev_id), !dbg !15
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
define internal fastcc range(i32 -1, 1) i32 @benchmark_add_compute_(ptr noalias align 64 %arg0, ptr noalias align 64 %arg1, ptr noalias align 64 %arg2, i32 %dev_id) unnamed_addr #3 !dbg !38 {
entry:
    #dbg_declare(ptr %arg0, !46, !DIExpression(), !47)
    #dbg_value(ptr %arg0, !42, !DIExpression(), !47)
    #dbg_value(ptr %arg1, !43, !DIExpression(), !47)
    #dbg_value(ptr %arg2, !44, !DIExpression(), !47)
    #dbg_value(i32 %dev_id, !45, !DIExpression(), !47)
  call void @llvm.assume(i1 true) [ "align"(ptr %arg0, i64 64) ], !dbg !47
    #dbg_declare(ptr %arg1, !48, !DIExpression(), !47)
  call void @llvm.assume(i1 true) [ "align"(ptr %arg1, i64 64) ], !dbg !47
    #dbg_declare(ptr %arg2, !49, !DIExpression(), !47)
  call void @llvm.assume(i1 true) [ "align"(ptr %arg2, i64 64) ], !dbg !47
    #dbg_value(i32 0, !50, !DIExpression(), !47)
  %invariant.gep = getelementptr inbounds nuw i8, ptr %arg0, i64 4
  %invariant.gep65 = getelementptr inbounds nuw i8, ptr %arg0, i64 8
  %invariant.gep69 = getelementptr inbounds nuw i8, ptr %arg0, i64 12
  %invariant.gep73 = getelementptr inbounds nuw i8, ptr %arg0, i64 16
  %invariant.gep77 = getelementptr inbounds nuw i8, ptr %arg0, i64 20
  %invariant.gep81 = getelementptr inbounds nuw i8, ptr %arg0, i64 24
  %invariant.gep85 = getelementptr inbounds nuw i8, ptr %arg0, i64 28
  %invariant.gep89 = getelementptr inbounds nuw i8, ptr %arg0, i64 32
  %invariant.gep93 = getelementptr inbounds nuw i8, ptr %arg0, i64 36
  %invariant.gep97 = getelementptr inbounds nuw i8, ptr %arg0, i64 40
  %invariant.gep101 = getelementptr inbounds nuw i8, ptr %arg0, i64 44
  %invariant.gep105 = getelementptr inbounds nuw i8, ptr %arg0, i64 48
  %invariant.gep109 = getelementptr inbounds nuw i8, ptr %arg0, i64 52
  %invariant.gep113 = getelementptr inbounds nuw i8, ptr %arg0, i64 56
  %invariant.gep117 = getelementptr inbounds nuw i8, ptr %arg0, i64 60
  %invariant.gep121 = getelementptr inbounds nuw i8, ptr %arg1, i64 4
  %invariant.gep125 = getelementptr inbounds nuw i8, ptr %arg1, i64 8
  %invariant.gep129 = getelementptr inbounds nuw i8, ptr %arg1, i64 12
  %invariant.gep133 = getelementptr inbounds nuw i8, ptr %arg1, i64 16
  %invariant.gep137 = getelementptr inbounds nuw i8, ptr %arg1, i64 20
  %invariant.gep141 = getelementptr inbounds nuw i8, ptr %arg1, i64 24
  %invariant.gep145 = getelementptr inbounds nuw i8, ptr %arg1, i64 28
  %invariant.gep149 = getelementptr inbounds nuw i8, ptr %arg1, i64 32
  %invariant.gep153 = getelementptr inbounds nuw i8, ptr %arg1, i64 36
  %invariant.gep157 = getelementptr inbounds nuw i8, ptr %arg1, i64 40
  %invariant.gep161 = getelementptr inbounds nuw i8, ptr %arg1, i64 44
  %invariant.gep165 = getelementptr inbounds nuw i8, ptr %arg1, i64 48
  %invariant.gep169 = getelementptr inbounds nuw i8, ptr %arg1, i64 52
  %invariant.gep173 = getelementptr inbounds nuw i8, ptr %arg1, i64 56
  %invariant.gep177 = getelementptr inbounds nuw i8, ptr %arg1, i64 60
  %invariant.gep181 = getelementptr inbounds nuw i8, ptr %arg2, i64 4
  %invariant.gep183 = getelementptr inbounds nuw i8, ptr %arg2, i64 8
  %invariant.gep185 = getelementptr inbounds nuw i8, ptr %arg2, i64 12
  %invariant.gep187 = getelementptr inbounds nuw i8, ptr %arg2, i64 16
  %invariant.gep189 = getelementptr inbounds nuw i8, ptr %arg2, i64 20
  %invariant.gep191 = getelementptr inbounds nuw i8, ptr %arg2, i64 24
  %invariant.gep193 = getelementptr inbounds nuw i8, ptr %arg2, i64 28
  %invariant.gep195 = getelementptr inbounds nuw i8, ptr %arg2, i64 32
  %invariant.gep197 = getelementptr inbounds nuw i8, ptr %arg2, i64 36
  %invariant.gep199 = getelementptr inbounds nuw i8, ptr %arg2, i64 40
  %invariant.gep201 = getelementptr inbounds nuw i8, ptr %arg2, i64 44
  %invariant.gep203 = getelementptr inbounds nuw i8, ptr %arg2, i64 48
  %invariant.gep205 = getelementptr inbounds nuw i8, ptr %arg2, i64 52
  %invariant.gep207 = getelementptr inbounds nuw i8, ptr %arg2, i64 56
  %invariant.gep209 = getelementptr inbounds nuw i8, ptr %arg2, i64 60
  br label %for_body_parallel_0, !dbg !47

for_begin_parallel_0:                             ; preds = %if_end21
  %0 = add nuw nsw i32 %parallel_033, 1, !dbg !47
    #dbg_value(i32 %0, !50, !DIExpression(), !47)
  %exitcond62.not = icmp eq i32 %0, 34, !dbg !47
  br i1 %exitcond62.not, label %common.ret, label %for_body_parallel_0, !dbg !47, !prof !51

for_body_parallel_0:                              ; preds = %entry, %for_begin_parallel_0
  %parallel_033 = phi i32 [ 0, %entry ], [ %0, %for_begin_parallel_0 ]
    #dbg_value(i32 %parallel_033, !50, !DIExpression(), !47)
  %1 = load ptr, ptr @__TVMBackendAllocWorkspace, align 8, !dbg !47, !tbaa !52
  %tile_storage_0 = tail call ptr %1(i32 1, i32 %dev_id, i64 1024, i32 2, i32 32), !dbg !47
    #dbg_declare(ptr %tile_storage_0, !55, !DIExpression(), !47)
  %2 = icmp eq ptr %tile_storage_0, null, !dbg !47
  br i1 %2, label %common.ret, label %if_end, !dbg !47, !prof !16

common.ret:                                       ; preds = %if_end21, %for_end_tile_i_7_pack, %for_end_tile_i_2_pack, %for_body_parallel_0, %for_begin_parallel_0
  %common.ret.op = phi i32 [ 0, %for_begin_parallel_0 ], [ -1, %for_body_parallel_0 ], [ -1, %for_end_tile_i_2_pack ], [ -1, %for_end_tile_i_7_pack ], [ -1, %if_end21 ]
  ret i32 %common.ret.op, !dbg !47

if_end:                                           ; preds = %for_body_parallel_0
  %cse_v1 = and i32 %parallel_033, 1, !dbg !47
    #dbg_value(i32 %cse_v1, !56, !DIExpression(), !47)
  %cse_v5 = shl nuw nsw i32 %cse_v1, 8, !dbg !47
    #dbg_value(i32 %cse_v5, !57, !DIExpression(), !47)
    #dbg_value(i32 %cse_v1, !58, !DIExpression(DW_OP_constu, 7, DW_OP_shl, DW_OP_stack_value), !47)
    #dbg_value(i32 %cse_v1, !59, !DIExpression(DW_OP_constu, 6, DW_OP_shl, DW_OP_stack_value), !47)
    #dbg_value(i32 %cse_v1, !60, !DIExpression(DW_OP_constu, 5, DW_OP_shl, DW_OP_stack_value), !47)
  %3 = lshr i32 %parallel_033, 1, !dbg !47
  %4 = mul nuw nsw i32 %3, 257, !dbg !47
  %cse_v13 = add nuw nsw i32 %4, %cse_v5, !dbg !47
    #dbg_value(i32 %cse_v13, !61, !DIExpression(), !47)
  %5 = icmp eq i32 %cse_v1, 0
    #dbg_value(i32 0, !62, !DIExpression(), !47)
  %6 = zext nneg i32 %cse_v13 to i64, !dbg !47
  %7 = zext nneg i32 %cse_v5 to i64, !dbg !47
  %invariant.gep63 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 4, !dbg !47
  %invariant.gep67 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 8, !dbg !47
  %invariant.gep71 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 12, !dbg !47
  %invariant.gep75 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 16, !dbg !47
  %invariant.gep79 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 20, !dbg !47
  %invariant.gep83 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 24, !dbg !47
  %invariant.gep87 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 28, !dbg !47
  %invariant.gep91 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 32, !dbg !47
  %invariant.gep95 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 36, !dbg !47
  %invariant.gep99 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 40, !dbg !47
  %invariant.gep103 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 44, !dbg !47
  %invariant.gep107 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 48, !dbg !47
  %invariant.gep111 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 52, !dbg !47
  %invariant.gep115 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 56, !dbg !47
  %invariant.gep119 = getelementptr inbounds nuw i8, ptr %tile_storage_0, i64 60, !dbg !47
  br label %for_body_tile_i_2_pack, !dbg !47

for_body_tile_i_2_pack:                           ; preds = %if_end, %if_end2
  %indvars.iv = phi i64 [ 0, %if_end ], [ %indvars.iv.next, %if_end2 ]
    #dbg_value(i64 %indvars.iv, !62, !DIExpression(), !47)
  %8 = shl nuw nsw i64 %indvars.iv, 4, !dbg !47
    #dbg_value(i64 %8, !63, !DIExpression(), !47)
  %9 = or disjoint i64 %8, %7, !dbg !47
    #dbg_value(i64 %9, !64, !DIExpression(), !47)
    #dbg_value(!DIArgList(i64 %indvars.iv, i32 %cse_v1), !65, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_constu, 3, DW_OP_shl, DW_OP_LLVM_arg, 1, DW_OP_constu, 7, DW_OP_shl, DW_OP_plus, DW_OP_stack_value), !47)
    #dbg_value(!DIArgList(i64 %indvars.iv, i32 %cse_v1), !66, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_constu, 2, DW_OP_shl, DW_OP_LLVM_arg, 1, DW_OP_constu, 6, DW_OP_shl, DW_OP_plus, DW_OP_stack_value), !47)
  %10 = add nuw nsw i64 %8, %6, !dbg !47
    #dbg_value(i64 %10, !67, !DIExpression(), !47)
  %11 = icmp samesign ult i64 %9, 243, !dbg !47
  %12 = and i1 %5, %11, !dbg !47
  br i1 %12, label %if_then1, label %for_body_tile_i_2_lane.s.preheader, !dbg !47

for_body_tile_i_2_lane.s.preheader:               ; preds = %for_body_tile_i_2_pack
    #dbg_value(i64 0, !68, !DIExpression(), !47)
  %13 = icmp samesign ult i64 %9, 257, !dbg !47
  br i1 %13, label %if_then3, label %if_end5, !dbg !47

for_end_tile_i_2_pack:                            ; preds = %if_end2
  %14 = load ptr, ptr @__TVMBackendAllocWorkspace, align 8, !dbg !47, !tbaa !52
  %tile_storage_3 = tail call ptr %14(i32 1, i32 %dev_id, i64 1024, i32 2, i32 32), !dbg !47
    #dbg_declare(ptr %tile_storage_3, !69, !DIExpression(), !47)
  %15 = icmp eq ptr %tile_storage_3, null, !dbg !47
  br i1 %15, label %common.ret, label %for_begin_tile_i_5_pack.preheader, !dbg !47, !prof !16

for_begin_tile_i_5_pack.preheader:                ; preds = %for_end_tile_i_2_pack
    #dbg_value(i32 0, !70, !DIExpression(), !47)
  %invariant.gep123 = getelementptr inbounds nuw i8, ptr %tile_storage_3, i64 4, !dbg !47
  %invariant.gep127 = getelementptr inbounds nuw i8, ptr %tile_storage_3, i64 8, !dbg !47
  %invariant.gep131 = getelementptr inbounds nuw i8, ptr %tile_storage_3, i64 12, !dbg !47
  %invariant.gep135 = getelementptr inbounds nuw i8, ptr %tile_storage_3, i64 16, !dbg !47
  %invariant.gep139 = getelementptr inbounds nuw i8, ptr %tile_storage_3, i64 20, !dbg !47
  %invariant.gep143 = getelementptr inbounds nuw i8, ptr %tile_storage_3, i64 24, !dbg !47
  %invariant.gep147 = getelementptr inbounds nuw i8, ptr %tile_storage_3, i64 28, !dbg !47
  %invariant.gep151 = getelementptr inbounds nuw i8, ptr %tile_storage_3, i64 32, !dbg !47
  %invariant.gep155 = getelementptr inbounds nuw i8, ptr %tile_storage_3, i64 36, !dbg !47
  %invariant.gep159 = getelementptr inbounds nuw i8, ptr %tile_storage_3, i64 40, !dbg !47
  %invariant.gep163 = getelementptr inbounds nuw i8, ptr %tile_storage_3, i64 44, !dbg !47
  %invariant.gep167 = getelementptr inbounds nuw i8, ptr %tile_storage_3, i64 48, !dbg !47
  %invariant.gep171 = getelementptr inbounds nuw i8, ptr %tile_storage_3, i64 52, !dbg !47
  %invariant.gep175 = getelementptr inbounds nuw i8, ptr %tile_storage_3, i64 56, !dbg !47
  %invariant.gep179 = getelementptr inbounds nuw i8, ptr %tile_storage_3, i64 60, !dbg !47
  br label %for_body_tile_i_5_pack, !dbg !47

if_then1:                                         ; preds = %for_body_tile_i_2_pack
  %16 = getelementptr inbounds nuw float, ptr %arg0, i64 %10, !dbg !47
  %17 = load <16 x float>, ptr %16, align 4, !dbg !47, !tbaa !71
  %18 = getelementptr inbounds nuw float, ptr %tile_storage_0, i64 %8, !dbg !47
  store <16 x float> %17, ptr %18, align 4, !dbg !47, !tbaa !73
  br label %if_end2, !dbg !47

if_end2:                                          ; preds = %if_end5.15, %if_then1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !47
    #dbg_value(i64 %indvars.iv.next, !62, !DIExpression(), !47)
  %exitcond.not = icmp eq i64 %indvars.iv.next, 16, !dbg !47
  br i1 %exitcond.not, label %for_end_tile_i_2_pack, label %for_body_tile_i_2_pack, !dbg !47, !prof !51

if_then3:                                         ; preds = %for_body_tile_i_2_lane.s.preheader
  %19 = getelementptr inbounds nuw float, ptr %arg0, i64 %10, !dbg !47
  %20 = load float, ptr %19, align 4, !dbg !47, !tbaa !71
  br label %if_end5, !dbg !47

if_end5:                                          ; preds = %for_body_tile_i_2_lane.s.preheader, %if_then3
  %21 = phi float [ %20, %if_then3 ], [ 0.000000e+00, %for_body_tile_i_2_lane.s.preheader ], !dbg !47
  %22 = getelementptr inbounds nuw float, ptr %tile_storage_0, i64 %8, !dbg !47
  store float %21, ptr %22, align 4, !dbg !47, !tbaa !73
    #dbg_value(i64 1, !68, !DIExpression(), !47)
  %23 = or disjoint i64 %9, 1, !dbg !47
  %24 = icmp samesign ult i64 %23, 257, !dbg !47
  br i1 %24, label %if_then3.1, label %if_end5.1, !dbg !47

if_then3.1:                                       ; preds = %if_end5
  %gep = getelementptr inbounds nuw float, ptr %invariant.gep, i64 %10, !dbg !47
  %25 = load float, ptr %gep, align 4, !dbg !47, !tbaa !71
  br label %if_end5.1, !dbg !47

if_end5.1:                                        ; preds = %if_then3.1, %if_end5
  %26 = phi float [ %25, %if_then3.1 ], [ 0.000000e+00, %if_end5 ], !dbg !47
  %gep64 = getelementptr inbounds nuw float, ptr %invariant.gep63, i64 %8, !dbg !47
  store float %26, ptr %gep64, align 4, !dbg !47, !tbaa !73
    #dbg_value(i64 2, !68, !DIExpression(), !47)
  %27 = or disjoint i64 %9, 2, !dbg !47
  %28 = icmp samesign ult i64 %27, 257, !dbg !47
  br i1 %28, label %if_then3.2, label %if_end5.2, !dbg !47

if_then3.2:                                       ; preds = %if_end5.1
  %gep66 = getelementptr inbounds nuw float, ptr %invariant.gep65, i64 %10, !dbg !47
  %29 = load float, ptr %gep66, align 4, !dbg !47, !tbaa !71
  br label %if_end5.2, !dbg !47

if_end5.2:                                        ; preds = %if_then3.2, %if_end5.1
  %30 = phi float [ %29, %if_then3.2 ], [ 0.000000e+00, %if_end5.1 ], !dbg !47
  %gep68 = getelementptr inbounds nuw float, ptr %invariant.gep67, i64 %8, !dbg !47
  store float %30, ptr %gep68, align 4, !dbg !47, !tbaa !73
    #dbg_value(i64 3, !68, !DIExpression(), !47)
  %31 = or disjoint i64 %9, 3, !dbg !47
  %32 = icmp samesign ult i64 %31, 257, !dbg !47
  br i1 %32, label %if_then3.3, label %if_end5.3, !dbg !47

if_then3.3:                                       ; preds = %if_end5.2
  %gep70 = getelementptr inbounds nuw float, ptr %invariant.gep69, i64 %10, !dbg !47
  %33 = load float, ptr %gep70, align 4, !dbg !47, !tbaa !71
  br label %if_end5.3, !dbg !47

if_end5.3:                                        ; preds = %if_then3.3, %if_end5.2
  %34 = phi float [ %33, %if_then3.3 ], [ 0.000000e+00, %if_end5.2 ], !dbg !47
  %gep72 = getelementptr inbounds nuw float, ptr %invariant.gep71, i64 %8, !dbg !47
  store float %34, ptr %gep72, align 4, !dbg !47, !tbaa !73
    #dbg_value(i64 4, !68, !DIExpression(), !47)
  %35 = or disjoint i64 %9, 4, !dbg !47
  %36 = icmp samesign ult i64 %35, 257, !dbg !47
  br i1 %36, label %if_then3.4, label %if_end5.4, !dbg !47

if_then3.4:                                       ; preds = %if_end5.3
  %gep74 = getelementptr inbounds nuw float, ptr %invariant.gep73, i64 %10, !dbg !47
  %37 = load float, ptr %gep74, align 4, !dbg !47, !tbaa !71
  br label %if_end5.4, !dbg !47

if_end5.4:                                        ; preds = %if_then3.4, %if_end5.3
  %38 = phi float [ %37, %if_then3.4 ], [ 0.000000e+00, %if_end5.3 ], !dbg !47
  %gep76 = getelementptr inbounds nuw float, ptr %invariant.gep75, i64 %8, !dbg !47
  store float %38, ptr %gep76, align 4, !dbg !47, !tbaa !73
    #dbg_value(i64 5, !68, !DIExpression(), !47)
  %39 = or disjoint i64 %9, 5, !dbg !47
  %40 = icmp samesign ult i64 %39, 257, !dbg !47
  br i1 %40, label %if_then3.5, label %if_end5.5, !dbg !47

if_then3.5:                                       ; preds = %if_end5.4
  %gep78 = getelementptr inbounds nuw float, ptr %invariant.gep77, i64 %10, !dbg !47
  %41 = load float, ptr %gep78, align 4, !dbg !47, !tbaa !71
  br label %if_end5.5, !dbg !47

if_end5.5:                                        ; preds = %if_then3.5, %if_end5.4
  %42 = phi float [ %41, %if_then3.5 ], [ 0.000000e+00, %if_end5.4 ], !dbg !47
  %gep80 = getelementptr inbounds nuw float, ptr %invariant.gep79, i64 %8, !dbg !47
  store float %42, ptr %gep80, align 4, !dbg !47, !tbaa !73
    #dbg_value(i64 6, !68, !DIExpression(), !47)
  %43 = or disjoint i64 %9, 6, !dbg !47
  %44 = icmp samesign ult i64 %43, 257, !dbg !47
  br i1 %44, label %if_then3.6, label %if_end5.6, !dbg !47

if_then3.6:                                       ; preds = %if_end5.5
  %gep82 = getelementptr inbounds nuw float, ptr %invariant.gep81, i64 %10, !dbg !47
  %45 = load float, ptr %gep82, align 4, !dbg !47, !tbaa !71
  br label %if_end5.6, !dbg !47

if_end5.6:                                        ; preds = %if_then3.6, %if_end5.5
  %46 = phi float [ %45, %if_then3.6 ], [ 0.000000e+00, %if_end5.5 ], !dbg !47
  %gep84 = getelementptr inbounds nuw float, ptr %invariant.gep83, i64 %8, !dbg !47
  store float %46, ptr %gep84, align 4, !dbg !47, !tbaa !73
    #dbg_value(i64 7, !68, !DIExpression(), !47)
  %47 = or disjoint i64 %9, 7, !dbg !47
  %48 = icmp samesign ult i64 %47, 257, !dbg !47
  br i1 %48, label %if_then3.7, label %if_end5.7, !dbg !47

if_then3.7:                                       ; preds = %if_end5.6
  %gep86 = getelementptr inbounds nuw float, ptr %invariant.gep85, i64 %10, !dbg !47
  %49 = load float, ptr %gep86, align 4, !dbg !47, !tbaa !71
  br label %if_end5.7, !dbg !47

if_end5.7:                                        ; preds = %if_then3.7, %if_end5.6
  %50 = phi float [ %49, %if_then3.7 ], [ 0.000000e+00, %if_end5.6 ], !dbg !47
  %gep88 = getelementptr inbounds nuw float, ptr %invariant.gep87, i64 %8, !dbg !47
  store float %50, ptr %gep88, align 4, !dbg !47, !tbaa !73
    #dbg_value(i64 8, !68, !DIExpression(), !47)
  %51 = or disjoint i64 %9, 8, !dbg !47
  %52 = icmp samesign ult i64 %51, 257, !dbg !47
  br i1 %52, label %if_then3.8, label %if_end5.8, !dbg !47

if_then3.8:                                       ; preds = %if_end5.7
  %gep90 = getelementptr inbounds nuw float, ptr %invariant.gep89, i64 %10, !dbg !47
  %53 = load float, ptr %gep90, align 4, !dbg !47, !tbaa !71
  br label %if_end5.8, !dbg !47

if_end5.8:                                        ; preds = %if_then3.8, %if_end5.7
  %54 = phi float [ %53, %if_then3.8 ], [ 0.000000e+00, %if_end5.7 ], !dbg !47
  %gep92 = getelementptr inbounds nuw float, ptr %invariant.gep91, i64 %8, !dbg !47
  store float %54, ptr %gep92, align 4, !dbg !47, !tbaa !73
    #dbg_value(i64 9, !68, !DIExpression(), !47)
  %55 = or disjoint i64 %9, 9, !dbg !47
  %56 = icmp samesign ult i64 %55, 257, !dbg !47
  br i1 %56, label %if_then3.9, label %if_end5.9, !dbg !47

if_then3.9:                                       ; preds = %if_end5.8
  %gep94 = getelementptr inbounds nuw float, ptr %invariant.gep93, i64 %10, !dbg !47
  %57 = load float, ptr %gep94, align 4, !dbg !47, !tbaa !71
  br label %if_end5.9, !dbg !47

if_end5.9:                                        ; preds = %if_then3.9, %if_end5.8
  %58 = phi float [ %57, %if_then3.9 ], [ 0.000000e+00, %if_end5.8 ], !dbg !47
  %gep96 = getelementptr inbounds nuw float, ptr %invariant.gep95, i64 %8, !dbg !47
  store float %58, ptr %gep96, align 4, !dbg !47, !tbaa !73
    #dbg_value(i64 10, !68, !DIExpression(), !47)
  %59 = or disjoint i64 %9, 10, !dbg !47
  %60 = icmp samesign ult i64 %59, 257, !dbg !47
  br i1 %60, label %if_then3.10, label %if_end5.10, !dbg !47

if_then3.10:                                      ; preds = %if_end5.9
  %gep98 = getelementptr inbounds nuw float, ptr %invariant.gep97, i64 %10, !dbg !47
  %61 = load float, ptr %gep98, align 4, !dbg !47, !tbaa !71
  br label %if_end5.10, !dbg !47

if_end5.10:                                       ; preds = %if_then3.10, %if_end5.9
  %62 = phi float [ %61, %if_then3.10 ], [ 0.000000e+00, %if_end5.9 ], !dbg !47
  %gep100 = getelementptr inbounds nuw float, ptr %invariant.gep99, i64 %8, !dbg !47
  store float %62, ptr %gep100, align 4, !dbg !47, !tbaa !73
    #dbg_value(i64 11, !68, !DIExpression(), !47)
  %63 = or disjoint i64 %9, 11, !dbg !47
  %64 = icmp samesign ult i64 %63, 257, !dbg !47
  br i1 %64, label %if_then3.11, label %if_end5.11, !dbg !47

if_then3.11:                                      ; preds = %if_end5.10
  %gep102 = getelementptr inbounds nuw float, ptr %invariant.gep101, i64 %10, !dbg !47
  %65 = load float, ptr %gep102, align 4, !dbg !47, !tbaa !71
  br label %if_end5.11, !dbg !47

if_end5.11:                                       ; preds = %if_then3.11, %if_end5.10
  %66 = phi float [ %65, %if_then3.11 ], [ 0.000000e+00, %if_end5.10 ], !dbg !47
  %gep104 = getelementptr inbounds nuw float, ptr %invariant.gep103, i64 %8, !dbg !47
  store float %66, ptr %gep104, align 4, !dbg !47, !tbaa !73
    #dbg_value(i64 12, !68, !DIExpression(), !47)
  %67 = or disjoint i64 %9, 12, !dbg !47
  %68 = icmp samesign ult i64 %67, 257, !dbg !47
  br i1 %68, label %if_then3.12, label %if_end5.12, !dbg !47

if_then3.12:                                      ; preds = %if_end5.11
  %gep106 = getelementptr inbounds nuw float, ptr %invariant.gep105, i64 %10, !dbg !47
  %69 = load float, ptr %gep106, align 4, !dbg !47, !tbaa !71
  br label %if_end5.12, !dbg !47

if_end5.12:                                       ; preds = %if_then3.12, %if_end5.11
  %70 = phi float [ %69, %if_then3.12 ], [ 0.000000e+00, %if_end5.11 ], !dbg !47
  %gep108 = getelementptr inbounds nuw float, ptr %invariant.gep107, i64 %8, !dbg !47
  store float %70, ptr %gep108, align 4, !dbg !47, !tbaa !73
    #dbg_value(i64 13, !68, !DIExpression(), !47)
  %71 = or disjoint i64 %9, 13, !dbg !47
  %72 = icmp samesign ult i64 %71, 257, !dbg !47
  br i1 %72, label %if_then3.13, label %if_end5.13, !dbg !47

if_then3.13:                                      ; preds = %if_end5.12
  %gep110 = getelementptr inbounds nuw float, ptr %invariant.gep109, i64 %10, !dbg !47
  %73 = load float, ptr %gep110, align 4, !dbg !47, !tbaa !71
  br label %if_end5.13, !dbg !47

if_end5.13:                                       ; preds = %if_then3.13, %if_end5.12
  %74 = phi float [ %73, %if_then3.13 ], [ 0.000000e+00, %if_end5.12 ], !dbg !47
  %gep112 = getelementptr inbounds nuw float, ptr %invariant.gep111, i64 %8, !dbg !47
  store float %74, ptr %gep112, align 4, !dbg !47, !tbaa !73
    #dbg_value(i64 14, !68, !DIExpression(), !47)
  %75 = or disjoint i64 %9, 14, !dbg !47
  %76 = icmp samesign ult i64 %75, 257, !dbg !47
  br i1 %76, label %if_then3.14, label %if_end5.14, !dbg !47

if_then3.14:                                      ; preds = %if_end5.13
  %gep114 = getelementptr inbounds nuw float, ptr %invariant.gep113, i64 %10, !dbg !47
  %77 = load float, ptr %gep114, align 4, !dbg !47, !tbaa !71
  br label %if_end5.14, !dbg !47

if_end5.14:                                       ; preds = %if_then3.14, %if_end5.13
  %78 = phi float [ %77, %if_then3.14 ], [ 0.000000e+00, %if_end5.13 ], !dbg !47
  %gep116 = getelementptr inbounds nuw float, ptr %invariant.gep115, i64 %8, !dbg !47
  store float %78, ptr %gep116, align 4, !dbg !47, !tbaa !73
    #dbg_value(i64 15, !68, !DIExpression(), !47)
  %79 = or disjoint i64 %9, 15, !dbg !47
  %80 = icmp samesign ult i64 %79, 257, !dbg !47
  br i1 %80, label %if_then3.15, label %if_end5.15, !dbg !47

if_then3.15:                                      ; preds = %if_end5.14
  %gep118 = getelementptr inbounds nuw float, ptr %invariant.gep117, i64 %10, !dbg !47
  %81 = load float, ptr %gep118, align 4, !dbg !47, !tbaa !71
  br label %if_end5.15, !dbg !47

if_end5.15:                                       ; preds = %if_then3.15, %if_end5.14
  %82 = phi float [ %81, %if_then3.15 ], [ 0.000000e+00, %if_end5.14 ], !dbg !47
  %gep120 = getelementptr inbounds nuw float, ptr %invariant.gep119, i64 %8, !dbg !47
  store float %82, ptr %gep120, align 4, !dbg !47, !tbaa !73
    #dbg_value(i64 16, !68, !DIExpression(), !47)
  br label %if_end2, !dbg !47

for_body_tile_i_5_pack:                           ; preds = %for_begin_tile_i_5_pack.preheader, %if_end10
  %indvars.iv43 = phi i64 [ 0, %for_begin_tile_i_5_pack.preheader ], [ %indvars.iv.next44, %if_end10 ]
    #dbg_value(i64 %indvars.iv43, !70, !DIExpression(), !47)
  %83 = shl nuw nsw i64 %indvars.iv43, 4, !dbg !47
    #dbg_value(i64 %83, !75, !DIExpression(), !47)
  %84 = or disjoint i64 %83, %7, !dbg !47
    #dbg_value(i64 %84, !76, !DIExpression(), !47)
    #dbg_value(!DIArgList(i64 %indvars.iv43, i32 %cse_v1), !77, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_constu, 3, DW_OP_shl, DW_OP_LLVM_arg, 1, DW_OP_constu, 7, DW_OP_shl, DW_OP_plus, DW_OP_stack_value), !47)
    #dbg_value(!DIArgList(i64 %indvars.iv43, i32 %cse_v1), !78, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_constu, 2, DW_OP_shl, DW_OP_LLVM_arg, 1, DW_OP_constu, 6, DW_OP_shl, DW_OP_plus, DW_OP_stack_value), !47)
  %85 = add nuw nsw i64 %83, %6, !dbg !47
    #dbg_value(i64 %85, !79, !DIExpression(), !47)
  %86 = icmp samesign ult i64 %84, 243, !dbg !47
  %87 = and i1 %5, %86, !dbg !47
  br i1 %87, label %if_then9, label %for_body_tile_i_5_lane.s.preheader, !dbg !47

for_body_tile_i_5_lane.s.preheader:               ; preds = %for_body_tile_i_5_pack
    #dbg_value(i64 0, !80, !DIExpression(), !47)
  %88 = icmp samesign ult i64 %84, 257, !dbg !47
  br i1 %88, label %if_then12, label %if_end14, !dbg !47

if_then9:                                         ; preds = %for_body_tile_i_5_pack
  %89 = getelementptr inbounds nuw float, ptr %arg1, i64 %85, !dbg !47
  %90 = load <16 x float>, ptr %89, align 4, !dbg !47, !tbaa !81
  %91 = getelementptr inbounds nuw float, ptr %tile_storage_3, i64 %83, !dbg !47
  store <16 x float> %90, ptr %91, align 4, !dbg !47, !tbaa !83
  br label %if_end10, !dbg !47

if_end10:                                         ; preds = %if_end14.15, %if_then9
  %indvars.iv.next44 = add nuw nsw i64 %indvars.iv43, 1, !dbg !47
    #dbg_value(i64 %indvars.iv.next44, !70, !DIExpression(), !47)
  %exitcond49.not = icmp eq i64 %indvars.iv.next44, 16, !dbg !47
  br i1 %exitcond49.not, label %for_body_tile_i_7_pack, label %for_body_tile_i_5_pack, !dbg !47, !prof !51

if_then12:                                        ; preds = %for_body_tile_i_5_lane.s.preheader
  %92 = getelementptr inbounds nuw float, ptr %arg1, i64 %85, !dbg !47
  %93 = load float, ptr %92, align 4, !dbg !47, !tbaa !81
  br label %if_end14, !dbg !47

if_end14:                                         ; preds = %for_body_tile_i_5_lane.s.preheader, %if_then12
  %94 = phi float [ %93, %if_then12 ], [ 0.000000e+00, %for_body_tile_i_5_lane.s.preheader ], !dbg !47
  %95 = getelementptr inbounds nuw float, ptr %tile_storage_3, i64 %83, !dbg !47
  store float %94, ptr %95, align 4, !dbg !47, !tbaa !83
    #dbg_value(i64 1, !80, !DIExpression(), !47)
  %96 = or disjoint i64 %84, 1, !dbg !47
  %97 = icmp samesign ult i64 %96, 257, !dbg !47
  br i1 %97, label %if_then12.1, label %if_end14.1, !dbg !47

if_then12.1:                                      ; preds = %if_end14
  %gep122 = getelementptr inbounds nuw float, ptr %invariant.gep121, i64 %85, !dbg !47
  %98 = load float, ptr %gep122, align 4, !dbg !47, !tbaa !81
  br label %if_end14.1, !dbg !47

if_end14.1:                                       ; preds = %if_then12.1, %if_end14
  %99 = phi float [ %98, %if_then12.1 ], [ 0.000000e+00, %if_end14 ], !dbg !47
  %gep124 = getelementptr inbounds nuw float, ptr %invariant.gep123, i64 %83, !dbg !47
  store float %99, ptr %gep124, align 4, !dbg !47, !tbaa !83
    #dbg_value(i64 2, !80, !DIExpression(), !47)
  %100 = or disjoint i64 %84, 2, !dbg !47
  %101 = icmp samesign ult i64 %100, 257, !dbg !47
  br i1 %101, label %if_then12.2, label %if_end14.2, !dbg !47

if_then12.2:                                      ; preds = %if_end14.1
  %gep126 = getelementptr inbounds nuw float, ptr %invariant.gep125, i64 %85, !dbg !47
  %102 = load float, ptr %gep126, align 4, !dbg !47, !tbaa !81
  br label %if_end14.2, !dbg !47

if_end14.2:                                       ; preds = %if_then12.2, %if_end14.1
  %103 = phi float [ %102, %if_then12.2 ], [ 0.000000e+00, %if_end14.1 ], !dbg !47
  %gep128 = getelementptr inbounds nuw float, ptr %invariant.gep127, i64 %83, !dbg !47
  store float %103, ptr %gep128, align 4, !dbg !47, !tbaa !83
    #dbg_value(i64 3, !80, !DIExpression(), !47)
  %104 = or disjoint i64 %84, 3, !dbg !47
  %105 = icmp samesign ult i64 %104, 257, !dbg !47
  br i1 %105, label %if_then12.3, label %if_end14.3, !dbg !47

if_then12.3:                                      ; preds = %if_end14.2
  %gep130 = getelementptr inbounds nuw float, ptr %invariant.gep129, i64 %85, !dbg !47
  %106 = load float, ptr %gep130, align 4, !dbg !47, !tbaa !81
  br label %if_end14.3, !dbg !47

if_end14.3:                                       ; preds = %if_then12.3, %if_end14.2
  %107 = phi float [ %106, %if_then12.3 ], [ 0.000000e+00, %if_end14.2 ], !dbg !47
  %gep132 = getelementptr inbounds nuw float, ptr %invariant.gep131, i64 %83, !dbg !47
  store float %107, ptr %gep132, align 4, !dbg !47, !tbaa !83
    #dbg_value(i64 4, !80, !DIExpression(), !47)
  %108 = or disjoint i64 %84, 4, !dbg !47
  %109 = icmp samesign ult i64 %108, 257, !dbg !47
  br i1 %109, label %if_then12.4, label %if_end14.4, !dbg !47

if_then12.4:                                      ; preds = %if_end14.3
  %gep134 = getelementptr inbounds nuw float, ptr %invariant.gep133, i64 %85, !dbg !47
  %110 = load float, ptr %gep134, align 4, !dbg !47, !tbaa !81
  br label %if_end14.4, !dbg !47

if_end14.4:                                       ; preds = %if_then12.4, %if_end14.3
  %111 = phi float [ %110, %if_then12.4 ], [ 0.000000e+00, %if_end14.3 ], !dbg !47
  %gep136 = getelementptr inbounds nuw float, ptr %invariant.gep135, i64 %83, !dbg !47
  store float %111, ptr %gep136, align 4, !dbg !47, !tbaa !83
    #dbg_value(i64 5, !80, !DIExpression(), !47)
  %112 = or disjoint i64 %84, 5, !dbg !47
  %113 = icmp samesign ult i64 %112, 257, !dbg !47
  br i1 %113, label %if_then12.5, label %if_end14.5, !dbg !47

if_then12.5:                                      ; preds = %if_end14.4
  %gep138 = getelementptr inbounds nuw float, ptr %invariant.gep137, i64 %85, !dbg !47
  %114 = load float, ptr %gep138, align 4, !dbg !47, !tbaa !81
  br label %if_end14.5, !dbg !47

if_end14.5:                                       ; preds = %if_then12.5, %if_end14.4
  %115 = phi float [ %114, %if_then12.5 ], [ 0.000000e+00, %if_end14.4 ], !dbg !47
  %gep140 = getelementptr inbounds nuw float, ptr %invariant.gep139, i64 %83, !dbg !47
  store float %115, ptr %gep140, align 4, !dbg !47, !tbaa !83
    #dbg_value(i64 6, !80, !DIExpression(), !47)
  %116 = or disjoint i64 %84, 6, !dbg !47
  %117 = icmp samesign ult i64 %116, 257, !dbg !47
  br i1 %117, label %if_then12.6, label %if_end14.6, !dbg !47

if_then12.6:                                      ; preds = %if_end14.5
  %gep142 = getelementptr inbounds nuw float, ptr %invariant.gep141, i64 %85, !dbg !47
  %118 = load float, ptr %gep142, align 4, !dbg !47, !tbaa !81
  br label %if_end14.6, !dbg !47

if_end14.6:                                       ; preds = %if_then12.6, %if_end14.5
  %119 = phi float [ %118, %if_then12.6 ], [ 0.000000e+00, %if_end14.5 ], !dbg !47
  %gep144 = getelementptr inbounds nuw float, ptr %invariant.gep143, i64 %83, !dbg !47
  store float %119, ptr %gep144, align 4, !dbg !47, !tbaa !83
    #dbg_value(i64 7, !80, !DIExpression(), !47)
  %120 = or disjoint i64 %84, 7, !dbg !47
  %121 = icmp samesign ult i64 %120, 257, !dbg !47
  br i1 %121, label %if_then12.7, label %if_end14.7, !dbg !47

if_then12.7:                                      ; preds = %if_end14.6
  %gep146 = getelementptr inbounds nuw float, ptr %invariant.gep145, i64 %85, !dbg !47
  %122 = load float, ptr %gep146, align 4, !dbg !47, !tbaa !81
  br label %if_end14.7, !dbg !47

if_end14.7:                                       ; preds = %if_then12.7, %if_end14.6
  %123 = phi float [ %122, %if_then12.7 ], [ 0.000000e+00, %if_end14.6 ], !dbg !47
  %gep148 = getelementptr inbounds nuw float, ptr %invariant.gep147, i64 %83, !dbg !47
  store float %123, ptr %gep148, align 4, !dbg !47, !tbaa !83
    #dbg_value(i64 8, !80, !DIExpression(), !47)
  %124 = or disjoint i64 %84, 8, !dbg !47
  %125 = icmp samesign ult i64 %124, 257, !dbg !47
  br i1 %125, label %if_then12.8, label %if_end14.8, !dbg !47

if_then12.8:                                      ; preds = %if_end14.7
  %gep150 = getelementptr inbounds nuw float, ptr %invariant.gep149, i64 %85, !dbg !47
  %126 = load float, ptr %gep150, align 4, !dbg !47, !tbaa !81
  br label %if_end14.8, !dbg !47

if_end14.8:                                       ; preds = %if_then12.8, %if_end14.7
  %127 = phi float [ %126, %if_then12.8 ], [ 0.000000e+00, %if_end14.7 ], !dbg !47
  %gep152 = getelementptr inbounds nuw float, ptr %invariant.gep151, i64 %83, !dbg !47
  store float %127, ptr %gep152, align 4, !dbg !47, !tbaa !83
    #dbg_value(i64 9, !80, !DIExpression(), !47)
  %128 = or disjoint i64 %84, 9, !dbg !47
  %129 = icmp samesign ult i64 %128, 257, !dbg !47
  br i1 %129, label %if_then12.9, label %if_end14.9, !dbg !47

if_then12.9:                                      ; preds = %if_end14.8
  %gep154 = getelementptr inbounds nuw float, ptr %invariant.gep153, i64 %85, !dbg !47
  %130 = load float, ptr %gep154, align 4, !dbg !47, !tbaa !81
  br label %if_end14.9, !dbg !47

if_end14.9:                                       ; preds = %if_then12.9, %if_end14.8
  %131 = phi float [ %130, %if_then12.9 ], [ 0.000000e+00, %if_end14.8 ], !dbg !47
  %gep156 = getelementptr inbounds nuw float, ptr %invariant.gep155, i64 %83, !dbg !47
  store float %131, ptr %gep156, align 4, !dbg !47, !tbaa !83
    #dbg_value(i64 10, !80, !DIExpression(), !47)
  %132 = or disjoint i64 %84, 10, !dbg !47
  %133 = icmp samesign ult i64 %132, 257, !dbg !47
  br i1 %133, label %if_then12.10, label %if_end14.10, !dbg !47

if_then12.10:                                     ; preds = %if_end14.9
  %gep158 = getelementptr inbounds nuw float, ptr %invariant.gep157, i64 %85, !dbg !47
  %134 = load float, ptr %gep158, align 4, !dbg !47, !tbaa !81
  br label %if_end14.10, !dbg !47

if_end14.10:                                      ; preds = %if_then12.10, %if_end14.9
  %135 = phi float [ %134, %if_then12.10 ], [ 0.000000e+00, %if_end14.9 ], !dbg !47
  %gep160 = getelementptr inbounds nuw float, ptr %invariant.gep159, i64 %83, !dbg !47
  store float %135, ptr %gep160, align 4, !dbg !47, !tbaa !83
    #dbg_value(i64 11, !80, !DIExpression(), !47)
  %136 = or disjoint i64 %84, 11, !dbg !47
  %137 = icmp samesign ult i64 %136, 257, !dbg !47
  br i1 %137, label %if_then12.11, label %if_end14.11, !dbg !47

if_then12.11:                                     ; preds = %if_end14.10
  %gep162 = getelementptr inbounds nuw float, ptr %invariant.gep161, i64 %85, !dbg !47
  %138 = load float, ptr %gep162, align 4, !dbg !47, !tbaa !81
  br label %if_end14.11, !dbg !47

if_end14.11:                                      ; preds = %if_then12.11, %if_end14.10
  %139 = phi float [ %138, %if_then12.11 ], [ 0.000000e+00, %if_end14.10 ], !dbg !47
  %gep164 = getelementptr inbounds nuw float, ptr %invariant.gep163, i64 %83, !dbg !47
  store float %139, ptr %gep164, align 4, !dbg !47, !tbaa !83
    #dbg_value(i64 12, !80, !DIExpression(), !47)
  %140 = or disjoint i64 %84, 12, !dbg !47
  %141 = icmp samesign ult i64 %140, 257, !dbg !47
  br i1 %141, label %if_then12.12, label %if_end14.12, !dbg !47

if_then12.12:                                     ; preds = %if_end14.11
  %gep166 = getelementptr inbounds nuw float, ptr %invariant.gep165, i64 %85, !dbg !47
  %142 = load float, ptr %gep166, align 4, !dbg !47, !tbaa !81
  br label %if_end14.12, !dbg !47

if_end14.12:                                      ; preds = %if_then12.12, %if_end14.11
  %143 = phi float [ %142, %if_then12.12 ], [ 0.000000e+00, %if_end14.11 ], !dbg !47
  %gep168 = getelementptr inbounds nuw float, ptr %invariant.gep167, i64 %83, !dbg !47
  store float %143, ptr %gep168, align 4, !dbg !47, !tbaa !83
    #dbg_value(i64 13, !80, !DIExpression(), !47)
  %144 = or disjoint i64 %84, 13, !dbg !47
  %145 = icmp samesign ult i64 %144, 257, !dbg !47
  br i1 %145, label %if_then12.13, label %if_end14.13, !dbg !47

if_then12.13:                                     ; preds = %if_end14.12
  %gep170 = getelementptr inbounds nuw float, ptr %invariant.gep169, i64 %85, !dbg !47
  %146 = load float, ptr %gep170, align 4, !dbg !47, !tbaa !81
  br label %if_end14.13, !dbg !47

if_end14.13:                                      ; preds = %if_then12.13, %if_end14.12
  %147 = phi float [ %146, %if_then12.13 ], [ 0.000000e+00, %if_end14.12 ], !dbg !47
  %gep172 = getelementptr inbounds nuw float, ptr %invariant.gep171, i64 %83, !dbg !47
  store float %147, ptr %gep172, align 4, !dbg !47, !tbaa !83
    #dbg_value(i64 14, !80, !DIExpression(), !47)
  %148 = or disjoint i64 %84, 14, !dbg !47
  %149 = icmp samesign ult i64 %148, 257, !dbg !47
  br i1 %149, label %if_then12.14, label %if_end14.14, !dbg !47

if_then12.14:                                     ; preds = %if_end14.13
  %gep174 = getelementptr inbounds nuw float, ptr %invariant.gep173, i64 %85, !dbg !47
  %150 = load float, ptr %gep174, align 4, !dbg !47, !tbaa !81
  br label %if_end14.14, !dbg !47

if_end14.14:                                      ; preds = %if_then12.14, %if_end14.13
  %151 = phi float [ %150, %if_then12.14 ], [ 0.000000e+00, %if_end14.13 ], !dbg !47
  %gep176 = getelementptr inbounds nuw float, ptr %invariant.gep175, i64 %83, !dbg !47
  store float %151, ptr %gep176, align 4, !dbg !47, !tbaa !83
    #dbg_value(i64 15, !80, !DIExpression(), !47)
  %152 = or disjoint i64 %84, 15, !dbg !47
  %153 = icmp samesign ult i64 %152, 257, !dbg !47
  br i1 %153, label %if_then12.15, label %if_end14.15, !dbg !47

if_then12.15:                                     ; preds = %if_end14.14
  %gep178 = getelementptr inbounds nuw float, ptr %invariant.gep177, i64 %85, !dbg !47
  %154 = load float, ptr %gep178, align 4, !dbg !47, !tbaa !81
  br label %if_end14.15, !dbg !47

if_end14.15:                                      ; preds = %if_then12.15, %if_end14.14
  %155 = phi float [ %154, %if_then12.15 ], [ 0.000000e+00, %if_end14.14 ], !dbg !47
  %gep180 = getelementptr inbounds nuw float, ptr %invariant.gep179, i64 %83, !dbg !47
  store float %155, ptr %gep180, align 4, !dbg !47, !tbaa !83
    #dbg_value(i64 16, !80, !DIExpression(), !47)
  br label %if_end10, !dbg !47

for_body_tile_i_7_pack:                           ; preds = %if_end10, %if_end16
  %indvars.iv55 = phi i64 [ %indvars.iv.next56, %if_end16 ], [ 0, %if_end10 ]
    #dbg_value(i64 %indvars.iv55, !85, !DIExpression(), !47)
  %156 = shl nuw nsw i64 %indvars.iv55, 4, !dbg !47
    #dbg_value(i64 %156, !86, !DIExpression(), !47)
  %157 = or disjoint i64 %156, %7, !dbg !47
    #dbg_value(i64 %157, !87, !DIExpression(), !47)
    #dbg_value(!DIArgList(i64 %indvars.iv55, i32 %cse_v1), !88, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_constu, 3, DW_OP_shl, DW_OP_LLVM_arg, 1, DW_OP_constu, 7, DW_OP_shl, DW_OP_plus, DW_OP_stack_value), !47)
    #dbg_value(!DIArgList(i64 %indvars.iv55, i32 %cse_v1), !89, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_constu, 2, DW_OP_shl, DW_OP_LLVM_arg, 1, DW_OP_constu, 6, DW_OP_shl, DW_OP_plus, DW_OP_stack_value), !47)
  %158 = add nuw nsw i64 %156, %6, !dbg !47
    #dbg_value(i64 %158, !90, !DIExpression(), !47)
  %159 = icmp samesign ult i64 %157, 243, !dbg !47
  %160 = and i1 %5, %159, !dbg !47
  br i1 %160, label %if_then15, label %for_body_tile_i_7_lane.s.preheader, !dbg !47

for_body_tile_i_7_lane.s.preheader:               ; preds = %for_body_tile_i_7_pack
    #dbg_value(i64 0, !91, !DIExpression(), !47)
  %161 = icmp samesign ult i64 %157, 257, !dbg !47
  br i1 %161, label %if_then18, label %if_end19, !dbg !47, !prof !16

for_end_tile_i_7_pack:                            ; preds = %if_end16
  %162 = load ptr, ptr @__TVMBackendFreeWorkspace, align 8, !dbg !47, !tbaa !52
  %163 = tail call i32 %162(i32 1, i32 %dev_id, ptr nonnull %tile_storage_3), !dbg !47
  %.not = icmp eq i32 %163, 0, !dbg !47
  br i1 %.not, label %if_end21, label %common.ret, !dbg !47, !prof !17

if_then15:                                        ; preds = %for_body_tile_i_7_pack
  %164 = getelementptr inbounds nuw float, ptr %tile_storage_0, i64 %156, !dbg !47
  %165 = load <16 x float>, ptr %164, align 4, !dbg !47, !tbaa !73
  %166 = getelementptr inbounds nuw float, ptr %tile_storage_3, i64 %156, !dbg !47
  %167 = load <16 x float>, ptr %166, align 4, !dbg !47, !tbaa !83
  %168 = fadd <16 x float> %165, %167, !dbg !47
  %169 = getelementptr inbounds nuw float, ptr %arg2, i64 %158, !dbg !47
  store <16 x float> %168, ptr %169, align 4, !dbg !47, !tbaa !92
  br label %if_end16, !dbg !47

if_end16:                                         ; preds = %if_end19.14, %if_then18.15, %if_then15
  %indvars.iv.next56 = add nuw nsw i64 %indvars.iv55, 1, !dbg !47
    #dbg_value(i64 %indvars.iv.next56, !85, !DIExpression(), !47)
  %exitcond61.not = icmp eq i64 %indvars.iv.next56, 16, !dbg !47
  br i1 %exitcond61.not, label %for_end_tile_i_7_pack, label %for_body_tile_i_7_pack, !dbg !47, !prof !51

if_then18:                                        ; preds = %for_body_tile_i_7_lane.s.preheader
    #dbg_value(i64 %156, !94, !DIExpression(), !47)
  %170 = getelementptr inbounds nuw float, ptr %tile_storage_0, i64 %156, !dbg !47
  %171 = load float, ptr %170, align 4, !dbg !47, !tbaa !73
  %172 = getelementptr inbounds nuw float, ptr %tile_storage_3, i64 %156, !dbg !47
  %173 = load float, ptr %172, align 4, !dbg !47, !tbaa !83
  %174 = fadd float %171, %173, !dbg !47
  %175 = getelementptr inbounds nuw float, ptr %arg2, i64 %158, !dbg !47
  store float %174, ptr %175, align 4, !dbg !47, !tbaa !92
  br label %if_end19, !dbg !47

if_end19:                                         ; preds = %if_then18, %for_body_tile_i_7_lane.s.preheader
    #dbg_value(i64 1, !91, !DIExpression(), !47)
  %176 = or disjoint i64 %157, 1, !dbg !47
  %177 = icmp samesign ult i64 %176, 257, !dbg !47
  br i1 %177, label %if_then18.1, label %if_end19.1, !dbg !47, !prof !16

if_then18.1:                                      ; preds = %if_end19
  %178 = or disjoint i64 %156, 1, !dbg !47
    #dbg_value(i64 %178, !94, !DIExpression(), !47)
  %179 = getelementptr inbounds nuw float, ptr %tile_storage_0, i64 %178, !dbg !47
  %180 = load float, ptr %179, align 4, !dbg !47, !tbaa !73
  %181 = getelementptr inbounds nuw float, ptr %tile_storage_3, i64 %178, !dbg !47
  %182 = load float, ptr %181, align 4, !dbg !47, !tbaa !83
  %183 = fadd float %180, %182, !dbg !47
  %gep182 = getelementptr inbounds nuw float, ptr %invariant.gep181, i64 %158, !dbg !47
  store float %183, ptr %gep182, align 4, !dbg !47, !tbaa !92
  br label %if_end19.1, !dbg !47

if_end19.1:                                       ; preds = %if_then18.1, %if_end19
    #dbg_value(i64 2, !91, !DIExpression(), !47)
  %184 = or disjoint i64 %157, 2, !dbg !47
  %185 = icmp samesign ult i64 %184, 257, !dbg !47
  br i1 %185, label %if_then18.2, label %if_end19.2, !dbg !47, !prof !16

if_then18.2:                                      ; preds = %if_end19.1
  %186 = or disjoint i64 %156, 2, !dbg !47
    #dbg_value(i64 %186, !94, !DIExpression(), !47)
  %187 = getelementptr inbounds nuw float, ptr %tile_storage_0, i64 %186, !dbg !47
  %188 = load float, ptr %187, align 4, !dbg !47, !tbaa !73
  %189 = getelementptr inbounds nuw float, ptr %tile_storage_3, i64 %186, !dbg !47
  %190 = load float, ptr %189, align 4, !dbg !47, !tbaa !83
  %191 = fadd float %188, %190, !dbg !47
  %gep184 = getelementptr inbounds nuw float, ptr %invariant.gep183, i64 %158, !dbg !47
  store float %191, ptr %gep184, align 4, !dbg !47, !tbaa !92
  br label %if_end19.2, !dbg !47

if_end19.2:                                       ; preds = %if_then18.2, %if_end19.1
    #dbg_value(i64 3, !91, !DIExpression(), !47)
  %192 = or disjoint i64 %157, 3, !dbg !47
  %193 = icmp samesign ult i64 %192, 257, !dbg !47
  br i1 %193, label %if_then18.3, label %if_end19.3, !dbg !47, !prof !16

if_then18.3:                                      ; preds = %if_end19.2
  %194 = or disjoint i64 %156, 3, !dbg !47
    #dbg_value(i64 %194, !94, !DIExpression(), !47)
  %195 = getelementptr inbounds nuw float, ptr %tile_storage_0, i64 %194, !dbg !47
  %196 = load float, ptr %195, align 4, !dbg !47, !tbaa !73
  %197 = getelementptr inbounds nuw float, ptr %tile_storage_3, i64 %194, !dbg !47
  %198 = load float, ptr %197, align 4, !dbg !47, !tbaa !83
  %199 = fadd float %196, %198, !dbg !47
  %gep186 = getelementptr inbounds nuw float, ptr %invariant.gep185, i64 %158, !dbg !47
  store float %199, ptr %gep186, align 4, !dbg !47, !tbaa !92
  br label %if_end19.3, !dbg !47

if_end19.3:                                       ; preds = %if_then18.3, %if_end19.2
    #dbg_value(i64 4, !91, !DIExpression(), !47)
  %200 = or disjoint i64 %157, 4, !dbg !47
  %201 = icmp samesign ult i64 %200, 257, !dbg !47
  br i1 %201, label %if_then18.4, label %if_end19.4, !dbg !47, !prof !16

if_then18.4:                                      ; preds = %if_end19.3
  %202 = or disjoint i64 %156, 4, !dbg !47
    #dbg_value(i64 %202, !94, !DIExpression(), !47)
  %203 = getelementptr inbounds nuw float, ptr %tile_storage_0, i64 %202, !dbg !47
  %204 = load float, ptr %203, align 4, !dbg !47, !tbaa !73
  %205 = getelementptr inbounds nuw float, ptr %tile_storage_3, i64 %202, !dbg !47
  %206 = load float, ptr %205, align 4, !dbg !47, !tbaa !83
  %207 = fadd float %204, %206, !dbg !47
  %gep188 = getelementptr inbounds nuw float, ptr %invariant.gep187, i64 %158, !dbg !47
  store float %207, ptr %gep188, align 4, !dbg !47, !tbaa !92
  br label %if_end19.4, !dbg !47

if_end19.4:                                       ; preds = %if_then18.4, %if_end19.3
    #dbg_value(i64 5, !91, !DIExpression(), !47)
  %208 = or disjoint i64 %157, 5, !dbg !47
  %209 = icmp samesign ult i64 %208, 257, !dbg !47
  br i1 %209, label %if_then18.5, label %if_end19.5, !dbg !47, !prof !16

if_then18.5:                                      ; preds = %if_end19.4
  %210 = or disjoint i64 %156, 5, !dbg !47
    #dbg_value(i64 %210, !94, !DIExpression(), !47)
  %211 = getelementptr inbounds nuw float, ptr %tile_storage_0, i64 %210, !dbg !47
  %212 = load float, ptr %211, align 4, !dbg !47, !tbaa !73
  %213 = getelementptr inbounds nuw float, ptr %tile_storage_3, i64 %210, !dbg !47
  %214 = load float, ptr %213, align 4, !dbg !47, !tbaa !83
  %215 = fadd float %212, %214, !dbg !47
  %gep190 = getelementptr inbounds nuw float, ptr %invariant.gep189, i64 %158, !dbg !47
  store float %215, ptr %gep190, align 4, !dbg !47, !tbaa !92
  br label %if_end19.5, !dbg !47

if_end19.5:                                       ; preds = %if_then18.5, %if_end19.4
    #dbg_value(i64 6, !91, !DIExpression(), !47)
  %216 = or disjoint i64 %157, 6, !dbg !47
  %217 = icmp samesign ult i64 %216, 257, !dbg !47
  br i1 %217, label %if_then18.6, label %if_end19.6, !dbg !47, !prof !16

if_then18.6:                                      ; preds = %if_end19.5
  %218 = or disjoint i64 %156, 6, !dbg !47
    #dbg_value(i64 %218, !94, !DIExpression(), !47)
  %219 = getelementptr inbounds nuw float, ptr %tile_storage_0, i64 %218, !dbg !47
  %220 = load float, ptr %219, align 4, !dbg !47, !tbaa !73
  %221 = getelementptr inbounds nuw float, ptr %tile_storage_3, i64 %218, !dbg !47
  %222 = load float, ptr %221, align 4, !dbg !47, !tbaa !83
  %223 = fadd float %220, %222, !dbg !47
  %gep192 = getelementptr inbounds nuw float, ptr %invariant.gep191, i64 %158, !dbg !47
  store float %223, ptr %gep192, align 4, !dbg !47, !tbaa !92
  br label %if_end19.6, !dbg !47

if_end19.6:                                       ; preds = %if_then18.6, %if_end19.5
    #dbg_value(i64 7, !91, !DIExpression(), !47)
  %224 = or disjoint i64 %157, 7, !dbg !47
  %225 = icmp samesign ult i64 %224, 257, !dbg !47
  br i1 %225, label %if_then18.7, label %if_end19.7, !dbg !47, !prof !16

if_then18.7:                                      ; preds = %if_end19.6
  %226 = or disjoint i64 %156, 7, !dbg !47
    #dbg_value(i64 %226, !94, !DIExpression(), !47)
  %227 = getelementptr inbounds nuw float, ptr %tile_storage_0, i64 %226, !dbg !47
  %228 = load float, ptr %227, align 4, !dbg !47, !tbaa !73
  %229 = getelementptr inbounds nuw float, ptr %tile_storage_3, i64 %226, !dbg !47
  %230 = load float, ptr %229, align 4, !dbg !47, !tbaa !83
  %231 = fadd float %228, %230, !dbg !47
  %gep194 = getelementptr inbounds nuw float, ptr %invariant.gep193, i64 %158, !dbg !47
  store float %231, ptr %gep194, align 4, !dbg !47, !tbaa !92
  br label %if_end19.7, !dbg !47

if_end19.7:                                       ; preds = %if_then18.7, %if_end19.6
    #dbg_value(i64 8, !91, !DIExpression(), !47)
  %232 = or disjoint i64 %157, 8, !dbg !47
  %233 = icmp samesign ult i64 %232, 257, !dbg !47
  br i1 %233, label %if_then18.8, label %if_end19.8, !dbg !47, !prof !16

if_then18.8:                                      ; preds = %if_end19.7
  %234 = or disjoint i64 %156, 8, !dbg !47
    #dbg_value(i64 %234, !94, !DIExpression(), !47)
  %235 = getelementptr inbounds nuw float, ptr %tile_storage_0, i64 %234, !dbg !47
  %236 = load float, ptr %235, align 4, !dbg !47, !tbaa !73
  %237 = getelementptr inbounds nuw float, ptr %tile_storage_3, i64 %234, !dbg !47
  %238 = load float, ptr %237, align 4, !dbg !47, !tbaa !83
  %239 = fadd float %236, %238, !dbg !47
  %gep196 = getelementptr inbounds nuw float, ptr %invariant.gep195, i64 %158, !dbg !47
  store float %239, ptr %gep196, align 4, !dbg !47, !tbaa !92
  br label %if_end19.8, !dbg !47

if_end19.8:                                       ; preds = %if_then18.8, %if_end19.7
    #dbg_value(i64 9, !91, !DIExpression(), !47)
  %240 = or disjoint i64 %157, 9, !dbg !47
  %241 = icmp samesign ult i64 %240, 257, !dbg !47
  br i1 %241, label %if_then18.9, label %if_end19.9, !dbg !47, !prof !16

if_then18.9:                                      ; preds = %if_end19.8
  %242 = or disjoint i64 %156, 9, !dbg !47
    #dbg_value(i64 %242, !94, !DIExpression(), !47)
  %243 = getelementptr inbounds nuw float, ptr %tile_storage_0, i64 %242, !dbg !47
  %244 = load float, ptr %243, align 4, !dbg !47, !tbaa !73
  %245 = getelementptr inbounds nuw float, ptr %tile_storage_3, i64 %242, !dbg !47
  %246 = load float, ptr %245, align 4, !dbg !47, !tbaa !83
  %247 = fadd float %244, %246, !dbg !47
  %gep198 = getelementptr inbounds nuw float, ptr %invariant.gep197, i64 %158, !dbg !47
  store float %247, ptr %gep198, align 4, !dbg !47, !tbaa !92
  br label %if_end19.9, !dbg !47

if_end19.9:                                       ; preds = %if_then18.9, %if_end19.8
    #dbg_value(i64 10, !91, !DIExpression(), !47)
  %248 = or disjoint i64 %157, 10, !dbg !47
  %249 = icmp samesign ult i64 %248, 257, !dbg !47
  br i1 %249, label %if_then18.10, label %if_end19.10, !dbg !47, !prof !16

if_then18.10:                                     ; preds = %if_end19.9
  %250 = or disjoint i64 %156, 10, !dbg !47
    #dbg_value(i64 %250, !94, !DIExpression(), !47)
  %251 = getelementptr inbounds nuw float, ptr %tile_storage_0, i64 %250, !dbg !47
  %252 = load float, ptr %251, align 4, !dbg !47, !tbaa !73
  %253 = getelementptr inbounds nuw float, ptr %tile_storage_3, i64 %250, !dbg !47
  %254 = load float, ptr %253, align 4, !dbg !47, !tbaa !83
  %255 = fadd float %252, %254, !dbg !47
  %gep200 = getelementptr inbounds nuw float, ptr %invariant.gep199, i64 %158, !dbg !47
  store float %255, ptr %gep200, align 4, !dbg !47, !tbaa !92
  br label %if_end19.10, !dbg !47

if_end19.10:                                      ; preds = %if_then18.10, %if_end19.9
    #dbg_value(i64 11, !91, !DIExpression(), !47)
  %256 = or disjoint i64 %157, 11, !dbg !47
  %257 = icmp samesign ult i64 %256, 257, !dbg !47
  br i1 %257, label %if_then18.11, label %if_end19.11, !dbg !47, !prof !16

if_then18.11:                                     ; preds = %if_end19.10
  %258 = or disjoint i64 %156, 11, !dbg !47
    #dbg_value(i64 %258, !94, !DIExpression(), !47)
  %259 = getelementptr inbounds nuw float, ptr %tile_storage_0, i64 %258, !dbg !47
  %260 = load float, ptr %259, align 4, !dbg !47, !tbaa !73
  %261 = getelementptr inbounds nuw float, ptr %tile_storage_3, i64 %258, !dbg !47
  %262 = load float, ptr %261, align 4, !dbg !47, !tbaa !83
  %263 = fadd float %260, %262, !dbg !47
  %gep202 = getelementptr inbounds nuw float, ptr %invariant.gep201, i64 %158, !dbg !47
  store float %263, ptr %gep202, align 4, !dbg !47, !tbaa !92
  br label %if_end19.11, !dbg !47

if_end19.11:                                      ; preds = %if_then18.11, %if_end19.10
    #dbg_value(i64 12, !91, !DIExpression(), !47)
  %264 = or disjoint i64 %157, 12, !dbg !47
  %265 = icmp samesign ult i64 %264, 257, !dbg !47
  br i1 %265, label %if_then18.12, label %if_end19.12, !dbg !47, !prof !16

if_then18.12:                                     ; preds = %if_end19.11
  %266 = or disjoint i64 %156, 12, !dbg !47
    #dbg_value(i64 %266, !94, !DIExpression(), !47)
  %267 = getelementptr inbounds nuw float, ptr %tile_storage_0, i64 %266, !dbg !47
  %268 = load float, ptr %267, align 4, !dbg !47, !tbaa !73
  %269 = getelementptr inbounds nuw float, ptr %tile_storage_3, i64 %266, !dbg !47
  %270 = load float, ptr %269, align 4, !dbg !47, !tbaa !83
  %271 = fadd float %268, %270, !dbg !47
  %gep204 = getelementptr inbounds nuw float, ptr %invariant.gep203, i64 %158, !dbg !47
  store float %271, ptr %gep204, align 4, !dbg !47, !tbaa !92
  br label %if_end19.12, !dbg !47

if_end19.12:                                      ; preds = %if_then18.12, %if_end19.11
    #dbg_value(i64 13, !91, !DIExpression(), !47)
  %272 = or disjoint i64 %157, 13, !dbg !47
  %273 = icmp samesign ult i64 %272, 257, !dbg !47
  br i1 %273, label %if_then18.13, label %if_end19.13, !dbg !47, !prof !16

if_then18.13:                                     ; preds = %if_end19.12
  %274 = or disjoint i64 %156, 13, !dbg !47
    #dbg_value(i64 %274, !94, !DIExpression(), !47)
  %275 = getelementptr inbounds nuw float, ptr %tile_storage_0, i64 %274, !dbg !47
  %276 = load float, ptr %275, align 4, !dbg !47, !tbaa !73
  %277 = getelementptr inbounds nuw float, ptr %tile_storage_3, i64 %274, !dbg !47
  %278 = load float, ptr %277, align 4, !dbg !47, !tbaa !83
  %279 = fadd float %276, %278, !dbg !47
  %gep206 = getelementptr inbounds nuw float, ptr %invariant.gep205, i64 %158, !dbg !47
  store float %279, ptr %gep206, align 4, !dbg !47, !tbaa !92
  br label %if_end19.13, !dbg !47

if_end19.13:                                      ; preds = %if_then18.13, %if_end19.12
    #dbg_value(i64 14, !91, !DIExpression(), !47)
  %280 = or disjoint i64 %157, 14, !dbg !47
  %281 = icmp samesign ult i64 %280, 257, !dbg !47
  br i1 %281, label %if_then18.14, label %if_end19.14, !dbg !47, !prof !16

if_then18.14:                                     ; preds = %if_end19.13
  %282 = or disjoint i64 %156, 14, !dbg !47
    #dbg_value(i64 %282, !94, !DIExpression(), !47)
  %283 = getelementptr inbounds nuw float, ptr %tile_storage_0, i64 %282, !dbg !47
  %284 = load float, ptr %283, align 4, !dbg !47, !tbaa !73
  %285 = getelementptr inbounds nuw float, ptr %tile_storage_3, i64 %282, !dbg !47
  %286 = load float, ptr %285, align 4, !dbg !47, !tbaa !83
  %287 = fadd float %284, %286, !dbg !47
  %gep208 = getelementptr inbounds nuw float, ptr %invariant.gep207, i64 %158, !dbg !47
  store float %287, ptr %gep208, align 4, !dbg !47, !tbaa !92
  br label %if_end19.14, !dbg !47

if_end19.14:                                      ; preds = %if_then18.14, %if_end19.13
    #dbg_value(i64 15, !91, !DIExpression(), !47)
  %288 = or disjoint i64 %157, 15, !dbg !47
  %289 = icmp samesign ult i64 %288, 257, !dbg !47
  br i1 %289, label %if_then18.15, label %if_end16, !dbg !47, !prof !16

if_then18.15:                                     ; preds = %if_end19.14
  %290 = or disjoint i64 %156, 15, !dbg !47
    #dbg_value(i64 %290, !94, !DIExpression(), !47)
  %291 = getelementptr inbounds nuw float, ptr %tile_storage_0, i64 %290, !dbg !47
  %292 = load float, ptr %291, align 4, !dbg !47, !tbaa !73
  %293 = getelementptr inbounds nuw float, ptr %tile_storage_3, i64 %290, !dbg !47
  %294 = load float, ptr %293, align 4, !dbg !47, !tbaa !83
  %295 = fadd float %292, %294, !dbg !47
  %gep210 = getelementptr inbounds nuw float, ptr %invariant.gep209, i64 %158, !dbg !47
  store float %295, ptr %gep210, align 4, !dbg !47, !tbaa !92
  br label %if_end16, !dbg !47

if_end21:                                         ; preds = %for_end_tile_i_7_pack
  %296 = load ptr, ptr @__TVMBackendFreeWorkspace, align 8, !dbg !47, !tbaa !52
  %297 = tail call i32 %296(i32 1, i32 %dev_id, ptr nonnull %tile_storage_0), !dbg !47
  %.not26 = icmp eq i32 %297, 0, !dbg !47
  br i1 %.not26, label %for_begin_parallel_0, label %common.ret, !dbg !47, !prof !17
}

define weak dllexport i32 @__tvm_ffi_main(ptr %0, ptr %1, i32 %2, ptr %3) local_unnamed_addr {
entry:
  %4 = tail call i32 @__tvm_ffi_benchmark_add(ptr poison, ptr %1, i32 %2, ptr poison), !dbg !15
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
!5 = distinct !DISubprogram(name: "__tvm_ffi_benchmark_add", scope: !1, file: !1, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
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
!24 = !DILocalVariable(name: "benchmark_add.arg0_shape", scope: !5, file: !1, type: !25)
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !26)
!26 = !DIBasicType(name: "int64", size: 64, encoding: DW_ATE_signed)
!27 = !DILocalVariable(name: "benchmark_add.arg0_strides", scope: !5, file: !1, type: !25)
!28 = !DILocalVariable(name: "dev_id", scope: !5, file: !1, type: !8)
!29 = !DILocalVariable(name: "benchmark_add.arg1_shape", scope: !5, file: !1, type: !25)
!30 = !DILocalVariable(name: "benchmark_add.arg1_strides", scope: !5, file: !1, type: !25)
!31 = !DILocalVariable(name: "benchmark_add.arg2_shape", scope: !5, file: !1, type: !25)
!32 = !DILocalVariable(name: "benchmark_add.arg2_strides", scope: !5, file: !1, type: !25)
!33 = !DILocalVariable(name: "arg0", scope: !5, file: !1, type: !34)
!34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !35)
!35 = !DIBasicType(name: "float32", size: 32, encoding: DW_ATE_float)
!36 = !DILocalVariable(name: "arg1", scope: !5, file: !1, type: !34)
!37 = !DILocalVariable(name: "arg2", scope: !5, file: !1, type: !34)
!38 = distinct !DISubprogram(name: "benchmark_add_compute_", scope: !1, file: !1, type: !39, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !41)
!39 = !DISubroutineType(types: !40)
!40 = !{!8, !34, !34, !34, !8}
!41 = !{!42, !43, !44, !45}
!42 = !DILocalVariable(name: "arg0", arg: 1, scope: !38, file: !1, type: !34)
!43 = !DILocalVariable(name: "arg1", arg: 2, scope: !38, file: !1, type: !34)
!44 = !DILocalVariable(name: "arg2", arg: 3, scope: !38, file: !1, type: !34)
!45 = !DILocalVariable(name: "dev_id", arg: 4, scope: !38, file: !1, type: !8)
!46 = !DILocalVariable(name: "arg0", scope: !38, file: !1, type: !34)
!47 = !DILocation(line: 0, scope: !38)
!48 = !DILocalVariable(name: "arg1", scope: !38, file: !1, type: !34)
!49 = !DILocalVariable(name: "arg2", scope: !38, file: !1, type: !34)
!50 = !DILocalVariable(name: "parallel_0", scope: !38, file: !1, type: !8)
!51 = !{!"branch_weights", i32 1, i32 1048575}
!52 = !{!53, !53, i64 0}
!53 = !{!"ctx_ptr", !54, i64 0}
!54 = !{!"tvm-tbaa"}
!55 = !DILocalVariable(name: "tile_storage_0", scope: !38, file: !1, type: !34)
!56 = !DILocalVariable(name: "cse_v1", scope: !38, file: !1, type: !8)
!57 = !DILocalVariable(name: "cse_v5", scope: !38, file: !1, type: !8)
!58 = !DILocalVariable(name: "cse_v6", scope: !38, file: !1, type: !8)
!59 = !DILocalVariable(name: "cse_v7", scope: !38, file: !1, type: !8)
!60 = !DILocalVariable(name: "cse_v8", scope: !38, file: !1, type: !8)
!61 = !DILocalVariable(name: "cse_v13", scope: !38, file: !1, type: !8)
!62 = !DILocalVariable(name: "tile_i_2_pack", scope: !38, file: !1, type: !8)
!63 = !DILocalVariable(name: "cse_v2", scope: !38, file: !1, type: !8)
!64 = !DILocalVariable(name: "cse_v10", scope: !38, file: !1, type: !8)
!65 = !DILocalVariable(name: "cse_v11", scope: !38, file: !1, type: !8)
!66 = !DILocalVariable(name: "cse_v12", scope: !38, file: !1, type: !8)
!67 = !DILocalVariable(name: "cse_v20", scope: !38, file: !1, type: !8)
!68 = !DILocalVariable(name: "tile_i_2_lane.s", scope: !38, file: !1, type: !8)
!69 = !DILocalVariable(name: "tile_storage_3", scope: !38, file: !1, type: !34)
!70 = !DILocalVariable(name: "tile_i_5_pack", scope: !38, file: !1, type: !8)
!71 = !{!72, !72, i64 0}
!72 = !{!"0x852f8a240", !54, i64 0}
!73 = !{!74, !74, i64 0}
!74 = !{!"0x85285a680", !54, i64 0}
!75 = !DILocalVariable(name: "cse_v3", scope: !38, file: !1, type: !8)
!76 = !DILocalVariable(name: "cse_v14", scope: !38, file: !1, type: !8)
!77 = !DILocalVariable(name: "cse_v15", scope: !38, file: !1, type: !8)
!78 = !DILocalVariable(name: "cse_v16", scope: !38, file: !1, type: !8)
!79 = !DILocalVariable(name: "cse_v21", scope: !38, file: !1, type: !8)
!80 = !DILocalVariable(name: "tile_i_5_lane.s", scope: !38, file: !1, type: !8)
!81 = !{!82, !82, i64 0}
!82 = !{!"0x852f8a280", !54, i64 0}
!83 = !{!84, !84, i64 0}
!84 = !{!"0x85285ad40", !54, i64 0}
!85 = !DILocalVariable(name: "tile_i_7_pack", scope: !38, file: !1, type: !8)
!86 = !DILocalVariable(name: "cse_v4", scope: !38, file: !1, type: !8)
!87 = !DILocalVariable(name: "cse_v17", scope: !38, file: !1, type: !8)
!88 = !DILocalVariable(name: "cse_v18", scope: !38, file: !1, type: !8)
!89 = !DILocalVariable(name: "cse_v19", scope: !38, file: !1, type: !8)
!90 = !DILocalVariable(name: "cse_v22", scope: !38, file: !1, type: !8)
!91 = !DILocalVariable(name: "tile_i_7_lane.s", scope: !38, file: !1, type: !8)
!92 = !{!93, !93, i64 0}
!93 = !{!"0x852f8a2c0", !54, i64 0}
!94 = !DILocalVariable(name: "cse_v9", scope: !38, file: !1, type: !8)
