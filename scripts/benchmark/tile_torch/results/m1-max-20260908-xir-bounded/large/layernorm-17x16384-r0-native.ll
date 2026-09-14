; ModuleID = 'luisa-simd-kernel'
source_filename = "luisa-simd-kernel"

define internal void @llm_rows(ptr noalias readonly %argument_buffer, ptr %return_lanes, ptr noalias nonnull readonly %launch_config, i32 %active_lane_count) {
prologue:
  %0 = getelementptr i8, ptr %launch_config, i64 136
  %private.workspace = load ptr, ptr %0, align 8
  %tile_snapshot.private = getelementptr inbounds i8, ptr %private.workspace, i64 0
  %.splatinsert = insertelement <8 x ptr> poison, ptr %tile_snapshot.private, i64 0
  %.splat = shufflevector <8 x ptr> %.splatinsert, <8 x ptr> poison, <8 x i32> zeroinitializer
  %1 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat, 0
  %2 = insertvalue { <8 x ptr>, <8 x i64> } %1, <8 x i64> <i64 0, i64 65536, i64 131072, i64 196608, i64 262144, i64 327680, i64 393216, i64 458752>, 1
  %3 = getelementptr i8, ptr %launch_config, i64 136
  %private.workspace1 = load ptr, ptr %3, align 8
  %tile_snapshot.private2 = getelementptr inbounds i8, ptr %private.workspace1, i64 524288
  %.splatinsert3 = insertelement <8 x ptr> poison, ptr %tile_snapshot.private2, i64 0
  %.splat4 = shufflevector <8 x ptr> %.splatinsert3, <8 x ptr> poison, <8 x i32> zeroinitializer
  %4 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat4, 0
  %5 = insertvalue { <8 x ptr>, <8 x i64> } %4, <8 x i64> <i64 0, i64 65536, i64 131072, i64 196608, i64 262144, i64 327680, i64 393216, i64 458752>, 1
  %6 = getelementptr i8, ptr %launch_config, i64 136
  %private.workspace5 = load ptr, ptr %6, align 8
  %tile_snapshot.private6 = getelementptr inbounds i8, ptr %private.workspace5, i64 1048576
  %.splatinsert7 = insertelement <8 x ptr> poison, ptr %tile_snapshot.private6, i64 0
  %.splat8 = shufflevector <8 x ptr> %.splatinsert7, <8 x ptr> poison, <8 x i32> zeroinitializer
  %7 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat8, 0
  %8 = insertvalue { <8 x ptr>, <8 x i64> } %7, <8 x i64> <i64 0, i64 65536, i64 131072, i64 196608, i64 262144, i64 327680, i64 393216, i64 458752>, 1
  %9 = getelementptr i8, ptr %launch_config, i64 136
  %private.workspace9 = load ptr, ptr %9, align 8
  %tile_snapshot.private10 = getelementptr inbounds i8, ptr %private.workspace9, i64 1572864
  %.splatinsert11 = insertelement <8 x ptr> poison, ptr %tile_snapshot.private10, i64 0
  %.splat12 = shufflevector <8 x ptr> %.splatinsert11, <8 x ptr> poison, <8 x i32> zeroinitializer
  %10 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat12, 0
  %11 = insertvalue { <8 x ptr>, <8 x i64> } %10, <8 x i64> <i64 0, i64 65536, i64 131072, i64 196608, i64 262144, i64 327680, i64 393216, i64 458752>, 1
  %.spill = alloca <8 x i64>, align 64
  store <8 x i64> zeroinitializer, ptr %.spill, align 64
  %tile_snapshot.spill = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill, align 64
  %.slot = alloca i64, align 8
  store i64 0, ptr %.slot, align 4
  %.spill13 = alloca float, align 4
  store float 0.000000e+00, ptr %.spill13, align 4
  %.spill14 = alloca i64, align 8
  store i64 0, ptr %.spill14, align 4
  %.spill15 = alloca i64, align 8
  store i64 0, ptr %.spill15, align 4
  %.spill16 = alloca i64, align 8
  store i64 0, ptr %.spill16, align 4
  %.spill17 = alloca i1, align 1
  store i1 false, ptr %.spill17, align 1
  %.slot18 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot18, align 32
  %.spill19 = alloca i64, align 8
  store i64 0, ptr %.spill19, align 4
  %.spill20 = alloca i1, align 1
  store i1 false, ptr %.spill20, align 1
  %.slot21 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot21, align 32
  %.spill22 = alloca i64, align 8
  store i64 0, ptr %.spill22, align 4
  %.spill23 = alloca i1, align 1
  store i1 false, ptr %.spill23, align 1
  %.slot24 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot24, align 32
  %.spill25 = alloca i64, align 8
  store i64 0, ptr %.spill25, align 4
  %.spill26 = alloca i1, align 1
  store i1 false, ptr %.spill26, align 1
  %.slot27 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot27, align 32
  %.slot28 = alloca i64, align 8
  store i64 0, ptr %.slot28, align 4
  %.slot29 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot29, align 32
  %.slot30 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot30, align 32
  %.slot31 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot31, align 32
  %.slot32 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot32, align 32
  %.spill33 = alloca i64, align 8
  store i64 0, ptr %.spill33, align 4
  %.slot34 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot34, align 32
  %.spill35 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill35, align 32
  %.spill36 = alloca i64, align 8
  store i64 0, ptr %.spill36, align 4
  %.slot37 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot37, align 32
  %.spill38 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill38, align 32
  %.spill39 = alloca i64, align 8
  store i64 0, ptr %.spill39, align 4
  %.slot40 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot40, align 32
  %.spill41 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill41, align 32
  %.spill42 = alloca i64, align 8
  store i64 0, ptr %.spill42, align 4
  %.slot43 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot43, align 32
  %.spill44 = alloca float, align 4
  store float 0.000000e+00, ptr %.spill44, align 4
  %.spill45 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill45, align 32
  %tile_snapshot.spill46 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill46, align 64
  %.slot47 = alloca i64, align 8
  store i64 0, ptr %.slot47, align 4
  %.slot48 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot48, align 32
  %.slot49 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot49, align 32
  %.slot50 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot50, align 32
  %.slot51 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot51, align 32
  %.slot52 = alloca i64, align 8
  store i64 0, ptr %.slot52, align 4
  %.slot53 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot53, align 32
  %.slot54 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot54, align 32
  %.slot55 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot55, align 32
  %.slot56 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot56, align 32
  %.spill57 = alloca i64, align 8
  store i64 0, ptr %.spill57, align 4
  %.slot58 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot58, align 32
  %.spill59 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill59, align 32
  %.spill60 = alloca i64, align 8
  store i64 0, ptr %.spill60, align 4
  %.slot61 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot61, align 32
  %.spill62 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill62, align 32
  %.spill63 = alloca i64, align 8
  store i64 0, ptr %.spill63, align 4
  %.slot64 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot64, align 32
  %.spill65 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill65, align 32
  %.spill66 = alloca i64, align 8
  store i64 0, ptr %.spill66, align 4
  %.slot67 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot67, align 32
  %.spill68 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill68, align 32
  %tile_snapshot.spill69 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill69, align 64
  %.slot70 = alloca i64, align 8
  store i64 0, ptr %.slot70, align 4
  %.spill71 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill71, align 32
  %tile_snapshot.spill72 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill72, align 64
  %.slot73 = alloca i64, align 8
  store i64 0, ptr %.slot73, align 4
  %.slot74 = alloca i64, align 8
  store i64 0, ptr %.slot74, align 4
  %12 = getelementptr i8, ptr %argument_buffer, i64 0
  %13 = load ptr, ptr %12, align 16
  %14 = getelementptr i8, ptr %12, i64 8
  %15 = load i64, ptr %14, align 8
  %16 = insertvalue { ptr, i64 } poison, ptr %13, 0
  %17 = insertvalue { ptr, i64 } %16, i64 %15, 1
  %18 = getelementptr i8, ptr %argument_buffer, i64 16
  %19 = load ptr, ptr %18, align 16
  %20 = getelementptr i8, ptr %18, i64 8
  %21 = load i64, ptr %20, align 8
  %22 = insertvalue { ptr, i64 } poison, ptr %19, 0
  %23 = insertvalue { ptr, i64 } %22, i64 %21, 1
  %24 = getelementptr i8, ptr %argument_buffer, i64 32
  %25 = load ptr, ptr %24, align 16
  %26 = getelementptr i8, ptr %24, i64 8
  %27 = load i64, ptr %26, align 8
  %28 = insertvalue { ptr, i64 } poison, ptr %25, 0
  %29 = insertvalue { ptr, i64 } %28, i64 %27, 1
  %30 = getelementptr i8, ptr %argument_buffer, i64 48
  %31 = load ptr, ptr %30, align 16
  %32 = getelementptr i8, ptr %30, i64 8
  %33 = load i64, ptr %32, align 8
  %34 = insertvalue { ptr, i64 } poison, ptr %31, 0
  %35 = insertvalue { ptr, i64 } %34, i64 %33, 1
  %36 = load i32, ptr %launch_config, align 4
  %37 = getelementptr i8, ptr %launch_config, i64 12
  %38 = load i32, ptr %37, align 4
  %39 = getelementptr i8, ptr %launch_config, i64 4
  %40 = load i32, ptr %39, align 4
  %41 = getelementptr i8, ptr %launch_config, i64 16
  %42 = load i32, ptr %41, align 4
  %43 = getelementptr i8, ptr %launch_config, i64 8
  %44 = load i32, ptr %43, align 4
  %45 = getelementptr i8, ptr %launch_config, i64 20
  %46 = load i32, ptr %45, align 4
  %47 = getelementptr i8, ptr %launch_config, i64 36
  %48 = load i32, ptr %47, align 4
  %.splatinsert75 = insertelement <8 x i32> poison, i32 %48, i64 0
  %.splat76 = shufflevector <8 x i32> %.splatinsert75, <8 x i32> poison, <8 x i32> zeroinitializer
  %49 = add <8 x i32> %.splat76, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %50 = mul i32 %36, 32
  %.splatinsert77 = insertelement <8 x i32> poison, i32 %50, i64 0
  %.splat78 = shufflevector <8 x i32> %.splatinsert77, <8 x i32> poison, <8 x i32> zeroinitializer
  %51 = add <8 x i32> %.splat78, %49
  %52 = mul i32 %40, 1
  %.splatinsert79 = insertelement <8 x i32> poison, i32 %52, i64 0
  %.splat80 = shufflevector <8 x i32> %.splatinsert79, <8 x i32> poison, <8 x i32> zeroinitializer
  %53 = add <8 x i32> %.splat80, zeroinitializer
  %54 = mul i32 %44, 1
  %.splatinsert81 = insertelement <8 x i32> poison, i32 %54, i64 0
  %.splat82 = shufflevector <8 x i32> %.splatinsert81, <8 x i32> poison, <8 x i32> zeroinitializer
  %55 = add <8 x i32> %.splat82, zeroinitializer
  %56 = insertvalue [3 x <8 x i32>] poison, <8 x i32> %51, 0
  %57 = insertvalue [3 x <8 x i32>] %56, <8 x i32> %53, 1
  %58 = insertvalue [3 x <8 x i32>] %57, <8 x i32> %55, 2
  %.splatinsert83 = insertelement <8 x i32> poison, i32 %active_lane_count, i64 0
  %.splat84 = shufflevector <8 x i32> %.splatinsert83, <8 x i32> poison, <8 x i32> zeroinitializer
  %59 = icmp ult <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, %.splat84
  %60 = call i1 @llvm.vector.reduce.or.v8i1(<8 x i1> %59)
  br i1 %60, label %direct.activate, label %direct.inactive

direct.schedule.0:                                ; preds = %direct.activate
  %61 = extractvalue [3 x <8 x i32>] %58, 0
  %62 = zext <8 x i32> %61 to <8 x i64>
  %63 = select <8 x i1> %59, <8 x i64> %62, <8 x i64> zeroinitializer
  %64 = select <8 x i1> %59, <8 x i64> splat (i64 1), <8 x i64> splat (i64 1)
  %65 = sdiv <8 x i64> %63, %64
  %66 = select <8 x i1> %59, <8 x i64> %65, <8 x i64> zeroinitializer
  %67 = select <8 x i1> %59, <8 x i64> splat (i64 17), <8 x i64> splat (i64 1)
  %68 = srem <8 x i64> %66, %67
  %69 = load <8 x i64>, ptr %.spill, align 64
  %70 = select <8 x i1> %59, <8 x i64> %68, <8 x i64> %69
  store <8 x i64> %70, ptr %.spill, align 64
  %71 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %72 = extractvalue { <8 x ptr>, <8 x i64> } %2, 0
  %73 = extractvalue { <8 x ptr>, <8 x i64> } %71, 0
  %74 = select <8 x i1> %59, <8 x ptr> %72, <8 x ptr> %73
  %75 = extractvalue { <8 x ptr>, <8 x i64> } %2, 1
  %76 = extractvalue { <8 x ptr>, <8 x i64> } %71, 1
  %77 = select <8 x i1> %59, <8 x i64> %75, <8 x i64> %76
  %78 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %74, 0
  %79 = insertvalue { <8 x ptr>, <8 x i64> } %78, <8 x i64> %77, 1
  store { <8 x ptr>, <8 x i64> } %79, ptr %tile_snapshot.spill, align 64
  store i64 0, ptr %.slot, align 4
  br label %direct.schedule.1

direct.schedule.1:                                ; preds = %direct.schedule.2, %direct.schedule.0
  %.state = load i64, ptr %.slot, align 4
  %80 = icmp slt i64 %.state, 16384
  br i1 %80, label %direct.true, label %direct.false

direct.schedule.2:                                ; preds = %direct.true
  %.state85 = load i64, ptr %.slot, align 4
  %81 = srem i64 %.state85, 16384
  %.state86 = load i64, ptr %.slot, align 4
  %82 = sdiv i64 %.state86, 16384
  %83 = srem i64 %82, 1
  %.spill.load = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert87 = insertelement <8 x i64> poison, i64 %83, i64 0
  %.splat88 = shufflevector <8 x i64> %.splatinsert87, <8 x i64> poison, <8 x i32> zeroinitializer
  %84 = add <8 x i64> %.spill.load, %.splat88
  %85 = add <8 x i64> zeroinitializer, %84
  %86 = add i64 0, %81
  %87 = mul <8 x i64> %85, splat (i64 16384)
  %.splatinsert89 = insertelement <8 x i64> poison, i64 %86, i64 0
  %.splat90 = shufflevector <8 x i64> %.splatinsert89, <8 x i64> poison, <8 x i32> zeroinitializer
  %88 = add <8 x i64> %87, %.splat90
  %89 = extractvalue { ptr, i64 } %17, 0
  %90 = mul <8 x i64> %88, splat (i64 4)
  %91 = getelementptr i8, ptr %89, <8 x i64> %90
  %92 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %91, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %93 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 0
  %94 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 1
  %.state91 = load i64, ptr %.slot, align 4
  %.splatinsert92 = insertelement <8 x i64> poison, i64 %.state91, i64 0
  %.splat93 = shufflevector <8 x i64> %.splatinsert92, <8 x i64> poison, <8 x i32> zeroinitializer
  %95 = mul <8 x i64> %.splat93, splat (i64 4)
  %96 = add <8 x i64> %94, %95
  %97 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %93, 0
  %98 = insertvalue { <8 x ptr>, <8 x i64> } %97, <8 x i64> %96, 1
  %99 = extractvalue { <8 x ptr>, <8 x i64> } %98, 0
  %100 = extractvalue { <8 x ptr>, <8 x i64> } %98, 1
  %101 = getelementptr i8, <8 x ptr> %99, <8 x i64> %100
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %92, <8 x ptr> %101, i32 1, <8 x i1> %59)
  %.state94 = load i64, ptr %.slot, align 4
  %102 = add i64 %.state94, 1
  store i64 %102, ptr %.slot, align 4
  br label %direct.schedule.1

direct.schedule.3:                                ; preds = %direct.false
  store float 0.000000e+00, ptr %.spill13, align 4
  store i64 0, ptr %.spill14, align 4
  store i64 0, ptr %.spill15, align 4
  store i64 0, ptr %.spill16, align 4
  store i1 true, ptr %.spill17, align 1
  br i1 true, label %direct.true95, label %direct.false96

direct.schedule.4:                                ; preds = %direct.true95
  %tile_snapshot.spill.load97 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %103 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load97, 0
  %104 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load97, 1
  %.spill.load98 = load i64, ptr %.spill16, align 4
  %.splatinsert99 = insertelement <8 x i64> poison, i64 %.spill.load98, i64 0
  %.splat100 = shufflevector <8 x i64> %.splatinsert99, <8 x i64> poison, <8 x i32> zeroinitializer
  %105 = mul <8 x i64> %.splat100, splat (i64 4)
  %106 = add <8 x i64> %104, %105
  %107 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %103, 0
  %108 = insertvalue { <8 x ptr>, <8 x i64> } %107, <8 x i64> %106, 1
  %109 = extractvalue { <8 x ptr>, <8 x i64> } %108, 0
  %110 = extractvalue { <8 x ptr>, <8 x i64> } %108, 1
  %111 = getelementptr i8, <8 x ptr> %109, <8 x i64> %110
  %112 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %111, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %113 = load <8 x float>, ptr %.slot18, align 32
  %114 = select <8 x i1> %59, <8 x float> %112, <8 x float> %113
  store <8 x float> %114, ptr %.slot18, align 32
  br label %direct.schedule.5

direct.schedule.5:                                ; preds = %direct.schedule.4, %direct.false96
  %.spill.load101 = load i64, ptr %.spill15, align 4
  %115 = add i64 %.spill.load101, 1
  store i64 %115, ptr %.spill19, align 4
  %116 = icmp sge i64 %115, 0
  %117 = icmp slt i64 %115, 16384
  %118 = and i1 %116, %117
  store i1 %118, ptr %.spill20, align 1
  br i1 %118, label %direct.true102, label %direct.false103

direct.schedule.6:                                ; preds = %direct.true102
  %tile_snapshot.spill.load104 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %119 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load104, 0
  %120 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load104, 1
  %.spill.load105 = load i64, ptr %.spill19, align 4
  %.splatinsert106 = insertelement <8 x i64> poison, i64 %.spill.load105, i64 0
  %.splat107 = shufflevector <8 x i64> %.splatinsert106, <8 x i64> poison, <8 x i32> zeroinitializer
  %121 = mul <8 x i64> %.splat107, splat (i64 4)
  %122 = add <8 x i64> %120, %121
  %123 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %119, 0
  %124 = insertvalue { <8 x ptr>, <8 x i64> } %123, <8 x i64> %122, 1
  %125 = extractvalue { <8 x ptr>, <8 x i64> } %124, 0
  %126 = extractvalue { <8 x ptr>, <8 x i64> } %124, 1
  %127 = getelementptr i8, <8 x ptr> %125, <8 x i64> %126
  %128 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %127, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %129 = load <8 x float>, ptr %.slot21, align 32
  %130 = select <8 x i1> %59, <8 x float> %128, <8 x float> %129
  store <8 x float> %130, ptr %.slot21, align 32
  br label %direct.schedule.7

direct.schedule.7:                                ; preds = %direct.schedule.6, %direct.false103
  %.spill.load108 = load i64, ptr %.spill15, align 4
  %131 = add i64 %.spill.load108, 2
  store i64 %131, ptr %.spill22, align 4
  %132 = icmp sge i64 %131, 0
  %133 = icmp slt i64 %131, 16384
  %134 = and i1 %132, %133
  store i1 %134, ptr %.spill23, align 1
  br i1 %134, label %direct.true109, label %direct.false110

direct.schedule.8:                                ; preds = %direct.true109
  %tile_snapshot.spill.load111 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %135 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load111, 0
  %136 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load111, 1
  %.spill.load112 = load i64, ptr %.spill22, align 4
  %.splatinsert113 = insertelement <8 x i64> poison, i64 %.spill.load112, i64 0
  %.splat114 = shufflevector <8 x i64> %.splatinsert113, <8 x i64> poison, <8 x i32> zeroinitializer
  %137 = mul <8 x i64> %.splat114, splat (i64 4)
  %138 = add <8 x i64> %136, %137
  %139 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %135, 0
  %140 = insertvalue { <8 x ptr>, <8 x i64> } %139, <8 x i64> %138, 1
  %141 = extractvalue { <8 x ptr>, <8 x i64> } %140, 0
  %142 = extractvalue { <8 x ptr>, <8 x i64> } %140, 1
  %143 = getelementptr i8, <8 x ptr> %141, <8 x i64> %142
  %144 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %143, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %145 = load <8 x float>, ptr %.slot24, align 32
  %146 = select <8 x i1> %59, <8 x float> %144, <8 x float> %145
  store <8 x float> %146, ptr %.slot24, align 32
  br label %direct.schedule.9

direct.schedule.9:                                ; preds = %direct.schedule.8, %direct.false110
  %.spill.load115 = load i64, ptr %.spill15, align 4
  %147 = add i64 %.spill.load115, 3
  store i64 %147, ptr %.spill25, align 4
  %148 = icmp sge i64 %147, 0
  %149 = icmp slt i64 %147, 16384
  %150 = and i1 %148, %149
  store i1 %150, ptr %.spill26, align 1
  br i1 %150, label %direct.true116, label %direct.false117

direct.schedule.10:                               ; preds = %direct.true116
  %tile_snapshot.spill.load118 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %151 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load118, 0
  %152 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load118, 1
  %.spill.load119 = load i64, ptr %.spill25, align 4
  %.splatinsert120 = insertelement <8 x i64> poison, i64 %.spill.load119, i64 0
  %.splat121 = shufflevector <8 x i64> %.splatinsert120, <8 x i64> poison, <8 x i32> zeroinitializer
  %153 = mul <8 x i64> %.splat121, splat (i64 4)
  %154 = add <8 x i64> %152, %153
  %155 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %151, 0
  %156 = insertvalue { <8 x ptr>, <8 x i64> } %155, <8 x i64> %154, 1
  %157 = extractvalue { <8 x ptr>, <8 x i64> } %156, 0
  %158 = extractvalue { <8 x ptr>, <8 x i64> } %156, 1
  %159 = getelementptr i8, <8 x ptr> %157, <8 x i64> %158
  %160 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %159, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %161 = load <8 x float>, ptr %.slot27, align 32
  %162 = select <8 x i1> %59, <8 x float> %160, <8 x float> %161
  store <8 x float> %162, ptr %.slot27, align 32
  br label %direct.schedule.11

direct.schedule.11:                               ; preds = %direct.schedule.10, %direct.false117
  %.state122 = load <8 x float>, ptr %.slot18, align 32
  %.state123 = load <8 x float>, ptr %.slot21, align 32
  %.state124 = load <8 x float>, ptr %.slot24, align 32
  %.state125 = load <8 x float>, ptr %.slot27, align 32
  store i64 4, ptr %.slot28, align 4
  %163 = load <8 x float>, ptr %.slot29, align 32
  %164 = select <8 x i1> %59, <8 x float> %.state122, <8 x float> %163
  store <8 x float> %164, ptr %.slot29, align 32
  %165 = load <8 x float>, ptr %.slot30, align 32
  %166 = select <8 x i1> %59, <8 x float> %.state123, <8 x float> %165
  store <8 x float> %166, ptr %.slot30, align 32
  %167 = load <8 x float>, ptr %.slot31, align 32
  %168 = select <8 x i1> %59, <8 x float> %.state124, <8 x float> %167
  store <8 x float> %168, ptr %.slot31, align 32
  %169 = load <8 x float>, ptr %.slot32, align 32
  %170 = select <8 x i1> %59, <8 x float> %.state125, <8 x float> %169
  store <8 x float> %170, ptr %.slot32, align 32
  br label %direct.schedule.12

direct.schedule.12:                               ; preds = %direct.schedule.21, %direct.schedule.11
  %.state126 = load i64, ptr %.slot28, align 4
  %171 = icmp slt i64 %.state126, 16384
  br i1 %171, label %direct.true127, label %direct.false128

direct.schedule.13:                               ; preds = %direct.true127
  %.state129 = load i64, ptr %.slot28, align 4
  %172 = add i64 %.state129, 0
  %173 = sdiv i64 %172, 1
  %174 = srem i64 %173, 16384
  %.spill.load130 = load i64, ptr %.spill15, align 4
  %175 = add i64 %.spill.load130, %174
  store i64 %175, ptr %.spill33, align 4
  %176 = icmp sge i64 %175, 0
  %177 = icmp slt i64 %175, 16384
  %178 = and i1 %176, %177
  br i1 %178, label %direct.true131, label %direct.false132

direct.schedule.14:                               ; preds = %direct.true131
  %tile_snapshot.spill.load133 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %179 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load133, 0
  %180 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load133, 1
  %.spill.load134 = load i64, ptr %.spill33, align 4
  %.splatinsert135 = insertelement <8 x i64> poison, i64 %.spill.load134, i64 0
  %.splat136 = shufflevector <8 x i64> %.splatinsert135, <8 x i64> poison, <8 x i32> zeroinitializer
  %181 = mul <8 x i64> %.splat136, splat (i64 4)
  %182 = add <8 x i64> %180, %181
  %183 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %179, 0
  %184 = insertvalue { <8 x ptr>, <8 x i64> } %183, <8 x i64> %182, 1
  %185 = extractvalue { <8 x ptr>, <8 x i64> } %184, 0
  %186 = extractvalue { <8 x ptr>, <8 x i64> } %184, 1
  %187 = getelementptr i8, <8 x ptr> %185, <8 x i64> %186
  %188 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %187, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %189 = load <8 x float>, ptr %.slot34, align 32
  %190 = select <8 x i1> %59, <8 x float> %188, <8 x float> %189
  store <8 x float> %190, ptr %.slot34, align 32
  br label %direct.schedule.15

direct.schedule.15:                               ; preds = %direct.schedule.14, %direct.false132
  %.state137 = load <8 x float>, ptr %.slot29, align 32
  %.state138 = load <8 x float>, ptr %.slot34, align 32
  %191 = fadd <8 x float> %.state137, %.state138
  %192 = load <8 x float>, ptr %.spill35, align 32
  %193 = select <8 x i1> %59, <8 x float> %191, <8 x float> %192
  store <8 x float> %193, ptr %.spill35, align 32
  %.state139 = load i64, ptr %.slot28, align 4
  %194 = add i64 %.state139, 1
  %195 = sdiv i64 %194, 1
  %196 = srem i64 %195, 16384
  %.spill.load140 = load i64, ptr %.spill15, align 4
  %197 = add i64 %.spill.load140, %196
  store i64 %197, ptr %.spill36, align 4
  %198 = icmp sge i64 %197, 0
  %199 = icmp slt i64 %197, 16384
  %200 = and i1 %198, %199
  br i1 %200, label %direct.true141, label %direct.false142

direct.schedule.16:                               ; preds = %direct.true141
  %tile_snapshot.spill.load143 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %201 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load143, 0
  %202 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load143, 1
  %.spill.load144 = load i64, ptr %.spill36, align 4
  %.splatinsert145 = insertelement <8 x i64> poison, i64 %.spill.load144, i64 0
  %.splat146 = shufflevector <8 x i64> %.splatinsert145, <8 x i64> poison, <8 x i32> zeroinitializer
  %203 = mul <8 x i64> %.splat146, splat (i64 4)
  %204 = add <8 x i64> %202, %203
  %205 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %201, 0
  %206 = insertvalue { <8 x ptr>, <8 x i64> } %205, <8 x i64> %204, 1
  %207 = extractvalue { <8 x ptr>, <8 x i64> } %206, 0
  %208 = extractvalue { <8 x ptr>, <8 x i64> } %206, 1
  %209 = getelementptr i8, <8 x ptr> %207, <8 x i64> %208
  %210 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %209, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %211 = load <8 x float>, ptr %.slot37, align 32
  %212 = select <8 x i1> %59, <8 x float> %210, <8 x float> %211
  store <8 x float> %212, ptr %.slot37, align 32
  br label %direct.schedule.17

direct.schedule.17:                               ; preds = %direct.schedule.16, %direct.false142
  %.state147 = load <8 x float>, ptr %.slot30, align 32
  %.state148 = load <8 x float>, ptr %.slot37, align 32
  %213 = fadd <8 x float> %.state147, %.state148
  %214 = load <8 x float>, ptr %.spill38, align 32
  %215 = select <8 x i1> %59, <8 x float> %213, <8 x float> %214
  store <8 x float> %215, ptr %.spill38, align 32
  %.state149 = load i64, ptr %.slot28, align 4
  %216 = add i64 %.state149, 2
  %217 = sdiv i64 %216, 1
  %218 = srem i64 %217, 16384
  %.spill.load150 = load i64, ptr %.spill15, align 4
  %219 = add i64 %.spill.load150, %218
  store i64 %219, ptr %.spill39, align 4
  %220 = icmp sge i64 %219, 0
  %221 = icmp slt i64 %219, 16384
  %222 = and i1 %220, %221
  br i1 %222, label %direct.true151, label %direct.false152

direct.schedule.18:                               ; preds = %direct.true151
  %tile_snapshot.spill.load153 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %223 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load153, 0
  %224 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load153, 1
  %.spill.load154 = load i64, ptr %.spill39, align 4
  %.splatinsert155 = insertelement <8 x i64> poison, i64 %.spill.load154, i64 0
  %.splat156 = shufflevector <8 x i64> %.splatinsert155, <8 x i64> poison, <8 x i32> zeroinitializer
  %225 = mul <8 x i64> %.splat156, splat (i64 4)
  %226 = add <8 x i64> %224, %225
  %227 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %223, 0
  %228 = insertvalue { <8 x ptr>, <8 x i64> } %227, <8 x i64> %226, 1
  %229 = extractvalue { <8 x ptr>, <8 x i64> } %228, 0
  %230 = extractvalue { <8 x ptr>, <8 x i64> } %228, 1
  %231 = getelementptr i8, <8 x ptr> %229, <8 x i64> %230
  %232 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %231, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %233 = load <8 x float>, ptr %.slot40, align 32
  %234 = select <8 x i1> %59, <8 x float> %232, <8 x float> %233
  store <8 x float> %234, ptr %.slot40, align 32
  br label %direct.schedule.19

direct.schedule.19:                               ; preds = %direct.schedule.18, %direct.false152
  %.state157 = load <8 x float>, ptr %.slot31, align 32
  %.state158 = load <8 x float>, ptr %.slot40, align 32
  %235 = fadd <8 x float> %.state157, %.state158
  %236 = load <8 x float>, ptr %.spill41, align 32
  %237 = select <8 x i1> %59, <8 x float> %235, <8 x float> %236
  store <8 x float> %237, ptr %.spill41, align 32
  %.state159 = load i64, ptr %.slot28, align 4
  %238 = add i64 %.state159, 3
  %239 = sdiv i64 %238, 1
  %240 = srem i64 %239, 16384
  %.spill.load160 = load i64, ptr %.spill15, align 4
  %241 = add i64 %.spill.load160, %240
  store i64 %241, ptr %.spill42, align 4
  %242 = icmp sge i64 %241, 0
  %243 = icmp slt i64 %241, 16384
  %244 = and i1 %242, %243
  br i1 %244, label %direct.true161, label %direct.false162

direct.schedule.20:                               ; preds = %direct.true161
  %tile_snapshot.spill.load163 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %245 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load163, 0
  %246 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load163, 1
  %.spill.load164 = load i64, ptr %.spill42, align 4
  %.splatinsert165 = insertelement <8 x i64> poison, i64 %.spill.load164, i64 0
  %.splat166 = shufflevector <8 x i64> %.splatinsert165, <8 x i64> poison, <8 x i32> zeroinitializer
  %247 = mul <8 x i64> %.splat166, splat (i64 4)
  %248 = add <8 x i64> %246, %247
  %249 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %245, 0
  %250 = insertvalue { <8 x ptr>, <8 x i64> } %249, <8 x i64> %248, 1
  %251 = extractvalue { <8 x ptr>, <8 x i64> } %250, 0
  %252 = extractvalue { <8 x ptr>, <8 x i64> } %250, 1
  %253 = getelementptr i8, <8 x ptr> %251, <8 x i64> %252
  %254 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %253, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %255 = load <8 x float>, ptr %.slot43, align 32
  %256 = select <8 x i1> %59, <8 x float> %254, <8 x float> %255
  store <8 x float> %256, ptr %.slot43, align 32
  br label %direct.schedule.21

direct.schedule.21:                               ; preds = %direct.schedule.20, %direct.false162
  %.state167 = load <8 x float>, ptr %.slot32, align 32
  %.state168 = load <8 x float>, ptr %.slot43, align 32
  %257 = fadd <8 x float> %.state167, %.state168
  %.state169 = load i64, ptr %.slot28, align 4
  %258 = add i64 %.state169, 4
  %.spill.load170 = load <8 x float>, ptr %.spill35, align 32
  %.spill.load171 = load <8 x float>, ptr %.spill38, align 32
  %.spill.load172 = load <8 x float>, ptr %.spill41, align 32
  store i64 %258, ptr %.slot28, align 4
  %259 = load <8 x float>, ptr %.slot29, align 32
  %260 = select <8 x i1> %59, <8 x float> %.spill.load170, <8 x float> %259
  store <8 x float> %260, ptr %.slot29, align 32
  %261 = load <8 x float>, ptr %.slot30, align 32
  %262 = select <8 x i1> %59, <8 x float> %.spill.load171, <8 x float> %261
  store <8 x float> %262, ptr %.slot30, align 32
  %263 = load <8 x float>, ptr %.slot31, align 32
  %264 = select <8 x i1> %59, <8 x float> %.spill.load172, <8 x float> %263
  store <8 x float> %264, ptr %.slot31, align 32
  %265 = load <8 x float>, ptr %.slot32, align 32
  %266 = select <8 x i1> %59, <8 x float> %257, <8 x float> %265
  store <8 x float> %266, ptr %.slot32, align 32
  br label %direct.schedule.12

direct.schedule.22:                               ; preds = %direct.false128
  %.spill.load173 = load float, ptr %.spill13, align 4
  %.splatinsert174 = insertelement <8 x float> poison, float %.spill.load173, i64 0
  %.splat175 = shufflevector <8 x float> %.splatinsert174, <8 x float> poison, <8 x i32> zeroinitializer
  %.state176 = load <8 x float>, ptr %.slot29, align 32
  %267 = fadd <8 x float> %.splat175, %.state176
  %.state177 = load <8 x float>, ptr %.slot30, align 32
  %268 = fadd <8 x float> %267, %.state177
  %.state178 = load <8 x float>, ptr %.slot31, align 32
  %269 = fadd <8 x float> %268, %.state178
  %.state179 = load <8 x float>, ptr %.slot32, align 32
  %270 = fadd <8 x float> %269, %.state179
  store float 1.638400e+04, ptr %.spill44, align 4
  %271 = fdiv <8 x float> %270, splat (float 1.638400e+04)
  %272 = load <8 x float>, ptr %.spill45, align 32
  %273 = select <8 x i1> %59, <8 x float> %271, <8 x float> %272
  store <8 x float> %273, ptr %.spill45, align 32
  %274 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill46, align 64
  %275 = extractvalue { <8 x ptr>, <8 x i64> } %5, 0
  %276 = extractvalue { <8 x ptr>, <8 x i64> } %274, 0
  %277 = select <8 x i1> %59, <8 x ptr> %275, <8 x ptr> %276
  %278 = extractvalue { <8 x ptr>, <8 x i64> } %5, 1
  %279 = extractvalue { <8 x ptr>, <8 x i64> } %274, 1
  %280 = select <8 x i1> %59, <8 x i64> %278, <8 x i64> %279
  %281 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %277, 0
  %282 = insertvalue { <8 x ptr>, <8 x i64> } %281, <8 x i64> %280, 1
  store { <8 x ptr>, <8 x i64> } %282, ptr %tile_snapshot.spill46, align 64
  store i64 0, ptr %.slot47, align 4
  br label %direct.schedule.23

direct.schedule.23:                               ; preds = %direct.schedule.24, %direct.schedule.22
  %.state180 = load i64, ptr %.slot47, align 4
  %283 = icmp slt i64 %.state180, 16384
  br i1 %283, label %direct.true181, label %direct.false182

direct.schedule.24:                               ; preds = %direct.true181
  %.state183 = load i64, ptr %.slot47, align 4
  %284 = srem i64 %.state183, 16384
  %285 = add i64 0, %284
  %tile_snapshot.spill.load184 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %286 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load184, 0
  %287 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load184, 1
  %.splatinsert185 = insertelement <8 x i64> poison, i64 %285, i64 0
  %.splat186 = shufflevector <8 x i64> %.splatinsert185, <8 x i64> poison, <8 x i32> zeroinitializer
  %288 = mul <8 x i64> %.splat186, splat (i64 4)
  %289 = add <8 x i64> %287, %288
  %290 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %286, 0
  %291 = insertvalue { <8 x ptr>, <8 x i64> } %290, <8 x i64> %289, 1
  %292 = extractvalue { <8 x ptr>, <8 x i64> } %291, 0
  %293 = extractvalue { <8 x ptr>, <8 x i64> } %291, 1
  %294 = getelementptr i8, <8 x ptr> %292, <8 x i64> %293
  %295 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %294, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %.spill.load187 = load <8 x float>, ptr %.spill45, align 32
  %296 = fsub <8 x float> %295, %.spill.load187
  %tile_snapshot.spill.load188 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill46, align 64
  %297 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load188, 0
  %298 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load188, 1
  %.state189 = load i64, ptr %.slot47, align 4
  %.splatinsert190 = insertelement <8 x i64> poison, i64 %.state189, i64 0
  %.splat191 = shufflevector <8 x i64> %.splatinsert190, <8 x i64> poison, <8 x i32> zeroinitializer
  %299 = mul <8 x i64> %.splat191, splat (i64 4)
  %300 = add <8 x i64> %298, %299
  %301 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %297, 0
  %302 = insertvalue { <8 x ptr>, <8 x i64> } %301, <8 x i64> %300, 1
  %303 = extractvalue { <8 x ptr>, <8 x i64> } %302, 0
  %304 = extractvalue { <8 x ptr>, <8 x i64> } %302, 1
  %305 = getelementptr i8, <8 x ptr> %303, <8 x i64> %304
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %296, <8 x ptr> %305, i32 1, <8 x i1> %59)
  %.state192 = load i64, ptr %.slot47, align 4
  %306 = add i64 %.state192, 1
  store i64 %306, ptr %.slot47, align 4
  br label %direct.schedule.23

direct.schedule.25:                               ; preds = %direct.false182
  %.spill.load193 = load i1, ptr %.spill17, align 1
  br i1 %.spill.load193, label %direct.true194, label %direct.false195

direct.schedule.26:                               ; preds = %direct.true194
  %.spill.load196 = load i64, ptr %.spill16, align 4
  %307 = srem i64 %.spill.load196, 16384
  %308 = add i64 0, %307
  %tile_snapshot.spill.load197 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill46, align 64
  %309 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load197, 0
  %310 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load197, 1
  %.splatinsert198 = insertelement <8 x i64> poison, i64 %308, i64 0
  %.splat199 = shufflevector <8 x i64> %.splatinsert198, <8 x i64> poison, <8 x i32> zeroinitializer
  %311 = mul <8 x i64> %.splat199, splat (i64 4)
  %312 = add <8 x i64> %310, %311
  %313 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %309, 0
  %314 = insertvalue { <8 x ptr>, <8 x i64> } %313, <8 x i64> %312, 1
  %315 = extractvalue { <8 x ptr>, <8 x i64> } %314, 0
  %316 = extractvalue { <8 x ptr>, <8 x i64> } %314, 1
  %317 = getelementptr i8, <8 x ptr> %315, <8 x i64> %316
  %318 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %317, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %319 = fmul <8 x float> %318, %318
  %320 = load <8 x float>, ptr %.slot48, align 32
  %321 = select <8 x i1> %59, <8 x float> %319, <8 x float> %320
  store <8 x float> %321, ptr %.slot48, align 32
  br label %direct.schedule.27

direct.schedule.27:                               ; preds = %direct.schedule.26, %direct.false195
  %.spill.load200 = load i1, ptr %.spill20, align 1
  br i1 %.spill.load200, label %direct.true201, label %direct.false202

direct.schedule.28:                               ; preds = %direct.true201
  %.spill.load203 = load i64, ptr %.spill19, align 4
  %322 = srem i64 %.spill.load203, 16384
  %323 = add i64 0, %322
  %tile_snapshot.spill.load204 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill46, align 64
  %324 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load204, 0
  %325 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load204, 1
  %.splatinsert205 = insertelement <8 x i64> poison, i64 %323, i64 0
  %.splat206 = shufflevector <8 x i64> %.splatinsert205, <8 x i64> poison, <8 x i32> zeroinitializer
  %326 = mul <8 x i64> %.splat206, splat (i64 4)
  %327 = add <8 x i64> %325, %326
  %328 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %324, 0
  %329 = insertvalue { <8 x ptr>, <8 x i64> } %328, <8 x i64> %327, 1
  %330 = extractvalue { <8 x ptr>, <8 x i64> } %329, 0
  %331 = extractvalue { <8 x ptr>, <8 x i64> } %329, 1
  %332 = getelementptr i8, <8 x ptr> %330, <8 x i64> %331
  %333 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %332, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %334 = fmul <8 x float> %333, %333
  %335 = load <8 x float>, ptr %.slot49, align 32
  %336 = select <8 x i1> %59, <8 x float> %334, <8 x float> %335
  store <8 x float> %336, ptr %.slot49, align 32
  br label %direct.schedule.29

direct.schedule.29:                               ; preds = %direct.schedule.28, %direct.false202
  %.spill.load207 = load i1, ptr %.spill23, align 1
  br i1 %.spill.load207, label %direct.true208, label %direct.false209

direct.schedule.30:                               ; preds = %direct.true208
  %.spill.load210 = load i64, ptr %.spill22, align 4
  %337 = srem i64 %.spill.load210, 16384
  %338 = add i64 0, %337
  %tile_snapshot.spill.load211 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill46, align 64
  %339 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load211, 0
  %340 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load211, 1
  %.splatinsert212 = insertelement <8 x i64> poison, i64 %338, i64 0
  %.splat213 = shufflevector <8 x i64> %.splatinsert212, <8 x i64> poison, <8 x i32> zeroinitializer
  %341 = mul <8 x i64> %.splat213, splat (i64 4)
  %342 = add <8 x i64> %340, %341
  %343 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %339, 0
  %344 = insertvalue { <8 x ptr>, <8 x i64> } %343, <8 x i64> %342, 1
  %345 = extractvalue { <8 x ptr>, <8 x i64> } %344, 0
  %346 = extractvalue { <8 x ptr>, <8 x i64> } %344, 1
  %347 = getelementptr i8, <8 x ptr> %345, <8 x i64> %346
  %348 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %347, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %349 = fmul <8 x float> %348, %348
  %350 = load <8 x float>, ptr %.slot50, align 32
  %351 = select <8 x i1> %59, <8 x float> %349, <8 x float> %350
  store <8 x float> %351, ptr %.slot50, align 32
  br label %direct.schedule.31

direct.schedule.31:                               ; preds = %direct.schedule.30, %direct.false209
  %.spill.load214 = load i1, ptr %.spill26, align 1
  br i1 %.spill.load214, label %direct.true215, label %direct.false216

direct.schedule.32:                               ; preds = %direct.true215
  %.spill.load217 = load i64, ptr %.spill25, align 4
  %352 = srem i64 %.spill.load217, 16384
  %353 = add i64 0, %352
  %tile_snapshot.spill.load218 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill46, align 64
  %354 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load218, 0
  %355 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load218, 1
  %.splatinsert219 = insertelement <8 x i64> poison, i64 %353, i64 0
  %.splat220 = shufflevector <8 x i64> %.splatinsert219, <8 x i64> poison, <8 x i32> zeroinitializer
  %356 = mul <8 x i64> %.splat220, splat (i64 4)
  %357 = add <8 x i64> %355, %356
  %358 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %354, 0
  %359 = insertvalue { <8 x ptr>, <8 x i64> } %358, <8 x i64> %357, 1
  %360 = extractvalue { <8 x ptr>, <8 x i64> } %359, 0
  %361 = extractvalue { <8 x ptr>, <8 x i64> } %359, 1
  %362 = getelementptr i8, <8 x ptr> %360, <8 x i64> %361
  %363 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %362, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %364 = fmul <8 x float> %363, %363
  %365 = load <8 x float>, ptr %.slot51, align 32
  %366 = select <8 x i1> %59, <8 x float> %364, <8 x float> %365
  store <8 x float> %366, ptr %.slot51, align 32
  br label %direct.schedule.33

direct.schedule.33:                               ; preds = %direct.schedule.32, %direct.false216
  %.state221 = load <8 x float>, ptr %.slot48, align 32
  %.state222 = load <8 x float>, ptr %.slot49, align 32
  %.state223 = load <8 x float>, ptr %.slot50, align 32
  %.state224 = load <8 x float>, ptr %.slot51, align 32
  store i64 4, ptr %.slot52, align 4
  %367 = load <8 x float>, ptr %.slot53, align 32
  %368 = select <8 x i1> %59, <8 x float> %.state221, <8 x float> %367
  store <8 x float> %368, ptr %.slot53, align 32
  %369 = load <8 x float>, ptr %.slot54, align 32
  %370 = select <8 x i1> %59, <8 x float> %.state222, <8 x float> %369
  store <8 x float> %370, ptr %.slot54, align 32
  %371 = load <8 x float>, ptr %.slot55, align 32
  %372 = select <8 x i1> %59, <8 x float> %.state223, <8 x float> %371
  store <8 x float> %372, ptr %.slot55, align 32
  %373 = load <8 x float>, ptr %.slot56, align 32
  %374 = select <8 x i1> %59, <8 x float> %.state224, <8 x float> %373
  store <8 x float> %374, ptr %.slot56, align 32
  br label %direct.schedule.34

direct.schedule.34:                               ; preds = %direct.schedule.43, %direct.schedule.33
  %.state225 = load i64, ptr %.slot52, align 4
  %375 = icmp slt i64 %.state225, 16384
  br i1 %375, label %direct.true226, label %direct.false227

direct.schedule.35:                               ; preds = %direct.true226
  %.state228 = load i64, ptr %.slot52, align 4
  %376 = add i64 %.state228, 0
  %377 = sdiv i64 %376, 1
  %378 = srem i64 %377, 16384
  %.spill.load229 = load i64, ptr %.spill15, align 4
  %379 = add i64 %.spill.load229, %378
  store i64 %379, ptr %.spill57, align 4
  %380 = icmp sge i64 %379, 0
  %381 = icmp slt i64 %379, 16384
  %382 = and i1 %380, %381
  br i1 %382, label %direct.true230, label %direct.false231

direct.schedule.36:                               ; preds = %direct.true230
  %.spill.load232 = load i64, ptr %.spill57, align 4
  %383 = srem i64 %.spill.load232, 16384
  %384 = add i64 0, %383
  %tile_snapshot.spill.load233 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill46, align 64
  %385 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load233, 0
  %386 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load233, 1
  %.splatinsert234 = insertelement <8 x i64> poison, i64 %384, i64 0
  %.splat235 = shufflevector <8 x i64> %.splatinsert234, <8 x i64> poison, <8 x i32> zeroinitializer
  %387 = mul <8 x i64> %.splat235, splat (i64 4)
  %388 = add <8 x i64> %386, %387
  %389 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %385, 0
  %390 = insertvalue { <8 x ptr>, <8 x i64> } %389, <8 x i64> %388, 1
  %391 = extractvalue { <8 x ptr>, <8 x i64> } %390, 0
  %392 = extractvalue { <8 x ptr>, <8 x i64> } %390, 1
  %393 = getelementptr i8, <8 x ptr> %391, <8 x i64> %392
  %394 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %393, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %395 = fmul <8 x float> %394, %394
  %396 = load <8 x float>, ptr %.slot58, align 32
  %397 = select <8 x i1> %59, <8 x float> %395, <8 x float> %396
  store <8 x float> %397, ptr %.slot58, align 32
  br label %direct.schedule.37

direct.schedule.37:                               ; preds = %direct.schedule.36, %direct.false231
  %.state236 = load <8 x float>, ptr %.slot53, align 32
  %.state237 = load <8 x float>, ptr %.slot58, align 32
  %398 = fadd <8 x float> %.state236, %.state237
  %399 = load <8 x float>, ptr %.spill59, align 32
  %400 = select <8 x i1> %59, <8 x float> %398, <8 x float> %399
  store <8 x float> %400, ptr %.spill59, align 32
  %.state238 = load i64, ptr %.slot52, align 4
  %401 = add i64 %.state238, 1
  %402 = sdiv i64 %401, 1
  %403 = srem i64 %402, 16384
  %.spill.load239 = load i64, ptr %.spill15, align 4
  %404 = add i64 %.spill.load239, %403
  store i64 %404, ptr %.spill60, align 4
  %405 = icmp sge i64 %404, 0
  %406 = icmp slt i64 %404, 16384
  %407 = and i1 %405, %406
  br i1 %407, label %direct.true240, label %direct.false241

direct.schedule.38:                               ; preds = %direct.true240
  %.spill.load242 = load i64, ptr %.spill60, align 4
  %408 = srem i64 %.spill.load242, 16384
  %409 = add i64 0, %408
  %tile_snapshot.spill.load243 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill46, align 64
  %410 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load243, 0
  %411 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load243, 1
  %.splatinsert244 = insertelement <8 x i64> poison, i64 %409, i64 0
  %.splat245 = shufflevector <8 x i64> %.splatinsert244, <8 x i64> poison, <8 x i32> zeroinitializer
  %412 = mul <8 x i64> %.splat245, splat (i64 4)
  %413 = add <8 x i64> %411, %412
  %414 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %410, 0
  %415 = insertvalue { <8 x ptr>, <8 x i64> } %414, <8 x i64> %413, 1
  %416 = extractvalue { <8 x ptr>, <8 x i64> } %415, 0
  %417 = extractvalue { <8 x ptr>, <8 x i64> } %415, 1
  %418 = getelementptr i8, <8 x ptr> %416, <8 x i64> %417
  %419 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %418, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %420 = fmul <8 x float> %419, %419
  %421 = load <8 x float>, ptr %.slot61, align 32
  %422 = select <8 x i1> %59, <8 x float> %420, <8 x float> %421
  store <8 x float> %422, ptr %.slot61, align 32
  br label %direct.schedule.39

direct.schedule.39:                               ; preds = %direct.schedule.38, %direct.false241
  %.state246 = load <8 x float>, ptr %.slot54, align 32
  %.state247 = load <8 x float>, ptr %.slot61, align 32
  %423 = fadd <8 x float> %.state246, %.state247
  %424 = load <8 x float>, ptr %.spill62, align 32
  %425 = select <8 x i1> %59, <8 x float> %423, <8 x float> %424
  store <8 x float> %425, ptr %.spill62, align 32
  %.state248 = load i64, ptr %.slot52, align 4
  %426 = add i64 %.state248, 2
  %427 = sdiv i64 %426, 1
  %428 = srem i64 %427, 16384
  %.spill.load249 = load i64, ptr %.spill15, align 4
  %429 = add i64 %.spill.load249, %428
  store i64 %429, ptr %.spill63, align 4
  %430 = icmp sge i64 %429, 0
  %431 = icmp slt i64 %429, 16384
  %432 = and i1 %430, %431
  br i1 %432, label %direct.true250, label %direct.false251

direct.schedule.40:                               ; preds = %direct.true250
  %.spill.load252 = load i64, ptr %.spill63, align 4
  %433 = srem i64 %.spill.load252, 16384
  %434 = add i64 0, %433
  %tile_snapshot.spill.load253 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill46, align 64
  %435 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load253, 0
  %436 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load253, 1
  %.splatinsert254 = insertelement <8 x i64> poison, i64 %434, i64 0
  %.splat255 = shufflevector <8 x i64> %.splatinsert254, <8 x i64> poison, <8 x i32> zeroinitializer
  %437 = mul <8 x i64> %.splat255, splat (i64 4)
  %438 = add <8 x i64> %436, %437
  %439 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %435, 0
  %440 = insertvalue { <8 x ptr>, <8 x i64> } %439, <8 x i64> %438, 1
  %441 = extractvalue { <8 x ptr>, <8 x i64> } %440, 0
  %442 = extractvalue { <8 x ptr>, <8 x i64> } %440, 1
  %443 = getelementptr i8, <8 x ptr> %441, <8 x i64> %442
  %444 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %443, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %445 = fmul <8 x float> %444, %444
  %446 = load <8 x float>, ptr %.slot64, align 32
  %447 = select <8 x i1> %59, <8 x float> %445, <8 x float> %446
  store <8 x float> %447, ptr %.slot64, align 32
  br label %direct.schedule.41

direct.schedule.41:                               ; preds = %direct.schedule.40, %direct.false251
  %.state256 = load <8 x float>, ptr %.slot55, align 32
  %.state257 = load <8 x float>, ptr %.slot64, align 32
  %448 = fadd <8 x float> %.state256, %.state257
  %449 = load <8 x float>, ptr %.spill65, align 32
  %450 = select <8 x i1> %59, <8 x float> %448, <8 x float> %449
  store <8 x float> %450, ptr %.spill65, align 32
  %.state258 = load i64, ptr %.slot52, align 4
  %451 = add i64 %.state258, 3
  %452 = sdiv i64 %451, 1
  %453 = srem i64 %452, 16384
  %.spill.load259 = load i64, ptr %.spill15, align 4
  %454 = add i64 %.spill.load259, %453
  store i64 %454, ptr %.spill66, align 4
  %455 = icmp sge i64 %454, 0
  %456 = icmp slt i64 %454, 16384
  %457 = and i1 %455, %456
  br i1 %457, label %direct.true260, label %direct.false261

direct.schedule.42:                               ; preds = %direct.true260
  %.spill.load262 = load i64, ptr %.spill66, align 4
  %458 = srem i64 %.spill.load262, 16384
  %459 = add i64 0, %458
  %tile_snapshot.spill.load263 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill46, align 64
  %460 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load263, 0
  %461 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load263, 1
  %.splatinsert264 = insertelement <8 x i64> poison, i64 %459, i64 0
  %.splat265 = shufflevector <8 x i64> %.splatinsert264, <8 x i64> poison, <8 x i32> zeroinitializer
  %462 = mul <8 x i64> %.splat265, splat (i64 4)
  %463 = add <8 x i64> %461, %462
  %464 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %460, 0
  %465 = insertvalue { <8 x ptr>, <8 x i64> } %464, <8 x i64> %463, 1
  %466 = extractvalue { <8 x ptr>, <8 x i64> } %465, 0
  %467 = extractvalue { <8 x ptr>, <8 x i64> } %465, 1
  %468 = getelementptr i8, <8 x ptr> %466, <8 x i64> %467
  %469 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %468, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %470 = fmul <8 x float> %469, %469
  %471 = load <8 x float>, ptr %.slot67, align 32
  %472 = select <8 x i1> %59, <8 x float> %470, <8 x float> %471
  store <8 x float> %472, ptr %.slot67, align 32
  br label %direct.schedule.43

direct.schedule.43:                               ; preds = %direct.schedule.42, %direct.false261
  %.state266 = load <8 x float>, ptr %.slot56, align 32
  %.state267 = load <8 x float>, ptr %.slot67, align 32
  %473 = fadd <8 x float> %.state266, %.state267
  %.state268 = load i64, ptr %.slot52, align 4
  %474 = add i64 %.state268, 4
  %.spill.load269 = load <8 x float>, ptr %.spill59, align 32
  %.spill.load270 = load <8 x float>, ptr %.spill62, align 32
  %.spill.load271 = load <8 x float>, ptr %.spill65, align 32
  store i64 %474, ptr %.slot52, align 4
  %475 = load <8 x float>, ptr %.slot53, align 32
  %476 = select <8 x i1> %59, <8 x float> %.spill.load269, <8 x float> %475
  store <8 x float> %476, ptr %.slot53, align 32
  %477 = load <8 x float>, ptr %.slot54, align 32
  %478 = select <8 x i1> %59, <8 x float> %.spill.load270, <8 x float> %477
  store <8 x float> %478, ptr %.slot54, align 32
  %479 = load <8 x float>, ptr %.slot55, align 32
  %480 = select <8 x i1> %59, <8 x float> %.spill.load271, <8 x float> %479
  store <8 x float> %480, ptr %.slot55, align 32
  %481 = load <8 x float>, ptr %.slot56, align 32
  %482 = select <8 x i1> %59, <8 x float> %473, <8 x float> %481
  store <8 x float> %482, ptr %.slot56, align 32
  br label %direct.schedule.34

direct.schedule.44:                               ; preds = %direct.false227
  %.spill.load272 = load float, ptr %.spill13, align 4
  %.splatinsert273 = insertelement <8 x float> poison, float %.spill.load272, i64 0
  %.splat274 = shufflevector <8 x float> %.splatinsert273, <8 x float> poison, <8 x i32> zeroinitializer
  %.state275 = load <8 x float>, ptr %.slot53, align 32
  %483 = fadd <8 x float> %.splat274, %.state275
  %.state276 = load <8 x float>, ptr %.slot54, align 32
  %484 = fadd <8 x float> %483, %.state276
  %.state277 = load <8 x float>, ptr %.slot55, align 32
  %485 = fadd <8 x float> %484, %.state277
  %.state278 = load <8 x float>, ptr %.slot56, align 32
  %486 = fadd <8 x float> %485, %.state278
  %.spill.load279 = load float, ptr %.spill44, align 4
  %.splatinsert280 = insertelement <8 x float> poison, float %.spill.load279, i64 0
  %.splat281 = shufflevector <8 x float> %.splatinsert280, <8 x float> poison, <8 x i32> zeroinitializer
  %487 = fdiv <8 x float> %486, %.splat281
  %488 = load <8 x float>, ptr %.spill68, align 32
  %489 = select <8 x i1> %59, <8 x float> %487, <8 x float> %488
  store <8 x float> %489, ptr %.spill68, align 32
  %490 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill69, align 64
  %491 = extractvalue { <8 x ptr>, <8 x i64> } %8, 0
  %492 = extractvalue { <8 x ptr>, <8 x i64> } %490, 0
  %493 = select <8 x i1> %59, <8 x ptr> %491, <8 x ptr> %492
  %494 = extractvalue { <8 x ptr>, <8 x i64> } %8, 1
  %495 = extractvalue { <8 x ptr>, <8 x i64> } %490, 1
  %496 = select <8 x i1> %59, <8 x i64> %494, <8 x i64> %495
  %497 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %493, 0
  %498 = insertvalue { <8 x ptr>, <8 x i64> } %497, <8 x i64> %496, 1
  store { <8 x ptr>, <8 x i64> } %498, ptr %tile_snapshot.spill69, align 64
  store i64 0, ptr %.slot70, align 4
  br label %direct.schedule.45

direct.schedule.45:                               ; preds = %direct.schedule.46, %direct.schedule.44
  %.state282 = load i64, ptr %.slot70, align 4
  %499 = icmp slt i64 %.state282, 16384
  br i1 %499, label %direct.true283, label %direct.false284

direct.schedule.46:                               ; preds = %direct.true283
  %.state285 = load i64, ptr %.slot70, align 4
  %500 = srem i64 %.state285, 16384
  %.state286 = load i64, ptr %.slot70, align 4
  %501 = sdiv i64 %.state286, 16384
  %502 = srem i64 %501, 1
  %503 = add i64 0, %502
  %.spill.load287 = load i64, ptr %.spill14, align 4
  %504 = add i64 %.spill.load287, %503
  %505 = add i64 0, %500
  %506 = mul i64 %504, 16384
  %507 = add i64 %506, %505
  %508 = extractvalue { ptr, i64 } %23, 0
  %509 = mul i64 %507, 4
  %510 = getelementptr i8, ptr %508, i64 %509
  %511 = load float, ptr %510, align 4
  %.splatinsert288 = insertelement <8 x float> poison, float %511, i64 0
  %.splat289 = shufflevector <8 x float> %.splatinsert288, <8 x float> poison, <8 x i32> zeroinitializer
  %tile_snapshot.spill.load290 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill69, align 64
  %512 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load290, 0
  %513 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load290, 1
  %.state291 = load i64, ptr %.slot70, align 4
  %.splatinsert292 = insertelement <8 x i64> poison, i64 %.state291, i64 0
  %.splat293 = shufflevector <8 x i64> %.splatinsert292, <8 x i64> poison, <8 x i32> zeroinitializer
  %514 = mul <8 x i64> %.splat293, splat (i64 4)
  %515 = add <8 x i64> %513, %514
  %516 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %512, 0
  %517 = insertvalue { <8 x ptr>, <8 x i64> } %516, <8 x i64> %515, 1
  %518 = extractvalue { <8 x ptr>, <8 x i64> } %517, 0
  %519 = extractvalue { <8 x ptr>, <8 x i64> } %517, 1
  %520 = getelementptr i8, <8 x ptr> %518, <8 x i64> %519
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %.splat289, <8 x ptr> %520, i32 1, <8 x i1> %59)
  %.state294 = load i64, ptr %.slot70, align 4
  %521 = add i64 %.state294, 1
  store i64 %521, ptr %.slot70, align 4
  br label %direct.schedule.45

direct.schedule.47:                               ; preds = %direct.false284
  %.spill.load295 = load <8 x float>, ptr %.spill68, align 32
  %522 = fadd <8 x float> %.spill.load295, splat (float 0x3EE4F8B580000000)
  %523 = call <8 x float> @llvm.sqrt.v8f32(<8 x float> %522)
  %524 = load <8 x float>, ptr %.spill71, align 32
  %525 = select <8 x i1> %59, <8 x float> %523, <8 x float> %524
  store <8 x float> %525, ptr %.spill71, align 32
  %526 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill72, align 64
  %527 = extractvalue { <8 x ptr>, <8 x i64> } %11, 0
  %528 = extractvalue { <8 x ptr>, <8 x i64> } %526, 0
  %529 = select <8 x i1> %59, <8 x ptr> %527, <8 x ptr> %528
  %530 = extractvalue { <8 x ptr>, <8 x i64> } %11, 1
  %531 = extractvalue { <8 x ptr>, <8 x i64> } %526, 1
  %532 = select <8 x i1> %59, <8 x i64> %530, <8 x i64> %531
  %533 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %529, 0
  %534 = insertvalue { <8 x ptr>, <8 x i64> } %533, <8 x i64> %532, 1
  store { <8 x ptr>, <8 x i64> } %534, ptr %tile_snapshot.spill72, align 64
  store i64 0, ptr %.slot73, align 4
  br label %direct.schedule.48

direct.schedule.48:                               ; preds = %direct.schedule.49, %direct.schedule.47
  %.state296 = load i64, ptr %.slot73, align 4
  %535 = icmp slt i64 %.state296, 16384
  br i1 %535, label %direct.true297, label %direct.false298

direct.schedule.49:                               ; preds = %direct.true297
  %.state299 = load i64, ptr %.slot73, align 4
  %536 = srem i64 %.state299, 16384
  %.state300 = load i64, ptr %.slot73, align 4
  %537 = sdiv i64 %.state300, 16384
  %538 = srem i64 %537, 1
  %539 = add i64 0, %538
  %.spill.load301 = load i64, ptr %.spill14, align 4
  %540 = add i64 %.spill.load301, %539
  %541 = add i64 0, %536
  %542 = mul i64 %540, 16384
  %543 = add i64 %542, %541
  %544 = extractvalue { ptr, i64 } %29, 0
  %545 = mul i64 %543, 4
  %546 = getelementptr i8, ptr %544, i64 %545
  %547 = load float, ptr %546, align 4
  %.splatinsert302 = insertelement <8 x float> poison, float %547, i64 0
  %.splat303 = shufflevector <8 x float> %.splatinsert302, <8 x float> poison, <8 x i32> zeroinitializer
  %tile_snapshot.spill.load304 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill72, align 64
  %548 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load304, 0
  %549 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load304, 1
  %.state305 = load i64, ptr %.slot73, align 4
  %.splatinsert306 = insertelement <8 x i64> poison, i64 %.state305, i64 0
  %.splat307 = shufflevector <8 x i64> %.splatinsert306, <8 x i64> poison, <8 x i32> zeroinitializer
  %550 = mul <8 x i64> %.splat307, splat (i64 4)
  %551 = add <8 x i64> %549, %550
  %552 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %548, 0
  %553 = insertvalue { <8 x ptr>, <8 x i64> } %552, <8 x i64> %551, 1
  %554 = extractvalue { <8 x ptr>, <8 x i64> } %553, 0
  %555 = extractvalue { <8 x ptr>, <8 x i64> } %553, 1
  %556 = getelementptr i8, <8 x ptr> %554, <8 x i64> %555
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %.splat303, <8 x ptr> %556, i32 1, <8 x i1> %59)
  %.state308 = load i64, ptr %.slot73, align 4
  %557 = add i64 %.state308, 1
  store i64 %557, ptr %.slot73, align 4
  br label %direct.schedule.48

direct.schedule.50:                               ; preds = %direct.false298
  store i64 0, ptr %.slot74, align 4
  br label %direct.schedule.51

direct.schedule.51:                               ; preds = %direct.schedule.52, %direct.schedule.50
  %.state309 = load i64, ptr %.slot74, align 4
  %558 = icmp slt i64 %.state309, 16384
  br i1 %558, label %direct.true310, label %direct.false311

direct.schedule.52:                               ; preds = %direct.true310
  %.state312 = load i64, ptr %.slot74, align 4
  %559 = srem i64 %.state312, 16384
  %.state313 = load i64, ptr %.slot74, align 4
  %560 = sdiv i64 %.state313, 16384
  %561 = srem i64 %560, 1
  %.spill.load314 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert315 = insertelement <8 x i64> poison, i64 %561, i64 0
  %.splat316 = shufflevector <8 x i64> %.splatinsert315, <8 x i64> poison, <8 x i32> zeroinitializer
  %562 = add <8 x i64> %.spill.load314, %.splat316
  %563 = add <8 x i64> zeroinitializer, %562
  %564 = add i64 0, %559
  %565 = mul <8 x i64> %563, splat (i64 16384)
  %.splatinsert317 = insertelement <8 x i64> poison, i64 %564, i64 0
  %.splat318 = shufflevector <8 x i64> %.splatinsert317, <8 x i64> poison, <8 x i32> zeroinitializer
  %566 = add <8 x i64> %565, %.splat318
  %567 = add i64 0, %559
  %568 = srem i64 %567, 16384
  %569 = add i64 0, %568
  %570 = srem i64 %569, 16384
  %571 = add i64 0, %570
  %tile_snapshot.spill.load319 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill46, align 64
  %572 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load319, 0
  %573 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load319, 1
  %.splatinsert320 = insertelement <8 x i64> poison, i64 %571, i64 0
  %.splat321 = shufflevector <8 x i64> %.splatinsert320, <8 x i64> poison, <8 x i32> zeroinitializer
  %574 = mul <8 x i64> %.splat321, splat (i64 4)
  %575 = add <8 x i64> %573, %574
  %576 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %572, 0
  %577 = insertvalue { <8 x ptr>, <8 x i64> } %576, <8 x i64> %575, 1
  %578 = extractvalue { <8 x ptr>, <8 x i64> } %577, 0
  %579 = extractvalue { <8 x ptr>, <8 x i64> } %577, 1
  %580 = getelementptr i8, <8 x ptr> %578, <8 x i64> %579
  %581 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %580, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %.spill.load322 = load <8 x float>, ptr %.spill71, align 32
  %582 = fdiv <8 x float> %581, %.spill.load322
  %tile_snapshot.spill.load323 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill69, align 64
  %583 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load323, 0
  %584 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load323, 1
  %.splatinsert324 = insertelement <8 x i64> poison, i64 %569, i64 0
  %.splat325 = shufflevector <8 x i64> %.splatinsert324, <8 x i64> poison, <8 x i32> zeroinitializer
  %585 = mul <8 x i64> %.splat325, splat (i64 4)
  %586 = add <8 x i64> %584, %585
  %587 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %583, 0
  %588 = insertvalue { <8 x ptr>, <8 x i64> } %587, <8 x i64> %586, 1
  %589 = extractvalue { <8 x ptr>, <8 x i64> } %588, 0
  %590 = extractvalue { <8 x ptr>, <8 x i64> } %588, 1
  %591 = getelementptr i8, <8 x ptr> %589, <8 x i64> %590
  %592 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %591, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %593 = fmul <8 x float> %582, %592
  %tile_snapshot.spill.load326 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill72, align 64
  %594 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load326, 0
  %595 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load326, 1
  %.splatinsert327 = insertelement <8 x i64> poison, i64 %567, i64 0
  %.splat328 = shufflevector <8 x i64> %.splatinsert327, <8 x i64> poison, <8 x i32> zeroinitializer
  %596 = mul <8 x i64> %.splat328, splat (i64 4)
  %597 = add <8 x i64> %595, %596
  %598 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %594, 0
  %599 = insertvalue { <8 x ptr>, <8 x i64> } %598, <8 x i64> %597, 1
  %600 = extractvalue { <8 x ptr>, <8 x i64> } %599, 0
  %601 = extractvalue { <8 x ptr>, <8 x i64> } %599, 1
  %602 = getelementptr i8, <8 x ptr> %600, <8 x i64> %601
  %603 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %602, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %604 = fadd <8 x float> %593, %603
  %605 = extractvalue { ptr, i64 } %35, 0
  %606 = mul <8 x i64> %566, splat (i64 4)
  %607 = getelementptr i8, ptr %605, <8 x i64> %606
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %604, <8 x ptr> %607, i32 1, <8 x i1> %59)
  %.state329 = load i64, ptr %.slot74, align 4
  %608 = add i64 %.state329, 1
  store i64 %608, ptr %.slot74, align 4
  br label %direct.schedule.51

direct.schedule.53:                               ; preds = %direct.false311
  ret void

direct.activate:                                  ; preds = %prologue
  br label %direct.schedule.0

direct.inactive:                                  ; preds = %prologue
  ret void

direct.true:                                      ; preds = %direct.schedule.1
  br label %direct.schedule.2

direct.false:                                     ; preds = %direct.schedule.1
  br label %direct.schedule.3

direct.true95:                                    ; preds = %direct.schedule.3
  br label %direct.schedule.4

direct.false96:                                   ; preds = %direct.schedule.3
  %609 = load <8 x float>, ptr %.slot18, align 32
  %610 = select <8 x i1> %59, <8 x float> zeroinitializer, <8 x float> %609
  store <8 x float> %610, ptr %.slot18, align 32
  br label %direct.schedule.5

direct.true102:                                   ; preds = %direct.schedule.5
  br label %direct.schedule.6

direct.false103:                                  ; preds = %direct.schedule.5
  %611 = load <8 x float>, ptr %.slot21, align 32
  %612 = select <8 x i1> %59, <8 x float> zeroinitializer, <8 x float> %611
  store <8 x float> %612, ptr %.slot21, align 32
  br label %direct.schedule.7

direct.true109:                                   ; preds = %direct.schedule.7
  br label %direct.schedule.8

direct.false110:                                  ; preds = %direct.schedule.7
  %613 = load <8 x float>, ptr %.slot24, align 32
  %614 = select <8 x i1> %59, <8 x float> zeroinitializer, <8 x float> %613
  store <8 x float> %614, ptr %.slot24, align 32
  br label %direct.schedule.9

direct.true116:                                   ; preds = %direct.schedule.9
  br label %direct.schedule.10

direct.false117:                                  ; preds = %direct.schedule.9
  %615 = load <8 x float>, ptr %.slot27, align 32
  %616 = select <8 x i1> %59, <8 x float> zeroinitializer, <8 x float> %615
  store <8 x float> %616, ptr %.slot27, align 32
  br label %direct.schedule.11

direct.true127:                                   ; preds = %direct.schedule.12
  br label %direct.schedule.13

direct.false128:                                  ; preds = %direct.schedule.12
  br label %direct.schedule.22

direct.true131:                                   ; preds = %direct.schedule.13
  br label %direct.schedule.14

direct.false132:                                  ; preds = %direct.schedule.13
  %617 = load <8 x float>, ptr %.slot34, align 32
  %618 = select <8 x i1> %59, <8 x float> zeroinitializer, <8 x float> %617
  store <8 x float> %618, ptr %.slot34, align 32
  br label %direct.schedule.15

direct.true141:                                   ; preds = %direct.schedule.15
  br label %direct.schedule.16

direct.false142:                                  ; preds = %direct.schedule.15
  %619 = load <8 x float>, ptr %.slot37, align 32
  %620 = select <8 x i1> %59, <8 x float> zeroinitializer, <8 x float> %619
  store <8 x float> %620, ptr %.slot37, align 32
  br label %direct.schedule.17

direct.true151:                                   ; preds = %direct.schedule.17
  br label %direct.schedule.18

direct.false152:                                  ; preds = %direct.schedule.17
  %621 = load <8 x float>, ptr %.slot40, align 32
  %622 = select <8 x i1> %59, <8 x float> zeroinitializer, <8 x float> %621
  store <8 x float> %622, ptr %.slot40, align 32
  br label %direct.schedule.19

direct.true161:                                   ; preds = %direct.schedule.19
  br label %direct.schedule.20

direct.false162:                                  ; preds = %direct.schedule.19
  %623 = load <8 x float>, ptr %.slot43, align 32
  %624 = select <8 x i1> %59, <8 x float> zeroinitializer, <8 x float> %623
  store <8 x float> %624, ptr %.slot43, align 32
  br label %direct.schedule.21

direct.true181:                                   ; preds = %direct.schedule.23
  br label %direct.schedule.24

direct.false182:                                  ; preds = %direct.schedule.23
  br label %direct.schedule.25

direct.true194:                                   ; preds = %direct.schedule.25
  br label %direct.schedule.26

direct.false195:                                  ; preds = %direct.schedule.25
  %625 = load <8 x float>, ptr %.slot48, align 32
  %626 = select <8 x i1> %59, <8 x float> zeroinitializer, <8 x float> %625
  store <8 x float> %626, ptr %.slot48, align 32
  br label %direct.schedule.27

direct.true201:                                   ; preds = %direct.schedule.27
  br label %direct.schedule.28

direct.false202:                                  ; preds = %direct.schedule.27
  %627 = load <8 x float>, ptr %.slot49, align 32
  %628 = select <8 x i1> %59, <8 x float> zeroinitializer, <8 x float> %627
  store <8 x float> %628, ptr %.slot49, align 32
  br label %direct.schedule.29

direct.true208:                                   ; preds = %direct.schedule.29
  br label %direct.schedule.30

direct.false209:                                  ; preds = %direct.schedule.29
  %629 = load <8 x float>, ptr %.slot50, align 32
  %630 = select <8 x i1> %59, <8 x float> zeroinitializer, <8 x float> %629
  store <8 x float> %630, ptr %.slot50, align 32
  br label %direct.schedule.31

direct.true215:                                   ; preds = %direct.schedule.31
  br label %direct.schedule.32

direct.false216:                                  ; preds = %direct.schedule.31
  %631 = load <8 x float>, ptr %.slot51, align 32
  %632 = select <8 x i1> %59, <8 x float> zeroinitializer, <8 x float> %631
  store <8 x float> %632, ptr %.slot51, align 32
  br label %direct.schedule.33

direct.true226:                                   ; preds = %direct.schedule.34
  br label %direct.schedule.35

direct.false227:                                  ; preds = %direct.schedule.34
  br label %direct.schedule.44

direct.true230:                                   ; preds = %direct.schedule.35
  br label %direct.schedule.36

direct.false231:                                  ; preds = %direct.schedule.35
  %633 = load <8 x float>, ptr %.slot58, align 32
  %634 = select <8 x i1> %59, <8 x float> zeroinitializer, <8 x float> %633
  store <8 x float> %634, ptr %.slot58, align 32
  br label %direct.schedule.37

direct.true240:                                   ; preds = %direct.schedule.37
  br label %direct.schedule.38

direct.false241:                                  ; preds = %direct.schedule.37
  %635 = load <8 x float>, ptr %.slot61, align 32
  %636 = select <8 x i1> %59, <8 x float> zeroinitializer, <8 x float> %635
  store <8 x float> %636, ptr %.slot61, align 32
  br label %direct.schedule.39

direct.true250:                                   ; preds = %direct.schedule.39
  br label %direct.schedule.40

direct.false251:                                  ; preds = %direct.schedule.39
  %637 = load <8 x float>, ptr %.slot64, align 32
  %638 = select <8 x i1> %59, <8 x float> zeroinitializer, <8 x float> %637
  store <8 x float> %638, ptr %.slot64, align 32
  br label %direct.schedule.41

direct.true260:                                   ; preds = %direct.schedule.41
  br label %direct.schedule.42

direct.false261:                                  ; preds = %direct.schedule.41
  %639 = load <8 x float>, ptr %.slot67, align 32
  %640 = select <8 x i1> %59, <8 x float> zeroinitializer, <8 x float> %639
  store <8 x float> %640, ptr %.slot67, align 32
  br label %direct.schedule.43

direct.true283:                                   ; preds = %direct.schedule.45
  br label %direct.schedule.46

direct.false284:                                  ; preds = %direct.schedule.45
  br label %direct.schedule.47

direct.true297:                                   ; preds = %direct.schedule.48
  br label %direct.schedule.49

direct.false298:                                  ; preds = %direct.schedule.48
  br label %direct.schedule.50

direct.true310:                                   ; preds = %direct.schedule.51
  br label %direct.schedule.52

direct.false311:                                  ; preds = %direct.schedule.51
  br label %direct.schedule.53
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.vector.reduce.or.v8i1(<8 x i1>) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr>, i32 immarg, <8 x i1>, <8 x float>) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.masked.scatter.v8f32.v8p0(<8 x float>, <8 x ptr>, i32 immarg, <8 x i1>) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.sqrt.v8f32(<8 x float>) #0

define internal void @llm_rows.packet_batch(ptr noalias readonly %argument_buffer, ptr %return_lanes, ptr noalias nonnull %launch_config, i32 %packet_count) {
packet.batch.prologue:
  %thread.index.address = getelementptr inbounds i8, ptr %launch_config, i64 36
  %base.thread.index = load i32, ptr %thread.index.address, align 4
  %block.x.address = getelementptr inbounds i8, ptr %launch_config, i64 0
  %block.x = load i32, ptr %block.x.address, align 4
  %dispatch.size.x.address = getelementptr inbounds i8, ptr %launch_config, i64 12
  %dispatch.size.x = load i32, ptr %dispatch.size.x.address, align 4
  %base.thread.index.i64 = zext i32 %base.thread.index to i64
  %0 = zext i32 %block.x to i64
  %block.origin.x = mul i64 %0, 32
  %dispatch.size.x.i64 = zext i32 %dispatch.size.x to i64
  %block.origin.in.range = icmp ule i64 %block.origin.x, %dispatch.size.x.i64
  %block.origin.safe = select i1 %block.origin.in.range, i64 %block.origin.x, i64 0
  %packet.range.start = add i64 %block.origin.safe, %base.thread.index.i64
  %1 = icmp ult i64 %packet.range.start, %dispatch.size.x.i64
  %packet.range.inside.dispatch = and i1 %block.origin.in.range, %1
  %2 = sub i64 %dispatch.size.x.i64, %packet.range.start
  %dispatch.remaining = select i1 %packet.range.inside.dispatch, i64 %2, i64 0
  %packet.range.inside.block = icmp ult i64 %base.thread.index.i64, 32
  %3 = sub i64 32, %base.thread.index.i64
  %block.remaining = select i1 %packet.range.inside.block, i64 %3, i64 0
  %4 = icmp ult i64 %dispatch.remaining, %block.remaining
  %packet.range.remaining = select i1 %4, i64 %dispatch.remaining, i64 %block.remaining
  %5 = zext i32 %packet_count to i64
  %packet.range.requested.threads = mul i64 %5, 8
  %6 = icmp ult i64 %packet.range.remaining, %packet.range.requested.threads
  %packet.range.active.threads = select i1 %6, i64 %packet.range.remaining, i64 %packet.range.requested.threads
  %7 = icmp eq i32 %packet_count, 4
  %8 = icmp uge i64 %packet.range.remaining, 32
  %packet.range.complete.static = and i1 %7, %8
  br i1 %packet.range.complete.static, label %packet.batch.full, label %packet.batch.partial

packet.batch.full:                                ; preds = %packet.batch.prologue
  store i32 %base.thread.index, ptr %thread.index.address, align 4
  call void @llm_rows(ptr %argument_buffer, ptr %return_lanes, ptr %launch_config, i32 8)
  %packet.thread.index = add i32 %base.thread.index, 8
  store i32 %packet.thread.index, ptr %thread.index.address, align 4
  call void @llm_rows(ptr %argument_buffer, ptr %return_lanes, ptr %launch_config, i32 8)
  %packet.thread.index1 = add i32 %base.thread.index, 16
  store i32 %packet.thread.index1, ptr %thread.index.address, align 4
  call void @llm_rows(ptr %argument_buffer, ptr %return_lanes, ptr %launch_config, i32 8)
  %packet.thread.index2 = add i32 %base.thread.index, 24
  store i32 %packet.thread.index2, ptr %thread.index.address, align 4
  call void @llm_rows(ptr %argument_buffer, ptr %return_lanes, ptr %launch_config, i32 8)
  br label %packet.batch.exit

packet.batch.partial:                             ; preds = %packet.batch.prologue
  %packet.full.count.i64 = udiv i64 %packet.range.active.threads, 8
  %packet.full.count = trunc i64 %packet.full.count.i64 to i32
  %packet.tail.lane.count.i64 = urem i64 %packet.range.active.threads, 8
  %packet.tail.lane.count = trunc i64 %packet.tail.lane.count.i64 to i32
  %9 = icmp ne i32 %packet.full.count, 0
  br i1 %9, label %packet.batch.partial.full.loop, label %packet.batch.tail.check

packet.batch.exit:                                ; preds = %packet.batch.partial.finish, %packet.batch.full
  ret void

packet.batch.partial.full.loop:                   ; preds = %packet.batch.partial.full.loop, %packet.batch.partial
  %partial.packet.index = phi i32 [ 0, %packet.batch.partial ], [ %partial.packet.index.next, %packet.batch.partial.full.loop ]
  %partial.packet.thread.offset = mul i32 %partial.packet.index, 8
  %partial.packet.thread.index = add i32 %base.thread.index, %partial.packet.thread.offset
  store i32 %partial.packet.thread.index, ptr %thread.index.address, align 4
  call void @llm_rows(ptr %argument_buffer, ptr %return_lanes, ptr %launch_config, i32 8)
  %partial.packet.index.next = add i32 %partial.packet.index, 1
  %partial.packet.has.more.full = icmp ult i32 %partial.packet.index.next, %packet.full.count
  br i1 %partial.packet.has.more.full, label %packet.batch.partial.full.loop, label %packet.batch.tail.check

packet.batch.tail.check:                          ; preds = %packet.batch.partial.full.loop, %packet.batch.partial
  %10 = icmp ne i32 %packet.tail.lane.count, 0
  br i1 %10, label %packet.batch.tail.call, label %packet.batch.partial.finish

packet.batch.tail.call:                           ; preds = %packet.batch.tail.check
  %packet.tail.thread.offset = mul i32 %packet.full.count, 8
  %packet.tail.thread.index = add i32 %base.thread.index, %packet.tail.thread.offset
  store i32 %packet.tail.thread.index, ptr %thread.index.address, align 4
  call void @llm_rows(ptr %argument_buffer, ptr %return_lanes, ptr %launch_config, i32 %packet.tail.lane.count)
  br label %packet.batch.partial.finish

packet.batch.partial.finish:                      ; preds = %packet.batch.tail.call, %packet.batch.tail.check
  %packet.batch.has.requested = icmp ne i32 %packet_count, 0
  %packet.batch.last.requested = sub i32 %packet_count, 1
  %packet.batch.last.requested.offset = mul i32 %packet.batch.last.requested, 8
  %packet.batch.last.requested.index = add i32 %base.thread.index, %packet.batch.last.requested.offset
  %11 = select i1 %packet.batch.has.requested, i32 %packet.batch.last.requested.index, i32 %base.thread.index
  store i32 %11, ptr %thread.index.address, align 4
  br label %packet.batch.exit
}

define dso_local void @llm_rows.packet_batch.blocks(ptr noalias readonly %argument_buffer, ptr %return_lanes, ptr noalias nonnull %launch_config, i32 %block_count) {
block.batch.prologue:
  %block.x.address = getelementptr inbounds i8, ptr %launch_config, i64 0
  %block.y.address = getelementptr inbounds i8, ptr %launch_config, i64 4
  %block.z.address = getelementptr inbounds i8, ptr %launch_config, i64 8
  %thread.index.address = getelementptr inbounds i8, ptr %launch_config, i64 36
  %grid.x.address = getelementptr inbounds i8, ptr %launch_config, i64 44
  %grid.y.address = getelementptr inbounds i8, ptr %launch_config, i64 48
  %initial.block.x = load i32, ptr %block.x.address, align 4
  %initial.block.y = load i32, ptr %block.y.address, align 4
  %initial.block.z = load i32, ptr %block.z.address, align 4
  %grid.x = load i32, ptr %grid.x.address, align 4
  %grid.y = load i32, ptr %grid.y.address, align 4
  %block.batch.empty = icmp eq i32 %block_count, 0
  br i1 %block.batch.empty, label %block.batch.exit, label %block.batch.loop

block.batch.loop:                                 ; preds = %block.batch.loop, %block.batch.prologue
  %block.index = phi i32 [ 0, %block.batch.prologue ], [ %block.index.next, %block.batch.loop ]
  %block.x = phi i32 [ %initial.block.x, %block.batch.prologue ], [ %block.x.next, %block.batch.loop ]
  %block.y = phi i32 [ %initial.block.y, %block.batch.prologue ], [ %block.y.next, %block.batch.loop ]
  %block.z = phi i32 [ %initial.block.z, %block.batch.prologue ], [ %block.z.next, %block.batch.loop ]
  store i32 %block.x, ptr %block.x.address, align 4
  store i32 %block.y, ptr %block.y.address, align 4
  store i32 %block.z, ptr %block.z.address, align 4
  store i32 0, ptr %thread.index.address, align 4
  call void @llm_rows.packet_batch(ptr %argument_buffer, ptr %return_lanes, ptr %launch_config, i32 4)
  %block.x.incremented = add i32 %block.x, 1
  %block.x.wrap = icmp eq i32 %block.x.incremented, %grid.x
  %block.x.next = select i1 %block.x.wrap, i32 0, i32 %block.x.incremented
  %block.y.increment = zext i1 %block.x.wrap to i32
  %block.y.incremented = add i32 %block.y, %block.y.increment
  %block.y.at.end = icmp eq i32 %block.y.incremented, %grid.y
  %block.y.wrap = and i1 %block.x.wrap, %block.y.at.end
  %block.y.next = select i1 %block.y.wrap, i32 0, i32 %block.y.incremented
  %block.z.increment = zext i1 %block.y.wrap to i32
  %block.z.next = add i32 %block.z, %block.z.increment
  %block.index.next = add i32 %block.index, 1
  %block.batch.has.more = icmp ult i32 %block.index.next, %block_count
  br i1 %block.batch.has.more, label %block.batch.loop, label %block.batch.exit

block.batch.exit:                                 ; preds = %block.batch.loop, %block.batch.prologue
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(read) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(write) }
