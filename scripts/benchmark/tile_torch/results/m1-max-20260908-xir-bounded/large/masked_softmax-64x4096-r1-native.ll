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
  %2 = insertvalue { <8 x ptr>, <8 x i64> } %1, <8 x i64> <i64 0, i64 16384, i64 32768, i64 49152, i64 65536, i64 81920, i64 98304, i64 114688>, 1
  %3 = getelementptr i8, ptr %launch_config, i64 136
  %private.workspace1 = load ptr, ptr %3, align 8
  %tile_snapshot.private2 = getelementptr inbounds i8, ptr %private.workspace1, i64 131072
  %.splatinsert3 = insertelement <8 x ptr> poison, ptr %tile_snapshot.private2, i64 0
  %.splat4 = shufflevector <8 x ptr> %.splatinsert3, <8 x ptr> poison, <8 x i32> zeroinitializer
  %4 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat4, 0
  %5 = insertvalue { <8 x ptr>, <8 x i64> } %4, <8 x i64> <i64 0, i64 32768, i64 65536, i64 98304, i64 131072, i64 163840, i64 196608, i64 229376>, 1
  %6 = getelementptr i8, ptr %launch_config, i64 136
  %private.workspace5 = load ptr, ptr %6, align 8
  %tile_snapshot.private6 = getelementptr inbounds i8, ptr %private.workspace5, i64 393216
  %.splatinsert7 = insertelement <8 x ptr> poison, ptr %tile_snapshot.private6, i64 0
  %.splat8 = shufflevector <8 x ptr> %.splatinsert7, <8 x ptr> poison, <8 x i32> zeroinitializer
  %7 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat8, 0
  %8 = insertvalue { <8 x ptr>, <8 x i64> } %7, <8 x i64> <i64 0, i64 4096, i64 8192, i64 12288, i64 16384, i64 20480, i64 24576, i64 28672>, 1
  %9 = getelementptr i8, ptr %launch_config, i64 136
  %private.workspace9 = load ptr, ptr %9, align 8
  %tile_snapshot.private10 = getelementptr inbounds i8, ptr %private.workspace9, i64 425984
  %.splatinsert11 = insertelement <8 x ptr> poison, ptr %tile_snapshot.private10, i64 0
  %.splat12 = shufflevector <8 x ptr> %.splatinsert11, <8 x ptr> poison, <8 x i32> zeroinitializer
  %10 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat12, 0
  %11 = insertvalue { <8 x ptr>, <8 x i64> } %10, <8 x i64> <i64 0, i64 16384, i64 32768, i64 49152, i64 65536, i64 81920, i64 98304, i64 114688>, 1
  %12 = getelementptr i8, ptr %launch_config, i64 136
  %private.workspace13 = load ptr, ptr %12, align 8
  %tile_snapshot.private14 = getelementptr inbounds i8, ptr %private.workspace13, i64 557056
  %.splatinsert15 = insertelement <8 x ptr> poison, ptr %tile_snapshot.private14, i64 0
  %.splat16 = shufflevector <8 x ptr> %.splatinsert15, <8 x ptr> poison, <8 x i32> zeroinitializer
  %13 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat16, 0
  %14 = insertvalue { <8 x ptr>, <8 x i64> } %13, <8 x i64> <i64 0, i64 16384, i64 32768, i64 49152, i64 65536, i64 81920, i64 98304, i64 114688>, 1
  %.spill = alloca <8 x i64>, align 64
  store <8 x i64> zeroinitializer, ptr %.spill, align 64
  %tile_snapshot.spill = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill, align 64
  %.slot = alloca i64, align 8
  store i64 0, ptr %.slot, align 4
  %tile_snapshot.spill17 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill17, align 64
  %.slot18 = alloca i64, align 8
  store i64 0, ptr %.slot18, align 4
  %.spill19 = alloca <8 x i64>, align 64
  store <8 x i64> zeroinitializer, ptr %.spill19, align 64
  %tile_snapshot.spill20 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill20, align 64
  %.slot21 = alloca i64, align 8
  store i64 0, ptr %.slot21, align 4
  %.spill22 = alloca float, align 4
  store float 0.000000e+00, ptr %.spill22, align 4
  %tile_snapshot.spill23 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill23, align 64
  %.slot24 = alloca i64, align 8
  store i64 0, ptr %.slot24, align 4
  %.spill25 = alloca float, align 4
  store float 0.000000e+00, ptr %.spill25, align 4
  %.spill26 = alloca i64, align 8
  store i64 0, ptr %.spill26, align 4
  %.spill27 = alloca i64, align 8
  store i64 0, ptr %.spill27, align 4
  %.spill28 = alloca i1, align 1
  store i1 false, ptr %.spill28, align 1
  %.slot29 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot29, align 32
  %.spill30 = alloca i64, align 8
  store i64 0, ptr %.spill30, align 4
  %.spill31 = alloca i1, align 1
  store i1 false, ptr %.spill31, align 1
  %.slot32 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot32, align 32
  %.spill33 = alloca i64, align 8
  store i64 0, ptr %.spill33, align 4
  %.spill34 = alloca i1, align 1
  store i1 false, ptr %.spill34, align 1
  %.slot35 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot35, align 32
  %.spill36 = alloca i64, align 8
  store i64 0, ptr %.spill36, align 4
  %.spill37 = alloca i1, align 1
  store i1 false, ptr %.spill37, align 1
  %.slot38 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot38, align 32
  %.slot39 = alloca i64, align 8
  store i64 0, ptr %.slot39, align 4
  %.slot40 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot40, align 32
  %.slot41 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot41, align 32
  %.slot42 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot42, align 32
  %.slot43 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot43, align 32
  %.spill44 = alloca i64, align 8
  store i64 0, ptr %.spill44, align 4
  %.slot45 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot45, align 32
  %.spill46 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill46, align 32
  %.spill47 = alloca i64, align 8
  store i64 0, ptr %.spill47, align 4
  %.slot48 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot48, align 32
  %.spill49 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill49, align 32
  %.spill50 = alloca i64, align 8
  store i64 0, ptr %.spill50, align 4
  %.slot51 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot51, align 32
  %.spill52 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill52, align 32
  %.spill53 = alloca i64, align 8
  store i64 0, ptr %.spill53, align 4
  %.slot54 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot54, align 32
  %.spill55 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill55, align 32
  %.spill56 = alloca float, align 4
  store float 0.000000e+00, ptr %.spill56, align 4
  %tile_snapshot.spill57 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill57, align 64
  %.slot58 = alloca i64, align 8
  store i64 0, ptr %.slot58, align 4
  %.slot59 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot59, align 32
  %.slot60 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot60, align 32
  %.slot61 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot61, align 32
  %.slot62 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot62, align 32
  %.slot63 = alloca i64, align 8
  store i64 0, ptr %.slot63, align 4
  %.slot64 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot64, align 32
  %.slot65 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot65, align 32
  %.slot66 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot66, align 32
  %.slot67 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot67, align 32
  %.spill68 = alloca i64, align 8
  store i64 0, ptr %.spill68, align 4
  %.slot69 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot69, align 32
  %.spill70 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill70, align 32
  %.spill71 = alloca i64, align 8
  store i64 0, ptr %.spill71, align 4
  %.slot72 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot72, align 32
  %.spill73 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill73, align 32
  %.spill74 = alloca i64, align 8
  store i64 0, ptr %.spill74, align 4
  %.slot75 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot75, align 32
  %.spill76 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill76, align 32
  %.spill77 = alloca i64, align 8
  store i64 0, ptr %.spill77, align 4
  %.slot78 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot78, align 32
  %.spill79 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill79, align 32
  %.slot80 = alloca i64, align 8
  store i64 0, ptr %.slot80, align 4
  %15 = getelementptr i8, ptr %argument_buffer, i64 0
  %16 = load ptr, ptr %15, align 16
  %17 = getelementptr i8, ptr %15, i64 8
  %18 = load i64, ptr %17, align 8
  %19 = insertvalue { ptr, i64 } poison, ptr %16, 0
  %20 = insertvalue { ptr, i64 } %19, i64 %18, 1
  %21 = getelementptr i8, ptr %argument_buffer, i64 16
  %22 = load ptr, ptr %21, align 16
  %23 = getelementptr i8, ptr %21, i64 8
  %24 = load i64, ptr %23, align 8
  %25 = insertvalue { ptr, i64 } poison, ptr %22, 0
  %26 = insertvalue { ptr, i64 } %25, i64 %24, 1
  %27 = getelementptr i8, ptr %argument_buffer, i64 32
  %28 = load ptr, ptr %27, align 16
  %29 = getelementptr i8, ptr %27, i64 8
  %30 = load i64, ptr %29, align 8
  %31 = insertvalue { ptr, i64 } poison, ptr %28, 0
  %32 = insertvalue { ptr, i64 } %31, i64 %30, 1
  %33 = getelementptr i8, ptr %argument_buffer, i64 48
  %34 = load ptr, ptr %33, align 16
  %35 = getelementptr i8, ptr %33, i64 8
  %36 = load i64, ptr %35, align 8
  %37 = insertvalue { ptr, i64 } poison, ptr %34, 0
  %38 = insertvalue { ptr, i64 } %37, i64 %36, 1
  %39 = load i32, ptr %launch_config, align 4
  %40 = getelementptr i8, ptr %launch_config, i64 12
  %41 = load i32, ptr %40, align 4
  %42 = getelementptr i8, ptr %launch_config, i64 4
  %43 = load i32, ptr %42, align 4
  %44 = getelementptr i8, ptr %launch_config, i64 16
  %45 = load i32, ptr %44, align 4
  %46 = getelementptr i8, ptr %launch_config, i64 8
  %47 = load i32, ptr %46, align 4
  %48 = getelementptr i8, ptr %launch_config, i64 20
  %49 = load i32, ptr %48, align 4
  %50 = getelementptr i8, ptr %launch_config, i64 36
  %51 = load i32, ptr %50, align 4
  %.splatinsert81 = insertelement <8 x i32> poison, i32 %51, i64 0
  %.splat82 = shufflevector <8 x i32> %.splatinsert81, <8 x i32> poison, <8 x i32> zeroinitializer
  %52 = add <8 x i32> %.splat82, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %53 = mul i32 %39, 32
  %.splatinsert83 = insertelement <8 x i32> poison, i32 %53, i64 0
  %.splat84 = shufflevector <8 x i32> %.splatinsert83, <8 x i32> poison, <8 x i32> zeroinitializer
  %54 = add <8 x i32> %.splat84, %52
  %55 = mul i32 %43, 1
  %.splatinsert85 = insertelement <8 x i32> poison, i32 %55, i64 0
  %.splat86 = shufflevector <8 x i32> %.splatinsert85, <8 x i32> poison, <8 x i32> zeroinitializer
  %56 = add <8 x i32> %.splat86, zeroinitializer
  %57 = mul i32 %47, 1
  %.splatinsert87 = insertelement <8 x i32> poison, i32 %57, i64 0
  %.splat88 = shufflevector <8 x i32> %.splatinsert87, <8 x i32> poison, <8 x i32> zeroinitializer
  %58 = add <8 x i32> %.splat88, zeroinitializer
  %59 = insertvalue [3 x <8 x i32>] poison, <8 x i32> %54, 0
  %60 = insertvalue [3 x <8 x i32>] %59, <8 x i32> %56, 1
  %61 = insertvalue [3 x <8 x i32>] %60, <8 x i32> %58, 2
  %.splatinsert89 = insertelement <8 x i32> poison, i32 %active_lane_count, i64 0
  %.splat90 = shufflevector <8 x i32> %.splatinsert89, <8 x i32> poison, <8 x i32> zeroinitializer
  %62 = icmp ult <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, %.splat90
  %63 = call i1 @llvm.vector.reduce.or.v8i1(<8 x i1> %62)
  br i1 %63, label %direct.activate, label %direct.inactive

direct.schedule.0:                                ; preds = %direct.activate
  %64 = extractvalue [3 x <8 x i32>] %61, 0
  %65 = zext <8 x i32> %64 to <8 x i64>
  %66 = select <8 x i1> %62, <8 x i64> %65, <8 x i64> zeroinitializer
  %67 = select <8 x i1> %62, <8 x i64> splat (i64 1), <8 x i64> splat (i64 1)
  %68 = sdiv <8 x i64> %66, %67
  %69 = select <8 x i1> %62, <8 x i64> %68, <8 x i64> zeroinitializer
  %70 = select <8 x i1> %62, <8 x i64> splat (i64 64), <8 x i64> splat (i64 1)
  %71 = srem <8 x i64> %69, %70
  %72 = load <8 x i64>, ptr %.spill, align 64
  %73 = select <8 x i1> %62, <8 x i64> %71, <8 x i64> %72
  store <8 x i64> %73, ptr %.spill, align 64
  %74 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %75 = extractvalue { <8 x ptr>, <8 x i64> } %2, 0
  %76 = extractvalue { <8 x ptr>, <8 x i64> } %74, 0
  %77 = select <8 x i1> %62, <8 x ptr> %75, <8 x ptr> %76
  %78 = extractvalue { <8 x ptr>, <8 x i64> } %2, 1
  %79 = extractvalue { <8 x ptr>, <8 x i64> } %74, 1
  %80 = select <8 x i1> %62, <8 x i64> %78, <8 x i64> %79
  %81 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %77, 0
  %82 = insertvalue { <8 x ptr>, <8 x i64> } %81, <8 x i64> %80, 1
  store { <8 x ptr>, <8 x i64> } %82, ptr %tile_snapshot.spill, align 64
  store i64 0, ptr %.slot, align 4
  br label %direct.schedule.1

direct.schedule.1:                                ; preds = %direct.schedule.2, %direct.schedule.0
  %.state = load i64, ptr %.slot, align 4
  %83 = icmp slt i64 %.state, 4096
  br i1 %83, label %direct.true, label %direct.false

direct.schedule.2:                                ; preds = %direct.true
  %.state91 = load i64, ptr %.slot, align 4
  %84 = srem i64 %.state91, 4096
  %.state92 = load i64, ptr %.slot, align 4
  %85 = sdiv i64 %.state92, 4096
  %86 = srem i64 %85, 1
  %.spill.load = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert93 = insertelement <8 x i64> poison, i64 %86, i64 0
  %.splat94 = shufflevector <8 x i64> %.splatinsert93, <8 x i64> poison, <8 x i32> zeroinitializer
  %87 = add <8 x i64> %.spill.load, %.splat94
  %88 = add <8 x i64> zeroinitializer, %87
  %89 = add i64 0, %84
  %90 = mul <8 x i64> %88, splat (i64 4096)
  %.splatinsert95 = insertelement <8 x i64> poison, i64 %89, i64 0
  %.splat96 = shufflevector <8 x i64> %.splatinsert95, <8 x i64> poison, <8 x i32> zeroinitializer
  %91 = add <8 x i64> %90, %.splat96
  %92 = extractvalue { ptr, i64 } %20, 0
  %93 = mul <8 x i64> %91, splat (i64 4)
  %94 = getelementptr i8, ptr %92, <8 x i64> %93
  %95 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %94, i32 1, <8 x i1> %62, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %96 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 0
  %97 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 1
  %.state97 = load i64, ptr %.slot, align 4
  %.splatinsert98 = insertelement <8 x i64> poison, i64 %.state97, i64 0
  %.splat99 = shufflevector <8 x i64> %.splatinsert98, <8 x i64> poison, <8 x i32> zeroinitializer
  %98 = mul <8 x i64> %.splat99, splat (i64 4)
  %99 = add <8 x i64> %97, %98
  %100 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %96, 0
  %101 = insertvalue { <8 x ptr>, <8 x i64> } %100, <8 x i64> %99, 1
  %102 = extractvalue { <8 x ptr>, <8 x i64> } %101, 0
  %103 = extractvalue { <8 x ptr>, <8 x i64> } %101, 1
  %104 = getelementptr i8, <8 x ptr> %102, <8 x i64> %103
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %95, <8 x ptr> %104, i32 1, <8 x i1> %62)
  %.state100 = load i64, ptr %.slot, align 4
  %105 = add i64 %.state100, 1
  store i64 %105, ptr %.slot, align 4
  br label %direct.schedule.1

direct.schedule.3:                                ; preds = %direct.false
  %106 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill17, align 64
  %107 = extractvalue { <8 x ptr>, <8 x i64> } %5, 0
  %108 = extractvalue { <8 x ptr>, <8 x i64> } %106, 0
  %109 = select <8 x i1> %62, <8 x ptr> %107, <8 x ptr> %108
  %110 = extractvalue { <8 x ptr>, <8 x i64> } %5, 1
  %111 = extractvalue { <8 x ptr>, <8 x i64> } %106, 1
  %112 = select <8 x i1> %62, <8 x i64> %110, <8 x i64> %111
  %113 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %109, 0
  %114 = insertvalue { <8 x ptr>, <8 x i64> } %113, <8 x i64> %112, 1
  store { <8 x ptr>, <8 x i64> } %114, ptr %tile_snapshot.spill17, align 64
  store i64 0, ptr %.slot18, align 4
  br label %direct.schedule.4

direct.schedule.4:                                ; preds = %direct.schedule.5, %direct.schedule.3
  %.state101 = load i64, ptr %.slot18, align 4
  %115 = icmp slt i64 %.state101, 4096
  br i1 %115, label %direct.true102, label %direct.false103

direct.schedule.5:                                ; preds = %direct.true102
  %.state104 = load i64, ptr %.slot18, align 4
  %116 = srem i64 %.state104, 4096
  %tile_snapshot.spill.load105 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill17, align 64
  %117 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load105, 0
  %118 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load105, 1
  %.state106 = load i64, ptr %.slot18, align 4
  %.splatinsert107 = insertelement <8 x i64> poison, i64 %.state106, i64 0
  %.splat108 = shufflevector <8 x i64> %.splatinsert107, <8 x i64> poison, <8 x i32> zeroinitializer
  %119 = mul <8 x i64> %.splat108, splat (i64 8)
  %120 = add <8 x i64> %118, %119
  %121 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %117, 0
  %122 = insertvalue { <8 x ptr>, <8 x i64> } %121, <8 x i64> %120, 1
  %.splatinsert109 = insertelement <8 x i64> poison, i64 %116, i64 0
  %.splat110 = shufflevector <8 x i64> %.splatinsert109, <8 x i64> poison, <8 x i32> zeroinitializer
  %123 = extractvalue { <8 x ptr>, <8 x i64> } %122, 0
  %124 = extractvalue { <8 x ptr>, <8 x i64> } %122, 1
  %125 = getelementptr i8, <8 x ptr> %123, <8 x i64> %124
  call void @llvm.masked.scatter.v8i64.v8p0(<8 x i64> %.splat110, <8 x ptr> %125, i32 1, <8 x i1> %62)
  %.state111 = load i64, ptr %.slot18, align 4
  %126 = add i64 %.state111, 1
  store i64 %126, ptr %.slot18, align 4
  br label %direct.schedule.4

direct.schedule.6:                                ; preds = %direct.false103
  %.spill.load112 = load <8 x i64>, ptr %.spill, align 64
  %127 = select <8 x i1> %62, <8 x i64> %.spill.load112, <8 x i64> zeroinitializer
  %128 = select <8 x i1> %62, <8 x i64> splat (i64 4096), <8 x i64> splat (i64 1)
  %129 = srem <8 x i64> %127, %128
  %130 = load <8 x i64>, ptr %.spill19, align 64
  %131 = select <8 x i1> %62, <8 x i64> %129, <8 x i64> %130
  store <8 x i64> %131, ptr %.spill19, align 64
  %132 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill20, align 64
  %133 = extractvalue { <8 x ptr>, <8 x i64> } %8, 0
  %134 = extractvalue { <8 x ptr>, <8 x i64> } %132, 0
  %135 = select <8 x i1> %62, <8 x ptr> %133, <8 x ptr> %134
  %136 = extractvalue { <8 x ptr>, <8 x i64> } %8, 1
  %137 = extractvalue { <8 x ptr>, <8 x i64> } %132, 1
  %138 = select <8 x i1> %62, <8 x i64> %136, <8 x i64> %137
  %139 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %135, 0
  %140 = insertvalue { <8 x ptr>, <8 x i64> } %139, <8 x i64> %138, 1
  store { <8 x ptr>, <8 x i64> } %140, ptr %tile_snapshot.spill20, align 64
  store i64 0, ptr %.slot21, align 4
  br label %direct.schedule.7

direct.schedule.7:                                ; preds = %direct.schedule.8, %direct.schedule.6
  %.state113 = load i64, ptr %.slot21, align 4
  %141 = icmp slt i64 %.state113, 4096
  br i1 %141, label %direct.true114, label %direct.false115

direct.schedule.8:                                ; preds = %direct.true114
  %.state116 = load i64, ptr %.slot21, align 4
  %142 = srem i64 %.state116, 4096
  %143 = add i64 0, %142
  %tile_snapshot.spill.load117 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill17, align 64
  %144 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load117, 0
  %145 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load117, 1
  %.splatinsert118 = insertelement <8 x i64> poison, i64 %143, i64 0
  %.splat119 = shufflevector <8 x i64> %.splatinsert118, <8 x i64> poison, <8 x i32> zeroinitializer
  %146 = mul <8 x i64> %.splat119, splat (i64 8)
  %147 = add <8 x i64> %145, %146
  %148 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %144, 0
  %149 = insertvalue { <8 x ptr>, <8 x i64> } %148, <8 x i64> %147, 1
  %150 = extractvalue { <8 x ptr>, <8 x i64> } %149, 0
  %151 = extractvalue { <8 x ptr>, <8 x i64> } %149, 1
  %152 = getelementptr i8, <8 x ptr> %150, <8 x i64> %151
  %153 = call <8 x i64> @llvm.masked.gather.v8i64.v8p0(<8 x ptr> %152, i32 1, <8 x i1> %62, <8 x i64> zeroinitializer)
  %.spill.load120 = load <8 x i64>, ptr %.spill19, align 64
  %154 = icmp sle <8 x i64> %153, %.spill.load120
  %tile_snapshot.spill.load121 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill20, align 64
  %155 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load121, 0
  %156 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load121, 1
  %.state122 = load i64, ptr %.slot21, align 4
  %.splatinsert123 = insertelement <8 x i64> poison, i64 %.state122, i64 0
  %.splat124 = shufflevector <8 x i64> %.splatinsert123, <8 x i64> poison, <8 x i32> zeroinitializer
  %157 = add <8 x i64> %156, %.splat124
  %158 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %155, 0
  %159 = insertvalue { <8 x ptr>, <8 x i64> } %158, <8 x i64> %157, 1
  %160 = extractvalue { <8 x ptr>, <8 x i64> } %159, 0
  %161 = extractvalue { <8 x ptr>, <8 x i64> } %159, 1
  %162 = getelementptr i8, <8 x ptr> %160, <8 x i64> %161
  %163 = zext <8 x i1> %154 to <8 x i8>
  call void @llvm.masked.scatter.v8i8.v8p0(<8 x i8> %163, <8 x ptr> %162, i32 1, <8 x i1> %62)
  %.state125 = load i64, ptr %.slot21, align 4
  %164 = add i64 %.state125, 1
  store i64 %164, ptr %.slot21, align 4
  br label %direct.schedule.7

direct.schedule.9:                                ; preds = %direct.false115
  store float 0xC6293E5940000000, ptr %.spill22, align 4
  %165 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill23, align 64
  %166 = extractvalue { <8 x ptr>, <8 x i64> } %11, 0
  %167 = extractvalue { <8 x ptr>, <8 x i64> } %165, 0
  %168 = select <8 x i1> %62, <8 x ptr> %166, <8 x ptr> %167
  %169 = extractvalue { <8 x ptr>, <8 x i64> } %11, 1
  %170 = extractvalue { <8 x ptr>, <8 x i64> } %165, 1
  %171 = select <8 x i1> %62, <8 x i64> %169, <8 x i64> %170
  %172 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %168, 0
  %173 = insertvalue { <8 x ptr>, <8 x i64> } %172, <8 x i64> %171, 1
  store { <8 x ptr>, <8 x i64> } %173, ptr %tile_snapshot.spill23, align 64
  store i64 0, ptr %.slot24, align 4
  br label %direct.schedule.10

direct.schedule.10:                               ; preds = %direct.schedule.11, %direct.schedule.9
  %.state126 = load i64, ptr %.slot24, align 4
  %174 = icmp slt i64 %.state126, 4096
  br i1 %174, label %direct.true127, label %direct.false128

direct.schedule.11:                               ; preds = %direct.true127
  %.state129 = load i64, ptr %.slot24, align 4
  %175 = srem i64 %.state129, 4096
  %176 = add i64 0, %175
  %tile_snapshot.spill.load130 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill20, align 64
  %177 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load130, 0
  %178 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load130, 1
  %.splatinsert131 = insertelement <8 x i64> poison, i64 %176, i64 0
  %.splat132 = shufflevector <8 x i64> %.splatinsert131, <8 x i64> poison, <8 x i32> zeroinitializer
  %179 = add <8 x i64> %178, %.splat132
  %180 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %177, 0
  %181 = insertvalue { <8 x ptr>, <8 x i64> } %180, <8 x i64> %179, 1
  %182 = extractvalue { <8 x ptr>, <8 x i64> } %181, 0
  %183 = extractvalue { <8 x ptr>, <8 x i64> } %181, 1
  %184 = getelementptr i8, <8 x ptr> %182, <8 x i64> %183
  %185 = call <8 x i8> @llvm.masked.gather.v8i8.v8p0(<8 x ptr> %184, i32 1, <8 x i1> %62, <8 x i8> zeroinitializer)
  %186 = icmp ne <8 x i8> %185, zeroinitializer
  %tile_snapshot.spill.load133 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %187 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load133, 0
  %188 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load133, 1
  %.splatinsert134 = insertelement <8 x i64> poison, i64 %176, i64 0
  %.splat135 = shufflevector <8 x i64> %.splatinsert134, <8 x i64> poison, <8 x i32> zeroinitializer
  %189 = mul <8 x i64> %.splat135, splat (i64 4)
  %190 = add <8 x i64> %188, %189
  %191 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %187, 0
  %192 = insertvalue { <8 x ptr>, <8 x i64> } %191, <8 x i64> %190, 1
  %193 = extractvalue { <8 x ptr>, <8 x i64> } %192, 0
  %194 = extractvalue { <8 x ptr>, <8 x i64> } %192, 1
  %195 = getelementptr i8, <8 x ptr> %193, <8 x i64> %194
  %196 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %195, i32 1, <8 x i1> %62, <8 x float> zeroinitializer)
  %.spill.load136 = load float, ptr %.spill22, align 4
  %.splatinsert137 = insertelement <8 x float> poison, float %.spill.load136, i64 0
  %.splat138 = shufflevector <8 x float> %.splatinsert137, <8 x float> poison, <8 x i32> zeroinitializer
  %197 = select <8 x i1> %186, <8 x float> %196, <8 x float> %.splat138
  %tile_snapshot.spill.load139 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill23, align 64
  %198 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load139, 0
  %199 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load139, 1
  %.state140 = load i64, ptr %.slot24, align 4
  %.splatinsert141 = insertelement <8 x i64> poison, i64 %.state140, i64 0
  %.splat142 = shufflevector <8 x i64> %.splatinsert141, <8 x i64> poison, <8 x i32> zeroinitializer
  %200 = mul <8 x i64> %.splat142, splat (i64 4)
  %201 = add <8 x i64> %199, %200
  %202 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %198, 0
  %203 = insertvalue { <8 x ptr>, <8 x i64> } %202, <8 x i64> %201, 1
  %204 = extractvalue { <8 x ptr>, <8 x i64> } %203, 0
  %205 = extractvalue { <8 x ptr>, <8 x i64> } %203, 1
  %206 = getelementptr i8, <8 x ptr> %204, <8 x i64> %205
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %197, <8 x ptr> %206, i32 1, <8 x i1> %62)
  %.state143 = load i64, ptr %.slot24, align 4
  %207 = add i64 %.state143, 1
  store i64 %207, ptr %.slot24, align 4
  br label %direct.schedule.10

direct.schedule.12:                               ; preds = %direct.false128
  store float 0xFFF0000000000000, ptr %.spill25, align 4
  store i64 0, ptr %.spill26, align 4
  store i64 0, ptr %.spill27, align 4
  store i1 true, ptr %.spill28, align 1
  br i1 true, label %direct.true144, label %direct.false145

direct.schedule.13:                               ; preds = %direct.true144
  %tile_snapshot.spill.load146 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill23, align 64
  %208 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load146, 0
  %209 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load146, 1
  %.spill.load147 = load i64, ptr %.spill27, align 4
  %.splatinsert148 = insertelement <8 x i64> poison, i64 %.spill.load147, i64 0
  %.splat149 = shufflevector <8 x i64> %.splatinsert148, <8 x i64> poison, <8 x i32> zeroinitializer
  %210 = mul <8 x i64> %.splat149, splat (i64 4)
  %211 = add <8 x i64> %209, %210
  %212 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %208, 0
  %213 = insertvalue { <8 x ptr>, <8 x i64> } %212, <8 x i64> %211, 1
  %214 = extractvalue { <8 x ptr>, <8 x i64> } %213, 0
  %215 = extractvalue { <8 x ptr>, <8 x i64> } %213, 1
  %216 = getelementptr i8, <8 x ptr> %214, <8 x i64> %215
  %217 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %216, i32 1, <8 x i1> %62, <8 x float> zeroinitializer)
  %218 = load <8 x float>, ptr %.slot29, align 32
  %219 = select <8 x i1> %62, <8 x float> %217, <8 x float> %218
  store <8 x float> %219, ptr %.slot29, align 32
  br label %direct.schedule.14

direct.schedule.14:                               ; preds = %direct.schedule.13, %direct.false145
  %.spill.load150 = load i64, ptr %.spill26, align 4
  %220 = add i64 %.spill.load150, 1
  store i64 %220, ptr %.spill30, align 4
  %221 = icmp sge i64 %220, 0
  %222 = icmp slt i64 %220, 4096
  %223 = and i1 %221, %222
  store i1 %223, ptr %.spill31, align 1
  br i1 %223, label %direct.true151, label %direct.false152

direct.schedule.15:                               ; preds = %direct.true151
  %tile_snapshot.spill.load153 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill23, align 64
  %224 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load153, 0
  %225 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load153, 1
  %.spill.load154 = load i64, ptr %.spill30, align 4
  %.splatinsert155 = insertelement <8 x i64> poison, i64 %.spill.load154, i64 0
  %.splat156 = shufflevector <8 x i64> %.splatinsert155, <8 x i64> poison, <8 x i32> zeroinitializer
  %226 = mul <8 x i64> %.splat156, splat (i64 4)
  %227 = add <8 x i64> %225, %226
  %228 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %224, 0
  %229 = insertvalue { <8 x ptr>, <8 x i64> } %228, <8 x i64> %227, 1
  %230 = extractvalue { <8 x ptr>, <8 x i64> } %229, 0
  %231 = extractvalue { <8 x ptr>, <8 x i64> } %229, 1
  %232 = getelementptr i8, <8 x ptr> %230, <8 x i64> %231
  %233 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %232, i32 1, <8 x i1> %62, <8 x float> zeroinitializer)
  %234 = load <8 x float>, ptr %.slot32, align 32
  %235 = select <8 x i1> %62, <8 x float> %233, <8 x float> %234
  store <8 x float> %235, ptr %.slot32, align 32
  br label %direct.schedule.16

direct.schedule.16:                               ; preds = %direct.schedule.15, %direct.false152
  %.spill.load157 = load i64, ptr %.spill26, align 4
  %236 = add i64 %.spill.load157, 2
  store i64 %236, ptr %.spill33, align 4
  %237 = icmp sge i64 %236, 0
  %238 = icmp slt i64 %236, 4096
  %239 = and i1 %237, %238
  store i1 %239, ptr %.spill34, align 1
  br i1 %239, label %direct.true158, label %direct.false159

direct.schedule.17:                               ; preds = %direct.true158
  %tile_snapshot.spill.load160 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill23, align 64
  %240 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load160, 0
  %241 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load160, 1
  %.spill.load161 = load i64, ptr %.spill33, align 4
  %.splatinsert162 = insertelement <8 x i64> poison, i64 %.spill.load161, i64 0
  %.splat163 = shufflevector <8 x i64> %.splatinsert162, <8 x i64> poison, <8 x i32> zeroinitializer
  %242 = mul <8 x i64> %.splat163, splat (i64 4)
  %243 = add <8 x i64> %241, %242
  %244 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %240, 0
  %245 = insertvalue { <8 x ptr>, <8 x i64> } %244, <8 x i64> %243, 1
  %246 = extractvalue { <8 x ptr>, <8 x i64> } %245, 0
  %247 = extractvalue { <8 x ptr>, <8 x i64> } %245, 1
  %248 = getelementptr i8, <8 x ptr> %246, <8 x i64> %247
  %249 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %248, i32 1, <8 x i1> %62, <8 x float> zeroinitializer)
  %250 = load <8 x float>, ptr %.slot35, align 32
  %251 = select <8 x i1> %62, <8 x float> %249, <8 x float> %250
  store <8 x float> %251, ptr %.slot35, align 32
  br label %direct.schedule.18

direct.schedule.18:                               ; preds = %direct.schedule.17, %direct.false159
  %.spill.load164 = load i64, ptr %.spill26, align 4
  %252 = add i64 %.spill.load164, 3
  store i64 %252, ptr %.spill36, align 4
  %253 = icmp sge i64 %252, 0
  %254 = icmp slt i64 %252, 4096
  %255 = and i1 %253, %254
  store i1 %255, ptr %.spill37, align 1
  br i1 %255, label %direct.true165, label %direct.false166

direct.schedule.19:                               ; preds = %direct.true165
  %tile_snapshot.spill.load167 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill23, align 64
  %256 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load167, 0
  %257 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load167, 1
  %.spill.load168 = load i64, ptr %.spill36, align 4
  %.splatinsert169 = insertelement <8 x i64> poison, i64 %.spill.load168, i64 0
  %.splat170 = shufflevector <8 x i64> %.splatinsert169, <8 x i64> poison, <8 x i32> zeroinitializer
  %258 = mul <8 x i64> %.splat170, splat (i64 4)
  %259 = add <8 x i64> %257, %258
  %260 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %256, 0
  %261 = insertvalue { <8 x ptr>, <8 x i64> } %260, <8 x i64> %259, 1
  %262 = extractvalue { <8 x ptr>, <8 x i64> } %261, 0
  %263 = extractvalue { <8 x ptr>, <8 x i64> } %261, 1
  %264 = getelementptr i8, <8 x ptr> %262, <8 x i64> %263
  %265 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %264, i32 1, <8 x i1> %62, <8 x float> zeroinitializer)
  %266 = load <8 x float>, ptr %.slot38, align 32
  %267 = select <8 x i1> %62, <8 x float> %265, <8 x float> %266
  store <8 x float> %267, ptr %.slot38, align 32
  br label %direct.schedule.20

direct.schedule.20:                               ; preds = %direct.schedule.19, %direct.false166
  %.state171 = load <8 x float>, ptr %.slot29, align 32
  %.state172 = load <8 x float>, ptr %.slot32, align 32
  %.state173 = load <8 x float>, ptr %.slot35, align 32
  %.state174 = load <8 x float>, ptr %.slot38, align 32
  store i64 4, ptr %.slot39, align 4
  %268 = load <8 x float>, ptr %.slot40, align 32
  %269 = select <8 x i1> %62, <8 x float> %.state171, <8 x float> %268
  store <8 x float> %269, ptr %.slot40, align 32
  %270 = load <8 x float>, ptr %.slot41, align 32
  %271 = select <8 x i1> %62, <8 x float> %.state172, <8 x float> %270
  store <8 x float> %271, ptr %.slot41, align 32
  %272 = load <8 x float>, ptr %.slot42, align 32
  %273 = select <8 x i1> %62, <8 x float> %.state173, <8 x float> %272
  store <8 x float> %273, ptr %.slot42, align 32
  %274 = load <8 x float>, ptr %.slot43, align 32
  %275 = select <8 x i1> %62, <8 x float> %.state174, <8 x float> %274
  store <8 x float> %275, ptr %.slot43, align 32
  br label %direct.schedule.21

direct.schedule.21:                               ; preds = %direct.schedule.30, %direct.schedule.20
  %.state175 = load i64, ptr %.slot39, align 4
  %276 = icmp slt i64 %.state175, 4096
  br i1 %276, label %direct.true176, label %direct.false177

direct.schedule.22:                               ; preds = %direct.true176
  %.state178 = load i64, ptr %.slot39, align 4
  %277 = add i64 %.state178, 0
  %278 = sdiv i64 %277, 1
  %279 = srem i64 %278, 4096
  %.spill.load179 = load i64, ptr %.spill26, align 4
  %280 = add i64 %.spill.load179, %279
  store i64 %280, ptr %.spill44, align 4
  %281 = icmp sge i64 %280, 0
  %282 = icmp slt i64 %280, 4096
  %283 = and i1 %281, %282
  br i1 %283, label %direct.true180, label %direct.false181

direct.schedule.23:                               ; preds = %direct.true180
  %tile_snapshot.spill.load182 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill23, align 64
  %284 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load182, 0
  %285 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load182, 1
  %.spill.load183 = load i64, ptr %.spill44, align 4
  %.splatinsert184 = insertelement <8 x i64> poison, i64 %.spill.load183, i64 0
  %.splat185 = shufflevector <8 x i64> %.splatinsert184, <8 x i64> poison, <8 x i32> zeroinitializer
  %286 = mul <8 x i64> %.splat185, splat (i64 4)
  %287 = add <8 x i64> %285, %286
  %288 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %284, 0
  %289 = insertvalue { <8 x ptr>, <8 x i64> } %288, <8 x i64> %287, 1
  %290 = extractvalue { <8 x ptr>, <8 x i64> } %289, 0
  %291 = extractvalue { <8 x ptr>, <8 x i64> } %289, 1
  %292 = getelementptr i8, <8 x ptr> %290, <8 x i64> %291
  %293 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %292, i32 1, <8 x i1> %62, <8 x float> zeroinitializer)
  %294 = load <8 x float>, ptr %.slot45, align 32
  %295 = select <8 x i1> %62, <8 x float> %293, <8 x float> %294
  store <8 x float> %295, ptr %.slot45, align 32
  br label %direct.schedule.24

direct.schedule.24:                               ; preds = %direct.schedule.23, %direct.false181
  %.state186 = load <8 x float>, ptr %.slot40, align 32
  %.state187 = load <8 x float>, ptr %.slot45, align 32
  %296 = call <8 x float> @llvm.maxnum.v8f32(<8 x float> %.state186, <8 x float> %.state187)
  %297 = load <8 x float>, ptr %.spill46, align 32
  %298 = select <8 x i1> %62, <8 x float> %296, <8 x float> %297
  store <8 x float> %298, ptr %.spill46, align 32
  %.state188 = load i64, ptr %.slot39, align 4
  %299 = add i64 %.state188, 1
  %300 = sdiv i64 %299, 1
  %301 = srem i64 %300, 4096
  %.spill.load189 = load i64, ptr %.spill26, align 4
  %302 = add i64 %.spill.load189, %301
  store i64 %302, ptr %.spill47, align 4
  %303 = icmp sge i64 %302, 0
  %304 = icmp slt i64 %302, 4096
  %305 = and i1 %303, %304
  br i1 %305, label %direct.true190, label %direct.false191

direct.schedule.25:                               ; preds = %direct.true190
  %tile_snapshot.spill.load192 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill23, align 64
  %306 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load192, 0
  %307 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load192, 1
  %.spill.load193 = load i64, ptr %.spill47, align 4
  %.splatinsert194 = insertelement <8 x i64> poison, i64 %.spill.load193, i64 0
  %.splat195 = shufflevector <8 x i64> %.splatinsert194, <8 x i64> poison, <8 x i32> zeroinitializer
  %308 = mul <8 x i64> %.splat195, splat (i64 4)
  %309 = add <8 x i64> %307, %308
  %310 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %306, 0
  %311 = insertvalue { <8 x ptr>, <8 x i64> } %310, <8 x i64> %309, 1
  %312 = extractvalue { <8 x ptr>, <8 x i64> } %311, 0
  %313 = extractvalue { <8 x ptr>, <8 x i64> } %311, 1
  %314 = getelementptr i8, <8 x ptr> %312, <8 x i64> %313
  %315 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %314, i32 1, <8 x i1> %62, <8 x float> zeroinitializer)
  %316 = load <8 x float>, ptr %.slot48, align 32
  %317 = select <8 x i1> %62, <8 x float> %315, <8 x float> %316
  store <8 x float> %317, ptr %.slot48, align 32
  br label %direct.schedule.26

direct.schedule.26:                               ; preds = %direct.schedule.25, %direct.false191
  %.state196 = load <8 x float>, ptr %.slot41, align 32
  %.state197 = load <8 x float>, ptr %.slot48, align 32
  %318 = call <8 x float> @llvm.maxnum.v8f32(<8 x float> %.state196, <8 x float> %.state197)
  %319 = load <8 x float>, ptr %.spill49, align 32
  %320 = select <8 x i1> %62, <8 x float> %318, <8 x float> %319
  store <8 x float> %320, ptr %.spill49, align 32
  %.state198 = load i64, ptr %.slot39, align 4
  %321 = add i64 %.state198, 2
  %322 = sdiv i64 %321, 1
  %323 = srem i64 %322, 4096
  %.spill.load199 = load i64, ptr %.spill26, align 4
  %324 = add i64 %.spill.load199, %323
  store i64 %324, ptr %.spill50, align 4
  %325 = icmp sge i64 %324, 0
  %326 = icmp slt i64 %324, 4096
  %327 = and i1 %325, %326
  br i1 %327, label %direct.true200, label %direct.false201

direct.schedule.27:                               ; preds = %direct.true200
  %tile_snapshot.spill.load202 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill23, align 64
  %328 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load202, 0
  %329 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load202, 1
  %.spill.load203 = load i64, ptr %.spill50, align 4
  %.splatinsert204 = insertelement <8 x i64> poison, i64 %.spill.load203, i64 0
  %.splat205 = shufflevector <8 x i64> %.splatinsert204, <8 x i64> poison, <8 x i32> zeroinitializer
  %330 = mul <8 x i64> %.splat205, splat (i64 4)
  %331 = add <8 x i64> %329, %330
  %332 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %328, 0
  %333 = insertvalue { <8 x ptr>, <8 x i64> } %332, <8 x i64> %331, 1
  %334 = extractvalue { <8 x ptr>, <8 x i64> } %333, 0
  %335 = extractvalue { <8 x ptr>, <8 x i64> } %333, 1
  %336 = getelementptr i8, <8 x ptr> %334, <8 x i64> %335
  %337 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %336, i32 1, <8 x i1> %62, <8 x float> zeroinitializer)
  %338 = load <8 x float>, ptr %.slot51, align 32
  %339 = select <8 x i1> %62, <8 x float> %337, <8 x float> %338
  store <8 x float> %339, ptr %.slot51, align 32
  br label %direct.schedule.28

direct.schedule.28:                               ; preds = %direct.schedule.27, %direct.false201
  %.state206 = load <8 x float>, ptr %.slot42, align 32
  %.state207 = load <8 x float>, ptr %.slot51, align 32
  %340 = call <8 x float> @llvm.maxnum.v8f32(<8 x float> %.state206, <8 x float> %.state207)
  %341 = load <8 x float>, ptr %.spill52, align 32
  %342 = select <8 x i1> %62, <8 x float> %340, <8 x float> %341
  store <8 x float> %342, ptr %.spill52, align 32
  %.state208 = load i64, ptr %.slot39, align 4
  %343 = add i64 %.state208, 3
  %344 = sdiv i64 %343, 1
  %345 = srem i64 %344, 4096
  %.spill.load209 = load i64, ptr %.spill26, align 4
  %346 = add i64 %.spill.load209, %345
  store i64 %346, ptr %.spill53, align 4
  %347 = icmp sge i64 %346, 0
  %348 = icmp slt i64 %346, 4096
  %349 = and i1 %347, %348
  br i1 %349, label %direct.true210, label %direct.false211

direct.schedule.29:                               ; preds = %direct.true210
  %tile_snapshot.spill.load212 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill23, align 64
  %350 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load212, 0
  %351 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load212, 1
  %.spill.load213 = load i64, ptr %.spill53, align 4
  %.splatinsert214 = insertelement <8 x i64> poison, i64 %.spill.load213, i64 0
  %.splat215 = shufflevector <8 x i64> %.splatinsert214, <8 x i64> poison, <8 x i32> zeroinitializer
  %352 = mul <8 x i64> %.splat215, splat (i64 4)
  %353 = add <8 x i64> %351, %352
  %354 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %350, 0
  %355 = insertvalue { <8 x ptr>, <8 x i64> } %354, <8 x i64> %353, 1
  %356 = extractvalue { <8 x ptr>, <8 x i64> } %355, 0
  %357 = extractvalue { <8 x ptr>, <8 x i64> } %355, 1
  %358 = getelementptr i8, <8 x ptr> %356, <8 x i64> %357
  %359 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %358, i32 1, <8 x i1> %62, <8 x float> zeroinitializer)
  %360 = load <8 x float>, ptr %.slot54, align 32
  %361 = select <8 x i1> %62, <8 x float> %359, <8 x float> %360
  store <8 x float> %361, ptr %.slot54, align 32
  br label %direct.schedule.30

direct.schedule.30:                               ; preds = %direct.schedule.29, %direct.false211
  %.state216 = load <8 x float>, ptr %.slot43, align 32
  %.state217 = load <8 x float>, ptr %.slot54, align 32
  %362 = call <8 x float> @llvm.maxnum.v8f32(<8 x float> %.state216, <8 x float> %.state217)
  %.state218 = load i64, ptr %.slot39, align 4
  %363 = add i64 %.state218, 4
  %.spill.load219 = load <8 x float>, ptr %.spill46, align 32
  %.spill.load220 = load <8 x float>, ptr %.spill49, align 32
  %.spill.load221 = load <8 x float>, ptr %.spill52, align 32
  store i64 %363, ptr %.slot39, align 4
  %364 = load <8 x float>, ptr %.slot40, align 32
  %365 = select <8 x i1> %62, <8 x float> %.spill.load219, <8 x float> %364
  store <8 x float> %365, ptr %.slot40, align 32
  %366 = load <8 x float>, ptr %.slot41, align 32
  %367 = select <8 x i1> %62, <8 x float> %.spill.load220, <8 x float> %366
  store <8 x float> %367, ptr %.slot41, align 32
  %368 = load <8 x float>, ptr %.slot42, align 32
  %369 = select <8 x i1> %62, <8 x float> %.spill.load221, <8 x float> %368
  store <8 x float> %369, ptr %.slot42, align 32
  %370 = load <8 x float>, ptr %.slot43, align 32
  %371 = select <8 x i1> %62, <8 x float> %362, <8 x float> %370
  store <8 x float> %371, ptr %.slot43, align 32
  br label %direct.schedule.21

direct.schedule.31:                               ; preds = %direct.false177
  %.spill.load222 = load float, ptr %.spill25, align 4
  %.splatinsert223 = insertelement <8 x float> poison, float %.spill.load222, i64 0
  %.splat224 = shufflevector <8 x float> %.splatinsert223, <8 x float> poison, <8 x i32> zeroinitializer
  %.state225 = load <8 x float>, ptr %.slot40, align 32
  %372 = call <8 x float> @llvm.maxnum.v8f32(<8 x float> %.splat224, <8 x float> %.state225)
  %.state226 = load <8 x float>, ptr %.slot41, align 32
  %373 = call <8 x float> @llvm.maxnum.v8f32(<8 x float> %372, <8 x float> %.state226)
  %.state227 = load <8 x float>, ptr %.slot42, align 32
  %374 = call <8 x float> @llvm.maxnum.v8f32(<8 x float> %373, <8 x float> %.state227)
  %.state228 = load <8 x float>, ptr %.slot43, align 32
  %375 = call <8 x float> @llvm.maxnum.v8f32(<8 x float> %374, <8 x float> %.state228)
  %376 = load <8 x float>, ptr %.spill55, align 32
  %377 = select <8 x i1> %62, <8 x float> %375, <8 x float> %376
  store <8 x float> %377, ptr %.spill55, align 32
  store float 0.000000e+00, ptr %.spill56, align 4
  %378 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill57, align 64
  %379 = extractvalue { <8 x ptr>, <8 x i64> } %14, 0
  %380 = extractvalue { <8 x ptr>, <8 x i64> } %378, 0
  %381 = select <8 x i1> %62, <8 x ptr> %379, <8 x ptr> %380
  %382 = extractvalue { <8 x ptr>, <8 x i64> } %14, 1
  %383 = extractvalue { <8 x ptr>, <8 x i64> } %378, 1
  %384 = select <8 x i1> %62, <8 x i64> %382, <8 x i64> %383
  %385 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %381, 0
  %386 = insertvalue { <8 x ptr>, <8 x i64> } %385, <8 x i64> %384, 1
  store { <8 x ptr>, <8 x i64> } %386, ptr %tile_snapshot.spill57, align 64
  store i64 0, ptr %.slot58, align 4
  br label %direct.schedule.32

direct.schedule.32:                               ; preds = %direct.schedule.33, %direct.schedule.31
  %.state229 = load i64, ptr %.slot58, align 4
  %387 = icmp slt i64 %.state229, 4096
  br i1 %387, label %direct.true230, label %direct.false231

direct.schedule.33:                               ; preds = %direct.true230
  %.state232 = load i64, ptr %.slot58, align 4
  %388 = srem i64 %.state232, 4096
  %389 = add i64 0, %388
  %tile_snapshot.spill.load233 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill20, align 64
  %390 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load233, 0
  %391 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load233, 1
  %.splatinsert234 = insertelement <8 x i64> poison, i64 %389, i64 0
  %.splat235 = shufflevector <8 x i64> %.splatinsert234, <8 x i64> poison, <8 x i32> zeroinitializer
  %392 = add <8 x i64> %391, %.splat235
  %393 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %390, 0
  %394 = insertvalue { <8 x ptr>, <8 x i64> } %393, <8 x i64> %392, 1
  %395 = extractvalue { <8 x ptr>, <8 x i64> } %394, 0
  %396 = extractvalue { <8 x ptr>, <8 x i64> } %394, 1
  %397 = getelementptr i8, <8 x ptr> %395, <8 x i64> %396
  %398 = call <8 x i8> @llvm.masked.gather.v8i8.v8p0(<8 x ptr> %397, i32 1, <8 x i1> %62, <8 x i8> zeroinitializer)
  %399 = icmp ne <8 x i8> %398, zeroinitializer
  %400 = srem i64 %389, 4096
  %401 = add i64 0, %400
  %402 = srem i64 %401, 4096
  %403 = add i64 0, %402
  %tile_snapshot.spill.load236 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill23, align 64
  %404 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load236, 0
  %405 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load236, 1
  %.splatinsert237 = insertelement <8 x i64> poison, i64 %403, i64 0
  %.splat238 = shufflevector <8 x i64> %.splatinsert237, <8 x i64> poison, <8 x i32> zeroinitializer
  %406 = mul <8 x i64> %.splat238, splat (i64 4)
  %407 = add <8 x i64> %405, %406
  %408 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %404, 0
  %409 = insertvalue { <8 x ptr>, <8 x i64> } %408, <8 x i64> %407, 1
  %410 = extractvalue { <8 x ptr>, <8 x i64> } %409, 0
  %411 = extractvalue { <8 x ptr>, <8 x i64> } %409, 1
  %412 = getelementptr i8, <8 x ptr> %410, <8 x i64> %411
  %413 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %412, i32 1, <8 x i1> %62, <8 x float> zeroinitializer)
  %.spill.load239 = load <8 x float>, ptr %.spill55, align 32
  %414 = fsub <8 x float> %413, %.spill.load239
  %415 = select <8 x i1> %62, <8 x float> %414, <8 x float> zeroinitializer
  %native.exp = call <8 x float> @__luisa_cpu_native_exp_f32_v8_u10(<8 x float> %415)
  %.spill.load240 = load float, ptr %.spill56, align 4
  %.splatinsert241 = insertelement <8 x float> poison, float %.spill.load240, i64 0
  %.splat242 = shufflevector <8 x float> %.splatinsert241, <8 x float> poison, <8 x i32> zeroinitializer
  %416 = select <8 x i1> %399, <8 x float> %native.exp, <8 x float> %.splat242
  %tile_snapshot.spill.load243 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill57, align 64
  %417 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load243, 0
  %418 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load243, 1
  %.state244 = load i64, ptr %.slot58, align 4
  %.splatinsert245 = insertelement <8 x i64> poison, i64 %.state244, i64 0
  %.splat246 = shufflevector <8 x i64> %.splatinsert245, <8 x i64> poison, <8 x i32> zeroinitializer
  %419 = mul <8 x i64> %.splat246, splat (i64 4)
  %420 = add <8 x i64> %418, %419
  %421 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %417, 0
  %422 = insertvalue { <8 x ptr>, <8 x i64> } %421, <8 x i64> %420, 1
  %423 = extractvalue { <8 x ptr>, <8 x i64> } %422, 0
  %424 = extractvalue { <8 x ptr>, <8 x i64> } %422, 1
  %425 = getelementptr i8, <8 x ptr> %423, <8 x i64> %424
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %416, <8 x ptr> %425, i32 1, <8 x i1> %62)
  %.state247 = load i64, ptr %.slot58, align 4
  %426 = add i64 %.state247, 1
  store i64 %426, ptr %.slot58, align 4
  br label %direct.schedule.32

direct.schedule.34:                               ; preds = %direct.false231
  %.spill.load248 = load i1, ptr %.spill28, align 1
  br i1 %.spill.load248, label %direct.true249, label %direct.false250

direct.schedule.35:                               ; preds = %direct.true249
  %tile_snapshot.spill.load251 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill57, align 64
  %427 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load251, 0
  %428 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load251, 1
  %.spill.load252 = load i64, ptr %.spill27, align 4
  %.splatinsert253 = insertelement <8 x i64> poison, i64 %.spill.load252, i64 0
  %.splat254 = shufflevector <8 x i64> %.splatinsert253, <8 x i64> poison, <8 x i32> zeroinitializer
  %429 = mul <8 x i64> %.splat254, splat (i64 4)
  %430 = add <8 x i64> %428, %429
  %431 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %427, 0
  %432 = insertvalue { <8 x ptr>, <8 x i64> } %431, <8 x i64> %430, 1
  %433 = extractvalue { <8 x ptr>, <8 x i64> } %432, 0
  %434 = extractvalue { <8 x ptr>, <8 x i64> } %432, 1
  %435 = getelementptr i8, <8 x ptr> %433, <8 x i64> %434
  %436 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %435, i32 1, <8 x i1> %62, <8 x float> zeroinitializer)
  %437 = load <8 x float>, ptr %.slot59, align 32
  %438 = select <8 x i1> %62, <8 x float> %436, <8 x float> %437
  store <8 x float> %438, ptr %.slot59, align 32
  br label %direct.schedule.36

direct.schedule.36:                               ; preds = %direct.schedule.35, %direct.false250
  %.spill.load255 = load i1, ptr %.spill31, align 1
  br i1 %.spill.load255, label %direct.true256, label %direct.false257

direct.schedule.37:                               ; preds = %direct.true256
  %tile_snapshot.spill.load258 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill57, align 64
  %439 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load258, 0
  %440 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load258, 1
  %.spill.load259 = load i64, ptr %.spill30, align 4
  %.splatinsert260 = insertelement <8 x i64> poison, i64 %.spill.load259, i64 0
  %.splat261 = shufflevector <8 x i64> %.splatinsert260, <8 x i64> poison, <8 x i32> zeroinitializer
  %441 = mul <8 x i64> %.splat261, splat (i64 4)
  %442 = add <8 x i64> %440, %441
  %443 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %439, 0
  %444 = insertvalue { <8 x ptr>, <8 x i64> } %443, <8 x i64> %442, 1
  %445 = extractvalue { <8 x ptr>, <8 x i64> } %444, 0
  %446 = extractvalue { <8 x ptr>, <8 x i64> } %444, 1
  %447 = getelementptr i8, <8 x ptr> %445, <8 x i64> %446
  %448 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %447, i32 1, <8 x i1> %62, <8 x float> zeroinitializer)
  %449 = load <8 x float>, ptr %.slot60, align 32
  %450 = select <8 x i1> %62, <8 x float> %448, <8 x float> %449
  store <8 x float> %450, ptr %.slot60, align 32
  br label %direct.schedule.38

direct.schedule.38:                               ; preds = %direct.schedule.37, %direct.false257
  %.spill.load262 = load i1, ptr %.spill34, align 1
  br i1 %.spill.load262, label %direct.true263, label %direct.false264

direct.schedule.39:                               ; preds = %direct.true263
  %tile_snapshot.spill.load265 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill57, align 64
  %451 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load265, 0
  %452 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load265, 1
  %.spill.load266 = load i64, ptr %.spill33, align 4
  %.splatinsert267 = insertelement <8 x i64> poison, i64 %.spill.load266, i64 0
  %.splat268 = shufflevector <8 x i64> %.splatinsert267, <8 x i64> poison, <8 x i32> zeroinitializer
  %453 = mul <8 x i64> %.splat268, splat (i64 4)
  %454 = add <8 x i64> %452, %453
  %455 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %451, 0
  %456 = insertvalue { <8 x ptr>, <8 x i64> } %455, <8 x i64> %454, 1
  %457 = extractvalue { <8 x ptr>, <8 x i64> } %456, 0
  %458 = extractvalue { <8 x ptr>, <8 x i64> } %456, 1
  %459 = getelementptr i8, <8 x ptr> %457, <8 x i64> %458
  %460 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %459, i32 1, <8 x i1> %62, <8 x float> zeroinitializer)
  %461 = load <8 x float>, ptr %.slot61, align 32
  %462 = select <8 x i1> %62, <8 x float> %460, <8 x float> %461
  store <8 x float> %462, ptr %.slot61, align 32
  br label %direct.schedule.40

direct.schedule.40:                               ; preds = %direct.schedule.39, %direct.false264
  %.spill.load269 = load i1, ptr %.spill37, align 1
  br i1 %.spill.load269, label %direct.true270, label %direct.false271

direct.schedule.41:                               ; preds = %direct.true270
  %tile_snapshot.spill.load272 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill57, align 64
  %463 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load272, 0
  %464 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load272, 1
  %.spill.load273 = load i64, ptr %.spill36, align 4
  %.splatinsert274 = insertelement <8 x i64> poison, i64 %.spill.load273, i64 0
  %.splat275 = shufflevector <8 x i64> %.splatinsert274, <8 x i64> poison, <8 x i32> zeroinitializer
  %465 = mul <8 x i64> %.splat275, splat (i64 4)
  %466 = add <8 x i64> %464, %465
  %467 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %463, 0
  %468 = insertvalue { <8 x ptr>, <8 x i64> } %467, <8 x i64> %466, 1
  %469 = extractvalue { <8 x ptr>, <8 x i64> } %468, 0
  %470 = extractvalue { <8 x ptr>, <8 x i64> } %468, 1
  %471 = getelementptr i8, <8 x ptr> %469, <8 x i64> %470
  %472 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %471, i32 1, <8 x i1> %62, <8 x float> zeroinitializer)
  %473 = load <8 x float>, ptr %.slot62, align 32
  %474 = select <8 x i1> %62, <8 x float> %472, <8 x float> %473
  store <8 x float> %474, ptr %.slot62, align 32
  br label %direct.schedule.42

direct.schedule.42:                               ; preds = %direct.schedule.41, %direct.false271
  %.state276 = load <8 x float>, ptr %.slot59, align 32
  %.state277 = load <8 x float>, ptr %.slot60, align 32
  %.state278 = load <8 x float>, ptr %.slot61, align 32
  %.state279 = load <8 x float>, ptr %.slot62, align 32
  store i64 4, ptr %.slot63, align 4
  %475 = load <8 x float>, ptr %.slot64, align 32
  %476 = select <8 x i1> %62, <8 x float> %.state276, <8 x float> %475
  store <8 x float> %476, ptr %.slot64, align 32
  %477 = load <8 x float>, ptr %.slot65, align 32
  %478 = select <8 x i1> %62, <8 x float> %.state277, <8 x float> %477
  store <8 x float> %478, ptr %.slot65, align 32
  %479 = load <8 x float>, ptr %.slot66, align 32
  %480 = select <8 x i1> %62, <8 x float> %.state278, <8 x float> %479
  store <8 x float> %480, ptr %.slot66, align 32
  %481 = load <8 x float>, ptr %.slot67, align 32
  %482 = select <8 x i1> %62, <8 x float> %.state279, <8 x float> %481
  store <8 x float> %482, ptr %.slot67, align 32
  br label %direct.schedule.43

direct.schedule.43:                               ; preds = %direct.schedule.52, %direct.schedule.42
  %.state280 = load i64, ptr %.slot63, align 4
  %483 = icmp slt i64 %.state280, 4096
  br i1 %483, label %direct.true281, label %direct.false282

direct.schedule.44:                               ; preds = %direct.true281
  %.state283 = load i64, ptr %.slot63, align 4
  %484 = add i64 %.state283, 0
  %485 = sdiv i64 %484, 1
  %486 = srem i64 %485, 4096
  %.spill.load284 = load i64, ptr %.spill26, align 4
  %487 = add i64 %.spill.load284, %486
  store i64 %487, ptr %.spill68, align 4
  %488 = icmp sge i64 %487, 0
  %489 = icmp slt i64 %487, 4096
  %490 = and i1 %488, %489
  br i1 %490, label %direct.true285, label %direct.false286

direct.schedule.45:                               ; preds = %direct.true285
  %tile_snapshot.spill.load287 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill57, align 64
  %491 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load287, 0
  %492 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load287, 1
  %.spill.load288 = load i64, ptr %.spill68, align 4
  %.splatinsert289 = insertelement <8 x i64> poison, i64 %.spill.load288, i64 0
  %.splat290 = shufflevector <8 x i64> %.splatinsert289, <8 x i64> poison, <8 x i32> zeroinitializer
  %493 = mul <8 x i64> %.splat290, splat (i64 4)
  %494 = add <8 x i64> %492, %493
  %495 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %491, 0
  %496 = insertvalue { <8 x ptr>, <8 x i64> } %495, <8 x i64> %494, 1
  %497 = extractvalue { <8 x ptr>, <8 x i64> } %496, 0
  %498 = extractvalue { <8 x ptr>, <8 x i64> } %496, 1
  %499 = getelementptr i8, <8 x ptr> %497, <8 x i64> %498
  %500 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %499, i32 1, <8 x i1> %62, <8 x float> zeroinitializer)
  %501 = load <8 x float>, ptr %.slot69, align 32
  %502 = select <8 x i1> %62, <8 x float> %500, <8 x float> %501
  store <8 x float> %502, ptr %.slot69, align 32
  br label %direct.schedule.46

direct.schedule.46:                               ; preds = %direct.schedule.45, %direct.false286
  %.state291 = load <8 x float>, ptr %.slot64, align 32
  %.state292 = load <8 x float>, ptr %.slot69, align 32
  %503 = fadd <8 x float> %.state291, %.state292
  %504 = load <8 x float>, ptr %.spill70, align 32
  %505 = select <8 x i1> %62, <8 x float> %503, <8 x float> %504
  store <8 x float> %505, ptr %.spill70, align 32
  %.state293 = load i64, ptr %.slot63, align 4
  %506 = add i64 %.state293, 1
  %507 = sdiv i64 %506, 1
  %508 = srem i64 %507, 4096
  %.spill.load294 = load i64, ptr %.spill26, align 4
  %509 = add i64 %.spill.load294, %508
  store i64 %509, ptr %.spill71, align 4
  %510 = icmp sge i64 %509, 0
  %511 = icmp slt i64 %509, 4096
  %512 = and i1 %510, %511
  br i1 %512, label %direct.true295, label %direct.false296

direct.schedule.47:                               ; preds = %direct.true295
  %tile_snapshot.spill.load297 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill57, align 64
  %513 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load297, 0
  %514 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load297, 1
  %.spill.load298 = load i64, ptr %.spill71, align 4
  %.splatinsert299 = insertelement <8 x i64> poison, i64 %.spill.load298, i64 0
  %.splat300 = shufflevector <8 x i64> %.splatinsert299, <8 x i64> poison, <8 x i32> zeroinitializer
  %515 = mul <8 x i64> %.splat300, splat (i64 4)
  %516 = add <8 x i64> %514, %515
  %517 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %513, 0
  %518 = insertvalue { <8 x ptr>, <8 x i64> } %517, <8 x i64> %516, 1
  %519 = extractvalue { <8 x ptr>, <8 x i64> } %518, 0
  %520 = extractvalue { <8 x ptr>, <8 x i64> } %518, 1
  %521 = getelementptr i8, <8 x ptr> %519, <8 x i64> %520
  %522 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %521, i32 1, <8 x i1> %62, <8 x float> zeroinitializer)
  %523 = load <8 x float>, ptr %.slot72, align 32
  %524 = select <8 x i1> %62, <8 x float> %522, <8 x float> %523
  store <8 x float> %524, ptr %.slot72, align 32
  br label %direct.schedule.48

direct.schedule.48:                               ; preds = %direct.schedule.47, %direct.false296
  %.state301 = load <8 x float>, ptr %.slot65, align 32
  %.state302 = load <8 x float>, ptr %.slot72, align 32
  %525 = fadd <8 x float> %.state301, %.state302
  %526 = load <8 x float>, ptr %.spill73, align 32
  %527 = select <8 x i1> %62, <8 x float> %525, <8 x float> %526
  store <8 x float> %527, ptr %.spill73, align 32
  %.state303 = load i64, ptr %.slot63, align 4
  %528 = add i64 %.state303, 2
  %529 = sdiv i64 %528, 1
  %530 = srem i64 %529, 4096
  %.spill.load304 = load i64, ptr %.spill26, align 4
  %531 = add i64 %.spill.load304, %530
  store i64 %531, ptr %.spill74, align 4
  %532 = icmp sge i64 %531, 0
  %533 = icmp slt i64 %531, 4096
  %534 = and i1 %532, %533
  br i1 %534, label %direct.true305, label %direct.false306

direct.schedule.49:                               ; preds = %direct.true305
  %tile_snapshot.spill.load307 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill57, align 64
  %535 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load307, 0
  %536 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load307, 1
  %.spill.load308 = load i64, ptr %.spill74, align 4
  %.splatinsert309 = insertelement <8 x i64> poison, i64 %.spill.load308, i64 0
  %.splat310 = shufflevector <8 x i64> %.splatinsert309, <8 x i64> poison, <8 x i32> zeroinitializer
  %537 = mul <8 x i64> %.splat310, splat (i64 4)
  %538 = add <8 x i64> %536, %537
  %539 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %535, 0
  %540 = insertvalue { <8 x ptr>, <8 x i64> } %539, <8 x i64> %538, 1
  %541 = extractvalue { <8 x ptr>, <8 x i64> } %540, 0
  %542 = extractvalue { <8 x ptr>, <8 x i64> } %540, 1
  %543 = getelementptr i8, <8 x ptr> %541, <8 x i64> %542
  %544 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %543, i32 1, <8 x i1> %62, <8 x float> zeroinitializer)
  %545 = load <8 x float>, ptr %.slot75, align 32
  %546 = select <8 x i1> %62, <8 x float> %544, <8 x float> %545
  store <8 x float> %546, ptr %.slot75, align 32
  br label %direct.schedule.50

direct.schedule.50:                               ; preds = %direct.schedule.49, %direct.false306
  %.state311 = load <8 x float>, ptr %.slot66, align 32
  %.state312 = load <8 x float>, ptr %.slot75, align 32
  %547 = fadd <8 x float> %.state311, %.state312
  %548 = load <8 x float>, ptr %.spill76, align 32
  %549 = select <8 x i1> %62, <8 x float> %547, <8 x float> %548
  store <8 x float> %549, ptr %.spill76, align 32
  %.state313 = load i64, ptr %.slot63, align 4
  %550 = add i64 %.state313, 3
  %551 = sdiv i64 %550, 1
  %552 = srem i64 %551, 4096
  %.spill.load314 = load i64, ptr %.spill26, align 4
  %553 = add i64 %.spill.load314, %552
  store i64 %553, ptr %.spill77, align 4
  %554 = icmp sge i64 %553, 0
  %555 = icmp slt i64 %553, 4096
  %556 = and i1 %554, %555
  br i1 %556, label %direct.true315, label %direct.false316

direct.schedule.51:                               ; preds = %direct.true315
  %tile_snapshot.spill.load317 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill57, align 64
  %557 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load317, 0
  %558 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load317, 1
  %.spill.load318 = load i64, ptr %.spill77, align 4
  %.splatinsert319 = insertelement <8 x i64> poison, i64 %.spill.load318, i64 0
  %.splat320 = shufflevector <8 x i64> %.splatinsert319, <8 x i64> poison, <8 x i32> zeroinitializer
  %559 = mul <8 x i64> %.splat320, splat (i64 4)
  %560 = add <8 x i64> %558, %559
  %561 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %557, 0
  %562 = insertvalue { <8 x ptr>, <8 x i64> } %561, <8 x i64> %560, 1
  %563 = extractvalue { <8 x ptr>, <8 x i64> } %562, 0
  %564 = extractvalue { <8 x ptr>, <8 x i64> } %562, 1
  %565 = getelementptr i8, <8 x ptr> %563, <8 x i64> %564
  %566 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %565, i32 1, <8 x i1> %62, <8 x float> zeroinitializer)
  %567 = load <8 x float>, ptr %.slot78, align 32
  %568 = select <8 x i1> %62, <8 x float> %566, <8 x float> %567
  store <8 x float> %568, ptr %.slot78, align 32
  br label %direct.schedule.52

direct.schedule.52:                               ; preds = %direct.schedule.51, %direct.false316
  %.state321 = load <8 x float>, ptr %.slot67, align 32
  %.state322 = load <8 x float>, ptr %.slot78, align 32
  %569 = fadd <8 x float> %.state321, %.state322
  %.state323 = load i64, ptr %.slot63, align 4
  %570 = add i64 %.state323, 4
  %.spill.load324 = load <8 x float>, ptr %.spill70, align 32
  %.spill.load325 = load <8 x float>, ptr %.spill73, align 32
  %.spill.load326 = load <8 x float>, ptr %.spill76, align 32
  store i64 %570, ptr %.slot63, align 4
  %571 = load <8 x float>, ptr %.slot64, align 32
  %572 = select <8 x i1> %62, <8 x float> %.spill.load324, <8 x float> %571
  store <8 x float> %572, ptr %.slot64, align 32
  %573 = load <8 x float>, ptr %.slot65, align 32
  %574 = select <8 x i1> %62, <8 x float> %.spill.load325, <8 x float> %573
  store <8 x float> %574, ptr %.slot65, align 32
  %575 = load <8 x float>, ptr %.slot66, align 32
  %576 = select <8 x i1> %62, <8 x float> %.spill.load326, <8 x float> %575
  store <8 x float> %576, ptr %.slot66, align 32
  %577 = load <8 x float>, ptr %.slot67, align 32
  %578 = select <8 x i1> %62, <8 x float> %569, <8 x float> %577
  store <8 x float> %578, ptr %.slot67, align 32
  br label %direct.schedule.43

direct.schedule.53:                               ; preds = %direct.false282
  %.spill.load327 = load float, ptr %.spill56, align 4
  %.splatinsert328 = insertelement <8 x float> poison, float %.spill.load327, i64 0
  %.splat329 = shufflevector <8 x float> %.splatinsert328, <8 x float> poison, <8 x i32> zeroinitializer
  %.state330 = load <8 x float>, ptr %.slot64, align 32
  %579 = fadd <8 x float> %.splat329, %.state330
  %.state331 = load <8 x float>, ptr %.slot65, align 32
  %580 = fadd <8 x float> %579, %.state331
  %.state332 = load <8 x float>, ptr %.slot66, align 32
  %581 = fadd <8 x float> %580, %.state332
  %.state333 = load <8 x float>, ptr %.slot67, align 32
  %582 = fadd <8 x float> %581, %.state333
  %583 = load <8 x float>, ptr %.spill79, align 32
  %584 = select <8 x i1> %62, <8 x float> %582, <8 x float> %583
  store <8 x float> %584, ptr %.spill79, align 32
  store i64 0, ptr %.slot80, align 4
  br label %direct.schedule.54

direct.schedule.54:                               ; preds = %direct.schedule.55, %direct.schedule.53
  %.state334 = load i64, ptr %.slot80, align 4
  %585 = icmp slt i64 %.state334, 4096
  br i1 %585, label %direct.true335, label %direct.false336

direct.schedule.55:                               ; preds = %direct.true335
  %.state337 = load i64, ptr %.slot80, align 4
  %586 = srem i64 %.state337, 4096
  %.state338 = load i64, ptr %.slot80, align 4
  %587 = sdiv i64 %.state338, 4096
  %588 = srem i64 %587, 1
  %.spill.load339 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert340 = insertelement <8 x i64> poison, i64 %588, i64 0
  %.splat341 = shufflevector <8 x i64> %.splatinsert340, <8 x i64> poison, <8 x i32> zeroinitializer
  %589 = add <8 x i64> %.spill.load339, %.splat341
  %590 = add <8 x i64> zeroinitializer, %589
  %591 = add i64 0, %586
  %592 = mul <8 x i64> %590, splat (i64 4096)
  %.splatinsert342 = insertelement <8 x i64> poison, i64 %591, i64 0
  %.splat343 = shufflevector <8 x i64> %.splatinsert342, <8 x i64> poison, <8 x i32> zeroinitializer
  %593 = add <8 x i64> %592, %.splat343
  %594 = add i64 0, %586
  %tile_snapshot.spill.load344 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill57, align 64
  %595 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load344, 0
  %596 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load344, 1
  %.splatinsert345 = insertelement <8 x i64> poison, i64 %594, i64 0
  %.splat346 = shufflevector <8 x i64> %.splatinsert345, <8 x i64> poison, <8 x i32> zeroinitializer
  %597 = mul <8 x i64> %.splat346, splat (i64 4)
  %598 = add <8 x i64> %596, %597
  %599 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %595, 0
  %600 = insertvalue { <8 x ptr>, <8 x i64> } %599, <8 x i64> %598, 1
  %601 = extractvalue { <8 x ptr>, <8 x i64> } %600, 0
  %602 = extractvalue { <8 x ptr>, <8 x i64> } %600, 1
  %603 = getelementptr i8, <8 x ptr> %601, <8 x i64> %602
  %604 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %603, i32 1, <8 x i1> %62, <8 x float> zeroinitializer)
  %.spill.load347 = load <8 x float>, ptr %.spill79, align 32
  %605 = fdiv <8 x float> %604, %.spill.load347
  %606 = extractvalue { ptr, i64 } %38, 0
  %607 = mul <8 x i64> %593, splat (i64 4)
  %608 = getelementptr i8, ptr %606, <8 x i64> %607
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %605, <8 x ptr> %608, i32 1, <8 x i1> %62)
  %.state348 = load i64, ptr %.slot80, align 4
  %609 = add i64 %.state348, 1
  store i64 %609, ptr %.slot80, align 4
  br label %direct.schedule.54

direct.schedule.56:                               ; preds = %direct.false336
  ret void

direct.activate:                                  ; preds = %prologue
  br label %direct.schedule.0

direct.inactive:                                  ; preds = %prologue
  ret void

direct.true:                                      ; preds = %direct.schedule.1
  br label %direct.schedule.2

direct.false:                                     ; preds = %direct.schedule.1
  br label %direct.schedule.3

direct.true102:                                   ; preds = %direct.schedule.4
  br label %direct.schedule.5

direct.false103:                                  ; preds = %direct.schedule.4
  br label %direct.schedule.6

direct.true114:                                   ; preds = %direct.schedule.7
  br label %direct.schedule.8

direct.false115:                                  ; preds = %direct.schedule.7
  br label %direct.schedule.9

direct.true127:                                   ; preds = %direct.schedule.10
  br label %direct.schedule.11

direct.false128:                                  ; preds = %direct.schedule.10
  br label %direct.schedule.12

direct.true144:                                   ; preds = %direct.schedule.12
  br label %direct.schedule.13

direct.false145:                                  ; preds = %direct.schedule.12
  %610 = load <8 x float>, ptr %.slot29, align 32
  %611 = select <8 x i1> %62, <8 x float> zeroinitializer, <8 x float> %610
  store <8 x float> %611, ptr %.slot29, align 32
  br label %direct.schedule.14

direct.true151:                                   ; preds = %direct.schedule.14
  br label %direct.schedule.15

direct.false152:                                  ; preds = %direct.schedule.14
  %612 = load <8 x float>, ptr %.slot32, align 32
  %613 = select <8 x i1> %62, <8 x float> zeroinitializer, <8 x float> %612
  store <8 x float> %613, ptr %.slot32, align 32
  br label %direct.schedule.16

direct.true158:                                   ; preds = %direct.schedule.16
  br label %direct.schedule.17

direct.false159:                                  ; preds = %direct.schedule.16
  %614 = load <8 x float>, ptr %.slot35, align 32
  %615 = select <8 x i1> %62, <8 x float> zeroinitializer, <8 x float> %614
  store <8 x float> %615, ptr %.slot35, align 32
  br label %direct.schedule.18

direct.true165:                                   ; preds = %direct.schedule.18
  br label %direct.schedule.19

direct.false166:                                  ; preds = %direct.schedule.18
  %616 = load <8 x float>, ptr %.slot38, align 32
  %617 = select <8 x i1> %62, <8 x float> zeroinitializer, <8 x float> %616
  store <8 x float> %617, ptr %.slot38, align 32
  br label %direct.schedule.20

direct.true176:                                   ; preds = %direct.schedule.21
  br label %direct.schedule.22

direct.false177:                                  ; preds = %direct.schedule.21
  br label %direct.schedule.31

direct.true180:                                   ; preds = %direct.schedule.22
  br label %direct.schedule.23

direct.false181:                                  ; preds = %direct.schedule.22
  %618 = load <8 x float>, ptr %.slot45, align 32
  %619 = select <8 x i1> %62, <8 x float> zeroinitializer, <8 x float> %618
  store <8 x float> %619, ptr %.slot45, align 32
  br label %direct.schedule.24

direct.true190:                                   ; preds = %direct.schedule.24
  br label %direct.schedule.25

direct.false191:                                  ; preds = %direct.schedule.24
  %620 = load <8 x float>, ptr %.slot48, align 32
  %621 = select <8 x i1> %62, <8 x float> zeroinitializer, <8 x float> %620
  store <8 x float> %621, ptr %.slot48, align 32
  br label %direct.schedule.26

direct.true200:                                   ; preds = %direct.schedule.26
  br label %direct.schedule.27

direct.false201:                                  ; preds = %direct.schedule.26
  %622 = load <8 x float>, ptr %.slot51, align 32
  %623 = select <8 x i1> %62, <8 x float> zeroinitializer, <8 x float> %622
  store <8 x float> %623, ptr %.slot51, align 32
  br label %direct.schedule.28

direct.true210:                                   ; preds = %direct.schedule.28
  br label %direct.schedule.29

direct.false211:                                  ; preds = %direct.schedule.28
  %624 = load <8 x float>, ptr %.slot54, align 32
  %625 = select <8 x i1> %62, <8 x float> zeroinitializer, <8 x float> %624
  store <8 x float> %625, ptr %.slot54, align 32
  br label %direct.schedule.30

direct.true230:                                   ; preds = %direct.schedule.32
  br label %direct.schedule.33

direct.false231:                                  ; preds = %direct.schedule.32
  br label %direct.schedule.34

direct.true249:                                   ; preds = %direct.schedule.34
  br label %direct.schedule.35

direct.false250:                                  ; preds = %direct.schedule.34
  %626 = load <8 x float>, ptr %.slot59, align 32
  %627 = select <8 x i1> %62, <8 x float> zeroinitializer, <8 x float> %626
  store <8 x float> %627, ptr %.slot59, align 32
  br label %direct.schedule.36

direct.true256:                                   ; preds = %direct.schedule.36
  br label %direct.schedule.37

direct.false257:                                  ; preds = %direct.schedule.36
  %628 = load <8 x float>, ptr %.slot60, align 32
  %629 = select <8 x i1> %62, <8 x float> zeroinitializer, <8 x float> %628
  store <8 x float> %629, ptr %.slot60, align 32
  br label %direct.schedule.38

direct.true263:                                   ; preds = %direct.schedule.38
  br label %direct.schedule.39

direct.false264:                                  ; preds = %direct.schedule.38
  %630 = load <8 x float>, ptr %.slot61, align 32
  %631 = select <8 x i1> %62, <8 x float> zeroinitializer, <8 x float> %630
  store <8 x float> %631, ptr %.slot61, align 32
  br label %direct.schedule.40

direct.true270:                                   ; preds = %direct.schedule.40
  br label %direct.schedule.41

direct.false271:                                  ; preds = %direct.schedule.40
  %632 = load <8 x float>, ptr %.slot62, align 32
  %633 = select <8 x i1> %62, <8 x float> zeroinitializer, <8 x float> %632
  store <8 x float> %633, ptr %.slot62, align 32
  br label %direct.schedule.42

direct.true281:                                   ; preds = %direct.schedule.43
  br label %direct.schedule.44

direct.false282:                                  ; preds = %direct.schedule.43
  br label %direct.schedule.53

direct.true285:                                   ; preds = %direct.schedule.44
  br label %direct.schedule.45

direct.false286:                                  ; preds = %direct.schedule.44
  %634 = load <8 x float>, ptr %.slot69, align 32
  %635 = select <8 x i1> %62, <8 x float> zeroinitializer, <8 x float> %634
  store <8 x float> %635, ptr %.slot69, align 32
  br label %direct.schedule.46

direct.true295:                                   ; preds = %direct.schedule.46
  br label %direct.schedule.47

direct.false296:                                  ; preds = %direct.schedule.46
  %636 = load <8 x float>, ptr %.slot72, align 32
  %637 = select <8 x i1> %62, <8 x float> zeroinitializer, <8 x float> %636
  store <8 x float> %637, ptr %.slot72, align 32
  br label %direct.schedule.48

direct.true305:                                   ; preds = %direct.schedule.48
  br label %direct.schedule.49

direct.false306:                                  ; preds = %direct.schedule.48
  %638 = load <8 x float>, ptr %.slot75, align 32
  %639 = select <8 x i1> %62, <8 x float> zeroinitializer, <8 x float> %638
  store <8 x float> %639, ptr %.slot75, align 32
  br label %direct.schedule.50

direct.true315:                                   ; preds = %direct.schedule.50
  br label %direct.schedule.51

direct.false316:                                  ; preds = %direct.schedule.50
  %640 = load <8 x float>, ptr %.slot78, align 32
  %641 = select <8 x i1> %62, <8 x float> zeroinitializer, <8 x float> %640
  store <8 x float> %641, ptr %.slot78, align 32
  br label %direct.schedule.52

direct.true335:                                   ; preds = %direct.schedule.54
  br label %direct.schedule.55

direct.false336:                                  ; preds = %direct.schedule.54
  br label %direct.schedule.56
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.vector.reduce.or.v8i1(<8 x i1>) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr>, i32 immarg, <8 x i1>, <8 x float>) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.masked.scatter.v8f32.v8p0(<8 x float>, <8 x ptr>, i32 immarg, <8 x i1>) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.masked.scatter.v8i64.v8p0(<8 x i64>, <8 x ptr>, i32 immarg, <8 x i1>) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <8 x i64> @llvm.masked.gather.v8i64.v8p0(<8 x ptr>, i32 immarg, <8 x i1>, <8 x i64>) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.masked.scatter.v8i8.v8p0(<8 x i8>, <8 x ptr>, i32 immarg, <8 x i1>) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <8 x i8> @llvm.masked.gather.v8i8.v8p0(<8 x ptr>, i32 immarg, <8 x i1>, <8 x i8>) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.maxnum.v8f32(<8 x float>, <8 x float>) #0

; Function Attrs: alwaysinline norecurse nounwind willreturn memory(none)
define internal <8 x float> @__luisa_cpu_native_exp_f32_v8_u10(<8 x float> %x) #3 {
entry:
  %0 = fcmp oge <8 x float> %x, splat (float -1.040000e+02)
  %1 = fcmp ole <8 x float> %x, splat (float 1.000000e+02)
  %2 = and <8 x i1> %0, %1
  %3 = select <8 x i1> %2, <8 x float> %x, <8 x float> zeroinitializer
  %4 = fmul <8 x float> %3, splat (float 0x3FF7154760000000)
  %5 = bitcast <8 x float> %4 to <8 x i32>
  %6 = and <8 x i32> %5, splat (i32 -2147483648)
  %7 = or <8 x i32> splat (i32 1258291200), %6
  %8 = bitcast <8 x i32> %7 to <8 x float>
  %9 = fadd <8 x float> %4, %8
  %10 = fsub <8 x float> %9, %8
  %11 = fptosi <8 x float> %10 to <8 x i32>
  %12 = sitofp <8 x i32> %11 to <8 x float>
  %13 = fmul <8 x float> %12, splat (float 0xBFE62E4000000000)
  %14 = fadd <8 x float> %13, %3
  %15 = fmul <8 x float> %12, splat (float 0xBEB7F7D1C0000000)
  %16 = fadd <8 x float> %15, %14
  %17 = fmul <8 x float> splat (float 0x3F2A057B40000000), %16
  %18 = fadd <8 x float> %17, splat (float 0x3F56D2D920000000)
  %19 = fmul <8 x float> %18, %16
  %20 = fadd <8 x float> %19, splat (float 0x3F811114C0000000)
  %21 = fmul <8 x float> %20, %16
  %22 = fadd <8 x float> %21, splat (float 0x3FA5554F40000000)
  %23 = fmul <8 x float> %22, %16
  %24 = fadd <8 x float> %23, splat (float 0x3FC5555560000000)
  %25 = fmul <8 x float> %24, %16
  %26 = fadd <8 x float> %25, splat (float 5.000000e-01)
  %27 = fmul <8 x float> %16, %16
  %28 = fmul <8 x float> %27, %26
  %29 = fadd <8 x float> %16, %28
  %30 = fadd <8 x float> splat (float 1.000000e+00), %29
  %31 = ashr <8 x i32> %11, splat (i32 1)
  %32 = add <8 x i32> %31, splat (i32 127)
  %33 = shl <8 x i32> %32, splat (i32 23)
  %34 = bitcast <8 x i32> %33 to <8 x float>
  %35 = fmul <8 x float> %30, %34
  %36 = sub <8 x i32> %11, %31
  %37 = add <8 x i32> %36, splat (i32 127)
  %38 = shl <8 x i32> %37, splat (i32 23)
  %39 = bitcast <8 x i32> %38 to <8 x float>
  %40 = fmul <8 x float> %35, %39
  %41 = fcmp olt <8 x float> %x, splat (float -1.040000e+02)
  %42 = select <8 x i1> %41, <8 x float> zeroinitializer, <8 x float> %40
  %43 = fcmp ogt <8 x float> %x, splat (float 1.000000e+02)
  %44 = select <8 x i1> %43, <8 x float> splat (float 0x7FF0000000000000), <8 x float> %42
  %45 = fcmp uno <8 x float> %x, %x
  %46 = select <8 x i1> %45, <8 x float> splat (float 0x7FF8000000000000), <8 x float> %44
  ret <8 x float> %46
}

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
attributes #3 = { alwaysinline norecurse nounwind willreturn memory(none) "luisa.cpu.native_math" }
