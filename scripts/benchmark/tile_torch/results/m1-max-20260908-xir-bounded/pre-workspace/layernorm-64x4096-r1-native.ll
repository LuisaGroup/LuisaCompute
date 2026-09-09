; ModuleID = 'luisa-simd-kernel'
source_filename = "luisa-simd-kernel"

define internal void @llm_rows(ptr noalias readonly %argument_buffer, ptr %return_lanes, ptr noalias nonnull readonly %launch_config, i32 %active_lane_count) {
prologue:
  %tile_snapshot.local = alloca [131072 x i8], align 4
  %.splatinsert = insertelement <8 x ptr> poison, ptr %tile_snapshot.local, i64 0
  %.splat = shufflevector <8 x ptr> %.splatinsert, <8 x ptr> poison, <8 x i32> zeroinitializer
  %0 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat, 0
  %1 = insertvalue { <8 x ptr>, <8 x i64> } %0, <8 x i64> <i64 0, i64 16384, i64 32768, i64 49152, i64 65536, i64 81920, i64 98304, i64 114688>, 1
  %tile_snapshot.local1 = alloca [131072 x i8], align 4
  %.splatinsert2 = insertelement <8 x ptr> poison, ptr %tile_snapshot.local1, i64 0
  %.splat3 = shufflevector <8 x ptr> %.splatinsert2, <8 x ptr> poison, <8 x i32> zeroinitializer
  %2 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat3, 0
  %3 = insertvalue { <8 x ptr>, <8 x i64> } %2, <8 x i64> <i64 0, i64 16384, i64 32768, i64 49152, i64 65536, i64 81920, i64 98304, i64 114688>, 1
  %tile_snapshot.local4 = alloca [131072 x i8], align 4
  %.splatinsert5 = insertelement <8 x ptr> poison, ptr %tile_snapshot.local4, i64 0
  %.splat6 = shufflevector <8 x ptr> %.splatinsert5, <8 x ptr> poison, <8 x i32> zeroinitializer
  %4 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat6, 0
  %5 = insertvalue { <8 x ptr>, <8 x i64> } %4, <8 x i64> <i64 0, i64 16384, i64 32768, i64 49152, i64 65536, i64 81920, i64 98304, i64 114688>, 1
  %tile_snapshot.local7 = alloca [131072 x i8], align 4
  %.splatinsert8 = insertelement <8 x ptr> poison, ptr %tile_snapshot.local7, i64 0
  %.splat9 = shufflevector <8 x ptr> %.splatinsert8, <8 x ptr> poison, <8 x i32> zeroinitializer
  %6 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat9, 0
  %7 = insertvalue { <8 x ptr>, <8 x i64> } %6, <8 x i64> <i64 0, i64 16384, i64 32768, i64 49152, i64 65536, i64 81920, i64 98304, i64 114688>, 1
  %.spill = alloca <8 x i64>, align 64
  store <8 x i64> zeroinitializer, ptr %.spill, align 64
  %tile_snapshot.spill = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill, align 64
  %.slot = alloca i64, align 8
  store i64 0, ptr %.slot, align 4
  %.spill10 = alloca float, align 4
  store float 0.000000e+00, ptr %.spill10, align 4
  %.spill11 = alloca i64, align 8
  store i64 0, ptr %.spill11, align 4
  %.spill12 = alloca i64, align 8
  store i64 0, ptr %.spill12, align 4
  %.spill13 = alloca i64, align 8
  store i64 0, ptr %.spill13, align 4
  %.spill14 = alloca i1, align 1
  store i1 false, ptr %.spill14, align 1
  %.slot15 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot15, align 32
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
  %.slot25 = alloca i64, align 8
  store i64 0, ptr %.slot25, align 4
  %.slot26 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot26, align 32
  %.slot27 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot27, align 32
  %.slot28 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot28, align 32
  %.slot29 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot29, align 32
  %.spill30 = alloca i64, align 8
  store i64 0, ptr %.spill30, align 4
  %.slot31 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot31, align 32
  %.spill32 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill32, align 32
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
  %.spill41 = alloca float, align 4
  store float 0.000000e+00, ptr %.spill41, align 4
  %.spill42 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill42, align 32
  %tile_snapshot.spill43 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill43, align 64
  %.slot44 = alloca i64, align 8
  store i64 0, ptr %.slot44, align 4
  %.slot45 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot45, align 32
  %.slot46 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot46, align 32
  %.slot47 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot47, align 32
  %.slot48 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot48, align 32
  %.slot49 = alloca i64, align 8
  store i64 0, ptr %.slot49, align 4
  %.slot50 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot50, align 32
  %.slot51 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot51, align 32
  %.slot52 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot52, align 32
  %.slot53 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot53, align 32
  %.spill54 = alloca i64, align 8
  store i64 0, ptr %.spill54, align 4
  %.slot55 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot55, align 32
  %.spill56 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill56, align 32
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
  %tile_snapshot.spill66 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill66, align 64
  %.slot67 = alloca i64, align 8
  store i64 0, ptr %.slot67, align 4
  %.spill68 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill68, align 32
  %tile_snapshot.spill69 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill69, align 64
  %.slot70 = alloca i64, align 8
  store i64 0, ptr %.slot70, align 4
  %.slot71 = alloca i64, align 8
  store i64 0, ptr %.slot71, align 4
  %8 = getelementptr i8, ptr %argument_buffer, i64 0
  %9 = load ptr, ptr %8, align 16
  %10 = getelementptr i8, ptr %8, i64 8
  %11 = load i64, ptr %10, align 8
  %12 = insertvalue { ptr, i64 } poison, ptr %9, 0
  %13 = insertvalue { ptr, i64 } %12, i64 %11, 1
  %14 = getelementptr i8, ptr %argument_buffer, i64 16
  %15 = load ptr, ptr %14, align 16
  %16 = getelementptr i8, ptr %14, i64 8
  %17 = load i64, ptr %16, align 8
  %18 = insertvalue { ptr, i64 } poison, ptr %15, 0
  %19 = insertvalue { ptr, i64 } %18, i64 %17, 1
  %20 = getelementptr i8, ptr %argument_buffer, i64 32
  %21 = load ptr, ptr %20, align 16
  %22 = getelementptr i8, ptr %20, i64 8
  %23 = load i64, ptr %22, align 8
  %24 = insertvalue { ptr, i64 } poison, ptr %21, 0
  %25 = insertvalue { ptr, i64 } %24, i64 %23, 1
  %26 = getelementptr i8, ptr %argument_buffer, i64 48
  %27 = load ptr, ptr %26, align 16
  %28 = getelementptr i8, ptr %26, i64 8
  %29 = load i64, ptr %28, align 8
  %30 = insertvalue { ptr, i64 } poison, ptr %27, 0
  %31 = insertvalue { ptr, i64 } %30, i64 %29, 1
  %32 = load i32, ptr %launch_config, align 4
  %33 = getelementptr i8, ptr %launch_config, i64 12
  %34 = load i32, ptr %33, align 4
  %35 = getelementptr i8, ptr %launch_config, i64 4
  %36 = load i32, ptr %35, align 4
  %37 = getelementptr i8, ptr %launch_config, i64 16
  %38 = load i32, ptr %37, align 4
  %39 = getelementptr i8, ptr %launch_config, i64 8
  %40 = load i32, ptr %39, align 4
  %41 = getelementptr i8, ptr %launch_config, i64 20
  %42 = load i32, ptr %41, align 4
  %43 = getelementptr i8, ptr %launch_config, i64 36
  %44 = load i32, ptr %43, align 4
  %.splatinsert72 = insertelement <8 x i32> poison, i32 %44, i64 0
  %.splat73 = shufflevector <8 x i32> %.splatinsert72, <8 x i32> poison, <8 x i32> zeroinitializer
  %45 = add <8 x i32> %.splat73, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %46 = mul i32 %32, 32
  %.splatinsert74 = insertelement <8 x i32> poison, i32 %46, i64 0
  %.splat75 = shufflevector <8 x i32> %.splatinsert74, <8 x i32> poison, <8 x i32> zeroinitializer
  %47 = add <8 x i32> %.splat75, %45
  %48 = mul i32 %36, 1
  %.splatinsert76 = insertelement <8 x i32> poison, i32 %48, i64 0
  %.splat77 = shufflevector <8 x i32> %.splatinsert76, <8 x i32> poison, <8 x i32> zeroinitializer
  %49 = add <8 x i32> %.splat77, zeroinitializer
  %50 = mul i32 %40, 1
  %.splatinsert78 = insertelement <8 x i32> poison, i32 %50, i64 0
  %.splat79 = shufflevector <8 x i32> %.splatinsert78, <8 x i32> poison, <8 x i32> zeroinitializer
  %51 = add <8 x i32> %.splat79, zeroinitializer
  %52 = insertvalue [3 x <8 x i32>] poison, <8 x i32> %47, 0
  %53 = insertvalue [3 x <8 x i32>] %52, <8 x i32> %49, 1
  %54 = insertvalue [3 x <8 x i32>] %53, <8 x i32> %51, 2
  %.splatinsert80 = insertelement <8 x i32> poison, i32 %active_lane_count, i64 0
  %.splat81 = shufflevector <8 x i32> %.splatinsert80, <8 x i32> poison, <8 x i32> zeroinitializer
  %55 = icmp ult <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, %.splat81
  %56 = call i1 @llvm.vector.reduce.or.v8i1(<8 x i1> %55)
  br i1 %56, label %direct.activate, label %direct.inactive

direct.schedule.0:                                ; preds = %direct.activate
  %57 = extractvalue [3 x <8 x i32>] %54, 0
  %58 = zext <8 x i32> %57 to <8 x i64>
  %59 = select <8 x i1> %55, <8 x i64> %58, <8 x i64> zeroinitializer
  %60 = select <8 x i1> %55, <8 x i64> splat (i64 1), <8 x i64> splat (i64 1)
  %61 = sdiv <8 x i64> %59, %60
  %62 = select <8 x i1> %55, <8 x i64> %61, <8 x i64> zeroinitializer
  %63 = select <8 x i1> %55, <8 x i64> splat (i64 64), <8 x i64> splat (i64 1)
  %64 = srem <8 x i64> %62, %63
  %65 = load <8 x i64>, ptr %.spill, align 64
  %66 = select <8 x i1> %55, <8 x i64> %64, <8 x i64> %65
  store <8 x i64> %66, ptr %.spill, align 64
  %67 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %68 = extractvalue { <8 x ptr>, <8 x i64> } %1, 0
  %69 = extractvalue { <8 x ptr>, <8 x i64> } %67, 0
  %70 = select <8 x i1> %55, <8 x ptr> %68, <8 x ptr> %69
  %71 = extractvalue { <8 x ptr>, <8 x i64> } %1, 1
  %72 = extractvalue { <8 x ptr>, <8 x i64> } %67, 1
  %73 = select <8 x i1> %55, <8 x i64> %71, <8 x i64> %72
  %74 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %70, 0
  %75 = insertvalue { <8 x ptr>, <8 x i64> } %74, <8 x i64> %73, 1
  store { <8 x ptr>, <8 x i64> } %75, ptr %tile_snapshot.spill, align 64
  store i64 0, ptr %.slot, align 4
  br label %direct.schedule.1

direct.schedule.1:                                ; preds = %direct.schedule.2, %direct.schedule.0
  %.state = load i64, ptr %.slot, align 4
  %76 = icmp slt i64 %.state, 4096
  br i1 %76, label %direct.true, label %direct.false

direct.schedule.2:                                ; preds = %direct.true
  %.state82 = load i64, ptr %.slot, align 4
  %77 = srem i64 %.state82, 4096
  %.state83 = load i64, ptr %.slot, align 4
  %78 = sdiv i64 %.state83, 4096
  %79 = srem i64 %78, 1
  %.spill.load = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert84 = insertelement <8 x i64> poison, i64 %79, i64 0
  %.splat85 = shufflevector <8 x i64> %.splatinsert84, <8 x i64> poison, <8 x i32> zeroinitializer
  %80 = add <8 x i64> %.spill.load, %.splat85
  %81 = add <8 x i64> zeroinitializer, %80
  %82 = add i64 0, %77
  %83 = mul <8 x i64> %81, splat (i64 4096)
  %.splatinsert86 = insertelement <8 x i64> poison, i64 %82, i64 0
  %.splat87 = shufflevector <8 x i64> %.splatinsert86, <8 x i64> poison, <8 x i32> zeroinitializer
  %84 = add <8 x i64> %83, %.splat87
  %85 = extractvalue { ptr, i64 } %13, 0
  %86 = mul <8 x i64> %84, splat (i64 4)
  %87 = getelementptr i8, ptr %85, <8 x i64> %86
  %88 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %87, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %89 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 0
  %90 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 1
  %.state88 = load i64, ptr %.slot, align 4
  %.splatinsert89 = insertelement <8 x i64> poison, i64 %.state88, i64 0
  %.splat90 = shufflevector <8 x i64> %.splatinsert89, <8 x i64> poison, <8 x i32> zeroinitializer
  %91 = mul <8 x i64> %.splat90, splat (i64 4)
  %92 = add <8 x i64> %90, %91
  %93 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %89, 0
  %94 = insertvalue { <8 x ptr>, <8 x i64> } %93, <8 x i64> %92, 1
  %95 = extractvalue { <8 x ptr>, <8 x i64> } %94, 0
  %96 = extractvalue { <8 x ptr>, <8 x i64> } %94, 1
  %97 = getelementptr i8, <8 x ptr> %95, <8 x i64> %96
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %88, <8 x ptr> %97, i32 1, <8 x i1> %55)
  %.state91 = load i64, ptr %.slot, align 4
  %98 = add i64 %.state91, 1
  store i64 %98, ptr %.slot, align 4
  br label %direct.schedule.1

direct.schedule.3:                                ; preds = %direct.false
  store float 0.000000e+00, ptr %.spill10, align 4
  store i64 0, ptr %.spill11, align 4
  store i64 0, ptr %.spill12, align 4
  store i64 0, ptr %.spill13, align 4
  store i1 true, ptr %.spill14, align 1
  br i1 true, label %direct.true92, label %direct.false93

direct.schedule.4:                                ; preds = %direct.true92
  %tile_snapshot.spill.load94 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %99 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load94, 0
  %100 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load94, 1
  %.spill.load95 = load i64, ptr %.spill13, align 4
  %.splatinsert96 = insertelement <8 x i64> poison, i64 %.spill.load95, i64 0
  %.splat97 = shufflevector <8 x i64> %.splatinsert96, <8 x i64> poison, <8 x i32> zeroinitializer
  %101 = mul <8 x i64> %.splat97, splat (i64 4)
  %102 = add <8 x i64> %100, %101
  %103 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %99, 0
  %104 = insertvalue { <8 x ptr>, <8 x i64> } %103, <8 x i64> %102, 1
  %105 = extractvalue { <8 x ptr>, <8 x i64> } %104, 0
  %106 = extractvalue { <8 x ptr>, <8 x i64> } %104, 1
  %107 = getelementptr i8, <8 x ptr> %105, <8 x i64> %106
  %108 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %107, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %109 = load <8 x float>, ptr %.slot15, align 32
  %110 = select <8 x i1> %55, <8 x float> %108, <8 x float> %109
  store <8 x float> %110, ptr %.slot15, align 32
  br label %direct.schedule.5

direct.schedule.5:                                ; preds = %direct.schedule.4, %direct.false93
  %.spill.load98 = load i64, ptr %.spill12, align 4
  %111 = add i64 %.spill.load98, 1
  store i64 %111, ptr %.spill16, align 4
  %112 = icmp sge i64 %111, 0
  %113 = icmp slt i64 %111, 4096
  %114 = and i1 %112, %113
  store i1 %114, ptr %.spill17, align 1
  br i1 %114, label %direct.true99, label %direct.false100

direct.schedule.6:                                ; preds = %direct.true99
  %tile_snapshot.spill.load101 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %115 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load101, 0
  %116 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load101, 1
  %.spill.load102 = load i64, ptr %.spill16, align 4
  %.splatinsert103 = insertelement <8 x i64> poison, i64 %.spill.load102, i64 0
  %.splat104 = shufflevector <8 x i64> %.splatinsert103, <8 x i64> poison, <8 x i32> zeroinitializer
  %117 = mul <8 x i64> %.splat104, splat (i64 4)
  %118 = add <8 x i64> %116, %117
  %119 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %115, 0
  %120 = insertvalue { <8 x ptr>, <8 x i64> } %119, <8 x i64> %118, 1
  %121 = extractvalue { <8 x ptr>, <8 x i64> } %120, 0
  %122 = extractvalue { <8 x ptr>, <8 x i64> } %120, 1
  %123 = getelementptr i8, <8 x ptr> %121, <8 x i64> %122
  %124 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %123, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %125 = load <8 x float>, ptr %.slot18, align 32
  %126 = select <8 x i1> %55, <8 x float> %124, <8 x float> %125
  store <8 x float> %126, ptr %.slot18, align 32
  br label %direct.schedule.7

direct.schedule.7:                                ; preds = %direct.schedule.6, %direct.false100
  %.spill.load105 = load i64, ptr %.spill12, align 4
  %127 = add i64 %.spill.load105, 2
  store i64 %127, ptr %.spill19, align 4
  %128 = icmp sge i64 %127, 0
  %129 = icmp slt i64 %127, 4096
  %130 = and i1 %128, %129
  store i1 %130, ptr %.spill20, align 1
  br i1 %130, label %direct.true106, label %direct.false107

direct.schedule.8:                                ; preds = %direct.true106
  %tile_snapshot.spill.load108 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %131 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load108, 0
  %132 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load108, 1
  %.spill.load109 = load i64, ptr %.spill19, align 4
  %.splatinsert110 = insertelement <8 x i64> poison, i64 %.spill.load109, i64 0
  %.splat111 = shufflevector <8 x i64> %.splatinsert110, <8 x i64> poison, <8 x i32> zeroinitializer
  %133 = mul <8 x i64> %.splat111, splat (i64 4)
  %134 = add <8 x i64> %132, %133
  %135 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %131, 0
  %136 = insertvalue { <8 x ptr>, <8 x i64> } %135, <8 x i64> %134, 1
  %137 = extractvalue { <8 x ptr>, <8 x i64> } %136, 0
  %138 = extractvalue { <8 x ptr>, <8 x i64> } %136, 1
  %139 = getelementptr i8, <8 x ptr> %137, <8 x i64> %138
  %140 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %139, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %141 = load <8 x float>, ptr %.slot21, align 32
  %142 = select <8 x i1> %55, <8 x float> %140, <8 x float> %141
  store <8 x float> %142, ptr %.slot21, align 32
  br label %direct.schedule.9

direct.schedule.9:                                ; preds = %direct.schedule.8, %direct.false107
  %.spill.load112 = load i64, ptr %.spill12, align 4
  %143 = add i64 %.spill.load112, 3
  store i64 %143, ptr %.spill22, align 4
  %144 = icmp sge i64 %143, 0
  %145 = icmp slt i64 %143, 4096
  %146 = and i1 %144, %145
  store i1 %146, ptr %.spill23, align 1
  br i1 %146, label %direct.true113, label %direct.false114

direct.schedule.10:                               ; preds = %direct.true113
  %tile_snapshot.spill.load115 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %147 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load115, 0
  %148 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load115, 1
  %.spill.load116 = load i64, ptr %.spill22, align 4
  %.splatinsert117 = insertelement <8 x i64> poison, i64 %.spill.load116, i64 0
  %.splat118 = shufflevector <8 x i64> %.splatinsert117, <8 x i64> poison, <8 x i32> zeroinitializer
  %149 = mul <8 x i64> %.splat118, splat (i64 4)
  %150 = add <8 x i64> %148, %149
  %151 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %147, 0
  %152 = insertvalue { <8 x ptr>, <8 x i64> } %151, <8 x i64> %150, 1
  %153 = extractvalue { <8 x ptr>, <8 x i64> } %152, 0
  %154 = extractvalue { <8 x ptr>, <8 x i64> } %152, 1
  %155 = getelementptr i8, <8 x ptr> %153, <8 x i64> %154
  %156 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %155, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %157 = load <8 x float>, ptr %.slot24, align 32
  %158 = select <8 x i1> %55, <8 x float> %156, <8 x float> %157
  store <8 x float> %158, ptr %.slot24, align 32
  br label %direct.schedule.11

direct.schedule.11:                               ; preds = %direct.schedule.10, %direct.false114
  %.state119 = load <8 x float>, ptr %.slot15, align 32
  %.state120 = load <8 x float>, ptr %.slot18, align 32
  %.state121 = load <8 x float>, ptr %.slot21, align 32
  %.state122 = load <8 x float>, ptr %.slot24, align 32
  store i64 4, ptr %.slot25, align 4
  %159 = load <8 x float>, ptr %.slot26, align 32
  %160 = select <8 x i1> %55, <8 x float> %.state119, <8 x float> %159
  store <8 x float> %160, ptr %.slot26, align 32
  %161 = load <8 x float>, ptr %.slot27, align 32
  %162 = select <8 x i1> %55, <8 x float> %.state120, <8 x float> %161
  store <8 x float> %162, ptr %.slot27, align 32
  %163 = load <8 x float>, ptr %.slot28, align 32
  %164 = select <8 x i1> %55, <8 x float> %.state121, <8 x float> %163
  store <8 x float> %164, ptr %.slot28, align 32
  %165 = load <8 x float>, ptr %.slot29, align 32
  %166 = select <8 x i1> %55, <8 x float> %.state122, <8 x float> %165
  store <8 x float> %166, ptr %.slot29, align 32
  br label %direct.schedule.12

direct.schedule.12:                               ; preds = %direct.schedule.21, %direct.schedule.11
  %.state123 = load i64, ptr %.slot25, align 4
  %167 = icmp slt i64 %.state123, 4096
  br i1 %167, label %direct.true124, label %direct.false125

direct.schedule.13:                               ; preds = %direct.true124
  %.state126 = load i64, ptr %.slot25, align 4
  %168 = add i64 %.state126, 0
  %169 = sdiv i64 %168, 1
  %170 = srem i64 %169, 4096
  %.spill.load127 = load i64, ptr %.spill12, align 4
  %171 = add i64 %.spill.load127, %170
  store i64 %171, ptr %.spill30, align 4
  %172 = icmp sge i64 %171, 0
  %173 = icmp slt i64 %171, 4096
  %174 = and i1 %172, %173
  br i1 %174, label %direct.true128, label %direct.false129

direct.schedule.14:                               ; preds = %direct.true128
  %tile_snapshot.spill.load130 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %175 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load130, 0
  %176 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load130, 1
  %.spill.load131 = load i64, ptr %.spill30, align 4
  %.splatinsert132 = insertelement <8 x i64> poison, i64 %.spill.load131, i64 0
  %.splat133 = shufflevector <8 x i64> %.splatinsert132, <8 x i64> poison, <8 x i32> zeroinitializer
  %177 = mul <8 x i64> %.splat133, splat (i64 4)
  %178 = add <8 x i64> %176, %177
  %179 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %175, 0
  %180 = insertvalue { <8 x ptr>, <8 x i64> } %179, <8 x i64> %178, 1
  %181 = extractvalue { <8 x ptr>, <8 x i64> } %180, 0
  %182 = extractvalue { <8 x ptr>, <8 x i64> } %180, 1
  %183 = getelementptr i8, <8 x ptr> %181, <8 x i64> %182
  %184 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %183, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %185 = load <8 x float>, ptr %.slot31, align 32
  %186 = select <8 x i1> %55, <8 x float> %184, <8 x float> %185
  store <8 x float> %186, ptr %.slot31, align 32
  br label %direct.schedule.15

direct.schedule.15:                               ; preds = %direct.schedule.14, %direct.false129
  %.state134 = load <8 x float>, ptr %.slot26, align 32
  %.state135 = load <8 x float>, ptr %.slot31, align 32
  %187 = fadd <8 x float> %.state134, %.state135
  %188 = load <8 x float>, ptr %.spill32, align 32
  %189 = select <8 x i1> %55, <8 x float> %187, <8 x float> %188
  store <8 x float> %189, ptr %.spill32, align 32
  %.state136 = load i64, ptr %.slot25, align 4
  %190 = add i64 %.state136, 1
  %191 = sdiv i64 %190, 1
  %192 = srem i64 %191, 4096
  %.spill.load137 = load i64, ptr %.spill12, align 4
  %193 = add i64 %.spill.load137, %192
  store i64 %193, ptr %.spill33, align 4
  %194 = icmp sge i64 %193, 0
  %195 = icmp slt i64 %193, 4096
  %196 = and i1 %194, %195
  br i1 %196, label %direct.true138, label %direct.false139

direct.schedule.16:                               ; preds = %direct.true138
  %tile_snapshot.spill.load140 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %197 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load140, 0
  %198 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load140, 1
  %.spill.load141 = load i64, ptr %.spill33, align 4
  %.splatinsert142 = insertelement <8 x i64> poison, i64 %.spill.load141, i64 0
  %.splat143 = shufflevector <8 x i64> %.splatinsert142, <8 x i64> poison, <8 x i32> zeroinitializer
  %199 = mul <8 x i64> %.splat143, splat (i64 4)
  %200 = add <8 x i64> %198, %199
  %201 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %197, 0
  %202 = insertvalue { <8 x ptr>, <8 x i64> } %201, <8 x i64> %200, 1
  %203 = extractvalue { <8 x ptr>, <8 x i64> } %202, 0
  %204 = extractvalue { <8 x ptr>, <8 x i64> } %202, 1
  %205 = getelementptr i8, <8 x ptr> %203, <8 x i64> %204
  %206 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %205, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %207 = load <8 x float>, ptr %.slot34, align 32
  %208 = select <8 x i1> %55, <8 x float> %206, <8 x float> %207
  store <8 x float> %208, ptr %.slot34, align 32
  br label %direct.schedule.17

direct.schedule.17:                               ; preds = %direct.schedule.16, %direct.false139
  %.state144 = load <8 x float>, ptr %.slot27, align 32
  %.state145 = load <8 x float>, ptr %.slot34, align 32
  %209 = fadd <8 x float> %.state144, %.state145
  %210 = load <8 x float>, ptr %.spill35, align 32
  %211 = select <8 x i1> %55, <8 x float> %209, <8 x float> %210
  store <8 x float> %211, ptr %.spill35, align 32
  %.state146 = load i64, ptr %.slot25, align 4
  %212 = add i64 %.state146, 2
  %213 = sdiv i64 %212, 1
  %214 = srem i64 %213, 4096
  %.spill.load147 = load i64, ptr %.spill12, align 4
  %215 = add i64 %.spill.load147, %214
  store i64 %215, ptr %.spill36, align 4
  %216 = icmp sge i64 %215, 0
  %217 = icmp slt i64 %215, 4096
  %218 = and i1 %216, %217
  br i1 %218, label %direct.true148, label %direct.false149

direct.schedule.18:                               ; preds = %direct.true148
  %tile_snapshot.spill.load150 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %219 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load150, 0
  %220 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load150, 1
  %.spill.load151 = load i64, ptr %.spill36, align 4
  %.splatinsert152 = insertelement <8 x i64> poison, i64 %.spill.load151, i64 0
  %.splat153 = shufflevector <8 x i64> %.splatinsert152, <8 x i64> poison, <8 x i32> zeroinitializer
  %221 = mul <8 x i64> %.splat153, splat (i64 4)
  %222 = add <8 x i64> %220, %221
  %223 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %219, 0
  %224 = insertvalue { <8 x ptr>, <8 x i64> } %223, <8 x i64> %222, 1
  %225 = extractvalue { <8 x ptr>, <8 x i64> } %224, 0
  %226 = extractvalue { <8 x ptr>, <8 x i64> } %224, 1
  %227 = getelementptr i8, <8 x ptr> %225, <8 x i64> %226
  %228 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %227, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %229 = load <8 x float>, ptr %.slot37, align 32
  %230 = select <8 x i1> %55, <8 x float> %228, <8 x float> %229
  store <8 x float> %230, ptr %.slot37, align 32
  br label %direct.schedule.19

direct.schedule.19:                               ; preds = %direct.schedule.18, %direct.false149
  %.state154 = load <8 x float>, ptr %.slot28, align 32
  %.state155 = load <8 x float>, ptr %.slot37, align 32
  %231 = fadd <8 x float> %.state154, %.state155
  %232 = load <8 x float>, ptr %.spill38, align 32
  %233 = select <8 x i1> %55, <8 x float> %231, <8 x float> %232
  store <8 x float> %233, ptr %.spill38, align 32
  %.state156 = load i64, ptr %.slot25, align 4
  %234 = add i64 %.state156, 3
  %235 = sdiv i64 %234, 1
  %236 = srem i64 %235, 4096
  %.spill.load157 = load i64, ptr %.spill12, align 4
  %237 = add i64 %.spill.load157, %236
  store i64 %237, ptr %.spill39, align 4
  %238 = icmp sge i64 %237, 0
  %239 = icmp slt i64 %237, 4096
  %240 = and i1 %238, %239
  br i1 %240, label %direct.true158, label %direct.false159

direct.schedule.20:                               ; preds = %direct.true158
  %tile_snapshot.spill.load160 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %241 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load160, 0
  %242 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load160, 1
  %.spill.load161 = load i64, ptr %.spill39, align 4
  %.splatinsert162 = insertelement <8 x i64> poison, i64 %.spill.load161, i64 0
  %.splat163 = shufflevector <8 x i64> %.splatinsert162, <8 x i64> poison, <8 x i32> zeroinitializer
  %243 = mul <8 x i64> %.splat163, splat (i64 4)
  %244 = add <8 x i64> %242, %243
  %245 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %241, 0
  %246 = insertvalue { <8 x ptr>, <8 x i64> } %245, <8 x i64> %244, 1
  %247 = extractvalue { <8 x ptr>, <8 x i64> } %246, 0
  %248 = extractvalue { <8 x ptr>, <8 x i64> } %246, 1
  %249 = getelementptr i8, <8 x ptr> %247, <8 x i64> %248
  %250 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %249, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %251 = load <8 x float>, ptr %.slot40, align 32
  %252 = select <8 x i1> %55, <8 x float> %250, <8 x float> %251
  store <8 x float> %252, ptr %.slot40, align 32
  br label %direct.schedule.21

direct.schedule.21:                               ; preds = %direct.schedule.20, %direct.false159
  %.state164 = load <8 x float>, ptr %.slot29, align 32
  %.state165 = load <8 x float>, ptr %.slot40, align 32
  %253 = fadd <8 x float> %.state164, %.state165
  %.state166 = load i64, ptr %.slot25, align 4
  %254 = add i64 %.state166, 4
  %.spill.load167 = load <8 x float>, ptr %.spill32, align 32
  %.spill.load168 = load <8 x float>, ptr %.spill35, align 32
  %.spill.load169 = load <8 x float>, ptr %.spill38, align 32
  store i64 %254, ptr %.slot25, align 4
  %255 = load <8 x float>, ptr %.slot26, align 32
  %256 = select <8 x i1> %55, <8 x float> %.spill.load167, <8 x float> %255
  store <8 x float> %256, ptr %.slot26, align 32
  %257 = load <8 x float>, ptr %.slot27, align 32
  %258 = select <8 x i1> %55, <8 x float> %.spill.load168, <8 x float> %257
  store <8 x float> %258, ptr %.slot27, align 32
  %259 = load <8 x float>, ptr %.slot28, align 32
  %260 = select <8 x i1> %55, <8 x float> %.spill.load169, <8 x float> %259
  store <8 x float> %260, ptr %.slot28, align 32
  %261 = load <8 x float>, ptr %.slot29, align 32
  %262 = select <8 x i1> %55, <8 x float> %253, <8 x float> %261
  store <8 x float> %262, ptr %.slot29, align 32
  br label %direct.schedule.12

direct.schedule.22:                               ; preds = %direct.false125
  %.spill.load170 = load float, ptr %.spill10, align 4
  %.splatinsert171 = insertelement <8 x float> poison, float %.spill.load170, i64 0
  %.splat172 = shufflevector <8 x float> %.splatinsert171, <8 x float> poison, <8 x i32> zeroinitializer
  %.state173 = load <8 x float>, ptr %.slot26, align 32
  %263 = fadd <8 x float> %.splat172, %.state173
  %.state174 = load <8 x float>, ptr %.slot27, align 32
  %264 = fadd <8 x float> %263, %.state174
  %.state175 = load <8 x float>, ptr %.slot28, align 32
  %265 = fadd <8 x float> %264, %.state175
  %.state176 = load <8 x float>, ptr %.slot29, align 32
  %266 = fadd <8 x float> %265, %.state176
  store float 4.096000e+03, ptr %.spill41, align 4
  %267 = fdiv <8 x float> %266, splat (float 4.096000e+03)
  %268 = load <8 x float>, ptr %.spill42, align 32
  %269 = select <8 x i1> %55, <8 x float> %267, <8 x float> %268
  store <8 x float> %269, ptr %.spill42, align 32
  %270 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill43, align 64
  %271 = extractvalue { <8 x ptr>, <8 x i64> } %3, 0
  %272 = extractvalue { <8 x ptr>, <8 x i64> } %270, 0
  %273 = select <8 x i1> %55, <8 x ptr> %271, <8 x ptr> %272
  %274 = extractvalue { <8 x ptr>, <8 x i64> } %3, 1
  %275 = extractvalue { <8 x ptr>, <8 x i64> } %270, 1
  %276 = select <8 x i1> %55, <8 x i64> %274, <8 x i64> %275
  %277 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %273, 0
  %278 = insertvalue { <8 x ptr>, <8 x i64> } %277, <8 x i64> %276, 1
  store { <8 x ptr>, <8 x i64> } %278, ptr %tile_snapshot.spill43, align 64
  store i64 0, ptr %.slot44, align 4
  br label %direct.schedule.23

direct.schedule.23:                               ; preds = %direct.schedule.24, %direct.schedule.22
  %.state177 = load i64, ptr %.slot44, align 4
  %279 = icmp slt i64 %.state177, 4096
  br i1 %279, label %direct.true178, label %direct.false179

direct.schedule.24:                               ; preds = %direct.true178
  %.state180 = load i64, ptr %.slot44, align 4
  %280 = srem i64 %.state180, 4096
  %281 = add i64 0, %280
  %tile_snapshot.spill.load181 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %282 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load181, 0
  %283 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load181, 1
  %.splatinsert182 = insertelement <8 x i64> poison, i64 %281, i64 0
  %.splat183 = shufflevector <8 x i64> %.splatinsert182, <8 x i64> poison, <8 x i32> zeroinitializer
  %284 = mul <8 x i64> %.splat183, splat (i64 4)
  %285 = add <8 x i64> %283, %284
  %286 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %282, 0
  %287 = insertvalue { <8 x ptr>, <8 x i64> } %286, <8 x i64> %285, 1
  %288 = extractvalue { <8 x ptr>, <8 x i64> } %287, 0
  %289 = extractvalue { <8 x ptr>, <8 x i64> } %287, 1
  %290 = getelementptr i8, <8 x ptr> %288, <8 x i64> %289
  %291 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %290, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %.spill.load184 = load <8 x float>, ptr %.spill42, align 32
  %292 = fsub <8 x float> %291, %.spill.load184
  %tile_snapshot.spill.load185 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill43, align 64
  %293 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load185, 0
  %294 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load185, 1
  %.state186 = load i64, ptr %.slot44, align 4
  %.splatinsert187 = insertelement <8 x i64> poison, i64 %.state186, i64 0
  %.splat188 = shufflevector <8 x i64> %.splatinsert187, <8 x i64> poison, <8 x i32> zeroinitializer
  %295 = mul <8 x i64> %.splat188, splat (i64 4)
  %296 = add <8 x i64> %294, %295
  %297 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %293, 0
  %298 = insertvalue { <8 x ptr>, <8 x i64> } %297, <8 x i64> %296, 1
  %299 = extractvalue { <8 x ptr>, <8 x i64> } %298, 0
  %300 = extractvalue { <8 x ptr>, <8 x i64> } %298, 1
  %301 = getelementptr i8, <8 x ptr> %299, <8 x i64> %300
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %292, <8 x ptr> %301, i32 1, <8 x i1> %55)
  %.state189 = load i64, ptr %.slot44, align 4
  %302 = add i64 %.state189, 1
  store i64 %302, ptr %.slot44, align 4
  br label %direct.schedule.23

direct.schedule.25:                               ; preds = %direct.false179
  %.spill.load190 = load i1, ptr %.spill14, align 1
  br i1 %.spill.load190, label %direct.true191, label %direct.false192

direct.schedule.26:                               ; preds = %direct.true191
  %.spill.load193 = load i64, ptr %.spill13, align 4
  %303 = srem i64 %.spill.load193, 4096
  %304 = add i64 0, %303
  %tile_snapshot.spill.load194 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill43, align 64
  %305 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load194, 0
  %306 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load194, 1
  %.splatinsert195 = insertelement <8 x i64> poison, i64 %304, i64 0
  %.splat196 = shufflevector <8 x i64> %.splatinsert195, <8 x i64> poison, <8 x i32> zeroinitializer
  %307 = mul <8 x i64> %.splat196, splat (i64 4)
  %308 = add <8 x i64> %306, %307
  %309 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %305, 0
  %310 = insertvalue { <8 x ptr>, <8 x i64> } %309, <8 x i64> %308, 1
  %311 = extractvalue { <8 x ptr>, <8 x i64> } %310, 0
  %312 = extractvalue { <8 x ptr>, <8 x i64> } %310, 1
  %313 = getelementptr i8, <8 x ptr> %311, <8 x i64> %312
  %314 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %313, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %315 = fmul <8 x float> %314, %314
  %316 = load <8 x float>, ptr %.slot45, align 32
  %317 = select <8 x i1> %55, <8 x float> %315, <8 x float> %316
  store <8 x float> %317, ptr %.slot45, align 32
  br label %direct.schedule.27

direct.schedule.27:                               ; preds = %direct.schedule.26, %direct.false192
  %.spill.load197 = load i1, ptr %.spill17, align 1
  br i1 %.spill.load197, label %direct.true198, label %direct.false199

direct.schedule.28:                               ; preds = %direct.true198
  %.spill.load200 = load i64, ptr %.spill16, align 4
  %318 = srem i64 %.spill.load200, 4096
  %319 = add i64 0, %318
  %tile_snapshot.spill.load201 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill43, align 64
  %320 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load201, 0
  %321 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load201, 1
  %.splatinsert202 = insertelement <8 x i64> poison, i64 %319, i64 0
  %.splat203 = shufflevector <8 x i64> %.splatinsert202, <8 x i64> poison, <8 x i32> zeroinitializer
  %322 = mul <8 x i64> %.splat203, splat (i64 4)
  %323 = add <8 x i64> %321, %322
  %324 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %320, 0
  %325 = insertvalue { <8 x ptr>, <8 x i64> } %324, <8 x i64> %323, 1
  %326 = extractvalue { <8 x ptr>, <8 x i64> } %325, 0
  %327 = extractvalue { <8 x ptr>, <8 x i64> } %325, 1
  %328 = getelementptr i8, <8 x ptr> %326, <8 x i64> %327
  %329 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %328, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %330 = fmul <8 x float> %329, %329
  %331 = load <8 x float>, ptr %.slot46, align 32
  %332 = select <8 x i1> %55, <8 x float> %330, <8 x float> %331
  store <8 x float> %332, ptr %.slot46, align 32
  br label %direct.schedule.29

direct.schedule.29:                               ; preds = %direct.schedule.28, %direct.false199
  %.spill.load204 = load i1, ptr %.spill20, align 1
  br i1 %.spill.load204, label %direct.true205, label %direct.false206

direct.schedule.30:                               ; preds = %direct.true205
  %.spill.load207 = load i64, ptr %.spill19, align 4
  %333 = srem i64 %.spill.load207, 4096
  %334 = add i64 0, %333
  %tile_snapshot.spill.load208 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill43, align 64
  %335 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load208, 0
  %336 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load208, 1
  %.splatinsert209 = insertelement <8 x i64> poison, i64 %334, i64 0
  %.splat210 = shufflevector <8 x i64> %.splatinsert209, <8 x i64> poison, <8 x i32> zeroinitializer
  %337 = mul <8 x i64> %.splat210, splat (i64 4)
  %338 = add <8 x i64> %336, %337
  %339 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %335, 0
  %340 = insertvalue { <8 x ptr>, <8 x i64> } %339, <8 x i64> %338, 1
  %341 = extractvalue { <8 x ptr>, <8 x i64> } %340, 0
  %342 = extractvalue { <8 x ptr>, <8 x i64> } %340, 1
  %343 = getelementptr i8, <8 x ptr> %341, <8 x i64> %342
  %344 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %343, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %345 = fmul <8 x float> %344, %344
  %346 = load <8 x float>, ptr %.slot47, align 32
  %347 = select <8 x i1> %55, <8 x float> %345, <8 x float> %346
  store <8 x float> %347, ptr %.slot47, align 32
  br label %direct.schedule.31

direct.schedule.31:                               ; preds = %direct.schedule.30, %direct.false206
  %.spill.load211 = load i1, ptr %.spill23, align 1
  br i1 %.spill.load211, label %direct.true212, label %direct.false213

direct.schedule.32:                               ; preds = %direct.true212
  %.spill.load214 = load i64, ptr %.spill22, align 4
  %348 = srem i64 %.spill.load214, 4096
  %349 = add i64 0, %348
  %tile_snapshot.spill.load215 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill43, align 64
  %350 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load215, 0
  %351 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load215, 1
  %.splatinsert216 = insertelement <8 x i64> poison, i64 %349, i64 0
  %.splat217 = shufflevector <8 x i64> %.splatinsert216, <8 x i64> poison, <8 x i32> zeroinitializer
  %352 = mul <8 x i64> %.splat217, splat (i64 4)
  %353 = add <8 x i64> %351, %352
  %354 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %350, 0
  %355 = insertvalue { <8 x ptr>, <8 x i64> } %354, <8 x i64> %353, 1
  %356 = extractvalue { <8 x ptr>, <8 x i64> } %355, 0
  %357 = extractvalue { <8 x ptr>, <8 x i64> } %355, 1
  %358 = getelementptr i8, <8 x ptr> %356, <8 x i64> %357
  %359 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %358, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %360 = fmul <8 x float> %359, %359
  %361 = load <8 x float>, ptr %.slot48, align 32
  %362 = select <8 x i1> %55, <8 x float> %360, <8 x float> %361
  store <8 x float> %362, ptr %.slot48, align 32
  br label %direct.schedule.33

direct.schedule.33:                               ; preds = %direct.schedule.32, %direct.false213
  %.state218 = load <8 x float>, ptr %.slot45, align 32
  %.state219 = load <8 x float>, ptr %.slot46, align 32
  %.state220 = load <8 x float>, ptr %.slot47, align 32
  %.state221 = load <8 x float>, ptr %.slot48, align 32
  store i64 4, ptr %.slot49, align 4
  %363 = load <8 x float>, ptr %.slot50, align 32
  %364 = select <8 x i1> %55, <8 x float> %.state218, <8 x float> %363
  store <8 x float> %364, ptr %.slot50, align 32
  %365 = load <8 x float>, ptr %.slot51, align 32
  %366 = select <8 x i1> %55, <8 x float> %.state219, <8 x float> %365
  store <8 x float> %366, ptr %.slot51, align 32
  %367 = load <8 x float>, ptr %.slot52, align 32
  %368 = select <8 x i1> %55, <8 x float> %.state220, <8 x float> %367
  store <8 x float> %368, ptr %.slot52, align 32
  %369 = load <8 x float>, ptr %.slot53, align 32
  %370 = select <8 x i1> %55, <8 x float> %.state221, <8 x float> %369
  store <8 x float> %370, ptr %.slot53, align 32
  br label %direct.schedule.34

direct.schedule.34:                               ; preds = %direct.schedule.43, %direct.schedule.33
  %.state222 = load i64, ptr %.slot49, align 4
  %371 = icmp slt i64 %.state222, 4096
  br i1 %371, label %direct.true223, label %direct.false224

direct.schedule.35:                               ; preds = %direct.true223
  %.state225 = load i64, ptr %.slot49, align 4
  %372 = add i64 %.state225, 0
  %373 = sdiv i64 %372, 1
  %374 = srem i64 %373, 4096
  %.spill.load226 = load i64, ptr %.spill12, align 4
  %375 = add i64 %.spill.load226, %374
  store i64 %375, ptr %.spill54, align 4
  %376 = icmp sge i64 %375, 0
  %377 = icmp slt i64 %375, 4096
  %378 = and i1 %376, %377
  br i1 %378, label %direct.true227, label %direct.false228

direct.schedule.36:                               ; preds = %direct.true227
  %.spill.load229 = load i64, ptr %.spill54, align 4
  %379 = srem i64 %.spill.load229, 4096
  %380 = add i64 0, %379
  %tile_snapshot.spill.load230 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill43, align 64
  %381 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load230, 0
  %382 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load230, 1
  %.splatinsert231 = insertelement <8 x i64> poison, i64 %380, i64 0
  %.splat232 = shufflevector <8 x i64> %.splatinsert231, <8 x i64> poison, <8 x i32> zeroinitializer
  %383 = mul <8 x i64> %.splat232, splat (i64 4)
  %384 = add <8 x i64> %382, %383
  %385 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %381, 0
  %386 = insertvalue { <8 x ptr>, <8 x i64> } %385, <8 x i64> %384, 1
  %387 = extractvalue { <8 x ptr>, <8 x i64> } %386, 0
  %388 = extractvalue { <8 x ptr>, <8 x i64> } %386, 1
  %389 = getelementptr i8, <8 x ptr> %387, <8 x i64> %388
  %390 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %389, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %391 = fmul <8 x float> %390, %390
  %392 = load <8 x float>, ptr %.slot55, align 32
  %393 = select <8 x i1> %55, <8 x float> %391, <8 x float> %392
  store <8 x float> %393, ptr %.slot55, align 32
  br label %direct.schedule.37

direct.schedule.37:                               ; preds = %direct.schedule.36, %direct.false228
  %.state233 = load <8 x float>, ptr %.slot50, align 32
  %.state234 = load <8 x float>, ptr %.slot55, align 32
  %394 = fadd <8 x float> %.state233, %.state234
  %395 = load <8 x float>, ptr %.spill56, align 32
  %396 = select <8 x i1> %55, <8 x float> %394, <8 x float> %395
  store <8 x float> %396, ptr %.spill56, align 32
  %.state235 = load i64, ptr %.slot49, align 4
  %397 = add i64 %.state235, 1
  %398 = sdiv i64 %397, 1
  %399 = srem i64 %398, 4096
  %.spill.load236 = load i64, ptr %.spill12, align 4
  %400 = add i64 %.spill.load236, %399
  store i64 %400, ptr %.spill57, align 4
  %401 = icmp sge i64 %400, 0
  %402 = icmp slt i64 %400, 4096
  %403 = and i1 %401, %402
  br i1 %403, label %direct.true237, label %direct.false238

direct.schedule.38:                               ; preds = %direct.true237
  %.spill.load239 = load i64, ptr %.spill57, align 4
  %404 = srem i64 %.spill.load239, 4096
  %405 = add i64 0, %404
  %tile_snapshot.spill.load240 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill43, align 64
  %406 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load240, 0
  %407 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load240, 1
  %.splatinsert241 = insertelement <8 x i64> poison, i64 %405, i64 0
  %.splat242 = shufflevector <8 x i64> %.splatinsert241, <8 x i64> poison, <8 x i32> zeroinitializer
  %408 = mul <8 x i64> %.splat242, splat (i64 4)
  %409 = add <8 x i64> %407, %408
  %410 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %406, 0
  %411 = insertvalue { <8 x ptr>, <8 x i64> } %410, <8 x i64> %409, 1
  %412 = extractvalue { <8 x ptr>, <8 x i64> } %411, 0
  %413 = extractvalue { <8 x ptr>, <8 x i64> } %411, 1
  %414 = getelementptr i8, <8 x ptr> %412, <8 x i64> %413
  %415 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %414, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %416 = fmul <8 x float> %415, %415
  %417 = load <8 x float>, ptr %.slot58, align 32
  %418 = select <8 x i1> %55, <8 x float> %416, <8 x float> %417
  store <8 x float> %418, ptr %.slot58, align 32
  br label %direct.schedule.39

direct.schedule.39:                               ; preds = %direct.schedule.38, %direct.false238
  %.state243 = load <8 x float>, ptr %.slot51, align 32
  %.state244 = load <8 x float>, ptr %.slot58, align 32
  %419 = fadd <8 x float> %.state243, %.state244
  %420 = load <8 x float>, ptr %.spill59, align 32
  %421 = select <8 x i1> %55, <8 x float> %419, <8 x float> %420
  store <8 x float> %421, ptr %.spill59, align 32
  %.state245 = load i64, ptr %.slot49, align 4
  %422 = add i64 %.state245, 2
  %423 = sdiv i64 %422, 1
  %424 = srem i64 %423, 4096
  %.spill.load246 = load i64, ptr %.spill12, align 4
  %425 = add i64 %.spill.load246, %424
  store i64 %425, ptr %.spill60, align 4
  %426 = icmp sge i64 %425, 0
  %427 = icmp slt i64 %425, 4096
  %428 = and i1 %426, %427
  br i1 %428, label %direct.true247, label %direct.false248

direct.schedule.40:                               ; preds = %direct.true247
  %.spill.load249 = load i64, ptr %.spill60, align 4
  %429 = srem i64 %.spill.load249, 4096
  %430 = add i64 0, %429
  %tile_snapshot.spill.load250 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill43, align 64
  %431 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load250, 0
  %432 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load250, 1
  %.splatinsert251 = insertelement <8 x i64> poison, i64 %430, i64 0
  %.splat252 = shufflevector <8 x i64> %.splatinsert251, <8 x i64> poison, <8 x i32> zeroinitializer
  %433 = mul <8 x i64> %.splat252, splat (i64 4)
  %434 = add <8 x i64> %432, %433
  %435 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %431, 0
  %436 = insertvalue { <8 x ptr>, <8 x i64> } %435, <8 x i64> %434, 1
  %437 = extractvalue { <8 x ptr>, <8 x i64> } %436, 0
  %438 = extractvalue { <8 x ptr>, <8 x i64> } %436, 1
  %439 = getelementptr i8, <8 x ptr> %437, <8 x i64> %438
  %440 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %439, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %441 = fmul <8 x float> %440, %440
  %442 = load <8 x float>, ptr %.slot61, align 32
  %443 = select <8 x i1> %55, <8 x float> %441, <8 x float> %442
  store <8 x float> %443, ptr %.slot61, align 32
  br label %direct.schedule.41

direct.schedule.41:                               ; preds = %direct.schedule.40, %direct.false248
  %.state253 = load <8 x float>, ptr %.slot52, align 32
  %.state254 = load <8 x float>, ptr %.slot61, align 32
  %444 = fadd <8 x float> %.state253, %.state254
  %445 = load <8 x float>, ptr %.spill62, align 32
  %446 = select <8 x i1> %55, <8 x float> %444, <8 x float> %445
  store <8 x float> %446, ptr %.spill62, align 32
  %.state255 = load i64, ptr %.slot49, align 4
  %447 = add i64 %.state255, 3
  %448 = sdiv i64 %447, 1
  %449 = srem i64 %448, 4096
  %.spill.load256 = load i64, ptr %.spill12, align 4
  %450 = add i64 %.spill.load256, %449
  store i64 %450, ptr %.spill63, align 4
  %451 = icmp sge i64 %450, 0
  %452 = icmp slt i64 %450, 4096
  %453 = and i1 %451, %452
  br i1 %453, label %direct.true257, label %direct.false258

direct.schedule.42:                               ; preds = %direct.true257
  %.spill.load259 = load i64, ptr %.spill63, align 4
  %454 = srem i64 %.spill.load259, 4096
  %455 = add i64 0, %454
  %tile_snapshot.spill.load260 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill43, align 64
  %456 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load260, 0
  %457 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load260, 1
  %.splatinsert261 = insertelement <8 x i64> poison, i64 %455, i64 0
  %.splat262 = shufflevector <8 x i64> %.splatinsert261, <8 x i64> poison, <8 x i32> zeroinitializer
  %458 = mul <8 x i64> %.splat262, splat (i64 4)
  %459 = add <8 x i64> %457, %458
  %460 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %456, 0
  %461 = insertvalue { <8 x ptr>, <8 x i64> } %460, <8 x i64> %459, 1
  %462 = extractvalue { <8 x ptr>, <8 x i64> } %461, 0
  %463 = extractvalue { <8 x ptr>, <8 x i64> } %461, 1
  %464 = getelementptr i8, <8 x ptr> %462, <8 x i64> %463
  %465 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %464, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %466 = fmul <8 x float> %465, %465
  %467 = load <8 x float>, ptr %.slot64, align 32
  %468 = select <8 x i1> %55, <8 x float> %466, <8 x float> %467
  store <8 x float> %468, ptr %.slot64, align 32
  br label %direct.schedule.43

direct.schedule.43:                               ; preds = %direct.schedule.42, %direct.false258
  %.state263 = load <8 x float>, ptr %.slot53, align 32
  %.state264 = load <8 x float>, ptr %.slot64, align 32
  %469 = fadd <8 x float> %.state263, %.state264
  %.state265 = load i64, ptr %.slot49, align 4
  %470 = add i64 %.state265, 4
  %.spill.load266 = load <8 x float>, ptr %.spill56, align 32
  %.spill.load267 = load <8 x float>, ptr %.spill59, align 32
  %.spill.load268 = load <8 x float>, ptr %.spill62, align 32
  store i64 %470, ptr %.slot49, align 4
  %471 = load <8 x float>, ptr %.slot50, align 32
  %472 = select <8 x i1> %55, <8 x float> %.spill.load266, <8 x float> %471
  store <8 x float> %472, ptr %.slot50, align 32
  %473 = load <8 x float>, ptr %.slot51, align 32
  %474 = select <8 x i1> %55, <8 x float> %.spill.load267, <8 x float> %473
  store <8 x float> %474, ptr %.slot51, align 32
  %475 = load <8 x float>, ptr %.slot52, align 32
  %476 = select <8 x i1> %55, <8 x float> %.spill.load268, <8 x float> %475
  store <8 x float> %476, ptr %.slot52, align 32
  %477 = load <8 x float>, ptr %.slot53, align 32
  %478 = select <8 x i1> %55, <8 x float> %469, <8 x float> %477
  store <8 x float> %478, ptr %.slot53, align 32
  br label %direct.schedule.34

direct.schedule.44:                               ; preds = %direct.false224
  %.spill.load269 = load float, ptr %.spill10, align 4
  %.splatinsert270 = insertelement <8 x float> poison, float %.spill.load269, i64 0
  %.splat271 = shufflevector <8 x float> %.splatinsert270, <8 x float> poison, <8 x i32> zeroinitializer
  %.state272 = load <8 x float>, ptr %.slot50, align 32
  %479 = fadd <8 x float> %.splat271, %.state272
  %.state273 = load <8 x float>, ptr %.slot51, align 32
  %480 = fadd <8 x float> %479, %.state273
  %.state274 = load <8 x float>, ptr %.slot52, align 32
  %481 = fadd <8 x float> %480, %.state274
  %.state275 = load <8 x float>, ptr %.slot53, align 32
  %482 = fadd <8 x float> %481, %.state275
  %.spill.load276 = load float, ptr %.spill41, align 4
  %.splatinsert277 = insertelement <8 x float> poison, float %.spill.load276, i64 0
  %.splat278 = shufflevector <8 x float> %.splatinsert277, <8 x float> poison, <8 x i32> zeroinitializer
  %483 = fdiv <8 x float> %482, %.splat278
  %484 = load <8 x float>, ptr %.spill65, align 32
  %485 = select <8 x i1> %55, <8 x float> %483, <8 x float> %484
  store <8 x float> %485, ptr %.spill65, align 32
  %486 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill66, align 64
  %487 = extractvalue { <8 x ptr>, <8 x i64> } %5, 0
  %488 = extractvalue { <8 x ptr>, <8 x i64> } %486, 0
  %489 = select <8 x i1> %55, <8 x ptr> %487, <8 x ptr> %488
  %490 = extractvalue { <8 x ptr>, <8 x i64> } %5, 1
  %491 = extractvalue { <8 x ptr>, <8 x i64> } %486, 1
  %492 = select <8 x i1> %55, <8 x i64> %490, <8 x i64> %491
  %493 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %489, 0
  %494 = insertvalue { <8 x ptr>, <8 x i64> } %493, <8 x i64> %492, 1
  store { <8 x ptr>, <8 x i64> } %494, ptr %tile_snapshot.spill66, align 64
  store i64 0, ptr %.slot67, align 4
  br label %direct.schedule.45

direct.schedule.45:                               ; preds = %direct.schedule.46, %direct.schedule.44
  %.state279 = load i64, ptr %.slot67, align 4
  %495 = icmp slt i64 %.state279, 4096
  br i1 %495, label %direct.true280, label %direct.false281

direct.schedule.46:                               ; preds = %direct.true280
  %.state282 = load i64, ptr %.slot67, align 4
  %496 = srem i64 %.state282, 4096
  %.state283 = load i64, ptr %.slot67, align 4
  %497 = sdiv i64 %.state283, 4096
  %498 = srem i64 %497, 1
  %499 = add i64 0, %498
  %.spill.load284 = load i64, ptr %.spill11, align 4
  %500 = add i64 %.spill.load284, %499
  %501 = add i64 0, %496
  %502 = mul i64 %500, 4096
  %503 = add i64 %502, %501
  %504 = extractvalue { ptr, i64 } %19, 0
  %505 = mul i64 %503, 4
  %506 = getelementptr i8, ptr %504, i64 %505
  %507 = load float, ptr %506, align 4
  %.splatinsert285 = insertelement <8 x float> poison, float %507, i64 0
  %.splat286 = shufflevector <8 x float> %.splatinsert285, <8 x float> poison, <8 x i32> zeroinitializer
  %tile_snapshot.spill.load287 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill66, align 64
  %508 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load287, 0
  %509 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load287, 1
  %.state288 = load i64, ptr %.slot67, align 4
  %.splatinsert289 = insertelement <8 x i64> poison, i64 %.state288, i64 0
  %.splat290 = shufflevector <8 x i64> %.splatinsert289, <8 x i64> poison, <8 x i32> zeroinitializer
  %510 = mul <8 x i64> %.splat290, splat (i64 4)
  %511 = add <8 x i64> %509, %510
  %512 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %508, 0
  %513 = insertvalue { <8 x ptr>, <8 x i64> } %512, <8 x i64> %511, 1
  %514 = extractvalue { <8 x ptr>, <8 x i64> } %513, 0
  %515 = extractvalue { <8 x ptr>, <8 x i64> } %513, 1
  %516 = getelementptr i8, <8 x ptr> %514, <8 x i64> %515
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %.splat286, <8 x ptr> %516, i32 1, <8 x i1> %55)
  %.state291 = load i64, ptr %.slot67, align 4
  %517 = add i64 %.state291, 1
  store i64 %517, ptr %.slot67, align 4
  br label %direct.schedule.45

direct.schedule.47:                               ; preds = %direct.false281
  %.spill.load292 = load <8 x float>, ptr %.spill65, align 32
  %518 = fadd <8 x float> %.spill.load292, splat (float 0x3EE4F8B580000000)
  %519 = call <8 x float> @llvm.sqrt.v8f32(<8 x float> %518)
  %520 = load <8 x float>, ptr %.spill68, align 32
  %521 = select <8 x i1> %55, <8 x float> %519, <8 x float> %520
  store <8 x float> %521, ptr %.spill68, align 32
  %522 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill69, align 64
  %523 = extractvalue { <8 x ptr>, <8 x i64> } %7, 0
  %524 = extractvalue { <8 x ptr>, <8 x i64> } %522, 0
  %525 = select <8 x i1> %55, <8 x ptr> %523, <8 x ptr> %524
  %526 = extractvalue { <8 x ptr>, <8 x i64> } %7, 1
  %527 = extractvalue { <8 x ptr>, <8 x i64> } %522, 1
  %528 = select <8 x i1> %55, <8 x i64> %526, <8 x i64> %527
  %529 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %525, 0
  %530 = insertvalue { <8 x ptr>, <8 x i64> } %529, <8 x i64> %528, 1
  store { <8 x ptr>, <8 x i64> } %530, ptr %tile_snapshot.spill69, align 64
  store i64 0, ptr %.slot70, align 4
  br label %direct.schedule.48

direct.schedule.48:                               ; preds = %direct.schedule.49, %direct.schedule.47
  %.state293 = load i64, ptr %.slot70, align 4
  %531 = icmp slt i64 %.state293, 4096
  br i1 %531, label %direct.true294, label %direct.false295

direct.schedule.49:                               ; preds = %direct.true294
  %.state296 = load i64, ptr %.slot70, align 4
  %532 = srem i64 %.state296, 4096
  %.state297 = load i64, ptr %.slot70, align 4
  %533 = sdiv i64 %.state297, 4096
  %534 = srem i64 %533, 1
  %535 = add i64 0, %534
  %.spill.load298 = load i64, ptr %.spill11, align 4
  %536 = add i64 %.spill.load298, %535
  %537 = add i64 0, %532
  %538 = mul i64 %536, 4096
  %539 = add i64 %538, %537
  %540 = extractvalue { ptr, i64 } %25, 0
  %541 = mul i64 %539, 4
  %542 = getelementptr i8, ptr %540, i64 %541
  %543 = load float, ptr %542, align 4
  %.splatinsert299 = insertelement <8 x float> poison, float %543, i64 0
  %.splat300 = shufflevector <8 x float> %.splatinsert299, <8 x float> poison, <8 x i32> zeroinitializer
  %tile_snapshot.spill.load301 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill69, align 64
  %544 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load301, 0
  %545 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load301, 1
  %.state302 = load i64, ptr %.slot70, align 4
  %.splatinsert303 = insertelement <8 x i64> poison, i64 %.state302, i64 0
  %.splat304 = shufflevector <8 x i64> %.splatinsert303, <8 x i64> poison, <8 x i32> zeroinitializer
  %546 = mul <8 x i64> %.splat304, splat (i64 4)
  %547 = add <8 x i64> %545, %546
  %548 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %544, 0
  %549 = insertvalue { <8 x ptr>, <8 x i64> } %548, <8 x i64> %547, 1
  %550 = extractvalue { <8 x ptr>, <8 x i64> } %549, 0
  %551 = extractvalue { <8 x ptr>, <8 x i64> } %549, 1
  %552 = getelementptr i8, <8 x ptr> %550, <8 x i64> %551
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %.splat300, <8 x ptr> %552, i32 1, <8 x i1> %55)
  %.state305 = load i64, ptr %.slot70, align 4
  %553 = add i64 %.state305, 1
  store i64 %553, ptr %.slot70, align 4
  br label %direct.schedule.48

direct.schedule.50:                               ; preds = %direct.false295
  store i64 0, ptr %.slot71, align 4
  br label %direct.schedule.51

direct.schedule.51:                               ; preds = %direct.schedule.52, %direct.schedule.50
  %.state306 = load i64, ptr %.slot71, align 4
  %554 = icmp slt i64 %.state306, 4096
  br i1 %554, label %direct.true307, label %direct.false308

direct.schedule.52:                               ; preds = %direct.true307
  %.state309 = load i64, ptr %.slot71, align 4
  %555 = srem i64 %.state309, 4096
  %.state310 = load i64, ptr %.slot71, align 4
  %556 = sdiv i64 %.state310, 4096
  %557 = srem i64 %556, 1
  %.spill.load311 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert312 = insertelement <8 x i64> poison, i64 %557, i64 0
  %.splat313 = shufflevector <8 x i64> %.splatinsert312, <8 x i64> poison, <8 x i32> zeroinitializer
  %558 = add <8 x i64> %.spill.load311, %.splat313
  %559 = add <8 x i64> zeroinitializer, %558
  %560 = add i64 0, %555
  %561 = mul <8 x i64> %559, splat (i64 4096)
  %.splatinsert314 = insertelement <8 x i64> poison, i64 %560, i64 0
  %.splat315 = shufflevector <8 x i64> %.splatinsert314, <8 x i64> poison, <8 x i32> zeroinitializer
  %562 = add <8 x i64> %561, %.splat315
  %563 = add i64 0, %555
  %564 = srem i64 %563, 4096
  %565 = add i64 0, %564
  %566 = srem i64 %565, 4096
  %567 = add i64 0, %566
  %tile_snapshot.spill.load316 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill43, align 64
  %568 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load316, 0
  %569 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load316, 1
  %.splatinsert317 = insertelement <8 x i64> poison, i64 %567, i64 0
  %.splat318 = shufflevector <8 x i64> %.splatinsert317, <8 x i64> poison, <8 x i32> zeroinitializer
  %570 = mul <8 x i64> %.splat318, splat (i64 4)
  %571 = add <8 x i64> %569, %570
  %572 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %568, 0
  %573 = insertvalue { <8 x ptr>, <8 x i64> } %572, <8 x i64> %571, 1
  %574 = extractvalue { <8 x ptr>, <8 x i64> } %573, 0
  %575 = extractvalue { <8 x ptr>, <8 x i64> } %573, 1
  %576 = getelementptr i8, <8 x ptr> %574, <8 x i64> %575
  %577 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %576, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %.spill.load319 = load <8 x float>, ptr %.spill68, align 32
  %578 = fdiv <8 x float> %577, %.spill.load319
  %tile_snapshot.spill.load320 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill66, align 64
  %579 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load320, 0
  %580 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load320, 1
  %.splatinsert321 = insertelement <8 x i64> poison, i64 %565, i64 0
  %.splat322 = shufflevector <8 x i64> %.splatinsert321, <8 x i64> poison, <8 x i32> zeroinitializer
  %581 = mul <8 x i64> %.splat322, splat (i64 4)
  %582 = add <8 x i64> %580, %581
  %583 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %579, 0
  %584 = insertvalue { <8 x ptr>, <8 x i64> } %583, <8 x i64> %582, 1
  %585 = extractvalue { <8 x ptr>, <8 x i64> } %584, 0
  %586 = extractvalue { <8 x ptr>, <8 x i64> } %584, 1
  %587 = getelementptr i8, <8 x ptr> %585, <8 x i64> %586
  %588 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %587, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %589 = fmul <8 x float> %578, %588
  %tile_snapshot.spill.load323 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill69, align 64
  %590 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load323, 0
  %591 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load323, 1
  %.splatinsert324 = insertelement <8 x i64> poison, i64 %563, i64 0
  %.splat325 = shufflevector <8 x i64> %.splatinsert324, <8 x i64> poison, <8 x i32> zeroinitializer
  %592 = mul <8 x i64> %.splat325, splat (i64 4)
  %593 = add <8 x i64> %591, %592
  %594 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %590, 0
  %595 = insertvalue { <8 x ptr>, <8 x i64> } %594, <8 x i64> %593, 1
  %596 = extractvalue { <8 x ptr>, <8 x i64> } %595, 0
  %597 = extractvalue { <8 x ptr>, <8 x i64> } %595, 1
  %598 = getelementptr i8, <8 x ptr> %596, <8 x i64> %597
  %599 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %598, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %600 = fadd <8 x float> %589, %599
  %601 = extractvalue { ptr, i64 } %31, 0
  %602 = mul <8 x i64> %562, splat (i64 4)
  %603 = getelementptr i8, ptr %601, <8 x i64> %602
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %600, <8 x ptr> %603, i32 1, <8 x i1> %55)
  %.state326 = load i64, ptr %.slot71, align 4
  %604 = add i64 %.state326, 1
  store i64 %604, ptr %.slot71, align 4
  br label %direct.schedule.51

direct.schedule.53:                               ; preds = %direct.false308
  ret void

direct.activate:                                  ; preds = %prologue
  br label %direct.schedule.0

direct.inactive:                                  ; preds = %prologue
  ret void

direct.true:                                      ; preds = %direct.schedule.1
  br label %direct.schedule.2

direct.false:                                     ; preds = %direct.schedule.1
  br label %direct.schedule.3

direct.true92:                                    ; preds = %direct.schedule.3
  br label %direct.schedule.4

direct.false93:                                   ; preds = %direct.schedule.3
  %605 = load <8 x float>, ptr %.slot15, align 32
  %606 = select <8 x i1> %55, <8 x float> zeroinitializer, <8 x float> %605
  store <8 x float> %606, ptr %.slot15, align 32
  br label %direct.schedule.5

direct.true99:                                    ; preds = %direct.schedule.5
  br label %direct.schedule.6

direct.false100:                                  ; preds = %direct.schedule.5
  %607 = load <8 x float>, ptr %.slot18, align 32
  %608 = select <8 x i1> %55, <8 x float> zeroinitializer, <8 x float> %607
  store <8 x float> %608, ptr %.slot18, align 32
  br label %direct.schedule.7

direct.true106:                                   ; preds = %direct.schedule.7
  br label %direct.schedule.8

direct.false107:                                  ; preds = %direct.schedule.7
  %609 = load <8 x float>, ptr %.slot21, align 32
  %610 = select <8 x i1> %55, <8 x float> zeroinitializer, <8 x float> %609
  store <8 x float> %610, ptr %.slot21, align 32
  br label %direct.schedule.9

direct.true113:                                   ; preds = %direct.schedule.9
  br label %direct.schedule.10

direct.false114:                                  ; preds = %direct.schedule.9
  %611 = load <8 x float>, ptr %.slot24, align 32
  %612 = select <8 x i1> %55, <8 x float> zeroinitializer, <8 x float> %611
  store <8 x float> %612, ptr %.slot24, align 32
  br label %direct.schedule.11

direct.true124:                                   ; preds = %direct.schedule.12
  br label %direct.schedule.13

direct.false125:                                  ; preds = %direct.schedule.12
  br label %direct.schedule.22

direct.true128:                                   ; preds = %direct.schedule.13
  br label %direct.schedule.14

direct.false129:                                  ; preds = %direct.schedule.13
  %613 = load <8 x float>, ptr %.slot31, align 32
  %614 = select <8 x i1> %55, <8 x float> zeroinitializer, <8 x float> %613
  store <8 x float> %614, ptr %.slot31, align 32
  br label %direct.schedule.15

direct.true138:                                   ; preds = %direct.schedule.15
  br label %direct.schedule.16

direct.false139:                                  ; preds = %direct.schedule.15
  %615 = load <8 x float>, ptr %.slot34, align 32
  %616 = select <8 x i1> %55, <8 x float> zeroinitializer, <8 x float> %615
  store <8 x float> %616, ptr %.slot34, align 32
  br label %direct.schedule.17

direct.true148:                                   ; preds = %direct.schedule.17
  br label %direct.schedule.18

direct.false149:                                  ; preds = %direct.schedule.17
  %617 = load <8 x float>, ptr %.slot37, align 32
  %618 = select <8 x i1> %55, <8 x float> zeroinitializer, <8 x float> %617
  store <8 x float> %618, ptr %.slot37, align 32
  br label %direct.schedule.19

direct.true158:                                   ; preds = %direct.schedule.19
  br label %direct.schedule.20

direct.false159:                                  ; preds = %direct.schedule.19
  %619 = load <8 x float>, ptr %.slot40, align 32
  %620 = select <8 x i1> %55, <8 x float> zeroinitializer, <8 x float> %619
  store <8 x float> %620, ptr %.slot40, align 32
  br label %direct.schedule.21

direct.true178:                                   ; preds = %direct.schedule.23
  br label %direct.schedule.24

direct.false179:                                  ; preds = %direct.schedule.23
  br label %direct.schedule.25

direct.true191:                                   ; preds = %direct.schedule.25
  br label %direct.schedule.26

direct.false192:                                  ; preds = %direct.schedule.25
  %621 = load <8 x float>, ptr %.slot45, align 32
  %622 = select <8 x i1> %55, <8 x float> zeroinitializer, <8 x float> %621
  store <8 x float> %622, ptr %.slot45, align 32
  br label %direct.schedule.27

direct.true198:                                   ; preds = %direct.schedule.27
  br label %direct.schedule.28

direct.false199:                                  ; preds = %direct.schedule.27
  %623 = load <8 x float>, ptr %.slot46, align 32
  %624 = select <8 x i1> %55, <8 x float> zeroinitializer, <8 x float> %623
  store <8 x float> %624, ptr %.slot46, align 32
  br label %direct.schedule.29

direct.true205:                                   ; preds = %direct.schedule.29
  br label %direct.schedule.30

direct.false206:                                  ; preds = %direct.schedule.29
  %625 = load <8 x float>, ptr %.slot47, align 32
  %626 = select <8 x i1> %55, <8 x float> zeroinitializer, <8 x float> %625
  store <8 x float> %626, ptr %.slot47, align 32
  br label %direct.schedule.31

direct.true212:                                   ; preds = %direct.schedule.31
  br label %direct.schedule.32

direct.false213:                                  ; preds = %direct.schedule.31
  %627 = load <8 x float>, ptr %.slot48, align 32
  %628 = select <8 x i1> %55, <8 x float> zeroinitializer, <8 x float> %627
  store <8 x float> %628, ptr %.slot48, align 32
  br label %direct.schedule.33

direct.true223:                                   ; preds = %direct.schedule.34
  br label %direct.schedule.35

direct.false224:                                  ; preds = %direct.schedule.34
  br label %direct.schedule.44

direct.true227:                                   ; preds = %direct.schedule.35
  br label %direct.schedule.36

direct.false228:                                  ; preds = %direct.schedule.35
  %629 = load <8 x float>, ptr %.slot55, align 32
  %630 = select <8 x i1> %55, <8 x float> zeroinitializer, <8 x float> %629
  store <8 x float> %630, ptr %.slot55, align 32
  br label %direct.schedule.37

direct.true237:                                   ; preds = %direct.schedule.37
  br label %direct.schedule.38

direct.false238:                                  ; preds = %direct.schedule.37
  %631 = load <8 x float>, ptr %.slot58, align 32
  %632 = select <8 x i1> %55, <8 x float> zeroinitializer, <8 x float> %631
  store <8 x float> %632, ptr %.slot58, align 32
  br label %direct.schedule.39

direct.true247:                                   ; preds = %direct.schedule.39
  br label %direct.schedule.40

direct.false248:                                  ; preds = %direct.schedule.39
  %633 = load <8 x float>, ptr %.slot61, align 32
  %634 = select <8 x i1> %55, <8 x float> zeroinitializer, <8 x float> %633
  store <8 x float> %634, ptr %.slot61, align 32
  br label %direct.schedule.41

direct.true257:                                   ; preds = %direct.schedule.41
  br label %direct.schedule.42

direct.false258:                                  ; preds = %direct.schedule.41
  %635 = load <8 x float>, ptr %.slot64, align 32
  %636 = select <8 x i1> %55, <8 x float> zeroinitializer, <8 x float> %635
  store <8 x float> %636, ptr %.slot64, align 32
  br label %direct.schedule.43

direct.true280:                                   ; preds = %direct.schedule.45
  br label %direct.schedule.46

direct.false281:                                  ; preds = %direct.schedule.45
  br label %direct.schedule.47

direct.true294:                                   ; preds = %direct.schedule.48
  br label %direct.schedule.49

direct.false295:                                  ; preds = %direct.schedule.48
  br label %direct.schedule.50

direct.true307:                                   ; preds = %direct.schedule.51
  br label %direct.schedule.52

direct.false308:                                  ; preds = %direct.schedule.51
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
