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
  %2 = insertvalue { <8 x ptr>, <8 x i64> } %1, <8 x i64> <i64 0, i64 6148, i64 12296, i64 18444, i64 24592, i64 30740, i64 36888, i64 43036>, 1
  %3 = getelementptr i8, ptr %launch_config, i64 136
  %private.workspace1 = load ptr, ptr %3, align 8
  %tile_snapshot.private2 = getelementptr inbounds i8, ptr %private.workspace1, i64 49184
  %.splatinsert3 = insertelement <8 x ptr> poison, ptr %tile_snapshot.private2, i64 0
  %.splat4 = shufflevector <8 x ptr> %.splatinsert3, <8 x ptr> poison, <8 x i32> zeroinitializer
  %4 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat4, 0
  %5 = insertvalue { <8 x ptr>, <8 x i64> } %4, <8 x i64> <i64 0, i64 6148, i64 12296, i64 18444, i64 24592, i64 30740, i64 36888, i64 43036>, 1
  %.spill = alloca <8 x i64>, align 64
  store <8 x i64> zeroinitializer, ptr %.spill, align 64
  %tile_snapshot.spill = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill, align 64
  %.slot = alloca i64, align 8
  store i64 0, ptr %.slot, align 4
  %.spill5 = alloca float, align 4
  store float 0.000000e+00, ptr %.spill5, align 4
  %.spill6 = alloca float, align 4
  store float 0.000000e+00, ptr %.spill6, align 4
  %.spill7 = alloca float, align 4
  store float 0.000000e+00, ptr %.spill7, align 4
  %.spill8 = alloca float, align 4
  store float 0.000000e+00, ptr %.spill8, align 4
  %tile_snapshot.spill9 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill9, align 64
  %.slot10 = alloca i64, align 8
  store i64 0, ptr %.slot10, align 4
  %.slot11 = alloca i64, align 8
  store i64 0, ptr %.slot11, align 4
  %6 = getelementptr i8, ptr %argument_buffer, i64 0
  %7 = load ptr, ptr %6, align 16
  %8 = getelementptr i8, ptr %6, i64 8
  %9 = load i64, ptr %8, align 8
  %10 = insertvalue { ptr, i64 } poison, ptr %7, 0
  %11 = insertvalue { ptr, i64 } %10, i64 %9, 1
  %12 = getelementptr i8, ptr %argument_buffer, i64 16
  %13 = load ptr, ptr %12, align 16
  %14 = getelementptr i8, ptr %12, i64 8
  %15 = load i64, ptr %14, align 8
  %16 = insertvalue { ptr, i64 } poison, ptr %13, 0
  %17 = insertvalue { ptr, i64 } %16, i64 %15, 1
  %18 = getelementptr i8, ptr %argument_buffer, i64 32
  %19 = load ptr, ptr %18, align 16
  %20 = getelementptr i8, ptr %18, i64 8
  %21 = load i64, ptr %20, align 8
  %22 = insertvalue { ptr, i64 } poison, ptr %19, 0
  %23 = insertvalue { ptr, i64 } %22, i64 %21, 1
  %24 = getelementptr i8, ptr %argument_buffer, i64 48
  %25 = load ptr, ptr %24, align 16
  %26 = getelementptr i8, ptr %24, i64 8
  %27 = load i64, ptr %26, align 8
  %28 = insertvalue { ptr, i64 } poison, ptr %25, 0
  %29 = insertvalue { ptr, i64 } %28, i64 %27, 1
  %30 = load i32, ptr %launch_config, align 4
  %31 = getelementptr i8, ptr %launch_config, i64 12
  %32 = load i32, ptr %31, align 4
  %33 = getelementptr i8, ptr %launch_config, i64 4
  %34 = load i32, ptr %33, align 4
  %35 = getelementptr i8, ptr %launch_config, i64 16
  %36 = load i32, ptr %35, align 4
  %37 = getelementptr i8, ptr %launch_config, i64 8
  %38 = load i32, ptr %37, align 4
  %39 = getelementptr i8, ptr %launch_config, i64 20
  %40 = load i32, ptr %39, align 4
  %41 = getelementptr i8, ptr %launch_config, i64 36
  %42 = load i32, ptr %41, align 4
  %.splatinsert12 = insertelement <8 x i32> poison, i32 %42, i64 0
  %.splat13 = shufflevector <8 x i32> %.splatinsert12, <8 x i32> poison, <8 x i32> zeroinitializer
  %43 = add <8 x i32> %.splat13, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %44 = mul i32 %30, 32
  %.splatinsert14 = insertelement <8 x i32> poison, i32 %44, i64 0
  %.splat15 = shufflevector <8 x i32> %.splatinsert14, <8 x i32> poison, <8 x i32> zeroinitializer
  %45 = add <8 x i32> %.splat15, %43
  %46 = mul i32 %34, 1
  %.splatinsert16 = insertelement <8 x i32> poison, i32 %46, i64 0
  %.splat17 = shufflevector <8 x i32> %.splatinsert16, <8 x i32> poison, <8 x i32> zeroinitializer
  %47 = add <8 x i32> %.splat17, zeroinitializer
  %48 = mul i32 %38, 1
  %.splatinsert18 = insertelement <8 x i32> poison, i32 %48, i64 0
  %.splat19 = shufflevector <8 x i32> %.splatinsert18, <8 x i32> poison, <8 x i32> zeroinitializer
  %49 = add <8 x i32> %.splat19, zeroinitializer
  %50 = insertvalue [3 x <8 x i32>] poison, <8 x i32> %45, 0
  %51 = insertvalue [3 x <8 x i32>] %50, <8 x i32> %47, 1
  %52 = insertvalue [3 x <8 x i32>] %51, <8 x i32> %49, 2
  %.splatinsert20 = insertelement <8 x i32> poison, i32 %active_lane_count, i64 0
  %.splat21 = shufflevector <8 x i32> %.splatinsert20, <8 x i32> poison, <8 x i32> zeroinitializer
  %53 = icmp ult <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, %.splat21
  %54 = call i1 @llvm.vector.reduce.or.v8i1(<8 x i1> %53)
  br i1 %54, label %direct.activate, label %direct.inactive

direct.schedule.0:                                ; preds = %direct.activate
  %55 = extractvalue [3 x <8 x i32>] %52, 0
  %56 = zext <8 x i32> %55 to <8 x i64>
  %57 = select <8 x i1> %53, <8 x i64> %56, <8 x i64> zeroinitializer
  %58 = select <8 x i1> %53, <8 x i64> splat (i64 1), <8 x i64> splat (i64 1)
  %59 = sdiv <8 x i64> %57, %58
  %60 = select <8 x i1> %53, <8 x i64> %59, <8 x i64> zeroinitializer
  %61 = select <8 x i1> %53, <8 x i64> splat (i64 64), <8 x i64> splat (i64 1)
  %62 = srem <8 x i64> %60, %61
  %63 = load <8 x i64>, ptr %.spill, align 64
  %64 = select <8 x i1> %53, <8 x i64> %62, <8 x i64> %63
  store <8 x i64> %64, ptr %.spill, align 64
  %65 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %66 = extractvalue { <8 x ptr>, <8 x i64> } %2, 0
  %67 = extractvalue { <8 x ptr>, <8 x i64> } %65, 0
  %68 = select <8 x i1> %53, <8 x ptr> %66, <8 x ptr> %67
  %69 = extractvalue { <8 x ptr>, <8 x i64> } %2, 1
  %70 = extractvalue { <8 x ptr>, <8 x i64> } %65, 1
  %71 = select <8 x i1> %53, <8 x i64> %69, <8 x i64> %70
  %72 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %68, 0
  %73 = insertvalue { <8 x ptr>, <8 x i64> } %72, <8 x i64> %71, 1
  store { <8 x ptr>, <8 x i64> } %73, ptr %tile_snapshot.spill, align 64
  store i64 0, ptr %.slot, align 4
  br label %direct.schedule.1

direct.schedule.1:                                ; preds = %direct.schedule.2, %direct.schedule.0
  %.state = load i64, ptr %.slot, align 4
  %74 = icmp slt i64 %.state, 1537
  br i1 %74, label %direct.true, label %direct.false

direct.schedule.2:                                ; preds = %direct.true
  %.state22 = load i64, ptr %.slot, align 4
  %75 = srem i64 %.state22, 1537
  %.state23 = load i64, ptr %.slot, align 4
  %76 = sdiv i64 %.state23, 1537
  %77 = srem i64 %76, 1
  %.spill.load = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert24 = insertelement <8 x i64> poison, i64 %77, i64 0
  %.splat25 = shufflevector <8 x i64> %.splatinsert24, <8 x i64> poison, <8 x i32> zeroinitializer
  %78 = add <8 x i64> %.spill.load, %.splat25
  %79 = add <8 x i64> zeroinitializer, %78
  %80 = add i64 0, %75
  %81 = mul <8 x i64> %79, splat (i64 1537)
  %.splatinsert26 = insertelement <8 x i64> poison, i64 %80, i64 0
  %.splat27 = shufflevector <8 x i64> %.splatinsert26, <8 x i64> poison, <8 x i32> zeroinitializer
  %82 = add <8 x i64> %81, %.splat27
  %83 = extractvalue { ptr, i64 } %11, 0
  %84 = mul <8 x i64> %82, splat (i64 4)
  %85 = getelementptr i8, ptr %83, <8 x i64> %84
  %86 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %85, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %87 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 0
  %88 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 1
  %.state28 = load i64, ptr %.slot, align 4
  %.splatinsert29 = insertelement <8 x i64> poison, i64 %.state28, i64 0
  %.splat30 = shufflevector <8 x i64> %.splatinsert29, <8 x i64> poison, <8 x i32> zeroinitializer
  %89 = mul <8 x i64> %.splat30, splat (i64 4)
  %90 = add <8 x i64> %88, %89
  %91 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %87, 0
  %92 = insertvalue { <8 x ptr>, <8 x i64> } %91, <8 x i64> %90, 1
  %93 = extractvalue { <8 x ptr>, <8 x i64> } %92, 0
  %94 = extractvalue { <8 x ptr>, <8 x i64> } %92, 1
  %95 = getelementptr i8, <8 x ptr> %93, <8 x i64> %94
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %86, <8 x ptr> %95, i32 1, <8 x i1> %53)
  %.state31 = load i64, ptr %.slot, align 4
  %96 = add i64 %.state31, 1
  store i64 %96, ptr %.slot, align 4
  br label %direct.schedule.1

direct.schedule.3:                                ; preds = %direct.false
  store float 5.000000e-01, ptr %.spill5, align 4
  store float 0x3FA6E4E260000000, ptr %.spill6, align 4
  store float 0x3FE9884540000000, ptr %.spill7, align 4
  store float 1.000000e+00, ptr %.spill8, align 4
  %97 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill9, align 64
  %98 = extractvalue { <8 x ptr>, <8 x i64> } %5, 0
  %99 = extractvalue { <8 x ptr>, <8 x i64> } %97, 0
  %100 = select <8 x i1> %53, <8 x ptr> %98, <8 x ptr> %99
  %101 = extractvalue { <8 x ptr>, <8 x i64> } %5, 1
  %102 = extractvalue { <8 x ptr>, <8 x i64> } %97, 1
  %103 = select <8 x i1> %53, <8 x i64> %101, <8 x i64> %102
  %104 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %100, 0
  %105 = insertvalue { <8 x ptr>, <8 x i64> } %104, <8 x i64> %103, 1
  store { <8 x ptr>, <8 x i64> } %105, ptr %tile_snapshot.spill9, align 64
  store i64 0, ptr %.slot10, align 4
  br label %direct.schedule.4

direct.schedule.4:                                ; preds = %direct.schedule.5, %direct.schedule.3
  %.state32 = load i64, ptr %.slot10, align 4
  %106 = icmp slt i64 %.state32, 1537
  br i1 %106, label %direct.true33, label %direct.false34

direct.schedule.5:                                ; preds = %direct.true33
  %.state35 = load i64, ptr %.slot10, align 4
  %107 = srem i64 %.state35, 1537
  %.state36 = load i64, ptr %.slot10, align 4
  %108 = sdiv i64 %.state36, 1537
  %109 = srem i64 %108, 1
  %.spill.load37 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert38 = insertelement <8 x i64> poison, i64 %109, i64 0
  %.splat39 = shufflevector <8 x i64> %.splatinsert38, <8 x i64> poison, <8 x i32> zeroinitializer
  %110 = add <8 x i64> %.spill.load37, %.splat39
  %111 = add <8 x i64> zeroinitializer, %110
  %112 = add i64 0, %107
  %113 = mul <8 x i64> %111, splat (i64 1537)
  %.splatinsert40 = insertelement <8 x i64> poison, i64 %112, i64 0
  %.splat41 = shufflevector <8 x i64> %.splatinsert40, <8 x i64> poison, <8 x i32> zeroinitializer
  %114 = add <8 x i64> %113, %.splat41
  %115 = extractvalue { ptr, i64 } %17, 0
  %116 = mul <8 x i64> %114, splat (i64 4)
  %117 = getelementptr i8, ptr %115, <8 x i64> %116
  %118 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %117, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load42 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill9, align 64
  %119 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load42, 0
  %120 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load42, 1
  %.state43 = load i64, ptr %.slot10, align 4
  %.splatinsert44 = insertelement <8 x i64> poison, i64 %.state43, i64 0
  %.splat45 = shufflevector <8 x i64> %.splatinsert44, <8 x i64> poison, <8 x i32> zeroinitializer
  %121 = mul <8 x i64> %.splat45, splat (i64 4)
  %122 = add <8 x i64> %120, %121
  %123 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %119, 0
  %124 = insertvalue { <8 x ptr>, <8 x i64> } %123, <8 x i64> %122, 1
  %125 = extractvalue { <8 x ptr>, <8 x i64> } %124, 0
  %126 = extractvalue { <8 x ptr>, <8 x i64> } %124, 1
  %127 = getelementptr i8, <8 x ptr> %125, <8 x i64> %126
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %118, <8 x ptr> %127, i32 1, <8 x i1> %53)
  %.state46 = load i64, ptr %.slot10, align 4
  %128 = add i64 %.state46, 1
  store i64 %128, ptr %.slot10, align 4
  br label %direct.schedule.4

direct.schedule.6:                                ; preds = %direct.false34
  store i64 0, ptr %.slot11, align 4
  br label %direct.schedule.7

direct.schedule.7:                                ; preds = %direct.schedule.8, %direct.schedule.6
  %.state47 = load i64, ptr %.slot11, align 4
  %129 = icmp slt i64 %.state47, 1537
  br i1 %129, label %direct.true48, label %direct.false49

direct.schedule.8:                                ; preds = %direct.true48
  %.state50 = load i64, ptr %.slot11, align 4
  %130 = srem i64 %.state50, 1537
  %.state51 = load i64, ptr %.slot11, align 4
  %131 = sdiv i64 %.state51, 1537
  %132 = srem i64 %131, 1
  %.spill.load52 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert53 = insertelement <8 x i64> poison, i64 %132, i64 0
  %.splat54 = shufflevector <8 x i64> %.splatinsert53, <8 x i64> poison, <8 x i32> zeroinitializer
  %133 = add <8 x i64> %.spill.load52, %.splat54
  %134 = add <8 x i64> zeroinitializer, %133
  %135 = add i64 0, %130
  %136 = mul <8 x i64> %134, splat (i64 1537)
  %.splatinsert55 = insertelement <8 x i64> poison, i64 %135, i64 0
  %.splat56 = shufflevector <8 x i64> %.splatinsert55, <8 x i64> poison, <8 x i32> zeroinitializer
  %137 = add <8 x i64> %136, %.splat56
  %138 = add i64 0, %130
  %139 = srem i64 %138, 1537
  %140 = add i64 0, %139
  %141 = srem i64 %140, 1537
  %142 = add i64 0, %141
  %tile_snapshot.spill.load57 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %143 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load57, 0
  %144 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load57, 1
  %.splatinsert58 = insertelement <8 x i64> poison, i64 %142, i64 0
  %.splat59 = shufflevector <8 x i64> %.splatinsert58, <8 x i64> poison, <8 x i32> zeroinitializer
  %145 = mul <8 x i64> %.splat59, splat (i64 4)
  %146 = add <8 x i64> %144, %145
  %147 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %143, 0
  %148 = insertvalue { <8 x ptr>, <8 x i64> } %147, <8 x i64> %146, 1
  %149 = extractvalue { <8 x ptr>, <8 x i64> } %148, 0
  %150 = extractvalue { <8 x ptr>, <8 x i64> } %148, 1
  %151 = getelementptr i8, <8 x ptr> %149, <8 x i64> %150
  %152 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %151, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %.spill.load60 = load float, ptr %.spill5, align 4
  %.splatinsert61 = insertelement <8 x float> poison, float %.spill.load60, i64 0
  %.splat62 = shufflevector <8 x float> %.splatinsert61, <8 x float> poison, <8 x i32> zeroinitializer
  %153 = fmul <8 x float> %.splat62, %152
  %154 = srem i64 %142, 1537
  %155 = add i64 0, %154
  %156 = srem i64 %155, 1537
  %157 = add i64 0, %156
  %158 = srem i64 %157, 1537
  %159 = add i64 0, %158
  %tile_snapshot.spill.load63 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %160 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load63, 0
  %161 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load63, 1
  %.splatinsert64 = insertelement <8 x i64> poison, i64 %159, i64 0
  %.splat65 = shufflevector <8 x i64> %.splatinsert64, <8 x i64> poison, <8 x i32> zeroinitializer
  %162 = mul <8 x i64> %.splat65, splat (i64 4)
  %163 = add <8 x i64> %161, %162
  %164 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %160, 0
  %165 = insertvalue { <8 x ptr>, <8 x i64> } %164, <8 x i64> %163, 1
  %166 = extractvalue { <8 x ptr>, <8 x i64> } %165, 0
  %167 = extractvalue { <8 x ptr>, <8 x i64> } %165, 1
  %168 = getelementptr i8, <8 x ptr> %166, <8 x i64> %167
  %169 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %168, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %170 = srem i64 %159, 1537
  %171 = add i64 0, %170
  %172 = srem i64 %171, 1537
  %173 = add i64 0, %172
  %174 = srem i64 %173, 1537
  %175 = add i64 0, %174
  %tile_snapshot.spill.load66 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %176 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load66, 0
  %177 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load66, 1
  %.splatinsert67 = insertelement <8 x i64> poison, i64 %175, i64 0
  %.splat68 = shufflevector <8 x i64> %.splatinsert67, <8 x i64> poison, <8 x i32> zeroinitializer
  %178 = mul <8 x i64> %.splat68, splat (i64 4)
  %179 = add <8 x i64> %177, %178
  %180 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %176, 0
  %181 = insertvalue { <8 x ptr>, <8 x i64> } %180, <8 x i64> %179, 1
  %182 = extractvalue { <8 x ptr>, <8 x i64> } %181, 0
  %183 = extractvalue { <8 x ptr>, <8 x i64> } %181, 1
  %184 = getelementptr i8, <8 x ptr> %182, <8 x i64> %183
  %185 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %184, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %.spill.load69 = load float, ptr %.spill6, align 4
  %.splatinsert70 = insertelement <8 x float> poison, float %.spill.load69, i64 0
  %.splat71 = shufflevector <8 x float> %.splatinsert70, <8 x float> poison, <8 x i32> zeroinitializer
  %186 = fmul <8 x float> %.splat71, %185
  %tile_snapshot.spill.load72 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %187 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load72, 0
  %188 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load72, 1
  %.splatinsert73 = insertelement <8 x i64> poison, i64 %173, i64 0
  %.splat74 = shufflevector <8 x i64> %.splatinsert73, <8 x i64> poison, <8 x i32> zeroinitializer
  %189 = mul <8 x i64> %.splat74, splat (i64 4)
  %190 = add <8 x i64> %188, %189
  %191 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %187, 0
  %192 = insertvalue { <8 x ptr>, <8 x i64> } %191, <8 x i64> %190, 1
  %193 = extractvalue { <8 x ptr>, <8 x i64> } %192, 0
  %194 = extractvalue { <8 x ptr>, <8 x i64> } %192, 1
  %195 = getelementptr i8, <8 x ptr> %193, <8 x i64> %194
  %196 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %195, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %197 = fmul <8 x float> %186, %196
  %tile_snapshot.spill.load75 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %198 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load75, 0
  %199 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load75, 1
  %.splatinsert76 = insertelement <8 x i64> poison, i64 %171, i64 0
  %.splat77 = shufflevector <8 x i64> %.splatinsert76, <8 x i64> poison, <8 x i32> zeroinitializer
  %200 = mul <8 x i64> %.splat77, splat (i64 4)
  %201 = add <8 x i64> %199, %200
  %202 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %198, 0
  %203 = insertvalue { <8 x ptr>, <8 x i64> } %202, <8 x i64> %201, 1
  %204 = extractvalue { <8 x ptr>, <8 x i64> } %203, 0
  %205 = extractvalue { <8 x ptr>, <8 x i64> } %203, 1
  %206 = getelementptr i8, <8 x ptr> %204, <8 x i64> %205
  %207 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %206, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %208 = fmul <8 x float> %197, %207
  %209 = fadd <8 x float> %169, %208
  %.spill.load78 = load float, ptr %.spill7, align 4
  %.splatinsert79 = insertelement <8 x float> poison, float %.spill.load78, i64 0
  %.splat80 = shufflevector <8 x float> %.splatinsert79, <8 x float> poison, <8 x i32> zeroinitializer
  %210 = fmul <8 x float> %.splat80, %209
  %211 = select <8 x i1> %53, <8 x float> %210, <8 x float> zeroinitializer
  %native.tanh = call <8 x float> @__luisa_cpu_native_tanh_f32_v8_precise(<8 x float> %211)
  %.spill.load81 = load float, ptr %.spill8, align 4
  %.splatinsert82 = insertelement <8 x float> poison, float %.spill.load81, i64 0
  %.splat83 = shufflevector <8 x float> %.splatinsert82, <8 x float> poison, <8 x i32> zeroinitializer
  %212 = fadd <8 x float> %.splat83, %native.tanh
  %213 = fmul <8 x float> %153, %212
  %tile_snapshot.spill.load84 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill9, align 64
  %214 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load84, 0
  %215 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load84, 1
  %.splatinsert85 = insertelement <8 x i64> poison, i64 %138, i64 0
  %.splat86 = shufflevector <8 x i64> %.splatinsert85, <8 x i64> poison, <8 x i32> zeroinitializer
  %216 = mul <8 x i64> %.splat86, splat (i64 4)
  %217 = add <8 x i64> %215, %216
  %218 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %214, 0
  %219 = insertvalue { <8 x ptr>, <8 x i64> } %218, <8 x i64> %217, 1
  %220 = extractvalue { <8 x ptr>, <8 x i64> } %219, 0
  %221 = extractvalue { <8 x ptr>, <8 x i64> } %219, 1
  %222 = getelementptr i8, <8 x ptr> %220, <8 x i64> %221
  %223 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %222, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %224 = fadd <8 x float> %213, %223
  %225 = extractvalue { ptr, i64 } %29, 0
  %226 = mul <8 x i64> %137, splat (i64 4)
  %227 = getelementptr i8, ptr %225, <8 x i64> %226
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %224, <8 x ptr> %227, i32 1, <8 x i1> %53)
  %.state87 = load i64, ptr %.slot11, align 4
  %228 = add i64 %.state87, 1
  store i64 %228, ptr %.slot11, align 4
  br label %direct.schedule.7

direct.schedule.9:                                ; preds = %direct.false49
  ret void

direct.activate:                                  ; preds = %prologue
  br label %direct.schedule.0

direct.inactive:                                  ; preds = %prologue
  ret void

direct.true:                                      ; preds = %direct.schedule.1
  br label %direct.schedule.2

direct.false:                                     ; preds = %direct.schedule.1
  br label %direct.schedule.3

direct.true33:                                    ; preds = %direct.schedule.4
  br label %direct.schedule.5

direct.false34:                                   ; preds = %direct.schedule.4
  br label %direct.schedule.6

direct.true48:                                    ; preds = %direct.schedule.7
  br label %direct.schedule.8

direct.false49:                                   ; preds = %direct.schedule.7
  br label %direct.schedule.9
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.vector.reduce.or.v8i1(<8 x i1>) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr>, i32 immarg, <8 x i1>, <8 x float>) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.masked.scatter.v8f32.v8p0(<8 x float>, <8 x ptr>, i32 immarg, <8 x i1>) #2

; Function Attrs: alwaysinline norecurse nounwind willreturn memory(none)
define internal <8 x float> @__luisa_cpu_native_tanh_f32_v8_precise(<8 x float> %x) #3 {
entry:
  %0 = bitcast <8 x float> %x to <8 x i32>
  %1 = and <8 x i32> %0, splat (i32 2147483647)
  %2 = bitcast <8 x i32> %1 to <8 x float>
  %3 = fcmp contract ole <8 x float> %2, splat (float 1.000000e+00)
  %4 = fmul contract <8 x float> %2, %2
  %5 = fmul contract <8 x float> splat (float 0x3DE6124620000000), %4
  %6 = fadd contract <8 x float> %5, splat (float 0x3E5AE64560000000)
  %7 = fmul contract <8 x float> %6, %4
  %8 = fadd contract <8 x float> %7, splat (float 0x3EC71DE3A0000000)
  %9 = fmul contract <8 x float> %8, %4
  %10 = fadd contract <8 x float> %9, splat (float 0x3F2A01A020000000)
  %11 = fmul contract <8 x float> %10, %4
  %12 = fadd contract <8 x float> %11, splat (float 0x3F81111120000000)
  %13 = fmul contract <8 x float> %12, %4
  %14 = fadd contract <8 x float> %13, splat (float 0x3FC5555560000000)
  %15 = fmul contract <8 x float> %14, %4
  %16 = fadd contract <8 x float> %15, splat (float 1.000000e+00)
  %17 = fmul contract <8 x float> %2, %16
  %18 = fmul contract <8 x float> %2, %2
  %19 = fmul contract <8 x float> splat (float 0x3E21EED8E0000000), %18
  %20 = fadd contract <8 x float> %19, splat (float 0x3E927E4FC0000000)
  %21 = fmul contract <8 x float> %20, %18
  %22 = fadd contract <8 x float> %21, splat (float 0x3EFA01A020000000)
  %23 = fmul contract <8 x float> %22, %18
  %24 = fadd contract <8 x float> %23, splat (float 0x3F56C16C20000000)
  %25 = fmul contract <8 x float> %24, %18
  %26 = fadd contract <8 x float> %25, splat (float 0x3FA5555560000000)
  %27 = fmul contract <8 x float> %26, %18
  %28 = fadd contract <8 x float> %27, splat (float 5.000000e-01)
  %29 = fmul contract <8 x float> %28, %18
  %30 = fadd contract <8 x float> %29, splat (float 1.000000e+00)
  %31 = fdiv contract <8 x float> %17, %30
  %32 = fcmp contract ole <8 x float> %2, splat (float 9.000000e+00)
  %33 = select contract <8 x i1> %32, <8 x float> %2, <8 x float> zeroinitializer
  %exp.half = call contract <8 x float> @__luisa_cpu_native_exp_half_f32_v8_u10(<8 x float> %33)
  %34 = fdiv contract <8 x float> splat (float 2.500000e-01), %exp.half
  %35 = fsub contract <8 x float> %exp.half, %34
  %36 = fadd contract <8 x float> %exp.half, %34
  %37 = fdiv contract <8 x float> %35, %36
  %38 = select contract <8 x i1> %32, <8 x float> %37, <8 x float> splat (float 1.000000e+00)
  %39 = select contract <8 x i1> %3, <8 x float> %31, <8 x float> %38
  %40 = bitcast <8 x float> %39 to <8 x i32>
  %41 = and <8 x i32> %40, splat (i32 2147483647)
  %42 = bitcast <8 x float> %x to <8 x i32>
  %43 = and <8 x i32> %42, splat (i32 -2147483648)
  %44 = or <8 x i32> %41, %43
  %45 = bitcast <8 x i32> %44 to <8 x float>
  %46 = fcmp contract uno <8 x float> %x, %x
  %47 = select contract <8 x i1> %46, <8 x float> splat (float 0x7FF8000000000000), <8 x float> %45
  ret <8 x float> %47
}

; Function Attrs: alwaysinline norecurse nounwind willreturn memory(none)
define internal <8 x float> @__luisa_cpu_native_exp_half_f32_v8_u10(<8 x float> %x) #3 {
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
  %31 = sub <8 x i32> %11, splat (i32 1)
  %32 = ashr <8 x i32> %31, splat (i32 1)
  %33 = add <8 x i32> %32, splat (i32 127)
  %34 = shl <8 x i32> %33, splat (i32 23)
  %35 = bitcast <8 x i32> %34 to <8 x float>
  %36 = fmul <8 x float> %30, %35
  %37 = sub <8 x i32> %31, %32
  %38 = add <8 x i32> %37, splat (i32 127)
  %39 = shl <8 x i32> %38, splat (i32 23)
  %40 = bitcast <8 x i32> %39 to <8 x float>
  %41 = fmul <8 x float> %36, %40
  %42 = fcmp olt <8 x float> %x, splat (float -1.040000e+02)
  %43 = select <8 x i1> %42, <8 x float> zeroinitializer, <8 x float> %41
  %44 = fcmp ogt <8 x float> %x, splat (float 1.000000e+02)
  %45 = select <8 x i1> %44, <8 x float> splat (float 0x7FF0000000000000), <8 x float> %43
  %46 = fcmp uno <8 x float> %x, %x
  %47 = select <8 x i1> %46, <8 x float> splat (float 0x7FF8000000000000), <8 x float> %45
  ret <8 x float> %47
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
