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
  %5 = insertvalue { <8 x ptr>, <8 x i64> } %4, <8 x i64> <i64 0, i64 16384, i64 32768, i64 49152, i64 65536, i64 81920, i64 98304, i64 114688>, 1
  %.spill = alloca <8 x i64>, align 64
  store <8 x i64> zeroinitializer, ptr %.spill, align 64
  %tile_snapshot.spill = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill, align 64
  %.slot = alloca i64, align 8
  store i64 0, ptr %.slot, align 4
  %.spill5 = alloca float, align 4
  store float 0.000000e+00, ptr %.spill5, align 4
  %tile_snapshot.spill6 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill6, align 64
  %.slot7 = alloca i64, align 8
  store i64 0, ptr %.slot7, align 4
  %.slot8 = alloca i64, align 8
  store i64 0, ptr %.slot8, align 4
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
  %.splatinsert9 = insertelement <8 x i32> poison, i32 %42, i64 0
  %.splat10 = shufflevector <8 x i32> %.splatinsert9, <8 x i32> poison, <8 x i32> zeroinitializer
  %43 = add <8 x i32> %.splat10, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %44 = mul i32 %30, 128
  %.splatinsert11 = insertelement <8 x i32> poison, i32 %44, i64 0
  %.splat12 = shufflevector <8 x i32> %.splatinsert11, <8 x i32> poison, <8 x i32> zeroinitializer
  %45 = add <8 x i32> %.splat12, %43
  %46 = mul i32 %34, 1
  %.splatinsert13 = insertelement <8 x i32> poison, i32 %46, i64 0
  %.splat14 = shufflevector <8 x i32> %.splatinsert13, <8 x i32> poison, <8 x i32> zeroinitializer
  %47 = add <8 x i32> %.splat14, zeroinitializer
  %48 = mul i32 %38, 1
  %.splatinsert15 = insertelement <8 x i32> poison, i32 %48, i64 0
  %.splat16 = shufflevector <8 x i32> %.splatinsert15, <8 x i32> poison, <8 x i32> zeroinitializer
  %49 = add <8 x i32> %.splat16, zeroinitializer
  %50 = insertvalue [3 x <8 x i32>] poison, <8 x i32> %45, 0
  %51 = insertvalue [3 x <8 x i32>] %50, <8 x i32> %47, 1
  %52 = insertvalue [3 x <8 x i32>] %51, <8 x i32> %49, 2
  %.splatinsert17 = insertelement <8 x i32> poison, i32 %active_lane_count, i64 0
  %.splat18 = shufflevector <8 x i32> %.splatinsert17, <8 x i32> poison, <8 x i32> zeroinitializer
  %53 = icmp ult <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, %.splat18
  %54 = call i1 @llvm.vector.reduce.or.v8i1(<8 x i1> %53)
  br i1 %54, label %direct.activate, label %direct.inactive

direct.schedule.0:                                ; preds = %direct.activate
  %55 = extractvalue [3 x <8 x i32>] %52, 0
  %56 = zext <8 x i32> %55 to <8 x i64>
  %57 = select <8 x i1> %53, <8 x i64> %56, <8 x i64> zeroinitializer
  %58 = select <8 x i1> %53, <8 x i64> splat (i64 1), <8 x i64> splat (i64 1)
  %59 = sdiv <8 x i64> %57, %58
  %60 = select <8 x i1> %53, <8 x i64> %59, <8 x i64> zeroinitializer
  %61 = select <8 x i1> %53, <8 x i64> splat (i64 1024), <8 x i64> splat (i64 1)
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
  %74 = icmp slt i64 %.state, 4096
  br i1 %74, label %direct.true, label %direct.false

direct.schedule.2:                                ; preds = %direct.true
  %.state19 = load i64, ptr %.slot, align 4
  %75 = srem i64 %.state19, 4096
  %.state20 = load i64, ptr %.slot, align 4
  %76 = sdiv i64 %.state20, 4096
  %77 = srem i64 %76, 1
  %.spill.load = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert21 = insertelement <8 x i64> poison, i64 %77, i64 0
  %.splat22 = shufflevector <8 x i64> %.splatinsert21, <8 x i64> poison, <8 x i32> zeroinitializer
  %78 = add <8 x i64> %.spill.load, %.splat22
  %79 = add <8 x i64> zeroinitializer, %78
  %80 = add i64 0, %75
  %81 = mul <8 x i64> %79, splat (i64 4096)
  %.splatinsert23 = insertelement <8 x i64> poison, i64 %80, i64 0
  %.splat24 = shufflevector <8 x i64> %.splatinsert23, <8 x i64> poison, <8 x i32> zeroinitializer
  %82 = add <8 x i64> %81, %.splat24
  %83 = extractvalue { ptr, i64 } %11, 0
  %84 = mul <8 x i64> %82, splat (i64 4)
  %85 = getelementptr i8, ptr %83, <8 x i64> %84
  %86 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %85, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %87 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 0
  %88 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 1
  %.state25 = load i64, ptr %.slot, align 4
  %.splatinsert26 = insertelement <8 x i64> poison, i64 %.state25, i64 0
  %.splat27 = shufflevector <8 x i64> %.splatinsert26, <8 x i64> poison, <8 x i32> zeroinitializer
  %89 = mul <8 x i64> %.splat27, splat (i64 4)
  %90 = add <8 x i64> %88, %89
  %91 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %87, 0
  %92 = insertvalue { <8 x ptr>, <8 x i64> } %91, <8 x i64> %90, 1
  %93 = extractvalue { <8 x ptr>, <8 x i64> } %92, 0
  %94 = extractvalue { <8 x ptr>, <8 x i64> } %92, 1
  %95 = getelementptr i8, <8 x ptr> %93, <8 x i64> %94
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %86, <8 x ptr> %95, i32 1, <8 x i1> %53)
  %.state28 = load i64, ptr %.slot, align 4
  %96 = add i64 %.state28, 1
  store i64 %96, ptr %.slot, align 4
  br label %direct.schedule.1

direct.schedule.3:                                ; preds = %direct.false
  store float 1.000000e+00, ptr %.spill5, align 4
  %97 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill6, align 64
  %98 = extractvalue { <8 x ptr>, <8 x i64> } %5, 0
  %99 = extractvalue { <8 x ptr>, <8 x i64> } %97, 0
  %100 = select <8 x i1> %53, <8 x ptr> %98, <8 x ptr> %99
  %101 = extractvalue { <8 x ptr>, <8 x i64> } %5, 1
  %102 = extractvalue { <8 x ptr>, <8 x i64> } %97, 1
  %103 = select <8 x i1> %53, <8 x i64> %101, <8 x i64> %102
  %104 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %100, 0
  %105 = insertvalue { <8 x ptr>, <8 x i64> } %104, <8 x i64> %103, 1
  store { <8 x ptr>, <8 x i64> } %105, ptr %tile_snapshot.spill6, align 64
  store i64 0, ptr %.slot7, align 4
  br label %direct.schedule.4

direct.schedule.4:                                ; preds = %direct.schedule.5, %direct.schedule.3
  %.state29 = load i64, ptr %.slot7, align 4
  %106 = icmp slt i64 %.state29, 4096
  br i1 %106, label %direct.true30, label %direct.false31

direct.schedule.5:                                ; preds = %direct.true30
  %.state32 = load i64, ptr %.slot7, align 4
  %107 = srem i64 %.state32, 4096
  %.state33 = load i64, ptr %.slot7, align 4
  %108 = sdiv i64 %.state33, 4096
  %109 = srem i64 %108, 1
  %.spill.load34 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert35 = insertelement <8 x i64> poison, i64 %109, i64 0
  %.splat36 = shufflevector <8 x i64> %.splatinsert35, <8 x i64> poison, <8 x i32> zeroinitializer
  %110 = add <8 x i64> %.spill.load34, %.splat36
  %111 = add <8 x i64> zeroinitializer, %110
  %112 = add i64 0, %107
  %113 = mul <8 x i64> %111, splat (i64 4096)
  %.splatinsert37 = insertelement <8 x i64> poison, i64 %112, i64 0
  %.splat38 = shufflevector <8 x i64> %.splatinsert37, <8 x i64> poison, <8 x i32> zeroinitializer
  %114 = add <8 x i64> %113, %.splat38
  %115 = extractvalue { ptr, i64 } %17, 0
  %116 = mul <8 x i64> %114, splat (i64 4)
  %117 = getelementptr i8, ptr %115, <8 x i64> %116
  %118 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %117, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load39 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill6, align 64
  %119 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load39, 0
  %120 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load39, 1
  %.state40 = load i64, ptr %.slot7, align 4
  %.splatinsert41 = insertelement <8 x i64> poison, i64 %.state40, i64 0
  %.splat42 = shufflevector <8 x i64> %.splatinsert41, <8 x i64> poison, <8 x i32> zeroinitializer
  %121 = mul <8 x i64> %.splat42, splat (i64 4)
  %122 = add <8 x i64> %120, %121
  %123 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %119, 0
  %124 = insertvalue { <8 x ptr>, <8 x i64> } %123, <8 x i64> %122, 1
  %125 = extractvalue { <8 x ptr>, <8 x i64> } %124, 0
  %126 = extractvalue { <8 x ptr>, <8 x i64> } %124, 1
  %127 = getelementptr i8, <8 x ptr> %125, <8 x i64> %126
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %118, <8 x ptr> %127, i32 1, <8 x i1> %53)
  %.state43 = load i64, ptr %.slot7, align 4
  %128 = add i64 %.state43, 1
  store i64 %128, ptr %.slot7, align 4
  br label %direct.schedule.4

direct.schedule.6:                                ; preds = %direct.false31
  store i64 0, ptr %.slot8, align 4
  br label %direct.schedule.7

direct.schedule.7:                                ; preds = %direct.schedule.8, %direct.schedule.6
  %.state44 = load i64, ptr %.slot8, align 4
  %129 = icmp slt i64 %.state44, 4096
  br i1 %129, label %direct.true45, label %direct.false46

direct.schedule.8:                                ; preds = %direct.true45
  %.state47 = load i64, ptr %.slot8, align 4
  %130 = srem i64 %.state47, 4096
  %.state48 = load i64, ptr %.slot8, align 4
  %131 = sdiv i64 %.state48, 4096
  %132 = srem i64 %131, 1
  %.spill.load49 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert50 = insertelement <8 x i64> poison, i64 %132, i64 0
  %.splat51 = shufflevector <8 x i64> %.splatinsert50, <8 x i64> poison, <8 x i32> zeroinitializer
  %133 = add <8 x i64> %.spill.load49, %.splat51
  %134 = add <8 x i64> zeroinitializer, %133
  %135 = add i64 0, %130
  %136 = mul <8 x i64> %134, splat (i64 4096)
  %.splatinsert52 = insertelement <8 x i64> poison, i64 %135, i64 0
  %.splat53 = shufflevector <8 x i64> %.splatinsert52, <8 x i64> poison, <8 x i32> zeroinitializer
  %137 = add <8 x i64> %136, %.splat53
  %138 = add i64 0, %130
  %139 = srem i64 %138, 4096
  %140 = add i64 0, %139
  %tile_snapshot.spill.load54 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %141 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load54, 0
  %142 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load54, 1
  %.splatinsert55 = insertelement <8 x i64> poison, i64 %140, i64 0
  %.splat56 = shufflevector <8 x i64> %.splatinsert55, <8 x i64> poison, <8 x i32> zeroinitializer
  %143 = mul <8 x i64> %.splat56, splat (i64 4)
  %144 = add <8 x i64> %142, %143
  %145 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %141, 0
  %146 = insertvalue { <8 x ptr>, <8 x i64> } %145, <8 x i64> %144, 1
  %147 = extractvalue { <8 x ptr>, <8 x i64> } %146, 0
  %148 = extractvalue { <8 x ptr>, <8 x i64> } %146, 1
  %149 = getelementptr i8, <8 x ptr> %147, <8 x i64> %148
  %150 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %149, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %151 = srem i64 %140, 4096
  %152 = add i64 0, %151
  %153 = srem i64 %152, 4096
  %154 = add i64 0, %153
  %155 = srem i64 %154, 4096
  %156 = add i64 0, %155
  %tile_snapshot.spill.load57 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %157 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load57, 0
  %158 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load57, 1
  %.splatinsert58 = insertelement <8 x i64> poison, i64 %156, i64 0
  %.splat59 = shufflevector <8 x i64> %.splatinsert58, <8 x i64> poison, <8 x i32> zeroinitializer
  %159 = mul <8 x i64> %.splat59, splat (i64 4)
  %160 = add <8 x i64> %158, %159
  %161 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %157, 0
  %162 = insertvalue { <8 x ptr>, <8 x i64> } %161, <8 x i64> %160, 1
  %163 = extractvalue { <8 x ptr>, <8 x i64> } %162, 0
  %164 = extractvalue { <8 x ptr>, <8 x i64> } %162, 1
  %165 = getelementptr i8, <8 x ptr> %163, <8 x i64> %164
  %166 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %165, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %167 = fneg <8 x float> %166
  %168 = select <8 x i1> %53, <8 x float> %167, <8 x float> zeroinitializer
  %native.exp = call <8 x float> @__luisa_cpu_native_exp_f32_v8_u10(<8 x float> %168)
  %.spill.load60 = load float, ptr %.spill5, align 4
  %.splatinsert61 = insertelement <8 x float> poison, float %.spill.load60, i64 0
  %.splat62 = shufflevector <8 x float> %.splatinsert61, <8 x float> poison, <8 x i32> zeroinitializer
  %169 = fadd <8 x float> %.splat62, %native.exp
  %170 = fdiv <8 x float> %150, %169
  %tile_snapshot.spill.load63 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill6, align 64
  %171 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load63, 0
  %172 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load63, 1
  %.splatinsert64 = insertelement <8 x i64> poison, i64 %138, i64 0
  %.splat65 = shufflevector <8 x i64> %.splatinsert64, <8 x i64> poison, <8 x i32> zeroinitializer
  %173 = mul <8 x i64> %.splat65, splat (i64 4)
  %174 = add <8 x i64> %172, %173
  %175 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %171, 0
  %176 = insertvalue { <8 x ptr>, <8 x i64> } %175, <8 x i64> %174, 1
  %177 = extractvalue { <8 x ptr>, <8 x i64> } %176, 0
  %178 = extractvalue { <8 x ptr>, <8 x i64> } %176, 1
  %179 = getelementptr i8, <8 x ptr> %177, <8 x i64> %178
  %180 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %179, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %181 = fmul <8 x float> %170, %180
  %182 = extractvalue { ptr, i64 } %29, 0
  %183 = mul <8 x i64> %137, splat (i64 4)
  %184 = getelementptr i8, ptr %182, <8 x i64> %183
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %181, <8 x ptr> %184, i32 1, <8 x i1> %53)
  %.state66 = load i64, ptr %.slot8, align 4
  %185 = add i64 %.state66, 1
  store i64 %185, ptr %.slot8, align 4
  br label %direct.schedule.7

direct.schedule.9:                                ; preds = %direct.false46
  ret void

direct.activate:                                  ; preds = %prologue
  br label %direct.schedule.0

direct.inactive:                                  ; preds = %prologue
  ret void

direct.true:                                      ; preds = %direct.schedule.1
  br label %direct.schedule.2

direct.false:                                     ; preds = %direct.schedule.1
  br label %direct.schedule.3

direct.true30:                                    ; preds = %direct.schedule.4
  br label %direct.schedule.5

direct.false31:                                   ; preds = %direct.schedule.4
  br label %direct.schedule.6

direct.true45:                                    ; preds = %direct.schedule.7
  br label %direct.schedule.8

direct.false46:                                   ; preds = %direct.schedule.7
  br label %direct.schedule.9
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.vector.reduce.or.v8i1(<8 x i1>) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr>, i32 immarg, <8 x i1>, <8 x float>) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.masked.scatter.v8f32.v8p0(<8 x float>, <8 x ptr>, i32 immarg, <8 x i1>) #2

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
  %block.origin.x = mul i64 %0, 128
  %dispatch.size.x.i64 = zext i32 %dispatch.size.x to i64
  %block.origin.in.range = icmp ule i64 %block.origin.x, %dispatch.size.x.i64
  %block.origin.safe = select i1 %block.origin.in.range, i64 %block.origin.x, i64 0
  %packet.range.start = add i64 %block.origin.safe, %base.thread.index.i64
  %1 = icmp ult i64 %packet.range.start, %dispatch.size.x.i64
  %packet.range.inside.dispatch = and i1 %block.origin.in.range, %1
  %2 = sub i64 %dispatch.size.x.i64, %packet.range.start
  %dispatch.remaining = select i1 %packet.range.inside.dispatch, i64 %2, i64 0
  %packet.range.inside.block = icmp ult i64 %base.thread.index.i64, 128
  %3 = sub i64 128, %base.thread.index.i64
  %block.remaining = select i1 %packet.range.inside.block, i64 %3, i64 0
  %4 = icmp ult i64 %dispatch.remaining, %block.remaining
  %packet.range.remaining = select i1 %4, i64 %dispatch.remaining, i64 %block.remaining
  %5 = zext i32 %packet_count to i64
  %packet.range.requested.threads = mul i64 %5, 8
  %6 = icmp ult i64 %packet.range.remaining, %packet.range.requested.threads
  %packet.range.active.threads = select i1 %6, i64 %packet.range.remaining, i64 %packet.range.requested.threads
  %7 = icmp eq i32 %packet_count, 16
  %8 = icmp uge i64 %packet.range.remaining, 128
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
  %packet.thread.index3 = add i32 %base.thread.index, 32
  store i32 %packet.thread.index3, ptr %thread.index.address, align 4
  call void @llm_rows(ptr %argument_buffer, ptr %return_lanes, ptr %launch_config, i32 8)
  %packet.thread.index4 = add i32 %base.thread.index, 40
  store i32 %packet.thread.index4, ptr %thread.index.address, align 4
  call void @llm_rows(ptr %argument_buffer, ptr %return_lanes, ptr %launch_config, i32 8)
  %packet.thread.index5 = add i32 %base.thread.index, 48
  store i32 %packet.thread.index5, ptr %thread.index.address, align 4
  call void @llm_rows(ptr %argument_buffer, ptr %return_lanes, ptr %launch_config, i32 8)
  %packet.thread.index6 = add i32 %base.thread.index, 56
  store i32 %packet.thread.index6, ptr %thread.index.address, align 4
  call void @llm_rows(ptr %argument_buffer, ptr %return_lanes, ptr %launch_config, i32 8)
  %packet.thread.index7 = add i32 %base.thread.index, 64
  store i32 %packet.thread.index7, ptr %thread.index.address, align 4
  call void @llm_rows(ptr %argument_buffer, ptr %return_lanes, ptr %launch_config, i32 8)
  %packet.thread.index8 = add i32 %base.thread.index, 72
  store i32 %packet.thread.index8, ptr %thread.index.address, align 4
  call void @llm_rows(ptr %argument_buffer, ptr %return_lanes, ptr %launch_config, i32 8)
  %packet.thread.index9 = add i32 %base.thread.index, 80
  store i32 %packet.thread.index9, ptr %thread.index.address, align 4
  call void @llm_rows(ptr %argument_buffer, ptr %return_lanes, ptr %launch_config, i32 8)
  %packet.thread.index10 = add i32 %base.thread.index, 88
  store i32 %packet.thread.index10, ptr %thread.index.address, align 4
  call void @llm_rows(ptr %argument_buffer, ptr %return_lanes, ptr %launch_config, i32 8)
  %packet.thread.index11 = add i32 %base.thread.index, 96
  store i32 %packet.thread.index11, ptr %thread.index.address, align 4
  call void @llm_rows(ptr %argument_buffer, ptr %return_lanes, ptr %launch_config, i32 8)
  %packet.thread.index12 = add i32 %base.thread.index, 104
  store i32 %packet.thread.index12, ptr %thread.index.address, align 4
  call void @llm_rows(ptr %argument_buffer, ptr %return_lanes, ptr %launch_config, i32 8)
  %packet.thread.index13 = add i32 %base.thread.index, 112
  store i32 %packet.thread.index13, ptr %thread.index.address, align 4
  call void @llm_rows(ptr %argument_buffer, ptr %return_lanes, ptr %launch_config, i32 8)
  %packet.thread.index14 = add i32 %base.thread.index, 120
  store i32 %packet.thread.index14, ptr %thread.index.address, align 4
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
  call void @llm_rows.packet_batch(ptr %argument_buffer, ptr %return_lanes, ptr %launch_config, i32 16)
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
