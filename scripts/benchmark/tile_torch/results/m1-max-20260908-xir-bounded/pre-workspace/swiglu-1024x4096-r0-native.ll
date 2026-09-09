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
  %.spill = alloca <8 x i64>, align 64
  store <8 x i64> zeroinitializer, ptr %.spill, align 64
  %tile_snapshot.spill = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill, align 64
  %.slot = alloca i64, align 8
  store i64 0, ptr %.slot, align 4
  %.spill4 = alloca float, align 4
  store float 0.000000e+00, ptr %.spill4, align 4
  %tile_snapshot.spill5 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill5, align 64
  %.slot6 = alloca i64, align 8
  store i64 0, ptr %.slot6, align 4
  %.slot7 = alloca i64, align 8
  store i64 0, ptr %.slot7, align 4
  %4 = getelementptr i8, ptr %argument_buffer, i64 0
  %5 = load ptr, ptr %4, align 16
  %6 = getelementptr i8, ptr %4, i64 8
  %7 = load i64, ptr %6, align 8
  %8 = insertvalue { ptr, i64 } poison, ptr %5, 0
  %9 = insertvalue { ptr, i64 } %8, i64 %7, 1
  %10 = getelementptr i8, ptr %argument_buffer, i64 16
  %11 = load ptr, ptr %10, align 16
  %12 = getelementptr i8, ptr %10, i64 8
  %13 = load i64, ptr %12, align 8
  %14 = insertvalue { ptr, i64 } poison, ptr %11, 0
  %15 = insertvalue { ptr, i64 } %14, i64 %13, 1
  %16 = getelementptr i8, ptr %argument_buffer, i64 32
  %17 = load ptr, ptr %16, align 16
  %18 = getelementptr i8, ptr %16, i64 8
  %19 = load i64, ptr %18, align 8
  %20 = insertvalue { ptr, i64 } poison, ptr %17, 0
  %21 = insertvalue { ptr, i64 } %20, i64 %19, 1
  %22 = getelementptr i8, ptr %argument_buffer, i64 48
  %23 = load ptr, ptr %22, align 16
  %24 = getelementptr i8, ptr %22, i64 8
  %25 = load i64, ptr %24, align 8
  %26 = insertvalue { ptr, i64 } poison, ptr %23, 0
  %27 = insertvalue { ptr, i64 } %26, i64 %25, 1
  %28 = load i32, ptr %launch_config, align 4
  %29 = getelementptr i8, ptr %launch_config, i64 12
  %30 = load i32, ptr %29, align 4
  %31 = getelementptr i8, ptr %launch_config, i64 4
  %32 = load i32, ptr %31, align 4
  %33 = getelementptr i8, ptr %launch_config, i64 16
  %34 = load i32, ptr %33, align 4
  %35 = getelementptr i8, ptr %launch_config, i64 8
  %36 = load i32, ptr %35, align 4
  %37 = getelementptr i8, ptr %launch_config, i64 20
  %38 = load i32, ptr %37, align 4
  %39 = getelementptr i8, ptr %launch_config, i64 36
  %40 = load i32, ptr %39, align 4
  %.splatinsert8 = insertelement <8 x i32> poison, i32 %40, i64 0
  %.splat9 = shufflevector <8 x i32> %.splatinsert8, <8 x i32> poison, <8 x i32> zeroinitializer
  %41 = add <8 x i32> %.splat9, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %42 = mul i32 %28, 128
  %.splatinsert10 = insertelement <8 x i32> poison, i32 %42, i64 0
  %.splat11 = shufflevector <8 x i32> %.splatinsert10, <8 x i32> poison, <8 x i32> zeroinitializer
  %43 = add <8 x i32> %.splat11, %41
  %44 = mul i32 %32, 1
  %.splatinsert12 = insertelement <8 x i32> poison, i32 %44, i64 0
  %.splat13 = shufflevector <8 x i32> %.splatinsert12, <8 x i32> poison, <8 x i32> zeroinitializer
  %45 = add <8 x i32> %.splat13, zeroinitializer
  %46 = mul i32 %36, 1
  %.splatinsert14 = insertelement <8 x i32> poison, i32 %46, i64 0
  %.splat15 = shufflevector <8 x i32> %.splatinsert14, <8 x i32> poison, <8 x i32> zeroinitializer
  %47 = add <8 x i32> %.splat15, zeroinitializer
  %48 = insertvalue [3 x <8 x i32>] poison, <8 x i32> %43, 0
  %49 = insertvalue [3 x <8 x i32>] %48, <8 x i32> %45, 1
  %50 = insertvalue [3 x <8 x i32>] %49, <8 x i32> %47, 2
  %.splatinsert16 = insertelement <8 x i32> poison, i32 %active_lane_count, i64 0
  %.splat17 = shufflevector <8 x i32> %.splatinsert16, <8 x i32> poison, <8 x i32> zeroinitializer
  %51 = icmp ult <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, %.splat17
  %52 = call i1 @llvm.vector.reduce.or.v8i1(<8 x i1> %51)
  br i1 %52, label %direct.activate, label %direct.inactive

direct.schedule.0:                                ; preds = %direct.activate
  %53 = extractvalue [3 x <8 x i32>] %50, 0
  %54 = zext <8 x i32> %53 to <8 x i64>
  %55 = select <8 x i1> %51, <8 x i64> %54, <8 x i64> zeroinitializer
  %56 = select <8 x i1> %51, <8 x i64> splat (i64 1), <8 x i64> splat (i64 1)
  %57 = sdiv <8 x i64> %55, %56
  %58 = select <8 x i1> %51, <8 x i64> %57, <8 x i64> zeroinitializer
  %59 = select <8 x i1> %51, <8 x i64> splat (i64 1024), <8 x i64> splat (i64 1)
  %60 = srem <8 x i64> %58, %59
  %61 = load <8 x i64>, ptr %.spill, align 64
  %62 = select <8 x i1> %51, <8 x i64> %60, <8 x i64> %61
  store <8 x i64> %62, ptr %.spill, align 64
  %63 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %64 = extractvalue { <8 x ptr>, <8 x i64> } %1, 0
  %65 = extractvalue { <8 x ptr>, <8 x i64> } %63, 0
  %66 = select <8 x i1> %51, <8 x ptr> %64, <8 x ptr> %65
  %67 = extractvalue { <8 x ptr>, <8 x i64> } %1, 1
  %68 = extractvalue { <8 x ptr>, <8 x i64> } %63, 1
  %69 = select <8 x i1> %51, <8 x i64> %67, <8 x i64> %68
  %70 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %66, 0
  %71 = insertvalue { <8 x ptr>, <8 x i64> } %70, <8 x i64> %69, 1
  store { <8 x ptr>, <8 x i64> } %71, ptr %tile_snapshot.spill, align 64
  store i64 0, ptr %.slot, align 4
  br label %direct.schedule.1

direct.schedule.1:                                ; preds = %direct.schedule.2, %direct.schedule.0
  %.state = load i64, ptr %.slot, align 4
  %72 = icmp slt i64 %.state, 4096
  br i1 %72, label %direct.true, label %direct.false

direct.schedule.2:                                ; preds = %direct.true
  %.state18 = load i64, ptr %.slot, align 4
  %73 = srem i64 %.state18, 4096
  %.state19 = load i64, ptr %.slot, align 4
  %74 = sdiv i64 %.state19, 4096
  %75 = srem i64 %74, 1
  %.spill.load = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert20 = insertelement <8 x i64> poison, i64 %75, i64 0
  %.splat21 = shufflevector <8 x i64> %.splatinsert20, <8 x i64> poison, <8 x i32> zeroinitializer
  %76 = add <8 x i64> %.spill.load, %.splat21
  %77 = add <8 x i64> zeroinitializer, %76
  %78 = add i64 0, %73
  %79 = mul <8 x i64> %77, splat (i64 4096)
  %.splatinsert22 = insertelement <8 x i64> poison, i64 %78, i64 0
  %.splat23 = shufflevector <8 x i64> %.splatinsert22, <8 x i64> poison, <8 x i32> zeroinitializer
  %80 = add <8 x i64> %79, %.splat23
  %81 = extractvalue { ptr, i64 } %9, 0
  %82 = mul <8 x i64> %80, splat (i64 4)
  %83 = getelementptr i8, ptr %81, <8 x i64> %82
  %84 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %83, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %85 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 0
  %86 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 1
  %.state24 = load i64, ptr %.slot, align 4
  %.splatinsert25 = insertelement <8 x i64> poison, i64 %.state24, i64 0
  %.splat26 = shufflevector <8 x i64> %.splatinsert25, <8 x i64> poison, <8 x i32> zeroinitializer
  %87 = mul <8 x i64> %.splat26, splat (i64 4)
  %88 = add <8 x i64> %86, %87
  %89 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %85, 0
  %90 = insertvalue { <8 x ptr>, <8 x i64> } %89, <8 x i64> %88, 1
  %91 = extractvalue { <8 x ptr>, <8 x i64> } %90, 0
  %92 = extractvalue { <8 x ptr>, <8 x i64> } %90, 1
  %93 = getelementptr i8, <8 x ptr> %91, <8 x i64> %92
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %84, <8 x ptr> %93, i32 1, <8 x i1> %51)
  %.state27 = load i64, ptr %.slot, align 4
  %94 = add i64 %.state27, 1
  store i64 %94, ptr %.slot, align 4
  br label %direct.schedule.1

direct.schedule.3:                                ; preds = %direct.false
  store float 1.000000e+00, ptr %.spill4, align 4
  %95 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill5, align 64
  %96 = extractvalue { <8 x ptr>, <8 x i64> } %3, 0
  %97 = extractvalue { <8 x ptr>, <8 x i64> } %95, 0
  %98 = select <8 x i1> %51, <8 x ptr> %96, <8 x ptr> %97
  %99 = extractvalue { <8 x ptr>, <8 x i64> } %3, 1
  %100 = extractvalue { <8 x ptr>, <8 x i64> } %95, 1
  %101 = select <8 x i1> %51, <8 x i64> %99, <8 x i64> %100
  %102 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %98, 0
  %103 = insertvalue { <8 x ptr>, <8 x i64> } %102, <8 x i64> %101, 1
  store { <8 x ptr>, <8 x i64> } %103, ptr %tile_snapshot.spill5, align 64
  store i64 0, ptr %.slot6, align 4
  br label %direct.schedule.4

direct.schedule.4:                                ; preds = %direct.schedule.5, %direct.schedule.3
  %.state28 = load i64, ptr %.slot6, align 4
  %104 = icmp slt i64 %.state28, 4096
  br i1 %104, label %direct.true29, label %direct.false30

direct.schedule.5:                                ; preds = %direct.true29
  %.state31 = load i64, ptr %.slot6, align 4
  %105 = srem i64 %.state31, 4096
  %.state32 = load i64, ptr %.slot6, align 4
  %106 = sdiv i64 %.state32, 4096
  %107 = srem i64 %106, 1
  %.spill.load33 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert34 = insertelement <8 x i64> poison, i64 %107, i64 0
  %.splat35 = shufflevector <8 x i64> %.splatinsert34, <8 x i64> poison, <8 x i32> zeroinitializer
  %108 = add <8 x i64> %.spill.load33, %.splat35
  %109 = add <8 x i64> zeroinitializer, %108
  %110 = add i64 0, %105
  %111 = mul <8 x i64> %109, splat (i64 4096)
  %.splatinsert36 = insertelement <8 x i64> poison, i64 %110, i64 0
  %.splat37 = shufflevector <8 x i64> %.splatinsert36, <8 x i64> poison, <8 x i32> zeroinitializer
  %112 = add <8 x i64> %111, %.splat37
  %113 = extractvalue { ptr, i64 } %15, 0
  %114 = mul <8 x i64> %112, splat (i64 4)
  %115 = getelementptr i8, ptr %113, <8 x i64> %114
  %116 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %115, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load38 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill5, align 64
  %117 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load38, 0
  %118 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load38, 1
  %.state39 = load i64, ptr %.slot6, align 4
  %.splatinsert40 = insertelement <8 x i64> poison, i64 %.state39, i64 0
  %.splat41 = shufflevector <8 x i64> %.splatinsert40, <8 x i64> poison, <8 x i32> zeroinitializer
  %119 = mul <8 x i64> %.splat41, splat (i64 4)
  %120 = add <8 x i64> %118, %119
  %121 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %117, 0
  %122 = insertvalue { <8 x ptr>, <8 x i64> } %121, <8 x i64> %120, 1
  %123 = extractvalue { <8 x ptr>, <8 x i64> } %122, 0
  %124 = extractvalue { <8 x ptr>, <8 x i64> } %122, 1
  %125 = getelementptr i8, <8 x ptr> %123, <8 x i64> %124
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %116, <8 x ptr> %125, i32 1, <8 x i1> %51)
  %.state42 = load i64, ptr %.slot6, align 4
  %126 = add i64 %.state42, 1
  store i64 %126, ptr %.slot6, align 4
  br label %direct.schedule.4

direct.schedule.6:                                ; preds = %direct.false30
  store i64 0, ptr %.slot7, align 4
  br label %direct.schedule.7

direct.schedule.7:                                ; preds = %direct.schedule.8, %direct.schedule.6
  %.state43 = load i64, ptr %.slot7, align 4
  %127 = icmp slt i64 %.state43, 4096
  br i1 %127, label %direct.true44, label %direct.false45

direct.schedule.8:                                ; preds = %direct.true44
  %.state46 = load i64, ptr %.slot7, align 4
  %128 = srem i64 %.state46, 4096
  %.state47 = load i64, ptr %.slot7, align 4
  %129 = sdiv i64 %.state47, 4096
  %130 = srem i64 %129, 1
  %.spill.load48 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert49 = insertelement <8 x i64> poison, i64 %130, i64 0
  %.splat50 = shufflevector <8 x i64> %.splatinsert49, <8 x i64> poison, <8 x i32> zeroinitializer
  %131 = add <8 x i64> %.spill.load48, %.splat50
  %132 = add <8 x i64> zeroinitializer, %131
  %133 = add i64 0, %128
  %134 = mul <8 x i64> %132, splat (i64 4096)
  %.splatinsert51 = insertelement <8 x i64> poison, i64 %133, i64 0
  %.splat52 = shufflevector <8 x i64> %.splatinsert51, <8 x i64> poison, <8 x i32> zeroinitializer
  %135 = add <8 x i64> %134, %.splat52
  %136 = add i64 0, %128
  %137 = srem i64 %136, 4096
  %138 = add i64 0, %137
  %tile_snapshot.spill.load53 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %139 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load53, 0
  %140 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load53, 1
  %.splatinsert54 = insertelement <8 x i64> poison, i64 %138, i64 0
  %.splat55 = shufflevector <8 x i64> %.splatinsert54, <8 x i64> poison, <8 x i32> zeroinitializer
  %141 = mul <8 x i64> %.splat55, splat (i64 4)
  %142 = add <8 x i64> %140, %141
  %143 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %139, 0
  %144 = insertvalue { <8 x ptr>, <8 x i64> } %143, <8 x i64> %142, 1
  %145 = extractvalue { <8 x ptr>, <8 x i64> } %144, 0
  %146 = extractvalue { <8 x ptr>, <8 x i64> } %144, 1
  %147 = getelementptr i8, <8 x ptr> %145, <8 x i64> %146
  %148 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %147, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %149 = srem i64 %138, 4096
  %150 = add i64 0, %149
  %151 = srem i64 %150, 4096
  %152 = add i64 0, %151
  %153 = srem i64 %152, 4096
  %154 = add i64 0, %153
  %tile_snapshot.spill.load56 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %155 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load56, 0
  %156 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load56, 1
  %.splatinsert57 = insertelement <8 x i64> poison, i64 %154, i64 0
  %.splat58 = shufflevector <8 x i64> %.splatinsert57, <8 x i64> poison, <8 x i32> zeroinitializer
  %157 = mul <8 x i64> %.splat58, splat (i64 4)
  %158 = add <8 x i64> %156, %157
  %159 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %155, 0
  %160 = insertvalue { <8 x ptr>, <8 x i64> } %159, <8 x i64> %158, 1
  %161 = extractvalue { <8 x ptr>, <8 x i64> } %160, 0
  %162 = extractvalue { <8 x ptr>, <8 x i64> } %160, 1
  %163 = getelementptr i8, <8 x ptr> %161, <8 x i64> %162
  %164 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %163, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %165 = fneg <8 x float> %164
  %166 = select <8 x i1> %51, <8 x float> %165, <8 x float> zeroinitializer
  %native.exp = call <8 x float> @__luisa_cpu_native_exp_f32_v8_u10(<8 x float> %166)
  %.spill.load59 = load float, ptr %.spill4, align 4
  %.splatinsert60 = insertelement <8 x float> poison, float %.spill.load59, i64 0
  %.splat61 = shufflevector <8 x float> %.splatinsert60, <8 x float> poison, <8 x i32> zeroinitializer
  %167 = fadd <8 x float> %.splat61, %native.exp
  %168 = fdiv <8 x float> %148, %167
  %tile_snapshot.spill.load62 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill5, align 64
  %169 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load62, 0
  %170 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load62, 1
  %.splatinsert63 = insertelement <8 x i64> poison, i64 %136, i64 0
  %.splat64 = shufflevector <8 x i64> %.splatinsert63, <8 x i64> poison, <8 x i32> zeroinitializer
  %171 = mul <8 x i64> %.splat64, splat (i64 4)
  %172 = add <8 x i64> %170, %171
  %173 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %169, 0
  %174 = insertvalue { <8 x ptr>, <8 x i64> } %173, <8 x i64> %172, 1
  %175 = extractvalue { <8 x ptr>, <8 x i64> } %174, 0
  %176 = extractvalue { <8 x ptr>, <8 x i64> } %174, 1
  %177 = getelementptr i8, <8 x ptr> %175, <8 x i64> %176
  %178 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %177, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %179 = fmul <8 x float> %168, %178
  %180 = extractvalue { ptr, i64 } %27, 0
  %181 = mul <8 x i64> %135, splat (i64 4)
  %182 = getelementptr i8, ptr %180, <8 x i64> %181
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %179, <8 x ptr> %182, i32 1, <8 x i1> %51)
  %.state65 = load i64, ptr %.slot7, align 4
  %183 = add i64 %.state65, 1
  store i64 %183, ptr %.slot7, align 4
  br label %direct.schedule.7

direct.schedule.9:                                ; preds = %direct.false45
  ret void

direct.activate:                                  ; preds = %prologue
  br label %direct.schedule.0

direct.inactive:                                  ; preds = %prologue
  ret void

direct.true:                                      ; preds = %direct.schedule.1
  br label %direct.schedule.2

direct.false:                                     ; preds = %direct.schedule.1
  br label %direct.schedule.3

direct.true29:                                    ; preds = %direct.schedule.4
  br label %direct.schedule.5

direct.false30:                                   ; preds = %direct.schedule.4
  br label %direct.schedule.6

direct.true44:                                    ; preds = %direct.schedule.7
  br label %direct.schedule.8

direct.false45:                                   ; preds = %direct.schedule.7
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
