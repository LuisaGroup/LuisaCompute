; ModuleID = 'luisa-simd-kernel'
source_filename = "luisa-simd-kernel"

define internal void @llm_rows(ptr noalias readonly %argument_buffer, ptr %return_lanes, ptr noalias nonnull readonly %launch_config, i32 %active_lane_count) {
prologue:
  %tile_snapshot.local = alloca [49184 x i8], align 4
  %.splatinsert = insertelement <8 x ptr> poison, ptr %tile_snapshot.local, i64 0
  %.splat = shufflevector <8 x ptr> %.splatinsert, <8 x ptr> poison, <8 x i32> zeroinitializer
  %0 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat, 0
  %1 = insertvalue { <8 x ptr>, <8 x i64> } %0, <8 x i64> <i64 0, i64 6148, i64 12296, i64 18444, i64 24592, i64 30740, i64 36888, i64 43036>, 1
  %tile_snapshot.local1 = alloca [49184 x i8], align 4
  %.splatinsert2 = insertelement <8 x ptr> poison, ptr %tile_snapshot.local1, i64 0
  %.splat3 = shufflevector <8 x ptr> %.splatinsert2, <8 x ptr> poison, <8 x i32> zeroinitializer
  %2 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat3, 0
  %3 = insertvalue { <8 x ptr>, <8 x i64> } %2, <8 x i64> <i64 0, i64 6148, i64 12296, i64 18444, i64 24592, i64 30740, i64 36888, i64 43036>, 1
  %.spill = alloca <8 x i64>, align 64
  store <8 x i64> zeroinitializer, ptr %.spill, align 64
  %tile_snapshot.spill = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill, align 64
  %.slot = alloca i64, align 8
  store i64 0, ptr %.slot, align 4
  %.spill4 = alloca float, align 4
  store float 0.000000e+00, ptr %.spill4, align 4
  %.spill5 = alloca float, align 4
  store float 0.000000e+00, ptr %.spill5, align 4
  %.spill6 = alloca float, align 4
  store float 0.000000e+00, ptr %.spill6, align 4
  %.spill7 = alloca float, align 4
  store float 0.000000e+00, ptr %.spill7, align 4
  %tile_snapshot.spill8 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill8, align 64
  %.slot9 = alloca i64, align 8
  store i64 0, ptr %.slot9, align 4
  %.slot10 = alloca i64, align 8
  store i64 0, ptr %.slot10, align 4
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
  %.splatinsert11 = insertelement <8 x i32> poison, i32 %40, i64 0
  %.splat12 = shufflevector <8 x i32> %.splatinsert11, <8 x i32> poison, <8 x i32> zeroinitializer
  %41 = add <8 x i32> %.splat12, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %42 = mul i32 %28, 32
  %.splatinsert13 = insertelement <8 x i32> poison, i32 %42, i64 0
  %.splat14 = shufflevector <8 x i32> %.splatinsert13, <8 x i32> poison, <8 x i32> zeroinitializer
  %43 = add <8 x i32> %.splat14, %41
  %44 = mul i32 %32, 1
  %.splatinsert15 = insertelement <8 x i32> poison, i32 %44, i64 0
  %.splat16 = shufflevector <8 x i32> %.splatinsert15, <8 x i32> poison, <8 x i32> zeroinitializer
  %45 = add <8 x i32> %.splat16, zeroinitializer
  %46 = mul i32 %36, 1
  %.splatinsert17 = insertelement <8 x i32> poison, i32 %46, i64 0
  %.splat18 = shufflevector <8 x i32> %.splatinsert17, <8 x i32> poison, <8 x i32> zeroinitializer
  %47 = add <8 x i32> %.splat18, zeroinitializer
  %48 = insertvalue [3 x <8 x i32>] poison, <8 x i32> %43, 0
  %49 = insertvalue [3 x <8 x i32>] %48, <8 x i32> %45, 1
  %50 = insertvalue [3 x <8 x i32>] %49, <8 x i32> %47, 2
  %.splatinsert19 = insertelement <8 x i32> poison, i32 %active_lane_count, i64 0
  %.splat20 = shufflevector <8 x i32> %.splatinsert19, <8 x i32> poison, <8 x i32> zeroinitializer
  %51 = icmp ult <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, %.splat20
  %52 = call i1 @llvm.vector.reduce.or.v8i1(<8 x i1> %51)
  br i1 %52, label %direct.activate, label %direct.inactive

direct.schedule.0:                                ; preds = %direct.activate
  %53 = extractvalue [3 x <8 x i32>] %50, 0
  %54 = zext <8 x i32> %53 to <8 x i64>
  %55 = select <8 x i1> %51, <8 x i64> %54, <8 x i64> zeroinitializer
  %56 = select <8 x i1> %51, <8 x i64> splat (i64 1), <8 x i64> splat (i64 1)
  %57 = sdiv <8 x i64> %55, %56
  %58 = select <8 x i1> %51, <8 x i64> %57, <8 x i64> zeroinitializer
  %59 = select <8 x i1> %51, <8 x i64> splat (i64 64), <8 x i64> splat (i64 1)
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
  %72 = icmp slt i64 %.state, 1537
  br i1 %72, label %direct.true, label %direct.false

direct.schedule.2:                                ; preds = %direct.true
  %.state21 = load i64, ptr %.slot, align 4
  %73 = srem i64 %.state21, 1537
  %.state22 = load i64, ptr %.slot, align 4
  %74 = sdiv i64 %.state22, 1537
  %75 = srem i64 %74, 1
  %.spill.load = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert23 = insertelement <8 x i64> poison, i64 %75, i64 0
  %.splat24 = shufflevector <8 x i64> %.splatinsert23, <8 x i64> poison, <8 x i32> zeroinitializer
  %76 = add <8 x i64> %.spill.load, %.splat24
  %77 = add <8 x i64> zeroinitializer, %76
  %78 = add i64 0, %73
  %79 = mul <8 x i64> %77, splat (i64 1537)
  %.splatinsert25 = insertelement <8 x i64> poison, i64 %78, i64 0
  %.splat26 = shufflevector <8 x i64> %.splatinsert25, <8 x i64> poison, <8 x i32> zeroinitializer
  %80 = add <8 x i64> %79, %.splat26
  %81 = extractvalue { ptr, i64 } %9, 0
  %82 = mul <8 x i64> %80, splat (i64 4)
  %83 = getelementptr i8, ptr %81, <8 x i64> %82
  %84 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %83, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %85 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 0
  %86 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 1
  %.state27 = load i64, ptr %.slot, align 4
  %.splatinsert28 = insertelement <8 x i64> poison, i64 %.state27, i64 0
  %.splat29 = shufflevector <8 x i64> %.splatinsert28, <8 x i64> poison, <8 x i32> zeroinitializer
  %87 = mul <8 x i64> %.splat29, splat (i64 4)
  %88 = add <8 x i64> %86, %87
  %89 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %85, 0
  %90 = insertvalue { <8 x ptr>, <8 x i64> } %89, <8 x i64> %88, 1
  %91 = extractvalue { <8 x ptr>, <8 x i64> } %90, 0
  %92 = extractvalue { <8 x ptr>, <8 x i64> } %90, 1
  %93 = getelementptr i8, <8 x ptr> %91, <8 x i64> %92
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %84, <8 x ptr> %93, i32 1, <8 x i1> %51)
  %.state30 = load i64, ptr %.slot, align 4
  %94 = add i64 %.state30, 1
  store i64 %94, ptr %.slot, align 4
  br label %direct.schedule.1

direct.schedule.3:                                ; preds = %direct.false
  store float 5.000000e-01, ptr %.spill4, align 4
  store float 0x3FA6E4E260000000, ptr %.spill5, align 4
  store float 0x3FE9884540000000, ptr %.spill6, align 4
  store float 1.000000e+00, ptr %.spill7, align 4
  %95 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill8, align 64
  %96 = extractvalue { <8 x ptr>, <8 x i64> } %3, 0
  %97 = extractvalue { <8 x ptr>, <8 x i64> } %95, 0
  %98 = select <8 x i1> %51, <8 x ptr> %96, <8 x ptr> %97
  %99 = extractvalue { <8 x ptr>, <8 x i64> } %3, 1
  %100 = extractvalue { <8 x ptr>, <8 x i64> } %95, 1
  %101 = select <8 x i1> %51, <8 x i64> %99, <8 x i64> %100
  %102 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %98, 0
  %103 = insertvalue { <8 x ptr>, <8 x i64> } %102, <8 x i64> %101, 1
  store { <8 x ptr>, <8 x i64> } %103, ptr %tile_snapshot.spill8, align 64
  store i64 0, ptr %.slot9, align 4
  br label %direct.schedule.4

direct.schedule.4:                                ; preds = %direct.schedule.5, %direct.schedule.3
  %.state31 = load i64, ptr %.slot9, align 4
  %104 = icmp slt i64 %.state31, 1537
  br i1 %104, label %direct.true32, label %direct.false33

direct.schedule.5:                                ; preds = %direct.true32
  %.state34 = load i64, ptr %.slot9, align 4
  %105 = srem i64 %.state34, 1537
  %.state35 = load i64, ptr %.slot9, align 4
  %106 = sdiv i64 %.state35, 1537
  %107 = srem i64 %106, 1
  %.spill.load36 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert37 = insertelement <8 x i64> poison, i64 %107, i64 0
  %.splat38 = shufflevector <8 x i64> %.splatinsert37, <8 x i64> poison, <8 x i32> zeroinitializer
  %108 = add <8 x i64> %.spill.load36, %.splat38
  %109 = add <8 x i64> zeroinitializer, %108
  %110 = add i64 0, %105
  %111 = mul <8 x i64> %109, splat (i64 1537)
  %.splatinsert39 = insertelement <8 x i64> poison, i64 %110, i64 0
  %.splat40 = shufflevector <8 x i64> %.splatinsert39, <8 x i64> poison, <8 x i32> zeroinitializer
  %112 = add <8 x i64> %111, %.splat40
  %113 = extractvalue { ptr, i64 } %15, 0
  %114 = mul <8 x i64> %112, splat (i64 4)
  %115 = getelementptr i8, ptr %113, <8 x i64> %114
  %116 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %115, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load41 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill8, align 64
  %117 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load41, 0
  %118 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load41, 1
  %.state42 = load i64, ptr %.slot9, align 4
  %.splatinsert43 = insertelement <8 x i64> poison, i64 %.state42, i64 0
  %.splat44 = shufflevector <8 x i64> %.splatinsert43, <8 x i64> poison, <8 x i32> zeroinitializer
  %119 = mul <8 x i64> %.splat44, splat (i64 4)
  %120 = add <8 x i64> %118, %119
  %121 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %117, 0
  %122 = insertvalue { <8 x ptr>, <8 x i64> } %121, <8 x i64> %120, 1
  %123 = extractvalue { <8 x ptr>, <8 x i64> } %122, 0
  %124 = extractvalue { <8 x ptr>, <8 x i64> } %122, 1
  %125 = getelementptr i8, <8 x ptr> %123, <8 x i64> %124
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %116, <8 x ptr> %125, i32 1, <8 x i1> %51)
  %.state45 = load i64, ptr %.slot9, align 4
  %126 = add i64 %.state45, 1
  store i64 %126, ptr %.slot9, align 4
  br label %direct.schedule.4

direct.schedule.6:                                ; preds = %direct.false33
  store i64 0, ptr %.slot10, align 4
  br label %direct.schedule.7

direct.schedule.7:                                ; preds = %direct.schedule.8, %direct.schedule.6
  %.state46 = load i64, ptr %.slot10, align 4
  %127 = icmp slt i64 %.state46, 1537
  br i1 %127, label %direct.true47, label %direct.false48

direct.schedule.8:                                ; preds = %direct.true47
  %.state49 = load i64, ptr %.slot10, align 4
  %128 = srem i64 %.state49, 1537
  %.state50 = load i64, ptr %.slot10, align 4
  %129 = sdiv i64 %.state50, 1537
  %130 = srem i64 %129, 1
  %.spill.load51 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert52 = insertelement <8 x i64> poison, i64 %130, i64 0
  %.splat53 = shufflevector <8 x i64> %.splatinsert52, <8 x i64> poison, <8 x i32> zeroinitializer
  %131 = add <8 x i64> %.spill.load51, %.splat53
  %132 = add <8 x i64> zeroinitializer, %131
  %133 = add i64 0, %128
  %134 = mul <8 x i64> %132, splat (i64 1537)
  %.splatinsert54 = insertelement <8 x i64> poison, i64 %133, i64 0
  %.splat55 = shufflevector <8 x i64> %.splatinsert54, <8 x i64> poison, <8 x i32> zeroinitializer
  %135 = add <8 x i64> %134, %.splat55
  %136 = add i64 0, %128
  %137 = srem i64 %136, 1537
  %138 = add i64 0, %137
  %139 = srem i64 %138, 1537
  %140 = add i64 0, %139
  %tile_snapshot.spill.load56 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %141 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load56, 0
  %142 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load56, 1
  %.splatinsert57 = insertelement <8 x i64> poison, i64 %140, i64 0
  %.splat58 = shufflevector <8 x i64> %.splatinsert57, <8 x i64> poison, <8 x i32> zeroinitializer
  %143 = mul <8 x i64> %.splat58, splat (i64 4)
  %144 = add <8 x i64> %142, %143
  %145 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %141, 0
  %146 = insertvalue { <8 x ptr>, <8 x i64> } %145, <8 x i64> %144, 1
  %147 = extractvalue { <8 x ptr>, <8 x i64> } %146, 0
  %148 = extractvalue { <8 x ptr>, <8 x i64> } %146, 1
  %149 = getelementptr i8, <8 x ptr> %147, <8 x i64> %148
  %150 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %149, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %.spill.load59 = load float, ptr %.spill4, align 4
  %.splatinsert60 = insertelement <8 x float> poison, float %.spill.load59, i64 0
  %.splat61 = shufflevector <8 x float> %.splatinsert60, <8 x float> poison, <8 x i32> zeroinitializer
  %151 = fmul <8 x float> %.splat61, %150
  %152 = srem i64 %140, 1537
  %153 = add i64 0, %152
  %154 = srem i64 %153, 1537
  %155 = add i64 0, %154
  %156 = srem i64 %155, 1537
  %157 = add i64 0, %156
  %tile_snapshot.spill.load62 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %158 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load62, 0
  %159 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load62, 1
  %.splatinsert63 = insertelement <8 x i64> poison, i64 %157, i64 0
  %.splat64 = shufflevector <8 x i64> %.splatinsert63, <8 x i64> poison, <8 x i32> zeroinitializer
  %160 = mul <8 x i64> %.splat64, splat (i64 4)
  %161 = add <8 x i64> %159, %160
  %162 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %158, 0
  %163 = insertvalue { <8 x ptr>, <8 x i64> } %162, <8 x i64> %161, 1
  %164 = extractvalue { <8 x ptr>, <8 x i64> } %163, 0
  %165 = extractvalue { <8 x ptr>, <8 x i64> } %163, 1
  %166 = getelementptr i8, <8 x ptr> %164, <8 x i64> %165
  %167 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %166, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %168 = srem i64 %157, 1537
  %169 = add i64 0, %168
  %170 = srem i64 %169, 1537
  %171 = add i64 0, %170
  %172 = srem i64 %171, 1537
  %173 = add i64 0, %172
  %tile_snapshot.spill.load65 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %174 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load65, 0
  %175 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load65, 1
  %.splatinsert66 = insertelement <8 x i64> poison, i64 %173, i64 0
  %.splat67 = shufflevector <8 x i64> %.splatinsert66, <8 x i64> poison, <8 x i32> zeroinitializer
  %176 = mul <8 x i64> %.splat67, splat (i64 4)
  %177 = add <8 x i64> %175, %176
  %178 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %174, 0
  %179 = insertvalue { <8 x ptr>, <8 x i64> } %178, <8 x i64> %177, 1
  %180 = extractvalue { <8 x ptr>, <8 x i64> } %179, 0
  %181 = extractvalue { <8 x ptr>, <8 x i64> } %179, 1
  %182 = getelementptr i8, <8 x ptr> %180, <8 x i64> %181
  %183 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %182, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %.spill.load68 = load float, ptr %.spill5, align 4
  %.splatinsert69 = insertelement <8 x float> poison, float %.spill.load68, i64 0
  %.splat70 = shufflevector <8 x float> %.splatinsert69, <8 x float> poison, <8 x i32> zeroinitializer
  %184 = fmul <8 x float> %.splat70, %183
  %tile_snapshot.spill.load71 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %185 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load71, 0
  %186 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load71, 1
  %.splatinsert72 = insertelement <8 x i64> poison, i64 %171, i64 0
  %.splat73 = shufflevector <8 x i64> %.splatinsert72, <8 x i64> poison, <8 x i32> zeroinitializer
  %187 = mul <8 x i64> %.splat73, splat (i64 4)
  %188 = add <8 x i64> %186, %187
  %189 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %185, 0
  %190 = insertvalue { <8 x ptr>, <8 x i64> } %189, <8 x i64> %188, 1
  %191 = extractvalue { <8 x ptr>, <8 x i64> } %190, 0
  %192 = extractvalue { <8 x ptr>, <8 x i64> } %190, 1
  %193 = getelementptr i8, <8 x ptr> %191, <8 x i64> %192
  %194 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %193, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %195 = fmul <8 x float> %184, %194
  %tile_snapshot.spill.load74 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %196 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load74, 0
  %197 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load74, 1
  %.splatinsert75 = insertelement <8 x i64> poison, i64 %169, i64 0
  %.splat76 = shufflevector <8 x i64> %.splatinsert75, <8 x i64> poison, <8 x i32> zeroinitializer
  %198 = mul <8 x i64> %.splat76, splat (i64 4)
  %199 = add <8 x i64> %197, %198
  %200 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %196, 0
  %201 = insertvalue { <8 x ptr>, <8 x i64> } %200, <8 x i64> %199, 1
  %202 = extractvalue { <8 x ptr>, <8 x i64> } %201, 0
  %203 = extractvalue { <8 x ptr>, <8 x i64> } %201, 1
  %204 = getelementptr i8, <8 x ptr> %202, <8 x i64> %203
  %205 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %204, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %206 = fmul <8 x float> %195, %205
  %207 = fadd <8 x float> %167, %206
  %.spill.load77 = load float, ptr %.spill6, align 4
  %.splatinsert78 = insertelement <8 x float> poison, float %.spill.load77, i64 0
  %.splat79 = shufflevector <8 x float> %.splatinsert78, <8 x float> poison, <8 x i32> zeroinitializer
  %208 = fmul <8 x float> %.splat79, %207
  %209 = select <8 x i1> %51, <8 x float> %208, <8 x float> zeroinitializer
  %native.tanh = call <8 x float> @__luisa_cpu_native_tanh_f32_v8_precise(<8 x float> %209)
  %.spill.load80 = load float, ptr %.spill7, align 4
  %.splatinsert81 = insertelement <8 x float> poison, float %.spill.load80, i64 0
  %.splat82 = shufflevector <8 x float> %.splatinsert81, <8 x float> poison, <8 x i32> zeroinitializer
  %210 = fadd <8 x float> %.splat82, %native.tanh
  %211 = fmul <8 x float> %151, %210
  %tile_snapshot.spill.load83 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill8, align 64
  %212 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load83, 0
  %213 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load83, 1
  %.splatinsert84 = insertelement <8 x i64> poison, i64 %136, i64 0
  %.splat85 = shufflevector <8 x i64> %.splatinsert84, <8 x i64> poison, <8 x i32> zeroinitializer
  %214 = mul <8 x i64> %.splat85, splat (i64 4)
  %215 = add <8 x i64> %213, %214
  %216 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %212, 0
  %217 = insertvalue { <8 x ptr>, <8 x i64> } %216, <8 x i64> %215, 1
  %218 = extractvalue { <8 x ptr>, <8 x i64> } %217, 0
  %219 = extractvalue { <8 x ptr>, <8 x i64> } %217, 1
  %220 = getelementptr i8, <8 x ptr> %218, <8 x i64> %219
  %221 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %220, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %222 = fadd <8 x float> %211, %221
  %223 = extractvalue { ptr, i64 } %27, 0
  %224 = mul <8 x i64> %135, splat (i64 4)
  %225 = getelementptr i8, ptr %223, <8 x i64> %224
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %222, <8 x ptr> %225, i32 1, <8 x i1> %51)
  %.state86 = load i64, ptr %.slot10, align 4
  %226 = add i64 %.state86, 1
  store i64 %226, ptr %.slot10, align 4
  br label %direct.schedule.7

direct.schedule.9:                                ; preds = %direct.false48
  ret void

direct.activate:                                  ; preds = %prologue
  br label %direct.schedule.0

direct.inactive:                                  ; preds = %prologue
  ret void

direct.true:                                      ; preds = %direct.schedule.1
  br label %direct.schedule.2

direct.false:                                     ; preds = %direct.schedule.1
  br label %direct.schedule.3

direct.true32:                                    ; preds = %direct.schedule.4
  br label %direct.schedule.5

direct.false33:                                   ; preds = %direct.schedule.4
  br label %direct.schedule.6

direct.true47:                                    ; preds = %direct.schedule.7
  br label %direct.schedule.8

direct.false48:                                   ; preds = %direct.schedule.7
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
