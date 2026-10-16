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
  %.spill6 = alloca i64, align 8
  store i64 0, ptr %.spill6, align 4
  %.spill7 = alloca i64, align 8
  store i64 0, ptr %.spill7, align 4
  %.spill8 = alloca i64, align 8
  store i64 0, ptr %.spill8, align 4
  %.slot9 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot9, align 32
  %.spill10 = alloca i64, align 8
  store i64 0, ptr %.spill10, align 4
  %.slot11 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot11, align 32
  %.spill12 = alloca i64, align 8
  store i64 0, ptr %.spill12, align 4
  %.slot13 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot13, align 32
  %.spill14 = alloca i64, align 8
  store i64 0, ptr %.spill14, align 4
  %.slot15 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot15, align 32
  %.slot16 = alloca i64, align 8
  store i64 0, ptr %.slot16, align 4
  %.slot17 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot17, align 32
  %.slot18 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot18, align 32
  %.slot19 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot19, align 32
  %.slot20 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot20, align 32
  %.spill21 = alloca i64, align 8
  store i64 0, ptr %.spill21, align 4
  %.slot22 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot22, align 32
  %.spill23 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill23, align 32
  %.spill24 = alloca i64, align 8
  store i64 0, ptr %.spill24, align 4
  %.slot25 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot25, align 32
  %.spill26 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill26, align 32
  %.spill27 = alloca i64, align 8
  store i64 0, ptr %.spill27, align 4
  %.slot28 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot28, align 32
  %.spill29 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill29, align 32
  %.spill30 = alloca i64, align 8
  store i64 0, ptr %.spill30, align 4
  %.slot31 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot31, align 32
  %.spill32 = alloca i64, align 8
  store i64 0, ptr %.spill32, align 4
  %.slot33 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot33, align 32
  %.spill34 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill34, align 32
  %tile_snapshot.spill35 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill35, align 64
  %.slot36 = alloca i64, align 8
  store i64 0, ptr %.slot36, align 4
  %.spill37 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill37, align 32
  %.slot38 = alloca i64, align 8
  store i64 0, ptr %.slot38, align 4
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
  %.splatinsert39 = insertelement <8 x i32> poison, i32 %42, i64 0
  %.splat40 = shufflevector <8 x i32> %.splatinsert39, <8 x i32> poison, <8 x i32> zeroinitializer
  %43 = add <8 x i32> %.splat40, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %44 = mul i32 %30, 32
  %.splatinsert41 = insertelement <8 x i32> poison, i32 %44, i64 0
  %.splat42 = shufflevector <8 x i32> %.splatinsert41, <8 x i32> poison, <8 x i32> zeroinitializer
  %45 = add <8 x i32> %.splat42, %43
  %46 = mul i32 %34, 1
  %.splatinsert43 = insertelement <8 x i32> poison, i32 %46, i64 0
  %.splat44 = shufflevector <8 x i32> %.splatinsert43, <8 x i32> poison, <8 x i32> zeroinitializer
  %47 = add <8 x i32> %.splat44, zeroinitializer
  %48 = mul i32 %38, 1
  %.splatinsert45 = insertelement <8 x i32> poison, i32 %48, i64 0
  %.splat46 = shufflevector <8 x i32> %.splatinsert45, <8 x i32> poison, <8 x i32> zeroinitializer
  %49 = add <8 x i32> %.splat46, zeroinitializer
  %50 = insertvalue [3 x <8 x i32>] poison, <8 x i32> %45, 0
  %51 = insertvalue [3 x <8 x i32>] %50, <8 x i32> %47, 1
  %52 = insertvalue [3 x <8 x i32>] %51, <8 x i32> %49, 2
  %.splatinsert47 = insertelement <8 x i32> poison, i32 %active_lane_count, i64 0
  %.splat48 = shufflevector <8 x i32> %.splatinsert47, <8 x i32> poison, <8 x i32> zeroinitializer
  %53 = icmp ult <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, %.splat48
  %54 = call i1 @llvm.vector.reduce.or.v8i1(<8 x i1> %53)
  br i1 %54, label %direct.activate, label %direct.inactive

direct.schedule.0:                                ; preds = %direct.activate
  %55 = extractvalue [3 x <8 x i32>] %52, 0
  %56 = zext <8 x i32> %55 to <8 x i64>
  %57 = select <8 x i1> %53, <8 x i64> %56, <8 x i64> zeroinitializer
  %58 = select <8 x i1> %53, <8 x i64> splat (i64 1), <8 x i64> splat (i64 1)
  %59 = sdiv <8 x i64> %57, %58
  %60 = select <8 x i1> %53, <8 x i64> %59, <8 x i64> zeroinitializer
  %61 = select <8 x i1> %53, <8 x i64> splat (i64 17), <8 x i64> splat (i64 1)
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
  %.state49 = load i64, ptr %.slot, align 4
  %75 = srem i64 %.state49, 1537
  %.state50 = load i64, ptr %.slot, align 4
  %76 = sdiv i64 %.state50, 1537
  %77 = srem i64 %76, 1
  %.spill.load = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert51 = insertelement <8 x i64> poison, i64 %77, i64 0
  %.splat52 = shufflevector <8 x i64> %.splatinsert51, <8 x i64> poison, <8 x i32> zeroinitializer
  %78 = add <8 x i64> %.spill.load, %.splat52
  %79 = add <8 x i64> zeroinitializer, %78
  %80 = add i64 0, %75
  %81 = mul <8 x i64> %79, splat (i64 1537)
  %.splatinsert53 = insertelement <8 x i64> poison, i64 %80, i64 0
  %.splat54 = shufflevector <8 x i64> %.splatinsert53, <8 x i64> poison, <8 x i32> zeroinitializer
  %82 = add <8 x i64> %81, %.splat54
  %83 = extractvalue { ptr, i64 } %11, 0
  %84 = mul <8 x i64> %82, splat (i64 4)
  %85 = getelementptr i8, ptr %83, <8 x i64> %84
  %86 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %85, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %87 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 0
  %88 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 1
  %.state55 = load i64, ptr %.slot, align 4
  %.splatinsert56 = insertelement <8 x i64> poison, i64 %.state55, i64 0
  %.splat57 = shufflevector <8 x i64> %.splatinsert56, <8 x i64> poison, <8 x i32> zeroinitializer
  %89 = mul <8 x i64> %.splat57, splat (i64 4)
  %90 = add <8 x i64> %88, %89
  %91 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %87, 0
  %92 = insertvalue { <8 x ptr>, <8 x i64> } %91, <8 x i64> %90, 1
  %93 = extractvalue { <8 x ptr>, <8 x i64> } %92, 0
  %94 = extractvalue { <8 x ptr>, <8 x i64> } %92, 1
  %95 = getelementptr i8, <8 x ptr> %93, <8 x i64> %94
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %86, <8 x ptr> %95, i32 1, <8 x i1> %53)
  %.state58 = load i64, ptr %.slot, align 4
  %96 = add i64 %.state58, 1
  store i64 %96, ptr %.slot, align 4
  br label %direct.schedule.1

direct.schedule.3:                                ; preds = %direct.false
  store float 0.000000e+00, ptr %.spill5, align 4
  store i64 0, ptr %.spill6, align 4
  store i64 0, ptr %.spill7, align 4
  store i64 0, ptr %.spill8, align 4
  br i1 true, label %direct.true59, label %direct.false60

direct.schedule.4:                                ; preds = %direct.true59
  %.spill.load61 = load i64, ptr %.spill8, align 4
  %97 = srem i64 %.spill.load61, 1537
  %98 = add i64 0, %97
  %tile_snapshot.spill.load62 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %99 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load62, 0
  %100 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load62, 1
  %.splatinsert63 = insertelement <8 x i64> poison, i64 %98, i64 0
  %.splat64 = shufflevector <8 x i64> %.splatinsert63, <8 x i64> poison, <8 x i32> zeroinitializer
  %101 = mul <8 x i64> %.splat64, splat (i64 4)
  %102 = add <8 x i64> %100, %101
  %103 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %99, 0
  %104 = insertvalue { <8 x ptr>, <8 x i64> } %103, <8 x i64> %102, 1
  %105 = extractvalue { <8 x ptr>, <8 x i64> } %104, 0
  %106 = extractvalue { <8 x ptr>, <8 x i64> } %104, 1
  %107 = getelementptr i8, <8 x ptr> %105, <8 x i64> %106
  %108 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %107, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %109 = fmul <8 x float> %108, %108
  %110 = load <8 x float>, ptr %.slot9, align 32
  %111 = select <8 x i1> %53, <8 x float> %109, <8 x float> %110
  store <8 x float> %111, ptr %.slot9, align 32
  br label %direct.schedule.5

direct.schedule.5:                                ; preds = %direct.schedule.4, %direct.false60
  %.spill.load65 = load i64, ptr %.spill7, align 4
  %112 = add i64 %.spill.load65, 1
  store i64 %112, ptr %.spill10, align 4
  %113 = icmp sge i64 %112, 0
  %114 = icmp slt i64 %112, 1537
  %115 = and i1 %113, %114
  br i1 %115, label %direct.true66, label %direct.false67

direct.schedule.6:                                ; preds = %direct.true66
  %.spill.load68 = load i64, ptr %.spill10, align 4
  %116 = srem i64 %.spill.load68, 1537
  %117 = add i64 0, %116
  %tile_snapshot.spill.load69 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %118 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load69, 0
  %119 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load69, 1
  %.splatinsert70 = insertelement <8 x i64> poison, i64 %117, i64 0
  %.splat71 = shufflevector <8 x i64> %.splatinsert70, <8 x i64> poison, <8 x i32> zeroinitializer
  %120 = mul <8 x i64> %.splat71, splat (i64 4)
  %121 = add <8 x i64> %119, %120
  %122 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %118, 0
  %123 = insertvalue { <8 x ptr>, <8 x i64> } %122, <8 x i64> %121, 1
  %124 = extractvalue { <8 x ptr>, <8 x i64> } %123, 0
  %125 = extractvalue { <8 x ptr>, <8 x i64> } %123, 1
  %126 = getelementptr i8, <8 x ptr> %124, <8 x i64> %125
  %127 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %126, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %128 = fmul <8 x float> %127, %127
  %129 = load <8 x float>, ptr %.slot11, align 32
  %130 = select <8 x i1> %53, <8 x float> %128, <8 x float> %129
  store <8 x float> %130, ptr %.slot11, align 32
  br label %direct.schedule.7

direct.schedule.7:                                ; preds = %direct.schedule.6, %direct.false67
  %.spill.load72 = load i64, ptr %.spill7, align 4
  %131 = add i64 %.spill.load72, 2
  store i64 %131, ptr %.spill12, align 4
  %132 = icmp sge i64 %131, 0
  %133 = icmp slt i64 %131, 1537
  %134 = and i1 %132, %133
  br i1 %134, label %direct.true73, label %direct.false74

direct.schedule.8:                                ; preds = %direct.true73
  %.spill.load75 = load i64, ptr %.spill12, align 4
  %135 = srem i64 %.spill.load75, 1537
  %136 = add i64 0, %135
  %tile_snapshot.spill.load76 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %137 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load76, 0
  %138 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load76, 1
  %.splatinsert77 = insertelement <8 x i64> poison, i64 %136, i64 0
  %.splat78 = shufflevector <8 x i64> %.splatinsert77, <8 x i64> poison, <8 x i32> zeroinitializer
  %139 = mul <8 x i64> %.splat78, splat (i64 4)
  %140 = add <8 x i64> %138, %139
  %141 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %137, 0
  %142 = insertvalue { <8 x ptr>, <8 x i64> } %141, <8 x i64> %140, 1
  %143 = extractvalue { <8 x ptr>, <8 x i64> } %142, 0
  %144 = extractvalue { <8 x ptr>, <8 x i64> } %142, 1
  %145 = getelementptr i8, <8 x ptr> %143, <8 x i64> %144
  %146 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %145, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %147 = fmul <8 x float> %146, %146
  %148 = load <8 x float>, ptr %.slot13, align 32
  %149 = select <8 x i1> %53, <8 x float> %147, <8 x float> %148
  store <8 x float> %149, ptr %.slot13, align 32
  br label %direct.schedule.9

direct.schedule.9:                                ; preds = %direct.schedule.8, %direct.false74
  %.spill.load79 = load i64, ptr %.spill7, align 4
  %150 = add i64 %.spill.load79, 3
  store i64 %150, ptr %.spill14, align 4
  %151 = icmp sge i64 %150, 0
  %152 = icmp slt i64 %150, 1537
  %153 = and i1 %151, %152
  br i1 %153, label %direct.true80, label %direct.false81

direct.schedule.10:                               ; preds = %direct.true80
  %.spill.load82 = load i64, ptr %.spill14, align 4
  %154 = srem i64 %.spill.load82, 1537
  %155 = add i64 0, %154
  %tile_snapshot.spill.load83 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %156 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load83, 0
  %157 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load83, 1
  %.splatinsert84 = insertelement <8 x i64> poison, i64 %155, i64 0
  %.splat85 = shufflevector <8 x i64> %.splatinsert84, <8 x i64> poison, <8 x i32> zeroinitializer
  %158 = mul <8 x i64> %.splat85, splat (i64 4)
  %159 = add <8 x i64> %157, %158
  %160 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %156, 0
  %161 = insertvalue { <8 x ptr>, <8 x i64> } %160, <8 x i64> %159, 1
  %162 = extractvalue { <8 x ptr>, <8 x i64> } %161, 0
  %163 = extractvalue { <8 x ptr>, <8 x i64> } %161, 1
  %164 = getelementptr i8, <8 x ptr> %162, <8 x i64> %163
  %165 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %164, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %166 = fmul <8 x float> %165, %165
  %167 = load <8 x float>, ptr %.slot15, align 32
  %168 = select <8 x i1> %53, <8 x float> %166, <8 x float> %167
  store <8 x float> %168, ptr %.slot15, align 32
  br label %direct.schedule.11

direct.schedule.11:                               ; preds = %direct.schedule.10, %direct.false81
  %.state86 = load <8 x float>, ptr %.slot9, align 32
  %.state87 = load <8 x float>, ptr %.slot11, align 32
  %.state88 = load <8 x float>, ptr %.slot13, align 32
  %.state89 = load <8 x float>, ptr %.slot15, align 32
  store i64 4, ptr %.slot16, align 4
  %169 = load <8 x float>, ptr %.slot17, align 32
  %170 = select <8 x i1> %53, <8 x float> %.state86, <8 x float> %169
  store <8 x float> %170, ptr %.slot17, align 32
  %171 = load <8 x float>, ptr %.slot18, align 32
  %172 = select <8 x i1> %53, <8 x float> %.state87, <8 x float> %171
  store <8 x float> %172, ptr %.slot18, align 32
  %173 = load <8 x float>, ptr %.slot19, align 32
  %174 = select <8 x i1> %53, <8 x float> %.state88, <8 x float> %173
  store <8 x float> %174, ptr %.slot19, align 32
  %175 = load <8 x float>, ptr %.slot20, align 32
  %176 = select <8 x i1> %53, <8 x float> %.state89, <8 x float> %175
  store <8 x float> %176, ptr %.slot20, align 32
  br label %direct.schedule.12

direct.schedule.12:                               ; preds = %direct.schedule.21, %direct.schedule.11
  %.state90 = load i64, ptr %.slot16, align 4
  %177 = icmp slt i64 %.state90, 1536
  br i1 %177, label %direct.true91, label %direct.false92

direct.schedule.13:                               ; preds = %direct.true91
  %.state93 = load i64, ptr %.slot16, align 4
  %178 = add i64 %.state93, 0
  %179 = sdiv i64 %178, 1
  %180 = srem i64 %179, 1537
  %.spill.load94 = load i64, ptr %.spill7, align 4
  %181 = add i64 %.spill.load94, %180
  store i64 %181, ptr %.spill21, align 4
  %182 = icmp sge i64 %181, 0
  %183 = icmp slt i64 %181, 1537
  %184 = and i1 %182, %183
  br i1 %184, label %direct.true95, label %direct.false96

direct.schedule.14:                               ; preds = %direct.true95
  %.spill.load97 = load i64, ptr %.spill21, align 4
  %185 = srem i64 %.spill.load97, 1537
  %186 = add i64 0, %185
  %tile_snapshot.spill.load98 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %187 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load98, 0
  %188 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load98, 1
  %.splatinsert99 = insertelement <8 x i64> poison, i64 %186, i64 0
  %.splat100 = shufflevector <8 x i64> %.splatinsert99, <8 x i64> poison, <8 x i32> zeroinitializer
  %189 = mul <8 x i64> %.splat100, splat (i64 4)
  %190 = add <8 x i64> %188, %189
  %191 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %187, 0
  %192 = insertvalue { <8 x ptr>, <8 x i64> } %191, <8 x i64> %190, 1
  %193 = extractvalue { <8 x ptr>, <8 x i64> } %192, 0
  %194 = extractvalue { <8 x ptr>, <8 x i64> } %192, 1
  %195 = getelementptr i8, <8 x ptr> %193, <8 x i64> %194
  %196 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %195, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %197 = fmul <8 x float> %196, %196
  %198 = load <8 x float>, ptr %.slot22, align 32
  %199 = select <8 x i1> %53, <8 x float> %197, <8 x float> %198
  store <8 x float> %199, ptr %.slot22, align 32
  br label %direct.schedule.15

direct.schedule.15:                               ; preds = %direct.schedule.14, %direct.false96
  %.state101 = load <8 x float>, ptr %.slot17, align 32
  %.state102 = load <8 x float>, ptr %.slot22, align 32
  %200 = fadd <8 x float> %.state101, %.state102
  %201 = load <8 x float>, ptr %.spill23, align 32
  %202 = select <8 x i1> %53, <8 x float> %200, <8 x float> %201
  store <8 x float> %202, ptr %.spill23, align 32
  %.state103 = load i64, ptr %.slot16, align 4
  %203 = add i64 %.state103, 1
  %204 = sdiv i64 %203, 1
  %205 = srem i64 %204, 1537
  %.spill.load104 = load i64, ptr %.spill7, align 4
  %206 = add i64 %.spill.load104, %205
  store i64 %206, ptr %.spill24, align 4
  %207 = icmp sge i64 %206, 0
  %208 = icmp slt i64 %206, 1537
  %209 = and i1 %207, %208
  br i1 %209, label %direct.true105, label %direct.false106

direct.schedule.16:                               ; preds = %direct.true105
  %.spill.load107 = load i64, ptr %.spill24, align 4
  %210 = srem i64 %.spill.load107, 1537
  %211 = add i64 0, %210
  %tile_snapshot.spill.load108 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %212 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load108, 0
  %213 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load108, 1
  %.splatinsert109 = insertelement <8 x i64> poison, i64 %211, i64 0
  %.splat110 = shufflevector <8 x i64> %.splatinsert109, <8 x i64> poison, <8 x i32> zeroinitializer
  %214 = mul <8 x i64> %.splat110, splat (i64 4)
  %215 = add <8 x i64> %213, %214
  %216 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %212, 0
  %217 = insertvalue { <8 x ptr>, <8 x i64> } %216, <8 x i64> %215, 1
  %218 = extractvalue { <8 x ptr>, <8 x i64> } %217, 0
  %219 = extractvalue { <8 x ptr>, <8 x i64> } %217, 1
  %220 = getelementptr i8, <8 x ptr> %218, <8 x i64> %219
  %221 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %220, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %222 = fmul <8 x float> %221, %221
  %223 = load <8 x float>, ptr %.slot25, align 32
  %224 = select <8 x i1> %53, <8 x float> %222, <8 x float> %223
  store <8 x float> %224, ptr %.slot25, align 32
  br label %direct.schedule.17

direct.schedule.17:                               ; preds = %direct.schedule.16, %direct.false106
  %.state111 = load <8 x float>, ptr %.slot18, align 32
  %.state112 = load <8 x float>, ptr %.slot25, align 32
  %225 = fadd <8 x float> %.state111, %.state112
  %226 = load <8 x float>, ptr %.spill26, align 32
  %227 = select <8 x i1> %53, <8 x float> %225, <8 x float> %226
  store <8 x float> %227, ptr %.spill26, align 32
  %.state113 = load i64, ptr %.slot16, align 4
  %228 = add i64 %.state113, 2
  %229 = sdiv i64 %228, 1
  %230 = srem i64 %229, 1537
  %.spill.load114 = load i64, ptr %.spill7, align 4
  %231 = add i64 %.spill.load114, %230
  store i64 %231, ptr %.spill27, align 4
  %232 = icmp sge i64 %231, 0
  %233 = icmp slt i64 %231, 1537
  %234 = and i1 %232, %233
  br i1 %234, label %direct.true115, label %direct.false116

direct.schedule.18:                               ; preds = %direct.true115
  %.spill.load117 = load i64, ptr %.spill27, align 4
  %235 = srem i64 %.spill.load117, 1537
  %236 = add i64 0, %235
  %tile_snapshot.spill.load118 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %237 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load118, 0
  %238 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load118, 1
  %.splatinsert119 = insertelement <8 x i64> poison, i64 %236, i64 0
  %.splat120 = shufflevector <8 x i64> %.splatinsert119, <8 x i64> poison, <8 x i32> zeroinitializer
  %239 = mul <8 x i64> %.splat120, splat (i64 4)
  %240 = add <8 x i64> %238, %239
  %241 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %237, 0
  %242 = insertvalue { <8 x ptr>, <8 x i64> } %241, <8 x i64> %240, 1
  %243 = extractvalue { <8 x ptr>, <8 x i64> } %242, 0
  %244 = extractvalue { <8 x ptr>, <8 x i64> } %242, 1
  %245 = getelementptr i8, <8 x ptr> %243, <8 x i64> %244
  %246 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %245, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %247 = fmul <8 x float> %246, %246
  %248 = load <8 x float>, ptr %.slot28, align 32
  %249 = select <8 x i1> %53, <8 x float> %247, <8 x float> %248
  store <8 x float> %249, ptr %.slot28, align 32
  br label %direct.schedule.19

direct.schedule.19:                               ; preds = %direct.schedule.18, %direct.false116
  %.state121 = load <8 x float>, ptr %.slot19, align 32
  %.state122 = load <8 x float>, ptr %.slot28, align 32
  %250 = fadd <8 x float> %.state121, %.state122
  %251 = load <8 x float>, ptr %.spill29, align 32
  %252 = select <8 x i1> %53, <8 x float> %250, <8 x float> %251
  store <8 x float> %252, ptr %.spill29, align 32
  %.state123 = load i64, ptr %.slot16, align 4
  %253 = add i64 %.state123, 3
  %254 = sdiv i64 %253, 1
  %255 = srem i64 %254, 1537
  %.spill.load124 = load i64, ptr %.spill7, align 4
  %256 = add i64 %.spill.load124, %255
  store i64 %256, ptr %.spill30, align 4
  %257 = icmp sge i64 %256, 0
  %258 = icmp slt i64 %256, 1537
  %259 = and i1 %257, %258
  br i1 %259, label %direct.true125, label %direct.false126

direct.schedule.20:                               ; preds = %direct.true125
  %.spill.load127 = load i64, ptr %.spill30, align 4
  %260 = srem i64 %.spill.load127, 1537
  %261 = add i64 0, %260
  %tile_snapshot.spill.load128 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %262 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load128, 0
  %263 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load128, 1
  %.splatinsert129 = insertelement <8 x i64> poison, i64 %261, i64 0
  %.splat130 = shufflevector <8 x i64> %.splatinsert129, <8 x i64> poison, <8 x i32> zeroinitializer
  %264 = mul <8 x i64> %.splat130, splat (i64 4)
  %265 = add <8 x i64> %263, %264
  %266 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %262, 0
  %267 = insertvalue { <8 x ptr>, <8 x i64> } %266, <8 x i64> %265, 1
  %268 = extractvalue { <8 x ptr>, <8 x i64> } %267, 0
  %269 = extractvalue { <8 x ptr>, <8 x i64> } %267, 1
  %270 = getelementptr i8, <8 x ptr> %268, <8 x i64> %269
  %271 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %270, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %272 = fmul <8 x float> %271, %271
  %273 = load <8 x float>, ptr %.slot31, align 32
  %274 = select <8 x i1> %53, <8 x float> %272, <8 x float> %273
  store <8 x float> %274, ptr %.slot31, align 32
  br label %direct.schedule.21

direct.schedule.21:                               ; preds = %direct.schedule.20, %direct.false126
  %.state131 = load <8 x float>, ptr %.slot20, align 32
  %.state132 = load <8 x float>, ptr %.slot31, align 32
  %275 = fadd <8 x float> %.state131, %.state132
  %.state133 = load i64, ptr %.slot16, align 4
  %276 = add i64 %.state133, 4
  %.spill.load134 = load <8 x float>, ptr %.spill23, align 32
  %.spill.load135 = load <8 x float>, ptr %.spill26, align 32
  %.spill.load136 = load <8 x float>, ptr %.spill29, align 32
  store i64 %276, ptr %.slot16, align 4
  %277 = load <8 x float>, ptr %.slot17, align 32
  %278 = select <8 x i1> %53, <8 x float> %.spill.load134, <8 x float> %277
  store <8 x float> %278, ptr %.slot17, align 32
  %279 = load <8 x float>, ptr %.slot18, align 32
  %280 = select <8 x i1> %53, <8 x float> %.spill.load135, <8 x float> %279
  store <8 x float> %280, ptr %.slot18, align 32
  %281 = load <8 x float>, ptr %.slot19, align 32
  %282 = select <8 x i1> %53, <8 x float> %.spill.load136, <8 x float> %281
  store <8 x float> %282, ptr %.slot19, align 32
  %283 = load <8 x float>, ptr %.slot20, align 32
  %284 = select <8 x i1> %53, <8 x float> %275, <8 x float> %283
  store <8 x float> %284, ptr %.slot20, align 32
  br label %direct.schedule.12

direct.schedule.22:                               ; preds = %direct.false92
  %.spill.load137 = load i64, ptr %.spill7, align 4
  %285 = add i64 %.spill.load137, 1536
  store i64 %285, ptr %.spill32, align 4
  %286 = icmp sge i64 %285, 0
  %287 = icmp slt i64 %285, 1537
  %288 = and i1 %286, %287
  br i1 %288, label %direct.true138, label %direct.false139

direct.schedule.23:                               ; preds = %direct.true138
  %.spill.load140 = load i64, ptr %.spill32, align 4
  %289 = srem i64 %.spill.load140, 1537
  %290 = add i64 0, %289
  %tile_snapshot.spill.load141 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %291 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load141, 0
  %292 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load141, 1
  %.splatinsert142 = insertelement <8 x i64> poison, i64 %290, i64 0
  %.splat143 = shufflevector <8 x i64> %.splatinsert142, <8 x i64> poison, <8 x i32> zeroinitializer
  %293 = mul <8 x i64> %.splat143, splat (i64 4)
  %294 = add <8 x i64> %292, %293
  %295 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %291, 0
  %296 = insertvalue { <8 x ptr>, <8 x i64> } %295, <8 x i64> %294, 1
  %297 = extractvalue { <8 x ptr>, <8 x i64> } %296, 0
  %298 = extractvalue { <8 x ptr>, <8 x i64> } %296, 1
  %299 = getelementptr i8, <8 x ptr> %297, <8 x i64> %298
  %300 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %299, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %301 = fmul <8 x float> %300, %300
  %302 = load <8 x float>, ptr %.slot33, align 32
  %303 = select <8 x i1> %53, <8 x float> %301, <8 x float> %302
  store <8 x float> %303, ptr %.slot33, align 32
  br label %direct.schedule.24

direct.schedule.24:                               ; preds = %direct.schedule.23, %direct.false139
  %.state144 = load <8 x float>, ptr %.slot17, align 32
  %.state145 = load <8 x float>, ptr %.slot33, align 32
  %304 = fadd <8 x float> %.state144, %.state145
  %.spill.load146 = load float, ptr %.spill5, align 4
  %.splatinsert147 = insertelement <8 x float> poison, float %.spill.load146, i64 0
  %.splat148 = shufflevector <8 x float> %.splatinsert147, <8 x float> poison, <8 x i32> zeroinitializer
  %305 = fadd <8 x float> %.splat148, %304
  %.state149 = load <8 x float>, ptr %.slot18, align 32
  %306 = fadd <8 x float> %305, %.state149
  %.state150 = load <8 x float>, ptr %.slot19, align 32
  %307 = fadd <8 x float> %306, %.state150
  %.state151 = load <8 x float>, ptr %.slot20, align 32
  %308 = fadd <8 x float> %307, %.state151
  %309 = fdiv <8 x float> %308, splat (float 1.537000e+03)
  %310 = load <8 x float>, ptr %.spill34, align 32
  %311 = select <8 x i1> %53, <8 x float> %309, <8 x float> %310
  store <8 x float> %311, ptr %.spill34, align 32
  %312 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill35, align 64
  %313 = extractvalue { <8 x ptr>, <8 x i64> } %5, 0
  %314 = extractvalue { <8 x ptr>, <8 x i64> } %312, 0
  %315 = select <8 x i1> %53, <8 x ptr> %313, <8 x ptr> %314
  %316 = extractvalue { <8 x ptr>, <8 x i64> } %5, 1
  %317 = extractvalue { <8 x ptr>, <8 x i64> } %312, 1
  %318 = select <8 x i1> %53, <8 x i64> %316, <8 x i64> %317
  %319 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %315, 0
  %320 = insertvalue { <8 x ptr>, <8 x i64> } %319, <8 x i64> %318, 1
  store { <8 x ptr>, <8 x i64> } %320, ptr %tile_snapshot.spill35, align 64
  store i64 0, ptr %.slot36, align 4
  br label %direct.schedule.25

direct.schedule.25:                               ; preds = %direct.schedule.26, %direct.schedule.24
  %.state152 = load i64, ptr %.slot36, align 4
  %321 = icmp slt i64 %.state152, 1537
  br i1 %321, label %direct.true153, label %direct.false154

direct.schedule.26:                               ; preds = %direct.true153
  %.state155 = load i64, ptr %.slot36, align 4
  %322 = srem i64 %.state155, 1537
  %.state156 = load i64, ptr %.slot36, align 4
  %323 = sdiv i64 %.state156, 1537
  %324 = srem i64 %323, 1
  %325 = add i64 0, %324
  %.spill.load157 = load i64, ptr %.spill6, align 4
  %326 = add i64 %.spill.load157, %325
  %327 = add i64 0, %322
  %328 = mul i64 %326, 1537
  %329 = add i64 %328, %327
  %330 = extractvalue { ptr, i64 } %17, 0
  %331 = mul i64 %329, 4
  %332 = getelementptr i8, ptr %330, i64 %331
  %333 = load float, ptr %332, align 4
  %.splatinsert158 = insertelement <8 x float> poison, float %333, i64 0
  %.splat159 = shufflevector <8 x float> %.splatinsert158, <8 x float> poison, <8 x i32> zeroinitializer
  %tile_snapshot.spill.load160 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill35, align 64
  %334 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load160, 0
  %335 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load160, 1
  %.state161 = load i64, ptr %.slot36, align 4
  %.splatinsert162 = insertelement <8 x i64> poison, i64 %.state161, i64 0
  %.splat163 = shufflevector <8 x i64> %.splatinsert162, <8 x i64> poison, <8 x i32> zeroinitializer
  %336 = mul <8 x i64> %.splat163, splat (i64 4)
  %337 = add <8 x i64> %335, %336
  %338 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %334, 0
  %339 = insertvalue { <8 x ptr>, <8 x i64> } %338, <8 x i64> %337, 1
  %340 = extractvalue { <8 x ptr>, <8 x i64> } %339, 0
  %341 = extractvalue { <8 x ptr>, <8 x i64> } %339, 1
  %342 = getelementptr i8, <8 x ptr> %340, <8 x i64> %341
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %.splat159, <8 x ptr> %342, i32 1, <8 x i1> %53)
  %.state164 = load i64, ptr %.slot36, align 4
  %343 = add i64 %.state164, 1
  store i64 %343, ptr %.slot36, align 4
  br label %direct.schedule.25

direct.schedule.27:                               ; preds = %direct.false154
  %.spill.load165 = load <8 x float>, ptr %.spill34, align 32
  %344 = fadd <8 x float> %.spill.load165, splat (float 0x3EE4F8B580000000)
  %345 = call <8 x float> @llvm.sqrt.v8f32(<8 x float> %344)
  %346 = load <8 x float>, ptr %.spill37, align 32
  %347 = select <8 x i1> %53, <8 x float> %345, <8 x float> %346
  store <8 x float> %347, ptr %.spill37, align 32
  store i64 0, ptr %.slot38, align 4
  br label %direct.schedule.28

direct.schedule.28:                               ; preds = %direct.schedule.29, %direct.schedule.27
  %.state166 = load i64, ptr %.slot38, align 4
  %348 = icmp slt i64 %.state166, 1537
  br i1 %348, label %direct.true167, label %direct.false168

direct.schedule.29:                               ; preds = %direct.true167
  %.state169 = load i64, ptr %.slot38, align 4
  %349 = srem i64 %.state169, 1537
  %.state170 = load i64, ptr %.slot38, align 4
  %350 = sdiv i64 %.state170, 1537
  %351 = srem i64 %350, 1
  %.spill.load171 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert172 = insertelement <8 x i64> poison, i64 %351, i64 0
  %.splat173 = shufflevector <8 x i64> %.splatinsert172, <8 x i64> poison, <8 x i32> zeroinitializer
  %352 = add <8 x i64> %.spill.load171, %.splat173
  %353 = add <8 x i64> zeroinitializer, %352
  %354 = add i64 0, %349
  %355 = mul <8 x i64> %353, splat (i64 1537)
  %.splatinsert174 = insertelement <8 x i64> poison, i64 %354, i64 0
  %.splat175 = shufflevector <8 x i64> %.splatinsert174, <8 x i64> poison, <8 x i32> zeroinitializer
  %356 = add <8 x i64> %355, %.splat175
  %357 = add i64 0, %349
  %358 = srem i64 %357, 1537
  %359 = add i64 0, %358
  %tile_snapshot.spill.load176 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %360 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load176, 0
  %361 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load176, 1
  %.splatinsert177 = insertelement <8 x i64> poison, i64 %359, i64 0
  %.splat178 = shufflevector <8 x i64> %.splatinsert177, <8 x i64> poison, <8 x i32> zeroinitializer
  %362 = mul <8 x i64> %.splat178, splat (i64 4)
  %363 = add <8 x i64> %361, %362
  %364 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %360, 0
  %365 = insertvalue { <8 x ptr>, <8 x i64> } %364, <8 x i64> %363, 1
  %366 = extractvalue { <8 x ptr>, <8 x i64> } %365, 0
  %367 = extractvalue { <8 x ptr>, <8 x i64> } %365, 1
  %368 = getelementptr i8, <8 x ptr> %366, <8 x i64> %367
  %369 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %368, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %.spill.load179 = load <8 x float>, ptr %.spill37, align 32
  %370 = fdiv <8 x float> %369, %.spill.load179
  %tile_snapshot.spill.load180 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill35, align 64
  %371 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load180, 0
  %372 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load180, 1
  %.splatinsert181 = insertelement <8 x i64> poison, i64 %357, i64 0
  %.splat182 = shufflevector <8 x i64> %.splatinsert181, <8 x i64> poison, <8 x i32> zeroinitializer
  %373 = mul <8 x i64> %.splat182, splat (i64 4)
  %374 = add <8 x i64> %372, %373
  %375 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %371, 0
  %376 = insertvalue { <8 x ptr>, <8 x i64> } %375, <8 x i64> %374, 1
  %377 = extractvalue { <8 x ptr>, <8 x i64> } %376, 0
  %378 = extractvalue { <8 x ptr>, <8 x i64> } %376, 1
  %379 = getelementptr i8, <8 x ptr> %377, <8 x i64> %378
  %380 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %379, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %381 = fmul <8 x float> %370, %380
  %382 = extractvalue { ptr, i64 } %29, 0
  %383 = mul <8 x i64> %356, splat (i64 4)
  %384 = getelementptr i8, ptr %382, <8 x i64> %383
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %381, <8 x ptr> %384, i32 1, <8 x i1> %53)
  %.state183 = load i64, ptr %.slot38, align 4
  %385 = add i64 %.state183, 1
  store i64 %385, ptr %.slot38, align 4
  br label %direct.schedule.28

direct.schedule.30:                               ; preds = %direct.false168
  ret void

direct.activate:                                  ; preds = %prologue
  br label %direct.schedule.0

direct.inactive:                                  ; preds = %prologue
  ret void

direct.true:                                      ; preds = %direct.schedule.1
  br label %direct.schedule.2

direct.false:                                     ; preds = %direct.schedule.1
  br label %direct.schedule.3

direct.true59:                                    ; preds = %direct.schedule.3
  br label %direct.schedule.4

direct.false60:                                   ; preds = %direct.schedule.3
  %386 = load <8 x float>, ptr %.slot9, align 32
  %387 = select <8 x i1> %53, <8 x float> zeroinitializer, <8 x float> %386
  store <8 x float> %387, ptr %.slot9, align 32
  br label %direct.schedule.5

direct.true66:                                    ; preds = %direct.schedule.5
  br label %direct.schedule.6

direct.false67:                                   ; preds = %direct.schedule.5
  %388 = load <8 x float>, ptr %.slot11, align 32
  %389 = select <8 x i1> %53, <8 x float> zeroinitializer, <8 x float> %388
  store <8 x float> %389, ptr %.slot11, align 32
  br label %direct.schedule.7

direct.true73:                                    ; preds = %direct.schedule.7
  br label %direct.schedule.8

direct.false74:                                   ; preds = %direct.schedule.7
  %390 = load <8 x float>, ptr %.slot13, align 32
  %391 = select <8 x i1> %53, <8 x float> zeroinitializer, <8 x float> %390
  store <8 x float> %391, ptr %.slot13, align 32
  br label %direct.schedule.9

direct.true80:                                    ; preds = %direct.schedule.9
  br label %direct.schedule.10

direct.false81:                                   ; preds = %direct.schedule.9
  %392 = load <8 x float>, ptr %.slot15, align 32
  %393 = select <8 x i1> %53, <8 x float> zeroinitializer, <8 x float> %392
  store <8 x float> %393, ptr %.slot15, align 32
  br label %direct.schedule.11

direct.true91:                                    ; preds = %direct.schedule.12
  br label %direct.schedule.13

direct.false92:                                   ; preds = %direct.schedule.12
  br label %direct.schedule.22

direct.true95:                                    ; preds = %direct.schedule.13
  br label %direct.schedule.14

direct.false96:                                   ; preds = %direct.schedule.13
  %394 = load <8 x float>, ptr %.slot22, align 32
  %395 = select <8 x i1> %53, <8 x float> zeroinitializer, <8 x float> %394
  store <8 x float> %395, ptr %.slot22, align 32
  br label %direct.schedule.15

direct.true105:                                   ; preds = %direct.schedule.15
  br label %direct.schedule.16

direct.false106:                                  ; preds = %direct.schedule.15
  %396 = load <8 x float>, ptr %.slot25, align 32
  %397 = select <8 x i1> %53, <8 x float> zeroinitializer, <8 x float> %396
  store <8 x float> %397, ptr %.slot25, align 32
  br label %direct.schedule.17

direct.true115:                                   ; preds = %direct.schedule.17
  br label %direct.schedule.18

direct.false116:                                  ; preds = %direct.schedule.17
  %398 = load <8 x float>, ptr %.slot28, align 32
  %399 = select <8 x i1> %53, <8 x float> zeroinitializer, <8 x float> %398
  store <8 x float> %399, ptr %.slot28, align 32
  br label %direct.schedule.19

direct.true125:                                   ; preds = %direct.schedule.19
  br label %direct.schedule.20

direct.false126:                                  ; preds = %direct.schedule.19
  %400 = load <8 x float>, ptr %.slot31, align 32
  %401 = select <8 x i1> %53, <8 x float> zeroinitializer, <8 x float> %400
  store <8 x float> %401, ptr %.slot31, align 32
  br label %direct.schedule.21

direct.true138:                                   ; preds = %direct.schedule.22
  br label %direct.schedule.23

direct.false139:                                  ; preds = %direct.schedule.22
  %402 = load <8 x float>, ptr %.slot33, align 32
  %403 = select <8 x i1> %53, <8 x float> zeroinitializer, <8 x float> %402
  store <8 x float> %403, ptr %.slot33, align 32
  br label %direct.schedule.24

direct.true153:                                   ; preds = %direct.schedule.25
  br label %direct.schedule.26

direct.false154:                                  ; preds = %direct.schedule.25
  br label %direct.schedule.27

direct.true167:                                   ; preds = %direct.schedule.28
  br label %direct.schedule.29

direct.false168:                                  ; preds = %direct.schedule.28
  br label %direct.schedule.30
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
