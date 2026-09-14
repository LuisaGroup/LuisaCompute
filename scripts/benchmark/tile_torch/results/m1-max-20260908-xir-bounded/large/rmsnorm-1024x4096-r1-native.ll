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
  %.spill32 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill32, align 32
  %tile_snapshot.spill33 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill33, align 64
  %.slot34 = alloca i64, align 8
  store i64 0, ptr %.slot34, align 4
  %.spill35 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill35, align 32
  %.slot36 = alloca i64, align 8
  store i64 0, ptr %.slot36, align 4
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
  %.splatinsert37 = insertelement <8 x i32> poison, i32 %42, i64 0
  %.splat38 = shufflevector <8 x i32> %.splatinsert37, <8 x i32> poison, <8 x i32> zeroinitializer
  %43 = add <8 x i32> %.splat38, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %44 = mul i32 %30, 128
  %.splatinsert39 = insertelement <8 x i32> poison, i32 %44, i64 0
  %.splat40 = shufflevector <8 x i32> %.splatinsert39, <8 x i32> poison, <8 x i32> zeroinitializer
  %45 = add <8 x i32> %.splat40, %43
  %46 = mul i32 %34, 1
  %.splatinsert41 = insertelement <8 x i32> poison, i32 %46, i64 0
  %.splat42 = shufflevector <8 x i32> %.splatinsert41, <8 x i32> poison, <8 x i32> zeroinitializer
  %47 = add <8 x i32> %.splat42, zeroinitializer
  %48 = mul i32 %38, 1
  %.splatinsert43 = insertelement <8 x i32> poison, i32 %48, i64 0
  %.splat44 = shufflevector <8 x i32> %.splatinsert43, <8 x i32> poison, <8 x i32> zeroinitializer
  %49 = add <8 x i32> %.splat44, zeroinitializer
  %50 = insertvalue [3 x <8 x i32>] poison, <8 x i32> %45, 0
  %51 = insertvalue [3 x <8 x i32>] %50, <8 x i32> %47, 1
  %52 = insertvalue [3 x <8 x i32>] %51, <8 x i32> %49, 2
  %.splatinsert45 = insertelement <8 x i32> poison, i32 %active_lane_count, i64 0
  %.splat46 = shufflevector <8 x i32> %.splatinsert45, <8 x i32> poison, <8 x i32> zeroinitializer
  %53 = icmp ult <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, %.splat46
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
  %.state47 = load i64, ptr %.slot, align 4
  %75 = srem i64 %.state47, 4096
  %.state48 = load i64, ptr %.slot, align 4
  %76 = sdiv i64 %.state48, 4096
  %77 = srem i64 %76, 1
  %.spill.load = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert49 = insertelement <8 x i64> poison, i64 %77, i64 0
  %.splat50 = shufflevector <8 x i64> %.splatinsert49, <8 x i64> poison, <8 x i32> zeroinitializer
  %78 = add <8 x i64> %.spill.load, %.splat50
  %79 = add <8 x i64> zeroinitializer, %78
  %80 = add i64 0, %75
  %81 = mul <8 x i64> %79, splat (i64 4096)
  %.splatinsert51 = insertelement <8 x i64> poison, i64 %80, i64 0
  %.splat52 = shufflevector <8 x i64> %.splatinsert51, <8 x i64> poison, <8 x i32> zeroinitializer
  %82 = add <8 x i64> %81, %.splat52
  %83 = extractvalue { ptr, i64 } %11, 0
  %84 = mul <8 x i64> %82, splat (i64 4)
  %85 = getelementptr i8, ptr %83, <8 x i64> %84
  %86 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %85, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %87 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 0
  %88 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 1
  %.state53 = load i64, ptr %.slot, align 4
  %.splatinsert54 = insertelement <8 x i64> poison, i64 %.state53, i64 0
  %.splat55 = shufflevector <8 x i64> %.splatinsert54, <8 x i64> poison, <8 x i32> zeroinitializer
  %89 = mul <8 x i64> %.splat55, splat (i64 4)
  %90 = add <8 x i64> %88, %89
  %91 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %87, 0
  %92 = insertvalue { <8 x ptr>, <8 x i64> } %91, <8 x i64> %90, 1
  %93 = extractvalue { <8 x ptr>, <8 x i64> } %92, 0
  %94 = extractvalue { <8 x ptr>, <8 x i64> } %92, 1
  %95 = getelementptr i8, <8 x ptr> %93, <8 x i64> %94
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %86, <8 x ptr> %95, i32 1, <8 x i1> %53)
  %.state56 = load i64, ptr %.slot, align 4
  %96 = add i64 %.state56, 1
  store i64 %96, ptr %.slot, align 4
  br label %direct.schedule.1

direct.schedule.3:                                ; preds = %direct.false
  store float 0.000000e+00, ptr %.spill5, align 4
  store i64 0, ptr %.spill6, align 4
  store i64 0, ptr %.spill7, align 4
  store i64 0, ptr %.spill8, align 4
  br i1 true, label %direct.true57, label %direct.false58

direct.schedule.4:                                ; preds = %direct.true57
  %.spill.load59 = load i64, ptr %.spill8, align 4
  %97 = srem i64 %.spill.load59, 4096
  %98 = add i64 0, %97
  %tile_snapshot.spill.load60 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %99 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load60, 0
  %100 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load60, 1
  %.splatinsert61 = insertelement <8 x i64> poison, i64 %98, i64 0
  %.splat62 = shufflevector <8 x i64> %.splatinsert61, <8 x i64> poison, <8 x i32> zeroinitializer
  %101 = mul <8 x i64> %.splat62, splat (i64 4)
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

direct.schedule.5:                                ; preds = %direct.schedule.4, %direct.false58
  %.spill.load63 = load i64, ptr %.spill7, align 4
  %112 = add i64 %.spill.load63, 1
  store i64 %112, ptr %.spill10, align 4
  %113 = icmp sge i64 %112, 0
  %114 = icmp slt i64 %112, 4096
  %115 = and i1 %113, %114
  br i1 %115, label %direct.true64, label %direct.false65

direct.schedule.6:                                ; preds = %direct.true64
  %.spill.load66 = load i64, ptr %.spill10, align 4
  %116 = srem i64 %.spill.load66, 4096
  %117 = add i64 0, %116
  %tile_snapshot.spill.load67 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %118 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load67, 0
  %119 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load67, 1
  %.splatinsert68 = insertelement <8 x i64> poison, i64 %117, i64 0
  %.splat69 = shufflevector <8 x i64> %.splatinsert68, <8 x i64> poison, <8 x i32> zeroinitializer
  %120 = mul <8 x i64> %.splat69, splat (i64 4)
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

direct.schedule.7:                                ; preds = %direct.schedule.6, %direct.false65
  %.spill.load70 = load i64, ptr %.spill7, align 4
  %131 = add i64 %.spill.load70, 2
  store i64 %131, ptr %.spill12, align 4
  %132 = icmp sge i64 %131, 0
  %133 = icmp slt i64 %131, 4096
  %134 = and i1 %132, %133
  br i1 %134, label %direct.true71, label %direct.false72

direct.schedule.8:                                ; preds = %direct.true71
  %.spill.load73 = load i64, ptr %.spill12, align 4
  %135 = srem i64 %.spill.load73, 4096
  %136 = add i64 0, %135
  %tile_snapshot.spill.load74 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %137 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load74, 0
  %138 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load74, 1
  %.splatinsert75 = insertelement <8 x i64> poison, i64 %136, i64 0
  %.splat76 = shufflevector <8 x i64> %.splatinsert75, <8 x i64> poison, <8 x i32> zeroinitializer
  %139 = mul <8 x i64> %.splat76, splat (i64 4)
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

direct.schedule.9:                                ; preds = %direct.schedule.8, %direct.false72
  %.spill.load77 = load i64, ptr %.spill7, align 4
  %150 = add i64 %.spill.load77, 3
  store i64 %150, ptr %.spill14, align 4
  %151 = icmp sge i64 %150, 0
  %152 = icmp slt i64 %150, 4096
  %153 = and i1 %151, %152
  br i1 %153, label %direct.true78, label %direct.false79

direct.schedule.10:                               ; preds = %direct.true78
  %.spill.load80 = load i64, ptr %.spill14, align 4
  %154 = srem i64 %.spill.load80, 4096
  %155 = add i64 0, %154
  %tile_snapshot.spill.load81 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %156 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load81, 0
  %157 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load81, 1
  %.splatinsert82 = insertelement <8 x i64> poison, i64 %155, i64 0
  %.splat83 = shufflevector <8 x i64> %.splatinsert82, <8 x i64> poison, <8 x i32> zeroinitializer
  %158 = mul <8 x i64> %.splat83, splat (i64 4)
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

direct.schedule.11:                               ; preds = %direct.schedule.10, %direct.false79
  %.state84 = load <8 x float>, ptr %.slot9, align 32
  %.state85 = load <8 x float>, ptr %.slot11, align 32
  %.state86 = load <8 x float>, ptr %.slot13, align 32
  %.state87 = load <8 x float>, ptr %.slot15, align 32
  store i64 4, ptr %.slot16, align 4
  %169 = load <8 x float>, ptr %.slot17, align 32
  %170 = select <8 x i1> %53, <8 x float> %.state84, <8 x float> %169
  store <8 x float> %170, ptr %.slot17, align 32
  %171 = load <8 x float>, ptr %.slot18, align 32
  %172 = select <8 x i1> %53, <8 x float> %.state85, <8 x float> %171
  store <8 x float> %172, ptr %.slot18, align 32
  %173 = load <8 x float>, ptr %.slot19, align 32
  %174 = select <8 x i1> %53, <8 x float> %.state86, <8 x float> %173
  store <8 x float> %174, ptr %.slot19, align 32
  %175 = load <8 x float>, ptr %.slot20, align 32
  %176 = select <8 x i1> %53, <8 x float> %.state87, <8 x float> %175
  store <8 x float> %176, ptr %.slot20, align 32
  br label %direct.schedule.12

direct.schedule.12:                               ; preds = %direct.schedule.21, %direct.schedule.11
  %.state88 = load i64, ptr %.slot16, align 4
  %177 = icmp slt i64 %.state88, 4096
  br i1 %177, label %direct.true89, label %direct.false90

direct.schedule.13:                               ; preds = %direct.true89
  %.state91 = load i64, ptr %.slot16, align 4
  %178 = add i64 %.state91, 0
  %179 = sdiv i64 %178, 1
  %180 = srem i64 %179, 4096
  %.spill.load92 = load i64, ptr %.spill7, align 4
  %181 = add i64 %.spill.load92, %180
  store i64 %181, ptr %.spill21, align 4
  %182 = icmp sge i64 %181, 0
  %183 = icmp slt i64 %181, 4096
  %184 = and i1 %182, %183
  br i1 %184, label %direct.true93, label %direct.false94

direct.schedule.14:                               ; preds = %direct.true93
  %.spill.load95 = load i64, ptr %.spill21, align 4
  %185 = srem i64 %.spill.load95, 4096
  %186 = add i64 0, %185
  %tile_snapshot.spill.load96 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %187 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load96, 0
  %188 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load96, 1
  %.splatinsert97 = insertelement <8 x i64> poison, i64 %186, i64 0
  %.splat98 = shufflevector <8 x i64> %.splatinsert97, <8 x i64> poison, <8 x i32> zeroinitializer
  %189 = mul <8 x i64> %.splat98, splat (i64 4)
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

direct.schedule.15:                               ; preds = %direct.schedule.14, %direct.false94
  %.state99 = load <8 x float>, ptr %.slot17, align 32
  %.state100 = load <8 x float>, ptr %.slot22, align 32
  %200 = fadd <8 x float> %.state99, %.state100
  %201 = load <8 x float>, ptr %.spill23, align 32
  %202 = select <8 x i1> %53, <8 x float> %200, <8 x float> %201
  store <8 x float> %202, ptr %.spill23, align 32
  %.state101 = load i64, ptr %.slot16, align 4
  %203 = add i64 %.state101, 1
  %204 = sdiv i64 %203, 1
  %205 = srem i64 %204, 4096
  %.spill.load102 = load i64, ptr %.spill7, align 4
  %206 = add i64 %.spill.load102, %205
  store i64 %206, ptr %.spill24, align 4
  %207 = icmp sge i64 %206, 0
  %208 = icmp slt i64 %206, 4096
  %209 = and i1 %207, %208
  br i1 %209, label %direct.true103, label %direct.false104

direct.schedule.16:                               ; preds = %direct.true103
  %.spill.load105 = load i64, ptr %.spill24, align 4
  %210 = srem i64 %.spill.load105, 4096
  %211 = add i64 0, %210
  %tile_snapshot.spill.load106 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %212 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load106, 0
  %213 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load106, 1
  %.splatinsert107 = insertelement <8 x i64> poison, i64 %211, i64 0
  %.splat108 = shufflevector <8 x i64> %.splatinsert107, <8 x i64> poison, <8 x i32> zeroinitializer
  %214 = mul <8 x i64> %.splat108, splat (i64 4)
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

direct.schedule.17:                               ; preds = %direct.schedule.16, %direct.false104
  %.state109 = load <8 x float>, ptr %.slot18, align 32
  %.state110 = load <8 x float>, ptr %.slot25, align 32
  %225 = fadd <8 x float> %.state109, %.state110
  %226 = load <8 x float>, ptr %.spill26, align 32
  %227 = select <8 x i1> %53, <8 x float> %225, <8 x float> %226
  store <8 x float> %227, ptr %.spill26, align 32
  %.state111 = load i64, ptr %.slot16, align 4
  %228 = add i64 %.state111, 2
  %229 = sdiv i64 %228, 1
  %230 = srem i64 %229, 4096
  %.spill.load112 = load i64, ptr %.spill7, align 4
  %231 = add i64 %.spill.load112, %230
  store i64 %231, ptr %.spill27, align 4
  %232 = icmp sge i64 %231, 0
  %233 = icmp slt i64 %231, 4096
  %234 = and i1 %232, %233
  br i1 %234, label %direct.true113, label %direct.false114

direct.schedule.18:                               ; preds = %direct.true113
  %.spill.load115 = load i64, ptr %.spill27, align 4
  %235 = srem i64 %.spill.load115, 4096
  %236 = add i64 0, %235
  %tile_snapshot.spill.load116 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %237 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load116, 0
  %238 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load116, 1
  %.splatinsert117 = insertelement <8 x i64> poison, i64 %236, i64 0
  %.splat118 = shufflevector <8 x i64> %.splatinsert117, <8 x i64> poison, <8 x i32> zeroinitializer
  %239 = mul <8 x i64> %.splat118, splat (i64 4)
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

direct.schedule.19:                               ; preds = %direct.schedule.18, %direct.false114
  %.state119 = load <8 x float>, ptr %.slot19, align 32
  %.state120 = load <8 x float>, ptr %.slot28, align 32
  %250 = fadd <8 x float> %.state119, %.state120
  %251 = load <8 x float>, ptr %.spill29, align 32
  %252 = select <8 x i1> %53, <8 x float> %250, <8 x float> %251
  store <8 x float> %252, ptr %.spill29, align 32
  %.state121 = load i64, ptr %.slot16, align 4
  %253 = add i64 %.state121, 3
  %254 = sdiv i64 %253, 1
  %255 = srem i64 %254, 4096
  %.spill.load122 = load i64, ptr %.spill7, align 4
  %256 = add i64 %.spill.load122, %255
  store i64 %256, ptr %.spill30, align 4
  %257 = icmp sge i64 %256, 0
  %258 = icmp slt i64 %256, 4096
  %259 = and i1 %257, %258
  br i1 %259, label %direct.true123, label %direct.false124

direct.schedule.20:                               ; preds = %direct.true123
  %.spill.load125 = load i64, ptr %.spill30, align 4
  %260 = srem i64 %.spill.load125, 4096
  %261 = add i64 0, %260
  %tile_snapshot.spill.load126 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %262 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load126, 0
  %263 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load126, 1
  %.splatinsert127 = insertelement <8 x i64> poison, i64 %261, i64 0
  %.splat128 = shufflevector <8 x i64> %.splatinsert127, <8 x i64> poison, <8 x i32> zeroinitializer
  %264 = mul <8 x i64> %.splat128, splat (i64 4)
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

direct.schedule.21:                               ; preds = %direct.schedule.20, %direct.false124
  %.state129 = load <8 x float>, ptr %.slot20, align 32
  %.state130 = load <8 x float>, ptr %.slot31, align 32
  %275 = fadd <8 x float> %.state129, %.state130
  %.state131 = load i64, ptr %.slot16, align 4
  %276 = add i64 %.state131, 4
  %.spill.load132 = load <8 x float>, ptr %.spill23, align 32
  %.spill.load133 = load <8 x float>, ptr %.spill26, align 32
  %.spill.load134 = load <8 x float>, ptr %.spill29, align 32
  store i64 %276, ptr %.slot16, align 4
  %277 = load <8 x float>, ptr %.slot17, align 32
  %278 = select <8 x i1> %53, <8 x float> %.spill.load132, <8 x float> %277
  store <8 x float> %278, ptr %.slot17, align 32
  %279 = load <8 x float>, ptr %.slot18, align 32
  %280 = select <8 x i1> %53, <8 x float> %.spill.load133, <8 x float> %279
  store <8 x float> %280, ptr %.slot18, align 32
  %281 = load <8 x float>, ptr %.slot19, align 32
  %282 = select <8 x i1> %53, <8 x float> %.spill.load134, <8 x float> %281
  store <8 x float> %282, ptr %.slot19, align 32
  %283 = load <8 x float>, ptr %.slot20, align 32
  %284 = select <8 x i1> %53, <8 x float> %275, <8 x float> %283
  store <8 x float> %284, ptr %.slot20, align 32
  br label %direct.schedule.12

direct.schedule.22:                               ; preds = %direct.false90
  %.spill.load135 = load float, ptr %.spill5, align 4
  %.splatinsert136 = insertelement <8 x float> poison, float %.spill.load135, i64 0
  %.splat137 = shufflevector <8 x float> %.splatinsert136, <8 x float> poison, <8 x i32> zeroinitializer
  %.state138 = load <8 x float>, ptr %.slot17, align 32
  %285 = fadd <8 x float> %.splat137, %.state138
  %.state139 = load <8 x float>, ptr %.slot18, align 32
  %286 = fadd <8 x float> %285, %.state139
  %.state140 = load <8 x float>, ptr %.slot19, align 32
  %287 = fadd <8 x float> %286, %.state140
  %.state141 = load <8 x float>, ptr %.slot20, align 32
  %288 = fadd <8 x float> %287, %.state141
  %289 = fdiv <8 x float> %288, splat (float 4.096000e+03)
  %290 = load <8 x float>, ptr %.spill32, align 32
  %291 = select <8 x i1> %53, <8 x float> %289, <8 x float> %290
  store <8 x float> %291, ptr %.spill32, align 32
  %292 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill33, align 64
  %293 = extractvalue { <8 x ptr>, <8 x i64> } %5, 0
  %294 = extractvalue { <8 x ptr>, <8 x i64> } %292, 0
  %295 = select <8 x i1> %53, <8 x ptr> %293, <8 x ptr> %294
  %296 = extractvalue { <8 x ptr>, <8 x i64> } %5, 1
  %297 = extractvalue { <8 x ptr>, <8 x i64> } %292, 1
  %298 = select <8 x i1> %53, <8 x i64> %296, <8 x i64> %297
  %299 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %295, 0
  %300 = insertvalue { <8 x ptr>, <8 x i64> } %299, <8 x i64> %298, 1
  store { <8 x ptr>, <8 x i64> } %300, ptr %tile_snapshot.spill33, align 64
  store i64 0, ptr %.slot34, align 4
  br label %direct.schedule.23

direct.schedule.23:                               ; preds = %direct.schedule.24, %direct.schedule.22
  %.state142 = load i64, ptr %.slot34, align 4
  %301 = icmp slt i64 %.state142, 4096
  br i1 %301, label %direct.true143, label %direct.false144

direct.schedule.24:                               ; preds = %direct.true143
  %.state145 = load i64, ptr %.slot34, align 4
  %302 = srem i64 %.state145, 4096
  %.state146 = load i64, ptr %.slot34, align 4
  %303 = sdiv i64 %.state146, 4096
  %304 = srem i64 %303, 1
  %305 = add i64 0, %304
  %.spill.load147 = load i64, ptr %.spill6, align 4
  %306 = add i64 %.spill.load147, %305
  %307 = add i64 0, %302
  %308 = mul i64 %306, 4096
  %309 = add i64 %308, %307
  %310 = extractvalue { ptr, i64 } %17, 0
  %311 = mul i64 %309, 4
  %312 = getelementptr i8, ptr %310, i64 %311
  %313 = load float, ptr %312, align 4
  %.splatinsert148 = insertelement <8 x float> poison, float %313, i64 0
  %.splat149 = shufflevector <8 x float> %.splatinsert148, <8 x float> poison, <8 x i32> zeroinitializer
  %tile_snapshot.spill.load150 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill33, align 64
  %314 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load150, 0
  %315 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load150, 1
  %.state151 = load i64, ptr %.slot34, align 4
  %.splatinsert152 = insertelement <8 x i64> poison, i64 %.state151, i64 0
  %.splat153 = shufflevector <8 x i64> %.splatinsert152, <8 x i64> poison, <8 x i32> zeroinitializer
  %316 = mul <8 x i64> %.splat153, splat (i64 4)
  %317 = add <8 x i64> %315, %316
  %318 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %314, 0
  %319 = insertvalue { <8 x ptr>, <8 x i64> } %318, <8 x i64> %317, 1
  %320 = extractvalue { <8 x ptr>, <8 x i64> } %319, 0
  %321 = extractvalue { <8 x ptr>, <8 x i64> } %319, 1
  %322 = getelementptr i8, <8 x ptr> %320, <8 x i64> %321
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %.splat149, <8 x ptr> %322, i32 1, <8 x i1> %53)
  %.state154 = load i64, ptr %.slot34, align 4
  %323 = add i64 %.state154, 1
  store i64 %323, ptr %.slot34, align 4
  br label %direct.schedule.23

direct.schedule.25:                               ; preds = %direct.false144
  %.spill.load155 = load <8 x float>, ptr %.spill32, align 32
  %324 = fadd <8 x float> %.spill.load155, splat (float 0x3EE4F8B580000000)
  %325 = call <8 x float> @llvm.sqrt.v8f32(<8 x float> %324)
  %326 = load <8 x float>, ptr %.spill35, align 32
  %327 = select <8 x i1> %53, <8 x float> %325, <8 x float> %326
  store <8 x float> %327, ptr %.spill35, align 32
  store i64 0, ptr %.slot36, align 4
  br label %direct.schedule.26

direct.schedule.26:                               ; preds = %direct.schedule.27, %direct.schedule.25
  %.state156 = load i64, ptr %.slot36, align 4
  %328 = icmp slt i64 %.state156, 4096
  br i1 %328, label %direct.true157, label %direct.false158

direct.schedule.27:                               ; preds = %direct.true157
  %.state159 = load i64, ptr %.slot36, align 4
  %329 = srem i64 %.state159, 4096
  %.state160 = load i64, ptr %.slot36, align 4
  %330 = sdiv i64 %.state160, 4096
  %331 = srem i64 %330, 1
  %.spill.load161 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert162 = insertelement <8 x i64> poison, i64 %331, i64 0
  %.splat163 = shufflevector <8 x i64> %.splatinsert162, <8 x i64> poison, <8 x i32> zeroinitializer
  %332 = add <8 x i64> %.spill.load161, %.splat163
  %333 = add <8 x i64> zeroinitializer, %332
  %334 = add i64 0, %329
  %335 = mul <8 x i64> %333, splat (i64 4096)
  %.splatinsert164 = insertelement <8 x i64> poison, i64 %334, i64 0
  %.splat165 = shufflevector <8 x i64> %.splatinsert164, <8 x i64> poison, <8 x i32> zeroinitializer
  %336 = add <8 x i64> %335, %.splat165
  %337 = add i64 0, %329
  %338 = srem i64 %337, 4096
  %339 = add i64 0, %338
  %tile_snapshot.spill.load166 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %340 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load166, 0
  %341 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load166, 1
  %.splatinsert167 = insertelement <8 x i64> poison, i64 %339, i64 0
  %.splat168 = shufflevector <8 x i64> %.splatinsert167, <8 x i64> poison, <8 x i32> zeroinitializer
  %342 = mul <8 x i64> %.splat168, splat (i64 4)
  %343 = add <8 x i64> %341, %342
  %344 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %340, 0
  %345 = insertvalue { <8 x ptr>, <8 x i64> } %344, <8 x i64> %343, 1
  %346 = extractvalue { <8 x ptr>, <8 x i64> } %345, 0
  %347 = extractvalue { <8 x ptr>, <8 x i64> } %345, 1
  %348 = getelementptr i8, <8 x ptr> %346, <8 x i64> %347
  %349 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %348, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %.spill.load169 = load <8 x float>, ptr %.spill35, align 32
  %350 = fdiv <8 x float> %349, %.spill.load169
  %tile_snapshot.spill.load170 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill33, align 64
  %351 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load170, 0
  %352 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load170, 1
  %.splatinsert171 = insertelement <8 x i64> poison, i64 %337, i64 0
  %.splat172 = shufflevector <8 x i64> %.splatinsert171, <8 x i64> poison, <8 x i32> zeroinitializer
  %353 = mul <8 x i64> %.splat172, splat (i64 4)
  %354 = add <8 x i64> %352, %353
  %355 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %351, 0
  %356 = insertvalue { <8 x ptr>, <8 x i64> } %355, <8 x i64> %354, 1
  %357 = extractvalue { <8 x ptr>, <8 x i64> } %356, 0
  %358 = extractvalue { <8 x ptr>, <8 x i64> } %356, 1
  %359 = getelementptr i8, <8 x ptr> %357, <8 x i64> %358
  %360 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %359, i32 1, <8 x i1> %53, <8 x float> zeroinitializer)
  %361 = fmul <8 x float> %350, %360
  %362 = extractvalue { ptr, i64 } %29, 0
  %363 = mul <8 x i64> %336, splat (i64 4)
  %364 = getelementptr i8, ptr %362, <8 x i64> %363
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %361, <8 x ptr> %364, i32 1, <8 x i1> %53)
  %.state173 = load i64, ptr %.slot36, align 4
  %365 = add i64 %.state173, 1
  store i64 %365, ptr %.slot36, align 4
  br label %direct.schedule.26

direct.schedule.28:                               ; preds = %direct.false158
  ret void

direct.activate:                                  ; preds = %prologue
  br label %direct.schedule.0

direct.inactive:                                  ; preds = %prologue
  ret void

direct.true:                                      ; preds = %direct.schedule.1
  br label %direct.schedule.2

direct.false:                                     ; preds = %direct.schedule.1
  br label %direct.schedule.3

direct.true57:                                    ; preds = %direct.schedule.3
  br label %direct.schedule.4

direct.false58:                                   ; preds = %direct.schedule.3
  %366 = load <8 x float>, ptr %.slot9, align 32
  %367 = select <8 x i1> %53, <8 x float> zeroinitializer, <8 x float> %366
  store <8 x float> %367, ptr %.slot9, align 32
  br label %direct.schedule.5

direct.true64:                                    ; preds = %direct.schedule.5
  br label %direct.schedule.6

direct.false65:                                   ; preds = %direct.schedule.5
  %368 = load <8 x float>, ptr %.slot11, align 32
  %369 = select <8 x i1> %53, <8 x float> zeroinitializer, <8 x float> %368
  store <8 x float> %369, ptr %.slot11, align 32
  br label %direct.schedule.7

direct.true71:                                    ; preds = %direct.schedule.7
  br label %direct.schedule.8

direct.false72:                                   ; preds = %direct.schedule.7
  %370 = load <8 x float>, ptr %.slot13, align 32
  %371 = select <8 x i1> %53, <8 x float> zeroinitializer, <8 x float> %370
  store <8 x float> %371, ptr %.slot13, align 32
  br label %direct.schedule.9

direct.true78:                                    ; preds = %direct.schedule.9
  br label %direct.schedule.10

direct.false79:                                   ; preds = %direct.schedule.9
  %372 = load <8 x float>, ptr %.slot15, align 32
  %373 = select <8 x i1> %53, <8 x float> zeroinitializer, <8 x float> %372
  store <8 x float> %373, ptr %.slot15, align 32
  br label %direct.schedule.11

direct.true89:                                    ; preds = %direct.schedule.12
  br label %direct.schedule.13

direct.false90:                                   ; preds = %direct.schedule.12
  br label %direct.schedule.22

direct.true93:                                    ; preds = %direct.schedule.13
  br label %direct.schedule.14

direct.false94:                                   ; preds = %direct.schedule.13
  %374 = load <8 x float>, ptr %.slot22, align 32
  %375 = select <8 x i1> %53, <8 x float> zeroinitializer, <8 x float> %374
  store <8 x float> %375, ptr %.slot22, align 32
  br label %direct.schedule.15

direct.true103:                                   ; preds = %direct.schedule.15
  br label %direct.schedule.16

direct.false104:                                  ; preds = %direct.schedule.15
  %376 = load <8 x float>, ptr %.slot25, align 32
  %377 = select <8 x i1> %53, <8 x float> zeroinitializer, <8 x float> %376
  store <8 x float> %377, ptr %.slot25, align 32
  br label %direct.schedule.17

direct.true113:                                   ; preds = %direct.schedule.17
  br label %direct.schedule.18

direct.false114:                                  ; preds = %direct.schedule.17
  %378 = load <8 x float>, ptr %.slot28, align 32
  %379 = select <8 x i1> %53, <8 x float> zeroinitializer, <8 x float> %378
  store <8 x float> %379, ptr %.slot28, align 32
  br label %direct.schedule.19

direct.true123:                                   ; preds = %direct.schedule.19
  br label %direct.schedule.20

direct.false124:                                  ; preds = %direct.schedule.19
  %380 = load <8 x float>, ptr %.slot31, align 32
  %381 = select <8 x i1> %53, <8 x float> zeroinitializer, <8 x float> %380
  store <8 x float> %381, ptr %.slot31, align 32
  br label %direct.schedule.21

direct.true143:                                   ; preds = %direct.schedule.23
  br label %direct.schedule.24

direct.false144:                                  ; preds = %direct.schedule.23
  br label %direct.schedule.25

direct.true157:                                   ; preds = %direct.schedule.26
  br label %direct.schedule.27

direct.false158:                                  ; preds = %direct.schedule.26
  br label %direct.schedule.28
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
