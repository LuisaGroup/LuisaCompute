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
  %2 = insertvalue { <8 x ptr>, <8 x i64> } %1, <8 x i64> <i64 0, i64 8192, i64 16384, i64 24576, i64 32768, i64 40960, i64 49152, i64 57344>, 1
  %3 = getelementptr i8, ptr %launch_config, i64 136
  %private.workspace1 = load ptr, ptr %3, align 8
  %tile_snapshot.private2 = getelementptr inbounds i8, ptr %private.workspace1, i64 65536
  %.splatinsert3 = insertelement <8 x ptr> poison, ptr %tile_snapshot.private2, i64 0
  %.splat4 = shufflevector <8 x ptr> %.splatinsert3, <8 x ptr> poison, <8 x i32> zeroinitializer
  %4 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat4, 0
  %5 = insertvalue { <8 x ptr>, <8 x i64> } %4, <8 x i64> <i64 0, i64 8192, i64 16384, i64 24576, i64 32768, i64 40960, i64 49152, i64 57344>, 1
  %6 = getelementptr i8, ptr %launch_config, i64 136
  %private.workspace5 = load ptr, ptr %6, align 8
  %tile_snapshot.private6 = getelementptr inbounds i8, ptr %private.workspace5, i64 131072
  %.splatinsert7 = insertelement <8 x ptr> poison, ptr %tile_snapshot.private6, i64 0
  %.splat8 = shufflevector <8 x ptr> %.splatinsert7, <8 x ptr> poison, <8 x i32> zeroinitializer
  %7 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat8, 0
  %8 = insertvalue { <8 x ptr>, <8 x i64> } %7, <8 x i64> <i64 0, i64 8192, i64 16384, i64 24576, i64 32768, i64 40960, i64 49152, i64 57344>, 1
  %9 = getelementptr i8, ptr %launch_config, i64 136
  %private.workspace9 = load ptr, ptr %9, align 8
  %tile_snapshot.private10 = getelementptr inbounds i8, ptr %private.workspace9, i64 196608
  %.splatinsert11 = insertelement <8 x ptr> poison, ptr %tile_snapshot.private10, i64 0
  %.splat12 = shufflevector <8 x ptr> %.splatinsert11, <8 x ptr> poison, <8 x i32> zeroinitializer
  %10 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat12, 0
  %11 = insertvalue { <8 x ptr>, <8 x i64> } %10, <8 x i64> <i64 0, i64 8192, i64 16384, i64 24576, i64 32768, i64 40960, i64 49152, i64 57344>, 1
  %.spill = alloca <8 x i64>, align 64
  store <8 x i64> zeroinitializer, ptr %.spill, align 64
  %tile_snapshot.spill = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill, align 64
  %.slot = alloca i64, align 8
  store i64 0, ptr %.slot, align 4
  %tile_snapshot.spill13 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill13, align 64
  %.slot14 = alloca i64, align 8
  store i64 0, ptr %.slot14, align 4
  %tile_snapshot.spill15 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill15, align 64
  %.slot16 = alloca i64, align 8
  store i64 0, ptr %.slot16, align 4
  %tile_snapshot.spill17 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill17, align 64
  %.slot18 = alloca i64, align 8
  store i64 0, ptr %.slot18, align 4
  %.slot19 = alloca i64, align 8
  store i64 0, ptr %.slot19, align 4
  %.slot20 = alloca i64, align 8
  store i64 0, ptr %.slot20, align 4
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
  %.splatinsert21 = insertelement <8 x i32> poison, i32 %48, i64 0
  %.splat22 = shufflevector <8 x i32> %.splatinsert21, <8 x i32> poison, <8 x i32> zeroinitializer
  %49 = add <8 x i32> %.splat22, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %50 = mul i32 %36, 32
  %.splatinsert23 = insertelement <8 x i32> poison, i32 %50, i64 0
  %.splat24 = shufflevector <8 x i32> %.splatinsert23, <8 x i32> poison, <8 x i32> zeroinitializer
  %51 = add <8 x i32> %.splat24, %49
  %52 = mul i32 %40, 1
  %.splatinsert25 = insertelement <8 x i32> poison, i32 %52, i64 0
  %.splat26 = shufflevector <8 x i32> %.splatinsert25, <8 x i32> poison, <8 x i32> zeroinitializer
  %53 = add <8 x i32> %.splat26, zeroinitializer
  %54 = mul i32 %44, 1
  %.splatinsert27 = insertelement <8 x i32> poison, i32 %54, i64 0
  %.splat28 = shufflevector <8 x i32> %.splatinsert27, <8 x i32> poison, <8 x i32> zeroinitializer
  %55 = add <8 x i32> %.splat28, zeroinitializer
  %56 = insertvalue [3 x <8 x i32>] poison, <8 x i32> %51, 0
  %57 = insertvalue [3 x <8 x i32>] %56, <8 x i32> %53, 1
  %58 = insertvalue [3 x <8 x i32>] %57, <8 x i32> %55, 2
  %.splatinsert29 = insertelement <8 x i32> poison, i32 %active_lane_count, i64 0
  %.splat30 = shufflevector <8 x i32> %.splatinsert29, <8 x i32> poison, <8 x i32> zeroinitializer
  %59 = icmp ult <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, %.splat30
  %60 = call i1 @llvm.vector.reduce.or.v8i1(<8 x i1> %59)
  br i1 %60, label %direct.activate, label %direct.inactive

direct.schedule.0:                                ; preds = %direct.activate
  %61 = extractvalue [3 x <8 x i32>] %58, 0
  %62 = zext <8 x i32> %61 to <8 x i64>
  %63 = select <8 x i1> %59, <8 x i64> %62, <8 x i64> zeroinitializer
  %64 = select <8 x i1> %59, <8 x i64> splat (i64 1), <8 x i64> splat (i64 1)
  %65 = sdiv <8 x i64> %63, %64
  %66 = select <8 x i1> %59, <8 x i64> %65, <8 x i64> zeroinitializer
  %67 = select <8 x i1> %59, <8 x i64> splat (i64 64), <8 x i64> splat (i64 1)
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
  %80 = icmp slt i64 %.state, 2048
  br i1 %80, label %direct.true, label %direct.false

direct.schedule.2:                                ; preds = %direct.true
  %.state31 = load i64, ptr %.slot, align 4
  %81 = srem i64 %.state31, 2048
  %.state32 = load i64, ptr %.slot, align 4
  %82 = sdiv i64 %.state32, 2048
  %83 = srem i64 %82, 1
  %.spill.load = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert33 = insertelement <8 x i64> poison, i64 %83, i64 0
  %.splat34 = shufflevector <8 x i64> %.splatinsert33, <8 x i64> poison, <8 x i32> zeroinitializer
  %84 = add <8 x i64> %.spill.load, %.splat34
  %85 = add <8 x i64> zeroinitializer, %84
  %86 = add i64 0, %81
  %87 = mul <8 x i64> %85, splat (i64 4096)
  %.splatinsert35 = insertelement <8 x i64> poison, i64 %86, i64 0
  %.splat36 = shufflevector <8 x i64> %.splatinsert35, <8 x i64> poison, <8 x i32> zeroinitializer
  %88 = add <8 x i64> %87, %.splat36
  %89 = extractvalue { ptr, i64 } %17, 0
  %90 = mul <8 x i64> %88, splat (i64 4)
  %91 = getelementptr i8, ptr %89, <8 x i64> %90
  %92 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %91, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %93 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 0
  %94 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 1
  %.state37 = load i64, ptr %.slot, align 4
  %.splatinsert38 = insertelement <8 x i64> poison, i64 %.state37, i64 0
  %.splat39 = shufflevector <8 x i64> %.splatinsert38, <8 x i64> poison, <8 x i32> zeroinitializer
  %95 = mul <8 x i64> %.splat39, splat (i64 4)
  %96 = add <8 x i64> %94, %95
  %97 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %93, 0
  %98 = insertvalue { <8 x ptr>, <8 x i64> } %97, <8 x i64> %96, 1
  %99 = extractvalue { <8 x ptr>, <8 x i64> } %98, 0
  %100 = extractvalue { <8 x ptr>, <8 x i64> } %98, 1
  %101 = getelementptr i8, <8 x ptr> %99, <8 x i64> %100
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %92, <8 x ptr> %101, i32 1, <8 x i1> %59)
  %.state40 = load i64, ptr %.slot, align 4
  %102 = add i64 %.state40, 1
  store i64 %102, ptr %.slot, align 4
  br label %direct.schedule.1

direct.schedule.3:                                ; preds = %direct.false
  %103 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill13, align 64
  %104 = extractvalue { <8 x ptr>, <8 x i64> } %5, 0
  %105 = extractvalue { <8 x ptr>, <8 x i64> } %103, 0
  %106 = select <8 x i1> %59, <8 x ptr> %104, <8 x ptr> %105
  %107 = extractvalue { <8 x ptr>, <8 x i64> } %5, 1
  %108 = extractvalue { <8 x ptr>, <8 x i64> } %103, 1
  %109 = select <8 x i1> %59, <8 x i64> %107, <8 x i64> %108
  %110 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %106, 0
  %111 = insertvalue { <8 x ptr>, <8 x i64> } %110, <8 x i64> %109, 1
  store { <8 x ptr>, <8 x i64> } %111, ptr %tile_snapshot.spill13, align 64
  store i64 0, ptr %.slot14, align 4
  br label %direct.schedule.4

direct.schedule.4:                                ; preds = %direct.schedule.5, %direct.schedule.3
  %.state41 = load i64, ptr %.slot14, align 4
  %112 = icmp slt i64 %.state41, 2048
  br i1 %112, label %direct.true42, label %direct.false43

direct.schedule.5:                                ; preds = %direct.true42
  %.state44 = load i64, ptr %.slot14, align 4
  %113 = srem i64 %.state44, 2048
  %.state45 = load i64, ptr %.slot14, align 4
  %114 = sdiv i64 %.state45, 2048
  %115 = srem i64 %114, 1
  %.spill.load46 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert47 = insertelement <8 x i64> poison, i64 %115, i64 0
  %.splat48 = shufflevector <8 x i64> %.splatinsert47, <8 x i64> poison, <8 x i32> zeroinitializer
  %116 = add <8 x i64> %.spill.load46, %.splat48
  %117 = add <8 x i64> zeroinitializer, %116
  %118 = add i64 2048, %113
  %119 = mul <8 x i64> %117, splat (i64 4096)
  %.splatinsert49 = insertelement <8 x i64> poison, i64 %118, i64 0
  %.splat50 = shufflevector <8 x i64> %.splatinsert49, <8 x i64> poison, <8 x i32> zeroinitializer
  %120 = add <8 x i64> %119, %.splat50
  %121 = extractvalue { ptr, i64 } %17, 0
  %122 = mul <8 x i64> %120, splat (i64 4)
  %123 = getelementptr i8, ptr %121, <8 x i64> %122
  %124 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %123, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load51 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill13, align 64
  %125 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load51, 0
  %126 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load51, 1
  %.state52 = load i64, ptr %.slot14, align 4
  %.splatinsert53 = insertelement <8 x i64> poison, i64 %.state52, i64 0
  %.splat54 = shufflevector <8 x i64> %.splatinsert53, <8 x i64> poison, <8 x i32> zeroinitializer
  %127 = mul <8 x i64> %.splat54, splat (i64 4)
  %128 = add <8 x i64> %126, %127
  %129 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %125, 0
  %130 = insertvalue { <8 x ptr>, <8 x i64> } %129, <8 x i64> %128, 1
  %131 = extractvalue { <8 x ptr>, <8 x i64> } %130, 0
  %132 = extractvalue { <8 x ptr>, <8 x i64> } %130, 1
  %133 = getelementptr i8, <8 x ptr> %131, <8 x i64> %132
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %124, <8 x ptr> %133, i32 1, <8 x i1> %59)
  %.state55 = load i64, ptr %.slot14, align 4
  %134 = add i64 %.state55, 1
  store i64 %134, ptr %.slot14, align 4
  br label %direct.schedule.4

direct.schedule.6:                                ; preds = %direct.false43
  %135 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill15, align 64
  %136 = extractvalue { <8 x ptr>, <8 x i64> } %8, 0
  %137 = extractvalue { <8 x ptr>, <8 x i64> } %135, 0
  %138 = select <8 x i1> %59, <8 x ptr> %136, <8 x ptr> %137
  %139 = extractvalue { <8 x ptr>, <8 x i64> } %8, 1
  %140 = extractvalue { <8 x ptr>, <8 x i64> } %135, 1
  %141 = select <8 x i1> %59, <8 x i64> %139, <8 x i64> %140
  %142 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %138, 0
  %143 = insertvalue { <8 x ptr>, <8 x i64> } %142, <8 x i64> %141, 1
  store { <8 x ptr>, <8 x i64> } %143, ptr %tile_snapshot.spill15, align 64
  store i64 0, ptr %.slot16, align 4
  br label %direct.schedule.7

direct.schedule.7:                                ; preds = %direct.schedule.8, %direct.schedule.6
  %.state56 = load i64, ptr %.slot16, align 4
  %144 = icmp slt i64 %.state56, 2048
  br i1 %144, label %direct.true57, label %direct.false58

direct.schedule.8:                                ; preds = %direct.true57
  %.state59 = load i64, ptr %.slot16, align 4
  %145 = srem i64 %.state59, 2048
  %.state60 = load i64, ptr %.slot16, align 4
  %146 = sdiv i64 %.state60, 2048
  %147 = srem i64 %146, 1
  %.spill.load61 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert62 = insertelement <8 x i64> poison, i64 %147, i64 0
  %.splat63 = shufflevector <8 x i64> %.splatinsert62, <8 x i64> poison, <8 x i32> zeroinitializer
  %148 = add <8 x i64> %.spill.load61, %.splat63
  %149 = add <8 x i64> zeroinitializer, %148
  %150 = add i64 0, %145
  %151 = mul <8 x i64> %149, splat (i64 2048)
  %.splatinsert64 = insertelement <8 x i64> poison, i64 %150, i64 0
  %.splat65 = shufflevector <8 x i64> %.splatinsert64, <8 x i64> poison, <8 x i32> zeroinitializer
  %152 = add <8 x i64> %151, %.splat65
  %153 = extractvalue { ptr, i64 } %23, 0
  %154 = mul <8 x i64> %152, splat (i64 4)
  %155 = getelementptr i8, ptr %153, <8 x i64> %154
  %156 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %155, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load66 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill15, align 64
  %157 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load66, 0
  %158 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load66, 1
  %.state67 = load i64, ptr %.slot16, align 4
  %.splatinsert68 = insertelement <8 x i64> poison, i64 %.state67, i64 0
  %.splat69 = shufflevector <8 x i64> %.splatinsert68, <8 x i64> poison, <8 x i32> zeroinitializer
  %159 = mul <8 x i64> %.splat69, splat (i64 4)
  %160 = add <8 x i64> %158, %159
  %161 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %157, 0
  %162 = insertvalue { <8 x ptr>, <8 x i64> } %161, <8 x i64> %160, 1
  %163 = extractvalue { <8 x ptr>, <8 x i64> } %162, 0
  %164 = extractvalue { <8 x ptr>, <8 x i64> } %162, 1
  %165 = getelementptr i8, <8 x ptr> %163, <8 x i64> %164
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %156, <8 x ptr> %165, i32 1, <8 x i1> %59)
  %.state70 = load i64, ptr %.slot16, align 4
  %166 = add i64 %.state70, 1
  store i64 %166, ptr %.slot16, align 4
  br label %direct.schedule.7

direct.schedule.9:                                ; preds = %direct.false58
  %167 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill17, align 64
  %168 = extractvalue { <8 x ptr>, <8 x i64> } %11, 0
  %169 = extractvalue { <8 x ptr>, <8 x i64> } %167, 0
  %170 = select <8 x i1> %59, <8 x ptr> %168, <8 x ptr> %169
  %171 = extractvalue { <8 x ptr>, <8 x i64> } %11, 1
  %172 = extractvalue { <8 x ptr>, <8 x i64> } %167, 1
  %173 = select <8 x i1> %59, <8 x i64> %171, <8 x i64> %172
  %174 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %170, 0
  %175 = insertvalue { <8 x ptr>, <8 x i64> } %174, <8 x i64> %173, 1
  store { <8 x ptr>, <8 x i64> } %175, ptr %tile_snapshot.spill17, align 64
  store i64 0, ptr %.slot18, align 4
  br label %direct.schedule.10

direct.schedule.10:                               ; preds = %direct.schedule.11, %direct.schedule.9
  %.state71 = load i64, ptr %.slot18, align 4
  %176 = icmp slt i64 %.state71, 2048
  br i1 %176, label %direct.true72, label %direct.false73

direct.schedule.11:                               ; preds = %direct.true72
  %.state74 = load i64, ptr %.slot18, align 4
  %177 = srem i64 %.state74, 2048
  %.state75 = load i64, ptr %.slot18, align 4
  %178 = sdiv i64 %.state75, 2048
  %179 = srem i64 %178, 1
  %.spill.load76 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert77 = insertelement <8 x i64> poison, i64 %179, i64 0
  %.splat78 = shufflevector <8 x i64> %.splatinsert77, <8 x i64> poison, <8 x i32> zeroinitializer
  %180 = add <8 x i64> %.spill.load76, %.splat78
  %181 = add <8 x i64> zeroinitializer, %180
  %182 = add i64 0, %177
  %183 = mul <8 x i64> %181, splat (i64 2048)
  %.splatinsert79 = insertelement <8 x i64> poison, i64 %182, i64 0
  %.splat80 = shufflevector <8 x i64> %.splatinsert79, <8 x i64> poison, <8 x i32> zeroinitializer
  %184 = add <8 x i64> %183, %.splat80
  %185 = extractvalue { ptr, i64 } %29, 0
  %186 = mul <8 x i64> %184, splat (i64 4)
  %187 = getelementptr i8, ptr %185, <8 x i64> %186
  %188 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %187, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load81 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill17, align 64
  %189 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load81, 0
  %190 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load81, 1
  %.state82 = load i64, ptr %.slot18, align 4
  %.splatinsert83 = insertelement <8 x i64> poison, i64 %.state82, i64 0
  %.splat84 = shufflevector <8 x i64> %.splatinsert83, <8 x i64> poison, <8 x i32> zeroinitializer
  %191 = mul <8 x i64> %.splat84, splat (i64 4)
  %192 = add <8 x i64> %190, %191
  %193 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %189, 0
  %194 = insertvalue { <8 x ptr>, <8 x i64> } %193, <8 x i64> %192, 1
  %195 = extractvalue { <8 x ptr>, <8 x i64> } %194, 0
  %196 = extractvalue { <8 x ptr>, <8 x i64> } %194, 1
  %197 = getelementptr i8, <8 x ptr> %195, <8 x i64> %196
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %188, <8 x ptr> %197, i32 1, <8 x i1> %59)
  %.state85 = load i64, ptr %.slot18, align 4
  %198 = add i64 %.state85, 1
  store i64 %198, ptr %.slot18, align 4
  br label %direct.schedule.10

direct.schedule.12:                               ; preds = %direct.false73
  store i64 0, ptr %.slot19, align 4
  br label %direct.schedule.13

direct.schedule.13:                               ; preds = %direct.schedule.14, %direct.schedule.12
  %.state86 = load i64, ptr %.slot19, align 4
  %199 = icmp slt i64 %.state86, 2048
  br i1 %199, label %direct.true87, label %direct.false88

direct.schedule.14:                               ; preds = %direct.true87
  %.state89 = load i64, ptr %.slot19, align 4
  %200 = srem i64 %.state89, 2048
  %.state90 = load i64, ptr %.slot19, align 4
  %201 = sdiv i64 %.state90, 2048
  %202 = srem i64 %201, 1
  %.spill.load91 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert92 = insertelement <8 x i64> poison, i64 %202, i64 0
  %.splat93 = shufflevector <8 x i64> %.splatinsert92, <8 x i64> poison, <8 x i32> zeroinitializer
  %203 = add <8 x i64> %.spill.load91, %.splat93
  %204 = add <8 x i64> zeroinitializer, %203
  %205 = add i64 0, %200
  %206 = mul <8 x i64> %204, splat (i64 4096)
  %.splatinsert94 = insertelement <8 x i64> poison, i64 %205, i64 0
  %.splat95 = shufflevector <8 x i64> %.splatinsert94, <8 x i64> poison, <8 x i32> zeroinitializer
  %207 = add <8 x i64> %206, %.splat95
  %208 = add i64 0, %200
  %209 = srem i64 %208, 2048
  %210 = add i64 0, %209
  %tile_snapshot.spill.load96 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %211 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load96, 0
  %212 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load96, 1
  %.splatinsert97 = insertelement <8 x i64> poison, i64 %210, i64 0
  %.splat98 = shufflevector <8 x i64> %.splatinsert97, <8 x i64> poison, <8 x i32> zeroinitializer
  %213 = mul <8 x i64> %.splat98, splat (i64 4)
  %214 = add <8 x i64> %212, %213
  %215 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %211, 0
  %216 = insertvalue { <8 x ptr>, <8 x i64> } %215, <8 x i64> %214, 1
  %217 = extractvalue { <8 x ptr>, <8 x i64> } %216, 0
  %218 = extractvalue { <8 x ptr>, <8 x i64> } %216, 1
  %219 = getelementptr i8, <8 x ptr> %217, <8 x i64> %218
  %220 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %219, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load99 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill15, align 64
  %221 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load99, 0
  %222 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load99, 1
  %.splatinsert100 = insertelement <8 x i64> poison, i64 %210, i64 0
  %.splat101 = shufflevector <8 x i64> %.splatinsert100, <8 x i64> poison, <8 x i32> zeroinitializer
  %223 = mul <8 x i64> %.splat101, splat (i64 4)
  %224 = add <8 x i64> %222, %223
  %225 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %221, 0
  %226 = insertvalue { <8 x ptr>, <8 x i64> } %225, <8 x i64> %224, 1
  %227 = extractvalue { <8 x ptr>, <8 x i64> } %226, 0
  %228 = extractvalue { <8 x ptr>, <8 x i64> } %226, 1
  %229 = getelementptr i8, <8 x ptr> %227, <8 x i64> %228
  %230 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %229, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %231 = fmul <8 x float> %220, %230
  %tile_snapshot.spill.load102 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill13, align 64
  %232 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load102, 0
  %233 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load102, 1
  %.splatinsert103 = insertelement <8 x i64> poison, i64 %210, i64 0
  %.splat104 = shufflevector <8 x i64> %.splatinsert103, <8 x i64> poison, <8 x i32> zeroinitializer
  %234 = mul <8 x i64> %.splat104, splat (i64 4)
  %235 = add <8 x i64> %233, %234
  %236 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %232, 0
  %237 = insertvalue { <8 x ptr>, <8 x i64> } %236, <8 x i64> %235, 1
  %238 = extractvalue { <8 x ptr>, <8 x i64> } %237, 0
  %239 = extractvalue { <8 x ptr>, <8 x i64> } %237, 1
  %240 = getelementptr i8, <8 x ptr> %238, <8 x i64> %239
  %241 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %240, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load105 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill17, align 64
  %242 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load105, 0
  %243 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load105, 1
  %.splatinsert106 = insertelement <8 x i64> poison, i64 %210, i64 0
  %.splat107 = shufflevector <8 x i64> %.splatinsert106, <8 x i64> poison, <8 x i32> zeroinitializer
  %244 = mul <8 x i64> %.splat107, splat (i64 4)
  %245 = add <8 x i64> %243, %244
  %246 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %242, 0
  %247 = insertvalue { <8 x ptr>, <8 x i64> } %246, <8 x i64> %245, 1
  %248 = extractvalue { <8 x ptr>, <8 x i64> } %247, 0
  %249 = extractvalue { <8 x ptr>, <8 x i64> } %247, 1
  %250 = getelementptr i8, <8 x ptr> %248, <8 x i64> %249
  %251 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %250, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %252 = fmul <8 x float> %241, %251
  %253 = fsub <8 x float> %231, %252
  %254 = extractvalue { ptr, i64 } %35, 0
  %255 = mul <8 x i64> %207, splat (i64 4)
  %256 = getelementptr i8, ptr %254, <8 x i64> %255
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %253, <8 x ptr> %256, i32 1, <8 x i1> %59)
  %.state108 = load i64, ptr %.slot19, align 4
  %257 = add i64 %.state108, 1
  store i64 %257, ptr %.slot19, align 4
  br label %direct.schedule.13

direct.schedule.15:                               ; preds = %direct.false88
  store i64 0, ptr %.slot20, align 4
  br label %direct.schedule.16

direct.schedule.16:                               ; preds = %direct.schedule.17, %direct.schedule.15
  %.state109 = load i64, ptr %.slot20, align 4
  %258 = icmp slt i64 %.state109, 2048
  br i1 %258, label %direct.true110, label %direct.false111

direct.schedule.17:                               ; preds = %direct.true110
  %.state112 = load i64, ptr %.slot20, align 4
  %259 = srem i64 %.state112, 2048
  %.state113 = load i64, ptr %.slot20, align 4
  %260 = sdiv i64 %.state113, 2048
  %261 = srem i64 %260, 1
  %.spill.load114 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert115 = insertelement <8 x i64> poison, i64 %261, i64 0
  %.splat116 = shufflevector <8 x i64> %.splatinsert115, <8 x i64> poison, <8 x i32> zeroinitializer
  %262 = add <8 x i64> %.spill.load114, %.splat116
  %263 = add <8 x i64> zeroinitializer, %262
  %264 = add i64 2048, %259
  %265 = mul <8 x i64> %263, splat (i64 4096)
  %.splatinsert117 = insertelement <8 x i64> poison, i64 %264, i64 0
  %.splat118 = shufflevector <8 x i64> %.splatinsert117, <8 x i64> poison, <8 x i32> zeroinitializer
  %266 = add <8 x i64> %265, %.splat118
  %267 = add i64 0, %259
  %268 = srem i64 %267, 2048
  %269 = add i64 0, %268
  %tile_snapshot.spill.load119 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %270 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load119, 0
  %271 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load119, 1
  %.splatinsert120 = insertelement <8 x i64> poison, i64 %269, i64 0
  %.splat121 = shufflevector <8 x i64> %.splatinsert120, <8 x i64> poison, <8 x i32> zeroinitializer
  %272 = mul <8 x i64> %.splat121, splat (i64 4)
  %273 = add <8 x i64> %271, %272
  %274 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %270, 0
  %275 = insertvalue { <8 x ptr>, <8 x i64> } %274, <8 x i64> %273, 1
  %276 = extractvalue { <8 x ptr>, <8 x i64> } %275, 0
  %277 = extractvalue { <8 x ptr>, <8 x i64> } %275, 1
  %278 = getelementptr i8, <8 x ptr> %276, <8 x i64> %277
  %279 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %278, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load122 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill17, align 64
  %280 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load122, 0
  %281 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load122, 1
  %.splatinsert123 = insertelement <8 x i64> poison, i64 %269, i64 0
  %.splat124 = shufflevector <8 x i64> %.splatinsert123, <8 x i64> poison, <8 x i32> zeroinitializer
  %282 = mul <8 x i64> %.splat124, splat (i64 4)
  %283 = add <8 x i64> %281, %282
  %284 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %280, 0
  %285 = insertvalue { <8 x ptr>, <8 x i64> } %284, <8 x i64> %283, 1
  %286 = extractvalue { <8 x ptr>, <8 x i64> } %285, 0
  %287 = extractvalue { <8 x ptr>, <8 x i64> } %285, 1
  %288 = getelementptr i8, <8 x ptr> %286, <8 x i64> %287
  %289 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %288, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %290 = fmul <8 x float> %279, %289
  %tile_snapshot.spill.load125 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill13, align 64
  %291 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load125, 0
  %292 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load125, 1
  %.splatinsert126 = insertelement <8 x i64> poison, i64 %269, i64 0
  %.splat127 = shufflevector <8 x i64> %.splatinsert126, <8 x i64> poison, <8 x i32> zeroinitializer
  %293 = mul <8 x i64> %.splat127, splat (i64 4)
  %294 = add <8 x i64> %292, %293
  %295 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %291, 0
  %296 = insertvalue { <8 x ptr>, <8 x i64> } %295, <8 x i64> %294, 1
  %297 = extractvalue { <8 x ptr>, <8 x i64> } %296, 0
  %298 = extractvalue { <8 x ptr>, <8 x i64> } %296, 1
  %299 = getelementptr i8, <8 x ptr> %297, <8 x i64> %298
  %300 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %299, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load128 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill15, align 64
  %301 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load128, 0
  %302 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load128, 1
  %.splatinsert129 = insertelement <8 x i64> poison, i64 %269, i64 0
  %.splat130 = shufflevector <8 x i64> %.splatinsert129, <8 x i64> poison, <8 x i32> zeroinitializer
  %303 = mul <8 x i64> %.splat130, splat (i64 4)
  %304 = add <8 x i64> %302, %303
  %305 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %301, 0
  %306 = insertvalue { <8 x ptr>, <8 x i64> } %305, <8 x i64> %304, 1
  %307 = extractvalue { <8 x ptr>, <8 x i64> } %306, 0
  %308 = extractvalue { <8 x ptr>, <8 x i64> } %306, 1
  %309 = getelementptr i8, <8 x ptr> %307, <8 x i64> %308
  %310 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %309, i32 1, <8 x i1> %59, <8 x float> zeroinitializer)
  %311 = fmul <8 x float> %300, %310
  %312 = fadd <8 x float> %290, %311
  %313 = extractvalue { ptr, i64 } %35, 0
  %314 = mul <8 x i64> %266, splat (i64 4)
  %315 = getelementptr i8, ptr %313, <8 x i64> %314
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %312, <8 x ptr> %315, i32 1, <8 x i1> %59)
  %.state131 = load i64, ptr %.slot20, align 4
  %316 = add i64 %.state131, 1
  store i64 %316, ptr %.slot20, align 4
  br label %direct.schedule.16

direct.schedule.18:                               ; preds = %direct.false111
  ret void

direct.activate:                                  ; preds = %prologue
  br label %direct.schedule.0

direct.inactive:                                  ; preds = %prologue
  ret void

direct.true:                                      ; preds = %direct.schedule.1
  br label %direct.schedule.2

direct.false:                                     ; preds = %direct.schedule.1
  br label %direct.schedule.3

direct.true42:                                    ; preds = %direct.schedule.4
  br label %direct.schedule.5

direct.false43:                                   ; preds = %direct.schedule.4
  br label %direct.schedule.6

direct.true57:                                    ; preds = %direct.schedule.7
  br label %direct.schedule.8

direct.false58:                                   ; preds = %direct.schedule.7
  br label %direct.schedule.9

direct.true72:                                    ; preds = %direct.schedule.10
  br label %direct.schedule.11

direct.false73:                                   ; preds = %direct.schedule.10
  br label %direct.schedule.12

direct.true87:                                    ; preds = %direct.schedule.13
  br label %direct.schedule.14

direct.false88:                                   ; preds = %direct.schedule.13
  br label %direct.schedule.15

direct.true110:                                   ; preds = %direct.schedule.16
  br label %direct.schedule.17

direct.false111:                                  ; preds = %direct.schedule.16
  br label %direct.schedule.18
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.vector.reduce.or.v8i1(<8 x i1>) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr>, i32 immarg, <8 x i1>, <8 x float>) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.masked.scatter.v8f32.v8p0(<8 x float>, <8 x ptr>, i32 immarg, <8 x i1>) #2

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
