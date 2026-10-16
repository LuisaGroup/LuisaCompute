; ModuleID = 'luisa-simd-kernel'
source_filename = "luisa-simd-kernel"

define internal void @llm_rows(ptr noalias readonly %argument_buffer, ptr %return_lanes, ptr noalias nonnull readonly %launch_config, i32 %active_lane_count) {
prologue:
  %tile_snapshot.local = alloca [65536 x i8], align 4
  %.splatinsert = insertelement <8 x ptr> poison, ptr %tile_snapshot.local, i64 0
  %.splat = shufflevector <8 x ptr> %.splatinsert, <8 x ptr> poison, <8 x i32> zeroinitializer
  %0 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat, 0
  %1 = insertvalue { <8 x ptr>, <8 x i64> } %0, <8 x i64> <i64 0, i64 8192, i64 16384, i64 24576, i64 32768, i64 40960, i64 49152, i64 57344>, 1
  %tile_snapshot.local1 = alloca [65536 x i8], align 4
  %.splatinsert2 = insertelement <8 x ptr> poison, ptr %tile_snapshot.local1, i64 0
  %.splat3 = shufflevector <8 x ptr> %.splatinsert2, <8 x ptr> poison, <8 x i32> zeroinitializer
  %2 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat3, 0
  %3 = insertvalue { <8 x ptr>, <8 x i64> } %2, <8 x i64> <i64 0, i64 8192, i64 16384, i64 24576, i64 32768, i64 40960, i64 49152, i64 57344>, 1
  %tile_snapshot.local4 = alloca [65536 x i8], align 4
  %.splatinsert5 = insertelement <8 x ptr> poison, ptr %tile_snapshot.local4, i64 0
  %.splat6 = shufflevector <8 x ptr> %.splatinsert5, <8 x ptr> poison, <8 x i32> zeroinitializer
  %4 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat6, 0
  %5 = insertvalue { <8 x ptr>, <8 x i64> } %4, <8 x i64> <i64 0, i64 8192, i64 16384, i64 24576, i64 32768, i64 40960, i64 49152, i64 57344>, 1
  %tile_snapshot.local7 = alloca [65536 x i8], align 4
  %.splatinsert8 = insertelement <8 x ptr> poison, ptr %tile_snapshot.local7, i64 0
  %.splat9 = shufflevector <8 x ptr> %.splatinsert8, <8 x ptr> poison, <8 x i32> zeroinitializer
  %6 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat9, 0
  %7 = insertvalue { <8 x ptr>, <8 x i64> } %6, <8 x i64> <i64 0, i64 8192, i64 16384, i64 24576, i64 32768, i64 40960, i64 49152, i64 57344>, 1
  %.spill = alloca <8 x i64>, align 64
  store <8 x i64> zeroinitializer, ptr %.spill, align 64
  %tile_snapshot.spill = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill, align 64
  %.slot = alloca i64, align 8
  store i64 0, ptr %.slot, align 4
  %tile_snapshot.spill10 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill10, align 64
  %.slot11 = alloca i64, align 8
  store i64 0, ptr %.slot11, align 4
  %tile_snapshot.spill12 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill12, align 64
  %.slot13 = alloca i64, align 8
  store i64 0, ptr %.slot13, align 4
  %tile_snapshot.spill14 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill14, align 64
  %.slot15 = alloca i64, align 8
  store i64 0, ptr %.slot15, align 4
  %.slot16 = alloca i64, align 8
  store i64 0, ptr %.slot16, align 4
  %.slot17 = alloca i64, align 8
  store i64 0, ptr %.slot17, align 4
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
  %.splatinsert18 = insertelement <8 x i32> poison, i32 %44, i64 0
  %.splat19 = shufflevector <8 x i32> %.splatinsert18, <8 x i32> poison, <8 x i32> zeroinitializer
  %45 = add <8 x i32> %.splat19, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %46 = mul i32 %32, 32
  %.splatinsert20 = insertelement <8 x i32> poison, i32 %46, i64 0
  %.splat21 = shufflevector <8 x i32> %.splatinsert20, <8 x i32> poison, <8 x i32> zeroinitializer
  %47 = add <8 x i32> %.splat21, %45
  %48 = mul i32 %36, 1
  %.splatinsert22 = insertelement <8 x i32> poison, i32 %48, i64 0
  %.splat23 = shufflevector <8 x i32> %.splatinsert22, <8 x i32> poison, <8 x i32> zeroinitializer
  %49 = add <8 x i32> %.splat23, zeroinitializer
  %50 = mul i32 %40, 1
  %.splatinsert24 = insertelement <8 x i32> poison, i32 %50, i64 0
  %.splat25 = shufflevector <8 x i32> %.splatinsert24, <8 x i32> poison, <8 x i32> zeroinitializer
  %51 = add <8 x i32> %.splat25, zeroinitializer
  %52 = insertvalue [3 x <8 x i32>] poison, <8 x i32> %47, 0
  %53 = insertvalue [3 x <8 x i32>] %52, <8 x i32> %49, 1
  %54 = insertvalue [3 x <8 x i32>] %53, <8 x i32> %51, 2
  %.splatinsert26 = insertelement <8 x i32> poison, i32 %active_lane_count, i64 0
  %.splat27 = shufflevector <8 x i32> %.splatinsert26, <8 x i32> poison, <8 x i32> zeroinitializer
  %55 = icmp ult <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, %.splat27
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
  %76 = icmp slt i64 %.state, 2048
  br i1 %76, label %direct.true, label %direct.false

direct.schedule.2:                                ; preds = %direct.true
  %.state28 = load i64, ptr %.slot, align 4
  %77 = srem i64 %.state28, 2048
  %.state29 = load i64, ptr %.slot, align 4
  %78 = sdiv i64 %.state29, 2048
  %79 = srem i64 %78, 1
  %.spill.load = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert30 = insertelement <8 x i64> poison, i64 %79, i64 0
  %.splat31 = shufflevector <8 x i64> %.splatinsert30, <8 x i64> poison, <8 x i32> zeroinitializer
  %80 = add <8 x i64> %.spill.load, %.splat31
  %81 = add <8 x i64> zeroinitializer, %80
  %82 = add i64 0, %77
  %83 = mul <8 x i64> %81, splat (i64 4096)
  %.splatinsert32 = insertelement <8 x i64> poison, i64 %82, i64 0
  %.splat33 = shufflevector <8 x i64> %.splatinsert32, <8 x i64> poison, <8 x i32> zeroinitializer
  %84 = add <8 x i64> %83, %.splat33
  %85 = extractvalue { ptr, i64 } %13, 0
  %86 = mul <8 x i64> %84, splat (i64 4)
  %87 = getelementptr i8, ptr %85, <8 x i64> %86
  %88 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %87, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %89 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 0
  %90 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 1
  %.state34 = load i64, ptr %.slot, align 4
  %.splatinsert35 = insertelement <8 x i64> poison, i64 %.state34, i64 0
  %.splat36 = shufflevector <8 x i64> %.splatinsert35, <8 x i64> poison, <8 x i32> zeroinitializer
  %91 = mul <8 x i64> %.splat36, splat (i64 4)
  %92 = add <8 x i64> %90, %91
  %93 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %89, 0
  %94 = insertvalue { <8 x ptr>, <8 x i64> } %93, <8 x i64> %92, 1
  %95 = extractvalue { <8 x ptr>, <8 x i64> } %94, 0
  %96 = extractvalue { <8 x ptr>, <8 x i64> } %94, 1
  %97 = getelementptr i8, <8 x ptr> %95, <8 x i64> %96
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %88, <8 x ptr> %97, i32 1, <8 x i1> %55)
  %.state37 = load i64, ptr %.slot, align 4
  %98 = add i64 %.state37, 1
  store i64 %98, ptr %.slot, align 4
  br label %direct.schedule.1

direct.schedule.3:                                ; preds = %direct.false
  %99 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill10, align 64
  %100 = extractvalue { <8 x ptr>, <8 x i64> } %3, 0
  %101 = extractvalue { <8 x ptr>, <8 x i64> } %99, 0
  %102 = select <8 x i1> %55, <8 x ptr> %100, <8 x ptr> %101
  %103 = extractvalue { <8 x ptr>, <8 x i64> } %3, 1
  %104 = extractvalue { <8 x ptr>, <8 x i64> } %99, 1
  %105 = select <8 x i1> %55, <8 x i64> %103, <8 x i64> %104
  %106 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %102, 0
  %107 = insertvalue { <8 x ptr>, <8 x i64> } %106, <8 x i64> %105, 1
  store { <8 x ptr>, <8 x i64> } %107, ptr %tile_snapshot.spill10, align 64
  store i64 0, ptr %.slot11, align 4
  br label %direct.schedule.4

direct.schedule.4:                                ; preds = %direct.schedule.5, %direct.schedule.3
  %.state38 = load i64, ptr %.slot11, align 4
  %108 = icmp slt i64 %.state38, 2048
  br i1 %108, label %direct.true39, label %direct.false40

direct.schedule.5:                                ; preds = %direct.true39
  %.state41 = load i64, ptr %.slot11, align 4
  %109 = srem i64 %.state41, 2048
  %.state42 = load i64, ptr %.slot11, align 4
  %110 = sdiv i64 %.state42, 2048
  %111 = srem i64 %110, 1
  %.spill.load43 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert44 = insertelement <8 x i64> poison, i64 %111, i64 0
  %.splat45 = shufflevector <8 x i64> %.splatinsert44, <8 x i64> poison, <8 x i32> zeroinitializer
  %112 = add <8 x i64> %.spill.load43, %.splat45
  %113 = add <8 x i64> zeroinitializer, %112
  %114 = add i64 2048, %109
  %115 = mul <8 x i64> %113, splat (i64 4096)
  %.splatinsert46 = insertelement <8 x i64> poison, i64 %114, i64 0
  %.splat47 = shufflevector <8 x i64> %.splatinsert46, <8 x i64> poison, <8 x i32> zeroinitializer
  %116 = add <8 x i64> %115, %.splat47
  %117 = extractvalue { ptr, i64 } %13, 0
  %118 = mul <8 x i64> %116, splat (i64 4)
  %119 = getelementptr i8, ptr %117, <8 x i64> %118
  %120 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %119, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load48 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill10, align 64
  %121 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load48, 0
  %122 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load48, 1
  %.state49 = load i64, ptr %.slot11, align 4
  %.splatinsert50 = insertelement <8 x i64> poison, i64 %.state49, i64 0
  %.splat51 = shufflevector <8 x i64> %.splatinsert50, <8 x i64> poison, <8 x i32> zeroinitializer
  %123 = mul <8 x i64> %.splat51, splat (i64 4)
  %124 = add <8 x i64> %122, %123
  %125 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %121, 0
  %126 = insertvalue { <8 x ptr>, <8 x i64> } %125, <8 x i64> %124, 1
  %127 = extractvalue { <8 x ptr>, <8 x i64> } %126, 0
  %128 = extractvalue { <8 x ptr>, <8 x i64> } %126, 1
  %129 = getelementptr i8, <8 x ptr> %127, <8 x i64> %128
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %120, <8 x ptr> %129, i32 1, <8 x i1> %55)
  %.state52 = load i64, ptr %.slot11, align 4
  %130 = add i64 %.state52, 1
  store i64 %130, ptr %.slot11, align 4
  br label %direct.schedule.4

direct.schedule.6:                                ; preds = %direct.false40
  %131 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill12, align 64
  %132 = extractvalue { <8 x ptr>, <8 x i64> } %5, 0
  %133 = extractvalue { <8 x ptr>, <8 x i64> } %131, 0
  %134 = select <8 x i1> %55, <8 x ptr> %132, <8 x ptr> %133
  %135 = extractvalue { <8 x ptr>, <8 x i64> } %5, 1
  %136 = extractvalue { <8 x ptr>, <8 x i64> } %131, 1
  %137 = select <8 x i1> %55, <8 x i64> %135, <8 x i64> %136
  %138 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %134, 0
  %139 = insertvalue { <8 x ptr>, <8 x i64> } %138, <8 x i64> %137, 1
  store { <8 x ptr>, <8 x i64> } %139, ptr %tile_snapshot.spill12, align 64
  store i64 0, ptr %.slot13, align 4
  br label %direct.schedule.7

direct.schedule.7:                                ; preds = %direct.schedule.8, %direct.schedule.6
  %.state53 = load i64, ptr %.slot13, align 4
  %140 = icmp slt i64 %.state53, 2048
  br i1 %140, label %direct.true54, label %direct.false55

direct.schedule.8:                                ; preds = %direct.true54
  %.state56 = load i64, ptr %.slot13, align 4
  %141 = srem i64 %.state56, 2048
  %.state57 = load i64, ptr %.slot13, align 4
  %142 = sdiv i64 %.state57, 2048
  %143 = srem i64 %142, 1
  %.spill.load58 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert59 = insertelement <8 x i64> poison, i64 %143, i64 0
  %.splat60 = shufflevector <8 x i64> %.splatinsert59, <8 x i64> poison, <8 x i32> zeroinitializer
  %144 = add <8 x i64> %.spill.load58, %.splat60
  %145 = add <8 x i64> zeroinitializer, %144
  %146 = add i64 0, %141
  %147 = mul <8 x i64> %145, splat (i64 2048)
  %.splatinsert61 = insertelement <8 x i64> poison, i64 %146, i64 0
  %.splat62 = shufflevector <8 x i64> %.splatinsert61, <8 x i64> poison, <8 x i32> zeroinitializer
  %148 = add <8 x i64> %147, %.splat62
  %149 = extractvalue { ptr, i64 } %19, 0
  %150 = mul <8 x i64> %148, splat (i64 4)
  %151 = getelementptr i8, ptr %149, <8 x i64> %150
  %152 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %151, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load63 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill12, align 64
  %153 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load63, 0
  %154 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load63, 1
  %.state64 = load i64, ptr %.slot13, align 4
  %.splatinsert65 = insertelement <8 x i64> poison, i64 %.state64, i64 0
  %.splat66 = shufflevector <8 x i64> %.splatinsert65, <8 x i64> poison, <8 x i32> zeroinitializer
  %155 = mul <8 x i64> %.splat66, splat (i64 4)
  %156 = add <8 x i64> %154, %155
  %157 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %153, 0
  %158 = insertvalue { <8 x ptr>, <8 x i64> } %157, <8 x i64> %156, 1
  %159 = extractvalue { <8 x ptr>, <8 x i64> } %158, 0
  %160 = extractvalue { <8 x ptr>, <8 x i64> } %158, 1
  %161 = getelementptr i8, <8 x ptr> %159, <8 x i64> %160
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %152, <8 x ptr> %161, i32 1, <8 x i1> %55)
  %.state67 = load i64, ptr %.slot13, align 4
  %162 = add i64 %.state67, 1
  store i64 %162, ptr %.slot13, align 4
  br label %direct.schedule.7

direct.schedule.9:                                ; preds = %direct.false55
  %163 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill14, align 64
  %164 = extractvalue { <8 x ptr>, <8 x i64> } %7, 0
  %165 = extractvalue { <8 x ptr>, <8 x i64> } %163, 0
  %166 = select <8 x i1> %55, <8 x ptr> %164, <8 x ptr> %165
  %167 = extractvalue { <8 x ptr>, <8 x i64> } %7, 1
  %168 = extractvalue { <8 x ptr>, <8 x i64> } %163, 1
  %169 = select <8 x i1> %55, <8 x i64> %167, <8 x i64> %168
  %170 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %166, 0
  %171 = insertvalue { <8 x ptr>, <8 x i64> } %170, <8 x i64> %169, 1
  store { <8 x ptr>, <8 x i64> } %171, ptr %tile_snapshot.spill14, align 64
  store i64 0, ptr %.slot15, align 4
  br label %direct.schedule.10

direct.schedule.10:                               ; preds = %direct.schedule.11, %direct.schedule.9
  %.state68 = load i64, ptr %.slot15, align 4
  %172 = icmp slt i64 %.state68, 2048
  br i1 %172, label %direct.true69, label %direct.false70

direct.schedule.11:                               ; preds = %direct.true69
  %.state71 = load i64, ptr %.slot15, align 4
  %173 = srem i64 %.state71, 2048
  %.state72 = load i64, ptr %.slot15, align 4
  %174 = sdiv i64 %.state72, 2048
  %175 = srem i64 %174, 1
  %.spill.load73 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert74 = insertelement <8 x i64> poison, i64 %175, i64 0
  %.splat75 = shufflevector <8 x i64> %.splatinsert74, <8 x i64> poison, <8 x i32> zeroinitializer
  %176 = add <8 x i64> %.spill.load73, %.splat75
  %177 = add <8 x i64> zeroinitializer, %176
  %178 = add i64 0, %173
  %179 = mul <8 x i64> %177, splat (i64 2048)
  %.splatinsert76 = insertelement <8 x i64> poison, i64 %178, i64 0
  %.splat77 = shufflevector <8 x i64> %.splatinsert76, <8 x i64> poison, <8 x i32> zeroinitializer
  %180 = add <8 x i64> %179, %.splat77
  %181 = extractvalue { ptr, i64 } %25, 0
  %182 = mul <8 x i64> %180, splat (i64 4)
  %183 = getelementptr i8, ptr %181, <8 x i64> %182
  %184 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %183, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load78 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill14, align 64
  %185 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load78, 0
  %186 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load78, 1
  %.state79 = load i64, ptr %.slot15, align 4
  %.splatinsert80 = insertelement <8 x i64> poison, i64 %.state79, i64 0
  %.splat81 = shufflevector <8 x i64> %.splatinsert80, <8 x i64> poison, <8 x i32> zeroinitializer
  %187 = mul <8 x i64> %.splat81, splat (i64 4)
  %188 = add <8 x i64> %186, %187
  %189 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %185, 0
  %190 = insertvalue { <8 x ptr>, <8 x i64> } %189, <8 x i64> %188, 1
  %191 = extractvalue { <8 x ptr>, <8 x i64> } %190, 0
  %192 = extractvalue { <8 x ptr>, <8 x i64> } %190, 1
  %193 = getelementptr i8, <8 x ptr> %191, <8 x i64> %192
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %184, <8 x ptr> %193, i32 1, <8 x i1> %55)
  %.state82 = load i64, ptr %.slot15, align 4
  %194 = add i64 %.state82, 1
  store i64 %194, ptr %.slot15, align 4
  br label %direct.schedule.10

direct.schedule.12:                               ; preds = %direct.false70
  store i64 0, ptr %.slot16, align 4
  br label %direct.schedule.13

direct.schedule.13:                               ; preds = %direct.schedule.14, %direct.schedule.12
  %.state83 = load i64, ptr %.slot16, align 4
  %195 = icmp slt i64 %.state83, 2048
  br i1 %195, label %direct.true84, label %direct.false85

direct.schedule.14:                               ; preds = %direct.true84
  %.state86 = load i64, ptr %.slot16, align 4
  %196 = srem i64 %.state86, 2048
  %.state87 = load i64, ptr %.slot16, align 4
  %197 = sdiv i64 %.state87, 2048
  %198 = srem i64 %197, 1
  %.spill.load88 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert89 = insertelement <8 x i64> poison, i64 %198, i64 0
  %.splat90 = shufflevector <8 x i64> %.splatinsert89, <8 x i64> poison, <8 x i32> zeroinitializer
  %199 = add <8 x i64> %.spill.load88, %.splat90
  %200 = add <8 x i64> zeroinitializer, %199
  %201 = add i64 0, %196
  %202 = mul <8 x i64> %200, splat (i64 4096)
  %.splatinsert91 = insertelement <8 x i64> poison, i64 %201, i64 0
  %.splat92 = shufflevector <8 x i64> %.splatinsert91, <8 x i64> poison, <8 x i32> zeroinitializer
  %203 = add <8 x i64> %202, %.splat92
  %204 = add i64 0, %196
  %205 = srem i64 %204, 2048
  %206 = add i64 0, %205
  %tile_snapshot.spill.load93 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %207 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load93, 0
  %208 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load93, 1
  %.splatinsert94 = insertelement <8 x i64> poison, i64 %206, i64 0
  %.splat95 = shufflevector <8 x i64> %.splatinsert94, <8 x i64> poison, <8 x i32> zeroinitializer
  %209 = mul <8 x i64> %.splat95, splat (i64 4)
  %210 = add <8 x i64> %208, %209
  %211 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %207, 0
  %212 = insertvalue { <8 x ptr>, <8 x i64> } %211, <8 x i64> %210, 1
  %213 = extractvalue { <8 x ptr>, <8 x i64> } %212, 0
  %214 = extractvalue { <8 x ptr>, <8 x i64> } %212, 1
  %215 = getelementptr i8, <8 x ptr> %213, <8 x i64> %214
  %216 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %215, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load96 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill12, align 64
  %217 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load96, 0
  %218 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load96, 1
  %.splatinsert97 = insertelement <8 x i64> poison, i64 %206, i64 0
  %.splat98 = shufflevector <8 x i64> %.splatinsert97, <8 x i64> poison, <8 x i32> zeroinitializer
  %219 = mul <8 x i64> %.splat98, splat (i64 4)
  %220 = add <8 x i64> %218, %219
  %221 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %217, 0
  %222 = insertvalue { <8 x ptr>, <8 x i64> } %221, <8 x i64> %220, 1
  %223 = extractvalue { <8 x ptr>, <8 x i64> } %222, 0
  %224 = extractvalue { <8 x ptr>, <8 x i64> } %222, 1
  %225 = getelementptr i8, <8 x ptr> %223, <8 x i64> %224
  %226 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %225, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %227 = fmul <8 x float> %216, %226
  %tile_snapshot.spill.load99 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill10, align 64
  %228 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load99, 0
  %229 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load99, 1
  %.splatinsert100 = insertelement <8 x i64> poison, i64 %206, i64 0
  %.splat101 = shufflevector <8 x i64> %.splatinsert100, <8 x i64> poison, <8 x i32> zeroinitializer
  %230 = mul <8 x i64> %.splat101, splat (i64 4)
  %231 = add <8 x i64> %229, %230
  %232 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %228, 0
  %233 = insertvalue { <8 x ptr>, <8 x i64> } %232, <8 x i64> %231, 1
  %234 = extractvalue { <8 x ptr>, <8 x i64> } %233, 0
  %235 = extractvalue { <8 x ptr>, <8 x i64> } %233, 1
  %236 = getelementptr i8, <8 x ptr> %234, <8 x i64> %235
  %237 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %236, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load102 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill14, align 64
  %238 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load102, 0
  %239 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load102, 1
  %.splatinsert103 = insertelement <8 x i64> poison, i64 %206, i64 0
  %.splat104 = shufflevector <8 x i64> %.splatinsert103, <8 x i64> poison, <8 x i32> zeroinitializer
  %240 = mul <8 x i64> %.splat104, splat (i64 4)
  %241 = add <8 x i64> %239, %240
  %242 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %238, 0
  %243 = insertvalue { <8 x ptr>, <8 x i64> } %242, <8 x i64> %241, 1
  %244 = extractvalue { <8 x ptr>, <8 x i64> } %243, 0
  %245 = extractvalue { <8 x ptr>, <8 x i64> } %243, 1
  %246 = getelementptr i8, <8 x ptr> %244, <8 x i64> %245
  %247 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %246, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %248 = fmul <8 x float> %237, %247
  %249 = fsub <8 x float> %227, %248
  %250 = extractvalue { ptr, i64 } %31, 0
  %251 = mul <8 x i64> %203, splat (i64 4)
  %252 = getelementptr i8, ptr %250, <8 x i64> %251
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %249, <8 x ptr> %252, i32 1, <8 x i1> %55)
  %.state105 = load i64, ptr %.slot16, align 4
  %253 = add i64 %.state105, 1
  store i64 %253, ptr %.slot16, align 4
  br label %direct.schedule.13

direct.schedule.15:                               ; preds = %direct.false85
  store i64 0, ptr %.slot17, align 4
  br label %direct.schedule.16

direct.schedule.16:                               ; preds = %direct.schedule.17, %direct.schedule.15
  %.state106 = load i64, ptr %.slot17, align 4
  %254 = icmp slt i64 %.state106, 2048
  br i1 %254, label %direct.true107, label %direct.false108

direct.schedule.17:                               ; preds = %direct.true107
  %.state109 = load i64, ptr %.slot17, align 4
  %255 = srem i64 %.state109, 2048
  %.state110 = load i64, ptr %.slot17, align 4
  %256 = sdiv i64 %.state110, 2048
  %257 = srem i64 %256, 1
  %.spill.load111 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert112 = insertelement <8 x i64> poison, i64 %257, i64 0
  %.splat113 = shufflevector <8 x i64> %.splatinsert112, <8 x i64> poison, <8 x i32> zeroinitializer
  %258 = add <8 x i64> %.spill.load111, %.splat113
  %259 = add <8 x i64> zeroinitializer, %258
  %260 = add i64 2048, %255
  %261 = mul <8 x i64> %259, splat (i64 4096)
  %.splatinsert114 = insertelement <8 x i64> poison, i64 %260, i64 0
  %.splat115 = shufflevector <8 x i64> %.splatinsert114, <8 x i64> poison, <8 x i32> zeroinitializer
  %262 = add <8 x i64> %261, %.splat115
  %263 = add i64 0, %255
  %264 = srem i64 %263, 2048
  %265 = add i64 0, %264
  %tile_snapshot.spill.load116 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %266 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load116, 0
  %267 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load116, 1
  %.splatinsert117 = insertelement <8 x i64> poison, i64 %265, i64 0
  %.splat118 = shufflevector <8 x i64> %.splatinsert117, <8 x i64> poison, <8 x i32> zeroinitializer
  %268 = mul <8 x i64> %.splat118, splat (i64 4)
  %269 = add <8 x i64> %267, %268
  %270 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %266, 0
  %271 = insertvalue { <8 x ptr>, <8 x i64> } %270, <8 x i64> %269, 1
  %272 = extractvalue { <8 x ptr>, <8 x i64> } %271, 0
  %273 = extractvalue { <8 x ptr>, <8 x i64> } %271, 1
  %274 = getelementptr i8, <8 x ptr> %272, <8 x i64> %273
  %275 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %274, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load119 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill14, align 64
  %276 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load119, 0
  %277 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load119, 1
  %.splatinsert120 = insertelement <8 x i64> poison, i64 %265, i64 0
  %.splat121 = shufflevector <8 x i64> %.splatinsert120, <8 x i64> poison, <8 x i32> zeroinitializer
  %278 = mul <8 x i64> %.splat121, splat (i64 4)
  %279 = add <8 x i64> %277, %278
  %280 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %276, 0
  %281 = insertvalue { <8 x ptr>, <8 x i64> } %280, <8 x i64> %279, 1
  %282 = extractvalue { <8 x ptr>, <8 x i64> } %281, 0
  %283 = extractvalue { <8 x ptr>, <8 x i64> } %281, 1
  %284 = getelementptr i8, <8 x ptr> %282, <8 x i64> %283
  %285 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %284, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %286 = fmul <8 x float> %275, %285
  %tile_snapshot.spill.load122 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill10, align 64
  %287 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load122, 0
  %288 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load122, 1
  %.splatinsert123 = insertelement <8 x i64> poison, i64 %265, i64 0
  %.splat124 = shufflevector <8 x i64> %.splatinsert123, <8 x i64> poison, <8 x i32> zeroinitializer
  %289 = mul <8 x i64> %.splat124, splat (i64 4)
  %290 = add <8 x i64> %288, %289
  %291 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %287, 0
  %292 = insertvalue { <8 x ptr>, <8 x i64> } %291, <8 x i64> %290, 1
  %293 = extractvalue { <8 x ptr>, <8 x i64> } %292, 0
  %294 = extractvalue { <8 x ptr>, <8 x i64> } %292, 1
  %295 = getelementptr i8, <8 x ptr> %293, <8 x i64> %294
  %296 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %295, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load125 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill12, align 64
  %297 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load125, 0
  %298 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load125, 1
  %.splatinsert126 = insertelement <8 x i64> poison, i64 %265, i64 0
  %.splat127 = shufflevector <8 x i64> %.splatinsert126, <8 x i64> poison, <8 x i32> zeroinitializer
  %299 = mul <8 x i64> %.splat127, splat (i64 4)
  %300 = add <8 x i64> %298, %299
  %301 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %297, 0
  %302 = insertvalue { <8 x ptr>, <8 x i64> } %301, <8 x i64> %300, 1
  %303 = extractvalue { <8 x ptr>, <8 x i64> } %302, 0
  %304 = extractvalue { <8 x ptr>, <8 x i64> } %302, 1
  %305 = getelementptr i8, <8 x ptr> %303, <8 x i64> %304
  %306 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %305, i32 1, <8 x i1> %55, <8 x float> zeroinitializer)
  %307 = fmul <8 x float> %296, %306
  %308 = fadd <8 x float> %286, %307
  %309 = extractvalue { ptr, i64 } %31, 0
  %310 = mul <8 x i64> %262, splat (i64 4)
  %311 = getelementptr i8, ptr %309, <8 x i64> %310
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %308, <8 x ptr> %311, i32 1, <8 x i1> %55)
  %.state128 = load i64, ptr %.slot17, align 4
  %312 = add i64 %.state128, 1
  store i64 %312, ptr %.slot17, align 4
  br label %direct.schedule.16

direct.schedule.18:                               ; preds = %direct.false108
  ret void

direct.activate:                                  ; preds = %prologue
  br label %direct.schedule.0

direct.inactive:                                  ; preds = %prologue
  ret void

direct.true:                                      ; preds = %direct.schedule.1
  br label %direct.schedule.2

direct.false:                                     ; preds = %direct.schedule.1
  br label %direct.schedule.3

direct.true39:                                    ; preds = %direct.schedule.4
  br label %direct.schedule.5

direct.false40:                                   ; preds = %direct.schedule.4
  br label %direct.schedule.6

direct.true54:                                    ; preds = %direct.schedule.7
  br label %direct.schedule.8

direct.false55:                                   ; preds = %direct.schedule.7
  br label %direct.schedule.9

direct.true69:                                    ; preds = %direct.schedule.10
  br label %direct.schedule.11

direct.false70:                                   ; preds = %direct.schedule.10
  br label %direct.schedule.12

direct.true84:                                    ; preds = %direct.schedule.13
  br label %direct.schedule.14

direct.false85:                                   ; preds = %direct.schedule.13
  br label %direct.schedule.15

direct.true107:                                   ; preds = %direct.schedule.16
  br label %direct.schedule.17

direct.false108:                                  ; preds = %direct.schedule.16
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
