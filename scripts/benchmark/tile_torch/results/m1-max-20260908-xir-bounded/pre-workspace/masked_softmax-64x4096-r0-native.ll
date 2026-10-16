; ModuleID = 'luisa-simd-kernel'
source_filename = "luisa-simd-kernel"

define internal void @llm_rows(ptr noalias readonly %argument_buffer, ptr %return_lanes, ptr noalias nonnull readonly %launch_config, i32 %active_lane_count) {
prologue:
  %tile_snapshot.local = alloca [131072 x i8], align 4
  %.splatinsert = insertelement <8 x ptr> poison, ptr %tile_snapshot.local, i64 0
  %.splat = shufflevector <8 x ptr> %.splatinsert, <8 x ptr> poison, <8 x i32> zeroinitializer
  %0 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat, 0
  %1 = insertvalue { <8 x ptr>, <8 x i64> } %0, <8 x i64> <i64 0, i64 16384, i64 32768, i64 49152, i64 65536, i64 81920, i64 98304, i64 114688>, 1
  %tile_snapshot.local1 = alloca [262144 x i8], align 8
  %.splatinsert2 = insertelement <8 x ptr> poison, ptr %tile_snapshot.local1, i64 0
  %.splat3 = shufflevector <8 x ptr> %.splatinsert2, <8 x ptr> poison, <8 x i32> zeroinitializer
  %2 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat3, 0
  %3 = insertvalue { <8 x ptr>, <8 x i64> } %2, <8 x i64> <i64 0, i64 32768, i64 65536, i64 98304, i64 131072, i64 163840, i64 196608, i64 229376>, 1
  %tile_snapshot.local4 = alloca [32768 x i8], align 1
  %.splatinsert5 = insertelement <8 x ptr> poison, ptr %tile_snapshot.local4, i64 0
  %.splat6 = shufflevector <8 x ptr> %.splatinsert5, <8 x ptr> poison, <8 x i32> zeroinitializer
  %4 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat6, 0
  %5 = insertvalue { <8 x ptr>, <8 x i64> } %4, <8 x i64> <i64 0, i64 4096, i64 8192, i64 12288, i64 16384, i64 20480, i64 24576, i64 28672>, 1
  %tile_snapshot.local7 = alloca [131072 x i8], align 4
  %.splatinsert8 = insertelement <8 x ptr> poison, ptr %tile_snapshot.local7, i64 0
  %.splat9 = shufflevector <8 x ptr> %.splatinsert8, <8 x ptr> poison, <8 x i32> zeroinitializer
  %6 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat9, 0
  %7 = insertvalue { <8 x ptr>, <8 x i64> } %6, <8 x i64> <i64 0, i64 16384, i64 32768, i64 49152, i64 65536, i64 81920, i64 98304, i64 114688>, 1
  %tile_snapshot.local10 = alloca [131072 x i8], align 4
  %.splatinsert11 = insertelement <8 x ptr> poison, ptr %tile_snapshot.local10, i64 0
  %.splat12 = shufflevector <8 x ptr> %.splatinsert11, <8 x ptr> poison, <8 x i32> zeroinitializer
  %8 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat12, 0
  %9 = insertvalue { <8 x ptr>, <8 x i64> } %8, <8 x i64> <i64 0, i64 16384, i64 32768, i64 49152, i64 65536, i64 81920, i64 98304, i64 114688>, 1
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
  %.spill15 = alloca <8 x i64>, align 64
  store <8 x i64> zeroinitializer, ptr %.spill15, align 64
  %tile_snapshot.spill16 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill16, align 64
  %.slot17 = alloca i64, align 8
  store i64 0, ptr %.slot17, align 4
  %.spill18 = alloca float, align 4
  store float 0.000000e+00, ptr %.spill18, align 4
  %tile_snapshot.spill19 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill19, align 64
  %.slot20 = alloca i64, align 8
  store i64 0, ptr %.slot20, align 4
  %.spill21 = alloca float, align 4
  store float 0.000000e+00, ptr %.spill21, align 4
  %.spill22 = alloca i64, align 8
  store i64 0, ptr %.spill22, align 4
  %.spill23 = alloca i64, align 8
  store i64 0, ptr %.spill23, align 4
  %.spill24 = alloca i1, align 1
  store i1 false, ptr %.spill24, align 1
  %.slot25 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot25, align 32
  %.spill26 = alloca i64, align 8
  store i64 0, ptr %.spill26, align 4
  %.spill27 = alloca i1, align 1
  store i1 false, ptr %.spill27, align 1
  %.slot28 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot28, align 32
  %.spill29 = alloca i64, align 8
  store i64 0, ptr %.spill29, align 4
  %.spill30 = alloca i1, align 1
  store i1 false, ptr %.spill30, align 1
  %.slot31 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot31, align 32
  %.spill32 = alloca i64, align 8
  store i64 0, ptr %.spill32, align 4
  %.spill33 = alloca i1, align 1
  store i1 false, ptr %.spill33, align 1
  %.slot34 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot34, align 32
  %.slot35 = alloca i64, align 8
  store i64 0, ptr %.slot35, align 4
  %.slot36 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot36, align 32
  %.slot37 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot37, align 32
  %.slot38 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot38, align 32
  %.slot39 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot39, align 32
  %.spill40 = alloca i64, align 8
  store i64 0, ptr %.spill40, align 4
  %.slot41 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot41, align 32
  %.spill42 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill42, align 32
  %.spill43 = alloca i64, align 8
  store i64 0, ptr %.spill43, align 4
  %.slot44 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot44, align 32
  %.spill45 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill45, align 32
  %.spill46 = alloca i64, align 8
  store i64 0, ptr %.spill46, align 4
  %.slot47 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot47, align 32
  %.spill48 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill48, align 32
  %.spill49 = alloca i64, align 8
  store i64 0, ptr %.spill49, align 4
  %.slot50 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot50, align 32
  %.spill51 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill51, align 32
  %.spill52 = alloca float, align 4
  store float 0.000000e+00, ptr %.spill52, align 4
  %tile_snapshot.spill53 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill53, align 64
  %.slot54 = alloca i64, align 8
  store i64 0, ptr %.slot54, align 4
  %.slot55 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot55, align 32
  %.slot56 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot56, align 32
  %.slot57 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot57, align 32
  %.slot58 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot58, align 32
  %.slot59 = alloca i64, align 8
  store i64 0, ptr %.slot59, align 4
  %.slot60 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot60, align 32
  %.slot61 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot61, align 32
  %.slot62 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot62, align 32
  %.slot63 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot63, align 32
  %.spill64 = alloca i64, align 8
  store i64 0, ptr %.spill64, align 4
  %.slot65 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot65, align 32
  %.spill66 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill66, align 32
  %.spill67 = alloca i64, align 8
  store i64 0, ptr %.spill67, align 4
  %.slot68 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot68, align 32
  %.spill69 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill69, align 32
  %.spill70 = alloca i64, align 8
  store i64 0, ptr %.spill70, align 4
  %.slot71 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot71, align 32
  %.spill72 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill72, align 32
  %.spill73 = alloca i64, align 8
  store i64 0, ptr %.spill73, align 4
  %.slot74 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot74, align 32
  %.spill75 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill75, align 32
  %.slot76 = alloca i64, align 8
  store i64 0, ptr %.slot76, align 4
  %10 = getelementptr i8, ptr %argument_buffer, i64 0
  %11 = load ptr, ptr %10, align 16
  %12 = getelementptr i8, ptr %10, i64 8
  %13 = load i64, ptr %12, align 8
  %14 = insertvalue { ptr, i64 } poison, ptr %11, 0
  %15 = insertvalue { ptr, i64 } %14, i64 %13, 1
  %16 = getelementptr i8, ptr %argument_buffer, i64 16
  %17 = load ptr, ptr %16, align 16
  %18 = getelementptr i8, ptr %16, i64 8
  %19 = load i64, ptr %18, align 8
  %20 = insertvalue { ptr, i64 } poison, ptr %17, 0
  %21 = insertvalue { ptr, i64 } %20, i64 %19, 1
  %22 = getelementptr i8, ptr %argument_buffer, i64 32
  %23 = load ptr, ptr %22, align 16
  %24 = getelementptr i8, ptr %22, i64 8
  %25 = load i64, ptr %24, align 8
  %26 = insertvalue { ptr, i64 } poison, ptr %23, 0
  %27 = insertvalue { ptr, i64 } %26, i64 %25, 1
  %28 = getelementptr i8, ptr %argument_buffer, i64 48
  %29 = load ptr, ptr %28, align 16
  %30 = getelementptr i8, ptr %28, i64 8
  %31 = load i64, ptr %30, align 8
  %32 = insertvalue { ptr, i64 } poison, ptr %29, 0
  %33 = insertvalue { ptr, i64 } %32, i64 %31, 1
  %34 = load i32, ptr %launch_config, align 4
  %35 = getelementptr i8, ptr %launch_config, i64 12
  %36 = load i32, ptr %35, align 4
  %37 = getelementptr i8, ptr %launch_config, i64 4
  %38 = load i32, ptr %37, align 4
  %39 = getelementptr i8, ptr %launch_config, i64 16
  %40 = load i32, ptr %39, align 4
  %41 = getelementptr i8, ptr %launch_config, i64 8
  %42 = load i32, ptr %41, align 4
  %43 = getelementptr i8, ptr %launch_config, i64 20
  %44 = load i32, ptr %43, align 4
  %45 = getelementptr i8, ptr %launch_config, i64 36
  %46 = load i32, ptr %45, align 4
  %.splatinsert77 = insertelement <8 x i32> poison, i32 %46, i64 0
  %.splat78 = shufflevector <8 x i32> %.splatinsert77, <8 x i32> poison, <8 x i32> zeroinitializer
  %47 = add <8 x i32> %.splat78, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %48 = mul i32 %34, 32
  %.splatinsert79 = insertelement <8 x i32> poison, i32 %48, i64 0
  %.splat80 = shufflevector <8 x i32> %.splatinsert79, <8 x i32> poison, <8 x i32> zeroinitializer
  %49 = add <8 x i32> %.splat80, %47
  %50 = mul i32 %38, 1
  %.splatinsert81 = insertelement <8 x i32> poison, i32 %50, i64 0
  %.splat82 = shufflevector <8 x i32> %.splatinsert81, <8 x i32> poison, <8 x i32> zeroinitializer
  %51 = add <8 x i32> %.splat82, zeroinitializer
  %52 = mul i32 %42, 1
  %.splatinsert83 = insertelement <8 x i32> poison, i32 %52, i64 0
  %.splat84 = shufflevector <8 x i32> %.splatinsert83, <8 x i32> poison, <8 x i32> zeroinitializer
  %53 = add <8 x i32> %.splat84, zeroinitializer
  %54 = insertvalue [3 x <8 x i32>] poison, <8 x i32> %49, 0
  %55 = insertvalue [3 x <8 x i32>] %54, <8 x i32> %51, 1
  %56 = insertvalue [3 x <8 x i32>] %55, <8 x i32> %53, 2
  %.splatinsert85 = insertelement <8 x i32> poison, i32 %active_lane_count, i64 0
  %.splat86 = shufflevector <8 x i32> %.splatinsert85, <8 x i32> poison, <8 x i32> zeroinitializer
  %57 = icmp ult <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, %.splat86
  %58 = call i1 @llvm.vector.reduce.or.v8i1(<8 x i1> %57)
  br i1 %58, label %direct.activate, label %direct.inactive

direct.schedule.0:                                ; preds = %direct.activate
  %59 = extractvalue [3 x <8 x i32>] %56, 0
  %60 = zext <8 x i32> %59 to <8 x i64>
  %61 = select <8 x i1> %57, <8 x i64> %60, <8 x i64> zeroinitializer
  %62 = select <8 x i1> %57, <8 x i64> splat (i64 1), <8 x i64> splat (i64 1)
  %63 = sdiv <8 x i64> %61, %62
  %64 = select <8 x i1> %57, <8 x i64> %63, <8 x i64> zeroinitializer
  %65 = select <8 x i1> %57, <8 x i64> splat (i64 64), <8 x i64> splat (i64 1)
  %66 = srem <8 x i64> %64, %65
  %67 = load <8 x i64>, ptr %.spill, align 64
  %68 = select <8 x i1> %57, <8 x i64> %66, <8 x i64> %67
  store <8 x i64> %68, ptr %.spill, align 64
  %69 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %70 = extractvalue { <8 x ptr>, <8 x i64> } %1, 0
  %71 = extractvalue { <8 x ptr>, <8 x i64> } %69, 0
  %72 = select <8 x i1> %57, <8 x ptr> %70, <8 x ptr> %71
  %73 = extractvalue { <8 x ptr>, <8 x i64> } %1, 1
  %74 = extractvalue { <8 x ptr>, <8 x i64> } %69, 1
  %75 = select <8 x i1> %57, <8 x i64> %73, <8 x i64> %74
  %76 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %72, 0
  %77 = insertvalue { <8 x ptr>, <8 x i64> } %76, <8 x i64> %75, 1
  store { <8 x ptr>, <8 x i64> } %77, ptr %tile_snapshot.spill, align 64
  store i64 0, ptr %.slot, align 4
  br label %direct.schedule.1

direct.schedule.1:                                ; preds = %direct.schedule.2, %direct.schedule.0
  %.state = load i64, ptr %.slot, align 4
  %78 = icmp slt i64 %.state, 4096
  br i1 %78, label %direct.true, label %direct.false

direct.schedule.2:                                ; preds = %direct.true
  %.state87 = load i64, ptr %.slot, align 4
  %79 = srem i64 %.state87, 4096
  %.state88 = load i64, ptr %.slot, align 4
  %80 = sdiv i64 %.state88, 4096
  %81 = srem i64 %80, 1
  %.spill.load = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert89 = insertelement <8 x i64> poison, i64 %81, i64 0
  %.splat90 = shufflevector <8 x i64> %.splatinsert89, <8 x i64> poison, <8 x i32> zeroinitializer
  %82 = add <8 x i64> %.spill.load, %.splat90
  %83 = add <8 x i64> zeroinitializer, %82
  %84 = add i64 0, %79
  %85 = mul <8 x i64> %83, splat (i64 4096)
  %.splatinsert91 = insertelement <8 x i64> poison, i64 %84, i64 0
  %.splat92 = shufflevector <8 x i64> %.splatinsert91, <8 x i64> poison, <8 x i32> zeroinitializer
  %86 = add <8 x i64> %85, %.splat92
  %87 = extractvalue { ptr, i64 } %15, 0
  %88 = mul <8 x i64> %86, splat (i64 4)
  %89 = getelementptr i8, ptr %87, <8 x i64> %88
  %90 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %89, i32 1, <8 x i1> %57, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %91 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 0
  %92 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 1
  %.state93 = load i64, ptr %.slot, align 4
  %.splatinsert94 = insertelement <8 x i64> poison, i64 %.state93, i64 0
  %.splat95 = shufflevector <8 x i64> %.splatinsert94, <8 x i64> poison, <8 x i32> zeroinitializer
  %93 = mul <8 x i64> %.splat95, splat (i64 4)
  %94 = add <8 x i64> %92, %93
  %95 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %91, 0
  %96 = insertvalue { <8 x ptr>, <8 x i64> } %95, <8 x i64> %94, 1
  %97 = extractvalue { <8 x ptr>, <8 x i64> } %96, 0
  %98 = extractvalue { <8 x ptr>, <8 x i64> } %96, 1
  %99 = getelementptr i8, <8 x ptr> %97, <8 x i64> %98
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %90, <8 x ptr> %99, i32 1, <8 x i1> %57)
  %.state96 = load i64, ptr %.slot, align 4
  %100 = add i64 %.state96, 1
  store i64 %100, ptr %.slot, align 4
  br label %direct.schedule.1

direct.schedule.3:                                ; preds = %direct.false
  %101 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill13, align 64
  %102 = extractvalue { <8 x ptr>, <8 x i64> } %3, 0
  %103 = extractvalue { <8 x ptr>, <8 x i64> } %101, 0
  %104 = select <8 x i1> %57, <8 x ptr> %102, <8 x ptr> %103
  %105 = extractvalue { <8 x ptr>, <8 x i64> } %3, 1
  %106 = extractvalue { <8 x ptr>, <8 x i64> } %101, 1
  %107 = select <8 x i1> %57, <8 x i64> %105, <8 x i64> %106
  %108 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %104, 0
  %109 = insertvalue { <8 x ptr>, <8 x i64> } %108, <8 x i64> %107, 1
  store { <8 x ptr>, <8 x i64> } %109, ptr %tile_snapshot.spill13, align 64
  store i64 0, ptr %.slot14, align 4
  br label %direct.schedule.4

direct.schedule.4:                                ; preds = %direct.schedule.5, %direct.schedule.3
  %.state97 = load i64, ptr %.slot14, align 4
  %110 = icmp slt i64 %.state97, 4096
  br i1 %110, label %direct.true98, label %direct.false99

direct.schedule.5:                                ; preds = %direct.true98
  %.state100 = load i64, ptr %.slot14, align 4
  %111 = srem i64 %.state100, 4096
  %tile_snapshot.spill.load101 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill13, align 64
  %112 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load101, 0
  %113 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load101, 1
  %.state102 = load i64, ptr %.slot14, align 4
  %.splatinsert103 = insertelement <8 x i64> poison, i64 %.state102, i64 0
  %.splat104 = shufflevector <8 x i64> %.splatinsert103, <8 x i64> poison, <8 x i32> zeroinitializer
  %114 = mul <8 x i64> %.splat104, splat (i64 8)
  %115 = add <8 x i64> %113, %114
  %116 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %112, 0
  %117 = insertvalue { <8 x ptr>, <8 x i64> } %116, <8 x i64> %115, 1
  %.splatinsert105 = insertelement <8 x i64> poison, i64 %111, i64 0
  %.splat106 = shufflevector <8 x i64> %.splatinsert105, <8 x i64> poison, <8 x i32> zeroinitializer
  %118 = extractvalue { <8 x ptr>, <8 x i64> } %117, 0
  %119 = extractvalue { <8 x ptr>, <8 x i64> } %117, 1
  %120 = getelementptr i8, <8 x ptr> %118, <8 x i64> %119
  call void @llvm.masked.scatter.v8i64.v8p0(<8 x i64> %.splat106, <8 x ptr> %120, i32 1, <8 x i1> %57)
  %.state107 = load i64, ptr %.slot14, align 4
  %121 = add i64 %.state107, 1
  store i64 %121, ptr %.slot14, align 4
  br label %direct.schedule.4

direct.schedule.6:                                ; preds = %direct.false99
  %.spill.load108 = load <8 x i64>, ptr %.spill, align 64
  %122 = select <8 x i1> %57, <8 x i64> %.spill.load108, <8 x i64> zeroinitializer
  %123 = select <8 x i1> %57, <8 x i64> splat (i64 4096), <8 x i64> splat (i64 1)
  %124 = srem <8 x i64> %122, %123
  %125 = load <8 x i64>, ptr %.spill15, align 64
  %126 = select <8 x i1> %57, <8 x i64> %124, <8 x i64> %125
  store <8 x i64> %126, ptr %.spill15, align 64
  %127 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill16, align 64
  %128 = extractvalue { <8 x ptr>, <8 x i64> } %5, 0
  %129 = extractvalue { <8 x ptr>, <8 x i64> } %127, 0
  %130 = select <8 x i1> %57, <8 x ptr> %128, <8 x ptr> %129
  %131 = extractvalue { <8 x ptr>, <8 x i64> } %5, 1
  %132 = extractvalue { <8 x ptr>, <8 x i64> } %127, 1
  %133 = select <8 x i1> %57, <8 x i64> %131, <8 x i64> %132
  %134 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %130, 0
  %135 = insertvalue { <8 x ptr>, <8 x i64> } %134, <8 x i64> %133, 1
  store { <8 x ptr>, <8 x i64> } %135, ptr %tile_snapshot.spill16, align 64
  store i64 0, ptr %.slot17, align 4
  br label %direct.schedule.7

direct.schedule.7:                                ; preds = %direct.schedule.8, %direct.schedule.6
  %.state109 = load i64, ptr %.slot17, align 4
  %136 = icmp slt i64 %.state109, 4096
  br i1 %136, label %direct.true110, label %direct.false111

direct.schedule.8:                                ; preds = %direct.true110
  %.state112 = load i64, ptr %.slot17, align 4
  %137 = srem i64 %.state112, 4096
  %138 = add i64 0, %137
  %tile_snapshot.spill.load113 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill13, align 64
  %139 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load113, 0
  %140 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load113, 1
  %.splatinsert114 = insertelement <8 x i64> poison, i64 %138, i64 0
  %.splat115 = shufflevector <8 x i64> %.splatinsert114, <8 x i64> poison, <8 x i32> zeroinitializer
  %141 = mul <8 x i64> %.splat115, splat (i64 8)
  %142 = add <8 x i64> %140, %141
  %143 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %139, 0
  %144 = insertvalue { <8 x ptr>, <8 x i64> } %143, <8 x i64> %142, 1
  %145 = extractvalue { <8 x ptr>, <8 x i64> } %144, 0
  %146 = extractvalue { <8 x ptr>, <8 x i64> } %144, 1
  %147 = getelementptr i8, <8 x ptr> %145, <8 x i64> %146
  %148 = call <8 x i64> @llvm.masked.gather.v8i64.v8p0(<8 x ptr> %147, i32 1, <8 x i1> %57, <8 x i64> zeroinitializer)
  %.spill.load116 = load <8 x i64>, ptr %.spill15, align 64
  %149 = icmp sle <8 x i64> %148, %.spill.load116
  %tile_snapshot.spill.load117 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill16, align 64
  %150 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load117, 0
  %151 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load117, 1
  %.state118 = load i64, ptr %.slot17, align 4
  %.splatinsert119 = insertelement <8 x i64> poison, i64 %.state118, i64 0
  %.splat120 = shufflevector <8 x i64> %.splatinsert119, <8 x i64> poison, <8 x i32> zeroinitializer
  %152 = add <8 x i64> %151, %.splat120
  %153 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %150, 0
  %154 = insertvalue { <8 x ptr>, <8 x i64> } %153, <8 x i64> %152, 1
  %155 = extractvalue { <8 x ptr>, <8 x i64> } %154, 0
  %156 = extractvalue { <8 x ptr>, <8 x i64> } %154, 1
  %157 = getelementptr i8, <8 x ptr> %155, <8 x i64> %156
  %158 = zext <8 x i1> %149 to <8 x i8>
  call void @llvm.masked.scatter.v8i8.v8p0(<8 x i8> %158, <8 x ptr> %157, i32 1, <8 x i1> %57)
  %.state121 = load i64, ptr %.slot17, align 4
  %159 = add i64 %.state121, 1
  store i64 %159, ptr %.slot17, align 4
  br label %direct.schedule.7

direct.schedule.9:                                ; preds = %direct.false111
  store float 0xC6293E5940000000, ptr %.spill18, align 4
  %160 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill19, align 64
  %161 = extractvalue { <8 x ptr>, <8 x i64> } %7, 0
  %162 = extractvalue { <8 x ptr>, <8 x i64> } %160, 0
  %163 = select <8 x i1> %57, <8 x ptr> %161, <8 x ptr> %162
  %164 = extractvalue { <8 x ptr>, <8 x i64> } %7, 1
  %165 = extractvalue { <8 x ptr>, <8 x i64> } %160, 1
  %166 = select <8 x i1> %57, <8 x i64> %164, <8 x i64> %165
  %167 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %163, 0
  %168 = insertvalue { <8 x ptr>, <8 x i64> } %167, <8 x i64> %166, 1
  store { <8 x ptr>, <8 x i64> } %168, ptr %tile_snapshot.spill19, align 64
  store i64 0, ptr %.slot20, align 4
  br label %direct.schedule.10

direct.schedule.10:                               ; preds = %direct.schedule.11, %direct.schedule.9
  %.state122 = load i64, ptr %.slot20, align 4
  %169 = icmp slt i64 %.state122, 4096
  br i1 %169, label %direct.true123, label %direct.false124

direct.schedule.11:                               ; preds = %direct.true123
  %.state125 = load i64, ptr %.slot20, align 4
  %170 = srem i64 %.state125, 4096
  %171 = add i64 0, %170
  %tile_snapshot.spill.load126 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill16, align 64
  %172 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load126, 0
  %173 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load126, 1
  %.splatinsert127 = insertelement <8 x i64> poison, i64 %171, i64 0
  %.splat128 = shufflevector <8 x i64> %.splatinsert127, <8 x i64> poison, <8 x i32> zeroinitializer
  %174 = add <8 x i64> %173, %.splat128
  %175 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %172, 0
  %176 = insertvalue { <8 x ptr>, <8 x i64> } %175, <8 x i64> %174, 1
  %177 = extractvalue { <8 x ptr>, <8 x i64> } %176, 0
  %178 = extractvalue { <8 x ptr>, <8 x i64> } %176, 1
  %179 = getelementptr i8, <8 x ptr> %177, <8 x i64> %178
  %180 = call <8 x i8> @llvm.masked.gather.v8i8.v8p0(<8 x ptr> %179, i32 1, <8 x i1> %57, <8 x i8> zeroinitializer)
  %181 = icmp ne <8 x i8> %180, zeroinitializer
  %tile_snapshot.spill.load129 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %182 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load129, 0
  %183 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load129, 1
  %.splatinsert130 = insertelement <8 x i64> poison, i64 %171, i64 0
  %.splat131 = shufflevector <8 x i64> %.splatinsert130, <8 x i64> poison, <8 x i32> zeroinitializer
  %184 = mul <8 x i64> %.splat131, splat (i64 4)
  %185 = add <8 x i64> %183, %184
  %186 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %182, 0
  %187 = insertvalue { <8 x ptr>, <8 x i64> } %186, <8 x i64> %185, 1
  %188 = extractvalue { <8 x ptr>, <8 x i64> } %187, 0
  %189 = extractvalue { <8 x ptr>, <8 x i64> } %187, 1
  %190 = getelementptr i8, <8 x ptr> %188, <8 x i64> %189
  %191 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %190, i32 1, <8 x i1> %57, <8 x float> zeroinitializer)
  %.spill.load132 = load float, ptr %.spill18, align 4
  %.splatinsert133 = insertelement <8 x float> poison, float %.spill.load132, i64 0
  %.splat134 = shufflevector <8 x float> %.splatinsert133, <8 x float> poison, <8 x i32> zeroinitializer
  %192 = select <8 x i1> %181, <8 x float> %191, <8 x float> %.splat134
  %tile_snapshot.spill.load135 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill19, align 64
  %193 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load135, 0
  %194 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load135, 1
  %.state136 = load i64, ptr %.slot20, align 4
  %.splatinsert137 = insertelement <8 x i64> poison, i64 %.state136, i64 0
  %.splat138 = shufflevector <8 x i64> %.splatinsert137, <8 x i64> poison, <8 x i32> zeroinitializer
  %195 = mul <8 x i64> %.splat138, splat (i64 4)
  %196 = add <8 x i64> %194, %195
  %197 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %193, 0
  %198 = insertvalue { <8 x ptr>, <8 x i64> } %197, <8 x i64> %196, 1
  %199 = extractvalue { <8 x ptr>, <8 x i64> } %198, 0
  %200 = extractvalue { <8 x ptr>, <8 x i64> } %198, 1
  %201 = getelementptr i8, <8 x ptr> %199, <8 x i64> %200
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %192, <8 x ptr> %201, i32 1, <8 x i1> %57)
  %.state139 = load i64, ptr %.slot20, align 4
  %202 = add i64 %.state139, 1
  store i64 %202, ptr %.slot20, align 4
  br label %direct.schedule.10

direct.schedule.12:                               ; preds = %direct.false124
  store float 0xFFF0000000000000, ptr %.spill21, align 4
  store i64 0, ptr %.spill22, align 4
  store i64 0, ptr %.spill23, align 4
  store i1 true, ptr %.spill24, align 1
  br i1 true, label %direct.true140, label %direct.false141

direct.schedule.13:                               ; preds = %direct.true140
  %tile_snapshot.spill.load142 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill19, align 64
  %203 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load142, 0
  %204 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load142, 1
  %.spill.load143 = load i64, ptr %.spill23, align 4
  %.splatinsert144 = insertelement <8 x i64> poison, i64 %.spill.load143, i64 0
  %.splat145 = shufflevector <8 x i64> %.splatinsert144, <8 x i64> poison, <8 x i32> zeroinitializer
  %205 = mul <8 x i64> %.splat145, splat (i64 4)
  %206 = add <8 x i64> %204, %205
  %207 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %203, 0
  %208 = insertvalue { <8 x ptr>, <8 x i64> } %207, <8 x i64> %206, 1
  %209 = extractvalue { <8 x ptr>, <8 x i64> } %208, 0
  %210 = extractvalue { <8 x ptr>, <8 x i64> } %208, 1
  %211 = getelementptr i8, <8 x ptr> %209, <8 x i64> %210
  %212 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %211, i32 1, <8 x i1> %57, <8 x float> zeroinitializer)
  %213 = load <8 x float>, ptr %.slot25, align 32
  %214 = select <8 x i1> %57, <8 x float> %212, <8 x float> %213
  store <8 x float> %214, ptr %.slot25, align 32
  br label %direct.schedule.14

direct.schedule.14:                               ; preds = %direct.schedule.13, %direct.false141
  %.spill.load146 = load i64, ptr %.spill22, align 4
  %215 = add i64 %.spill.load146, 1
  store i64 %215, ptr %.spill26, align 4
  %216 = icmp sge i64 %215, 0
  %217 = icmp slt i64 %215, 4096
  %218 = and i1 %216, %217
  store i1 %218, ptr %.spill27, align 1
  br i1 %218, label %direct.true147, label %direct.false148

direct.schedule.15:                               ; preds = %direct.true147
  %tile_snapshot.spill.load149 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill19, align 64
  %219 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load149, 0
  %220 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load149, 1
  %.spill.load150 = load i64, ptr %.spill26, align 4
  %.splatinsert151 = insertelement <8 x i64> poison, i64 %.spill.load150, i64 0
  %.splat152 = shufflevector <8 x i64> %.splatinsert151, <8 x i64> poison, <8 x i32> zeroinitializer
  %221 = mul <8 x i64> %.splat152, splat (i64 4)
  %222 = add <8 x i64> %220, %221
  %223 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %219, 0
  %224 = insertvalue { <8 x ptr>, <8 x i64> } %223, <8 x i64> %222, 1
  %225 = extractvalue { <8 x ptr>, <8 x i64> } %224, 0
  %226 = extractvalue { <8 x ptr>, <8 x i64> } %224, 1
  %227 = getelementptr i8, <8 x ptr> %225, <8 x i64> %226
  %228 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %227, i32 1, <8 x i1> %57, <8 x float> zeroinitializer)
  %229 = load <8 x float>, ptr %.slot28, align 32
  %230 = select <8 x i1> %57, <8 x float> %228, <8 x float> %229
  store <8 x float> %230, ptr %.slot28, align 32
  br label %direct.schedule.16

direct.schedule.16:                               ; preds = %direct.schedule.15, %direct.false148
  %.spill.load153 = load i64, ptr %.spill22, align 4
  %231 = add i64 %.spill.load153, 2
  store i64 %231, ptr %.spill29, align 4
  %232 = icmp sge i64 %231, 0
  %233 = icmp slt i64 %231, 4096
  %234 = and i1 %232, %233
  store i1 %234, ptr %.spill30, align 1
  br i1 %234, label %direct.true154, label %direct.false155

direct.schedule.17:                               ; preds = %direct.true154
  %tile_snapshot.spill.load156 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill19, align 64
  %235 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load156, 0
  %236 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load156, 1
  %.spill.load157 = load i64, ptr %.spill29, align 4
  %.splatinsert158 = insertelement <8 x i64> poison, i64 %.spill.load157, i64 0
  %.splat159 = shufflevector <8 x i64> %.splatinsert158, <8 x i64> poison, <8 x i32> zeroinitializer
  %237 = mul <8 x i64> %.splat159, splat (i64 4)
  %238 = add <8 x i64> %236, %237
  %239 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %235, 0
  %240 = insertvalue { <8 x ptr>, <8 x i64> } %239, <8 x i64> %238, 1
  %241 = extractvalue { <8 x ptr>, <8 x i64> } %240, 0
  %242 = extractvalue { <8 x ptr>, <8 x i64> } %240, 1
  %243 = getelementptr i8, <8 x ptr> %241, <8 x i64> %242
  %244 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %243, i32 1, <8 x i1> %57, <8 x float> zeroinitializer)
  %245 = load <8 x float>, ptr %.slot31, align 32
  %246 = select <8 x i1> %57, <8 x float> %244, <8 x float> %245
  store <8 x float> %246, ptr %.slot31, align 32
  br label %direct.schedule.18

direct.schedule.18:                               ; preds = %direct.schedule.17, %direct.false155
  %.spill.load160 = load i64, ptr %.spill22, align 4
  %247 = add i64 %.spill.load160, 3
  store i64 %247, ptr %.spill32, align 4
  %248 = icmp sge i64 %247, 0
  %249 = icmp slt i64 %247, 4096
  %250 = and i1 %248, %249
  store i1 %250, ptr %.spill33, align 1
  br i1 %250, label %direct.true161, label %direct.false162

direct.schedule.19:                               ; preds = %direct.true161
  %tile_snapshot.spill.load163 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill19, align 64
  %251 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load163, 0
  %252 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load163, 1
  %.spill.load164 = load i64, ptr %.spill32, align 4
  %.splatinsert165 = insertelement <8 x i64> poison, i64 %.spill.load164, i64 0
  %.splat166 = shufflevector <8 x i64> %.splatinsert165, <8 x i64> poison, <8 x i32> zeroinitializer
  %253 = mul <8 x i64> %.splat166, splat (i64 4)
  %254 = add <8 x i64> %252, %253
  %255 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %251, 0
  %256 = insertvalue { <8 x ptr>, <8 x i64> } %255, <8 x i64> %254, 1
  %257 = extractvalue { <8 x ptr>, <8 x i64> } %256, 0
  %258 = extractvalue { <8 x ptr>, <8 x i64> } %256, 1
  %259 = getelementptr i8, <8 x ptr> %257, <8 x i64> %258
  %260 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %259, i32 1, <8 x i1> %57, <8 x float> zeroinitializer)
  %261 = load <8 x float>, ptr %.slot34, align 32
  %262 = select <8 x i1> %57, <8 x float> %260, <8 x float> %261
  store <8 x float> %262, ptr %.slot34, align 32
  br label %direct.schedule.20

direct.schedule.20:                               ; preds = %direct.schedule.19, %direct.false162
  %.state167 = load <8 x float>, ptr %.slot25, align 32
  %.state168 = load <8 x float>, ptr %.slot28, align 32
  %.state169 = load <8 x float>, ptr %.slot31, align 32
  %.state170 = load <8 x float>, ptr %.slot34, align 32
  store i64 4, ptr %.slot35, align 4
  %263 = load <8 x float>, ptr %.slot36, align 32
  %264 = select <8 x i1> %57, <8 x float> %.state167, <8 x float> %263
  store <8 x float> %264, ptr %.slot36, align 32
  %265 = load <8 x float>, ptr %.slot37, align 32
  %266 = select <8 x i1> %57, <8 x float> %.state168, <8 x float> %265
  store <8 x float> %266, ptr %.slot37, align 32
  %267 = load <8 x float>, ptr %.slot38, align 32
  %268 = select <8 x i1> %57, <8 x float> %.state169, <8 x float> %267
  store <8 x float> %268, ptr %.slot38, align 32
  %269 = load <8 x float>, ptr %.slot39, align 32
  %270 = select <8 x i1> %57, <8 x float> %.state170, <8 x float> %269
  store <8 x float> %270, ptr %.slot39, align 32
  br label %direct.schedule.21

direct.schedule.21:                               ; preds = %direct.schedule.30, %direct.schedule.20
  %.state171 = load i64, ptr %.slot35, align 4
  %271 = icmp slt i64 %.state171, 4096
  br i1 %271, label %direct.true172, label %direct.false173

direct.schedule.22:                               ; preds = %direct.true172
  %.state174 = load i64, ptr %.slot35, align 4
  %272 = add i64 %.state174, 0
  %273 = sdiv i64 %272, 1
  %274 = srem i64 %273, 4096
  %.spill.load175 = load i64, ptr %.spill22, align 4
  %275 = add i64 %.spill.load175, %274
  store i64 %275, ptr %.spill40, align 4
  %276 = icmp sge i64 %275, 0
  %277 = icmp slt i64 %275, 4096
  %278 = and i1 %276, %277
  br i1 %278, label %direct.true176, label %direct.false177

direct.schedule.23:                               ; preds = %direct.true176
  %tile_snapshot.spill.load178 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill19, align 64
  %279 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load178, 0
  %280 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load178, 1
  %.spill.load179 = load i64, ptr %.spill40, align 4
  %.splatinsert180 = insertelement <8 x i64> poison, i64 %.spill.load179, i64 0
  %.splat181 = shufflevector <8 x i64> %.splatinsert180, <8 x i64> poison, <8 x i32> zeroinitializer
  %281 = mul <8 x i64> %.splat181, splat (i64 4)
  %282 = add <8 x i64> %280, %281
  %283 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %279, 0
  %284 = insertvalue { <8 x ptr>, <8 x i64> } %283, <8 x i64> %282, 1
  %285 = extractvalue { <8 x ptr>, <8 x i64> } %284, 0
  %286 = extractvalue { <8 x ptr>, <8 x i64> } %284, 1
  %287 = getelementptr i8, <8 x ptr> %285, <8 x i64> %286
  %288 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %287, i32 1, <8 x i1> %57, <8 x float> zeroinitializer)
  %289 = load <8 x float>, ptr %.slot41, align 32
  %290 = select <8 x i1> %57, <8 x float> %288, <8 x float> %289
  store <8 x float> %290, ptr %.slot41, align 32
  br label %direct.schedule.24

direct.schedule.24:                               ; preds = %direct.schedule.23, %direct.false177
  %.state182 = load <8 x float>, ptr %.slot36, align 32
  %.state183 = load <8 x float>, ptr %.slot41, align 32
  %291 = call <8 x float> @llvm.maxnum.v8f32(<8 x float> %.state182, <8 x float> %.state183)
  %292 = load <8 x float>, ptr %.spill42, align 32
  %293 = select <8 x i1> %57, <8 x float> %291, <8 x float> %292
  store <8 x float> %293, ptr %.spill42, align 32
  %.state184 = load i64, ptr %.slot35, align 4
  %294 = add i64 %.state184, 1
  %295 = sdiv i64 %294, 1
  %296 = srem i64 %295, 4096
  %.spill.load185 = load i64, ptr %.spill22, align 4
  %297 = add i64 %.spill.load185, %296
  store i64 %297, ptr %.spill43, align 4
  %298 = icmp sge i64 %297, 0
  %299 = icmp slt i64 %297, 4096
  %300 = and i1 %298, %299
  br i1 %300, label %direct.true186, label %direct.false187

direct.schedule.25:                               ; preds = %direct.true186
  %tile_snapshot.spill.load188 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill19, align 64
  %301 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load188, 0
  %302 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load188, 1
  %.spill.load189 = load i64, ptr %.spill43, align 4
  %.splatinsert190 = insertelement <8 x i64> poison, i64 %.spill.load189, i64 0
  %.splat191 = shufflevector <8 x i64> %.splatinsert190, <8 x i64> poison, <8 x i32> zeroinitializer
  %303 = mul <8 x i64> %.splat191, splat (i64 4)
  %304 = add <8 x i64> %302, %303
  %305 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %301, 0
  %306 = insertvalue { <8 x ptr>, <8 x i64> } %305, <8 x i64> %304, 1
  %307 = extractvalue { <8 x ptr>, <8 x i64> } %306, 0
  %308 = extractvalue { <8 x ptr>, <8 x i64> } %306, 1
  %309 = getelementptr i8, <8 x ptr> %307, <8 x i64> %308
  %310 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %309, i32 1, <8 x i1> %57, <8 x float> zeroinitializer)
  %311 = load <8 x float>, ptr %.slot44, align 32
  %312 = select <8 x i1> %57, <8 x float> %310, <8 x float> %311
  store <8 x float> %312, ptr %.slot44, align 32
  br label %direct.schedule.26

direct.schedule.26:                               ; preds = %direct.schedule.25, %direct.false187
  %.state192 = load <8 x float>, ptr %.slot37, align 32
  %.state193 = load <8 x float>, ptr %.slot44, align 32
  %313 = call <8 x float> @llvm.maxnum.v8f32(<8 x float> %.state192, <8 x float> %.state193)
  %314 = load <8 x float>, ptr %.spill45, align 32
  %315 = select <8 x i1> %57, <8 x float> %313, <8 x float> %314
  store <8 x float> %315, ptr %.spill45, align 32
  %.state194 = load i64, ptr %.slot35, align 4
  %316 = add i64 %.state194, 2
  %317 = sdiv i64 %316, 1
  %318 = srem i64 %317, 4096
  %.spill.load195 = load i64, ptr %.spill22, align 4
  %319 = add i64 %.spill.load195, %318
  store i64 %319, ptr %.spill46, align 4
  %320 = icmp sge i64 %319, 0
  %321 = icmp slt i64 %319, 4096
  %322 = and i1 %320, %321
  br i1 %322, label %direct.true196, label %direct.false197

direct.schedule.27:                               ; preds = %direct.true196
  %tile_snapshot.spill.load198 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill19, align 64
  %323 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load198, 0
  %324 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load198, 1
  %.spill.load199 = load i64, ptr %.spill46, align 4
  %.splatinsert200 = insertelement <8 x i64> poison, i64 %.spill.load199, i64 0
  %.splat201 = shufflevector <8 x i64> %.splatinsert200, <8 x i64> poison, <8 x i32> zeroinitializer
  %325 = mul <8 x i64> %.splat201, splat (i64 4)
  %326 = add <8 x i64> %324, %325
  %327 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %323, 0
  %328 = insertvalue { <8 x ptr>, <8 x i64> } %327, <8 x i64> %326, 1
  %329 = extractvalue { <8 x ptr>, <8 x i64> } %328, 0
  %330 = extractvalue { <8 x ptr>, <8 x i64> } %328, 1
  %331 = getelementptr i8, <8 x ptr> %329, <8 x i64> %330
  %332 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %331, i32 1, <8 x i1> %57, <8 x float> zeroinitializer)
  %333 = load <8 x float>, ptr %.slot47, align 32
  %334 = select <8 x i1> %57, <8 x float> %332, <8 x float> %333
  store <8 x float> %334, ptr %.slot47, align 32
  br label %direct.schedule.28

direct.schedule.28:                               ; preds = %direct.schedule.27, %direct.false197
  %.state202 = load <8 x float>, ptr %.slot38, align 32
  %.state203 = load <8 x float>, ptr %.slot47, align 32
  %335 = call <8 x float> @llvm.maxnum.v8f32(<8 x float> %.state202, <8 x float> %.state203)
  %336 = load <8 x float>, ptr %.spill48, align 32
  %337 = select <8 x i1> %57, <8 x float> %335, <8 x float> %336
  store <8 x float> %337, ptr %.spill48, align 32
  %.state204 = load i64, ptr %.slot35, align 4
  %338 = add i64 %.state204, 3
  %339 = sdiv i64 %338, 1
  %340 = srem i64 %339, 4096
  %.spill.load205 = load i64, ptr %.spill22, align 4
  %341 = add i64 %.spill.load205, %340
  store i64 %341, ptr %.spill49, align 4
  %342 = icmp sge i64 %341, 0
  %343 = icmp slt i64 %341, 4096
  %344 = and i1 %342, %343
  br i1 %344, label %direct.true206, label %direct.false207

direct.schedule.29:                               ; preds = %direct.true206
  %tile_snapshot.spill.load208 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill19, align 64
  %345 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load208, 0
  %346 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load208, 1
  %.spill.load209 = load i64, ptr %.spill49, align 4
  %.splatinsert210 = insertelement <8 x i64> poison, i64 %.spill.load209, i64 0
  %.splat211 = shufflevector <8 x i64> %.splatinsert210, <8 x i64> poison, <8 x i32> zeroinitializer
  %347 = mul <8 x i64> %.splat211, splat (i64 4)
  %348 = add <8 x i64> %346, %347
  %349 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %345, 0
  %350 = insertvalue { <8 x ptr>, <8 x i64> } %349, <8 x i64> %348, 1
  %351 = extractvalue { <8 x ptr>, <8 x i64> } %350, 0
  %352 = extractvalue { <8 x ptr>, <8 x i64> } %350, 1
  %353 = getelementptr i8, <8 x ptr> %351, <8 x i64> %352
  %354 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %353, i32 1, <8 x i1> %57, <8 x float> zeroinitializer)
  %355 = load <8 x float>, ptr %.slot50, align 32
  %356 = select <8 x i1> %57, <8 x float> %354, <8 x float> %355
  store <8 x float> %356, ptr %.slot50, align 32
  br label %direct.schedule.30

direct.schedule.30:                               ; preds = %direct.schedule.29, %direct.false207
  %.state212 = load <8 x float>, ptr %.slot39, align 32
  %.state213 = load <8 x float>, ptr %.slot50, align 32
  %357 = call <8 x float> @llvm.maxnum.v8f32(<8 x float> %.state212, <8 x float> %.state213)
  %.state214 = load i64, ptr %.slot35, align 4
  %358 = add i64 %.state214, 4
  %.spill.load215 = load <8 x float>, ptr %.spill42, align 32
  %.spill.load216 = load <8 x float>, ptr %.spill45, align 32
  %.spill.load217 = load <8 x float>, ptr %.spill48, align 32
  store i64 %358, ptr %.slot35, align 4
  %359 = load <8 x float>, ptr %.slot36, align 32
  %360 = select <8 x i1> %57, <8 x float> %.spill.load215, <8 x float> %359
  store <8 x float> %360, ptr %.slot36, align 32
  %361 = load <8 x float>, ptr %.slot37, align 32
  %362 = select <8 x i1> %57, <8 x float> %.spill.load216, <8 x float> %361
  store <8 x float> %362, ptr %.slot37, align 32
  %363 = load <8 x float>, ptr %.slot38, align 32
  %364 = select <8 x i1> %57, <8 x float> %.spill.load217, <8 x float> %363
  store <8 x float> %364, ptr %.slot38, align 32
  %365 = load <8 x float>, ptr %.slot39, align 32
  %366 = select <8 x i1> %57, <8 x float> %357, <8 x float> %365
  store <8 x float> %366, ptr %.slot39, align 32
  br label %direct.schedule.21

direct.schedule.31:                               ; preds = %direct.false173
  %.spill.load218 = load float, ptr %.spill21, align 4
  %.splatinsert219 = insertelement <8 x float> poison, float %.spill.load218, i64 0
  %.splat220 = shufflevector <8 x float> %.splatinsert219, <8 x float> poison, <8 x i32> zeroinitializer
  %.state221 = load <8 x float>, ptr %.slot36, align 32
  %367 = call <8 x float> @llvm.maxnum.v8f32(<8 x float> %.splat220, <8 x float> %.state221)
  %.state222 = load <8 x float>, ptr %.slot37, align 32
  %368 = call <8 x float> @llvm.maxnum.v8f32(<8 x float> %367, <8 x float> %.state222)
  %.state223 = load <8 x float>, ptr %.slot38, align 32
  %369 = call <8 x float> @llvm.maxnum.v8f32(<8 x float> %368, <8 x float> %.state223)
  %.state224 = load <8 x float>, ptr %.slot39, align 32
  %370 = call <8 x float> @llvm.maxnum.v8f32(<8 x float> %369, <8 x float> %.state224)
  %371 = load <8 x float>, ptr %.spill51, align 32
  %372 = select <8 x i1> %57, <8 x float> %370, <8 x float> %371
  store <8 x float> %372, ptr %.spill51, align 32
  store float 0.000000e+00, ptr %.spill52, align 4
  %373 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill53, align 64
  %374 = extractvalue { <8 x ptr>, <8 x i64> } %9, 0
  %375 = extractvalue { <8 x ptr>, <8 x i64> } %373, 0
  %376 = select <8 x i1> %57, <8 x ptr> %374, <8 x ptr> %375
  %377 = extractvalue { <8 x ptr>, <8 x i64> } %9, 1
  %378 = extractvalue { <8 x ptr>, <8 x i64> } %373, 1
  %379 = select <8 x i1> %57, <8 x i64> %377, <8 x i64> %378
  %380 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %376, 0
  %381 = insertvalue { <8 x ptr>, <8 x i64> } %380, <8 x i64> %379, 1
  store { <8 x ptr>, <8 x i64> } %381, ptr %tile_snapshot.spill53, align 64
  store i64 0, ptr %.slot54, align 4
  br label %direct.schedule.32

direct.schedule.32:                               ; preds = %direct.schedule.33, %direct.schedule.31
  %.state225 = load i64, ptr %.slot54, align 4
  %382 = icmp slt i64 %.state225, 4096
  br i1 %382, label %direct.true226, label %direct.false227

direct.schedule.33:                               ; preds = %direct.true226
  %.state228 = load i64, ptr %.slot54, align 4
  %383 = srem i64 %.state228, 4096
  %384 = add i64 0, %383
  %tile_snapshot.spill.load229 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill16, align 64
  %385 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load229, 0
  %386 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load229, 1
  %.splatinsert230 = insertelement <8 x i64> poison, i64 %384, i64 0
  %.splat231 = shufflevector <8 x i64> %.splatinsert230, <8 x i64> poison, <8 x i32> zeroinitializer
  %387 = add <8 x i64> %386, %.splat231
  %388 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %385, 0
  %389 = insertvalue { <8 x ptr>, <8 x i64> } %388, <8 x i64> %387, 1
  %390 = extractvalue { <8 x ptr>, <8 x i64> } %389, 0
  %391 = extractvalue { <8 x ptr>, <8 x i64> } %389, 1
  %392 = getelementptr i8, <8 x ptr> %390, <8 x i64> %391
  %393 = call <8 x i8> @llvm.masked.gather.v8i8.v8p0(<8 x ptr> %392, i32 1, <8 x i1> %57, <8 x i8> zeroinitializer)
  %394 = icmp ne <8 x i8> %393, zeroinitializer
  %395 = srem i64 %384, 4096
  %396 = add i64 0, %395
  %397 = srem i64 %396, 4096
  %398 = add i64 0, %397
  %tile_snapshot.spill.load232 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill19, align 64
  %399 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load232, 0
  %400 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load232, 1
  %.splatinsert233 = insertelement <8 x i64> poison, i64 %398, i64 0
  %.splat234 = shufflevector <8 x i64> %.splatinsert233, <8 x i64> poison, <8 x i32> zeroinitializer
  %401 = mul <8 x i64> %.splat234, splat (i64 4)
  %402 = add <8 x i64> %400, %401
  %403 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %399, 0
  %404 = insertvalue { <8 x ptr>, <8 x i64> } %403, <8 x i64> %402, 1
  %405 = extractvalue { <8 x ptr>, <8 x i64> } %404, 0
  %406 = extractvalue { <8 x ptr>, <8 x i64> } %404, 1
  %407 = getelementptr i8, <8 x ptr> %405, <8 x i64> %406
  %408 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %407, i32 1, <8 x i1> %57, <8 x float> zeroinitializer)
  %.spill.load235 = load <8 x float>, ptr %.spill51, align 32
  %409 = fsub <8 x float> %408, %.spill.load235
  %410 = select <8 x i1> %57, <8 x float> %409, <8 x float> zeroinitializer
  %native.exp = call <8 x float> @__luisa_cpu_native_exp_f32_v8_u10(<8 x float> %410)
  %.spill.load236 = load float, ptr %.spill52, align 4
  %.splatinsert237 = insertelement <8 x float> poison, float %.spill.load236, i64 0
  %.splat238 = shufflevector <8 x float> %.splatinsert237, <8 x float> poison, <8 x i32> zeroinitializer
  %411 = select <8 x i1> %394, <8 x float> %native.exp, <8 x float> %.splat238
  %tile_snapshot.spill.load239 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill53, align 64
  %412 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load239, 0
  %413 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load239, 1
  %.state240 = load i64, ptr %.slot54, align 4
  %.splatinsert241 = insertelement <8 x i64> poison, i64 %.state240, i64 0
  %.splat242 = shufflevector <8 x i64> %.splatinsert241, <8 x i64> poison, <8 x i32> zeroinitializer
  %414 = mul <8 x i64> %.splat242, splat (i64 4)
  %415 = add <8 x i64> %413, %414
  %416 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %412, 0
  %417 = insertvalue { <8 x ptr>, <8 x i64> } %416, <8 x i64> %415, 1
  %418 = extractvalue { <8 x ptr>, <8 x i64> } %417, 0
  %419 = extractvalue { <8 x ptr>, <8 x i64> } %417, 1
  %420 = getelementptr i8, <8 x ptr> %418, <8 x i64> %419
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %411, <8 x ptr> %420, i32 1, <8 x i1> %57)
  %.state243 = load i64, ptr %.slot54, align 4
  %421 = add i64 %.state243, 1
  store i64 %421, ptr %.slot54, align 4
  br label %direct.schedule.32

direct.schedule.34:                               ; preds = %direct.false227
  %.spill.load244 = load i1, ptr %.spill24, align 1
  br i1 %.spill.load244, label %direct.true245, label %direct.false246

direct.schedule.35:                               ; preds = %direct.true245
  %tile_snapshot.spill.load247 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill53, align 64
  %422 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load247, 0
  %423 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load247, 1
  %.spill.load248 = load i64, ptr %.spill23, align 4
  %.splatinsert249 = insertelement <8 x i64> poison, i64 %.spill.load248, i64 0
  %.splat250 = shufflevector <8 x i64> %.splatinsert249, <8 x i64> poison, <8 x i32> zeroinitializer
  %424 = mul <8 x i64> %.splat250, splat (i64 4)
  %425 = add <8 x i64> %423, %424
  %426 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %422, 0
  %427 = insertvalue { <8 x ptr>, <8 x i64> } %426, <8 x i64> %425, 1
  %428 = extractvalue { <8 x ptr>, <8 x i64> } %427, 0
  %429 = extractvalue { <8 x ptr>, <8 x i64> } %427, 1
  %430 = getelementptr i8, <8 x ptr> %428, <8 x i64> %429
  %431 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %430, i32 1, <8 x i1> %57, <8 x float> zeroinitializer)
  %432 = load <8 x float>, ptr %.slot55, align 32
  %433 = select <8 x i1> %57, <8 x float> %431, <8 x float> %432
  store <8 x float> %433, ptr %.slot55, align 32
  br label %direct.schedule.36

direct.schedule.36:                               ; preds = %direct.schedule.35, %direct.false246
  %.spill.load251 = load i1, ptr %.spill27, align 1
  br i1 %.spill.load251, label %direct.true252, label %direct.false253

direct.schedule.37:                               ; preds = %direct.true252
  %tile_snapshot.spill.load254 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill53, align 64
  %434 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load254, 0
  %435 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load254, 1
  %.spill.load255 = load i64, ptr %.spill26, align 4
  %.splatinsert256 = insertelement <8 x i64> poison, i64 %.spill.load255, i64 0
  %.splat257 = shufflevector <8 x i64> %.splatinsert256, <8 x i64> poison, <8 x i32> zeroinitializer
  %436 = mul <8 x i64> %.splat257, splat (i64 4)
  %437 = add <8 x i64> %435, %436
  %438 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %434, 0
  %439 = insertvalue { <8 x ptr>, <8 x i64> } %438, <8 x i64> %437, 1
  %440 = extractvalue { <8 x ptr>, <8 x i64> } %439, 0
  %441 = extractvalue { <8 x ptr>, <8 x i64> } %439, 1
  %442 = getelementptr i8, <8 x ptr> %440, <8 x i64> %441
  %443 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %442, i32 1, <8 x i1> %57, <8 x float> zeroinitializer)
  %444 = load <8 x float>, ptr %.slot56, align 32
  %445 = select <8 x i1> %57, <8 x float> %443, <8 x float> %444
  store <8 x float> %445, ptr %.slot56, align 32
  br label %direct.schedule.38

direct.schedule.38:                               ; preds = %direct.schedule.37, %direct.false253
  %.spill.load258 = load i1, ptr %.spill30, align 1
  br i1 %.spill.load258, label %direct.true259, label %direct.false260

direct.schedule.39:                               ; preds = %direct.true259
  %tile_snapshot.spill.load261 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill53, align 64
  %446 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load261, 0
  %447 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load261, 1
  %.spill.load262 = load i64, ptr %.spill29, align 4
  %.splatinsert263 = insertelement <8 x i64> poison, i64 %.spill.load262, i64 0
  %.splat264 = shufflevector <8 x i64> %.splatinsert263, <8 x i64> poison, <8 x i32> zeroinitializer
  %448 = mul <8 x i64> %.splat264, splat (i64 4)
  %449 = add <8 x i64> %447, %448
  %450 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %446, 0
  %451 = insertvalue { <8 x ptr>, <8 x i64> } %450, <8 x i64> %449, 1
  %452 = extractvalue { <8 x ptr>, <8 x i64> } %451, 0
  %453 = extractvalue { <8 x ptr>, <8 x i64> } %451, 1
  %454 = getelementptr i8, <8 x ptr> %452, <8 x i64> %453
  %455 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %454, i32 1, <8 x i1> %57, <8 x float> zeroinitializer)
  %456 = load <8 x float>, ptr %.slot57, align 32
  %457 = select <8 x i1> %57, <8 x float> %455, <8 x float> %456
  store <8 x float> %457, ptr %.slot57, align 32
  br label %direct.schedule.40

direct.schedule.40:                               ; preds = %direct.schedule.39, %direct.false260
  %.spill.load265 = load i1, ptr %.spill33, align 1
  br i1 %.spill.load265, label %direct.true266, label %direct.false267

direct.schedule.41:                               ; preds = %direct.true266
  %tile_snapshot.spill.load268 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill53, align 64
  %458 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load268, 0
  %459 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load268, 1
  %.spill.load269 = load i64, ptr %.spill32, align 4
  %.splatinsert270 = insertelement <8 x i64> poison, i64 %.spill.load269, i64 0
  %.splat271 = shufflevector <8 x i64> %.splatinsert270, <8 x i64> poison, <8 x i32> zeroinitializer
  %460 = mul <8 x i64> %.splat271, splat (i64 4)
  %461 = add <8 x i64> %459, %460
  %462 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %458, 0
  %463 = insertvalue { <8 x ptr>, <8 x i64> } %462, <8 x i64> %461, 1
  %464 = extractvalue { <8 x ptr>, <8 x i64> } %463, 0
  %465 = extractvalue { <8 x ptr>, <8 x i64> } %463, 1
  %466 = getelementptr i8, <8 x ptr> %464, <8 x i64> %465
  %467 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %466, i32 1, <8 x i1> %57, <8 x float> zeroinitializer)
  %468 = load <8 x float>, ptr %.slot58, align 32
  %469 = select <8 x i1> %57, <8 x float> %467, <8 x float> %468
  store <8 x float> %469, ptr %.slot58, align 32
  br label %direct.schedule.42

direct.schedule.42:                               ; preds = %direct.schedule.41, %direct.false267
  %.state272 = load <8 x float>, ptr %.slot55, align 32
  %.state273 = load <8 x float>, ptr %.slot56, align 32
  %.state274 = load <8 x float>, ptr %.slot57, align 32
  %.state275 = load <8 x float>, ptr %.slot58, align 32
  store i64 4, ptr %.slot59, align 4
  %470 = load <8 x float>, ptr %.slot60, align 32
  %471 = select <8 x i1> %57, <8 x float> %.state272, <8 x float> %470
  store <8 x float> %471, ptr %.slot60, align 32
  %472 = load <8 x float>, ptr %.slot61, align 32
  %473 = select <8 x i1> %57, <8 x float> %.state273, <8 x float> %472
  store <8 x float> %473, ptr %.slot61, align 32
  %474 = load <8 x float>, ptr %.slot62, align 32
  %475 = select <8 x i1> %57, <8 x float> %.state274, <8 x float> %474
  store <8 x float> %475, ptr %.slot62, align 32
  %476 = load <8 x float>, ptr %.slot63, align 32
  %477 = select <8 x i1> %57, <8 x float> %.state275, <8 x float> %476
  store <8 x float> %477, ptr %.slot63, align 32
  br label %direct.schedule.43

direct.schedule.43:                               ; preds = %direct.schedule.52, %direct.schedule.42
  %.state276 = load i64, ptr %.slot59, align 4
  %478 = icmp slt i64 %.state276, 4096
  br i1 %478, label %direct.true277, label %direct.false278

direct.schedule.44:                               ; preds = %direct.true277
  %.state279 = load i64, ptr %.slot59, align 4
  %479 = add i64 %.state279, 0
  %480 = sdiv i64 %479, 1
  %481 = srem i64 %480, 4096
  %.spill.load280 = load i64, ptr %.spill22, align 4
  %482 = add i64 %.spill.load280, %481
  store i64 %482, ptr %.spill64, align 4
  %483 = icmp sge i64 %482, 0
  %484 = icmp slt i64 %482, 4096
  %485 = and i1 %483, %484
  br i1 %485, label %direct.true281, label %direct.false282

direct.schedule.45:                               ; preds = %direct.true281
  %tile_snapshot.spill.load283 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill53, align 64
  %486 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load283, 0
  %487 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load283, 1
  %.spill.load284 = load i64, ptr %.spill64, align 4
  %.splatinsert285 = insertelement <8 x i64> poison, i64 %.spill.load284, i64 0
  %.splat286 = shufflevector <8 x i64> %.splatinsert285, <8 x i64> poison, <8 x i32> zeroinitializer
  %488 = mul <8 x i64> %.splat286, splat (i64 4)
  %489 = add <8 x i64> %487, %488
  %490 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %486, 0
  %491 = insertvalue { <8 x ptr>, <8 x i64> } %490, <8 x i64> %489, 1
  %492 = extractvalue { <8 x ptr>, <8 x i64> } %491, 0
  %493 = extractvalue { <8 x ptr>, <8 x i64> } %491, 1
  %494 = getelementptr i8, <8 x ptr> %492, <8 x i64> %493
  %495 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %494, i32 1, <8 x i1> %57, <8 x float> zeroinitializer)
  %496 = load <8 x float>, ptr %.slot65, align 32
  %497 = select <8 x i1> %57, <8 x float> %495, <8 x float> %496
  store <8 x float> %497, ptr %.slot65, align 32
  br label %direct.schedule.46

direct.schedule.46:                               ; preds = %direct.schedule.45, %direct.false282
  %.state287 = load <8 x float>, ptr %.slot60, align 32
  %.state288 = load <8 x float>, ptr %.slot65, align 32
  %498 = fadd <8 x float> %.state287, %.state288
  %499 = load <8 x float>, ptr %.spill66, align 32
  %500 = select <8 x i1> %57, <8 x float> %498, <8 x float> %499
  store <8 x float> %500, ptr %.spill66, align 32
  %.state289 = load i64, ptr %.slot59, align 4
  %501 = add i64 %.state289, 1
  %502 = sdiv i64 %501, 1
  %503 = srem i64 %502, 4096
  %.spill.load290 = load i64, ptr %.spill22, align 4
  %504 = add i64 %.spill.load290, %503
  store i64 %504, ptr %.spill67, align 4
  %505 = icmp sge i64 %504, 0
  %506 = icmp slt i64 %504, 4096
  %507 = and i1 %505, %506
  br i1 %507, label %direct.true291, label %direct.false292

direct.schedule.47:                               ; preds = %direct.true291
  %tile_snapshot.spill.load293 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill53, align 64
  %508 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load293, 0
  %509 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load293, 1
  %.spill.load294 = load i64, ptr %.spill67, align 4
  %.splatinsert295 = insertelement <8 x i64> poison, i64 %.spill.load294, i64 0
  %.splat296 = shufflevector <8 x i64> %.splatinsert295, <8 x i64> poison, <8 x i32> zeroinitializer
  %510 = mul <8 x i64> %.splat296, splat (i64 4)
  %511 = add <8 x i64> %509, %510
  %512 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %508, 0
  %513 = insertvalue { <8 x ptr>, <8 x i64> } %512, <8 x i64> %511, 1
  %514 = extractvalue { <8 x ptr>, <8 x i64> } %513, 0
  %515 = extractvalue { <8 x ptr>, <8 x i64> } %513, 1
  %516 = getelementptr i8, <8 x ptr> %514, <8 x i64> %515
  %517 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %516, i32 1, <8 x i1> %57, <8 x float> zeroinitializer)
  %518 = load <8 x float>, ptr %.slot68, align 32
  %519 = select <8 x i1> %57, <8 x float> %517, <8 x float> %518
  store <8 x float> %519, ptr %.slot68, align 32
  br label %direct.schedule.48

direct.schedule.48:                               ; preds = %direct.schedule.47, %direct.false292
  %.state297 = load <8 x float>, ptr %.slot61, align 32
  %.state298 = load <8 x float>, ptr %.slot68, align 32
  %520 = fadd <8 x float> %.state297, %.state298
  %521 = load <8 x float>, ptr %.spill69, align 32
  %522 = select <8 x i1> %57, <8 x float> %520, <8 x float> %521
  store <8 x float> %522, ptr %.spill69, align 32
  %.state299 = load i64, ptr %.slot59, align 4
  %523 = add i64 %.state299, 2
  %524 = sdiv i64 %523, 1
  %525 = srem i64 %524, 4096
  %.spill.load300 = load i64, ptr %.spill22, align 4
  %526 = add i64 %.spill.load300, %525
  store i64 %526, ptr %.spill70, align 4
  %527 = icmp sge i64 %526, 0
  %528 = icmp slt i64 %526, 4096
  %529 = and i1 %527, %528
  br i1 %529, label %direct.true301, label %direct.false302

direct.schedule.49:                               ; preds = %direct.true301
  %tile_snapshot.spill.load303 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill53, align 64
  %530 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load303, 0
  %531 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load303, 1
  %.spill.load304 = load i64, ptr %.spill70, align 4
  %.splatinsert305 = insertelement <8 x i64> poison, i64 %.spill.load304, i64 0
  %.splat306 = shufflevector <8 x i64> %.splatinsert305, <8 x i64> poison, <8 x i32> zeroinitializer
  %532 = mul <8 x i64> %.splat306, splat (i64 4)
  %533 = add <8 x i64> %531, %532
  %534 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %530, 0
  %535 = insertvalue { <8 x ptr>, <8 x i64> } %534, <8 x i64> %533, 1
  %536 = extractvalue { <8 x ptr>, <8 x i64> } %535, 0
  %537 = extractvalue { <8 x ptr>, <8 x i64> } %535, 1
  %538 = getelementptr i8, <8 x ptr> %536, <8 x i64> %537
  %539 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %538, i32 1, <8 x i1> %57, <8 x float> zeroinitializer)
  %540 = load <8 x float>, ptr %.slot71, align 32
  %541 = select <8 x i1> %57, <8 x float> %539, <8 x float> %540
  store <8 x float> %541, ptr %.slot71, align 32
  br label %direct.schedule.50

direct.schedule.50:                               ; preds = %direct.schedule.49, %direct.false302
  %.state307 = load <8 x float>, ptr %.slot62, align 32
  %.state308 = load <8 x float>, ptr %.slot71, align 32
  %542 = fadd <8 x float> %.state307, %.state308
  %543 = load <8 x float>, ptr %.spill72, align 32
  %544 = select <8 x i1> %57, <8 x float> %542, <8 x float> %543
  store <8 x float> %544, ptr %.spill72, align 32
  %.state309 = load i64, ptr %.slot59, align 4
  %545 = add i64 %.state309, 3
  %546 = sdiv i64 %545, 1
  %547 = srem i64 %546, 4096
  %.spill.load310 = load i64, ptr %.spill22, align 4
  %548 = add i64 %.spill.load310, %547
  store i64 %548, ptr %.spill73, align 4
  %549 = icmp sge i64 %548, 0
  %550 = icmp slt i64 %548, 4096
  %551 = and i1 %549, %550
  br i1 %551, label %direct.true311, label %direct.false312

direct.schedule.51:                               ; preds = %direct.true311
  %tile_snapshot.spill.load313 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill53, align 64
  %552 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load313, 0
  %553 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load313, 1
  %.spill.load314 = load i64, ptr %.spill73, align 4
  %.splatinsert315 = insertelement <8 x i64> poison, i64 %.spill.load314, i64 0
  %.splat316 = shufflevector <8 x i64> %.splatinsert315, <8 x i64> poison, <8 x i32> zeroinitializer
  %554 = mul <8 x i64> %.splat316, splat (i64 4)
  %555 = add <8 x i64> %553, %554
  %556 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %552, 0
  %557 = insertvalue { <8 x ptr>, <8 x i64> } %556, <8 x i64> %555, 1
  %558 = extractvalue { <8 x ptr>, <8 x i64> } %557, 0
  %559 = extractvalue { <8 x ptr>, <8 x i64> } %557, 1
  %560 = getelementptr i8, <8 x ptr> %558, <8 x i64> %559
  %561 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %560, i32 1, <8 x i1> %57, <8 x float> zeroinitializer)
  %562 = load <8 x float>, ptr %.slot74, align 32
  %563 = select <8 x i1> %57, <8 x float> %561, <8 x float> %562
  store <8 x float> %563, ptr %.slot74, align 32
  br label %direct.schedule.52

direct.schedule.52:                               ; preds = %direct.schedule.51, %direct.false312
  %.state317 = load <8 x float>, ptr %.slot63, align 32
  %.state318 = load <8 x float>, ptr %.slot74, align 32
  %564 = fadd <8 x float> %.state317, %.state318
  %.state319 = load i64, ptr %.slot59, align 4
  %565 = add i64 %.state319, 4
  %.spill.load320 = load <8 x float>, ptr %.spill66, align 32
  %.spill.load321 = load <8 x float>, ptr %.spill69, align 32
  %.spill.load322 = load <8 x float>, ptr %.spill72, align 32
  store i64 %565, ptr %.slot59, align 4
  %566 = load <8 x float>, ptr %.slot60, align 32
  %567 = select <8 x i1> %57, <8 x float> %.spill.load320, <8 x float> %566
  store <8 x float> %567, ptr %.slot60, align 32
  %568 = load <8 x float>, ptr %.slot61, align 32
  %569 = select <8 x i1> %57, <8 x float> %.spill.load321, <8 x float> %568
  store <8 x float> %569, ptr %.slot61, align 32
  %570 = load <8 x float>, ptr %.slot62, align 32
  %571 = select <8 x i1> %57, <8 x float> %.spill.load322, <8 x float> %570
  store <8 x float> %571, ptr %.slot62, align 32
  %572 = load <8 x float>, ptr %.slot63, align 32
  %573 = select <8 x i1> %57, <8 x float> %564, <8 x float> %572
  store <8 x float> %573, ptr %.slot63, align 32
  br label %direct.schedule.43

direct.schedule.53:                               ; preds = %direct.false278
  %.spill.load323 = load float, ptr %.spill52, align 4
  %.splatinsert324 = insertelement <8 x float> poison, float %.spill.load323, i64 0
  %.splat325 = shufflevector <8 x float> %.splatinsert324, <8 x float> poison, <8 x i32> zeroinitializer
  %.state326 = load <8 x float>, ptr %.slot60, align 32
  %574 = fadd <8 x float> %.splat325, %.state326
  %.state327 = load <8 x float>, ptr %.slot61, align 32
  %575 = fadd <8 x float> %574, %.state327
  %.state328 = load <8 x float>, ptr %.slot62, align 32
  %576 = fadd <8 x float> %575, %.state328
  %.state329 = load <8 x float>, ptr %.slot63, align 32
  %577 = fadd <8 x float> %576, %.state329
  %578 = load <8 x float>, ptr %.spill75, align 32
  %579 = select <8 x i1> %57, <8 x float> %577, <8 x float> %578
  store <8 x float> %579, ptr %.spill75, align 32
  store i64 0, ptr %.slot76, align 4
  br label %direct.schedule.54

direct.schedule.54:                               ; preds = %direct.schedule.55, %direct.schedule.53
  %.state330 = load i64, ptr %.slot76, align 4
  %580 = icmp slt i64 %.state330, 4096
  br i1 %580, label %direct.true331, label %direct.false332

direct.schedule.55:                               ; preds = %direct.true331
  %.state333 = load i64, ptr %.slot76, align 4
  %581 = srem i64 %.state333, 4096
  %.state334 = load i64, ptr %.slot76, align 4
  %582 = sdiv i64 %.state334, 4096
  %583 = srem i64 %582, 1
  %.spill.load335 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert336 = insertelement <8 x i64> poison, i64 %583, i64 0
  %.splat337 = shufflevector <8 x i64> %.splatinsert336, <8 x i64> poison, <8 x i32> zeroinitializer
  %584 = add <8 x i64> %.spill.load335, %.splat337
  %585 = add <8 x i64> zeroinitializer, %584
  %586 = add i64 0, %581
  %587 = mul <8 x i64> %585, splat (i64 4096)
  %.splatinsert338 = insertelement <8 x i64> poison, i64 %586, i64 0
  %.splat339 = shufflevector <8 x i64> %.splatinsert338, <8 x i64> poison, <8 x i32> zeroinitializer
  %588 = add <8 x i64> %587, %.splat339
  %589 = add i64 0, %581
  %tile_snapshot.spill.load340 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill53, align 64
  %590 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load340, 0
  %591 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load340, 1
  %.splatinsert341 = insertelement <8 x i64> poison, i64 %589, i64 0
  %.splat342 = shufflevector <8 x i64> %.splatinsert341, <8 x i64> poison, <8 x i32> zeroinitializer
  %592 = mul <8 x i64> %.splat342, splat (i64 4)
  %593 = add <8 x i64> %591, %592
  %594 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %590, 0
  %595 = insertvalue { <8 x ptr>, <8 x i64> } %594, <8 x i64> %593, 1
  %596 = extractvalue { <8 x ptr>, <8 x i64> } %595, 0
  %597 = extractvalue { <8 x ptr>, <8 x i64> } %595, 1
  %598 = getelementptr i8, <8 x ptr> %596, <8 x i64> %597
  %599 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %598, i32 1, <8 x i1> %57, <8 x float> zeroinitializer)
  %.spill.load343 = load <8 x float>, ptr %.spill75, align 32
  %600 = fdiv <8 x float> %599, %.spill.load343
  %601 = extractvalue { ptr, i64 } %33, 0
  %602 = mul <8 x i64> %588, splat (i64 4)
  %603 = getelementptr i8, ptr %601, <8 x i64> %602
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %600, <8 x ptr> %603, i32 1, <8 x i1> %57)
  %.state344 = load i64, ptr %.slot76, align 4
  %604 = add i64 %.state344, 1
  store i64 %604, ptr %.slot76, align 4
  br label %direct.schedule.54

direct.schedule.56:                               ; preds = %direct.false332
  ret void

direct.activate:                                  ; preds = %prologue
  br label %direct.schedule.0

direct.inactive:                                  ; preds = %prologue
  ret void

direct.true:                                      ; preds = %direct.schedule.1
  br label %direct.schedule.2

direct.false:                                     ; preds = %direct.schedule.1
  br label %direct.schedule.3

direct.true98:                                    ; preds = %direct.schedule.4
  br label %direct.schedule.5

direct.false99:                                   ; preds = %direct.schedule.4
  br label %direct.schedule.6

direct.true110:                                   ; preds = %direct.schedule.7
  br label %direct.schedule.8

direct.false111:                                  ; preds = %direct.schedule.7
  br label %direct.schedule.9

direct.true123:                                   ; preds = %direct.schedule.10
  br label %direct.schedule.11

direct.false124:                                  ; preds = %direct.schedule.10
  br label %direct.schedule.12

direct.true140:                                   ; preds = %direct.schedule.12
  br label %direct.schedule.13

direct.false141:                                  ; preds = %direct.schedule.12
  %605 = load <8 x float>, ptr %.slot25, align 32
  %606 = select <8 x i1> %57, <8 x float> zeroinitializer, <8 x float> %605
  store <8 x float> %606, ptr %.slot25, align 32
  br label %direct.schedule.14

direct.true147:                                   ; preds = %direct.schedule.14
  br label %direct.schedule.15

direct.false148:                                  ; preds = %direct.schedule.14
  %607 = load <8 x float>, ptr %.slot28, align 32
  %608 = select <8 x i1> %57, <8 x float> zeroinitializer, <8 x float> %607
  store <8 x float> %608, ptr %.slot28, align 32
  br label %direct.schedule.16

direct.true154:                                   ; preds = %direct.schedule.16
  br label %direct.schedule.17

direct.false155:                                  ; preds = %direct.schedule.16
  %609 = load <8 x float>, ptr %.slot31, align 32
  %610 = select <8 x i1> %57, <8 x float> zeroinitializer, <8 x float> %609
  store <8 x float> %610, ptr %.slot31, align 32
  br label %direct.schedule.18

direct.true161:                                   ; preds = %direct.schedule.18
  br label %direct.schedule.19

direct.false162:                                  ; preds = %direct.schedule.18
  %611 = load <8 x float>, ptr %.slot34, align 32
  %612 = select <8 x i1> %57, <8 x float> zeroinitializer, <8 x float> %611
  store <8 x float> %612, ptr %.slot34, align 32
  br label %direct.schedule.20

direct.true172:                                   ; preds = %direct.schedule.21
  br label %direct.schedule.22

direct.false173:                                  ; preds = %direct.schedule.21
  br label %direct.schedule.31

direct.true176:                                   ; preds = %direct.schedule.22
  br label %direct.schedule.23

direct.false177:                                  ; preds = %direct.schedule.22
  %613 = load <8 x float>, ptr %.slot41, align 32
  %614 = select <8 x i1> %57, <8 x float> zeroinitializer, <8 x float> %613
  store <8 x float> %614, ptr %.slot41, align 32
  br label %direct.schedule.24

direct.true186:                                   ; preds = %direct.schedule.24
  br label %direct.schedule.25

direct.false187:                                  ; preds = %direct.schedule.24
  %615 = load <8 x float>, ptr %.slot44, align 32
  %616 = select <8 x i1> %57, <8 x float> zeroinitializer, <8 x float> %615
  store <8 x float> %616, ptr %.slot44, align 32
  br label %direct.schedule.26

direct.true196:                                   ; preds = %direct.schedule.26
  br label %direct.schedule.27

direct.false197:                                  ; preds = %direct.schedule.26
  %617 = load <8 x float>, ptr %.slot47, align 32
  %618 = select <8 x i1> %57, <8 x float> zeroinitializer, <8 x float> %617
  store <8 x float> %618, ptr %.slot47, align 32
  br label %direct.schedule.28

direct.true206:                                   ; preds = %direct.schedule.28
  br label %direct.schedule.29

direct.false207:                                  ; preds = %direct.schedule.28
  %619 = load <8 x float>, ptr %.slot50, align 32
  %620 = select <8 x i1> %57, <8 x float> zeroinitializer, <8 x float> %619
  store <8 x float> %620, ptr %.slot50, align 32
  br label %direct.schedule.30

direct.true226:                                   ; preds = %direct.schedule.32
  br label %direct.schedule.33

direct.false227:                                  ; preds = %direct.schedule.32
  br label %direct.schedule.34

direct.true245:                                   ; preds = %direct.schedule.34
  br label %direct.schedule.35

direct.false246:                                  ; preds = %direct.schedule.34
  %621 = load <8 x float>, ptr %.slot55, align 32
  %622 = select <8 x i1> %57, <8 x float> zeroinitializer, <8 x float> %621
  store <8 x float> %622, ptr %.slot55, align 32
  br label %direct.schedule.36

direct.true252:                                   ; preds = %direct.schedule.36
  br label %direct.schedule.37

direct.false253:                                  ; preds = %direct.schedule.36
  %623 = load <8 x float>, ptr %.slot56, align 32
  %624 = select <8 x i1> %57, <8 x float> zeroinitializer, <8 x float> %623
  store <8 x float> %624, ptr %.slot56, align 32
  br label %direct.schedule.38

direct.true259:                                   ; preds = %direct.schedule.38
  br label %direct.schedule.39

direct.false260:                                  ; preds = %direct.schedule.38
  %625 = load <8 x float>, ptr %.slot57, align 32
  %626 = select <8 x i1> %57, <8 x float> zeroinitializer, <8 x float> %625
  store <8 x float> %626, ptr %.slot57, align 32
  br label %direct.schedule.40

direct.true266:                                   ; preds = %direct.schedule.40
  br label %direct.schedule.41

direct.false267:                                  ; preds = %direct.schedule.40
  %627 = load <8 x float>, ptr %.slot58, align 32
  %628 = select <8 x i1> %57, <8 x float> zeroinitializer, <8 x float> %627
  store <8 x float> %628, ptr %.slot58, align 32
  br label %direct.schedule.42

direct.true277:                                   ; preds = %direct.schedule.43
  br label %direct.schedule.44

direct.false278:                                  ; preds = %direct.schedule.43
  br label %direct.schedule.53

direct.true281:                                   ; preds = %direct.schedule.44
  br label %direct.schedule.45

direct.false282:                                  ; preds = %direct.schedule.44
  %629 = load <8 x float>, ptr %.slot65, align 32
  %630 = select <8 x i1> %57, <8 x float> zeroinitializer, <8 x float> %629
  store <8 x float> %630, ptr %.slot65, align 32
  br label %direct.schedule.46

direct.true291:                                   ; preds = %direct.schedule.46
  br label %direct.schedule.47

direct.false292:                                  ; preds = %direct.schedule.46
  %631 = load <8 x float>, ptr %.slot68, align 32
  %632 = select <8 x i1> %57, <8 x float> zeroinitializer, <8 x float> %631
  store <8 x float> %632, ptr %.slot68, align 32
  br label %direct.schedule.48

direct.true301:                                   ; preds = %direct.schedule.48
  br label %direct.schedule.49

direct.false302:                                  ; preds = %direct.schedule.48
  %633 = load <8 x float>, ptr %.slot71, align 32
  %634 = select <8 x i1> %57, <8 x float> zeroinitializer, <8 x float> %633
  store <8 x float> %634, ptr %.slot71, align 32
  br label %direct.schedule.50

direct.true311:                                   ; preds = %direct.schedule.50
  br label %direct.schedule.51

direct.false312:                                  ; preds = %direct.schedule.50
  %635 = load <8 x float>, ptr %.slot74, align 32
  %636 = select <8 x i1> %57, <8 x float> zeroinitializer, <8 x float> %635
  store <8 x float> %636, ptr %.slot74, align 32
  br label %direct.schedule.52

direct.true331:                                   ; preds = %direct.schedule.54
  br label %direct.schedule.55

direct.false332:                                  ; preds = %direct.schedule.54
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
