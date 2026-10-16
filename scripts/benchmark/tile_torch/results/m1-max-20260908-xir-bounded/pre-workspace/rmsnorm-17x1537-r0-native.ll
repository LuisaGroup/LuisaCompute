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
  %.spill5 = alloca i64, align 8
  store i64 0, ptr %.spill5, align 4
  %.spill6 = alloca i64, align 8
  store i64 0, ptr %.spill6, align 4
  %.spill7 = alloca i64, align 8
  store i64 0, ptr %.spill7, align 4
  %.slot8 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot8, align 32
  %.spill9 = alloca i64, align 8
  store i64 0, ptr %.spill9, align 4
  %.slot10 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot10, align 32
  %.spill11 = alloca i64, align 8
  store i64 0, ptr %.spill11, align 4
  %.slot12 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot12, align 32
  %.spill13 = alloca i64, align 8
  store i64 0, ptr %.spill13, align 4
  %.slot14 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot14, align 32
  %.slot15 = alloca i64, align 8
  store i64 0, ptr %.slot15, align 4
  %.slot16 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot16, align 32
  %.slot17 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot17, align 32
  %.slot18 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot18, align 32
  %.slot19 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot19, align 32
  %.spill20 = alloca i64, align 8
  store i64 0, ptr %.spill20, align 4
  %.slot21 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot21, align 32
  %.spill22 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill22, align 32
  %.spill23 = alloca i64, align 8
  store i64 0, ptr %.spill23, align 4
  %.slot24 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot24, align 32
  %.spill25 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill25, align 32
  %.spill26 = alloca i64, align 8
  store i64 0, ptr %.spill26, align 4
  %.slot27 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot27, align 32
  %.spill28 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill28, align 32
  %.spill29 = alloca i64, align 8
  store i64 0, ptr %.spill29, align 4
  %.slot30 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot30, align 32
  %.spill31 = alloca i64, align 8
  store i64 0, ptr %.spill31, align 4
  %.slot32 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot32, align 32
  %.spill33 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill33, align 32
  %tile_snapshot.spill34 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill34, align 64
  %.slot35 = alloca i64, align 8
  store i64 0, ptr %.slot35, align 4
  %.spill36 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill36, align 32
  %.slot37 = alloca i64, align 8
  store i64 0, ptr %.slot37, align 4
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
  %.splatinsert38 = insertelement <8 x i32> poison, i32 %40, i64 0
  %.splat39 = shufflevector <8 x i32> %.splatinsert38, <8 x i32> poison, <8 x i32> zeroinitializer
  %41 = add <8 x i32> %.splat39, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %42 = mul i32 %28, 32
  %.splatinsert40 = insertelement <8 x i32> poison, i32 %42, i64 0
  %.splat41 = shufflevector <8 x i32> %.splatinsert40, <8 x i32> poison, <8 x i32> zeroinitializer
  %43 = add <8 x i32> %.splat41, %41
  %44 = mul i32 %32, 1
  %.splatinsert42 = insertelement <8 x i32> poison, i32 %44, i64 0
  %.splat43 = shufflevector <8 x i32> %.splatinsert42, <8 x i32> poison, <8 x i32> zeroinitializer
  %45 = add <8 x i32> %.splat43, zeroinitializer
  %46 = mul i32 %36, 1
  %.splatinsert44 = insertelement <8 x i32> poison, i32 %46, i64 0
  %.splat45 = shufflevector <8 x i32> %.splatinsert44, <8 x i32> poison, <8 x i32> zeroinitializer
  %47 = add <8 x i32> %.splat45, zeroinitializer
  %48 = insertvalue [3 x <8 x i32>] poison, <8 x i32> %43, 0
  %49 = insertvalue [3 x <8 x i32>] %48, <8 x i32> %45, 1
  %50 = insertvalue [3 x <8 x i32>] %49, <8 x i32> %47, 2
  %.splatinsert46 = insertelement <8 x i32> poison, i32 %active_lane_count, i64 0
  %.splat47 = shufflevector <8 x i32> %.splatinsert46, <8 x i32> poison, <8 x i32> zeroinitializer
  %51 = icmp ult <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, %.splat47
  %52 = call i1 @llvm.vector.reduce.or.v8i1(<8 x i1> %51)
  br i1 %52, label %direct.activate, label %direct.inactive

direct.schedule.0:                                ; preds = %direct.activate
  %53 = extractvalue [3 x <8 x i32>] %50, 0
  %54 = zext <8 x i32> %53 to <8 x i64>
  %55 = select <8 x i1> %51, <8 x i64> %54, <8 x i64> zeroinitializer
  %56 = select <8 x i1> %51, <8 x i64> splat (i64 1), <8 x i64> splat (i64 1)
  %57 = sdiv <8 x i64> %55, %56
  %58 = select <8 x i1> %51, <8 x i64> %57, <8 x i64> zeroinitializer
  %59 = select <8 x i1> %51, <8 x i64> splat (i64 17), <8 x i64> splat (i64 1)
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
  %.state48 = load i64, ptr %.slot, align 4
  %73 = srem i64 %.state48, 1537
  %.state49 = load i64, ptr %.slot, align 4
  %74 = sdiv i64 %.state49, 1537
  %75 = srem i64 %74, 1
  %.spill.load = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert50 = insertelement <8 x i64> poison, i64 %75, i64 0
  %.splat51 = shufflevector <8 x i64> %.splatinsert50, <8 x i64> poison, <8 x i32> zeroinitializer
  %76 = add <8 x i64> %.spill.load, %.splat51
  %77 = add <8 x i64> zeroinitializer, %76
  %78 = add i64 0, %73
  %79 = mul <8 x i64> %77, splat (i64 1537)
  %.splatinsert52 = insertelement <8 x i64> poison, i64 %78, i64 0
  %.splat53 = shufflevector <8 x i64> %.splatinsert52, <8 x i64> poison, <8 x i32> zeroinitializer
  %80 = add <8 x i64> %79, %.splat53
  %81 = extractvalue { ptr, i64 } %9, 0
  %82 = mul <8 x i64> %80, splat (i64 4)
  %83 = getelementptr i8, ptr %81, <8 x i64> %82
  %84 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %83, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %85 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 0
  %86 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 1
  %.state54 = load i64, ptr %.slot, align 4
  %.splatinsert55 = insertelement <8 x i64> poison, i64 %.state54, i64 0
  %.splat56 = shufflevector <8 x i64> %.splatinsert55, <8 x i64> poison, <8 x i32> zeroinitializer
  %87 = mul <8 x i64> %.splat56, splat (i64 4)
  %88 = add <8 x i64> %86, %87
  %89 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %85, 0
  %90 = insertvalue { <8 x ptr>, <8 x i64> } %89, <8 x i64> %88, 1
  %91 = extractvalue { <8 x ptr>, <8 x i64> } %90, 0
  %92 = extractvalue { <8 x ptr>, <8 x i64> } %90, 1
  %93 = getelementptr i8, <8 x ptr> %91, <8 x i64> %92
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %84, <8 x ptr> %93, i32 1, <8 x i1> %51)
  %.state57 = load i64, ptr %.slot, align 4
  %94 = add i64 %.state57, 1
  store i64 %94, ptr %.slot, align 4
  br label %direct.schedule.1

direct.schedule.3:                                ; preds = %direct.false
  store float 0.000000e+00, ptr %.spill4, align 4
  store i64 0, ptr %.spill5, align 4
  store i64 0, ptr %.spill6, align 4
  store i64 0, ptr %.spill7, align 4
  br i1 true, label %direct.true58, label %direct.false59

direct.schedule.4:                                ; preds = %direct.true58
  %.spill.load60 = load i64, ptr %.spill7, align 4
  %95 = srem i64 %.spill.load60, 1537
  %96 = add i64 0, %95
  %tile_snapshot.spill.load61 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %97 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load61, 0
  %98 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load61, 1
  %.splatinsert62 = insertelement <8 x i64> poison, i64 %96, i64 0
  %.splat63 = shufflevector <8 x i64> %.splatinsert62, <8 x i64> poison, <8 x i32> zeroinitializer
  %99 = mul <8 x i64> %.splat63, splat (i64 4)
  %100 = add <8 x i64> %98, %99
  %101 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %97, 0
  %102 = insertvalue { <8 x ptr>, <8 x i64> } %101, <8 x i64> %100, 1
  %103 = extractvalue { <8 x ptr>, <8 x i64> } %102, 0
  %104 = extractvalue { <8 x ptr>, <8 x i64> } %102, 1
  %105 = getelementptr i8, <8 x ptr> %103, <8 x i64> %104
  %106 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %105, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %107 = fmul <8 x float> %106, %106
  %108 = load <8 x float>, ptr %.slot8, align 32
  %109 = select <8 x i1> %51, <8 x float> %107, <8 x float> %108
  store <8 x float> %109, ptr %.slot8, align 32
  br label %direct.schedule.5

direct.schedule.5:                                ; preds = %direct.schedule.4, %direct.false59
  %.spill.load64 = load i64, ptr %.spill6, align 4
  %110 = add i64 %.spill.load64, 1
  store i64 %110, ptr %.spill9, align 4
  %111 = icmp sge i64 %110, 0
  %112 = icmp slt i64 %110, 1537
  %113 = and i1 %111, %112
  br i1 %113, label %direct.true65, label %direct.false66

direct.schedule.6:                                ; preds = %direct.true65
  %.spill.load67 = load i64, ptr %.spill9, align 4
  %114 = srem i64 %.spill.load67, 1537
  %115 = add i64 0, %114
  %tile_snapshot.spill.load68 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %116 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load68, 0
  %117 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load68, 1
  %.splatinsert69 = insertelement <8 x i64> poison, i64 %115, i64 0
  %.splat70 = shufflevector <8 x i64> %.splatinsert69, <8 x i64> poison, <8 x i32> zeroinitializer
  %118 = mul <8 x i64> %.splat70, splat (i64 4)
  %119 = add <8 x i64> %117, %118
  %120 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %116, 0
  %121 = insertvalue { <8 x ptr>, <8 x i64> } %120, <8 x i64> %119, 1
  %122 = extractvalue { <8 x ptr>, <8 x i64> } %121, 0
  %123 = extractvalue { <8 x ptr>, <8 x i64> } %121, 1
  %124 = getelementptr i8, <8 x ptr> %122, <8 x i64> %123
  %125 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %124, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %126 = fmul <8 x float> %125, %125
  %127 = load <8 x float>, ptr %.slot10, align 32
  %128 = select <8 x i1> %51, <8 x float> %126, <8 x float> %127
  store <8 x float> %128, ptr %.slot10, align 32
  br label %direct.schedule.7

direct.schedule.7:                                ; preds = %direct.schedule.6, %direct.false66
  %.spill.load71 = load i64, ptr %.spill6, align 4
  %129 = add i64 %.spill.load71, 2
  store i64 %129, ptr %.spill11, align 4
  %130 = icmp sge i64 %129, 0
  %131 = icmp slt i64 %129, 1537
  %132 = and i1 %130, %131
  br i1 %132, label %direct.true72, label %direct.false73

direct.schedule.8:                                ; preds = %direct.true72
  %.spill.load74 = load i64, ptr %.spill11, align 4
  %133 = srem i64 %.spill.load74, 1537
  %134 = add i64 0, %133
  %tile_snapshot.spill.load75 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %135 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load75, 0
  %136 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load75, 1
  %.splatinsert76 = insertelement <8 x i64> poison, i64 %134, i64 0
  %.splat77 = shufflevector <8 x i64> %.splatinsert76, <8 x i64> poison, <8 x i32> zeroinitializer
  %137 = mul <8 x i64> %.splat77, splat (i64 4)
  %138 = add <8 x i64> %136, %137
  %139 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %135, 0
  %140 = insertvalue { <8 x ptr>, <8 x i64> } %139, <8 x i64> %138, 1
  %141 = extractvalue { <8 x ptr>, <8 x i64> } %140, 0
  %142 = extractvalue { <8 x ptr>, <8 x i64> } %140, 1
  %143 = getelementptr i8, <8 x ptr> %141, <8 x i64> %142
  %144 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %143, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %145 = fmul <8 x float> %144, %144
  %146 = load <8 x float>, ptr %.slot12, align 32
  %147 = select <8 x i1> %51, <8 x float> %145, <8 x float> %146
  store <8 x float> %147, ptr %.slot12, align 32
  br label %direct.schedule.9

direct.schedule.9:                                ; preds = %direct.schedule.8, %direct.false73
  %.spill.load78 = load i64, ptr %.spill6, align 4
  %148 = add i64 %.spill.load78, 3
  store i64 %148, ptr %.spill13, align 4
  %149 = icmp sge i64 %148, 0
  %150 = icmp slt i64 %148, 1537
  %151 = and i1 %149, %150
  br i1 %151, label %direct.true79, label %direct.false80

direct.schedule.10:                               ; preds = %direct.true79
  %.spill.load81 = load i64, ptr %.spill13, align 4
  %152 = srem i64 %.spill.load81, 1537
  %153 = add i64 0, %152
  %tile_snapshot.spill.load82 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %154 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load82, 0
  %155 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load82, 1
  %.splatinsert83 = insertelement <8 x i64> poison, i64 %153, i64 0
  %.splat84 = shufflevector <8 x i64> %.splatinsert83, <8 x i64> poison, <8 x i32> zeroinitializer
  %156 = mul <8 x i64> %.splat84, splat (i64 4)
  %157 = add <8 x i64> %155, %156
  %158 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %154, 0
  %159 = insertvalue { <8 x ptr>, <8 x i64> } %158, <8 x i64> %157, 1
  %160 = extractvalue { <8 x ptr>, <8 x i64> } %159, 0
  %161 = extractvalue { <8 x ptr>, <8 x i64> } %159, 1
  %162 = getelementptr i8, <8 x ptr> %160, <8 x i64> %161
  %163 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %162, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %164 = fmul <8 x float> %163, %163
  %165 = load <8 x float>, ptr %.slot14, align 32
  %166 = select <8 x i1> %51, <8 x float> %164, <8 x float> %165
  store <8 x float> %166, ptr %.slot14, align 32
  br label %direct.schedule.11

direct.schedule.11:                               ; preds = %direct.schedule.10, %direct.false80
  %.state85 = load <8 x float>, ptr %.slot8, align 32
  %.state86 = load <8 x float>, ptr %.slot10, align 32
  %.state87 = load <8 x float>, ptr %.slot12, align 32
  %.state88 = load <8 x float>, ptr %.slot14, align 32
  store i64 4, ptr %.slot15, align 4
  %167 = load <8 x float>, ptr %.slot16, align 32
  %168 = select <8 x i1> %51, <8 x float> %.state85, <8 x float> %167
  store <8 x float> %168, ptr %.slot16, align 32
  %169 = load <8 x float>, ptr %.slot17, align 32
  %170 = select <8 x i1> %51, <8 x float> %.state86, <8 x float> %169
  store <8 x float> %170, ptr %.slot17, align 32
  %171 = load <8 x float>, ptr %.slot18, align 32
  %172 = select <8 x i1> %51, <8 x float> %.state87, <8 x float> %171
  store <8 x float> %172, ptr %.slot18, align 32
  %173 = load <8 x float>, ptr %.slot19, align 32
  %174 = select <8 x i1> %51, <8 x float> %.state88, <8 x float> %173
  store <8 x float> %174, ptr %.slot19, align 32
  br label %direct.schedule.12

direct.schedule.12:                               ; preds = %direct.schedule.21, %direct.schedule.11
  %.state89 = load i64, ptr %.slot15, align 4
  %175 = icmp slt i64 %.state89, 1536
  br i1 %175, label %direct.true90, label %direct.false91

direct.schedule.13:                               ; preds = %direct.true90
  %.state92 = load i64, ptr %.slot15, align 4
  %176 = add i64 %.state92, 0
  %177 = sdiv i64 %176, 1
  %178 = srem i64 %177, 1537
  %.spill.load93 = load i64, ptr %.spill6, align 4
  %179 = add i64 %.spill.load93, %178
  store i64 %179, ptr %.spill20, align 4
  %180 = icmp sge i64 %179, 0
  %181 = icmp slt i64 %179, 1537
  %182 = and i1 %180, %181
  br i1 %182, label %direct.true94, label %direct.false95

direct.schedule.14:                               ; preds = %direct.true94
  %.spill.load96 = load i64, ptr %.spill20, align 4
  %183 = srem i64 %.spill.load96, 1537
  %184 = add i64 0, %183
  %tile_snapshot.spill.load97 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %185 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load97, 0
  %186 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load97, 1
  %.splatinsert98 = insertelement <8 x i64> poison, i64 %184, i64 0
  %.splat99 = shufflevector <8 x i64> %.splatinsert98, <8 x i64> poison, <8 x i32> zeroinitializer
  %187 = mul <8 x i64> %.splat99, splat (i64 4)
  %188 = add <8 x i64> %186, %187
  %189 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %185, 0
  %190 = insertvalue { <8 x ptr>, <8 x i64> } %189, <8 x i64> %188, 1
  %191 = extractvalue { <8 x ptr>, <8 x i64> } %190, 0
  %192 = extractvalue { <8 x ptr>, <8 x i64> } %190, 1
  %193 = getelementptr i8, <8 x ptr> %191, <8 x i64> %192
  %194 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %193, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %195 = fmul <8 x float> %194, %194
  %196 = load <8 x float>, ptr %.slot21, align 32
  %197 = select <8 x i1> %51, <8 x float> %195, <8 x float> %196
  store <8 x float> %197, ptr %.slot21, align 32
  br label %direct.schedule.15

direct.schedule.15:                               ; preds = %direct.schedule.14, %direct.false95
  %.state100 = load <8 x float>, ptr %.slot16, align 32
  %.state101 = load <8 x float>, ptr %.slot21, align 32
  %198 = fadd <8 x float> %.state100, %.state101
  %199 = load <8 x float>, ptr %.spill22, align 32
  %200 = select <8 x i1> %51, <8 x float> %198, <8 x float> %199
  store <8 x float> %200, ptr %.spill22, align 32
  %.state102 = load i64, ptr %.slot15, align 4
  %201 = add i64 %.state102, 1
  %202 = sdiv i64 %201, 1
  %203 = srem i64 %202, 1537
  %.spill.load103 = load i64, ptr %.spill6, align 4
  %204 = add i64 %.spill.load103, %203
  store i64 %204, ptr %.spill23, align 4
  %205 = icmp sge i64 %204, 0
  %206 = icmp slt i64 %204, 1537
  %207 = and i1 %205, %206
  br i1 %207, label %direct.true104, label %direct.false105

direct.schedule.16:                               ; preds = %direct.true104
  %.spill.load106 = load i64, ptr %.spill23, align 4
  %208 = srem i64 %.spill.load106, 1537
  %209 = add i64 0, %208
  %tile_snapshot.spill.load107 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %210 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load107, 0
  %211 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load107, 1
  %.splatinsert108 = insertelement <8 x i64> poison, i64 %209, i64 0
  %.splat109 = shufflevector <8 x i64> %.splatinsert108, <8 x i64> poison, <8 x i32> zeroinitializer
  %212 = mul <8 x i64> %.splat109, splat (i64 4)
  %213 = add <8 x i64> %211, %212
  %214 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %210, 0
  %215 = insertvalue { <8 x ptr>, <8 x i64> } %214, <8 x i64> %213, 1
  %216 = extractvalue { <8 x ptr>, <8 x i64> } %215, 0
  %217 = extractvalue { <8 x ptr>, <8 x i64> } %215, 1
  %218 = getelementptr i8, <8 x ptr> %216, <8 x i64> %217
  %219 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %218, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %220 = fmul <8 x float> %219, %219
  %221 = load <8 x float>, ptr %.slot24, align 32
  %222 = select <8 x i1> %51, <8 x float> %220, <8 x float> %221
  store <8 x float> %222, ptr %.slot24, align 32
  br label %direct.schedule.17

direct.schedule.17:                               ; preds = %direct.schedule.16, %direct.false105
  %.state110 = load <8 x float>, ptr %.slot17, align 32
  %.state111 = load <8 x float>, ptr %.slot24, align 32
  %223 = fadd <8 x float> %.state110, %.state111
  %224 = load <8 x float>, ptr %.spill25, align 32
  %225 = select <8 x i1> %51, <8 x float> %223, <8 x float> %224
  store <8 x float> %225, ptr %.spill25, align 32
  %.state112 = load i64, ptr %.slot15, align 4
  %226 = add i64 %.state112, 2
  %227 = sdiv i64 %226, 1
  %228 = srem i64 %227, 1537
  %.spill.load113 = load i64, ptr %.spill6, align 4
  %229 = add i64 %.spill.load113, %228
  store i64 %229, ptr %.spill26, align 4
  %230 = icmp sge i64 %229, 0
  %231 = icmp slt i64 %229, 1537
  %232 = and i1 %230, %231
  br i1 %232, label %direct.true114, label %direct.false115

direct.schedule.18:                               ; preds = %direct.true114
  %.spill.load116 = load i64, ptr %.spill26, align 4
  %233 = srem i64 %.spill.load116, 1537
  %234 = add i64 0, %233
  %tile_snapshot.spill.load117 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %235 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load117, 0
  %236 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load117, 1
  %.splatinsert118 = insertelement <8 x i64> poison, i64 %234, i64 0
  %.splat119 = shufflevector <8 x i64> %.splatinsert118, <8 x i64> poison, <8 x i32> zeroinitializer
  %237 = mul <8 x i64> %.splat119, splat (i64 4)
  %238 = add <8 x i64> %236, %237
  %239 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %235, 0
  %240 = insertvalue { <8 x ptr>, <8 x i64> } %239, <8 x i64> %238, 1
  %241 = extractvalue { <8 x ptr>, <8 x i64> } %240, 0
  %242 = extractvalue { <8 x ptr>, <8 x i64> } %240, 1
  %243 = getelementptr i8, <8 x ptr> %241, <8 x i64> %242
  %244 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %243, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %245 = fmul <8 x float> %244, %244
  %246 = load <8 x float>, ptr %.slot27, align 32
  %247 = select <8 x i1> %51, <8 x float> %245, <8 x float> %246
  store <8 x float> %247, ptr %.slot27, align 32
  br label %direct.schedule.19

direct.schedule.19:                               ; preds = %direct.schedule.18, %direct.false115
  %.state120 = load <8 x float>, ptr %.slot18, align 32
  %.state121 = load <8 x float>, ptr %.slot27, align 32
  %248 = fadd <8 x float> %.state120, %.state121
  %249 = load <8 x float>, ptr %.spill28, align 32
  %250 = select <8 x i1> %51, <8 x float> %248, <8 x float> %249
  store <8 x float> %250, ptr %.spill28, align 32
  %.state122 = load i64, ptr %.slot15, align 4
  %251 = add i64 %.state122, 3
  %252 = sdiv i64 %251, 1
  %253 = srem i64 %252, 1537
  %.spill.load123 = load i64, ptr %.spill6, align 4
  %254 = add i64 %.spill.load123, %253
  store i64 %254, ptr %.spill29, align 4
  %255 = icmp sge i64 %254, 0
  %256 = icmp slt i64 %254, 1537
  %257 = and i1 %255, %256
  br i1 %257, label %direct.true124, label %direct.false125

direct.schedule.20:                               ; preds = %direct.true124
  %.spill.load126 = load i64, ptr %.spill29, align 4
  %258 = srem i64 %.spill.load126, 1537
  %259 = add i64 0, %258
  %tile_snapshot.spill.load127 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %260 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load127, 0
  %261 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load127, 1
  %.splatinsert128 = insertelement <8 x i64> poison, i64 %259, i64 0
  %.splat129 = shufflevector <8 x i64> %.splatinsert128, <8 x i64> poison, <8 x i32> zeroinitializer
  %262 = mul <8 x i64> %.splat129, splat (i64 4)
  %263 = add <8 x i64> %261, %262
  %264 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %260, 0
  %265 = insertvalue { <8 x ptr>, <8 x i64> } %264, <8 x i64> %263, 1
  %266 = extractvalue { <8 x ptr>, <8 x i64> } %265, 0
  %267 = extractvalue { <8 x ptr>, <8 x i64> } %265, 1
  %268 = getelementptr i8, <8 x ptr> %266, <8 x i64> %267
  %269 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %268, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %270 = fmul <8 x float> %269, %269
  %271 = load <8 x float>, ptr %.slot30, align 32
  %272 = select <8 x i1> %51, <8 x float> %270, <8 x float> %271
  store <8 x float> %272, ptr %.slot30, align 32
  br label %direct.schedule.21

direct.schedule.21:                               ; preds = %direct.schedule.20, %direct.false125
  %.state130 = load <8 x float>, ptr %.slot19, align 32
  %.state131 = load <8 x float>, ptr %.slot30, align 32
  %273 = fadd <8 x float> %.state130, %.state131
  %.state132 = load i64, ptr %.slot15, align 4
  %274 = add i64 %.state132, 4
  %.spill.load133 = load <8 x float>, ptr %.spill22, align 32
  %.spill.load134 = load <8 x float>, ptr %.spill25, align 32
  %.spill.load135 = load <8 x float>, ptr %.spill28, align 32
  store i64 %274, ptr %.slot15, align 4
  %275 = load <8 x float>, ptr %.slot16, align 32
  %276 = select <8 x i1> %51, <8 x float> %.spill.load133, <8 x float> %275
  store <8 x float> %276, ptr %.slot16, align 32
  %277 = load <8 x float>, ptr %.slot17, align 32
  %278 = select <8 x i1> %51, <8 x float> %.spill.load134, <8 x float> %277
  store <8 x float> %278, ptr %.slot17, align 32
  %279 = load <8 x float>, ptr %.slot18, align 32
  %280 = select <8 x i1> %51, <8 x float> %.spill.load135, <8 x float> %279
  store <8 x float> %280, ptr %.slot18, align 32
  %281 = load <8 x float>, ptr %.slot19, align 32
  %282 = select <8 x i1> %51, <8 x float> %273, <8 x float> %281
  store <8 x float> %282, ptr %.slot19, align 32
  br label %direct.schedule.12

direct.schedule.22:                               ; preds = %direct.false91
  %.spill.load136 = load i64, ptr %.spill6, align 4
  %283 = add i64 %.spill.load136, 1536
  store i64 %283, ptr %.spill31, align 4
  %284 = icmp sge i64 %283, 0
  %285 = icmp slt i64 %283, 1537
  %286 = and i1 %284, %285
  br i1 %286, label %direct.true137, label %direct.false138

direct.schedule.23:                               ; preds = %direct.true137
  %.spill.load139 = load i64, ptr %.spill31, align 4
  %287 = srem i64 %.spill.load139, 1537
  %288 = add i64 0, %287
  %tile_snapshot.spill.load140 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %289 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load140, 0
  %290 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load140, 1
  %.splatinsert141 = insertelement <8 x i64> poison, i64 %288, i64 0
  %.splat142 = shufflevector <8 x i64> %.splatinsert141, <8 x i64> poison, <8 x i32> zeroinitializer
  %291 = mul <8 x i64> %.splat142, splat (i64 4)
  %292 = add <8 x i64> %290, %291
  %293 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %289, 0
  %294 = insertvalue { <8 x ptr>, <8 x i64> } %293, <8 x i64> %292, 1
  %295 = extractvalue { <8 x ptr>, <8 x i64> } %294, 0
  %296 = extractvalue { <8 x ptr>, <8 x i64> } %294, 1
  %297 = getelementptr i8, <8 x ptr> %295, <8 x i64> %296
  %298 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %297, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %299 = fmul <8 x float> %298, %298
  %300 = load <8 x float>, ptr %.slot32, align 32
  %301 = select <8 x i1> %51, <8 x float> %299, <8 x float> %300
  store <8 x float> %301, ptr %.slot32, align 32
  br label %direct.schedule.24

direct.schedule.24:                               ; preds = %direct.schedule.23, %direct.false138
  %.state143 = load <8 x float>, ptr %.slot16, align 32
  %.state144 = load <8 x float>, ptr %.slot32, align 32
  %302 = fadd <8 x float> %.state143, %.state144
  %.spill.load145 = load float, ptr %.spill4, align 4
  %.splatinsert146 = insertelement <8 x float> poison, float %.spill.load145, i64 0
  %.splat147 = shufflevector <8 x float> %.splatinsert146, <8 x float> poison, <8 x i32> zeroinitializer
  %303 = fadd <8 x float> %.splat147, %302
  %.state148 = load <8 x float>, ptr %.slot17, align 32
  %304 = fadd <8 x float> %303, %.state148
  %.state149 = load <8 x float>, ptr %.slot18, align 32
  %305 = fadd <8 x float> %304, %.state149
  %.state150 = load <8 x float>, ptr %.slot19, align 32
  %306 = fadd <8 x float> %305, %.state150
  %307 = fdiv <8 x float> %306, splat (float 1.537000e+03)
  %308 = load <8 x float>, ptr %.spill33, align 32
  %309 = select <8 x i1> %51, <8 x float> %307, <8 x float> %308
  store <8 x float> %309, ptr %.spill33, align 32
  %310 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill34, align 64
  %311 = extractvalue { <8 x ptr>, <8 x i64> } %3, 0
  %312 = extractvalue { <8 x ptr>, <8 x i64> } %310, 0
  %313 = select <8 x i1> %51, <8 x ptr> %311, <8 x ptr> %312
  %314 = extractvalue { <8 x ptr>, <8 x i64> } %3, 1
  %315 = extractvalue { <8 x ptr>, <8 x i64> } %310, 1
  %316 = select <8 x i1> %51, <8 x i64> %314, <8 x i64> %315
  %317 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %313, 0
  %318 = insertvalue { <8 x ptr>, <8 x i64> } %317, <8 x i64> %316, 1
  store { <8 x ptr>, <8 x i64> } %318, ptr %tile_snapshot.spill34, align 64
  store i64 0, ptr %.slot35, align 4
  br label %direct.schedule.25

direct.schedule.25:                               ; preds = %direct.schedule.26, %direct.schedule.24
  %.state151 = load i64, ptr %.slot35, align 4
  %319 = icmp slt i64 %.state151, 1537
  br i1 %319, label %direct.true152, label %direct.false153

direct.schedule.26:                               ; preds = %direct.true152
  %.state154 = load i64, ptr %.slot35, align 4
  %320 = srem i64 %.state154, 1537
  %.state155 = load i64, ptr %.slot35, align 4
  %321 = sdiv i64 %.state155, 1537
  %322 = srem i64 %321, 1
  %323 = add i64 0, %322
  %.spill.load156 = load i64, ptr %.spill5, align 4
  %324 = add i64 %.spill.load156, %323
  %325 = add i64 0, %320
  %326 = mul i64 %324, 1537
  %327 = add i64 %326, %325
  %328 = extractvalue { ptr, i64 } %15, 0
  %329 = mul i64 %327, 4
  %330 = getelementptr i8, ptr %328, i64 %329
  %331 = load float, ptr %330, align 4
  %.splatinsert157 = insertelement <8 x float> poison, float %331, i64 0
  %.splat158 = shufflevector <8 x float> %.splatinsert157, <8 x float> poison, <8 x i32> zeroinitializer
  %tile_snapshot.spill.load159 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill34, align 64
  %332 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load159, 0
  %333 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load159, 1
  %.state160 = load i64, ptr %.slot35, align 4
  %.splatinsert161 = insertelement <8 x i64> poison, i64 %.state160, i64 0
  %.splat162 = shufflevector <8 x i64> %.splatinsert161, <8 x i64> poison, <8 x i32> zeroinitializer
  %334 = mul <8 x i64> %.splat162, splat (i64 4)
  %335 = add <8 x i64> %333, %334
  %336 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %332, 0
  %337 = insertvalue { <8 x ptr>, <8 x i64> } %336, <8 x i64> %335, 1
  %338 = extractvalue { <8 x ptr>, <8 x i64> } %337, 0
  %339 = extractvalue { <8 x ptr>, <8 x i64> } %337, 1
  %340 = getelementptr i8, <8 x ptr> %338, <8 x i64> %339
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %.splat158, <8 x ptr> %340, i32 1, <8 x i1> %51)
  %.state163 = load i64, ptr %.slot35, align 4
  %341 = add i64 %.state163, 1
  store i64 %341, ptr %.slot35, align 4
  br label %direct.schedule.25

direct.schedule.27:                               ; preds = %direct.false153
  %.spill.load164 = load <8 x float>, ptr %.spill33, align 32
  %342 = fadd <8 x float> %.spill.load164, splat (float 0x3EE4F8B580000000)
  %343 = call <8 x float> @llvm.sqrt.v8f32(<8 x float> %342)
  %344 = load <8 x float>, ptr %.spill36, align 32
  %345 = select <8 x i1> %51, <8 x float> %343, <8 x float> %344
  store <8 x float> %345, ptr %.spill36, align 32
  store i64 0, ptr %.slot37, align 4
  br label %direct.schedule.28

direct.schedule.28:                               ; preds = %direct.schedule.29, %direct.schedule.27
  %.state165 = load i64, ptr %.slot37, align 4
  %346 = icmp slt i64 %.state165, 1537
  br i1 %346, label %direct.true166, label %direct.false167

direct.schedule.29:                               ; preds = %direct.true166
  %.state168 = load i64, ptr %.slot37, align 4
  %347 = srem i64 %.state168, 1537
  %.state169 = load i64, ptr %.slot37, align 4
  %348 = sdiv i64 %.state169, 1537
  %349 = srem i64 %348, 1
  %.spill.load170 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert171 = insertelement <8 x i64> poison, i64 %349, i64 0
  %.splat172 = shufflevector <8 x i64> %.splatinsert171, <8 x i64> poison, <8 x i32> zeroinitializer
  %350 = add <8 x i64> %.spill.load170, %.splat172
  %351 = add <8 x i64> zeroinitializer, %350
  %352 = add i64 0, %347
  %353 = mul <8 x i64> %351, splat (i64 1537)
  %.splatinsert173 = insertelement <8 x i64> poison, i64 %352, i64 0
  %.splat174 = shufflevector <8 x i64> %.splatinsert173, <8 x i64> poison, <8 x i32> zeroinitializer
  %354 = add <8 x i64> %353, %.splat174
  %355 = add i64 0, %347
  %356 = srem i64 %355, 1537
  %357 = add i64 0, %356
  %tile_snapshot.spill.load175 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %358 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load175, 0
  %359 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load175, 1
  %.splatinsert176 = insertelement <8 x i64> poison, i64 %357, i64 0
  %.splat177 = shufflevector <8 x i64> %.splatinsert176, <8 x i64> poison, <8 x i32> zeroinitializer
  %360 = mul <8 x i64> %.splat177, splat (i64 4)
  %361 = add <8 x i64> %359, %360
  %362 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %358, 0
  %363 = insertvalue { <8 x ptr>, <8 x i64> } %362, <8 x i64> %361, 1
  %364 = extractvalue { <8 x ptr>, <8 x i64> } %363, 0
  %365 = extractvalue { <8 x ptr>, <8 x i64> } %363, 1
  %366 = getelementptr i8, <8 x ptr> %364, <8 x i64> %365
  %367 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %366, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %.spill.load178 = load <8 x float>, ptr %.spill36, align 32
  %368 = fdiv <8 x float> %367, %.spill.load178
  %tile_snapshot.spill.load179 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill34, align 64
  %369 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load179, 0
  %370 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load179, 1
  %.splatinsert180 = insertelement <8 x i64> poison, i64 %355, i64 0
  %.splat181 = shufflevector <8 x i64> %.splatinsert180, <8 x i64> poison, <8 x i32> zeroinitializer
  %371 = mul <8 x i64> %.splat181, splat (i64 4)
  %372 = add <8 x i64> %370, %371
  %373 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %369, 0
  %374 = insertvalue { <8 x ptr>, <8 x i64> } %373, <8 x i64> %372, 1
  %375 = extractvalue { <8 x ptr>, <8 x i64> } %374, 0
  %376 = extractvalue { <8 x ptr>, <8 x i64> } %374, 1
  %377 = getelementptr i8, <8 x ptr> %375, <8 x i64> %376
  %378 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %377, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %379 = fmul <8 x float> %368, %378
  %380 = extractvalue { ptr, i64 } %27, 0
  %381 = mul <8 x i64> %354, splat (i64 4)
  %382 = getelementptr i8, ptr %380, <8 x i64> %381
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %379, <8 x ptr> %382, i32 1, <8 x i1> %51)
  %.state182 = load i64, ptr %.slot37, align 4
  %383 = add i64 %.state182, 1
  store i64 %383, ptr %.slot37, align 4
  br label %direct.schedule.28

direct.schedule.30:                               ; preds = %direct.false167
  ret void

direct.activate:                                  ; preds = %prologue
  br label %direct.schedule.0

direct.inactive:                                  ; preds = %prologue
  ret void

direct.true:                                      ; preds = %direct.schedule.1
  br label %direct.schedule.2

direct.false:                                     ; preds = %direct.schedule.1
  br label %direct.schedule.3

direct.true58:                                    ; preds = %direct.schedule.3
  br label %direct.schedule.4

direct.false59:                                   ; preds = %direct.schedule.3
  %384 = load <8 x float>, ptr %.slot8, align 32
  %385 = select <8 x i1> %51, <8 x float> zeroinitializer, <8 x float> %384
  store <8 x float> %385, ptr %.slot8, align 32
  br label %direct.schedule.5

direct.true65:                                    ; preds = %direct.schedule.5
  br label %direct.schedule.6

direct.false66:                                   ; preds = %direct.schedule.5
  %386 = load <8 x float>, ptr %.slot10, align 32
  %387 = select <8 x i1> %51, <8 x float> zeroinitializer, <8 x float> %386
  store <8 x float> %387, ptr %.slot10, align 32
  br label %direct.schedule.7

direct.true72:                                    ; preds = %direct.schedule.7
  br label %direct.schedule.8

direct.false73:                                   ; preds = %direct.schedule.7
  %388 = load <8 x float>, ptr %.slot12, align 32
  %389 = select <8 x i1> %51, <8 x float> zeroinitializer, <8 x float> %388
  store <8 x float> %389, ptr %.slot12, align 32
  br label %direct.schedule.9

direct.true79:                                    ; preds = %direct.schedule.9
  br label %direct.schedule.10

direct.false80:                                   ; preds = %direct.schedule.9
  %390 = load <8 x float>, ptr %.slot14, align 32
  %391 = select <8 x i1> %51, <8 x float> zeroinitializer, <8 x float> %390
  store <8 x float> %391, ptr %.slot14, align 32
  br label %direct.schedule.11

direct.true90:                                    ; preds = %direct.schedule.12
  br label %direct.schedule.13

direct.false91:                                   ; preds = %direct.schedule.12
  br label %direct.schedule.22

direct.true94:                                    ; preds = %direct.schedule.13
  br label %direct.schedule.14

direct.false95:                                   ; preds = %direct.schedule.13
  %392 = load <8 x float>, ptr %.slot21, align 32
  %393 = select <8 x i1> %51, <8 x float> zeroinitializer, <8 x float> %392
  store <8 x float> %393, ptr %.slot21, align 32
  br label %direct.schedule.15

direct.true104:                                   ; preds = %direct.schedule.15
  br label %direct.schedule.16

direct.false105:                                  ; preds = %direct.schedule.15
  %394 = load <8 x float>, ptr %.slot24, align 32
  %395 = select <8 x i1> %51, <8 x float> zeroinitializer, <8 x float> %394
  store <8 x float> %395, ptr %.slot24, align 32
  br label %direct.schedule.17

direct.true114:                                   ; preds = %direct.schedule.17
  br label %direct.schedule.18

direct.false115:                                  ; preds = %direct.schedule.17
  %396 = load <8 x float>, ptr %.slot27, align 32
  %397 = select <8 x i1> %51, <8 x float> zeroinitializer, <8 x float> %396
  store <8 x float> %397, ptr %.slot27, align 32
  br label %direct.schedule.19

direct.true124:                                   ; preds = %direct.schedule.19
  br label %direct.schedule.20

direct.false125:                                  ; preds = %direct.schedule.19
  %398 = load <8 x float>, ptr %.slot30, align 32
  %399 = select <8 x i1> %51, <8 x float> zeroinitializer, <8 x float> %398
  store <8 x float> %399, ptr %.slot30, align 32
  br label %direct.schedule.21

direct.true137:                                   ; preds = %direct.schedule.22
  br label %direct.schedule.23

direct.false138:                                  ; preds = %direct.schedule.22
  %400 = load <8 x float>, ptr %.slot32, align 32
  %401 = select <8 x i1> %51, <8 x float> zeroinitializer, <8 x float> %400
  store <8 x float> %401, ptr %.slot32, align 32
  br label %direct.schedule.24

direct.true152:                                   ; preds = %direct.schedule.25
  br label %direct.schedule.26

direct.false153:                                  ; preds = %direct.schedule.25
  br label %direct.schedule.27

direct.true166:                                   ; preds = %direct.schedule.28
  br label %direct.schedule.29

direct.false167:                                  ; preds = %direct.schedule.28
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
