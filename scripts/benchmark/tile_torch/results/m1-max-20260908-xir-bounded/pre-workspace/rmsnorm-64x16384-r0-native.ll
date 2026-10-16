; ModuleID = 'luisa-simd-kernel'
source_filename = "luisa-simd-kernel"

define internal void @llm_rows(ptr noalias readonly %argument_buffer, ptr %return_lanes, ptr noalias nonnull readonly %launch_config, i32 %active_lane_count) {
prologue:
  %tile_snapshot.local = alloca [524288 x i8], align 4
  %.splatinsert = insertelement <8 x ptr> poison, ptr %tile_snapshot.local, i64 0
  %.splat = shufflevector <8 x ptr> %.splatinsert, <8 x ptr> poison, <8 x i32> zeroinitializer
  %0 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat, 0
  %1 = insertvalue { <8 x ptr>, <8 x i64> } %0, <8 x i64> <i64 0, i64 65536, i64 131072, i64 196608, i64 262144, i64 327680, i64 393216, i64 458752>, 1
  %tile_snapshot.local1 = alloca [524288 x i8], align 4
  %.splatinsert2 = insertelement <8 x ptr> poison, ptr %tile_snapshot.local1, i64 0
  %.splat3 = shufflevector <8 x ptr> %.splatinsert2, <8 x ptr> poison, <8 x i32> zeroinitializer
  %2 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %.splat3, 0
  %3 = insertvalue { <8 x ptr>, <8 x i64> } %2, <8 x i64> <i64 0, i64 65536, i64 131072, i64 196608, i64 262144, i64 327680, i64 393216, i64 458752>, 1
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
  %.spill31 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill31, align 32
  %tile_snapshot.spill32 = alloca { <8 x ptr>, <8 x i64> }, align 64
  store { <8 x ptr>, <8 x i64> } zeroinitializer, ptr %tile_snapshot.spill32, align 64
  %.slot33 = alloca i64, align 8
  store i64 0, ptr %.slot33, align 4
  %.spill34 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill34, align 32
  %.slot35 = alloca i64, align 8
  store i64 0, ptr %.slot35, align 4
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
  %.splatinsert36 = insertelement <8 x i32> poison, i32 %40, i64 0
  %.splat37 = shufflevector <8 x i32> %.splatinsert36, <8 x i32> poison, <8 x i32> zeroinitializer
  %41 = add <8 x i32> %.splat37, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %42 = mul i32 %28, 32
  %.splatinsert38 = insertelement <8 x i32> poison, i32 %42, i64 0
  %.splat39 = shufflevector <8 x i32> %.splatinsert38, <8 x i32> poison, <8 x i32> zeroinitializer
  %43 = add <8 x i32> %.splat39, %41
  %44 = mul i32 %32, 1
  %.splatinsert40 = insertelement <8 x i32> poison, i32 %44, i64 0
  %.splat41 = shufflevector <8 x i32> %.splatinsert40, <8 x i32> poison, <8 x i32> zeroinitializer
  %45 = add <8 x i32> %.splat41, zeroinitializer
  %46 = mul i32 %36, 1
  %.splatinsert42 = insertelement <8 x i32> poison, i32 %46, i64 0
  %.splat43 = shufflevector <8 x i32> %.splatinsert42, <8 x i32> poison, <8 x i32> zeroinitializer
  %47 = add <8 x i32> %.splat43, zeroinitializer
  %48 = insertvalue [3 x <8 x i32>] poison, <8 x i32> %43, 0
  %49 = insertvalue [3 x <8 x i32>] %48, <8 x i32> %45, 1
  %50 = insertvalue [3 x <8 x i32>] %49, <8 x i32> %47, 2
  %.splatinsert44 = insertelement <8 x i32> poison, i32 %active_lane_count, i64 0
  %.splat45 = shufflevector <8 x i32> %.splatinsert44, <8 x i32> poison, <8 x i32> zeroinitializer
  %51 = icmp ult <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, %.splat45
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
  %72 = icmp slt i64 %.state, 16384
  br i1 %72, label %direct.true, label %direct.false

direct.schedule.2:                                ; preds = %direct.true
  %.state46 = load i64, ptr %.slot, align 4
  %73 = srem i64 %.state46, 16384
  %.state47 = load i64, ptr %.slot, align 4
  %74 = sdiv i64 %.state47, 16384
  %75 = srem i64 %74, 1
  %.spill.load = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert48 = insertelement <8 x i64> poison, i64 %75, i64 0
  %.splat49 = shufflevector <8 x i64> %.splatinsert48, <8 x i64> poison, <8 x i32> zeroinitializer
  %76 = add <8 x i64> %.spill.load, %.splat49
  %77 = add <8 x i64> zeroinitializer, %76
  %78 = add i64 0, %73
  %79 = mul <8 x i64> %77, splat (i64 16384)
  %.splatinsert50 = insertelement <8 x i64> poison, i64 %78, i64 0
  %.splat51 = shufflevector <8 x i64> %.splatinsert50, <8 x i64> poison, <8 x i32> zeroinitializer
  %80 = add <8 x i64> %79, %.splat51
  %81 = extractvalue { ptr, i64 } %9, 0
  %82 = mul <8 x i64> %80, splat (i64 4)
  %83 = getelementptr i8, ptr %81, <8 x i64> %82
  %84 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %83, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %tile_snapshot.spill.load = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %85 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 0
  %86 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load, 1
  %.state52 = load i64, ptr %.slot, align 4
  %.splatinsert53 = insertelement <8 x i64> poison, i64 %.state52, i64 0
  %.splat54 = shufflevector <8 x i64> %.splatinsert53, <8 x i64> poison, <8 x i32> zeroinitializer
  %87 = mul <8 x i64> %.splat54, splat (i64 4)
  %88 = add <8 x i64> %86, %87
  %89 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %85, 0
  %90 = insertvalue { <8 x ptr>, <8 x i64> } %89, <8 x i64> %88, 1
  %91 = extractvalue { <8 x ptr>, <8 x i64> } %90, 0
  %92 = extractvalue { <8 x ptr>, <8 x i64> } %90, 1
  %93 = getelementptr i8, <8 x ptr> %91, <8 x i64> %92
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %84, <8 x ptr> %93, i32 1, <8 x i1> %51)
  %.state55 = load i64, ptr %.slot, align 4
  %94 = add i64 %.state55, 1
  store i64 %94, ptr %.slot, align 4
  br label %direct.schedule.1

direct.schedule.3:                                ; preds = %direct.false
  store float 0.000000e+00, ptr %.spill4, align 4
  store i64 0, ptr %.spill5, align 4
  store i64 0, ptr %.spill6, align 4
  store i64 0, ptr %.spill7, align 4
  br i1 true, label %direct.true56, label %direct.false57

direct.schedule.4:                                ; preds = %direct.true56
  %.spill.load58 = load i64, ptr %.spill7, align 4
  %95 = srem i64 %.spill.load58, 16384
  %96 = add i64 0, %95
  %tile_snapshot.spill.load59 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %97 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load59, 0
  %98 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load59, 1
  %.splatinsert60 = insertelement <8 x i64> poison, i64 %96, i64 0
  %.splat61 = shufflevector <8 x i64> %.splatinsert60, <8 x i64> poison, <8 x i32> zeroinitializer
  %99 = mul <8 x i64> %.splat61, splat (i64 4)
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

direct.schedule.5:                                ; preds = %direct.schedule.4, %direct.false57
  %.spill.load62 = load i64, ptr %.spill6, align 4
  %110 = add i64 %.spill.load62, 1
  store i64 %110, ptr %.spill9, align 4
  %111 = icmp sge i64 %110, 0
  %112 = icmp slt i64 %110, 16384
  %113 = and i1 %111, %112
  br i1 %113, label %direct.true63, label %direct.false64

direct.schedule.6:                                ; preds = %direct.true63
  %.spill.load65 = load i64, ptr %.spill9, align 4
  %114 = srem i64 %.spill.load65, 16384
  %115 = add i64 0, %114
  %tile_snapshot.spill.load66 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %116 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load66, 0
  %117 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load66, 1
  %.splatinsert67 = insertelement <8 x i64> poison, i64 %115, i64 0
  %.splat68 = shufflevector <8 x i64> %.splatinsert67, <8 x i64> poison, <8 x i32> zeroinitializer
  %118 = mul <8 x i64> %.splat68, splat (i64 4)
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

direct.schedule.7:                                ; preds = %direct.schedule.6, %direct.false64
  %.spill.load69 = load i64, ptr %.spill6, align 4
  %129 = add i64 %.spill.load69, 2
  store i64 %129, ptr %.spill11, align 4
  %130 = icmp sge i64 %129, 0
  %131 = icmp slt i64 %129, 16384
  %132 = and i1 %130, %131
  br i1 %132, label %direct.true70, label %direct.false71

direct.schedule.8:                                ; preds = %direct.true70
  %.spill.load72 = load i64, ptr %.spill11, align 4
  %133 = srem i64 %.spill.load72, 16384
  %134 = add i64 0, %133
  %tile_snapshot.spill.load73 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %135 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load73, 0
  %136 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load73, 1
  %.splatinsert74 = insertelement <8 x i64> poison, i64 %134, i64 0
  %.splat75 = shufflevector <8 x i64> %.splatinsert74, <8 x i64> poison, <8 x i32> zeroinitializer
  %137 = mul <8 x i64> %.splat75, splat (i64 4)
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

direct.schedule.9:                                ; preds = %direct.schedule.8, %direct.false71
  %.spill.load76 = load i64, ptr %.spill6, align 4
  %148 = add i64 %.spill.load76, 3
  store i64 %148, ptr %.spill13, align 4
  %149 = icmp sge i64 %148, 0
  %150 = icmp slt i64 %148, 16384
  %151 = and i1 %149, %150
  br i1 %151, label %direct.true77, label %direct.false78

direct.schedule.10:                               ; preds = %direct.true77
  %.spill.load79 = load i64, ptr %.spill13, align 4
  %152 = srem i64 %.spill.load79, 16384
  %153 = add i64 0, %152
  %tile_snapshot.spill.load80 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %154 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load80, 0
  %155 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load80, 1
  %.splatinsert81 = insertelement <8 x i64> poison, i64 %153, i64 0
  %.splat82 = shufflevector <8 x i64> %.splatinsert81, <8 x i64> poison, <8 x i32> zeroinitializer
  %156 = mul <8 x i64> %.splat82, splat (i64 4)
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

direct.schedule.11:                               ; preds = %direct.schedule.10, %direct.false78
  %.state83 = load <8 x float>, ptr %.slot8, align 32
  %.state84 = load <8 x float>, ptr %.slot10, align 32
  %.state85 = load <8 x float>, ptr %.slot12, align 32
  %.state86 = load <8 x float>, ptr %.slot14, align 32
  store i64 4, ptr %.slot15, align 4
  %167 = load <8 x float>, ptr %.slot16, align 32
  %168 = select <8 x i1> %51, <8 x float> %.state83, <8 x float> %167
  store <8 x float> %168, ptr %.slot16, align 32
  %169 = load <8 x float>, ptr %.slot17, align 32
  %170 = select <8 x i1> %51, <8 x float> %.state84, <8 x float> %169
  store <8 x float> %170, ptr %.slot17, align 32
  %171 = load <8 x float>, ptr %.slot18, align 32
  %172 = select <8 x i1> %51, <8 x float> %.state85, <8 x float> %171
  store <8 x float> %172, ptr %.slot18, align 32
  %173 = load <8 x float>, ptr %.slot19, align 32
  %174 = select <8 x i1> %51, <8 x float> %.state86, <8 x float> %173
  store <8 x float> %174, ptr %.slot19, align 32
  br label %direct.schedule.12

direct.schedule.12:                               ; preds = %direct.schedule.21, %direct.schedule.11
  %.state87 = load i64, ptr %.slot15, align 4
  %175 = icmp slt i64 %.state87, 16384
  br i1 %175, label %direct.true88, label %direct.false89

direct.schedule.13:                               ; preds = %direct.true88
  %.state90 = load i64, ptr %.slot15, align 4
  %176 = add i64 %.state90, 0
  %177 = sdiv i64 %176, 1
  %178 = srem i64 %177, 16384
  %.spill.load91 = load i64, ptr %.spill6, align 4
  %179 = add i64 %.spill.load91, %178
  store i64 %179, ptr %.spill20, align 4
  %180 = icmp sge i64 %179, 0
  %181 = icmp slt i64 %179, 16384
  %182 = and i1 %180, %181
  br i1 %182, label %direct.true92, label %direct.false93

direct.schedule.14:                               ; preds = %direct.true92
  %.spill.load94 = load i64, ptr %.spill20, align 4
  %183 = srem i64 %.spill.load94, 16384
  %184 = add i64 0, %183
  %tile_snapshot.spill.load95 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %185 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load95, 0
  %186 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load95, 1
  %.splatinsert96 = insertelement <8 x i64> poison, i64 %184, i64 0
  %.splat97 = shufflevector <8 x i64> %.splatinsert96, <8 x i64> poison, <8 x i32> zeroinitializer
  %187 = mul <8 x i64> %.splat97, splat (i64 4)
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

direct.schedule.15:                               ; preds = %direct.schedule.14, %direct.false93
  %.state98 = load <8 x float>, ptr %.slot16, align 32
  %.state99 = load <8 x float>, ptr %.slot21, align 32
  %198 = fadd <8 x float> %.state98, %.state99
  %199 = load <8 x float>, ptr %.spill22, align 32
  %200 = select <8 x i1> %51, <8 x float> %198, <8 x float> %199
  store <8 x float> %200, ptr %.spill22, align 32
  %.state100 = load i64, ptr %.slot15, align 4
  %201 = add i64 %.state100, 1
  %202 = sdiv i64 %201, 1
  %203 = srem i64 %202, 16384
  %.spill.load101 = load i64, ptr %.spill6, align 4
  %204 = add i64 %.spill.load101, %203
  store i64 %204, ptr %.spill23, align 4
  %205 = icmp sge i64 %204, 0
  %206 = icmp slt i64 %204, 16384
  %207 = and i1 %205, %206
  br i1 %207, label %direct.true102, label %direct.false103

direct.schedule.16:                               ; preds = %direct.true102
  %.spill.load104 = load i64, ptr %.spill23, align 4
  %208 = srem i64 %.spill.load104, 16384
  %209 = add i64 0, %208
  %tile_snapshot.spill.load105 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %210 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load105, 0
  %211 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load105, 1
  %.splatinsert106 = insertelement <8 x i64> poison, i64 %209, i64 0
  %.splat107 = shufflevector <8 x i64> %.splatinsert106, <8 x i64> poison, <8 x i32> zeroinitializer
  %212 = mul <8 x i64> %.splat107, splat (i64 4)
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

direct.schedule.17:                               ; preds = %direct.schedule.16, %direct.false103
  %.state108 = load <8 x float>, ptr %.slot17, align 32
  %.state109 = load <8 x float>, ptr %.slot24, align 32
  %223 = fadd <8 x float> %.state108, %.state109
  %224 = load <8 x float>, ptr %.spill25, align 32
  %225 = select <8 x i1> %51, <8 x float> %223, <8 x float> %224
  store <8 x float> %225, ptr %.spill25, align 32
  %.state110 = load i64, ptr %.slot15, align 4
  %226 = add i64 %.state110, 2
  %227 = sdiv i64 %226, 1
  %228 = srem i64 %227, 16384
  %.spill.load111 = load i64, ptr %.spill6, align 4
  %229 = add i64 %.spill.load111, %228
  store i64 %229, ptr %.spill26, align 4
  %230 = icmp sge i64 %229, 0
  %231 = icmp slt i64 %229, 16384
  %232 = and i1 %230, %231
  br i1 %232, label %direct.true112, label %direct.false113

direct.schedule.18:                               ; preds = %direct.true112
  %.spill.load114 = load i64, ptr %.spill26, align 4
  %233 = srem i64 %.spill.load114, 16384
  %234 = add i64 0, %233
  %tile_snapshot.spill.load115 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %235 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load115, 0
  %236 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load115, 1
  %.splatinsert116 = insertelement <8 x i64> poison, i64 %234, i64 0
  %.splat117 = shufflevector <8 x i64> %.splatinsert116, <8 x i64> poison, <8 x i32> zeroinitializer
  %237 = mul <8 x i64> %.splat117, splat (i64 4)
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

direct.schedule.19:                               ; preds = %direct.schedule.18, %direct.false113
  %.state118 = load <8 x float>, ptr %.slot18, align 32
  %.state119 = load <8 x float>, ptr %.slot27, align 32
  %248 = fadd <8 x float> %.state118, %.state119
  %249 = load <8 x float>, ptr %.spill28, align 32
  %250 = select <8 x i1> %51, <8 x float> %248, <8 x float> %249
  store <8 x float> %250, ptr %.spill28, align 32
  %.state120 = load i64, ptr %.slot15, align 4
  %251 = add i64 %.state120, 3
  %252 = sdiv i64 %251, 1
  %253 = srem i64 %252, 16384
  %.spill.load121 = load i64, ptr %.spill6, align 4
  %254 = add i64 %.spill.load121, %253
  store i64 %254, ptr %.spill29, align 4
  %255 = icmp sge i64 %254, 0
  %256 = icmp slt i64 %254, 16384
  %257 = and i1 %255, %256
  br i1 %257, label %direct.true122, label %direct.false123

direct.schedule.20:                               ; preds = %direct.true122
  %.spill.load124 = load i64, ptr %.spill29, align 4
  %258 = srem i64 %.spill.load124, 16384
  %259 = add i64 0, %258
  %tile_snapshot.spill.load125 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %260 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load125, 0
  %261 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load125, 1
  %.splatinsert126 = insertelement <8 x i64> poison, i64 %259, i64 0
  %.splat127 = shufflevector <8 x i64> %.splatinsert126, <8 x i64> poison, <8 x i32> zeroinitializer
  %262 = mul <8 x i64> %.splat127, splat (i64 4)
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

direct.schedule.21:                               ; preds = %direct.schedule.20, %direct.false123
  %.state128 = load <8 x float>, ptr %.slot19, align 32
  %.state129 = load <8 x float>, ptr %.slot30, align 32
  %273 = fadd <8 x float> %.state128, %.state129
  %.state130 = load i64, ptr %.slot15, align 4
  %274 = add i64 %.state130, 4
  %.spill.load131 = load <8 x float>, ptr %.spill22, align 32
  %.spill.load132 = load <8 x float>, ptr %.spill25, align 32
  %.spill.load133 = load <8 x float>, ptr %.spill28, align 32
  store i64 %274, ptr %.slot15, align 4
  %275 = load <8 x float>, ptr %.slot16, align 32
  %276 = select <8 x i1> %51, <8 x float> %.spill.load131, <8 x float> %275
  store <8 x float> %276, ptr %.slot16, align 32
  %277 = load <8 x float>, ptr %.slot17, align 32
  %278 = select <8 x i1> %51, <8 x float> %.spill.load132, <8 x float> %277
  store <8 x float> %278, ptr %.slot17, align 32
  %279 = load <8 x float>, ptr %.slot18, align 32
  %280 = select <8 x i1> %51, <8 x float> %.spill.load133, <8 x float> %279
  store <8 x float> %280, ptr %.slot18, align 32
  %281 = load <8 x float>, ptr %.slot19, align 32
  %282 = select <8 x i1> %51, <8 x float> %273, <8 x float> %281
  store <8 x float> %282, ptr %.slot19, align 32
  br label %direct.schedule.12

direct.schedule.22:                               ; preds = %direct.false89
  %.spill.load134 = load float, ptr %.spill4, align 4
  %.splatinsert135 = insertelement <8 x float> poison, float %.spill.load134, i64 0
  %.splat136 = shufflevector <8 x float> %.splatinsert135, <8 x float> poison, <8 x i32> zeroinitializer
  %.state137 = load <8 x float>, ptr %.slot16, align 32
  %283 = fadd <8 x float> %.splat136, %.state137
  %.state138 = load <8 x float>, ptr %.slot17, align 32
  %284 = fadd <8 x float> %283, %.state138
  %.state139 = load <8 x float>, ptr %.slot18, align 32
  %285 = fadd <8 x float> %284, %.state139
  %.state140 = load <8 x float>, ptr %.slot19, align 32
  %286 = fadd <8 x float> %285, %.state140
  %287 = fdiv <8 x float> %286, splat (float 1.638400e+04)
  %288 = load <8 x float>, ptr %.spill31, align 32
  %289 = select <8 x i1> %51, <8 x float> %287, <8 x float> %288
  store <8 x float> %289, ptr %.spill31, align 32
  %290 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill32, align 64
  %291 = extractvalue { <8 x ptr>, <8 x i64> } %3, 0
  %292 = extractvalue { <8 x ptr>, <8 x i64> } %290, 0
  %293 = select <8 x i1> %51, <8 x ptr> %291, <8 x ptr> %292
  %294 = extractvalue { <8 x ptr>, <8 x i64> } %3, 1
  %295 = extractvalue { <8 x ptr>, <8 x i64> } %290, 1
  %296 = select <8 x i1> %51, <8 x i64> %294, <8 x i64> %295
  %297 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %293, 0
  %298 = insertvalue { <8 x ptr>, <8 x i64> } %297, <8 x i64> %296, 1
  store { <8 x ptr>, <8 x i64> } %298, ptr %tile_snapshot.spill32, align 64
  store i64 0, ptr %.slot33, align 4
  br label %direct.schedule.23

direct.schedule.23:                               ; preds = %direct.schedule.24, %direct.schedule.22
  %.state141 = load i64, ptr %.slot33, align 4
  %299 = icmp slt i64 %.state141, 16384
  br i1 %299, label %direct.true142, label %direct.false143

direct.schedule.24:                               ; preds = %direct.true142
  %.state144 = load i64, ptr %.slot33, align 4
  %300 = srem i64 %.state144, 16384
  %.state145 = load i64, ptr %.slot33, align 4
  %301 = sdiv i64 %.state145, 16384
  %302 = srem i64 %301, 1
  %303 = add i64 0, %302
  %.spill.load146 = load i64, ptr %.spill5, align 4
  %304 = add i64 %.spill.load146, %303
  %305 = add i64 0, %300
  %306 = mul i64 %304, 16384
  %307 = add i64 %306, %305
  %308 = extractvalue { ptr, i64 } %15, 0
  %309 = mul i64 %307, 4
  %310 = getelementptr i8, ptr %308, i64 %309
  %311 = load float, ptr %310, align 4
  %.splatinsert147 = insertelement <8 x float> poison, float %311, i64 0
  %.splat148 = shufflevector <8 x float> %.splatinsert147, <8 x float> poison, <8 x i32> zeroinitializer
  %tile_snapshot.spill.load149 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill32, align 64
  %312 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load149, 0
  %313 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load149, 1
  %.state150 = load i64, ptr %.slot33, align 4
  %.splatinsert151 = insertelement <8 x i64> poison, i64 %.state150, i64 0
  %.splat152 = shufflevector <8 x i64> %.splatinsert151, <8 x i64> poison, <8 x i32> zeroinitializer
  %314 = mul <8 x i64> %.splat152, splat (i64 4)
  %315 = add <8 x i64> %313, %314
  %316 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %312, 0
  %317 = insertvalue { <8 x ptr>, <8 x i64> } %316, <8 x i64> %315, 1
  %318 = extractvalue { <8 x ptr>, <8 x i64> } %317, 0
  %319 = extractvalue { <8 x ptr>, <8 x i64> } %317, 1
  %320 = getelementptr i8, <8 x ptr> %318, <8 x i64> %319
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %.splat148, <8 x ptr> %320, i32 1, <8 x i1> %51)
  %.state153 = load i64, ptr %.slot33, align 4
  %321 = add i64 %.state153, 1
  store i64 %321, ptr %.slot33, align 4
  br label %direct.schedule.23

direct.schedule.25:                               ; preds = %direct.false143
  %.spill.load154 = load <8 x float>, ptr %.spill31, align 32
  %322 = fadd <8 x float> %.spill.load154, splat (float 0x3EE4F8B580000000)
  %323 = call <8 x float> @llvm.sqrt.v8f32(<8 x float> %322)
  %324 = load <8 x float>, ptr %.spill34, align 32
  %325 = select <8 x i1> %51, <8 x float> %323, <8 x float> %324
  store <8 x float> %325, ptr %.spill34, align 32
  store i64 0, ptr %.slot35, align 4
  br label %direct.schedule.26

direct.schedule.26:                               ; preds = %direct.schedule.27, %direct.schedule.25
  %.state155 = load i64, ptr %.slot35, align 4
  %326 = icmp slt i64 %.state155, 16384
  br i1 %326, label %direct.true156, label %direct.false157

direct.schedule.27:                               ; preds = %direct.true156
  %.state158 = load i64, ptr %.slot35, align 4
  %327 = srem i64 %.state158, 16384
  %.state159 = load i64, ptr %.slot35, align 4
  %328 = sdiv i64 %.state159, 16384
  %329 = srem i64 %328, 1
  %.spill.load160 = load <8 x i64>, ptr %.spill, align 64
  %.splatinsert161 = insertelement <8 x i64> poison, i64 %329, i64 0
  %.splat162 = shufflevector <8 x i64> %.splatinsert161, <8 x i64> poison, <8 x i32> zeroinitializer
  %330 = add <8 x i64> %.spill.load160, %.splat162
  %331 = add <8 x i64> zeroinitializer, %330
  %332 = add i64 0, %327
  %333 = mul <8 x i64> %331, splat (i64 16384)
  %.splatinsert163 = insertelement <8 x i64> poison, i64 %332, i64 0
  %.splat164 = shufflevector <8 x i64> %.splatinsert163, <8 x i64> poison, <8 x i32> zeroinitializer
  %334 = add <8 x i64> %333, %.splat164
  %335 = add i64 0, %327
  %336 = srem i64 %335, 16384
  %337 = add i64 0, %336
  %tile_snapshot.spill.load165 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill, align 64
  %338 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load165, 0
  %339 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load165, 1
  %.splatinsert166 = insertelement <8 x i64> poison, i64 %337, i64 0
  %.splat167 = shufflevector <8 x i64> %.splatinsert166, <8 x i64> poison, <8 x i32> zeroinitializer
  %340 = mul <8 x i64> %.splat167, splat (i64 4)
  %341 = add <8 x i64> %339, %340
  %342 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %338, 0
  %343 = insertvalue { <8 x ptr>, <8 x i64> } %342, <8 x i64> %341, 1
  %344 = extractvalue { <8 x ptr>, <8 x i64> } %343, 0
  %345 = extractvalue { <8 x ptr>, <8 x i64> } %343, 1
  %346 = getelementptr i8, <8 x ptr> %344, <8 x i64> %345
  %347 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %346, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %.spill.load168 = load <8 x float>, ptr %.spill34, align 32
  %348 = fdiv <8 x float> %347, %.spill.load168
  %tile_snapshot.spill.load169 = load { <8 x ptr>, <8 x i64> }, ptr %tile_snapshot.spill32, align 64
  %349 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load169, 0
  %350 = extractvalue { <8 x ptr>, <8 x i64> } %tile_snapshot.spill.load169, 1
  %.splatinsert170 = insertelement <8 x i64> poison, i64 %335, i64 0
  %.splat171 = shufflevector <8 x i64> %.splatinsert170, <8 x i64> poison, <8 x i32> zeroinitializer
  %351 = mul <8 x i64> %.splat171, splat (i64 4)
  %352 = add <8 x i64> %350, %351
  %353 = insertvalue { <8 x ptr>, <8 x i64> } poison, <8 x ptr> %349, 0
  %354 = insertvalue { <8 x ptr>, <8 x i64> } %353, <8 x i64> %352, 1
  %355 = extractvalue { <8 x ptr>, <8 x i64> } %354, 0
  %356 = extractvalue { <8 x ptr>, <8 x i64> } %354, 1
  %357 = getelementptr i8, <8 x ptr> %355, <8 x i64> %356
  %358 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %357, i32 1, <8 x i1> %51, <8 x float> zeroinitializer)
  %359 = fmul <8 x float> %348, %358
  %360 = extractvalue { ptr, i64 } %27, 0
  %361 = mul <8 x i64> %334, splat (i64 4)
  %362 = getelementptr i8, ptr %360, <8 x i64> %361
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %359, <8 x ptr> %362, i32 1, <8 x i1> %51)
  %.state172 = load i64, ptr %.slot35, align 4
  %363 = add i64 %.state172, 1
  store i64 %363, ptr %.slot35, align 4
  br label %direct.schedule.26

direct.schedule.28:                               ; preds = %direct.false157
  ret void

direct.activate:                                  ; preds = %prologue
  br label %direct.schedule.0

direct.inactive:                                  ; preds = %prologue
  ret void

direct.true:                                      ; preds = %direct.schedule.1
  br label %direct.schedule.2

direct.false:                                     ; preds = %direct.schedule.1
  br label %direct.schedule.3

direct.true56:                                    ; preds = %direct.schedule.3
  br label %direct.schedule.4

direct.false57:                                   ; preds = %direct.schedule.3
  %364 = load <8 x float>, ptr %.slot8, align 32
  %365 = select <8 x i1> %51, <8 x float> zeroinitializer, <8 x float> %364
  store <8 x float> %365, ptr %.slot8, align 32
  br label %direct.schedule.5

direct.true63:                                    ; preds = %direct.schedule.5
  br label %direct.schedule.6

direct.false64:                                   ; preds = %direct.schedule.5
  %366 = load <8 x float>, ptr %.slot10, align 32
  %367 = select <8 x i1> %51, <8 x float> zeroinitializer, <8 x float> %366
  store <8 x float> %367, ptr %.slot10, align 32
  br label %direct.schedule.7

direct.true70:                                    ; preds = %direct.schedule.7
  br label %direct.schedule.8

direct.false71:                                   ; preds = %direct.schedule.7
  %368 = load <8 x float>, ptr %.slot12, align 32
  %369 = select <8 x i1> %51, <8 x float> zeroinitializer, <8 x float> %368
  store <8 x float> %369, ptr %.slot12, align 32
  br label %direct.schedule.9

direct.true77:                                    ; preds = %direct.schedule.9
  br label %direct.schedule.10

direct.false78:                                   ; preds = %direct.schedule.9
  %370 = load <8 x float>, ptr %.slot14, align 32
  %371 = select <8 x i1> %51, <8 x float> zeroinitializer, <8 x float> %370
  store <8 x float> %371, ptr %.slot14, align 32
  br label %direct.schedule.11

direct.true88:                                    ; preds = %direct.schedule.12
  br label %direct.schedule.13

direct.false89:                                   ; preds = %direct.schedule.12
  br label %direct.schedule.22

direct.true92:                                    ; preds = %direct.schedule.13
  br label %direct.schedule.14

direct.false93:                                   ; preds = %direct.schedule.13
  %372 = load <8 x float>, ptr %.slot21, align 32
  %373 = select <8 x i1> %51, <8 x float> zeroinitializer, <8 x float> %372
  store <8 x float> %373, ptr %.slot21, align 32
  br label %direct.schedule.15

direct.true102:                                   ; preds = %direct.schedule.15
  br label %direct.schedule.16

direct.false103:                                  ; preds = %direct.schedule.15
  %374 = load <8 x float>, ptr %.slot24, align 32
  %375 = select <8 x i1> %51, <8 x float> zeroinitializer, <8 x float> %374
  store <8 x float> %375, ptr %.slot24, align 32
  br label %direct.schedule.17

direct.true112:                                   ; preds = %direct.schedule.17
  br label %direct.schedule.18

direct.false113:                                  ; preds = %direct.schedule.17
  %376 = load <8 x float>, ptr %.slot27, align 32
  %377 = select <8 x i1> %51, <8 x float> zeroinitializer, <8 x float> %376
  store <8 x float> %377, ptr %.slot27, align 32
  br label %direct.schedule.19

direct.true122:                                   ; preds = %direct.schedule.19
  br label %direct.schedule.20

direct.false123:                                  ; preds = %direct.schedule.19
  %378 = load <8 x float>, ptr %.slot30, align 32
  %379 = select <8 x i1> %51, <8 x float> zeroinitializer, <8 x float> %378
  store <8 x float> %379, ptr %.slot30, align 32
  br label %direct.schedule.21

direct.true142:                                   ; preds = %direct.schedule.23
  br label %direct.schedule.24

direct.false143:                                  ; preds = %direct.schedule.23
  br label %direct.schedule.25

direct.true156:                                   ; preds = %direct.schedule.26
  br label %direct.schedule.27

direct.false157:                                  ; preds = %direct.schedule.26
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
