; ModuleID = 'luisa-simd-kernel'
source_filename = "luisa-simd-kernel"

define internal void @xir_gemm(ptr noalias readonly %argument_buffer, ptr %return_lanes, ptr noalias nonnull readonly %launch_config, i32 %active_lane_count) {
prologue:
  %.spill = alloca <8 x i64>, align 64
  store <8 x i64> zeroinitializer, ptr %.spill, align 64
  %.spill1 = alloca <8 x i64>, align 64
  store <8 x i64> zeroinitializer, ptr %.spill1, align 64
  %.slot = alloca i64, align 8
  store i64 0, ptr %.slot, align 4
  %.slot2 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot2, align 32
  %.spill3 = alloca i64, align 8
  store i64 0, ptr %.spill3, align 4
  %.spill4 = alloca i64, align 8
  store i64 0, ptr %.spill4, align 4
  %.spill5 = alloca <8 x i64>, align 64
  store <8 x i64> zeroinitializer, ptr %.spill5, align 64
  %.spill6 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill6, align 32
  %.spill7 = alloca i64, align 8
  store i64 0, ptr %.spill7, align 4
  %.spill8 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill8, align 32
  %.spill9 = alloca i64, align 8
  store i64 0, ptr %.spill9, align 4
  %.spill10 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill10, align 32
  %.spill11 = alloca i64, align 8
  store i64 0, ptr %.spill11, align 4
  %.spill12 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill12, align 32
  %.spill13 = alloca i64, align 8
  store i64 0, ptr %.spill13, align 4
  %.spill14 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill14, align 32
  %.spill15 = alloca i64, align 8
  store i64 0, ptr %.spill15, align 4
  %.spill16 = alloca i1, align 1
  store i1 false, ptr %.spill16, align 1
  %.spill17 = alloca <8 x i64>, align 64
  store <8 x i64> zeroinitializer, ptr %.spill17, align 64
  %.slot18 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot18, align 32
  %.spill19 = alloca i64, align 8
  store i64 0, ptr %.spill19, align 4
  %.spill20 = alloca i1, align 1
  store i1 false, ptr %.spill20, align 1
  %.spill21 = alloca <8 x i64>, align 64
  store <8 x i64> zeroinitializer, ptr %.spill21, align 64
  %.slot22 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot22, align 32
  %.spill23 = alloca i64, align 8
  store i64 0, ptr %.spill23, align 4
  %.spill24 = alloca i1, align 1
  store i1 false, ptr %.spill24, align 1
  %.spill25 = alloca <8 x i64>, align 64
  store <8 x i64> zeroinitializer, ptr %.spill25, align 64
  %.slot26 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot26, align 32
  %.spill27 = alloca i64, align 8
  store i64 0, ptr %.spill27, align 4
  %.spill28 = alloca <8 x i64>, align 64
  store <8 x i64> zeroinitializer, ptr %.spill28, align 64
  %.spill29 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill29, align 32
  %.spill30 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill30, align 32
  %.spill31 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill31, align 32
  %.spill32 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill32, align 32
  %.spill33 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.spill33, align 32
  %.spill34 = alloca <8 x i64>, align 64
  store <8 x i64> zeroinitializer, ptr %.spill34, align 64
  %.slot35 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot35, align 32
  %.spill36 = alloca <8 x i64>, align 64
  store <8 x i64> zeroinitializer, ptr %.spill36, align 64
  %.slot37 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot37, align 32
  %.spill38 = alloca <8 x i64>, align 64
  store <8 x i64> zeroinitializer, ptr %.spill38, align 64
  %.slot39 = alloca <8 x float>, align 32
  store <8 x float> zeroinitializer, ptr %.slot39, align 32
  %0 = getelementptr i8, ptr %argument_buffer, i64 0
  %1 = load ptr, ptr %0, align 16
  %2 = getelementptr i8, ptr %0, i64 8
  %3 = load i64, ptr %2, align 8
  %4 = insertvalue { ptr, i64 } poison, ptr %1, 0
  %5 = insertvalue { ptr, i64 } %4, i64 %3, 1
  %6 = getelementptr i8, ptr %argument_buffer, i64 16
  %7 = load ptr, ptr %6, align 16
  %8 = getelementptr i8, ptr %6, i64 8
  %9 = load i64, ptr %8, align 8
  %10 = insertvalue { ptr, i64 } poison, ptr %7, 0
  %11 = insertvalue { ptr, i64 } %10, i64 %9, 1
  %12 = getelementptr i8, ptr %argument_buffer, i64 32
  %13 = load ptr, ptr %12, align 16
  %14 = getelementptr i8, ptr %12, i64 8
  %15 = load i64, ptr %14, align 8
  %16 = insertvalue { ptr, i64 } poison, ptr %13, 0
  %17 = insertvalue { ptr, i64 } %16, i64 %15, 1
  %18 = load i32, ptr %launch_config, align 4
  %19 = getelementptr i8, ptr %launch_config, i64 12
  %20 = load i32, ptr %19, align 4
  %21 = getelementptr i8, ptr %launch_config, i64 4
  %22 = load i32, ptr %21, align 4
  %23 = getelementptr i8, ptr %launch_config, i64 16
  %24 = load i32, ptr %23, align 4
  %25 = getelementptr i8, ptr %launch_config, i64 8
  %26 = load i32, ptr %25, align 4
  %27 = getelementptr i8, ptr %launch_config, i64 20
  %28 = load i32, ptr %27, align 4
  %29 = getelementptr i8, ptr %launch_config, i64 36
  %30 = load i32, ptr %29, align 4
  %.splatinsert = insertelement <8 x i32> poison, i32 %30, i64 0
  %.splat = shufflevector <8 x i32> %.splatinsert, <8 x i32> poison, <8 x i32> zeroinitializer
  %31 = add <8 x i32> %.splat, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %32 = mul i32 %18, 1024
  %.splatinsert40 = insertelement <8 x i32> poison, i32 %32, i64 0
  %.splat41 = shufflevector <8 x i32> %.splatinsert40, <8 x i32> poison, <8 x i32> zeroinitializer
  %33 = add <8 x i32> %.splat41, %31
  %34 = mul i32 %22, 1
  %.splatinsert42 = insertelement <8 x i32> poison, i32 %34, i64 0
  %.splat43 = shufflevector <8 x i32> %.splatinsert42, <8 x i32> poison, <8 x i32> zeroinitializer
  %35 = add <8 x i32> %.splat43, zeroinitializer
  %36 = mul i32 %26, 1
  %.splatinsert44 = insertelement <8 x i32> poison, i32 %36, i64 0
  %.splat45 = shufflevector <8 x i32> %.splatinsert44, <8 x i32> poison, <8 x i32> zeroinitializer
  %37 = add <8 x i32> %.splat45, zeroinitializer
  %38 = insertvalue [3 x <8 x i32>] poison, <8 x i32> %33, 0
  %39 = insertvalue [3 x <8 x i32>] %38, <8 x i32> %35, 1
  %40 = insertvalue [3 x <8 x i32>] %39, <8 x i32> %37, 2
  %.splatinsert46 = insertelement <8 x i32> poison, i32 %active_lane_count, i64 0
  %.splat47 = shufflevector <8 x i32> %.splatinsert46, <8 x i32> poison, <8 x i32> zeroinitializer
  %41 = icmp ult <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, %.splat47
  %42 = call i1 @llvm.vector.reduce.or.v8i1(<8 x i1> %41)
  br i1 %42, label %direct.activate, label %direct.inactive

direct.schedule.0:                                ; preds = %direct.activate
  %43 = extractvalue [3 x <8 x i32>] %40, 0
  %44 = zext <8 x i32> %43 to <8 x i64>
  %45 = select <8 x i1> %41, <8 x i64> %44, <8 x i64> zeroinitializer
  %46 = select <8 x i1> %41, <8 x i64> splat (i64 193), <8 x i64> splat (i64 1)
  %47 = sdiv <8 x i64> %45, %46
  %48 = select <8 x i1> %41, <8 x i64> %47, <8 x i64> zeroinitializer
  %49 = select <8 x i1> %41, <8 x i64> splat (i64 127), <8 x i64> splat (i64 1)
  %50 = srem <8 x i64> %48, %49
  %51 = select <8 x i1> %41, <8 x i64> %44, <8 x i64> zeroinitializer
  %52 = select <8 x i1> %41, <8 x i64> splat (i64 1), <8 x i64> splat (i64 1)
  %53 = sdiv <8 x i64> %51, %52
  %54 = select <8 x i1> %41, <8 x i64> %53, <8 x i64> zeroinitializer
  %55 = select <8 x i1> %41, <8 x i64> splat (i64 193), <8 x i64> splat (i64 1)
  %56 = srem <8 x i64> %54, %55
  %57 = mul <8 x i64> %50, splat (i64 1)
  %58 = load <8 x i64>, ptr %.spill, align 64
  %59 = select <8 x i1> %41, <8 x i64> %57, <8 x i64> %58
  store <8 x i64> %59, ptr %.spill, align 64
  %60 = mul <8 x i64> %56, splat (i64 1)
  %61 = load <8 x i64>, ptr %.spill1, align 64
  %62 = select <8 x i1> %41, <8 x i64> %60, <8 x i64> %61
  store <8 x i64> %62, ptr %.spill1, align 64
  store i64 0, ptr %.slot, align 4
  %63 = load <8 x float>, ptr %.slot2, align 32
  %64 = select <8 x i1> %41, <8 x float> zeroinitializer, <8 x float> %63
  store <8 x float> %64, ptr %.slot2, align 32
  br label %direct.schedule.1

direct.schedule.1:                                ; preds = %direct.schedule.14, %direct.schedule.0
  %.state = load i64, ptr %.slot, align 4
  %65 = icmp slt i64 %.state, 8
  br i1 %65, label %direct.true, label %direct.false

direct.schedule.2:                                ; preds = %direct.true
  %.state48 = load i64, ptr %.slot, align 4
  %66 = sdiv i64 %.state48, 1
  %67 = srem i64 %66, 8
  %68 = mul i64 %67, 8
  store i64 %68, ptr %.spill3, align 4
  %.spill.load = load <8 x i64>, ptr %.spill, align 64
  %69 = add <8 x i64> %.spill.load, zeroinitializer
  %70 = add <8 x i64> zeroinitializer, %69
  %71 = add i64 %68, 0
  store i64 %71, ptr %.spill4, align 4
  %72 = mul <8 x i64> %70, splat (i64 61)
  %73 = load <8 x i64>, ptr %.spill5, align 64
  %74 = select <8 x i1> %41, <8 x i64> %72, <8 x i64> %73
  store <8 x i64> %74, ptr %.spill5, align 64
  %.splatinsert49 = insertelement <8 x i64> poison, i64 %71, i64 0
  %.splat50 = shufflevector <8 x i64> %.splatinsert49, <8 x i64> poison, <8 x i32> zeroinitializer
  %75 = add <8 x i64> %72, %.splat50
  %76 = extractvalue { ptr, i64 } %5, 0
  %77 = mul <8 x i64> %75, splat (i64 4)
  %78 = getelementptr i8, ptr %76, <8 x i64> %77
  %79 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %78, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %80 = load <8 x float>, ptr %.spill6, align 32
  %81 = select <8 x i1> %41, <8 x float> %79, <8 x float> %80
  store <8 x float> %81, ptr %.spill6, align 32
  %82 = add i64 %68, 1
  store i64 %82, ptr %.spill7, align 4
  %.splatinsert51 = insertelement <8 x i64> poison, i64 %82, i64 0
  %.splat52 = shufflevector <8 x i64> %.splatinsert51, <8 x i64> poison, <8 x i32> zeroinitializer
  %83 = add <8 x i64> %72, %.splat52
  %84 = extractvalue { ptr, i64 } %5, 0
  %85 = mul <8 x i64> %83, splat (i64 4)
  %86 = getelementptr i8, ptr %84, <8 x i64> %85
  %87 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %86, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %88 = load <8 x float>, ptr %.spill8, align 32
  %89 = select <8 x i1> %41, <8 x float> %87, <8 x float> %88
  store <8 x float> %89, ptr %.spill8, align 32
  %90 = add i64 %68, 2
  store i64 %90, ptr %.spill9, align 4
  %.splatinsert53 = insertelement <8 x i64> poison, i64 %90, i64 0
  %.splat54 = shufflevector <8 x i64> %.splatinsert53, <8 x i64> poison, <8 x i32> zeroinitializer
  %91 = add <8 x i64> %72, %.splat54
  %92 = extractvalue { ptr, i64 } %5, 0
  %93 = mul <8 x i64> %91, splat (i64 4)
  %94 = getelementptr i8, ptr %92, <8 x i64> %93
  %95 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %94, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %96 = load <8 x float>, ptr %.spill10, align 32
  %97 = select <8 x i1> %41, <8 x float> %95, <8 x float> %96
  store <8 x float> %97, ptr %.spill10, align 32
  %98 = add i64 %68, 3
  store i64 %98, ptr %.spill11, align 4
  %.splatinsert55 = insertelement <8 x i64> poison, i64 %98, i64 0
  %.splat56 = shufflevector <8 x i64> %.splatinsert55, <8 x i64> poison, <8 x i32> zeroinitializer
  %99 = add <8 x i64> %72, %.splat56
  %100 = extractvalue { ptr, i64 } %5, 0
  %101 = mul <8 x i64> %99, splat (i64 4)
  %102 = getelementptr i8, ptr %100, <8 x i64> %101
  %103 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %102, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %104 = load <8 x float>, ptr %.spill12, align 32
  %105 = select <8 x i1> %41, <8 x float> %103, <8 x float> %104
  store <8 x float> %105, ptr %.spill12, align 32
  %106 = add i64 %68, 4
  store i64 %106, ptr %.spill13, align 4
  %.splatinsert57 = insertelement <8 x i64> poison, i64 %106, i64 0
  %.splat58 = shufflevector <8 x i64> %.splatinsert57, <8 x i64> poison, <8 x i32> zeroinitializer
  %107 = add <8 x i64> %72, %.splat58
  %108 = extractvalue { ptr, i64 } %5, 0
  %109 = mul <8 x i64> %107, splat (i64 4)
  %110 = getelementptr i8, ptr %108, <8 x i64> %109
  %111 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %110, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %112 = load <8 x float>, ptr %.spill14, align 32
  %113 = select <8 x i1> %41, <8 x float> %111, <8 x float> %112
  store <8 x float> %113, ptr %.spill14, align 32
  %114 = add i64 %68, 5
  store i64 %114, ptr %.spill15, align 4
  %.splatinsert59 = insertelement <8 x i64> poison, i64 %114, i64 0
  %.splat60 = shufflevector <8 x i64> %.splatinsert59, <8 x i64> poison, <8 x i32> zeroinitializer
  %115 = add <8 x i64> %72, %.splat60
  %116 = icmp sge i64 %114, 0
  %117 = and i1 true, %116
  %118 = icmp slt i64 %114, 61
  %119 = and i1 %117, %118
  store i1 %119, ptr %.spill16, align 1
  %120 = load <8 x i64>, ptr %.spill17, align 64
  %121 = select <8 x i1> %41, <8 x i64> %115, <8 x i64> %120
  store <8 x i64> %121, ptr %.spill17, align 64
  br i1 %119, label %direct.true61, label %direct.false62

direct.schedule.3:                                ; preds = %direct.true61
  %.spill.load63 = load <8 x i64>, ptr %.spill17, align 64
  %122 = extractvalue { ptr, i64 } %5, 0
  %123 = mul <8 x i64> %.spill.load63, splat (i64 4)
  %124 = getelementptr i8, ptr %122, <8 x i64> %123
  %125 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %124, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %126 = load <8 x float>, ptr %.slot18, align 32
  %127 = select <8 x i1> %41, <8 x float> %125, <8 x float> %126
  store <8 x float> %127, ptr %.slot18, align 32
  br label %direct.schedule.4

direct.schedule.4:                                ; preds = %direct.schedule.3, %direct.false62
  %.spill.load64 = load i64, ptr %.spill3, align 4
  %128 = add i64 %.spill.load64, 6
  store i64 %128, ptr %.spill19, align 4
  %.spill.load65 = load <8 x i64>, ptr %.spill5, align 64
  %.splatinsert66 = insertelement <8 x i64> poison, i64 %128, i64 0
  %.splat67 = shufflevector <8 x i64> %.splatinsert66, <8 x i64> poison, <8 x i32> zeroinitializer
  %129 = add <8 x i64> %.spill.load65, %.splat67
  %130 = icmp sge i64 %128, 0
  %131 = and i1 true, %130
  %132 = icmp slt i64 %128, 61
  %133 = and i1 %131, %132
  store i1 %133, ptr %.spill20, align 1
  %134 = load <8 x i64>, ptr %.spill21, align 64
  %135 = select <8 x i1> %41, <8 x i64> %129, <8 x i64> %134
  store <8 x i64> %135, ptr %.spill21, align 64
  br i1 %133, label %direct.true68, label %direct.false69

direct.schedule.5:                                ; preds = %direct.true68
  %.spill.load70 = load <8 x i64>, ptr %.spill21, align 64
  %136 = extractvalue { ptr, i64 } %5, 0
  %137 = mul <8 x i64> %.spill.load70, splat (i64 4)
  %138 = getelementptr i8, ptr %136, <8 x i64> %137
  %139 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %138, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %140 = load <8 x float>, ptr %.slot22, align 32
  %141 = select <8 x i1> %41, <8 x float> %139, <8 x float> %140
  store <8 x float> %141, ptr %.slot22, align 32
  br label %direct.schedule.6

direct.schedule.6:                                ; preds = %direct.schedule.5, %direct.false69
  %.spill.load71 = load i64, ptr %.spill3, align 4
  %142 = add i64 %.spill.load71, 7
  store i64 %142, ptr %.spill23, align 4
  %.spill.load72 = load <8 x i64>, ptr %.spill5, align 64
  %.splatinsert73 = insertelement <8 x i64> poison, i64 %142, i64 0
  %.splat74 = shufflevector <8 x i64> %.splatinsert73, <8 x i64> poison, <8 x i32> zeroinitializer
  %143 = add <8 x i64> %.spill.load72, %.splat74
  %144 = icmp sge i64 %142, 0
  %145 = and i1 true, %144
  %146 = icmp slt i64 %142, 61
  %147 = and i1 %145, %146
  store i1 %147, ptr %.spill24, align 1
  %148 = load <8 x i64>, ptr %.spill25, align 64
  %149 = select <8 x i1> %41, <8 x i64> %143, <8 x i64> %148
  store <8 x i64> %149, ptr %.spill25, align 64
  br i1 %147, label %direct.true75, label %direct.false76

direct.schedule.7:                                ; preds = %direct.true75
  %.spill.load77 = load <8 x i64>, ptr %.spill25, align 64
  %150 = extractvalue { ptr, i64 } %5, 0
  %151 = mul <8 x i64> %.spill.load77, splat (i64 4)
  %152 = getelementptr i8, ptr %150, <8 x i64> %151
  %153 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %152, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %154 = load <8 x float>, ptr %.slot26, align 32
  %155 = select <8 x i1> %41, <8 x float> %153, <8 x float> %154
  store <8 x float> %155, ptr %.slot26, align 32
  br label %direct.schedule.8

direct.schedule.8:                                ; preds = %direct.schedule.7, %direct.false76
  store i64 0, ptr %.spill27, align 4
  %.spill.load78 = load i64, ptr %.spill4, align 4
  %156 = add i64 0, %.spill.load78
  %.spill.load79 = load <8 x i64>, ptr %.spill1, align 64
  %157 = add <8 x i64> %.spill.load79, zeroinitializer
  %158 = load <8 x i64>, ptr %.spill28, align 64
  %159 = select <8 x i1> %41, <8 x i64> %157, <8 x i64> %158
  store <8 x i64> %159, ptr %.spill28, align 64
  %160 = mul i64 %156, 193
  %.splatinsert80 = insertelement <8 x i64> poison, i64 %160, i64 0
  %.splat81 = shufflevector <8 x i64> %.splatinsert80, <8 x i64> poison, <8 x i32> zeroinitializer
  %161 = add <8 x i64> %.splat81, %157
  %162 = extractvalue { ptr, i64 } %11, 0
  %163 = mul <8 x i64> %161, splat (i64 4)
  %164 = getelementptr i8, ptr %162, <8 x i64> %163
  %165 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %164, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %166 = load <8 x float>, ptr %.spill29, align 32
  %167 = select <8 x i1> %41, <8 x float> %165, <8 x float> %166
  store <8 x float> %167, ptr %.spill29, align 32
  %.spill.load82 = load i64, ptr %.spill7, align 4
  %168 = add i64 0, %.spill.load82
  %169 = mul i64 %168, 193
  %.splatinsert83 = insertelement <8 x i64> poison, i64 %169, i64 0
  %.splat84 = shufflevector <8 x i64> %.splatinsert83, <8 x i64> poison, <8 x i32> zeroinitializer
  %170 = add <8 x i64> %.splat84, %157
  %171 = extractvalue { ptr, i64 } %11, 0
  %172 = mul <8 x i64> %170, splat (i64 4)
  %173 = getelementptr i8, ptr %171, <8 x i64> %172
  %174 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %173, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %175 = load <8 x float>, ptr %.spill30, align 32
  %176 = select <8 x i1> %41, <8 x float> %174, <8 x float> %175
  store <8 x float> %176, ptr %.spill30, align 32
  %.spill.load85 = load i64, ptr %.spill9, align 4
  %177 = add i64 0, %.spill.load85
  %178 = mul i64 %177, 193
  %.splatinsert86 = insertelement <8 x i64> poison, i64 %178, i64 0
  %.splat87 = shufflevector <8 x i64> %.splatinsert86, <8 x i64> poison, <8 x i32> zeroinitializer
  %179 = add <8 x i64> %.splat87, %157
  %180 = extractvalue { ptr, i64 } %11, 0
  %181 = mul <8 x i64> %179, splat (i64 4)
  %182 = getelementptr i8, ptr %180, <8 x i64> %181
  %183 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %182, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %184 = load <8 x float>, ptr %.spill31, align 32
  %185 = select <8 x i1> %41, <8 x float> %183, <8 x float> %184
  store <8 x float> %185, ptr %.spill31, align 32
  %.spill.load88 = load i64, ptr %.spill11, align 4
  %186 = add i64 0, %.spill.load88
  %187 = mul i64 %186, 193
  %.splatinsert89 = insertelement <8 x i64> poison, i64 %187, i64 0
  %.splat90 = shufflevector <8 x i64> %.splatinsert89, <8 x i64> poison, <8 x i32> zeroinitializer
  %188 = add <8 x i64> %.splat90, %157
  %189 = extractvalue { ptr, i64 } %11, 0
  %190 = mul <8 x i64> %188, splat (i64 4)
  %191 = getelementptr i8, ptr %189, <8 x i64> %190
  %192 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %191, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %193 = load <8 x float>, ptr %.spill32, align 32
  %194 = select <8 x i1> %41, <8 x float> %192, <8 x float> %193
  store <8 x float> %194, ptr %.spill32, align 32
  %.spill.load91 = load i64, ptr %.spill13, align 4
  %195 = add i64 0, %.spill.load91
  %196 = mul i64 %195, 193
  %.splatinsert92 = insertelement <8 x i64> poison, i64 %196, i64 0
  %.splat93 = shufflevector <8 x i64> %.splatinsert92, <8 x i64> poison, <8 x i32> zeroinitializer
  %197 = add <8 x i64> %.splat93, %157
  %198 = extractvalue { ptr, i64 } %11, 0
  %199 = mul <8 x i64> %197, splat (i64 4)
  %200 = getelementptr i8, ptr %198, <8 x i64> %199
  %201 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %200, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %202 = load <8 x float>, ptr %.spill33, align 32
  %203 = select <8 x i1> %41, <8 x float> %201, <8 x float> %202
  store <8 x float> %203, ptr %.spill33, align 32
  %.spill.load94 = load i64, ptr %.spill15, align 4
  %204 = add i64 0, %.spill.load94
  %205 = mul i64 %204, 193
  %.splatinsert95 = insertelement <8 x i64> poison, i64 %205, i64 0
  %.splat96 = shufflevector <8 x i64> %.splatinsert95, <8 x i64> poison, <8 x i32> zeroinitializer
  %206 = add <8 x i64> %.splat96, %157
  %207 = load <8 x i64>, ptr %.spill34, align 64
  %208 = select <8 x i1> %41, <8 x i64> %206, <8 x i64> %207
  store <8 x i64> %208, ptr %.spill34, align 64
  %.spill.load97 = load i1, ptr %.spill16, align 1
  br i1 %.spill.load97, label %direct.true98, label %direct.false99

direct.schedule.9:                                ; preds = %direct.true98
  %.spill.load100 = load <8 x i64>, ptr %.spill34, align 64
  %209 = extractvalue { ptr, i64 } %11, 0
  %210 = mul <8 x i64> %.spill.load100, splat (i64 4)
  %211 = getelementptr i8, ptr %209, <8 x i64> %210
  %212 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %211, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %213 = load <8 x float>, ptr %.slot35, align 32
  %214 = select <8 x i1> %41, <8 x float> %212, <8 x float> %213
  store <8 x float> %214, ptr %.slot35, align 32
  br label %direct.schedule.10

direct.schedule.10:                               ; preds = %direct.schedule.9, %direct.false99
  %.spill.load101 = load i64, ptr %.spill27, align 4
  %.spill.load102 = load i64, ptr %.spill19, align 4
  %215 = add i64 %.spill.load101, %.spill.load102
  %216 = mul i64 %215, 193
  %.splatinsert103 = insertelement <8 x i64> poison, i64 %216, i64 0
  %.splat104 = shufflevector <8 x i64> %.splatinsert103, <8 x i64> poison, <8 x i32> zeroinitializer
  %.spill.load105 = load <8 x i64>, ptr %.spill28, align 64
  %217 = add <8 x i64> %.splat104, %.spill.load105
  %218 = load <8 x i64>, ptr %.spill36, align 64
  %219 = select <8 x i1> %41, <8 x i64> %217, <8 x i64> %218
  store <8 x i64> %219, ptr %.spill36, align 64
  %.spill.load106 = load i1, ptr %.spill20, align 1
  br i1 %.spill.load106, label %direct.true107, label %direct.false108

direct.schedule.11:                               ; preds = %direct.true107
  %.spill.load109 = load <8 x i64>, ptr %.spill36, align 64
  %220 = extractvalue { ptr, i64 } %11, 0
  %221 = mul <8 x i64> %.spill.load109, splat (i64 4)
  %222 = getelementptr i8, ptr %220, <8 x i64> %221
  %223 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %222, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %224 = load <8 x float>, ptr %.slot37, align 32
  %225 = select <8 x i1> %41, <8 x float> %223, <8 x float> %224
  store <8 x float> %225, ptr %.slot37, align 32
  br label %direct.schedule.12

direct.schedule.12:                               ; preds = %direct.schedule.11, %direct.false108
  %.spill.load110 = load i64, ptr %.spill27, align 4
  %.spill.load111 = load i64, ptr %.spill23, align 4
  %226 = add i64 %.spill.load110, %.spill.load111
  %227 = mul i64 %226, 193
  %.splatinsert112 = insertelement <8 x i64> poison, i64 %227, i64 0
  %.splat113 = shufflevector <8 x i64> %.splatinsert112, <8 x i64> poison, <8 x i32> zeroinitializer
  %.spill.load114 = load <8 x i64>, ptr %.spill28, align 64
  %228 = add <8 x i64> %.splat113, %.spill.load114
  %229 = load <8 x i64>, ptr %.spill38, align 64
  %230 = select <8 x i1> %41, <8 x i64> %228, <8 x i64> %229
  store <8 x i64> %230, ptr %.spill38, align 64
  %.spill.load115 = load i1, ptr %.spill24, align 1
  br i1 %.spill.load115, label %direct.true116, label %direct.false117

direct.schedule.13:                               ; preds = %direct.true116
  %.spill.load118 = load <8 x i64>, ptr %.spill38, align 64
  %231 = extractvalue { ptr, i64 } %11, 0
  %232 = mul <8 x i64> %.spill.load118, splat (i64 4)
  %233 = getelementptr i8, ptr %231, <8 x i64> %232
  %234 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %233, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %235 = load <8 x float>, ptr %.slot39, align 32
  %236 = select <8 x i1> %41, <8 x float> %234, <8 x float> %235
  store <8 x float> %236, ptr %.slot39, align 32
  br label %direct.schedule.14

direct.schedule.14:                               ; preds = %direct.schedule.13, %direct.false117
  %.spill.load119 = load <8 x float>, ptr %.spill6, align 32
  %.spill.load120 = load <8 x float>, ptr %.spill29, align 32
  %237 = fmul <8 x float> %.spill.load119, %.spill.load120
  %.state121 = load <8 x float>, ptr %.slot2, align 32
  %238 = fadd <8 x float> %.state121, %237
  %.spill.load122 = load <8 x float>, ptr %.spill8, align 32
  %.spill.load123 = load <8 x float>, ptr %.spill30, align 32
  %239 = fmul <8 x float> %.spill.load122, %.spill.load123
  %240 = fadd <8 x float> %238, %239
  %.spill.load124 = load <8 x float>, ptr %.spill10, align 32
  %.spill.load125 = load <8 x float>, ptr %.spill31, align 32
  %241 = fmul <8 x float> %.spill.load124, %.spill.load125
  %242 = fadd <8 x float> %240, %241
  %.spill.load126 = load <8 x float>, ptr %.spill12, align 32
  %.spill.load127 = load <8 x float>, ptr %.spill32, align 32
  %243 = fmul <8 x float> %.spill.load126, %.spill.load127
  %244 = fadd <8 x float> %242, %243
  %.spill.load128 = load <8 x float>, ptr %.spill14, align 32
  %.spill.load129 = load <8 x float>, ptr %.spill33, align 32
  %245 = fmul <8 x float> %.spill.load128, %.spill.load129
  %246 = fadd <8 x float> %244, %245
  %.state130 = load <8 x float>, ptr %.slot18, align 32
  %.state131 = load <8 x float>, ptr %.slot35, align 32
  %247 = fmul <8 x float> %.state130, %.state131
  %248 = fadd <8 x float> %246, %247
  %.state132 = load <8 x float>, ptr %.slot22, align 32
  %.state133 = load <8 x float>, ptr %.slot37, align 32
  %249 = fmul <8 x float> %.state132, %.state133
  %250 = fadd <8 x float> %248, %249
  %.state134 = load <8 x float>, ptr %.slot26, align 32
  %.state135 = load <8 x float>, ptr %.slot39, align 32
  %251 = fmul <8 x float> %.state134, %.state135
  %252 = fadd <8 x float> %250, %251
  %.state136 = load i64, ptr %.slot, align 4
  %253 = add i64 %.state136, 1
  store i64 %253, ptr %.slot, align 4
  %254 = load <8 x float>, ptr %.slot2, align 32
  %255 = select <8 x i1> %41, <8 x float> %252, <8 x float> %254
  store <8 x float> %255, ptr %.slot2, align 32
  br label %direct.schedule.1

direct.schedule.15:                               ; preds = %direct.false
  %.spill.load137 = load <8 x i64>, ptr %.spill, align 64
  %256 = add <8 x i64> %.spill.load137, zeroinitializer
  %257 = add <8 x i64> zeroinitializer, %256
  %.spill.load138 = load <8 x i64>, ptr %.spill1, align 64
  %258 = add <8 x i64> %.spill.load138, zeroinitializer
  %259 = mul <8 x i64> %257, splat (i64 193)
  %260 = add <8 x i64> %259, %258
  %.state139 = load <8 x float>, ptr %.slot2, align 32
  %261 = extractvalue { ptr, i64 } %17, 0
  %262 = mul <8 x i64> %260, splat (i64 4)
  %263 = getelementptr i8, ptr %261, <8 x i64> %262
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %.state139, <8 x ptr> %263, i32 1, <8 x i1> %41)
  ret void

direct.activate:                                  ; preds = %prologue
  br label %direct.schedule.0

direct.inactive:                                  ; preds = %prologue
  ret void

direct.true:                                      ; preds = %direct.schedule.1
  br label %direct.schedule.2

direct.false:                                     ; preds = %direct.schedule.1
  br label %direct.schedule.15

direct.true61:                                    ; preds = %direct.schedule.2
  br label %direct.schedule.3

direct.false62:                                   ; preds = %direct.schedule.2
  %264 = load <8 x float>, ptr %.slot18, align 32
  %265 = select <8 x i1> %41, <8 x float> zeroinitializer, <8 x float> %264
  store <8 x float> %265, ptr %.slot18, align 32
  br label %direct.schedule.4

direct.true68:                                    ; preds = %direct.schedule.4
  br label %direct.schedule.5

direct.false69:                                   ; preds = %direct.schedule.4
  %266 = load <8 x float>, ptr %.slot22, align 32
  %267 = select <8 x i1> %41, <8 x float> zeroinitializer, <8 x float> %266
  store <8 x float> %267, ptr %.slot22, align 32
  br label %direct.schedule.6

direct.true75:                                    ; preds = %direct.schedule.6
  br label %direct.schedule.7

direct.false76:                                   ; preds = %direct.schedule.6
  %268 = load <8 x float>, ptr %.slot26, align 32
  %269 = select <8 x i1> %41, <8 x float> zeroinitializer, <8 x float> %268
  store <8 x float> %269, ptr %.slot26, align 32
  br label %direct.schedule.8

direct.true98:                                    ; preds = %direct.schedule.8
  br label %direct.schedule.9

direct.false99:                                   ; preds = %direct.schedule.8
  %270 = load <8 x float>, ptr %.slot35, align 32
  %271 = select <8 x i1> %41, <8 x float> zeroinitializer, <8 x float> %270
  store <8 x float> %271, ptr %.slot35, align 32
  br label %direct.schedule.10

direct.true107:                                   ; preds = %direct.schedule.10
  br label %direct.schedule.11

direct.false108:                                  ; preds = %direct.schedule.10
  %272 = load <8 x float>, ptr %.slot37, align 32
  %273 = select <8 x i1> %41, <8 x float> zeroinitializer, <8 x float> %272
  store <8 x float> %273, ptr %.slot37, align 32
  br label %direct.schedule.12

direct.true116:                                   ; preds = %direct.schedule.12
  br label %direct.schedule.13

direct.false117:                                  ; preds = %direct.schedule.12
  %274 = load <8 x float>, ptr %.slot39, align 32
  %275 = select <8 x i1> %41, <8 x float> zeroinitializer, <8 x float> %274
  store <8 x float> %275, ptr %.slot39, align 32
  br label %direct.schedule.14
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.vector.reduce.or.v8i1(<8 x i1>) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr>, i32 immarg, <8 x i1>, <8 x float>) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.masked.scatter.v8f32.v8p0(<8 x float>, <8 x ptr>, i32 immarg, <8 x i1>) #2

define internal void @xir_gemm.packet_batch(ptr noalias readonly %argument_buffer, ptr %return_lanes, ptr noalias nonnull %launch_config, i32 %packet_count) {
packet.batch.prologue:
  %thread.index.address = getelementptr inbounds i8, ptr %launch_config, i64 36
  %base.thread.index = load i32, ptr %thread.index.address, align 4
  %block.x.address = getelementptr inbounds i8, ptr %launch_config, i64 0
  %block.x = load i32, ptr %block.x.address, align 4
  %dispatch.size.x.address = getelementptr inbounds i8, ptr %launch_config, i64 12
  %dispatch.size.x = load i32, ptr %dispatch.size.x.address, align 4
  %base.thread.index.i64 = zext i32 %base.thread.index to i64
  %0 = zext i32 %block.x to i64
  %block.origin.x = mul i64 %0, 1024
  %dispatch.size.x.i64 = zext i32 %dispatch.size.x to i64
  %block.origin.in.range = icmp ule i64 %block.origin.x, %dispatch.size.x.i64
  %block.origin.safe = select i1 %block.origin.in.range, i64 %block.origin.x, i64 0
  %packet.range.start = add i64 %block.origin.safe, %base.thread.index.i64
  %1 = icmp ult i64 %packet.range.start, %dispatch.size.x.i64
  %packet.range.inside.dispatch = and i1 %block.origin.in.range, %1
  %2 = sub i64 %dispatch.size.x.i64, %packet.range.start
  %dispatch.remaining = select i1 %packet.range.inside.dispatch, i64 %2, i64 0
  %packet.range.inside.block = icmp ult i64 %base.thread.index.i64, 1024
  %3 = sub i64 1024, %base.thread.index.i64
  %block.remaining = select i1 %packet.range.inside.block, i64 %3, i64 0
  %4 = icmp ult i64 %dispatch.remaining, %block.remaining
  %packet.range.remaining = select i1 %4, i64 %dispatch.remaining, i64 %block.remaining
  %5 = zext i32 %packet_count to i64
  %packet.range.requested.threads = mul i64 %5, 8
  %6 = icmp ult i64 %packet.range.remaining, %packet.range.requested.threads
  %packet.range.active.threads = select i1 %6, i64 %packet.range.remaining, i64 %packet.range.requested.threads
  %7 = icmp eq i32 %packet_count, 128
  %8 = icmp uge i64 %packet.range.remaining, 1024
  %packet.range.complete.static = and i1 %7, %8
  br i1 %packet.range.complete.static, label %packet.batch.full, label %packet.batch.partial

packet.batch.full:                                ; preds = %packet.batch.prologue
  br label %packet.batch.full.loop

packet.batch.partial:                             ; preds = %packet.batch.prologue
  %packet.full.count.i64 = udiv i64 %packet.range.active.threads, 8
  %packet.full.count = trunc i64 %packet.full.count.i64 to i32
  %packet.tail.lane.count.i64 = urem i64 %packet.range.active.threads, 8
  %packet.tail.lane.count = trunc i64 %packet.tail.lane.count.i64 to i32
  %9 = icmp ne i32 %packet.full.count, 0
  br i1 %9, label %packet.batch.partial.full.loop, label %packet.batch.tail.check

packet.batch.exit:                                ; preds = %packet.batch.partial.finish, %packet.batch.full.loop
  ret void

packet.batch.full.loop:                           ; preds = %packet.batch.full.loop, %packet.batch.full
  %packet.index = phi i32 [ 0, %packet.batch.full ], [ %packet.index.next, %packet.batch.full.loop ]
  %packet.thread.offset = mul i32 %packet.index, 8
  %packet.thread.index = add i32 %base.thread.index, %packet.thread.offset
  store i32 %packet.thread.index, ptr %thread.index.address, align 4
  call void @xir_gemm(ptr %argument_buffer, ptr %return_lanes, ptr %launch_config, i32 8)
  %packet.index.next = add i32 %packet.index, 1
  %packet.batch.has.more = icmp ult i32 %packet.index.next, %packet_count
  br i1 %packet.batch.has.more, label %packet.batch.full.loop, label %packet.batch.exit

packet.batch.partial.full.loop:                   ; preds = %packet.batch.partial.full.loop, %packet.batch.partial
  %partial.packet.index = phi i32 [ 0, %packet.batch.partial ], [ %partial.packet.index.next, %packet.batch.partial.full.loop ]
  %partial.packet.thread.offset = mul i32 %partial.packet.index, 8
  %partial.packet.thread.index = add i32 %base.thread.index, %partial.packet.thread.offset
  store i32 %partial.packet.thread.index, ptr %thread.index.address, align 4
  call void @xir_gemm(ptr %argument_buffer, ptr %return_lanes, ptr %launch_config, i32 8)
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
  call void @xir_gemm(ptr %argument_buffer, ptr %return_lanes, ptr %launch_config, i32 %packet.tail.lane.count)
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

define dso_local void @xir_gemm.packet_batch.blocks(ptr noalias readonly %argument_buffer, ptr %return_lanes, ptr noalias nonnull %launch_config, i32 %block_count) {
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
  call void @xir_gemm.packet_batch(ptr %argument_buffer, ptr %return_lanes, ptr %launch_config, i32 128)
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
