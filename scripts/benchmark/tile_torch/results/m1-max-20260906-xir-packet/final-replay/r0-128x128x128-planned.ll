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
  %.splatinsert3 = insertelement <8 x i32> poison, i32 %32, i64 0
  %.splat4 = shufflevector <8 x i32> %.splatinsert3, <8 x i32> poison, <8 x i32> zeroinitializer
  %33 = add <8 x i32> %.splat4, %31
  %34 = mul i32 %22, 1
  %.splatinsert5 = insertelement <8 x i32> poison, i32 %34, i64 0
  %.splat6 = shufflevector <8 x i32> %.splatinsert5, <8 x i32> poison, <8 x i32> zeroinitializer
  %35 = add <8 x i32> %.splat6, zeroinitializer
  %36 = mul i32 %26, 1
  %.splatinsert7 = insertelement <8 x i32> poison, i32 %36, i64 0
  %.splat8 = shufflevector <8 x i32> %.splatinsert7, <8 x i32> poison, <8 x i32> zeroinitializer
  %37 = add <8 x i32> %.splat8, zeroinitializer
  %38 = insertvalue [3 x <8 x i32>] poison, <8 x i32> %33, 0
  %39 = insertvalue [3 x <8 x i32>] %38, <8 x i32> %35, 1
  %40 = insertvalue [3 x <8 x i32>] %39, <8 x i32> %37, 2
  %.splatinsert9 = insertelement <8 x i32> poison, i32 %active_lane_count, i64 0
  %.splat10 = shufflevector <8 x i32> %.splatinsert9, <8 x i32> poison, <8 x i32> zeroinitializer
  %41 = icmp ult <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, %.splat10
  %42 = call i1 @llvm.vector.reduce.or.v8i1(<8 x i1> %41)
  br i1 %42, label %direct.activate, label %direct.inactive

direct.schedule.0:                                ; preds = %direct.activate
  %43 = extractvalue [3 x <8 x i32>] %40, 0
  %44 = zext <8 x i32> %43 to <8 x i64>
  %45 = select <8 x i1> %41, <8 x i64> %44, <8 x i64> zeroinitializer
  %46 = select <8 x i1> %41, <8 x i64> splat (i64 128), <8 x i64> splat (i64 1)
  %47 = sdiv <8 x i64> %45, %46
  %48 = select <8 x i1> %41, <8 x i64> %47, <8 x i64> zeroinitializer
  %49 = select <8 x i1> %41, <8 x i64> splat (i64 128), <8 x i64> splat (i64 1)
  %50 = srem <8 x i64> %48, %49
  %51 = select <8 x i1> %41, <8 x i64> %44, <8 x i64> zeroinitializer
  %52 = select <8 x i1> %41, <8 x i64> splat (i64 1), <8 x i64> splat (i64 1)
  %53 = sdiv <8 x i64> %51, %52
  %54 = select <8 x i1> %41, <8 x i64> %53, <8 x i64> zeroinitializer
  %55 = select <8 x i1> %41, <8 x i64> splat (i64 128), <8 x i64> splat (i64 1)
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

direct.schedule.1:                                ; preds = %direct.schedule.2, %direct.schedule.0
  %.state = load i64, ptr %.slot, align 4
  %65 = icmp slt i64 %.state, 16
  br i1 %65, label %direct.true, label %direct.false

direct.schedule.2:                                ; preds = %direct.true
  %.state11 = load i64, ptr %.slot, align 4
  %66 = sdiv i64 %.state11, 1
  %67 = srem i64 %66, 16
  %68 = mul i64 %67, 8
  %.spill.load = load <8 x i64>, ptr %.spill, align 64
  %69 = add <8 x i64> %.spill.load, zeroinitializer
  %70 = add <8 x i64> zeroinitializer, %69
  %71 = add i64 %68, 0
  %72 = mul <8 x i64> %70, splat (i64 128)
  %.splatinsert12 = insertelement <8 x i64> poison, i64 %71, i64 0
  %.splat13 = shufflevector <8 x i64> %.splatinsert12, <8 x i64> poison, <8 x i32> zeroinitializer
  %73 = add <8 x i64> %72, %.splat13
  %74 = extractvalue { ptr, i64 } %5, 0
  %75 = extractelement <8 x i64> %73, i32 0
  %76 = mul i64 %75, 4
  %77 = getelementptr i8, ptr %74, i64 %76
  %78 = load float, ptr %77, align 4
  %.splatinsert14 = insertelement <8 x float> poison, float %78, i64 0
  %.splat15 = shufflevector <8 x float> %.splatinsert14, <8 x float> poison, <8 x i32> zeroinitializer
  %79 = add i64 %68, 1
  %.splatinsert16 = insertelement <8 x i64> poison, i64 %79, i64 0
  %.splat17 = shufflevector <8 x i64> %.splatinsert16, <8 x i64> poison, <8 x i32> zeroinitializer
  %80 = add <8 x i64> %72, %.splat17
  %81 = extractvalue { ptr, i64 } %5, 0
  %82 = extractelement <8 x i64> %80, i32 0
  %83 = mul i64 %82, 4
  %84 = getelementptr i8, ptr %81, i64 %83
  %85 = load float, ptr %84, align 4
  %.splatinsert18 = insertelement <8 x float> poison, float %85, i64 0
  %.splat19 = shufflevector <8 x float> %.splatinsert18, <8 x float> poison, <8 x i32> zeroinitializer
  %86 = add i64 %68, 2
  %.splatinsert20 = insertelement <8 x i64> poison, i64 %86, i64 0
  %.splat21 = shufflevector <8 x i64> %.splatinsert20, <8 x i64> poison, <8 x i32> zeroinitializer
  %87 = add <8 x i64> %72, %.splat21
  %88 = extractvalue { ptr, i64 } %5, 0
  %89 = extractelement <8 x i64> %87, i32 0
  %90 = mul i64 %89, 4
  %91 = getelementptr i8, ptr %88, i64 %90
  %92 = load float, ptr %91, align 4
  %.splatinsert22 = insertelement <8 x float> poison, float %92, i64 0
  %.splat23 = shufflevector <8 x float> %.splatinsert22, <8 x float> poison, <8 x i32> zeroinitializer
  %93 = add i64 %68, 3
  %.splatinsert24 = insertelement <8 x i64> poison, i64 %93, i64 0
  %.splat25 = shufflevector <8 x i64> %.splatinsert24, <8 x i64> poison, <8 x i32> zeroinitializer
  %94 = add <8 x i64> %72, %.splat25
  %95 = extractvalue { ptr, i64 } %5, 0
  %96 = extractelement <8 x i64> %94, i32 0
  %97 = mul i64 %96, 4
  %98 = getelementptr i8, ptr %95, i64 %97
  %99 = load float, ptr %98, align 4
  %.splatinsert26 = insertelement <8 x float> poison, float %99, i64 0
  %.splat27 = shufflevector <8 x float> %.splatinsert26, <8 x float> poison, <8 x i32> zeroinitializer
  %100 = add i64 %68, 4
  %.splatinsert28 = insertelement <8 x i64> poison, i64 %100, i64 0
  %.splat29 = shufflevector <8 x i64> %.splatinsert28, <8 x i64> poison, <8 x i32> zeroinitializer
  %101 = add <8 x i64> %72, %.splat29
  %102 = extractvalue { ptr, i64 } %5, 0
  %103 = extractelement <8 x i64> %101, i32 0
  %104 = mul i64 %103, 4
  %105 = getelementptr i8, ptr %102, i64 %104
  %106 = load float, ptr %105, align 4
  %.splatinsert30 = insertelement <8 x float> poison, float %106, i64 0
  %.splat31 = shufflevector <8 x float> %.splatinsert30, <8 x float> poison, <8 x i32> zeroinitializer
  %107 = add i64 %68, 5
  %.splatinsert32 = insertelement <8 x i64> poison, i64 %107, i64 0
  %.splat33 = shufflevector <8 x i64> %.splatinsert32, <8 x i64> poison, <8 x i32> zeroinitializer
  %108 = add <8 x i64> %72, %.splat33
  %109 = extractvalue { ptr, i64 } %5, 0
  %110 = extractelement <8 x i64> %108, i32 0
  %111 = mul i64 %110, 4
  %112 = getelementptr i8, ptr %109, i64 %111
  %113 = load float, ptr %112, align 4
  %.splatinsert34 = insertelement <8 x float> poison, float %113, i64 0
  %.splat35 = shufflevector <8 x float> %.splatinsert34, <8 x float> poison, <8 x i32> zeroinitializer
  %114 = add i64 %68, 6
  %.splatinsert36 = insertelement <8 x i64> poison, i64 %114, i64 0
  %.splat37 = shufflevector <8 x i64> %.splatinsert36, <8 x i64> poison, <8 x i32> zeroinitializer
  %115 = add <8 x i64> %72, %.splat37
  %116 = extractvalue { ptr, i64 } %5, 0
  %117 = extractelement <8 x i64> %115, i32 0
  %118 = mul i64 %117, 4
  %119 = getelementptr i8, ptr %116, i64 %118
  %120 = load float, ptr %119, align 4
  %.splatinsert38 = insertelement <8 x float> poison, float %120, i64 0
  %.splat39 = shufflevector <8 x float> %.splatinsert38, <8 x float> poison, <8 x i32> zeroinitializer
  %121 = add i64 %68, 7
  %.splatinsert40 = insertelement <8 x i64> poison, i64 %121, i64 0
  %.splat41 = shufflevector <8 x i64> %.splatinsert40, <8 x i64> poison, <8 x i32> zeroinitializer
  %122 = add <8 x i64> %72, %.splat41
  %123 = extractvalue { ptr, i64 } %5, 0
  %124 = extractelement <8 x i64> %122, i32 0
  %125 = mul i64 %124, 4
  %126 = getelementptr i8, ptr %123, i64 %125
  %127 = load float, ptr %126, align 4
  %.splatinsert42 = insertelement <8 x float> poison, float %127, i64 0
  %.splat43 = shufflevector <8 x float> %.splatinsert42, <8 x float> poison, <8 x i32> zeroinitializer
  %128 = add i64 0, %71
  %.spill.load44 = load <8 x i64>, ptr %.spill1, align 64
  %129 = add <8 x i64> %.spill.load44, zeroinitializer
  %130 = mul i64 %128, 128
  %.splatinsert45 = insertelement <8 x i64> poison, i64 %130, i64 0
  %.splat46 = shufflevector <8 x i64> %.splatinsert45, <8 x i64> poison, <8 x i32> zeroinitializer
  %131 = add <8 x i64> %.splat46, %129
  %132 = extractvalue { ptr, i64 } %11, 0
  %133 = extractelement <8 x i64> %131, i32 0
  %134 = sub i64 %133, 0
  %135 = mul i64 %134, 4
  %buffer.contiguous.address = getelementptr i8, ptr %132, i64 %135
  %buffer.contiguous.load = call <8 x float> @llvm.masked.load.v8f32.p0(ptr %buffer.contiguous.address, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %136 = add i64 0, %79
  %137 = mul i64 %136, 128
  %.splatinsert47 = insertelement <8 x i64> poison, i64 %137, i64 0
  %.splat48 = shufflevector <8 x i64> %.splatinsert47, <8 x i64> poison, <8 x i32> zeroinitializer
  %138 = add <8 x i64> %.splat48, %129
  %139 = extractvalue { ptr, i64 } %11, 0
  %140 = extractelement <8 x i64> %138, i32 0
  %141 = sub i64 %140, 0
  %142 = mul i64 %141, 4
  %buffer.contiguous.address49 = getelementptr i8, ptr %139, i64 %142
  %buffer.contiguous.load50 = call <8 x float> @llvm.masked.load.v8f32.p0(ptr %buffer.contiguous.address49, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %143 = add i64 0, %86
  %144 = mul i64 %143, 128
  %.splatinsert51 = insertelement <8 x i64> poison, i64 %144, i64 0
  %.splat52 = shufflevector <8 x i64> %.splatinsert51, <8 x i64> poison, <8 x i32> zeroinitializer
  %145 = add <8 x i64> %.splat52, %129
  %146 = extractvalue { ptr, i64 } %11, 0
  %147 = extractelement <8 x i64> %145, i32 0
  %148 = sub i64 %147, 0
  %149 = mul i64 %148, 4
  %buffer.contiguous.address53 = getelementptr i8, ptr %146, i64 %149
  %buffer.contiguous.load54 = call <8 x float> @llvm.masked.load.v8f32.p0(ptr %buffer.contiguous.address53, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %150 = add i64 0, %93
  %151 = mul i64 %150, 128
  %.splatinsert55 = insertelement <8 x i64> poison, i64 %151, i64 0
  %.splat56 = shufflevector <8 x i64> %.splatinsert55, <8 x i64> poison, <8 x i32> zeroinitializer
  %152 = add <8 x i64> %.splat56, %129
  %153 = extractvalue { ptr, i64 } %11, 0
  %154 = extractelement <8 x i64> %152, i32 0
  %155 = sub i64 %154, 0
  %156 = mul i64 %155, 4
  %buffer.contiguous.address57 = getelementptr i8, ptr %153, i64 %156
  %buffer.contiguous.load58 = call <8 x float> @llvm.masked.load.v8f32.p0(ptr %buffer.contiguous.address57, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %157 = add i64 0, %100
  %158 = mul i64 %157, 128
  %.splatinsert59 = insertelement <8 x i64> poison, i64 %158, i64 0
  %.splat60 = shufflevector <8 x i64> %.splatinsert59, <8 x i64> poison, <8 x i32> zeroinitializer
  %159 = add <8 x i64> %.splat60, %129
  %160 = extractvalue { ptr, i64 } %11, 0
  %161 = extractelement <8 x i64> %159, i32 0
  %162 = sub i64 %161, 0
  %163 = mul i64 %162, 4
  %buffer.contiguous.address61 = getelementptr i8, ptr %160, i64 %163
  %buffer.contiguous.load62 = call <8 x float> @llvm.masked.load.v8f32.p0(ptr %buffer.contiguous.address61, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %164 = add i64 0, %107
  %165 = mul i64 %164, 128
  %.splatinsert63 = insertelement <8 x i64> poison, i64 %165, i64 0
  %.splat64 = shufflevector <8 x i64> %.splatinsert63, <8 x i64> poison, <8 x i32> zeroinitializer
  %166 = add <8 x i64> %.splat64, %129
  %167 = extractvalue { ptr, i64 } %11, 0
  %168 = extractelement <8 x i64> %166, i32 0
  %169 = sub i64 %168, 0
  %170 = mul i64 %169, 4
  %buffer.contiguous.address65 = getelementptr i8, ptr %167, i64 %170
  %buffer.contiguous.load66 = call <8 x float> @llvm.masked.load.v8f32.p0(ptr %buffer.contiguous.address65, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %171 = add i64 0, %114
  %172 = mul i64 %171, 128
  %.splatinsert67 = insertelement <8 x i64> poison, i64 %172, i64 0
  %.splat68 = shufflevector <8 x i64> %.splatinsert67, <8 x i64> poison, <8 x i32> zeroinitializer
  %173 = add <8 x i64> %.splat68, %129
  %174 = extractvalue { ptr, i64 } %11, 0
  %175 = extractelement <8 x i64> %173, i32 0
  %176 = sub i64 %175, 0
  %177 = mul i64 %176, 4
  %buffer.contiguous.address69 = getelementptr i8, ptr %174, i64 %177
  %buffer.contiguous.load70 = call <8 x float> @llvm.masked.load.v8f32.p0(ptr %buffer.contiguous.address69, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %178 = add i64 0, %121
  %179 = mul i64 %178, 128
  %.splatinsert71 = insertelement <8 x i64> poison, i64 %179, i64 0
  %.splat72 = shufflevector <8 x i64> %.splatinsert71, <8 x i64> poison, <8 x i32> zeroinitializer
  %180 = add <8 x i64> %.splat72, %129
  %181 = extractvalue { ptr, i64 } %11, 0
  %182 = extractelement <8 x i64> %180, i32 0
  %183 = sub i64 %182, 0
  %184 = mul i64 %183, 4
  %buffer.contiguous.address73 = getelementptr i8, ptr %181, i64 %184
  %buffer.contiguous.load74 = call <8 x float> @llvm.masked.load.v8f32.p0(ptr %buffer.contiguous.address73, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %185 = fmul <8 x float> %.splat15, %buffer.contiguous.load
  %.state75 = load <8 x float>, ptr %.slot2, align 32
  %186 = fadd <8 x float> %.state75, %185
  %187 = fmul <8 x float> %.splat19, %buffer.contiguous.load50
  %188 = fadd <8 x float> %186, %187
  %189 = fmul <8 x float> %.splat23, %buffer.contiguous.load54
  %190 = fadd <8 x float> %188, %189
  %191 = fmul <8 x float> %.splat27, %buffer.contiguous.load58
  %192 = fadd <8 x float> %190, %191
  %193 = fmul <8 x float> %.splat31, %buffer.contiguous.load62
  %194 = fadd <8 x float> %192, %193
  %195 = fmul <8 x float> %.splat35, %buffer.contiguous.load66
  %196 = fadd <8 x float> %194, %195
  %197 = fmul <8 x float> %.splat39, %buffer.contiguous.load70
  %198 = fadd <8 x float> %196, %197
  %199 = fmul <8 x float> %.splat43, %buffer.contiguous.load74
  %200 = fadd <8 x float> %198, %199
  %.state76 = load i64, ptr %.slot, align 4
  %201 = add i64 %.state76, 1
  store i64 %201, ptr %.slot, align 4
  %202 = load <8 x float>, ptr %.slot2, align 32
  %203 = select <8 x i1> %41, <8 x float> %200, <8 x float> %202
  store <8 x float> %203, ptr %.slot2, align 32
  br label %direct.schedule.1

direct.schedule.3:                                ; preds = %direct.false
  %.spill.load77 = load <8 x i64>, ptr %.spill, align 64
  %204 = add <8 x i64> %.spill.load77, zeroinitializer
  %205 = add <8 x i64> zeroinitializer, %204
  %.spill.load78 = load <8 x i64>, ptr %.spill1, align 64
  %206 = add <8 x i64> %.spill.load78, zeroinitializer
  %207 = mul <8 x i64> %205, splat (i64 128)
  %208 = add <8 x i64> %207, %206
  %.state79 = load <8 x float>, ptr %.slot2, align 32
  %209 = extractvalue { ptr, i64 } %17, 0
  %210 = extractelement <8 x i64> %208, i32 0
  %211 = sub i64 %210, 0
  %212 = mul i64 %211, 4
  %buffer.contiguous.address80 = getelementptr i8, ptr %209, i64 %212
  call void @llvm.masked.store.v8f32.p0(<8 x float> %.state79, ptr %buffer.contiguous.address80, i32 1, <8 x i1> %41)
  ret void

direct.activate:                                  ; preds = %prologue
  br label %direct.schedule.0

direct.inactive:                                  ; preds = %prologue
  ret void

direct.true:                                      ; preds = %direct.schedule.1
  br label %direct.schedule.2

direct.false:                                     ; preds = %direct.schedule.1
  br label %direct.schedule.3
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.vector.reduce.or.v8i1(<8 x i1>) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <8 x float> @llvm.masked.load.v8f32.p0(ptr captures(none), i32 immarg, <8 x i1>, <8 x float>) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.masked.store.v8f32.p0(<8 x float>, ptr captures(none), i32 immarg, <8 x i1>) #2

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
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
