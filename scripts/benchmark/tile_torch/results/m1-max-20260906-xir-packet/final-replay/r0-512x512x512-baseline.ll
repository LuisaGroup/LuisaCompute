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
  %46 = select <8 x i1> %41, <8 x i64> splat (i64 512), <8 x i64> splat (i64 1)
  %47 = sdiv <8 x i64> %45, %46
  %48 = select <8 x i1> %41, <8 x i64> %47, <8 x i64> zeroinitializer
  %49 = select <8 x i1> %41, <8 x i64> splat (i64 512), <8 x i64> splat (i64 1)
  %50 = srem <8 x i64> %48, %49
  %51 = select <8 x i1> %41, <8 x i64> %44, <8 x i64> zeroinitializer
  %52 = select <8 x i1> %41, <8 x i64> splat (i64 1), <8 x i64> splat (i64 1)
  %53 = sdiv <8 x i64> %51, %52
  %54 = select <8 x i1> %41, <8 x i64> %53, <8 x i64> zeroinitializer
  %55 = select <8 x i1> %41, <8 x i64> splat (i64 512), <8 x i64> splat (i64 1)
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
  %65 = icmp slt i64 %.state, 64
  br i1 %65, label %direct.true, label %direct.false

direct.schedule.2:                                ; preds = %direct.true
  %.state11 = load i64, ptr %.slot, align 4
  %66 = sdiv i64 %.state11, 1
  %67 = srem i64 %66, 64
  %68 = mul i64 %67, 8
  %.spill.load = load <8 x i64>, ptr %.spill, align 64
  %69 = add <8 x i64> %.spill.load, zeroinitializer
  %70 = add <8 x i64> zeroinitializer, %69
  %71 = add i64 %68, 0
  %72 = mul <8 x i64> %70, splat (i64 512)
  %.splatinsert12 = insertelement <8 x i64> poison, i64 %71, i64 0
  %.splat13 = shufflevector <8 x i64> %.splatinsert12, <8 x i64> poison, <8 x i32> zeroinitializer
  %73 = add <8 x i64> %72, %.splat13
  %74 = extractvalue { ptr, i64 } %5, 0
  %75 = mul <8 x i64> %73, splat (i64 4)
  %76 = getelementptr i8, ptr %74, <8 x i64> %75
  %77 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %76, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %78 = add i64 %68, 1
  %.splatinsert14 = insertelement <8 x i64> poison, i64 %78, i64 0
  %.splat15 = shufflevector <8 x i64> %.splatinsert14, <8 x i64> poison, <8 x i32> zeroinitializer
  %79 = add <8 x i64> %72, %.splat15
  %80 = extractvalue { ptr, i64 } %5, 0
  %81 = mul <8 x i64> %79, splat (i64 4)
  %82 = getelementptr i8, ptr %80, <8 x i64> %81
  %83 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %82, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %84 = add i64 %68, 2
  %.splatinsert16 = insertelement <8 x i64> poison, i64 %84, i64 0
  %.splat17 = shufflevector <8 x i64> %.splatinsert16, <8 x i64> poison, <8 x i32> zeroinitializer
  %85 = add <8 x i64> %72, %.splat17
  %86 = extractvalue { ptr, i64 } %5, 0
  %87 = mul <8 x i64> %85, splat (i64 4)
  %88 = getelementptr i8, ptr %86, <8 x i64> %87
  %89 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %88, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %90 = add i64 %68, 3
  %.splatinsert18 = insertelement <8 x i64> poison, i64 %90, i64 0
  %.splat19 = shufflevector <8 x i64> %.splatinsert18, <8 x i64> poison, <8 x i32> zeroinitializer
  %91 = add <8 x i64> %72, %.splat19
  %92 = extractvalue { ptr, i64 } %5, 0
  %93 = mul <8 x i64> %91, splat (i64 4)
  %94 = getelementptr i8, ptr %92, <8 x i64> %93
  %95 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %94, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %96 = add i64 %68, 4
  %.splatinsert20 = insertelement <8 x i64> poison, i64 %96, i64 0
  %.splat21 = shufflevector <8 x i64> %.splatinsert20, <8 x i64> poison, <8 x i32> zeroinitializer
  %97 = add <8 x i64> %72, %.splat21
  %98 = extractvalue { ptr, i64 } %5, 0
  %99 = mul <8 x i64> %97, splat (i64 4)
  %100 = getelementptr i8, ptr %98, <8 x i64> %99
  %101 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %100, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %102 = add i64 %68, 5
  %.splatinsert22 = insertelement <8 x i64> poison, i64 %102, i64 0
  %.splat23 = shufflevector <8 x i64> %.splatinsert22, <8 x i64> poison, <8 x i32> zeroinitializer
  %103 = add <8 x i64> %72, %.splat23
  %104 = extractvalue { ptr, i64 } %5, 0
  %105 = mul <8 x i64> %103, splat (i64 4)
  %106 = getelementptr i8, ptr %104, <8 x i64> %105
  %107 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %106, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %108 = add i64 %68, 6
  %.splatinsert24 = insertelement <8 x i64> poison, i64 %108, i64 0
  %.splat25 = shufflevector <8 x i64> %.splatinsert24, <8 x i64> poison, <8 x i32> zeroinitializer
  %109 = add <8 x i64> %72, %.splat25
  %110 = extractvalue { ptr, i64 } %5, 0
  %111 = mul <8 x i64> %109, splat (i64 4)
  %112 = getelementptr i8, ptr %110, <8 x i64> %111
  %113 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %112, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %114 = add i64 %68, 7
  %.splatinsert26 = insertelement <8 x i64> poison, i64 %114, i64 0
  %.splat27 = shufflevector <8 x i64> %.splatinsert26, <8 x i64> poison, <8 x i32> zeroinitializer
  %115 = add <8 x i64> %72, %.splat27
  %116 = extractvalue { ptr, i64 } %5, 0
  %117 = mul <8 x i64> %115, splat (i64 4)
  %118 = getelementptr i8, ptr %116, <8 x i64> %117
  %119 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %118, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %120 = add i64 0, %71
  %.spill.load28 = load <8 x i64>, ptr %.spill1, align 64
  %121 = add <8 x i64> %.spill.load28, zeroinitializer
  %122 = mul i64 %120, 512
  %.splatinsert29 = insertelement <8 x i64> poison, i64 %122, i64 0
  %.splat30 = shufflevector <8 x i64> %.splatinsert29, <8 x i64> poison, <8 x i32> zeroinitializer
  %123 = add <8 x i64> %.splat30, %121
  %124 = extractvalue { ptr, i64 } %11, 0
  %125 = mul <8 x i64> %123, splat (i64 4)
  %126 = getelementptr i8, ptr %124, <8 x i64> %125
  %127 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %126, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %128 = add i64 0, %78
  %129 = mul i64 %128, 512
  %.splatinsert31 = insertelement <8 x i64> poison, i64 %129, i64 0
  %.splat32 = shufflevector <8 x i64> %.splatinsert31, <8 x i64> poison, <8 x i32> zeroinitializer
  %130 = add <8 x i64> %.splat32, %121
  %131 = extractvalue { ptr, i64 } %11, 0
  %132 = mul <8 x i64> %130, splat (i64 4)
  %133 = getelementptr i8, ptr %131, <8 x i64> %132
  %134 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %133, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %135 = add i64 0, %84
  %136 = mul i64 %135, 512
  %.splatinsert33 = insertelement <8 x i64> poison, i64 %136, i64 0
  %.splat34 = shufflevector <8 x i64> %.splatinsert33, <8 x i64> poison, <8 x i32> zeroinitializer
  %137 = add <8 x i64> %.splat34, %121
  %138 = extractvalue { ptr, i64 } %11, 0
  %139 = mul <8 x i64> %137, splat (i64 4)
  %140 = getelementptr i8, ptr %138, <8 x i64> %139
  %141 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %140, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %142 = add i64 0, %90
  %143 = mul i64 %142, 512
  %.splatinsert35 = insertelement <8 x i64> poison, i64 %143, i64 0
  %.splat36 = shufflevector <8 x i64> %.splatinsert35, <8 x i64> poison, <8 x i32> zeroinitializer
  %144 = add <8 x i64> %.splat36, %121
  %145 = extractvalue { ptr, i64 } %11, 0
  %146 = mul <8 x i64> %144, splat (i64 4)
  %147 = getelementptr i8, ptr %145, <8 x i64> %146
  %148 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %147, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %149 = add i64 0, %96
  %150 = mul i64 %149, 512
  %.splatinsert37 = insertelement <8 x i64> poison, i64 %150, i64 0
  %.splat38 = shufflevector <8 x i64> %.splatinsert37, <8 x i64> poison, <8 x i32> zeroinitializer
  %151 = add <8 x i64> %.splat38, %121
  %152 = extractvalue { ptr, i64 } %11, 0
  %153 = mul <8 x i64> %151, splat (i64 4)
  %154 = getelementptr i8, ptr %152, <8 x i64> %153
  %155 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %154, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %156 = add i64 0, %102
  %157 = mul i64 %156, 512
  %.splatinsert39 = insertelement <8 x i64> poison, i64 %157, i64 0
  %.splat40 = shufflevector <8 x i64> %.splatinsert39, <8 x i64> poison, <8 x i32> zeroinitializer
  %158 = add <8 x i64> %.splat40, %121
  %159 = extractvalue { ptr, i64 } %11, 0
  %160 = mul <8 x i64> %158, splat (i64 4)
  %161 = getelementptr i8, ptr %159, <8 x i64> %160
  %162 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %161, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %163 = add i64 0, %108
  %164 = mul i64 %163, 512
  %.splatinsert41 = insertelement <8 x i64> poison, i64 %164, i64 0
  %.splat42 = shufflevector <8 x i64> %.splatinsert41, <8 x i64> poison, <8 x i32> zeroinitializer
  %165 = add <8 x i64> %.splat42, %121
  %166 = extractvalue { ptr, i64 } %11, 0
  %167 = mul <8 x i64> %165, splat (i64 4)
  %168 = getelementptr i8, ptr %166, <8 x i64> %167
  %169 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %168, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %170 = add i64 0, %114
  %171 = mul i64 %170, 512
  %.splatinsert43 = insertelement <8 x i64> poison, i64 %171, i64 0
  %.splat44 = shufflevector <8 x i64> %.splatinsert43, <8 x i64> poison, <8 x i32> zeroinitializer
  %172 = add <8 x i64> %.splat44, %121
  %173 = extractvalue { ptr, i64 } %11, 0
  %174 = mul <8 x i64> %172, splat (i64 4)
  %175 = getelementptr i8, ptr %173, <8 x i64> %174
  %176 = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %175, i32 1, <8 x i1> %41, <8 x float> zeroinitializer)
  %177 = fmul <8 x float> %77, %127
  %.state45 = load <8 x float>, ptr %.slot2, align 32
  %178 = fadd <8 x float> %.state45, %177
  %179 = fmul <8 x float> %83, %134
  %180 = fadd <8 x float> %178, %179
  %181 = fmul <8 x float> %89, %141
  %182 = fadd <8 x float> %180, %181
  %183 = fmul <8 x float> %95, %148
  %184 = fadd <8 x float> %182, %183
  %185 = fmul <8 x float> %101, %155
  %186 = fadd <8 x float> %184, %185
  %187 = fmul <8 x float> %107, %162
  %188 = fadd <8 x float> %186, %187
  %189 = fmul <8 x float> %113, %169
  %190 = fadd <8 x float> %188, %189
  %191 = fmul <8 x float> %119, %176
  %192 = fadd <8 x float> %190, %191
  %.state46 = load i64, ptr %.slot, align 4
  %193 = add i64 %.state46, 1
  store i64 %193, ptr %.slot, align 4
  %194 = load <8 x float>, ptr %.slot2, align 32
  %195 = select <8 x i1> %41, <8 x float> %192, <8 x float> %194
  store <8 x float> %195, ptr %.slot2, align 32
  br label %direct.schedule.1

direct.schedule.3:                                ; preds = %direct.false
  %.spill.load47 = load <8 x i64>, ptr %.spill, align 64
  %196 = add <8 x i64> %.spill.load47, zeroinitializer
  %197 = add <8 x i64> zeroinitializer, %196
  %.spill.load48 = load <8 x i64>, ptr %.spill1, align 64
  %198 = add <8 x i64> %.spill.load48, zeroinitializer
  %199 = mul <8 x i64> %197, splat (i64 512)
  %200 = add <8 x i64> %199, %198
  %.state49 = load <8 x float>, ptr %.slot2, align 32
  %201 = extractvalue { ptr, i64 } %17, 0
  %202 = mul <8 x i64> %200, splat (i64 4)
  %203 = getelementptr i8, ptr %201, <8 x i64> %202
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %.state49, <8 x ptr> %203, i32 1, <8 x i1> %41)
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
