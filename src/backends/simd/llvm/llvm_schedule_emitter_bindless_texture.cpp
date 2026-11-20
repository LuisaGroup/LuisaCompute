#include "llvm_schedule_emitter.h"

#include <limits>

#include <llvm/IR/MDBuilder.h>

#include "../../common/env_flag.h"
#include "../../common/llvm_native_math.h"
#include <luisa/core/stl/string.h>

namespace luisa::compute::simd::detail {

namespace {

struct TextureLinearCoordinateIR {
    ::llvm::Value *lo;
    ::llvm::Value *hi;
    ::llvm::Value *t;
    ::llvm::Value *valid;
};

[[nodiscard]] ::llvm::Constant *float_vector_constant(
    ::llvm::IRBuilder<> &builder, uint32_t width, float value) {
    return ::llvm::ConstantVector::getSplat(
        ::llvm::ElementCount::getFixed(width),
        ::llvm::ConstantFP::get(builder.getFloatTy(), value));
}

[[nodiscard]] TextureLinearCoordinateIR
emit_mirror_linear_coordinate(
    ::llvm::IRBuilder<> &builder, uint32_t width,
    ::llvm::Value *coordinate, ::llvm::Value *size) {
    auto *zero = float_vector_constant(builder, width, 0.0f);
    auto *half = float_vector_constant(builder, width, 0.5f);
    auto *one = float_vector_constant(builder, width, 1.0f);
    auto *two = float_vector_constant(builder, width, 2.0f);
    auto *one_minus_epsilon = float_vector_constant(
        builder, width, 0x1.fffffep-1f);
    auto *infinity = ::llvm::ConstantVector::getSplat(
        ::llvm::ElementCount::getFixed(width),
        ::llvm::ConstantFP::getInfinity(builder.getFloatTy()));
    auto *absolute = builder.CreateUnaryIntrinsic(
        ::llvm::Intrinsic::fabs, coordinate);
    auto *finite = builder.CreateFCmpOLT(
        absolute, infinity, "bindless.texture.byte1.finite");
    auto *half_inverse_size = builder.CreateFDiv(half, size);

    auto mirror = [&](::llvm::Value *value) {
        value = builder.CreateUnaryIntrinsic(
            ::llvm::Intrinsic::fabs, value);
        auto *period = builder.CreateUnaryIntrinsic(
            ::llvm::Intrinsic::floor,
            builder.CreateFMul(value, half));
        value = builder.CreateFSub(
            value, builder.CreateFMul(period, two));
        value = builder.CreateSelect(
            builder.CreateFCmpOLT(value, one), value,
            builder.CreateFSub(two, value));
        value = builder.CreateMinNum(value, one_minus_epsilon);
        return builder.CreateFMul(value, size);
    };
    auto *a = mirror(builder.CreateFSub(
        coordinate, half_inverse_size));
    auto *b = mirror(builder.CreateFAdd(
        coordinate, half_inverse_size));
    auto *lo = builder.CreateMinNum(a, b);
    auto *hi = builder.CreateMaxNum(a, b);
    auto *reduced_in_range = builder.CreateAnd(
        builder.CreateAnd(
            builder.CreateFCmpOGE(lo, zero),
            builder.CreateFCmpOLT(lo, size)),
        builder.CreateAnd(
            builder.CreateFCmpOGE(hi, zero),
            builder.CreateFCmpOLT(hi, size)));
    auto *valid = builder.CreateAnd(
        finite, reduced_in_range,
        "bindless.texture.byte1.reduced.valid");
    auto *safe_lo = builder.CreateSelect(valid, lo, zero);
    auto *safe_hi = builder.CreateSelect(valid, hi, zero);
    auto *i32_lanes = ::llvm::FixedVectorType::get(
        builder.getInt32Ty(), width);
    auto *lo_index = builder.CreateFPToUI(
        safe_lo, i32_lanes,
        "bindless.texture.byte1.linear.lo");
    auto *hi_index = builder.CreateFPToUI(
        safe_hi, i32_lanes,
        "bindless.texture.byte1.linear.hi");
    auto *fraction = builder.CreateFSub(
        safe_hi, builder.CreateUnaryIntrinsic(::llvm::Intrinsic::floor, safe_hi),
        "bindless.texture.byte1.linear.fraction");
    return {lo_index, hi_index, fraction, valid};
}

void emit_byte1_linear_mirror_sample(
    ::llvm::IRBuilder<> &builder, uint32_t width,
    ::llvm::Value *active_mask, ::llvm::Value *data,
    ::llvm::Value *width_value, ::llvm::Value *height_value,
    ::llvm::Value *u, ::llvm::Value *v,
    ::llvm::AllocaInst *scratch,
    ::llvm::ArrayType *scratch_type,
    bool enable_wide_gathers, bool little_endian) {
    auto *float_lanes = ::llvm::FixedVectorType::get(
        builder.getFloatTy(), width);
    auto *i8_lanes = ::llvm::FixedVectorType::get(
        builder.getInt8Ty(), width);
    auto *i32_lanes = ::llvm::FixedVectorType::get(
        builder.getInt32Ty(), width);
    auto *i64_lanes = ::llvm::FixedVectorType::get(
        builder.getInt64Ty(), width);
    auto *width_float = builder.CreateVectorSplat(
        width, builder.CreateUIToFP(
                   width_value, builder.getFloatTy()));
    auto *height_float = builder.CreateVectorSplat(
        width, builder.CreateUIToFP(
                   height_value, builder.getFloatTy()));
    auto x = emit_mirror_linear_coordinate(
        builder, width, u, width_float);
    auto y = emit_mirror_linear_coordinate(
        builder, width, v, height_float);
    auto *sample_mask = builder.CreateAnd(
        active_mask, builder.CreateAnd(x.valid, y.valid),
        "bindless.texture.byte1.sample.mask");
    auto *width_i64 = builder.CreateVectorSplat(
        width, builder.CreateZExt(
                   width_value, builder.getInt64Ty()));
    auto *data_lanes = builder.CreateVectorSplat(width, data);
    auto offset = [&](::llvm::Value *x_index,
                      ::llvm::Value *y_index) {
        auto *x_i64 = builder.CreateZExt(x_index, i64_lanes);
        auto *y_i64 = builder.CreateZExt(y_index, i64_lanes);
        return builder.CreateAdd(
            builder.CreateMul(y_i64, width_i64), x_i64);
    };
    std::array<::llvm::Value *, 4u> offsets{
        offset(x.lo, y.lo), offset(x.hi, y.lo),
        offset(x.lo, y.hi), offset(x.hi, y.hi)};
    std::array<luisa::string_view, 4u> names{
        "bindless.texture.byte1.v00",
        "bindless.texture.byte1.v01",
        "bindless.texture.byte1.v10",
        "bindless.texture.byte1.v11"};
    auto narrow_gather = [&](::llvm::Value *texel_offset,
                             luisa::string_view name) {
        auto *pointers = builder.CreateGEP(
            builder.getInt8Ty(), data_lanes, texel_offset);
        auto *bytes = builder.CreateMaskedGather(
            i8_lanes, pointers, ::llvm::Align{1u},
            sample_mask, ::llvm::Constant::getNullValue(i8_lanes),
            name);
        return builder.CreateZExt(bytes, i32_lanes);
    };
    std::array<::llvm::Value *, 4u> texels{};
    if (enable_wide_gathers) {
        auto *width_scalar_i64 = builder.CreateZExt(
            width_value, builder.getInt64Ty());
        auto *height_scalar_i64 = builder.CreateZExt(
            height_value, builder.getInt64Ty());
        auto *pixel_count = builder.CreateMul(
            width_scalar_i64, height_scalar_i64);
        auto *has_wide_load = builder.CreateICmpUGE(
            pixel_count, builder.getInt64(4u));
        auto *last_wide_offset = builder.CreateSelect(
            has_wide_load,
            builder.CreateSub(pixel_count, builder.getInt64(4u)),
            builder.getInt64(0u));
        auto *last_wide_offsets = builder.CreateVectorSplat(
            width, last_wide_offset);
        ::llvm::Value *safe_lanes = ::llvm::Constant::getAllOnesValue(
            sample_mask->getType());
        for (auto *texel_offset : offsets) {
            auto *safe_offset = builder.CreateICmpULE(
                texel_offset, last_wide_offsets);
            safe_lanes = builder.CreateAnd(
                safe_lanes,
                builder.CreateOr(
                    builder.CreateNot(sample_mask), safe_offset));
        }
        auto *use_wide_gathers = builder.CreateAnd(
            has_wide_load, builder.CreateAndReduce(safe_lanes),
            "bindless.texture.byte1.wide.eligible");
        auto *function = builder.GetInsertBlock()->getParent();
        auto *wide = ::llvm::BasicBlock::Create(
            builder.getContext(),
            "bindless.texture.byte1.wide", function);
        auto *narrow = ::llvm::BasicBlock::Create(
            builder.getContext(),
            "bindless.texture.byte1.narrow", function);
        auto *gather_merge = ::llvm::BasicBlock::Create(
            builder.getContext(),
            "bindless.texture.byte1.gather.merge", function);
        builder.CreateCondBr(use_wide_gathers, wide, narrow);

        builder.SetInsertPoint(wide);
        std::array<::llvm::Value *, 4u> wide_texels{};
        for (auto i = 0u; i < offsets.size(); i++) {
            auto *pointers = builder.CreateGEP(
                builder.getInt8Ty(), data_lanes, offsets[i]);
            ::llvm::Value *words = builder.CreateMaskedGather(
                i32_lanes, pointers, ::llvm::Align{1u},
                sample_mask,
                ::llvm::Constant::getNullValue(i32_lanes),
                std::string{names[i]} + ".wide");
            if (!little_endian) {
                words = builder.CreateLShr(
                    words, builder.CreateVectorSplat(
                               width, builder.getInt32(24u)));
            }
            wide_texels[i] = builder.CreateAnd(
                words, builder.CreateVectorSplat(
                           width, builder.getInt32(0xffu)));
        }
        builder.CreateBr(gather_merge);
        auto *wide_end = builder.GetInsertBlock();

        builder.SetInsertPoint(narrow);
        std::array<::llvm::Value *, 4u> narrow_texels{};
        for (auto i = 0u; i < offsets.size(); i++) {
            narrow_texels[i] = narrow_gather(
                offsets[i], names[i]);
        }
        builder.CreateBr(gather_merge);
        auto *narrow_end = builder.GetInsertBlock();

        builder.SetInsertPoint(gather_merge);
        for (auto i = 0u; i < offsets.size(); i++) {
            auto *phi = builder.CreatePHI(
                i32_lanes, 2u,
                std::string{names[i]} + ".gathered");
            phi->addIncoming(wide_texels[i], wide_end);
            phi->addIncoming(narrow_texels[i], narrow_end);
            texels[i] = phi;
        }
    } else {
        for (auto i = 0u; i < offsets.size(); i++) {
            texels[i] = narrow_gather(offsets[i], names[i]);
        }
    }
    auto normalize = [&](::llvm::Value *bytes) {
        return builder.CreateFMul(
            builder.CreateUIToFP(bytes, float_lanes),
            float_vector_constant(builder, width, 1.0f / 255.0f));
    };
    auto *v00 = normalize(texels[0u]);
    auto *v01 = normalize(texels[1u]);
    auto *v10 = normalize(texels[2u]);
    auto *v11 = normalize(texels[3u]);
    auto lerp = [&](::llvm::Value *lhs, ::llvm::Value *rhs,
                    ::llvm::Value *t) {
        return builder.CreateFAdd(
            lhs, builder.CreateFMul(
                     builder.CreateFSub(rhs, lhs), t));
    };
    auto *value = lerp(
        lerp(v00, v01, x.t),
        lerp(v10, v11, x.t), y.t);
    value = builder.CreateSelect(
        sample_mask, value,
        ::llvm::Constant::getNullValue(float_lanes),
        "bindless.texture.byte1.result");
    auto *component = builder.CreateGEP(
        scratch_type, scratch,
        {builder.getInt32(0u), builder.getInt32(0u)});
    builder.CreateStore(value, component);
}

[[nodiscard]] ::llvm::Function *get_or_create_w1_rgba_linear_sample_helper(
    ::llvm::Module &module, bool is_float4) {
    auto fn_name = is_float4 ?
                       std::string{"__luisa_simd_bindless_sample2d_float4_linear_w1"} :
                       std::string{"__luisa_simd_bindless_sample2d_byte4_linear_w1"};
    if (auto *existing = module.getFunction(fn_name)) {
        return existing;
    }
    auto &ctx = module.getContext();
    auto *ptr_ty = ::llvm::PointerType::getUnqual(ctx);
    auto *i8_ty = ::llvm::Type::getInt8Ty(ctx);
    auto *i32_ty = ::llvm::Type::getInt32Ty(ctx);
    auto *i64_ty = ::llvm::Type::getInt64Ty(ctx);
    auto *f32_ty = ::llvm::Type::getFloatTy(ctx);
    auto *f32x4_ty = ::llvm::FixedVectorType::get(f32_ty, 4u);
    auto *i32x4_ty = ::llvm::FixedVectorType::get(i32_ty, 4u);
    auto *i8x4_ty = ::llvm::FixedVectorType::get(i8_ty, 4u);

    auto *fn_ty = ::llvm::FunctionType::get(
        f32x4_ty, {ptr_ty, i64_ty, f32_ty, f32_ty}, false);
    auto *fn = ::llvm::Function::Create(
        fn_ty, ::llvm::GlobalValue::InternalLinkage, fn_name, module);
    fn->setDSOLocal(true);
    fn->setCallingConv(::llvm::CallingConv::Fast);
    fn->addFnAttr(::llvm::Attribute::NoInline);
    fn->addFnAttr(::llvm::Attribute::NoUnwind);
    fn->addFnAttr(::llvm::Attribute::WillReturn);
    fn->addParamAttr(0u, ::llvm::Attribute::ReadOnly);
    fn->addParamAttr(0u, ::llvm::Attribute::NonNull);

    auto *entry_bb = ::llvm::BasicBlock::Create(ctx, "entry", fn);
    ::llvm::IRBuilder<> b{entry_bb};
    auto *data = fn->getArg(0u);
    auto *metadata = fn->getArg(1u);
    auto *u_arg = fn->getArg(2u);
    auto *v_arg = fn->getArg(3u);
    auto *width_val = b.CreateTrunc(
        b.CreateAnd(
            b.CreateLShr(metadata, b.getInt64(4u)),
            b.getInt64(simd_bindless_texture_extent_mask)),
        i32_ty);
    auto *height_val = b.CreateTrunc(
        b.CreateAnd(
            b.CreateLShr(metadata, b.getInt64(24u)),
            b.getInt64(simd_bindless_texture_extent_mask)),
        i32_ty);
    auto *addr_mode = b.CreateTrunc(
        b.CreateAnd(metadata, b.getInt64(0x03u)), i32_ty);

    auto *zero4 = float_vector_constant(b, 4u, 0.0f);
    auto *half4 = float_vector_constant(b, 4u, 0.5f);
    auto *one4 = float_vector_constant(b, 4u, 1.0f);
    auto *two4 = float_vector_constant(b, 4u, 2.0f);
    auto *one_minus_eps4 = float_vector_constant(b, 4u, 0x1.fffffep-1f);
    auto *inf_f = ::llvm::ConstantFP::getInfinity(f32_ty);

    // Pack [u - 0.5/w, u + 0.5/w, v - 0.5/h, v + 0.5/h] into one <4 x float>
    // so both axes share a single SIMD coordinate reduction and single switch.
    auto *w_f = b.CreateUIToFP(width_val, f32_ty);
    auto *h_f = b.CreateUIToFP(height_val, f32_ty);
    ::llvm::Value *size4 = ::llvm::PoisonValue::get(f32x4_ty);
    size4 = b.CreateInsertElement(size4, w_f, b.getInt32(0u));
    size4 = b.CreateInsertElement(size4, w_f, b.getInt32(1u));
    size4 = b.CreateInsertElement(size4, h_f, b.getInt32(2u));
    size4 = b.CreateInsertElement(size4, h_f, b.getInt32(3u));
    auto *half_inv4 = b.CreateFDiv(half4, size4);
    auto *sign_mask = ::llvm::ConstantVector::get({
        ::llvm::ConstantFP::get(f32_ty, -1.0f),
        ::llvm::ConstantFP::get(f32_ty, 1.0f),
        ::llvm::ConstantFP::get(f32_ty, -1.0f),
        ::llvm::ConstantFP::get(f32_ty, 1.0f),
    });
    ::llvm::Value *uv4 = ::llvm::PoisonValue::get(f32x4_ty);
    uv4 = b.CreateInsertElement(uv4, u_arg, b.getInt32(0u));
    uv4 = b.CreateInsertElement(uv4, u_arg, b.getInt32(1u));
    uv4 = b.CreateInsertElement(uv4, v_arg, b.getInt32(2u));
    uv4 = b.CreateInsertElement(uv4, v_arg, b.getInt32(3u));
    auto *coords_in = b.CreateIntrinsic(
        ::llvm::Intrinsic::fma, {f32x4_ty}, {half_inv4, sign_mask, uv4});

    auto *fin_u = b.CreateFCmpOLT(
        b.CreateUnaryIntrinsic(::llvm::Intrinsic::fabs, u_arg), inf_f);
    auto *fin_v = b.CreateFCmpOLT(
        b.CreateUnaryIntrinsic(::llvm::Intrinsic::fabs, v_arg), inf_f);
    auto *i1x4_ty = ::llvm::FixedVectorType::get(b.getInt1Ty(), 4u);
    ::llvm::Value *fin4 = ::llvm::PoisonValue::get(i1x4_ty);
    fin4 = b.CreateInsertElement(fin4, fin_u, b.getInt32(0u));
    fin4 = b.CreateInsertElement(fin4, fin_u, b.getInt32(1u));
    fin4 = b.CreateInsertElement(fin4, fin_v, b.getInt32(2u));
    fin4 = b.CreateInsertElement(fin4, fin_v, b.getInt32(3u));

    auto *repeat_bb = ::llvm::BasicBlock::Create(ctx, "repeat", fn);
    auto *edge_bb = ::llvm::BasicBlock::Create(ctx, "edge", fn);
    auto *mirror_bb = ::llvm::BasicBlock::Create(ctx, "mirror", fn);
    auto *merge_bb = ::llvm::BasicBlock::Create(ctx, "merge", fn);

    auto *sw = b.CreateSwitch(addr_mode, repeat_bb, 2u);
    sw->addCase(b.getInt32(0u), edge_bb);
    sw->addCase(b.getInt32(2u), mirror_bb);

    // REPEAT (1)
    b.SetInsertPoint(repeat_bb);
    auto *rep_pos = b.CreateFMul(
        b.CreateFSub(
            coords_in, b.CreateUnaryIntrinsic(::llvm::Intrinsic::floor, coords_in)),
        size4);
    b.CreateBr(merge_bb);

    // EDGE (0)
    b.SetInsertPoint(edge_bb);
    auto *edge_pos = b.CreateFMul(
        b.CreateMinNum(b.CreateMaxNum(coords_in, zero4), one_minus_eps4), size4);
    b.CreateBr(merge_bb);

    // MIRROR (2)
    b.SetInsertPoint(mirror_bb);
    auto *mir_abs = b.CreateUnaryIntrinsic(::llvm::Intrinsic::fabs, coords_in);
    auto *mir_per = b.CreateUnaryIntrinsic(
        ::llvm::Intrinsic::floor, b.CreateFMul(mir_abs, half4));
    auto *mir_fold = b.CreateFSub(mir_abs, b.CreateFMul(mir_per, two4));
    mir_fold = b.CreateSelect(
        b.CreateFCmpOLT(mir_fold, one4), mir_fold, b.CreateFSub(two4, mir_fold));
    auto *mir_scaled = b.CreateFMul(
        b.CreateMinNum(mir_fold, one_minus_eps4), size4);
    auto *mir_swap = b.CreateShuffleVector(mir_scaled, {1, 0, 3, 2});
    auto *mir_min = b.CreateMinNum(mir_scaled, mir_swap);
    auto *mir_max = b.CreateMaxNum(mir_scaled, mir_swap);
    auto *mir_pos = b.CreateShuffleVector(mir_min, mir_max, {0, 5, 2, 7});
    b.CreateBr(merge_bb);

    b.SetInsertPoint(merge_bb);
    auto *pos4 = b.CreatePHI(f32x4_ty, 3u);
    pos4->addIncoming(rep_pos, repeat_bb);
    pos4->addIncoming(edge_pos, edge_bb);
    pos4->addIncoming(mir_pos, mirror_bb);

    auto *valid4 = b.CreateAnd(
        fin4, b.CreateAnd(b.CreateFCmpOGE(pos4, zero4), b.CreateFCmpOLT(pos4, size4)));
    auto *safe_pos4 = b.CreateSelect(valid4, pos4, zero4);
    auto *idx4 = b.CreateFPToUI(safe_pos4, i32x4_ty);
    auto *frac4 = b.CreateFSub(safe_pos4, b.CreateUIToFP(idx4, f32x4_ty));
    auto *xt_v4 = b.CreateShuffleVector(frac4, {1, 1, 1, 1});
    auto *yt_v4 = b.CreateShuffleVector(frac4, {3, 3, 3, 3});

    auto *u_lo = b.CreateZExt(b.CreateExtractElement(idx4, b.getInt32(0u)), i64_ty);
    auto *u_hi = b.CreateZExt(b.CreateExtractElement(idx4, b.getInt32(1u)), i64_ty);
    auto *v_lo = b.CreateZExt(b.CreateExtractElement(idx4, b.getInt32(2u)), i64_ty);
    auto *v_hi = b.CreateZExt(b.CreateExtractElement(idx4, b.getInt32(3u)), i64_ty);
    auto *w64 = b.CreateZExt(width_val, i64_ty);
    auto *row0_base = b.CreateMul(v_lo, w64);
    auto *row1_base = b.CreateMul(v_hi, w64);
    auto *off00 = b.CreateAdd(row0_base, u_lo);
    auto *off01 = b.CreateAdd(row0_base, u_hi);
    auto *off10 = b.CreateAdd(row1_base, u_lo);
    auto *off11 = b.CreateAdd(row1_base, u_hi);

    auto load_texel = [&](::llvm::Value *off) -> ::llvm::Value * {
        if (is_float4) {
            auto *ptr = b.CreateGEP(f32x4_ty, data, off);
            auto *ld = b.CreateLoad(f32x4_ty, ptr);
            ld->setAlignment(::llvm::Align{4u});
            return ld;
        }
        auto *ptr = b.CreateGEP(i8x4_ty, data, off);
        auto *bytes = b.CreateLoad(i8x4_ty, ptr);
        bytes->setAlignment(::llvm::Align{4u});
        return b.CreateUIToFP(b.CreateZExt(bytes, i32x4_ty), f32x4_ty);
    };

    auto *all_valid = b.CreateAndReduce(valid4);
    auto *fast_bb = ::llvm::BasicBlock::Create(ctx, "fast", fn);
    auto *slow_bb = ::llvm::BasicBlock::Create(ctx, "slow", fn);
    auto *blend_bb = ::llvm::BasicBlock::Create(ctx, "blend", fn);
    auto *br = b.CreateCondBr(all_valid, fast_bb, slow_bb);
    br->setMetadata(
        ::llvm::LLVMContext::MD_prof,
        ::llvm::MDBuilder{ctx}.createBranchWeights(1000000u, 1u));

    b.SetInsertPoint(fast_bb);
    auto *f00 = load_texel(off00);
    auto *f01 = load_texel(off01);
    auto *f10 = load_texel(off10);
    auto *f11 = load_texel(off11);
    b.CreateBr(blend_bb);

    b.SetInsertPoint(slow_bb);
    auto load_guarded = [&](::llvm::Value *cond, ::llvm::Value *off,
                            const char *name) -> ::llvm::Value * {
        auto *cur = b.GetInsertBlock();
        auto *ld_bb = ::llvm::BasicBlock::Create(ctx, std::string{name} + ".ld", fn);
        auto *dn_bb = ::llvm::BasicBlock::Create(ctx, std::string{name} + ".dn", fn);
        b.CreateCondBr(cond, ld_bb, dn_bb);
        b.SetInsertPoint(ld_bb);
        auto *val = load_texel(off);
        b.CreateBr(dn_bb);
        b.SetInsertPoint(dn_bb);
        auto *phi = b.CreatePHI(f32x4_ty, 2u);
        phi->addIncoming(zero4, cur);
        phi->addIncoming(val, ld_bb);
        return phi;
    };
    auto *vu0 = b.CreateExtractElement(valid4, b.getInt32(0u));
    auto *vu1 = b.CreateExtractElement(valid4, b.getInt32(1u));
    auto *vv0 = b.CreateExtractElement(valid4, b.getInt32(2u));
    auto *vv1 = b.CreateExtractElement(valid4, b.getInt32(3u));
    auto *s00 = load_guarded(b.CreateAnd(vu0, vv0), off00, "t00");
    auto *s01 = load_guarded(b.CreateAnd(vu1, vv0), off01, "t01");
    auto *s10 = load_guarded(b.CreateAnd(vu0, vv1), off10, "t10");
    auto *s11 = load_guarded(b.CreateAnd(vu1, vv1), off11, "t11");
    auto *slow_end = b.GetInsertBlock();
    b.CreateBr(blend_bb);

    b.SetInsertPoint(blend_bb);
    auto make_phi = [&](::llvm::Value *fv, ::llvm::Value *sv) {
        auto *p = b.CreatePHI(f32x4_ty, 2u);
        p->addIncoming(fv, fast_bb);
        p->addIncoming(sv, slow_end);
        return p;
    };
    auto *c00 = make_phi(f00, s00);
    auto *c01 = make_phi(f01, s01);
    auto *c10 = make_phi(f10, s10);
    auto *c11 = make_phi(f11, s11);
    auto lerp4 = [&](::llvm::Value *lhs, ::llvm::Value *rhs, ::llvm::Value *t) {
        return b.CreateIntrinsic(
            ::llvm::Intrinsic::fma, {f32x4_ty}, {b.CreateFSub(rhs, lhs), t, lhs});
    };
    ::llvm::Value *res = lerp4(lerp4(c00, c01, xt_v4), lerp4(c10, c11, xt_v4), yt_v4);
    if (!is_float4) {
        res = b.CreateFMul(res, float_vector_constant(b, 4u, 1.0f / 255.0f));
    }
    b.CreateRet(res);
    return fn;
}

[[nodiscard]] ::llvm::Function *get_or_create_wide_rgba_linear_sample_helper(
    ::llvm::Module &module, uint32_t width, bool is_float4, bool little_endian) {
    auto fn_name = (is_float4 ?
                        std::string{"__luisa_simd_bindless_sample2d_float4_linear_w"} :
                        std::string{"__luisa_simd_bindless_sample2d_byte4_linear_w"}) +
                   std::to_string(width);
    if (auto *existing = module.getFunction(fn_name)) {
        return existing;
    }
    auto &ctx = module.getContext();
    auto *ptr_ty = ::llvm::PointerType::getUnqual(ctx);
    auto *i32_ty = ::llvm::Type::getInt32Ty(ctx);
    auto *f32_ty = ::llvm::Type::getFloatTy(ctx);
    auto *f32_lanes = ::llvm::FixedVectorType::get(f32_ty, width);
    auto *i32_lanes = ::llvm::FixedVectorType::get(i32_ty, width);
    auto *i64_lanes = ::llvm::FixedVectorType::get(
        ::llvm::Type::getInt64Ty(ctx), width);
    auto *mask_ty = ::llvm::FixedVectorType::get(
        ::llvm::Type::getInt1Ty(ctx), width);
    auto *scratch_ty = ::llvm::ArrayType::get(f32_lanes, 4u);

    auto *fn_ty = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(ctx),
        {ptr_ty, i32_ty, i32_ty, i32_ty, f32_lanes, f32_lanes, mask_ty, ptr_ty},
        false);
    auto *fn = ::llvm::Function::Create(
        fn_ty, ::llvm::GlobalValue::InternalLinkage, fn_name, module);
    fn->setDSOLocal(true);
    fn->setCallingConv(::llvm::CallingConv::Fast);
    fn->addFnAttr(::llvm::Attribute::NoInline);
    fn->addFnAttr(::llvm::Attribute::NoUnwind);
    fn->addFnAttr(::llvm::Attribute::WillReturn);
    fn->addParamAttr(0u, ::llvm::Attribute::ReadOnly);
    fn->addParamAttr(0u, ::llvm::Attribute::NonNull);
    fn->addParamAttr(7u, ::llvm::Attribute::NoAlias);
    fn->addParamAttr(7u, ::llvm::Attribute::NonNull);

    auto *entry_bb = ::llvm::BasicBlock::Create(ctx, "entry", fn);
    ::llvm::IRBuilder<> b{entry_bb};
    auto *data = fn->getArg(0u);
    auto *width_val = fn->getArg(1u);
    auto *height_val = fn->getArg(2u);
    auto *addr_mode = fn->getArg(3u);
    auto *u_arg = fn->getArg(4u);
    auto *v_arg = fn->getArg(5u);
    auto *active_mask = fn->getArg(6u);
    auto *out_scratch = fn->getArg(7u);

    auto *zero = float_vector_constant(b, width, 0.0f);
    auto *half = float_vector_constant(b, width, 0.5f);
    auto *one = float_vector_constant(b, width, 1.0f);
    auto *two = float_vector_constant(b, width, 2.0f);
    auto *one_minus_eps = float_vector_constant(b, width, 0x1.fffffep-1f);
    auto *inf_vec = ::llvm::ConstantVector::getSplat(
        ::llvm::ElementCount::getFixed(width),
        ::llvm::ConstantFP::getInfinity(f32_ty));

    auto *w_f_s = b.CreateUIToFP(width_val, f32_ty);
    auto *h_f_s = b.CreateUIToFP(height_val, f32_ty);
    auto *w_f = b.CreateVectorSplat(width, w_f_s);
    auto *h_f = b.CreateVectorSplat(width, h_f_s);
    auto *u_fin = b.CreateFCmpOLT(
        b.CreateUnaryIntrinsic(::llvm::Intrinsic::fabs, u_arg), inf_vec);
    auto *v_fin = b.CreateFCmpOLT(
        b.CreateUnaryIntrinsic(::llvm::Intrinsic::fabs, v_arg), inf_vec);
    auto *hu = b.CreateVectorSplat(
        width, b.CreateFDiv(::llvm::ConstantFP::get(f32_ty, 0.5f), w_f_s));
    auto *hv = b.CreateVectorSplat(
        width, b.CreateFDiv(::llvm::ConstantFP::get(f32_ty, 0.5f), h_f_s));
    auto *u_lo_in = b.CreateFSub(u_arg, hu);
    auto *u_hi_in = b.CreateFAdd(u_arg, hu);
    auto *v_lo_in = b.CreateFSub(v_arg, hv);
    auto *v_hi_in = b.CreateFAdd(v_arg, hv);

    // Single address-mode switch covering both U and V axes.
    auto *repeat_bb = ::llvm::BasicBlock::Create(ctx, "uv.repeat", fn);
    auto *edge_bb = ::llvm::BasicBlock::Create(ctx, "uv.edge", fn);
    auto *mirror_bb = ::llvm::BasicBlock::Create(ctx, "uv.mirror", fn);
    auto *merge_bb = ::llvm::BasicBlock::Create(ctx, "uv.merge", fn);

    auto *sw = b.CreateSwitch(addr_mode, repeat_bb, 2u);
    sw->addCase(b.getInt32(0u), edge_bb);
    sw->addCase(b.getInt32(2u), mirror_bb);

    // REPEAT (1)
    b.SetInsertPoint(repeat_bb);
    auto repeat_one = [&](::llvm::Value *c, ::llvm::Value *sz) {
        return b.CreateFMul(
            b.CreateFSub(c, b.CreateUnaryIntrinsic(::llvm::Intrinsic::floor, c)),
            sz);
    };
    auto *rep_ulo = repeat_one(u_lo_in, w_f);
    auto *rep_uhi = repeat_one(u_hi_in, w_f);
    auto *rep_vlo = repeat_one(v_lo_in, h_f);
    auto *rep_vhi = repeat_one(v_hi_in, h_f);
    b.CreateBr(merge_bb);

    // EDGE (0)
    b.SetInsertPoint(edge_bb);
    auto edge_one = [&](::llvm::Value *c, ::llvm::Value *sz) {
        return b.CreateFMul(
            b.CreateMinNum(b.CreateMaxNum(c, zero), one_minus_eps), sz);
    };
    auto *edge_ulo = edge_one(u_lo_in, w_f);
    auto *edge_uhi = edge_one(u_hi_in, w_f);
    auto *edge_vlo = edge_one(v_lo_in, h_f);
    auto *edge_vhi = edge_one(v_hi_in, h_f);
    b.CreateBr(merge_bb);

    // MIRROR (2)
    b.SetInsertPoint(mirror_bb);
    auto mirror_one = [&](::llvm::Value *val, ::llvm::Value *sz) {
        val = b.CreateUnaryIntrinsic(::llvm::Intrinsic::fabs, val);
        auto *period = b.CreateUnaryIntrinsic(
            ::llvm::Intrinsic::floor, b.CreateFMul(val, half));
        val = b.CreateFSub(val, b.CreateFMul(period, two));
        val = b.CreateSelect(
            b.CreateFCmpOLT(val, one), val, b.CreateFSub(two, val));
        return b.CreateFMul(b.CreateMinNum(val, one_minus_eps), sz);
    };
    auto *mu_a = mirror_one(u_lo_in, w_f);
    auto *mu_b = mirror_one(u_hi_in, w_f);
    auto *mir_ulo = b.CreateMinNum(mu_a, mu_b);
    auto *mir_uhi = b.CreateMaxNum(mu_a, mu_b);
    auto *mv_a = mirror_one(v_lo_in, h_f);
    auto *mv_b = mirror_one(v_hi_in, h_f);
    auto *mir_vlo = b.CreateMinNum(mv_a, mv_b);
    auto *mir_vhi = b.CreateMaxNum(mv_a, mv_b);
    b.CreateBr(merge_bb);

    b.SetInsertPoint(merge_bb);
    auto make_phi = [&](::llvm::Value *rv, ::llvm::Value *ev, ::llvm::Value *mv) {
        auto *p = b.CreatePHI(f32_lanes, 3u);
        p->addIncoming(rv, repeat_bb);
        p->addIncoming(ev, edge_bb);
        p->addIncoming(mv, mirror_bb);
        return p;
    };
    struct AxisCoord {
        ::llvm::Value *lo_idx;
        ::llvm::Value *hi_idx;
        ::llvm::Value *t;
        ::llvm::Value *lo_valid;
        ::llvm::Value *hi_valid;
    };
    auto finish_axis = [&](::llvm::Value *lo_f, ::llvm::Value *hi_f,
                           ::llvm::Value *sz, ::llvm::Value *fin) -> AxisCoord {
        auto *lo_valid = b.CreateAnd(
            fin, b.CreateAnd(b.CreateFCmpOGE(lo_f, zero), b.CreateFCmpOLT(lo_f, sz)));
        auto *hi_valid = b.CreateAnd(
            fin, b.CreateAnd(b.CreateFCmpOGE(hi_f, zero), b.CreateFCmpOLT(hi_f, sz)));
        auto *safe_lo = b.CreateSelect(lo_valid, lo_f, zero);
        auto *safe_hi = b.CreateSelect(hi_valid, hi_f, zero);
        auto *lo_idx = b.CreateFPToUI(safe_lo, i32_lanes);
        auto *hi_idx = b.CreateFPToUI(safe_hi, i32_lanes);
        auto *frac = b.CreateFSub(safe_hi, b.CreateUIToFP(hi_idx, f32_lanes));
        return {lo_idx, hi_idx, frac, lo_valid, hi_valid};
    };
    auto *ulo_f = make_phi(rep_ulo, edge_ulo, mir_ulo);
    auto *uhi_f = make_phi(rep_uhi, edge_uhi, mir_uhi);
    auto *vlo_f = make_phi(rep_vlo, edge_vlo, mir_vlo);
    auto *vhi_f = make_phi(rep_vhi, edge_vhi, mir_vhi);
    auto x = finish_axis(ulo_f, uhi_f, w_f, u_fin);
    auto y = finish_axis(vlo_f, vhi_f, h_f, v_fin);

    auto *width_i64 = b.CreateVectorSplat(
        width, b.CreateZExt(width_val, b.getInt64Ty()));
    auto *data_lanes = b.CreateVectorSplat(width, data);
    auto *m00 = b.CreateAnd(active_mask, b.CreateAnd(x.lo_valid, y.lo_valid));
    auto *m01 = b.CreateAnd(active_mask, b.CreateAnd(x.hi_valid, y.lo_valid));
    auto *m10 = b.CreateAnd(active_mask, b.CreateAnd(x.lo_valid, y.hi_valid));
    auto *m11 = b.CreateAnd(active_mask, b.CreateAnd(x.hi_valid, y.hi_valid));
    auto *row0_i64 = b.CreateMul(b.CreateZExt(y.lo_idx, i64_lanes), width_i64);
    auto *row1_i64 = b.CreateMul(b.CreateZExt(y.hi_idx, i64_lanes), width_i64);
    auto *x_lo_i64 = b.CreateZExt(x.lo_idx, i64_lanes);
    auto *x_hi_i64 = b.CreateZExt(x.hi_idx, i64_lanes);
    auto *off00 = b.CreateAdd(row0_i64, x_lo_i64);
    auto *off01 = b.CreateAdd(row0_i64, x_hi_i64);
    auto *off10 = b.CreateAdd(row1_i64, x_lo_i64);
    auto *off11 = b.CreateAdd(row1_i64, x_hi_i64);

    auto lerp = [&](::llvm::Value *lhs, ::llvm::Value *rhs, ::llvm::Value *t) {
        return b.CreateIntrinsic(
            ::llvm::Intrinsic::fma, {f32_lanes}, {b.CreateFSub(rhs, lhs), t, lhs});
    };

    if (is_float4) {
        auto *shift2 = ::llvm::ConstantVector::getSplat(
            ::llvm::ElementCount::getFixed(width), b.getInt64(2u));
        auto *f00_base = b.CreateShl(off00, shift2);
        auto *f01_base = b.CreateShl(off01, shift2);
        auto *f10_base = b.CreateShl(off10, shift2);
        auto *f11_base = b.CreateShl(off11, shift2);
        auto gather_f32 = [&](::llvm::Value *base_off, uint32_t ch,
                              ::llvm::Value *mask) {
            auto *idx = ch == 0u ?
                            base_off :
                            b.CreateAdd(
                                base_off,
                                ::llvm::ConstantVector::getSplat(
                                    ::llvm::ElementCount::getFixed(width),
                                    b.getInt64(ch)));
            auto *ptrs = b.CreateGEP(f32_ty, data_lanes, idx);
            return b.CreateMaskedGather(
                f32_lanes, ptrs, ::llvm::Align{4u}, mask, zero);
        };
        for (uint32_t ch = 0u; ch < 4u; ++ch) {
            auto *c00 = gather_f32(f00_base, ch, m00);
            auto *c01 = gather_f32(f01_base, ch, m01);
            auto *c10 = gather_f32(f10_base, ch, m10);
            auto *c11 = gather_f32(f11_base, ch, m11);
            ::llvm::Value *val = lerp(lerp(c00, c01, x.t), lerp(c10, c11, x.t), y.t);
            val = b.CreateSelect(active_mask, val, zero);
            auto *slot_ptr = b.CreateGEP(
                scratch_ty, out_scratch, {b.getInt32(0u), b.getInt32(ch)});
            b.CreateStore(val, slot_ptr);
        }
        b.CreateRetVoid();
        return fn;
    }

    auto *zero_i32 = ::llvm::Constant::getNullValue(i32_lanes);
    auto gather_rgba = [&](::llvm::Value *off, ::llvm::Value *mask) {
        auto *ptrs = b.CreateGEP(i32_ty, data_lanes, off);
        return b.CreateMaskedGather(
            i32_lanes, ptrs, ::llvm::Align{4u}, mask, zero_i32);
    };
    auto *w00 = gather_rgba(off00, m00);
    auto *w01 = gather_rgba(off01, m01);
    auto *w10 = gather_rgba(off10, m10);
    auto *w11 = gather_rgba(off11, m11);

    auto *mask_ff = ::llvm::ConstantVector::getSplat(
        ::llvm::ElementCount::getFixed(width), b.getInt32(0xffu));
    auto *inv_255 = float_vector_constant(b, width, 1.0f / 255.0f);
    for (uint32_t ch = 0u; ch < 4u; ++ch) {
        uint32_t shift = little_endian ? (ch * 8u) : ((3u - ch) * 8u);
        auto extract_ch = [&](::llvm::Value *word) {
            auto *shifted = shift == 0u ?
                                word :
                                b.CreateLShr(
                                    word,
                                    ::llvm::ConstantVector::getSplat(
                                        ::llvm::ElementCount::getFixed(width),
                                        b.getInt32(shift)));
            auto *byte_i32 = shift == 24u ? shifted : b.CreateAnd(shifted, mask_ff);
            return b.CreateUIToFP(byte_i32, f32_lanes);
        };
        auto *c00 = extract_ch(w00);
        auto *c01 = extract_ch(w01);
        auto *c10 = extract_ch(w10);
        auto *c11 = extract_ch(w11);
        auto *val = b.CreateFMul(
            lerp(lerp(c00, c01, x.t), lerp(c10, c11, x.t), y.t), inv_255);
        val = b.CreateSelect(active_mask, val, zero);
        auto *slot_ptr = b.CreateGEP(
            scratch_ty, out_scratch, {b.getInt32(0u), b.getInt32(ch)});
        b.CreateStore(val, slot_ptr);
    }
    b.CreateRetVoid();
    return fn;
}

}// namespace

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_bindless_texture_read(
    const schedule::Instruction &instruction) {
    if (!instruction.result || !instruction.source_op) {
        _fail("bindless texture read instruction is malformed");
        return nullptr;
    }
    auto op = static_cast<xir::ResourceReadOp>(
        *instruction.source_op);
    auto dimension =
        op == xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ ||
                op == xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL ?
            2u :
        op == xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ ||
                op == xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL ?
            3u :
            0u;
    auto has_level =
        op == xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL ||
        op == xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL;
    if (dimension == 0u || instruction.operands.size() !=
                               (has_level ? 4u : 3u)) {
        _fail("unsupported bindless texture read operation");
        return nullptr;
    }
    if (_width > 64u) {
        _fail("bindless texture packet callbacks support widths up to 64 lanes");
        return nullptr;
    }
    auto *result = _source.value(*instruction.result);
    auto *coordinate_value = _source.value(instruction.operands[2u]);
    auto *coordinate = coordinate_value == nullptr ? nullptr :
                                                     _as_lane_vector(
                                                         _load_value(instruction.operands[2u]), *coordinate_value);
    if (result == nullptr || result->type == nullptr ||
        !result->type->is_vector() || result->type->dimension() != 4u ||
        !result->type->element()->is_float32() ||
        coordinate_value == nullptr || coordinate == nullptr ||
        coordinate_value->type == nullptr ||
        !coordinate_value->type->is_vector() ||
        coordinate_value->type->dimension() != dimension ||
        !coordinate_value->type->element()->is_uint32()) {
        _fail("bindless texture read requires uint coordinates and a float4 result");
        return nullptr;
    }
    auto array = _bindless_array_lanes(
        instruction.operands[0u], instruction.operands[1u]);
    if (array.view == nullptr) { return nullptr; }

    auto &context = _module.getContext();
    auto *pointer_type = ::llvm::PointerType::getUnqual(context);
    auto *null_pointer = ::llvm::ConstantPointerNull::get(pointer_type);
    auto *i32_lanes = ::llvm::FixedVectorType::get(
        _builder.getInt32Ty(), _width);
    auto *float_lanes = ::llvm::FixedVectorType::get(
        _builder.getFloatTy(), _width);
    auto *safe_slot_indices = _builder.CreateTrunc(
        array.slot_indices, i32_lanes);
    auto *slot_scratch = _entry_scratch(
        i32_lanes, "bindless.texture.read.slots");
    _builder.CreateStore(safe_slot_indices, slot_scratch);

    std::array<::llvm::AllocaInst *, 3u> coordinate_scratch{};
    auto *zero_i32 = ::llvm::Constant::getNullValue(i32_lanes);
    for (auto axis = uint32_t{0u}; axis < 3u; axis++) {
        auto *lanes = axis < dimension ?
                          _extract_child(
                              coordinate, coordinate_value->type, axis, true) :
                          zero_i32;
        lanes = _builder.CreateSelect(
            _active_mask, lanes, zero_i32,
            "bindless.texture.read.safe.coordinate");
        coordinate_scratch[axis] = _entry_scratch(
            i32_lanes,
            "bindless.texture.read.coordinate." + std::to_string(axis));
        _builder.CreateStore(lanes, coordinate_scratch[axis]);
    }

    ::llvm::Value *level_pointer = null_pointer;
    if (has_level) {
        auto *level_value = _source.value(instruction.operands[3u]);
        auto *level = level_value == nullptr ? nullptr :
                                               _as_lane_vector(
                                                   _load_value(instruction.operands[3u]), *level_value);
        if (level_value == nullptr || level == nullptr ||
            level_value->type == nullptr ||
            !level_value->type->is_scalar() ||
            level_value->type->is_float()) {
            _fail("bindless texture read has an invalid mip level");
            return nullptr;
        }
        level = _builder.CreateZExtOrTrunc(level, i32_lanes);
        level = _builder.CreateSelect(_active_mask, level, zero_i32);
        auto *level_scratch = _entry_scratch(
            i32_lanes, "bindless.texture.read.levels");
        _builder.CreateStore(level, level_scratch);
        level_pointer = level_scratch;
    }

    auto *scratch_type = ::llvm::ArrayType::get(float_lanes, 4u);
    auto *scratch = _entry_scratch(
        scratch_type,
        "bindless.texture.read.result." +
            std::to_string(instruction.result->value));
    _builder.CreateStore(
        ::llvm::Constant::getNullValue(scratch_type), scratch);
    auto *callback = _builder.CreateExtractValue(array.view, {3u});
    auto *missing_callback = _builder.CreateICmpEQ(
        callback, null_pointer);
    _trap_if(
        _builder.CreateAnd(
            _builder.CreateOrReduce(_active_mask), missing_callback),
        "bindless.texture.read.callback.null");
    auto *active_mask_bits = _bindless_callback_mask(
        result->value_class == schedule::ValueClass::varying);
    auto *callback_type = ::llvm::FunctionType::get(
        _builder.getVoidTy(),
        {pointer_type, _builder.getInt64Ty(),
         _builder.getInt32Ty(), _builder.getInt32Ty(),
         _builder.getInt64Ty(), pointer_type,
         pointer_type, pointer_type, pointer_type,
         pointer_type, pointer_type},
        false);
    _builder.CreateCall(
        callback_type, callback,
        {array.slots, array.slot_count,
         _builder.getInt32(dimension), _builder.getInt32(_width),
         active_mask_bits, slot_scratch,
         coordinate_scratch[0u], coordinate_scratch[1u],
         coordinate_scratch[2u], level_pointer, scratch});

    auto *pixels = static_cast<::llvm::Value *>(
        ::llvm::PoisonValue::get(_data_type(result->type, true)));
    for (auto component = uint32_t{0u}; component < 4u; component++) {
        auto *component_pointer = _builder.CreateGEP(
            scratch_type, scratch,
            {_builder.getInt32(0u), _builder.getInt32(component)});
        pixels = _insert_child(
            pixels, _builder.CreateLoad(float_lanes, component_pointer),
            result->type, component, true);
    }
    return result->value_class == schedule::ValueClass::varying ?
               pixels :
               _extract_lane(pixels, result->type,
                             _safe_first_lane(_active_mask));
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_bindless_texture_query(
    const schedule::Instruction &instruction) {
    if (!instruction.result || !instruction.source_op ||
        instruction.operands.size() < 2u) {
        _fail("bindless texture query instruction is malformed");
        return nullptr;
    }
    auto op = static_cast<xir::ResourceQueryOp>(
        *instruction.source_op);
    auto size_query =
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL;
    auto sample_query = !size_query &&
                        op >= xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE &&
                        op <= xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER;
    if (!size_query && !sample_query) {
        _fail("unsupported bindless texture query operation");
        return nullptr;
    }
    auto dimension =
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE ||
                op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL ||
                op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE ||
                op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL ||
                op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD ||
                op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL ||
                op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER ||
                op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER ||
                op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER ||
                op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER ?
            2u :
            3u;
    auto has_level =
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER;
    auto has_gradient =
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER;
    auto has_gradient_level =
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER;
    auto explicit_sampler =
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER;
    if (_width > 64u) {
        _fail("bindless texture packet callbacks support widths up to 64 lanes");
        return nullptr;
    }

    auto array = _bindless_array_lanes(
        instruction.operands[0u], instruction.operands[1u]);
    if (array.view == nullptr) { return nullptr; }
    auto &context = _module.getContext();
    auto *pointer_type = ::llvm::PointerType::getUnqual(context);
    auto *null_pointer = ::llvm::ConstantPointerNull::get(pointer_type);
    auto *i32_lanes = ::llvm::FixedVectorType::get(
        _builder.getInt32Ty(), _width);
    auto *safe_slot_indices = _builder.CreateTrunc(
        array.slot_indices, i32_lanes);
    auto *slot_scratch = _entry_scratch(
        i32_lanes, "bindless.texture.query.slots");
    _builder.CreateStore(safe_slot_indices, slot_scratch);
    auto *zero_i32 = ::llvm::Constant::getNullValue(i32_lanes);

    if (size_query) {
        auto expected_operands = has_level ? 3u : 2u;
        auto *result = _source.value(*instruction.result);
        if (instruction.operands.size() != expected_operands ||
            result == nullptr || result->type == nullptr ||
            !result->type->is_vector() ||
            result->type->dimension() != dimension ||
            !result->type->element()->is_uint32()) {
            _fail("bindless texture size query has invalid operands or result type");
            return nullptr;
        }
        ::llvm::Value *level_pointer = null_pointer;
        if (has_level) {
            auto *level_value = _source.value(instruction.operands[2u]);
            auto *level = level_value == nullptr ? nullptr :
                                                   _as_lane_vector(
                                                       _load_value(instruction.operands[2u]), *level_value);
            if (level_value == nullptr || level == nullptr ||
                level_value->type == nullptr ||
                !level_value->type->is_scalar() ||
                level_value->type->is_float()) {
                _fail("bindless texture size query has an invalid mip level");
                return nullptr;
            }
            level = _builder.CreateZExtOrTrunc(level, i32_lanes);
            level = _builder.CreateSelect(_active_mask, level, zero_i32);
            auto *level_scratch = _entry_scratch(
                i32_lanes, "bindless.texture.size.levels");
            _builder.CreateStore(level, level_scratch);
            level_pointer = level_scratch;
        }
        auto *scratch_type = ::llvm::ArrayType::get(i32_lanes, 3u);
        auto *scratch = _entry_scratch(
            scratch_type,
            "bindless.texture.size.result." +
                std::to_string(instruction.result->value));
        _builder.CreateStore(
            ::llvm::Constant::getNullValue(scratch_type), scratch);
        auto *callback = _builder.CreateExtractValue(array.view, {4u});
        auto *missing_callback = _builder.CreateICmpEQ(
            callback, null_pointer);
        _trap_if(
            _builder.CreateAnd(
                _builder.CreateOrReduce(_active_mask), missing_callback),
            "bindless.texture.size.callback.null");
        auto *callback_type = ::llvm::FunctionType::get(
            _builder.getVoidTy(),
            {pointer_type, _builder.getInt64Ty(),
             _builder.getInt32Ty(), _builder.getInt32Ty(),
             _builder.getInt64Ty(), pointer_type,
             pointer_type, pointer_type},
            false);
        auto *active_mask_bits = _bindless_callback_mask(
            result->value_class == schedule::ValueClass::varying);
        _builder.CreateCall(
            callback_type, callback,
            {array.slots, array.slot_count,
             _builder.getInt32(dimension), _builder.getInt32(_width),
             active_mask_bits, slot_scratch, level_pointer, scratch});
        auto *sizes = static_cast<::llvm::Value *>(
            ::llvm::PoisonValue::get(_data_type(result->type, true)));
        for (auto axis = uint32_t{0u}; axis < dimension; axis++) {
            auto *axis_pointer = _builder.CreateGEP(
                scratch_type, scratch,
                {_builder.getInt32(0u), _builder.getInt32(axis)});
            sizes = _insert_child(
                sizes, _builder.CreateLoad(i32_lanes, axis_pointer),
                result->type, axis, true);
        }
        return result->value_class == schedule::ValueClass::varying ?
                   sizes :
                   _extract_lane(sizes, result->type,
                                 _safe_first_lane(_active_mask));
    }

    auto expected_operands =
        3u + (has_gradient ? 2u : 0u) +
        ((has_level || has_gradient_level) ? 1u : 0u) +
        (explicit_sampler ? 2u : 0u);
    auto *result = _source.value(*instruction.result);
    auto *coordinate_value = _source.value(instruction.operands[2u]);
    auto *coordinate = coordinate_value == nullptr ? nullptr :
                                                     _as_lane_vector(
                                                         _load_value(instruction.operands[2u]), *coordinate_value);
    if (instruction.operands.size() != expected_operands ||
        result == nullptr || result->type == nullptr ||
        !result->type->is_vector() || result->type->dimension() != 4u ||
        !result->type->element()->is_float32() ||
        coordinate_value == nullptr || coordinate == nullptr ||
        coordinate_value->type == nullptr ||
        !coordinate_value->type->is_vector() ||
        coordinate_value->type->dimension() != dimension ||
        !coordinate_value->type->element()->is_float32()) {
        _fail("bindless texture sample has invalid operands or result type");
        return nullptr;
    }

    auto varying_result = result->value_class ==
                          schedule::ValueClass::varying;
    auto *float_lanes = ::llvm::FixedVectorType::get(
        _builder.getFloatTy(), _width);
    auto *zero_float = ::llvm::Constant::getNullValue(float_lanes);
    auto *one_float = ::llvm::ConstantVector::getSplat(
        ::llvm::ElementCount::getFixed(_width),
        ::llvm::ConstantFP::get(_builder.getFloatTy(), 1.0));
    std::array<::llvm::AllocaInst *, 3u> coordinate_scratch{};
    for (auto axis = uint32_t{0u}; axis < 3u; axis++) {
        auto *lanes = axis < dimension ?
                          _extract_child(
                              coordinate, coordinate_value->type, axis, true) :
                          zero_float;
        lanes = _builder.CreateSelect(
            _active_mask, lanes, zero_float,
            "bindless.texture.sample.safe.coordinate");
        coordinate_scratch[axis] = _entry_scratch(
            float_lanes,
            "bindless.texture.sample.coordinate." +
                std::to_string(axis));
        _builder.CreateStore(lanes, coordinate_scratch[axis]);
    }

    auto operand_index = size_t{3u};
    ::llvm::Value *gradient_level = nullptr;
    ::llvm::Value *uniform_gradient_level = nullptr;
    ::llvm::Value *gradient_mask = nullptr;
    auto gradient_level_is_uniform = false;
    if (has_gradient) {
        auto *slot_value = _source.value(instruction.operands[1u]);
        std::array<const schedule::Value *, 2u> gradient_values{};
        auto uniform_gradient_lod =
            slot_value != nullptr &&
            schedule::is_uniform(slot_value->value_class);
        for (auto derivative = 0u; derivative < 2u; derivative++) {
            auto *gradient_value = _source.value(
                instruction.operands[operand_index + derivative]);
            gradient_values[derivative] = gradient_value;
            if (gradient_value == nullptr ||
                gradient_value->type == nullptr ||
                !gradient_value->type->is_vector() ||
                gradient_value->type->dimension() != dimension ||
                !gradient_value->type->element()->is_float32()) {
                _fail("bindless texture sample has an invalid gradient");
                return nullptr;
            }
            uniform_gradient_lod &=
                schedule::is_uniform(gradient_value->value_class);
        }
        constexpr auto texture_offset =
            offsetof(SIMDHostBindlessSlot, texture2d);
        constexpr auto texture_stride =
            offsetof(SIMDHostBindlessSlot, texture3d) -
            offsetof(SIMDHostBindlessSlot, texture2d);
        auto descriptor_offset = texture_offset +
                                 (dimension - 2u) * texture_stride;
        auto *first_active_mask = _builder.CreateAnd(
            _active_mask,
            _builder.CreateICmpEQ(
                _lane_ids(), _builder.CreateVectorSplat(
                                 _width, _safe_first_lane(_active_mask))));
        gradient_mask = varying_result ? _active_mask : first_active_mask;
        auto slot_uniform = schedule::is_uniform(slot_value->value_class);
        std::array<::llvm::Value *, 3u> extents{};
        auto *i64_lanes = ::llvm::FixedVectorType::get(
            _builder.getInt64Ty(), _width);
        auto metadata_offset = descriptor_offset +
                               offsetof(SIMDHostBindlessTextureSlot, metadata);
        ::llvm::Value *metadata = nullptr;
        if (slot_uniform) {
            auto *scalar_slot = _builder.CreateExtractElement(
                array.slot_indices, _safe_first_lane(_active_mask));
            auto *metadata_pointer = _builder.CreateGEP(
                _builder.getInt8Ty(), array.slots,
                _builder.CreateAdd(
                    _builder.CreateMul(
                        scalar_slot,
                        _builder.getInt64(
                            sizeof(SIMDHostBindlessSlot))),
                    _builder.getInt64(metadata_offset)));
            auto *zero_metadata = _entry_scratch(
                _builder.getInt64Ty(),
                "bindless.texture.sample.uniform.metadata.zero");
            _builder.CreateStore(
                _builder.getInt64(0u), zero_metadata);
            auto *safe_metadata_pointer = _builder.CreateSelect(
                _builder.CreateOrReduce(_active_mask),
                metadata_pointer, zero_metadata,
                "bindless.texture.sample.uniform.metadata.pointer");
            auto *metadata_load = _builder.CreateLoad(
                _builder.getInt64Ty(), safe_metadata_pointer,
                "bindless.texture.sample.uniform.metadata");
            metadata_load->setAlignment(
                ::llvm::Align{alignof(uint64_t)});
            metadata = metadata_load;
        } else {
            auto *slot_offsets = _builder.CreateMul(
                array.slot_indices,
                _builder.CreateVectorSplat(
                    _width,
                    _builder.getInt64(sizeof(SIMDHostBindlessSlot))));
            auto *slot_bases = _builder.CreateGEP(
                _builder.getInt8Ty(),
                _builder.CreateVectorSplat(_width, array.slots),
                slot_offsets);
            auto *extent_pointers = _builder.CreateGEP(
                _builder.getInt8Ty(), slot_bases,
                _builder.CreateVectorSplat(
                    _width, _builder.getInt64(metadata_offset)));
            metadata = _builder.CreateMaskedGather(
                i64_lanes, extent_pointers,
                ::llvm::Align{alignof(uint64_t)}, gradient_mask,
                ::llvm::Constant::getNullValue(i64_lanes));
        }
        for (auto axis = 0u; axis < dimension; axis++) {
            auto shift = 4u +
                         axis * simd_bindless_texture_extent_bits;
            if (slot_uniform) {
                extents[axis] = _builder.CreateUIToFP(
                    _builder.CreateTrunc(
                        _builder.CreateAnd(
                            _builder.CreateLShr(
                                metadata, _builder.getInt64(shift)),
                            _builder.getInt64(
                                simd_bindless_texture_extent_mask)),
                        _builder.getInt32Ty()),
                    _builder.getFloatTy());
                if (!uniform_gradient_lod) {
                    extents[axis] = _builder.CreateVectorSplat(
                        _width, extents[axis]);
                }
            } else {
                extents[axis] = _builder.CreateUIToFP(
                    _builder.CreateTrunc(
                        _builder.CreateAnd(
                            _builder.CreateLShr(
                                metadata,
                                _builder.CreateVectorSplat(
                                    _width, _builder.getInt64(shift))),
                            _builder.CreateVectorSplat(
                                _width, _builder.getInt64(
                                            simd_bindless_texture_extent_mask))),
                        i32_lanes),
                    float_lanes);
            }
        }
        auto native_math_mode = _enable_fast_math ?
                                    cpu::LLVMNativeMathMode::fast :
                                    cpu::LLVMNativeMathMode::precise;
        if (uniform_gradient_lod) {
            gradient_level_is_uniform = true;
            auto *zero_scalar = ::llvm::ConstantFP::get(
                _builder.getFloatTy(), 0.0);
            auto *one_scalar = ::llvm::ConstantFP::get(
                _builder.getFloatTy(), 1.0);
            std::array<::llvm::Value *, 2u> gradient_norms{
                zero_scalar, zero_scalar};
            ::llvm::Value *gradient_has_nan =
                _builder.getFalse();
            for (auto derivative = 0u; derivative < 2u; derivative++) {
                auto *gradient = _load_value(
                    instruction.operands[operand_index + derivative]);
                if (gradient == nullptr) { return nullptr; }
                for (auto axis = 0u; axis < dimension; axis++) {
                    auto *component = _extract_child(
                        gradient, gradient_values[derivative]->type,
                        axis, false);
                    gradient_has_nan = _builder.CreateOr(
                        gradient_has_nan,
                        _builder.CreateFCmpUNO(component, component));
                    auto *scaled = _builder.CreateFMul(
                        component, extents[axis]);
                    gradient_norms[derivative] = _builder.CreateFAdd(
                        gradient_norms[derivative],
                        _builder.CreateFMul(scaled, scaled));
                }
            }
            auto *rho2 = _builder.CreateMaxNum(
                gradient_norms[0u], gradient_norms[1u]);
            rho2 = _builder.CreateMaxNum(rho2, one_scalar);
            rho2 = _builder.CreateSelect(
                gradient_has_nan, one_scalar, rho2,
                "bindless.texture.sample.uniform.nan.gradient.lod");
            uniform_gradient_level = _builder.CreateUnaryIntrinsic(
                ::llvm::Intrinsic::log2, rho2,
                nullptr, "bindless.texture.sample.uniform.gradient.log2");
            uniform_gradient_level = _builder.CreateFMul(
                uniform_gradient_level,
                ::llvm::ConstantFP::get(
                    _builder.getFloatTy(), 0.5));
        } else {
            std::array<::llvm::Value *, 2u> gradient_norms{
                zero_float, zero_float};
            ::llvm::Value *gradient_has_nan =
                ::llvm::Constant::getNullValue(
                    ::llvm::FixedVectorType::get(
                        _builder.getInt1Ty(), _width));
            for (auto derivative = 0u; derivative < 2u; derivative++) {
                auto *gradient = _as_lane_vector(
                    _load_value(instruction.operands[operand_index + derivative]),
                    *gradient_values[derivative]);
                if (gradient == nullptr) { return nullptr; }
                for (auto axis = 0u; axis < dimension; axis++) {
                    auto *lanes = _extract_child(
                        gradient, gradient_values[derivative]->type,
                        axis, true);
                    lanes = _builder.CreateSelect(
                        gradient_mask, lanes, zero_float,
                        "bindless.texture.sample.safe.gradient");
                    gradient_has_nan = _builder.CreateOr(
                        gradient_has_nan,
                        _builder.CreateFCmpUNO(lanes, lanes));
                    auto *scaled = _builder.CreateFMul(
                        lanes, extents[axis]);
                    gradient_norms[derivative] = _builder.CreateFAdd(
                        gradient_norms[derivative],
                        _builder.CreateFMul(scaled, scaled));
                }
            }
            auto *rho2 = _builder.CreateMaxNum(
                gradient_norms[0u], gradient_norms[1u]);
            rho2 = _builder.CreateMaxNum(rho2, one_float);
            rho2 = _builder.CreateSelect(
                gradient_has_nan, one_float, rho2,
                "bindless.texture.sample.nan.gradient.lod");
            gradient_level = cpu::LLVMNativeMath::emit_log2_f32(
                _module, _builder, rho2, native_math_mode);
            if (gradient_level == nullptr) {
                _fail("bindless texture gradient LOD requires fixed f32 vectors");
                return nullptr;
            }
            gradient_level = _builder.CreateFMul(
                gradient_level,
                ::llvm::ConstantVector::getSplat(
                    ::llvm::ElementCount::getFixed(_width),
                    ::llvm::ConstantFP::get(
                        _builder.getFloatTy(), 0.5)));
        }
        if (!gradient_level_is_uniform) {
            gradient_level = _builder.CreateSelect(
                gradient_mask, gradient_level, zero_float,
                "bindless.texture.sample.safe.gradient.level");
        }
        operand_index += 2u;
    }

    ::llvm::Value *level_pointer = null_pointer;
    if (has_level || has_gradient_level) {
        auto *level_value = _source.value(
            instruction.operands[operand_index]);
        auto *raw_level = level_value == nullptr ? nullptr :
                                                   _load_value(
                                                       instruction.operands[operand_index]);
        if (level_value == nullptr || raw_level == nullptr ||
            level_value->type == nullptr ||
            !level_value->type->is_float32()) {
            _fail(
                has_gradient_level ?
                    "bindless texture sample has an invalid minimum mip level" :
                    "bindless texture sample has an invalid mip level");
            return nullptr;
        }
        ::llvm::Value *level = nullptr;
        auto uniform_minimum_level =
            has_gradient_level && gradient_level_is_uniform &&
            schedule::is_uniform(level_value->value_class);
        if (uniform_minimum_level) {
            auto *scalar_level = _builder.CreateMaxNum(
                uniform_gradient_level, raw_level);
            level = _builder.CreateVectorSplat(
                _width, scalar_level,
                "bindless.texture.sample.uniform.gradient.minimum.splat");
            level = _builder.CreateSelect(
                gradient_mask, level, zero_float);
        } else {
            level = _as_lane_vector(raw_level, *level_value);
            level = _builder.CreateSelect(
                has_gradient_level ? gradient_mask : _active_mask,
                level, zero_float);
            if (has_gradient_level) {
                if (gradient_level_is_uniform) {
                    gradient_level = _builder.CreateVectorSplat(
                        _width, uniform_gradient_level,
                        "bindless.texture.sample.uniform.gradient.splat");
                }
                level = _builder.CreateMaxNum(
                    gradient_level, level);
                level = _builder.CreateSelect(
                    gradient_mask, level, zero_float);
            }
        }
        auto *level_scratch = _entry_scratch(
            float_lanes, "bindless.texture.sample.levels");
        _builder.CreateStore(level, level_scratch);
        level_pointer = level_scratch;
        operand_index++;
    } else if (has_gradient) {
        if (gradient_level_is_uniform) {
            gradient_level = _builder.CreateVectorSplat(
                _width, uniform_gradient_level,
                "bindless.texture.sample.uniform.gradient.splat");
            gradient_level = _builder.CreateSelect(
                gradient_mask, gradient_level, zero_float,
                "bindless.texture.sample.safe.gradient.level");
        }
        auto *level_scratch = _entry_scratch(
            float_lanes, "bindless.texture.sample.gradient.levels");
        _builder.CreateStore(gradient_level, level_scratch);
        level_pointer = level_scratch;
    }

    ::llvm::Value *sampler_pointer = null_pointer;
    if (explicit_sampler) {
        auto *filter_value = _source.value(
            instruction.operands[operand_index]);
        auto *address_value = _source.value(
            instruction.operands[operand_index + 1u]);
        auto *filter = filter_value == nullptr ? nullptr :
                                                 _as_lane_vector(
                                                     _load_value(instruction.operands[operand_index]),
                                                     *filter_value);
        auto *address = address_value == nullptr ? nullptr :
                                                   _as_lane_vector(
                                                       _load_value(instruction.operands[operand_index + 1u]),
                                                       *address_value);
        if (filter_value == nullptr || address_value == nullptr ||
            filter == nullptr || address == nullptr ||
            filter_value->type == nullptr ||
            address_value->type == nullptr ||
            !filter_value->type->is_scalar() ||
            !address_value->type->is_scalar() ||
            filter_value->type->is_float() ||
            address_value->type->is_float()) {
            _fail("bindless texture sample has an invalid explicit sampler");
            return nullptr;
        }
        filter = _builder.CreateZExtOrTrunc(filter, i32_lanes);
        address = _builder.CreateZExtOrTrunc(address, i32_lanes);
        auto *sampler = _builder.CreateOr(
            _builder.CreateShl(
                filter,
                _builder.CreateVectorSplat(
                    _width, _builder.getInt32(2u))),
            address);
        sampler = _builder.CreateSelect(
            _active_mask, sampler, zero_i32);
        auto *sampler_scratch = _entry_scratch(
            i32_lanes, "bindless.texture.sample.samplers");
        _builder.CreateStore(sampler, sampler_scratch);
        sampler_pointer = sampler_scratch;
    }

    auto *scratch_type = ::llvm::ArrayType::get(float_lanes, 4u);
    auto *scratch = _entry_scratch(
        scratch_type,
        "bindless.texture.sample.result." +
            std::to_string(instruction.result->value));
    _builder.CreateStore(
        ::llvm::Constant::getNullValue(scratch_type), scratch);
    auto *callback = _builder.CreateExtractValue(array.view, {2u});
    auto *callback_type = ::llvm::FunctionType::get(
        _builder.getVoidTy(),
        {pointer_type, _builder.getInt64Ty(),
         _builder.getInt32Ty(), _builder.getInt32Ty(),
         _builder.getInt64Ty(), pointer_type, pointer_type,
         pointer_type, pointer_type, pointer_type,
         pointer_type, pointer_type},
        false);
    auto emit_callback = [&] {
        auto *missing_callback = _builder.CreateICmpEQ(
            callback, null_pointer);
        _trap_if(
            _builder.CreateAnd(
                _builder.CreateOrReduce(_active_mask),
                missing_callback),
            "bindless.texture.sample.callback.null");
        auto *active_mask_bits = _bindless_callback_mask(
            result->value_class == schedule::ValueClass::varying);
        _builder.CreateCall(
            callback_type, callback,
            {array.slots, array.slot_count,
             _builder.getInt32(dimension), _builder.getInt32(_width),
             active_mask_bits, slot_scratch, sampler_pointer,
             coordinate_scratch[0u], coordinate_scratch[1u],
             coordinate_scratch[2u], level_pointer, scratch});
    };

    auto *slot_value = _source.value(instruction.operands[1u]);
    auto direct_mip0_candidate =
        dimension == 2u && !has_level && !has_gradient &&
        !explicit_sampler && varying_result && slot_value != nullptr &&
        (_width == 1u || schedule::is_uniform(slot_value->value_class)) &&
        !luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_IR_BYTE1_TEXTURE_SAMPLING");
    if (direct_mip0_candidate) {
        auto *any_active = _width == 1u ?
                               _builder.CreateExtractElement(_active_mask, _builder.getInt32(0u)) :
                               _builder.CreateOrReduce(_active_mask);
        auto *inspect_bb = ::llvm::BasicBlock::Create(
            _module.getContext(), "bindless.texture.mip0.inspect", _entry);
        auto *direct_byte4 = ::llvm::BasicBlock::Create(
            _module.getContext(), "bindless.texture.byte4.direct", _entry);
        auto *check_float4 = ::llvm::BasicBlock::Create(
            _module.getContext(), "bindless.texture.float4.check", _entry);
        auto *direct_float4 = ::llvm::BasicBlock::Create(
            _module.getContext(), "bindless.texture.float4.direct", _entry);
        auto *check_byte1 = ::llvm::BasicBlock::Create(
            _module.getContext(), "bindless.texture.byte1.check", _entry);
        auto *direct_byte1 = ::llvm::BasicBlock::Create(
            _module.getContext(), "bindless.texture.byte1.direct", _entry);
        auto *fallback = ::llvm::BasicBlock::Create(
            _module.getContext(), "bindless.texture.mip0.callback", _entry);
        auto *merge = ::llvm::BasicBlock::Create(
            _module.getContext(), "bindless.texture.mip0.merge", _entry);

        auto *active_br = _builder.CreateCondBr(any_active, inspect_bb, merge);
        active_br->setMetadata(
            ::llvm::LLVMContext::MD_prof,
            ::llvm::MDBuilder{_module.getContext()}.createBranchWeights(1000u, 1u));

        _builder.SetInsertPoint(inspect_bb);
        auto *first_active_lane = _safe_first_lane(_active_mask);
        auto *scalar_slot = _builder.CreateExtractElement(
            array.slot_indices, first_active_lane);
        auto *slot_base = _builder.CreateGEP(
            _builder.getInt8Ty(), array.slots,
            _builder.CreateMul(
                scalar_slot,
                _builder.getInt64(sizeof(SIMDHostBindlessSlot))));
        auto *descriptor = _builder.CreateGEP(
            _builder.getInt8Ty(), slot_base,
            _builder.getInt64(
                offsetof(SIMDHostBindlessSlot, texture2d)));
        auto *data_pointer = _builder.CreateGEP(
            _builder.getInt8Ty(), descriptor,
            _builder.getInt64(offsetof(
                SIMDHostBindlessTextureSlot, byte1_mip0)));
        auto *data = _builder.CreateLoad(
            pointer_type, data_pointer,
            "bindless.texture.mip0.data");
        auto *metadata_pointer = _builder.CreateGEP(
            _builder.getInt8Ty(), descriptor,
            _builder.getInt64(offsetof(
                SIMDHostBindlessTextureSlot, metadata)));
        auto *metadata = _builder.CreateLoad(
            _builder.getInt64Ty(), metadata_pointer,
            "bindless.texture.mip0.metadata");
        auto extent = [&](uint32_t shift,
                          luisa::string_view name) {
            return _builder.CreateTrunc(
                _builder.CreateAnd(
                    _builder.CreateLShr(
                        metadata, _builder.getInt64(shift)),
                    _builder.getInt64(
                        simd_bindless_texture_extent_mask)),
                _builder.getInt32Ty(), name);
        };
        auto *width = extent(
            4u, "bindless.texture.mip0.width");
        auto *height = extent(
            24u, "bindless.texture.mip0.height");
        auto *sampler_code = _builder.CreateTrunc(
            _builder.CreateAnd(metadata, _builder.getInt64(0x0fu)),
            _builder.getInt32Ty());
        auto *mip0_format = _builder.CreateTrunc(
            _builder.CreateLShr(
                metadata,
                _builder.getInt64(simd_bindless_texture_mip0_format_shift)),
            _builder.getInt32Ty());
        auto *has_valid_mip0 = _builder.CreateAnd(
            _builder.CreateICmpNE(data, null_pointer),
            _builder.CreateAnd(
                _builder.CreateICmpNE(width, _builder.getInt32(0u)),
                _builder.CreateICmpNE(height, _builder.getInt32(0u))));
        auto *address_mode = _builder.CreateAnd(
            sampler_code, _builder.getInt32(0x03u));
        auto *linear_non_zero = _builder.CreateAnd(
            has_valid_mip0,
            _builder.CreateAnd(
                _builder.CreateICmpUGE(sampler_code, _builder.getInt32(4u)),
                _builder.CreateICmpNE(address_mode, _builder.getInt32(3u))));
        auto *byte4_eligible = _builder.CreateAnd(
            linear_non_zero,
            _builder.CreateICmpEQ(
                mip0_format,
                _builder.getInt32(static_cast<uint32_t>(
                    simd_bindless_texture_mip0_format_byte4))),
            "bindless.texture.byte4.eligible");
        auto *float4_eligible = _builder.CreateAnd(
            linear_non_zero,
            _builder.CreateICmpEQ(
                mip0_format,
                _builder.getInt32(static_cast<uint32_t>(
                    simd_bindless_texture_mip0_format_float4))),
            "bindless.texture.float4.eligible");
        auto *byte1_eligible = _builder.CreateAnd(
            has_valid_mip0,
            _builder.CreateAnd(
                _builder.CreateICmpEQ(
                    mip0_format,
                    _builder.getInt32(static_cast<uint32_t>(
                        simd_bindless_texture_mip0_format_byte1))),
                _builder.CreateICmpEQ(
                    sampler_code,
                    _builder.getInt32(
                        simd_bindless_linear_point_mirror_sampler_code))),
            "bindless.texture.byte1.eligible");

        auto emit_direct_rgba = [&](bool is_float4) {
            auto *u_lanes = _builder.CreateLoad(float_lanes, coordinate_scratch[0u]);
            auto *v_lanes = _builder.CreateLoad(float_lanes, coordinate_scratch[1u]);
            if (_width == 1u) {
                auto *helper = get_or_create_w1_rgba_linear_sample_helper(
                    _module, is_float4);
                auto *u_scalar = _builder.CreateExtractElement(
                    u_lanes, _builder.getInt32(0u));
                auto *v_scalar = _builder.CreateExtractElement(
                    v_lanes, _builder.getInt32(0u));
                auto *call = _builder.CreateCall(
                    helper,
                    {data, metadata, u_scalar, v_scalar});
                call->setCallingConv(::llvm::CallingConv::Fast);
                for (uint32_t ch = 0u; ch < 4u; ++ch) {
                    auto *elem = _builder.CreateExtractElement(
                        call, _builder.getInt32(ch));
                    auto *v1 = _builder.CreateInsertElement(
                        ::llvm::PoisonValue::get(float_lanes), elem,
                        _builder.getInt32(0u));
                    auto *slot_ptr = _builder.CreateGEP(
                        scratch_type, scratch,
                        {_builder.getInt32(0u), _builder.getInt32(ch)});
                    _builder.CreateStore(v1, slot_ptr);
                }
            } else {
                auto *helper = get_or_create_wide_rgba_linear_sample_helper(
                    _module, _width, is_float4,
                    _module.getDataLayout().isLittleEndian());
                auto *call = _builder.CreateCall(
                    helper,
                    {data, width, height, address_mode,
                     u_lanes, v_lanes, _active_mask, scratch});
                call->setCallingConv(::llvm::CallingConv::Fast);
            }
        };

        _builder.CreateCondBr(byte4_eligible, direct_byte4, check_float4);

        _builder.SetInsertPoint(direct_byte4);
        emit_direct_rgba(false);
        _builder.CreateBr(merge);

        _builder.SetInsertPoint(check_float4);
        _builder.CreateCondBr(float4_eligible, direct_float4, check_byte1);

        _builder.SetInsertPoint(direct_float4);
        emit_direct_rgba(true);
        _builder.CreateBr(merge);

        _builder.SetInsertPoint(check_byte1);
        _builder.CreateCondBr(byte1_eligible, direct_byte1, fallback);

        _builder.SetInsertPoint(direct_byte1);
        emit_byte1_linear_mirror_sample(
            _builder, _width, _active_mask,
            data, width, height,
            _builder.CreateLoad(
                float_lanes, coordinate_scratch[0u]),
            _builder.CreateLoad(
                float_lanes, coordinate_scratch[1u]),
            scratch, scratch_type,
            (_width == 4u || _width == 8u) &&
                !luisa::compute::detail::env_flag(
                    "LUISA_SIMD_DISABLE_WIDE_BYTE1_GATHERS"),
            _module.getDataLayout().isLittleEndian());
        _builder.CreateBr(merge);

        _builder.SetInsertPoint(fallback);
        emit_callback();
        _builder.CreateBr(merge);
        _builder.SetInsertPoint(merge);
    } else {
        emit_callback();
    }
    auto *pixels = static_cast<::llvm::Value *>(
        ::llvm::PoisonValue::get(_data_type(result->type, true)));
    for (auto component = uint32_t{0u}; component < 4u; component++) {
        auto *component_pointer = _builder.CreateGEP(
            scratch_type, scratch,
            {_builder.getInt32(0u), _builder.getInt32(component)});
        pixels = _insert_child(
            pixels, _builder.CreateLoad(float_lanes, component_pointer),
            result->type, component, true);
    }
    return result->value_class == schedule::ValueClass::varying ?
               pixels :
               _extract_lane(pixels, result->type,
                             _safe_first_lane(_active_mask));
}

}// namespace luisa::compute::simd::detail
