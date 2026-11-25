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
    auto *infinity = ::llvm::ConstantVector::getSplat(
        ::llvm::ElementCount::getFixed(width),
        ::llvm::ConstantFP::getInfinity(builder.getFloatTy()));
    auto *i32_lanes = ::llvm::FixedVectorType::get(
        builder.getInt32Ty(), width);
    auto *sz_minus_1_f = builder.CreateFSub(size, one);

    auto *raw_a = builder.CreateUnaryIntrinsic(
        ::llvm::Intrinsic::fabs, coordinate);
    auto *finite = builder.CreateFCmpOLT(
        raw_a, infinity, "bindless.texture.byte1.finite");
    auto *positive_size = builder.CreateFCmpOGE(size, one);
    auto *valid = builder.CreateAnd(
        finite, positive_size, "bindless.texture.byte1.reduced.valid");
    auto *safe_raw_a = builder.CreateSelect(valid, raw_a, zero);
    auto *period = builder.CreateUnaryIntrinsic(
        ::llvm::Intrinsic::floor,
        builder.CreateFMul(safe_raw_a, half));
    auto *a_raw = builder.CreateFSub(
        safe_raw_a, builder.CreateFMul(period, two));
    auto *a_valid = builder.CreateAnd(
        builder.CreateFCmpOGE(a_raw, zero),
        builder.CreateFCmpOLT(a_raw, two));
    auto *a = builder.CreateSelect(a_valid, a_raw, zero);
    auto *folded = builder.CreateSelect(
        builder.CreateFCmpOLT(a, one), a,
        builder.CreateFSub(two, a));
    auto *px = builder.CreateFSub(
        builder.CreateFMul(folded, size), half);
    auto *px_clamped = builder.CreateMinNum(
        builder.CreateMaxNum(px, zero),
        builder.CreateMaxNum(sz_minus_1_f, zero));
    auto *safe_px = builder.CreateSelect(valid, px_clamped, zero);
    auto *fl = builder.CreateUnaryIntrinsic(
        ::llvm::Intrinsic::floor, safe_px);
    auto *fraction = builder.CreateFSub(
        safe_px, fl, "bindless.texture.byte1.linear.fraction");
    auto *i0 = builder.CreateFPToUI(fl, i32_lanes);
    auto *sz_i32 = builder.CreateFPToUI(size, i32_lanes);
    auto *max_idx = builder.CreateSub(
        sz_i32, ::llvm::ConstantVector::getSplat(
                    ::llvm::ElementCount::getFixed(width),
                    builder.getInt32(1u)));
    auto *lo_index = builder.CreateSelect(
        builder.CreateICmpULT(i0, sz_i32), i0, max_idx,
        "bindless.texture.byte1.linear.lo");
    auto *i1 = builder.CreateAdd(
        i0, ::llvm::ConstantVector::getSplat(
                ::llvm::ElementCount::getFixed(width), builder.getInt32(1u)));
    auto *hi_index = builder.CreateSelect(
        builder.CreateICmpULT(i1, sz_i32), i1, max_idx,
        "bindless.texture.byte1.linear.hi");
    return {lo_index, hi_index, fraction, valid};
}

}// namespace

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
    ::llvm::Module &module, bool is_float4, bool enable_fast_math) {
    auto fn_name = std::string{
        is_float4 ? "__luisa_simd_bindless_sample2d_float4_linear_w1" :
                    "__luisa_simd_bindless_sample2d_byte4_linear_w1"};
    if (enable_fast_math) { fn_name += "_fm"; }
    if (auto *existing = module.getFunction(fn_name)) {
        return existing;
    }
    auto &ctx = module.getContext();
    auto *ptr_ty = ::llvm::PointerType::getUnqual(ctx);
    auto *i8_ty = ::llvm::Type::getInt8Ty(ctx);
    auto *i16_ty = ::llvm::Type::getInt16Ty(ctx);
    auto *i32_ty = ::llvm::Type::getInt32Ty(ctx);
    auto *i64_ty = ::llvm::Type::getInt64Ty(ctx);
    auto *f32_ty = ::llvm::Type::getFloatTy(ctx);
    auto *f32x2_ty = ::llvm::FixedVectorType::get(f32_ty, 2u);
    auto *f32x4_ty = ::llvm::FixedVectorType::get(f32_ty, 4u);
    auto *i32x2_ty = ::llvm::FixedVectorType::get(i32_ty, 2u);
    auto *i32x4_ty = ::llvm::FixedVectorType::get(i32_ty, 4u);
    auto *i16x8_ty = ::llvm::FixedVectorType::get(i16_ty, 8u);
    auto *i8x8_ty = ::llvm::FixedVectorType::get(i8_ty, 8u);
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
    if (enable_fast_math) {
        ::llvm::FastMathFlags fm_flags;
        fm_flags.setAllowReciprocal(true);
        fm_flags.setAllowContract(true);
        fm_flags.setAllowReassoc(true);
        fm_flags.setNoSignedZeros(true);
        b.setFastMathFlags(fm_flags);
    }
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

    // Division-free interior fast path for REPEAT (1) and EDGE (0).
    auto *interior_check_bb = ::llvm::BasicBlock::Create(
        ctx, "interior.check", fn);
    auto *interior_idx_bb = ::llvm::BasicBlock::Create(
        ctx, "interior.idx", fn);
    auto *interior_load_bb = ::llvm::BasicBlock::Create(
        ctx, "interior.load", fn);
    auto *general_bb = ::llvm::BasicBlock::Create(ctx, "general", fn);

    auto *u_bits = b.CreateBitCast(u_arg, i32_ty);
    auto *v_bits = b.CreateBitCast(v_arg, i32_ty);
    auto *exp_mask = b.getInt32(0x7f800000u);
    auto *u_fin_fast = b.CreateICmpNE(b.CreateAnd(u_bits, exp_mask), exp_mask);
    auto *v_fin_fast = b.CreateICmpNE(b.CreateAnd(v_bits, exp_mask), exp_mask);
    auto *fast_eligible = b.CreateAnd(
        b.CreateAnd(
            b.CreateICmpULE(addr_mode, b.getInt32(1u)),
            b.CreateAnd(
                b.CreateICmpUGE(width_val, b.getInt32(2u)),
                b.CreateICmpUGE(height_val, b.getInt32(2u)))),
        b.CreateAnd(u_fin_fast, v_fin_fast));
    auto *fast_br = b.CreateCondBr(fast_eligible, interior_check_bb, general_bb);
    fast_br->setMetadata(
        ::llvm::LLVMContext::MD_prof,
        ::llvm::MDBuilder{ctx}.createBranchWeights(1000000u, 1u));

    b.SetInsertPoint(interior_check_bb);
    auto *w_f_fast = b.CreateUIToFP(width_val, f32_ty);
    auto *h_f_fast = b.CreateUIToFP(height_val, f32_ty);
    ::llvm::Value *uv2 = ::llvm::PoisonValue::get(f32x2_ty);
    uv2 = b.CreateInsertElement(uv2, u_arg, b.getInt32(0u));
    uv2 = b.CreateInsertElement(uv2, v_arg, b.getInt32(1u));
    ::llvm::Value *wh2 = ::llvm::PoisonValue::get(f32x2_ty);
    wh2 = b.CreateInsertElement(wh2, w_f_fast, b.getInt32(0u));
    wh2 = b.CreateInsertElement(wh2, h_f_fast, b.getInt32(1u));
    auto *uv_frac = b.CreateFSub(
        uv2, b.CreateUnaryIntrinsic(::llvm::Intrinsic::floor, uv2));
    auto *is_repeat = b.CreateICmpEQ(addr_mode, b.getInt32(1u));
    auto *uv_wrapped = b.CreateSelect(is_repeat, uv_frac, uv2);
    auto *minus_half2 = float_vector_constant(b, 2u, -0.5f);
    auto *pos2 = b.CreateIntrinsic(
        ::llvm::Intrinsic::fma, {f32x2_ty}, {uv_wrapped, wh2, minus_half2});
    auto *zero2 = float_vector_constant(b, 2u, 0.0f);
    auto *one2 = float_vector_constant(b, 2u, 1.0f);
    auto *in_interior_f = b.CreateAndReduce(
        b.CreateAnd(
            b.CreateFCmpOGE(pos2, zero2),
            b.CreateFCmpOLT(pos2, b.CreateFSub(wh2, one2))));
    auto *interior_f_br = b.CreateCondBr(
        in_interior_f, interior_idx_bb, general_bb);
    interior_f_br->setMetadata(
        ::llvm::LLVMContext::MD_prof,
        ::llvm::MDBuilder{ctx}.createBranchWeights(1000000u, 1u));

    b.SetInsertPoint(interior_idx_bb);
    auto *idx2 = b.CreateFPToUI(pos2, i32x2_ty);
    auto *x0 = b.CreateExtractElement(idx2, b.getInt32(0u));
    auto *y0 = b.CreateExtractElement(idx2, b.getInt32(1u));
    auto *idx_ok = b.CreateAnd(
        b.CreateICmpULT(b.CreateAdd(x0, b.getInt32(1u)), width_val),
        b.CreateICmpULT(b.CreateAdd(y0, b.getInt32(1u)), height_val));
    auto *idx_br = b.CreateCondBr(idx_ok, interior_load_bb, general_bb);
    idx_br->setMetadata(
        ::llvm::LLVMContext::MD_prof,
        ::llvm::MDBuilder{ctx}.createBranchWeights(1000000u, 1u));

    b.SetInsertPoint(interior_load_bb);
    auto *frac2 = b.CreateFSub(pos2, b.CreateUIToFP(idx2, f32x2_ty));
    auto *xt4_fast = b.CreateVectorSplat(
        4u, b.CreateExtractElement(frac2, b.getInt32(0u)));
    auto *yt4_fast = b.CreateVectorSplat(
        4u, b.CreateExtractElement(frac2, b.getInt32(1u)));
    auto *w64_fast = b.CreateZExt(width_val, i64_ty);
    auto *idx0 = b.CreateAdd(
        b.CreateMul(b.CreateZExt(y0, i64_ty), w64_fast),
        b.CreateZExt(x0, i64_ty));
    auto *idx1 = b.CreateAdd(idx0, w64_fast);
    auto lerp4_fast = [&](::llvm::Value *lhs, ::llvm::Value *rhs,
                          ::llvm::Value *t) {
        return b.CreateIntrinsic(
            ::llvm::Intrinsic::fma, {f32x4_ty}, {b.CreateFSub(rhs, lhs), t, lhs});
    };
    if (is_float4) {
        auto load_f4 = [&](::llvm::Value *off) {
            auto *ptr = b.CreateGEP(f32x4_ty, data, off);
            auto *ld = b.CreateLoad(f32x4_ty, ptr);
            ld->setAlignment(::llvm::Align{4u});
            return ld;
        };
        auto *one64 = b.getInt64(1u);
        auto *v00 = load_f4(idx0);
        auto *v01 = load_f4(b.CreateAdd(idx0, one64));
        auto *v10 = load_f4(idx1);
        auto *v11 = load_f4(b.CreateAdd(idx1, one64));
        b.CreateRet(
            lerp4_fast(lerp4_fast(v00, v01, xt4_fast),
                       lerp4_fast(v10, v11, xt4_fast), yt4_fast));
    } else {
        auto load_row_pair = [&](::llvm::Value *row_off) {
            auto *ptr = b.CreateGEP(i8x4_ty, data, row_off);
            auto *bytes = b.CreateLoad(i8x8_ty, ptr);
            bytes->setAlignment(::llvm::Align{4u});
            auto *u16 = b.CreateZExt(bytes, i16x8_ty);
            auto *lo16 = b.CreateShuffleVector(u16, {0, 1, 2, 3});
            auto *hi16 = b.CreateShuffleVector(u16, {4, 5, 6, 7});
            return std::make_pair(
                b.CreateUIToFP(b.CreateZExt(lo16, i32x4_ty), f32x4_ty),
                b.CreateUIToFP(b.CreateZExt(hi16, i32x4_ty), f32x4_ty));
        };
        auto [v00, v01] = load_row_pair(idx0);
        auto [v10, v11] = load_row_pair(idx1);
        ::llvm::Value *res_fast = lerp4_fast(
            lerp4_fast(v00, v01, xt4_fast),
            lerp4_fast(v10, v11, xt4_fast), yt4_fast);
        res_fast = b.CreateFMul(
            res_fast, float_vector_constant(b, 4u, 1.0f / 255.0f));
        b.CreateRet(res_fast);
    }

    b.SetInsertPoint(general_bb);
    auto *zero4 = float_vector_constant(b, 4u, 0.0f);
    auto *zero2_g = float_vector_constant(b, 2u, 0.0f);
    auto *half2_g = float_vector_constant(b, 2u, 0.5f);
    auto *minus_half2_g = float_vector_constant(b, 2u, -0.5f);
    auto *one2_g = float_vector_constant(b, 2u, 1.0f);
    auto *two2_g = float_vector_constant(b, 2u, 2.0f);
    auto *inf_f = ::llvm::ConstantFP::getInfinity(f32_ty);
    auto *i1x2_ty = ::llvm::FixedVectorType::get(b.getInt1Ty(), 2u);
    auto *zero2_i32 = ::llvm::Constant::getNullValue(i32x2_ty);
    auto *one2_i32 = ::llvm::ConstantVector::getSplat(
        ::llvm::ElementCount::getFixed(2u), b.getInt32(1u));

    auto *w_f = b.CreateUIToFP(width_val, f32_ty);
    auto *h_f = b.CreateUIToFP(height_val, f32_ty);
    ::llvm::Value *uv2_g = ::llvm::PoisonValue::get(f32x2_ty);
    uv2_g = b.CreateInsertElement(uv2_g, u_arg, b.getInt32(0u));
    uv2_g = b.CreateInsertElement(uv2_g, v_arg, b.getInt32(1u));
    ::llvm::Value *size2 = ::llvm::PoisonValue::get(f32x2_ty);
    size2 = b.CreateInsertElement(size2, w_f, b.getInt32(0u));
    size2 = b.CreateInsertElement(size2, h_f, b.getInt32(1u));
    ::llvm::Value *size2_i = ::llvm::PoisonValue::get(i32x2_ty);
    size2_i = b.CreateInsertElement(size2_i, width_val, b.getInt32(0u));
    size2_i = b.CreateInsertElement(size2_i, height_val, b.getInt32(1u));
    auto *max_idx2 = b.CreateSub(size2_i, one2_i32);

    auto *fin_u = b.CreateFCmpOLT(
        b.CreateUnaryIntrinsic(::llvm::Intrinsic::fabs, u_arg), inf_f);
    auto *fin_v = b.CreateFCmpOLT(
        b.CreateUnaryIntrinsic(::llvm::Intrinsic::fabs, v_arg), inf_f);
    ::llvm::Value *fin2 = ::llvm::PoisonValue::get(i1x2_ty);
    fin2 = b.CreateInsertElement(fin2, fin_u, b.getInt32(0u));
    fin2 = b.CreateInsertElement(fin2, fin_v, b.getInt32(1u));

    auto *repeat_bb = ::llvm::BasicBlock::Create(ctx, "repeat", fn);
    auto *edge_bb = ::llvm::BasicBlock::Create(ctx, "edge", fn);
    auto *mirror_bb = ::llvm::BasicBlock::Create(ctx, "mirror", fn);
    auto *merge_bb = ::llvm::BasicBlock::Create(ctx, "merge", fn);

    auto *sw = b.CreateSwitch(addr_mode, repeat_bb, 2u);
    sw->addCase(b.getInt32(0u), edge_bb);
    sw->addCase(b.getInt32(2u), mirror_bb);

    // REPEAT (1)
    b.SetInsertPoint(repeat_bb);
    auto *safe_rep = b.CreateSelect(fin2, uv2_g, zero2_g);
    auto *frac_raw_g = b.CreateFSub(
        safe_rep, b.CreateUnaryIntrinsic(::llvm::Intrinsic::floor, safe_rep));
    auto *frac_ok_g = b.CreateAnd(
        b.CreateFCmpOGE(frac_raw_g, zero2_g),
        b.CreateFCmpOLT(frac_raw_g, one2_g));
    auto *f_rep = b.CreateSelect(frac_ok_g, frac_raw_g, zero2_g);
    auto *px_rep = b.CreateIntrinsic(
        ::llvm::Intrinsic::fma, {f32x2_ty}, {f_rep, size2, minus_half2_g});
    auto *fl_rep = b.CreateUnaryIntrinsic(::llvm::Intrinsic::floor, px_rep);
    auto *t_rep = b.CreateFSub(px_rep, fl_rep);
    auto *i0_rep = b.CreateFPToSI(b.CreateSelect(fin2, fl_rep, zero2_g), i32x2_ty);
    auto *i1_rep = b.CreateAdd(i0_rep, one2_i32);
    auto *lo_rep = b.CreateSelect(
        b.CreateICmpSLT(i0_rep, zero2_i32), max_idx2,
        b.CreateSelect(b.CreateICmpSGE(i0_rep, size2_i), max_idx2, i0_rep));
    auto *hi_rep = b.CreateSelect(
        b.CreateICmpSLT(i1_rep, zero2_i32), zero2_i32,
        b.CreateSelect(b.CreateICmpSGE(i1_rep, size2_i), zero2_i32, i1_rep));
    b.CreateBr(merge_bb);

    auto emit_w1_clamped = [&](::llvm::Value *folded) {
        auto *safe_c = b.CreateSelect(fin2, folded, zero2_g);
        auto *px = b.CreateIntrinsic(
            ::llvm::Intrinsic::fma, {f32x2_ty}, {safe_c, size2, minus_half2_g});
        auto *max_f = b.CreateMaxNum(b.CreateFSub(size2, one2_g), zero2_g);
        auto *px_c = b.CreateMinNum(b.CreateMaxNum(px, zero2_g), max_f);
        auto *fl = b.CreateUnaryIntrinsic(::llvm::Intrinsic::floor, px_c);
        auto *t = b.CreateFSub(px_c, fl);
        auto *i0 = b.CreateFPToUI(b.CreateSelect(fin2, fl, zero2_g), i32x2_ty);
        auto *i1 = b.CreateAdd(i0, one2_i32);
        auto *lo = b.CreateSelect(b.CreateICmpULT(i0, size2_i), i0, max_idx2);
        auto *hi = b.CreateSelect(b.CreateICmpULT(i1, size2_i), i1, max_idx2);
        return std::tuple{lo, hi, t};
    };

    // EDGE (0)
    b.SetInsertPoint(edge_bb);
    auto [lo_edge, hi_edge, t_edge] = emit_w1_clamped(uv2_g);
    b.CreateBr(merge_bb);

    // MIRROR (2)
    b.SetInsertPoint(mirror_bb);
    auto *mir_abs = b.CreateSelect(
        fin2, b.CreateUnaryIntrinsic(::llvm::Intrinsic::fabs, uv2_g), zero2_g);
    auto *mir_per = b.CreateUnaryIntrinsic(
        ::llvm::Intrinsic::floor, b.CreateFMul(mir_abs, half2_g));
    auto *mir_raw = b.CreateFSub(mir_abs, b.CreateFMul(mir_per, two2_g));
    auto *mir_ok = b.CreateAnd(
        b.CreateFCmpOGE(mir_raw, zero2_g), b.CreateFCmpOLT(mir_raw, two2_g));
    auto *mir_a = b.CreateSelect(mir_ok, mir_raw, zero2_g);
    auto *mir_fold = b.CreateSelect(
        b.CreateFCmpOLT(mir_a, one2_g), mir_a, b.CreateFSub(two2_g, mir_a));
    auto [lo_mir, hi_mir, t_mir] = emit_w1_clamped(mir_fold);
    b.CreateBr(merge_bb);

    b.SetInsertPoint(merge_bb);
    auto *lo2 = b.CreatePHI(i32x2_ty, 3u);
    lo2->addIncoming(lo_rep, repeat_bb);
    lo2->addIncoming(lo_edge, edge_bb);
    lo2->addIncoming(lo_mir, mirror_bb);
    auto *hi2 = b.CreatePHI(i32x2_ty, 3u);
    hi2->addIncoming(hi_rep, repeat_bb);
    hi2->addIncoming(hi_edge, edge_bb);
    hi2->addIncoming(hi_mir, mirror_bb);
    auto *t2 = b.CreatePHI(f32x2_ty, 3u);
    t2->addIncoming(t_rep, repeat_bb);
    t2->addIncoming(t_edge, edge_bb);
    t2->addIncoming(t_mir, mirror_bb);

    auto *xt_v4 = b.CreateVectorSplat(
        4u, b.CreateExtractElement(t2, b.getInt32(0u)));
    auto *yt_v4 = b.CreateVectorSplat(
        4u, b.CreateExtractElement(t2, b.getInt32(1u)));

    auto *u_lo = b.CreateZExt(b.CreateExtractElement(lo2, b.getInt32(0u)), i64_ty);
    auto *u_hi = b.CreateZExt(b.CreateExtractElement(hi2, b.getInt32(0u)), i64_ty);
    auto *v_lo = b.CreateZExt(b.CreateExtractElement(lo2, b.getInt32(1u)), i64_ty);
    auto *v_hi = b.CreateZExt(b.CreateExtractElement(hi2, b.getInt32(1u)), i64_ty);
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

    auto *all_valid = b.CreateAnd(
        b.CreateAnd(fin_u, fin_v),
        b.CreateAnd(
            b.CreateICmpNE(width_val, b.getInt32(0u)),
            b.CreateICmpNE(height_val, b.getInt32(0u))));
    auto *fast_bb = ::llvm::BasicBlock::Create(ctx, "fast", fn);
    auto *invalid_bb = ::llvm::BasicBlock::Create(ctx, "invalid", fn);
    auto *br = b.CreateCondBr(all_valid, fast_bb, invalid_bb);
    br->setMetadata(
        ::llvm::LLVMContext::MD_prof,
        ::llvm::MDBuilder{ctx}.createBranchWeights(1000000u, 1u));

    b.SetInsertPoint(fast_bb);
    auto *c00 = load_texel(off00);
    auto *c01 = load_texel(off01);
    auto *c10 = load_texel(off10);
    auto *c11 = load_texel(off11);
    auto lerp4 = [&](::llvm::Value *lhs, ::llvm::Value *rhs, ::llvm::Value *t) {
        return b.CreateIntrinsic(
            ::llvm::Intrinsic::fma, {f32x4_ty}, {b.CreateFSub(rhs, lhs), t, lhs});
    };
    ::llvm::Value *res = lerp4(lerp4(c00, c01, xt_v4), lerp4(c10, c11, xt_v4), yt_v4);
    if (!is_float4) {
        res = b.CreateFMul(res, float_vector_constant(b, 4u, 1.0f / 255.0f));
    }
    b.CreateRet(res);

    b.SetInsertPoint(invalid_bb);
    b.CreateRet(zero4);
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
    auto *inf_vec = ::llvm::ConstantVector::getSplat(
        ::llvm::ElementCount::getFixed(width),
        ::llvm::ConstantFP::getInfinity(f32_ty));
    auto *zero_i32_vec = ::llvm::Constant::getNullValue(i32_lanes);
    auto *one_i32_vec = ::llvm::ConstantVector::getSplat(
        ::llvm::ElementCount::getFixed(width), b.getInt32(1u));

    auto *w_i32 = b.CreateVectorSplat(width, width_val);
    auto *h_i32 = b.CreateVectorSplat(width, height_val);
    auto *w_f = b.CreateVectorSplat(width, b.CreateUIToFP(width_val, f32_ty));
    auto *h_f = b.CreateVectorSplat(width, b.CreateUIToFP(height_val, f32_ty));
    auto *u_fin = b.CreateFCmpOLT(
        b.CreateUnaryIntrinsic(::llvm::Intrinsic::fabs, u_arg), inf_vec);
    auto *v_fin = b.CreateFCmpOLT(
        b.CreateUnaryIntrinsic(::llvm::Intrinsic::fabs, v_arg), inf_vec);

    struct AxisCoord {
        ::llvm::Value *lo_idx;
        ::llvm::Value *hi_idx;
        ::llvm::Value *t;
        ::llvm::Value *valid;
    };

    // Single address-mode switch covering both U and V axes in exact texel space.
    auto *repeat_bb = ::llvm::BasicBlock::Create(ctx, "uv.repeat", fn);
    auto *edge_bb = ::llvm::BasicBlock::Create(ctx, "uv.edge", fn);
    auto *mirror_bb = ::llvm::BasicBlock::Create(ctx, "uv.mirror", fn);
    auto *merge_bb = ::llvm::BasicBlock::Create(ctx, "uv.merge", fn);

    auto *sw = b.CreateSwitch(addr_mode, repeat_bb, 2u);
    sw->addCase(b.getInt32(0u), edge_bb);
    sw->addCase(b.getInt32(2u), mirror_bb);

    auto build_repeat_axis = [&](::llvm::Value *coord, ::llvm::Value *sz_f,
                                 ::llvm::Value *sz_i, ::llvm::Value *fin) -> AxisCoord {
        auto *safe_c = b.CreateSelect(fin, coord, zero);
        auto *frac_raw = b.CreateFSub(
            safe_c, b.CreateUnaryIntrinsic(::llvm::Intrinsic::floor, safe_c));
        auto *frac_ok = b.CreateAnd(
            b.CreateFCmpOGE(frac_raw, zero),
            b.CreateFCmpOLT(frac_raw, one));
        auto *f = b.CreateSelect(frac_ok, frac_raw, zero);
        auto *px = b.CreateFSub(b.CreateFMul(f, sz_f), half);
        auto *fl = b.CreateUnaryIntrinsic(::llvm::Intrinsic::floor, px);
        auto *t = b.CreateFSub(px, fl);
        auto *i0 = b.CreateFPToSI(b.CreateSelect(fin, fl, zero), i32_lanes);
        auto *i1 = b.CreateAdd(i0, one_i32_vec);
        auto *max_idx = b.CreateSub(sz_i, one_i32_vec);
        auto *lo = b.CreateSelect(
            b.CreateICmpSLT(i0, zero_i32_vec), max_idx,
            b.CreateSelect(b.CreateICmpSGE(i0, sz_i), max_idx, i0));
        auto *hi = b.CreateSelect(
            b.CreateICmpSLT(i1, zero_i32_vec), zero_i32_vec,
            b.CreateSelect(b.CreateICmpSGE(i1, sz_i), zero_i32_vec, i1));
        return {lo, hi, t, fin};
    };

    auto build_edge_or_mirror_axis = [&](::llvm::Value *folded, ::llvm::Value *sz_f,
                                         ::llvm::Value *sz_i, ::llvm::Value *fin) -> AxisCoord {
        auto *safe_c = b.CreateSelect(fin, folded, zero);
        auto *px = b.CreateFSub(b.CreateFMul(safe_c, sz_f), half);
        auto *max_f = b.CreateMaxNum(b.CreateFSub(sz_f, one), zero);
        auto *px_c = b.CreateMinNum(b.CreateMaxNum(px, zero), max_f);
        auto *fl = b.CreateUnaryIntrinsic(::llvm::Intrinsic::floor, px_c);
        auto *t = b.CreateFSub(px_c, fl);
        auto *i0 = b.CreateFPToUI(b.CreateSelect(fin, fl, zero), i32_lanes);
        auto *i1 = b.CreateAdd(i0, one_i32_vec);
        auto *max_idx = b.CreateSub(sz_i, one_i32_vec);
        auto *lo = b.CreateSelect(b.CreateICmpULT(i0, sz_i), i0, max_idx);
        auto *hi = b.CreateSelect(b.CreateICmpULT(i1, sz_i), i1, max_idx);
        return {lo, hi, t, fin};
    };

    // REPEAT (1)
    b.SetInsertPoint(repeat_bb);
    auto rep_x = build_repeat_axis(u_arg, w_f, w_i32, u_fin);
    auto rep_y = build_repeat_axis(v_arg, h_f, h_i32, v_fin);
    b.CreateBr(merge_bb);

    // EDGE (0)
    b.SetInsertPoint(edge_bb);
    auto edge_x = build_edge_or_mirror_axis(u_arg, w_f, w_i32, u_fin);
    auto edge_y = build_edge_or_mirror_axis(v_arg, h_f, h_i32, v_fin);
    b.CreateBr(merge_bb);

    // MIRROR (2)
    b.SetInsertPoint(mirror_bb);
    auto fold_mirror = [&](::llvm::Value *val, ::llvm::Value *fin) {
        auto *raw_a = b.CreateSelect(
            fin, b.CreateUnaryIntrinsic(::llvm::Intrinsic::fabs, val), zero);
        auto *period = b.CreateUnaryIntrinsic(
            ::llvm::Intrinsic::floor, b.CreateFMul(raw_a, half));
        auto *a_raw = b.CreateFSub(raw_a, b.CreateFMul(period, two));
        auto *a_ok = b.CreateAnd(
            b.CreateFCmpOGE(a_raw, zero), b.CreateFCmpOLT(a_raw, two));
        auto *a = b.CreateSelect(a_ok, a_raw, zero);
        return b.CreateSelect(
            b.CreateFCmpOLT(a, one), a, b.CreateFSub(two, a));
    };
    auto mir_x = build_edge_or_mirror_axis(fold_mirror(u_arg, u_fin), w_f, w_i32, u_fin);
    auto mir_y = build_edge_or_mirror_axis(fold_mirror(v_arg, v_fin), h_f, h_i32, v_fin);
    b.CreateBr(merge_bb);

    b.SetInsertPoint(merge_bb);
    auto make_phi_f = [&](::llvm::Value *rv, ::llvm::Value *ev, ::llvm::Value *mv) {
        auto *p = b.CreatePHI(f32_lanes, 3u);
        p->addIncoming(rv, repeat_bb);
        p->addIncoming(ev, edge_bb);
        p->addIncoming(mv, mirror_bb);
        return p;
    };
    auto make_phi_i = [&](::llvm::Value *rv, ::llvm::Value *ev, ::llvm::Value *mv) {
        auto *p = b.CreatePHI(i32_lanes, 3u);
        p->addIncoming(rv, repeat_bb);
        p->addIncoming(ev, edge_bb);
        p->addIncoming(mv, mirror_bb);
        return p;
    };
    AxisCoord x{
        make_phi_i(rep_x.lo_idx, edge_x.lo_idx, mir_x.lo_idx),
        make_phi_i(rep_x.hi_idx, edge_x.hi_idx, mir_x.hi_idx),
        make_phi_f(rep_x.t, edge_x.t, mir_x.t),
        u_fin};
    AxisCoord y{
        make_phi_i(rep_y.lo_idx, edge_y.lo_idx, mir_y.lo_idx),
        make_phi_i(rep_y.hi_idx, edge_y.hi_idx, mir_y.hi_idx),
        make_phi_f(rep_y.t, edge_y.t, mir_y.t),
        v_fin};

    auto *width_i64 = b.CreateVectorSplat(
        width, b.CreateZExt(width_val, b.getInt64Ty()));
    auto *data_lanes = b.CreateVectorSplat(width, data);
    auto *finite_mask = b.CreateAnd(active_mask, b.CreateAnd(x.valid, y.valid));
    auto *m00 = finite_mask;
    auto *m01 = finite_mask;
    auto *m10 = finite_mask;
    auto *m11 = finite_mask;
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

}// namespace luisa::compute::simd::detail
