//
// Created by mike on 11/1/25.
//

#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

llvm::Value *CUDACodegenLLVMImpl::_convert_llvm_reg_value_to_mem(IB &b, llvm::Value *reg_v, const Type *type) noexcept {
    auto reg_type = reg_v->getType();
    auto type_info = _get_llvm_type(type);
    LUISA_DEBUG_ASSERT(reg_type == type_info->reg_type);
    if (reg_type == type_info->mem_type) { return reg_v; }
    // vectors are stored as padded arrays in memory
    if (reg_type->isVectorTy()) {
        auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(type_info->mem_type));
        auto dim = llvm::cast<llvm::VectorType>(reg_type)->getElementCount().getFixedValue();
        for (auto i = 0; i < dim; i++) {
            auto llvm_elem = b.CreateExtractElement(reg_v, i);
            result = b.CreateInsertValue(result, llvm_elem, i);
        }
        result->setName(reg_v->getName().str() + ".to.mem");
        return result;
    }
    // array or matrix types, we need to convert element by element
    if (reg_type->isArrayTy()) {
        auto llvm_dst = static_cast<llvm::Value *>(llvm::PoisonValue::get(type_info->mem_type));
        auto dim = type->dimension();
        LUISA_DEBUG_ASSERT(reg_type->getArrayNumElements() == dim);
        auto elem_type = type->is_matrix() ? Type::vector(type->element(), dim) : type->element();
        for (auto i = 0u; i < dim; i++) {
            auto llvm_reg_elem = b.CreateExtractValue(reg_v, i);
            auto llvm_mem_elem = _convert_llvm_reg_value_to_mem(b, llvm_reg_elem, elem_type);
            llvm_dst = b.CreateInsertValue(llvm_dst, llvm_mem_elem, i);
        }
        llvm_dst->setName(reg_v->getName().str() + ".to.mem");
        return llvm_dst;
    }
    // must be structure type now
    LUISA_DEBUG_ASSERT(reg_type->isStructTy());
    auto llvm_dst = static_cast<llvm::Value *>(llvm::PoisonValue::get(type_info->mem_type));
    auto member_count = type->members().size();
    for (auto i = 0u; i < member_count; i++) {
        auto llvm_reg_member = b.CreateExtractValue(reg_v, i);
        auto llvm_mem_member = _convert_llvm_reg_value_to_mem(b, llvm_reg_member, type->members()[i]);
        llvm_dst = b.CreateInsertValue(llvm_dst, llvm_mem_member, type_info->member_indices[i]);
    }
    llvm_dst->setName(reg_v->getName().str() + ".to.mem");
    return llvm_dst;
}

llvm::Value *CUDACodegenLLVMImpl::_convert_llvm_mem_value_to_reg(IB &b, llvm::Value *mem_v, const Type *type) noexcept {
    auto mem_type = mem_v->getType();
    auto type_info = _get_llvm_type(type);
    LUISA_DEBUG_ASSERT(mem_type == type_info->mem_type);
    if (mem_type == type_info->reg_type) { return mem_v; }
    // vectors are stored as padded arrays in memory
    if (type_info->reg_type->isVectorTy()) {
        auto reg_v = static_cast<llvm::Value *>(llvm::PoisonValue::get(type_info->reg_type));
        auto dim = llvm::cast<llvm::VectorType>(type_info->reg_type)->getElementCount().getFixedValue();
        for (auto i = 0; i < dim; i++) {
            auto llvm_elem = b.CreateExtractValue(mem_v, i);
            reg_v = b.CreateInsertElement(reg_v, llvm_elem, i);
        }
        reg_v->setName(mem_v->getName().str() + ".to.reg");
        return reg_v;
    }
    // array or matrix types, we need to convert element by element
    if (type_info->reg_type->isArrayTy()) {
        auto reg_v = static_cast<llvm::Value *>(llvm::PoisonValue::get(type_info->reg_type));
        auto dim = type->dimension();
        LUISA_DEBUG_ASSERT(mem_type->getArrayNumElements() == dim);
        auto elem_type = type->is_matrix() ? Type::vector(type->element(), dim) : type->element();
        for (auto i = 0u; i < dim; i++) {
            auto llvm_mem_elem = b.CreateExtractValue(mem_v, i);
            auto llvm_reg_elem = _convert_llvm_mem_value_to_reg(b, llvm_mem_elem, elem_type);
            reg_v = b.CreateInsertValue(reg_v, llvm_reg_elem, i);
        }
        reg_v->setName(mem_v->getName().str() + ".to.reg");
        return reg_v;
    }
    // must be structure type now
    LUISA_DEBUG_ASSERT(type_info->reg_type->isStructTy());
    auto reg_v = static_cast<llvm::Value *>(llvm::PoisonValue::get(type_info->reg_type));
    auto member_count = type->members().size();
    for (auto i = 0u; i < member_count; i++) {
        auto llvm_mem_member = b.CreateExtractValue(mem_v, type_info->member_indices[i]);
        auto llvm_reg_member = _convert_llvm_mem_value_to_reg(b, llvm_mem_member, type->members()[i]);
        reg_v = b.CreateInsertValue(reg_v, llvm_reg_member, i);
    }
    reg_v->setName(mem_v->getName().str() + ".to.reg");
    return reg_v;
}

llvm::Value *CUDACodegenLLVMImpl::_bitwise_cast(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_src, const Type *src_type, const Type *dst_type) noexcept {
    if (src_type == dst_type) { return llvm_src; }
    auto llvm_dst_type = _get_llvm_type(dst_type);
    // scalars or vectors, we can use the built-in llvm bitwise cast
    if ((src_type->is_scalar() || src_type->is_vector()) && (dst_type->is_scalar() || dst_type->is_vector()) &&
        !src_type->is_bool_or_bool_vector() && !dst_type->is_bool_or_bool_vector()) {
        return b.CreateBitCast(llvm_src, llvm_dst_type->reg_type);
    }
    // generic, we make a temporary alloca, store the src value, and load as the dst type
    auto llvm_src_mem = _convert_llvm_reg_value_to_mem(b, llvm_src, src_type);
    auto llvm_temp = _create_temp_in_alloca_block(func_ctx, llvm_src_mem->getType(), src_type->alignment());
    b.CreateAlignedStore(llvm_src_mem, llvm_temp, llvm::Align{src_type->alignment()});
    auto llvm_dst_mem = b.CreateAlignedLoad(llvm_dst_type->mem_type, llvm_temp, llvm::Align{dst_type->alignment()});
    return _convert_llvm_mem_value_to_reg(b, llvm_dst_mem, dst_type);
}

llvm::Value *CUDACodegenLLVMImpl::_static_cast(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_src, const Type *src_type, const Type *dst_type) noexcept {
    if (src_type == dst_type) { return llvm_src; }
    // scalar to scalar
    if (src_type->is_scalar() && dst_type->is_scalar()) {
        return _static_cast_scalar_to_scalar(b, func_ctx, llvm_src, src_type, dst_type);
    }
    // scalar to vector
    if (src_type->is_scalar() && dst_type->is_vector()) {
        return _static_cast_scalar_to_vector(b, func_ctx, llvm_src, src_type, dst_type);
    }
    // vector to vector
    if (src_type->is_vector() && dst_type->is_vector()) {
        return _static_cast_vector_to_vector(b, func_ctx, llvm_src, src_type, dst_type);
    }
    // other conversions are not supported
    LUISA_ERROR_WITH_LOCATION("Invalid types for static cast: {} -> {}",
                              src_type->description(), dst_type->description());
}

llvm::Value *CUDACodegenLLVMImpl::_static_cast_scalar_to_vector(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_src, const Type *src_type, const Type *dst_type) noexcept {
    auto dst_elem_type = dst_type->element();
    // cast the src to dst element type first
    auto llvm_dst_elem = _static_cast_scalar_to_scalar(b, func_ctx, llvm_src, src_type, dst_elem_type);
    // splat to vector
    auto dim = dst_type->dimension();
    return b.CreateVectorSplat(dim, llvm_dst_elem, llvm_src->getName().str() + ".to.vec");
}

namespace detail {

// OptiX's ptx2llvm module has bugs when casting half vectors, we need to do it element by element
[[nodiscard]] inline llvm::Value *safe_fp_cast_impl(CUDACodegenLLVMImpl::IB &b, bool uses_optix,
                                                    llvm::Value *llvm_src, llvm::Type *dst_type,
                                                    const llvm::Twine &name = "") noexcept {
    auto src_type = llvm_src->getType();
    if (src_type == dst_type) { return llvm_src; }
    if (!uses_optix || (!src_type->getScalarType()->isHalfTy() && !dst_type->getScalarType()->isHalfTy())) {
        return b.CreateFPCast(llvm_src, dst_type);
    }
    auto inline_asm = [&] {
        auto src_scalar_t = src_type->getScalarType();
        auto dst_scalar_t = dst_type->getScalarType();
        auto get_ptx_type_and_reg = [](llvm::Type *t) noexcept -> std::pair<const char *, const char *> {
            if (t->isHalfTy()) { return {"f16", "h"}; }
            if (t->isFloatTy()) { return {"f32", "f"}; }
            if (t->isDoubleTy()) { return {"f64", "d"}; }
            LUISA_ERROR_WITH_LOCATION("Invalid floating-point type for safe_fp_cast_impl.");
        };
        auto [src_ptx_type, src_asm_reg] = get_ptx_type_and_reg(src_scalar_t);
        auto [dst_ptx_type, dst_asm_reg] = get_ptx_type_and_reg(dst_scalar_t);
        auto ptx = src_scalar_t->isHalfTy() ? fmt::format("cvt.{}.{} $0, $1;", dst_ptx_type, src_ptx_type) :
                                              fmt::format("cvt.rn.{}.{} $0, $1;", dst_ptx_type, src_ptx_type);
        auto constraints = fmt::format("={},{}", dst_asm_reg, src_asm_reg);
        auto asm_type = llvm::FunctionType::get(dst_scalar_t, {src_scalar_t}, false);
        return llvm::InlineAsm::get(asm_type, std::string_view{ptx}, std::string_view{constraints}, false);
    }();
    if (dst_type->isVectorTy()) {
        auto dim = llvm::cast<llvm::VectorType>(dst_type)->getElementCount().getFixedValue();
        LUISA_DEBUG_ASSERT(src_type->isVectorTy() &&
                           llvm::cast<llvm::VectorType>(src_type)->getElementCount().getFixedValue() == dim);
        auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(dst_type));
        for (auto i = 0; i < dim; i++) {
            auto llvm_elem = b.CreateExtractElement(llvm_src, i);
            auto llvm_casted_elem = b.CreateCall(inline_asm, llvm_elem);
            result = b.CreateInsertElement(result, llvm_casted_elem, i);
        }
        result->setName(name);
        return result;
    }
    auto result = b.CreateCall(inline_asm, llvm_src);
    result->setName(name);
    return result;
}

[[nodiscard]] inline llvm::Value *cast_llvm_value_based_on_elem_types(CUDACodegenLLVMImpl::IB &b, bool uses_optix,
                                                                      llvm::Type *llvm_dst_type, llvm::Value *llvm_src,
                                                                      const Type *src_elem_type, const Type *dst_elem_type) noexcept {
    // other types to bool, implemented as src != 0
    if (dst_elem_type->is_bool()) {
        auto zero = llvm::Constant::getNullValue(llvm_src->getType());
        return src_elem_type->is_float() ?
                   b.CreateFCmpUNE(llvm_src, zero, llvm_src->getName().str() + ".to.bool") :
                   b.CreateICmpNE(llvm_src, zero, llvm_src->getName().str() + ".to.bool");
    }
    // bool to other types, implemented as selection
    if (src_elem_type->is_bool()) {
        auto llvm_zero = llvm::Constant::getNullValue(llvm_dst_type);
        auto llvm_one = dst_elem_type->is_float() ?
                            static_cast<llvm::Constant *>(llvm::ConstantFP::get(llvm_dst_type, 1.)) :
                            static_cast<llvm::Constant *>(llvm::ConstantInt::get(llvm_dst_type, 1));
        return b.CreateSelect(llvm_src, llvm_one, llvm_zero, llvm_src->getName().str() + ".from.bool");
    }
    struct ScalarTraits {
        bool is_fp;
        bool is_signed;// only valid if is_fp == false
    };
    auto get_scalar_traits = [](const Type *type) noexcept -> ScalarTraits {
        if (type->is_float()) { return {.is_fp = true, .is_signed = false}; }
        switch (type->tag()) {
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::INT16: [[fallthrough]];
            case Type::Tag::INT32: [[fallthrough]];
            case Type::Tag::INT64:
            case Type::Tag::INT4:// placeholder: treated as signed i8
                return {.is_fp = false, .is_signed = true};
            case Type::Tag::UINT8: [[fallthrough]];
            case Type::Tag::UINT16: [[fallthrough]];
            case Type::Tag::UINT32: [[fallthrough]];
            case Type::Tag::UINT64:
            case Type::Tag::FLOAT8_E4M3: [[fallthrough]];// placeholder: unsigned i8
            case Type::Tag::FLOAT8_E5M2: [[fallthrough]];
            case Type::Tag::FP4_E2M1:
                return {.is_fp = false, .is_signed = false};
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid scalar type: {}.", type->description());
    };
    auto [src_is_fp, src_is_signed] = get_scalar_traits(src_elem_type);
    auto [dst_is_fp, dst_is_signed] = get_scalar_traits(dst_elem_type);
    // float to other types
    if (src_is_fp) {
        return dst_is_fp     ? safe_fp_cast_impl(b, uses_optix, llvm_src, llvm_dst_type, llvm_src->getName().str() + ".fp.to.fp") :
               dst_is_signed ? b.CreateFPToSI(llvm_src, llvm_dst_type, llvm_src->getName().str() + ".fp.to.sint") :
                               b.CreateFPToUI(llvm_src, llvm_dst_type, llvm_src->getName().str() + ".fp.to.uint");
    }
    // int to float
    if (dst_is_fp) {
        return src_is_signed ? b.CreateSIToFP(llvm_src, llvm_dst_type, llvm_src->getName().str() + ".sint.to.fp") :
                               b.CreateUIToFP(llvm_src, llvm_dst_type, llvm_src->getName().str() + ".uint.to.fp");
    }
    // int to int, signedness is determined by src type
    return b.CreateIntCast(llvm_src, llvm_dst_type, src_is_signed, llvm_src->getName().str() + ".int.to.int");
}

}// namespace detail

llvm::Value *CUDACodegenLLVMImpl::_static_cast_scalar_to_scalar(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_src, const Type *src_type, const Type *dst_type) noexcept {
    if (src_type == dst_type) { return llvm_src; }
    auto llvm_dst_type = _get_llvm_type(dst_type)->reg_type;
    return detail::cast_llvm_value_based_on_elem_types(b, _rt_analysis.uses_ray_tracing, llvm_dst_type, llvm_src, src_type, dst_type);
}

llvm::Value *CUDACodegenLLVMImpl::_static_cast_vector_to_vector(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_src, const Type *src_type, const Type *dst_type) noexcept {
    if (src_type == dst_type) { return llvm_src; }
    LUISA_DEBUG_ASSERT(src_type->dimension() == dst_type->dimension());
    auto src_elem_type = src_type->element();
    auto dst_elem_type = dst_type->element();
    auto llvm_dst_type = _get_llvm_type(dst_type)->reg_type;
    return detail::cast_llvm_value_based_on_elem_types(b, _rt_analysis.uses_ray_tracing, llvm_dst_type, llvm_src, src_elem_type, dst_elem_type);
}

llvm::Value *CUDACodegenLLVMImpl::_texel_cast(IB &b, llvm::Value *llvm_src, llvm::Type *dst_type) noexcept {
    auto src_type = llvm_src->getType();
    LUISA_DEBUG_ASSERT(src_type->isVectorTy() && dst_type->isVectorTy() &&
                       llvm::cast<llvm::VectorType>(src_type)->getElementCount().getFixedValue() == 4 &&
                       llvm::cast<llvm::VectorType>(dst_type)->getElementCount().getFixedValue() == 4);
    // float to float
    if (src_type->isFPOrFPVectorTy() && dst_type->isFPOrFPVectorTy()) {
        return _safe_fp_cast(b, llvm_src, dst_type, llvm_src->getName().str() + ".texel.cast.fp.to.fp");
    }
    // int to float, normalize the int into [0, 1]
    if (src_type->isIntOrIntVectorTy() && dst_type->isFPOrFPVectorTy()) {
        auto src_max = llvm::Constant::getAllOnesValue(src_type);
        auto src_to_fp = b.CreateUIToFP(llvm_src, dst_type);
        auto src_max_fp = b.CreateUIToFP(src_max, dst_type);
        return b.CreateFDiv(src_to_fp, src_max_fp, llvm_src->getName().str() + ".texel.cast.int.to.fp");
    }
    // float to int, denormalize the float into [0, max_int]
    if (src_type->isFPOrFPVectorTy() && dst_type->isIntOrIntVectorTy()) {
        auto dst_max = llvm::Constant::getAllOnesValue(dst_type);
        auto dst_max_fp = b.CreateUIToFP(dst_max, src_type);
        auto src_zero = llvm::Constant::getNullValue(src_type);
        auto src_one = llvm::ConstantFP::get(src_type, 1.);
        auto src_clamped = b.CreateMinNum(b.CreateMaxNum(llvm_src, src_zero), src_one);
        auto src_scaled = b.CreateFMul(src_clamped, dst_max_fp);
        auto src_round = b.CreateUnaryIntrinsic(llvm::Intrinsic::rint, src_scaled);
        return b.CreateFPToUI(src_round, dst_type, llvm_src->getName().str() + ".texel.cast.fp.to.int");
    }
    // must be int to int
    LUISA_DEBUG_ASSERT(src_type->isIntOrIntVectorTy() && dst_type->isIntOrIntVectorTy());
    return b.CreateIntCast(llvm_src, dst_type, false, llvm_src->getName().str() + ".texel.cast.int.to.int");
}

llvm::Value *CUDACodegenLLVMImpl::_safe_fp_cast(IB &b, llvm::Value *llvm_src, llvm::Type *dst_type, const llvm::Twine &name) const noexcept {
    return detail::safe_fp_cast_impl(b, _rt_analysis.uses_ray_tracing, llvm_src, dst_type, name);
}

llvm::Value *CUDACodegenLLVMImpl::_translate_cast_inst(IB &b, FunctionContext &func_ctx, const xir::CastInst *inst) noexcept {
    auto src = inst->value();
    auto src_type = src->type();
    auto dst_type = inst->type();
    auto llvm_src = _get_llvm_value(b, func_ctx, src);
    switch (inst->op()) {
        case xir::CastOp::STATIC_CAST: return _static_cast(b, func_ctx, llvm_src, src_type, dst_type);
        case xir::CastOp::BITWISE_CAST: return _bitwise_cast(b, func_ctx, llvm_src, src_type, dst_type);
    }
    LUISA_ERROR_WITH_LOCATION("Invalid cast instruction.");
}

}// namespace luisa::compute::cuda
