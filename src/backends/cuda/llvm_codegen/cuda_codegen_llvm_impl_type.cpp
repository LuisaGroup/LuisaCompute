//
// Created by mike on 9/21/25.
//

#include <luisa/dsl/rtx/ray_query.h>

#include "../cuda_buffer.h"
#include "../cuda_texture.h"
#include "../cuda_bindless_array.h"
#include "../cuda_accel.h"
#include "../optix_api.h"
#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

namespace detail {

static void luisa_check_llvm_type_size_and_alignment(
    const llvm::DataLayout &data_layout, llvm::Type *type,
    size_t expected_size, size_t expected_alignment) noexcept {
    auto llvm_size = data_layout.getTypeAllocSize(type);
    auto llvm_align = data_layout.getABITypeAlign(type);
    auto type_str = [&]() noexcept {
        std::string str;
        llvm::raw_string_ostream os{str};
        type->print(os);
        return str;
    };
    LUISA_ASSERT(llvm_size.getFixedValue() == expected_size && llvm_align.value() <= expected_alignment,
                 "Type '{}' size or alignment mismatch: "
                 "expected size {}, alignment {}; "
                 "actual size {}, alignment {}.",
                 type_str(), expected_size, expected_alignment,
                 llvm_size.getFixedValue(), llvm_align.value());
}

}// namespace detail

size_t CUDACodegenLLVMImpl::_get_type_alignment(const Type *type) noexcept {
    if (type->is_basic() || type->is_array() || type->is_structure()) {
        return type->alignment();
    }
    return 16;
}

const CUDACodegenLLVMImpl::LLVMTypeInfo *
CUDACodegenLLVMImpl::_get_llvm_type(const Type *type) noexcept {
    if (auto iter = _xir_to_llvm_type.find(type); iter != _xir_to_llvm_type.end()) {
        return iter->second.get();
    }
    auto llvm_type_info = [this, type]() noexcept {
        if (type == nullptr) {
            auto llvm_void_type = llvm::Type::getVoidTy(_llvm_context);
            return luisa::make_unique<LLVMTypeInfo>(llvm_void_type, llvm_void_type, luisa::vector<size_t>{});
        }
        auto make_llvm_type_info = [this](auto mem_t, auto reg_t, size_t s, size_t a,
                                          luisa::vector<size_t> member_indices = {},
                                          luisa::vector<size_t> member_offsets = {}) noexcept {
            detail::luisa_check_llvm_type_size_and_alignment(*_data_layout, mem_t, s, a);
            return luisa::make_unique<LLVMTypeInfo>(mem_t, reg_t, std::move(member_indices), std::move(member_offsets));
        };
        switch (type->tag()) {
            case Type::Tag::BOOL: {
                auto llvm_i1_type = llvm::Type::getInt1Ty(_llvm_context);
                return make_llvm_type_info(llvm_i1_type, llvm_i1_type, sizeof(bool), alignof(bool));
            }
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::UINT8: {
                auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
                return make_llvm_type_info(llvm_i8_type, llvm_i8_type, sizeof(int8_t), alignof(int8_t));
            }
            case Type::Tag::INT16: [[fallthrough]];
            case Type::Tag::UINT16: {
                auto llvm_i16_type = llvm::Type::getInt16Ty(_llvm_context);
                return make_llvm_type_info(llvm_i16_type, llvm_i16_type, sizeof(int16_t), alignof(int16_t));
            }
            case Type::Tag::INT32: [[fallthrough]];
            case Type::Tag::UINT32: {
                auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
                return make_llvm_type_info(llvm_i32_type, llvm_i32_type, sizeof(int32_t), alignof(int32_t));
            }
            case Type::Tag::INT64: [[fallthrough]];
            case Type::Tag::UINT64: {
                auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
                return make_llvm_type_info(llvm_i64_type, llvm_i64_type, sizeof(int64_t), alignof(int64_t));
            }
            case Type::Tag::FLOAT16: {
                auto llvm_half_type = llvm::Type::getHalfTy(_llvm_context);
                return make_llvm_type_info(llvm_half_type, llvm_half_type, sizeof(luisa::half), alignof(luisa::half));
            }
            case Type::Tag::FLOAT32: {
                auto llvm_float_type = llvm::Type::getFloatTy(_llvm_context);
                return make_llvm_type_info(llvm_float_type, llvm_float_type, sizeof(float), alignof(float));
            }
            case Type::Tag::FLOAT64: {
                auto llvm_double_type = llvm::Type::getDoubleTy(_llvm_context);
                return make_llvm_type_info(llvm_double_type, llvm_double_type, sizeof(double), alignof(double));
            }
            case Type::Tag::VECTOR: {
                auto elem = _get_llvm_type(type->element());
                auto dim = type->dimension();
                LUISA_DEBUG_ASSERT(dim == 2 || dim == 3 || dim == 4);
                auto llvm_reg_type = llvm::VectorType::get(elem->reg_type, dim, false);
                auto llvm_mem_type = llvm::ArrayType::get(elem->mem_type, luisa::align(dim, 2));
                // other vector types are the same in memory and registers
                return make_llvm_type_info(llvm_mem_type, llvm_reg_type, type->size(), type->alignment());
            }
            case Type::Tag::MATRIX: {
                auto dim = type->dimension();
                auto llvm_col_type = _get_llvm_type(Type::vector(type->element(), dim));
                auto llvm_reg_type = llvm::ArrayType::get(llvm_col_type->reg_type, dim);
                auto llvm_mem_type = llvm::ArrayType::get(llvm_col_type->mem_type, dim);
                return make_llvm_type_info(llvm_mem_type, llvm_reg_type, type->size(), type->alignment());
            }
            case Type::Tag::ARRAY: {
                auto llvm_elem_type = _get_llvm_type(type->element());
                auto llvm_reg_type = llvm::ArrayType::get(llvm_elem_type->reg_type, type->dimension());
                auto llvm_mem_type = llvm::ArrayType::get(llvm_elem_type->mem_type, type->dimension());
                return make_llvm_type_info(llvm_mem_type, llvm_reg_type, type->size(), type->alignment());
            }
            case Type::Tag::STRUCTURE: {
                luisa::vector<size_t> member_indices;
                luisa::vector<size_t> member_offsets;
                auto member_count = type->members().size();
                member_indices.reserve(member_count);
                member_offsets.reserve(member_count);
                llvm::SmallVector<llvm::Type *> llvm_member_reg_types;
                llvm::SmallVector<llvm::Type *> llvm_member_mem_types;
                auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
                auto current_offset = static_cast<size_t>(0u);
                for (auto member : type->members()) {
                    auto llvm_member_type = _get_llvm_type(member);
                    // reg type is trivial
                    llvm_member_reg_types.emplace_back(llvm_member_type->reg_type);
                    // for mem type, we should take care of alignment
                    auto next_offset = luisa::align(current_offset, member->alignment());
                    if (next_offset > current_offset) {
                        llvm_member_mem_types.emplace_back(
                            llvm::ArrayType::get(llvm_i8_type, next_offset - current_offset));
                    }
                    member_indices.emplace_back(llvm_member_mem_types.size());
                    member_offsets.emplace_back(next_offset);
                    llvm_member_mem_types.emplace_back(llvm_member_type->mem_type);
                    current_offset = next_offset + member->size();
                }
                if (current_offset < type->size()) {
                    llvm_member_mem_types.emplace_back(
                        llvm::ArrayType::get(llvm_i8_type, type->size() - current_offset));
                }
                auto llvm_reg_type = llvm::StructType::get(_llvm_context, llvm_member_reg_types, false);
                auto llvm_mem_type = llvm::StructType::get(_llvm_context, llvm_member_mem_types, false);
                return make_llvm_type_info(llvm_mem_type, llvm_reg_type, type->size(), type->alignment(),
                                           std::move(member_indices), std::move(member_offsets));
            }
            case Type::Tag::BUFFER: {
                auto llvm_type = _get_llvm_buffer_type();
                return make_llvm_type_info(llvm_type, llvm_type,
                                           sizeof(CUDABuffer::Binding),
                                           alignof(CUDABuffer::Binding));
            }
            case Type::Tag::TEXTURE: {
                auto llvm_type = _get_llvm_texture_type();
                return make_llvm_type_info(llvm_type, llvm_type,
                                           sizeof(CUDATexture::Binding),
                                           alignof(CUDATexture::Binding));
            }
            case Type::Tag::BINDLESS_ARRAY: {
                auto llvm_type = _get_llvm_bindless_array_type();
                return make_llvm_type_info(llvm_type, llvm_type,
                                           sizeof(CUDABindlessArray::Binding),
                                           alignof(CUDABindlessArray::Binding));
            }
            case Type::Tag::ACCEL: {
                auto llvm_type = _get_llvm_accel_type();
                return make_llvm_type_info(llvm_type, llvm_type,
                                           sizeof(CUDAAccel::Binding),
                                           alignof(CUDAAccel::Binding));
            }
            case Type::Tag::COOPERATIVE_VECTOR: LUISA_NOT_IMPLEMENTED("Cooperative vector type");
            case Type::Tag::COOPERATIVE_VECTOR_REF: LUISA_NOT_IMPLEMENTED("Cooperative vector ref type");
            case Type::Tag::COOPERATIVE_MATRIX_REF: LUISA_NOT_IMPLEMENTED("Cooperative matrix ref type");
            case Type::Tag::CUSTOM: {
                if (type == Type::of<RayQueryAll>() || type == Type::of<RayQueryAny>()) {
                    auto llvm_type = _get_llvm_ray_query_type();
                    return make_llvm_type_info(llvm_type, llvm_type, sizeof(uint8_t), alignof(uint8_t));
                }
                LUISA_NOT_IMPLEMENTED("Custom type: {}.", type->description());
            }
        }
        LUISA_ERROR_WITH_LOCATION("Invalid type: {}.", type->description());
    }();
    auto [iter, success] = _xir_to_llvm_type.try_emplace(type, std::move(llvm_type_info));
    LUISA_ASSERT(success, "Failed to insert LLVM type.");
    return iter->second.get();
}

const CUDACodegenLLVMImpl::KernelArgumentStruct *CUDACodegenLLVMImpl::_get_kernel_argument_struct(const xir::KernelFunction *func) noexcept {
    if (auto iter = _kernel_arg_struct_types.find(func); iter != _kernel_arg_struct_types.end()) {
        return iter->second.get();
    }
    std::vector<llvm::Type *> llvm_arg_members;
    std::vector<size_t> llvm_arg_member_indices;
    auto current_offset = static_cast<size_t>(0u);
    static constexpr auto alignment = KernelArgumentStruct::argument_alignment;
    for (auto arg : func->arguments()) {
        auto next_offset = luisa::align(current_offset, alignment);
        if (next_offset > current_offset) {
            auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
            llvm_arg_members.emplace_back(llvm::ArrayType::get(llvm_i8_type, next_offset - current_offset));
        }
        llvm_arg_member_indices.emplace_back(llvm_arg_members.size());
        auto llvm_arg_type = _get_llvm_type(arg->type());
        llvm_arg_members.emplace_back(llvm_arg_type->mem_type);
        current_offset = next_offset + _data_layout->getTypeAllocSize(llvm_arg_members.back()).getFixedValue();
    }
    // tail padding and <i32 x 4> for dispatch_size and kernel_id
    if (current_offset % alignment != 0u) {
        auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
        auto count = luisa::align(current_offset, alignment) - current_offset;
        llvm_arg_members.emplace_back(llvm::ArrayType::get(llvm_i8_type, count));
    }
    auto dispatch_size_and_kernel_id_index = llvm_arg_members.size();
    auto llvm_i32x4_type = llvm::ArrayType::get(llvm::Type::getInt32Ty(_llvm_context), 4);
    llvm_arg_members.emplace_back(llvm_i32x4_type);
    auto llvm_arg_struct_type = llvm::StructType::create(_llvm_context, llvm_arg_members, "kernel.params.struct");
    auto kernel_arg_struct = luisa::make_unique<KernelArgumentStruct>(KernelArgumentStruct{
        .llvm_type = llvm_arg_struct_type,
        .argument_indices = std::move(llvm_arg_member_indices),
        .dispatch_size_and_kernel_id_index = dispatch_size_and_kernel_id_index,
    });
    auto [iter, success] = _kernel_arg_struct_types.try_emplace(func, std::move(kernel_arg_struct));
    LUISA_ASSERT(success, "Failed to insert kernel argument struct.");
    return iter->second.get();
}

llvm::Type *CUDACodegenLLVMImpl::_get_llvm_buffer_type() noexcept {
    if (_llvm_buffer_type == nullptr) {
        auto llvm_ptr_type = llvm::PointerType::get(_llvm_context, nvptx_address_space_global);
        auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
        _llvm_buffer_type = llvm::StructType::get(_llvm_context, {llvm_ptr_type, llvm_i64_type}, false);
        detail::luisa_check_llvm_type_size_and_alignment(
            *_data_layout, _llvm_buffer_type,
            sizeof(CUDABuffer::Binding),
            alignof(CUDABuffer::Binding));
    }
    return _llvm_buffer_type;
}

llvm::Type *CUDACodegenLLVMImpl::_get_llvm_texture_type() noexcept {
    if (_llvm_texture_type == nullptr) {
        auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
        _llvm_texture_type = llvm::StructType::get(_llvm_context, {llvm_i64_type, llvm_i64_type}, false);
        detail::luisa_check_llvm_type_size_and_alignment(
            *_data_layout, _llvm_texture_type,
            sizeof(CUDATexture::Binding),
            alignof(CUDATexture::Binding));
    }
    return _llvm_texture_type;
}

llvm::Type *CUDACodegenLLVMImpl::_get_llvm_bindless_array_type() noexcept {
    if (_llvm_bindless_array_type == nullptr) {
        auto llvm_ptr_type = llvm::PointerType::get(_llvm_context, nvptx_address_space_global);
        auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
        _llvm_bindless_array_type = llvm::StructType::get(_llvm_context, {llvm_ptr_type, llvm_i64_type}, false);
        detail::luisa_check_llvm_type_size_and_alignment(
            *_data_layout, _llvm_bindless_array_type,
            sizeof(CUDABindlessArray::Binding),
            alignof(CUDABindlessArray::Binding));
    }
    return _llvm_bindless_array_type;
}

llvm::Type *CUDACodegenLLVMImpl::_get_llvm_bindless_array_slot_type() noexcept {
    if (_llvm_bindless_array_slot_type == nullptr) {
        auto llvm_ptr_type = llvm::PointerType::get(_llvm_context, nvptx_address_space_global);
        auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
        _llvm_bindless_array_slot_type = llvm::StructType::get(
            _llvm_context,
            {
                llvm_ptr_type,// buffer
                llvm_i64_type,// size
                llvm_i64_type,// tex2d
                llvm_i64_type // tex3d
            },
            false);
        detail::luisa_check_llvm_type_size_and_alignment(
            *_data_layout, _llvm_bindless_array_slot_type,
            sizeof(CUDABindlessArray::Slot),
            alignof(CUDABindlessArray::Slot));
    }
    return _llvm_bindless_array_slot_type;
}

llvm::Type *CUDACodegenLLVMImpl::_get_llvm_accel_type() noexcept {
    if (_llvm_accel_type == nullptr) {
        auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
        auto llvm_ptr_type = llvm::PointerType::get(_llvm_context, nvptx_address_space_global);
        _llvm_accel_type = llvm::StructType::get(_llvm_context, {llvm_i64_type, llvm_ptr_type}, false);
        detail::luisa_check_llvm_type_size_and_alignment(
            *_data_layout, _llvm_accel_type,
            sizeof(CUDAAccel::Binding),
            alignof(CUDAAccel::Binding));
    }
    return _llvm_accel_type;
}

llvm::Type *CUDACodegenLLVMImpl::_get_llvm_accel_instance_type() noexcept {
    if (_llvm_accel_instance_type == nullptr) {
        auto llvm_f32x4_type = llvm::VectorType::get(llvm::Type::getFloatTy(_llvm_context), 4, false);
        auto llvm_affine_type = llvm::ArrayType::get(llvm_f32x4_type, 3);
        auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
        auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
        auto llvm_padding_type = llvm::ArrayType::get(llvm_i32_type, 2);
        _llvm_accel_instance_type = llvm::StructType::get(
            _llvm_context,
            {
                llvm_affine_type,// affine
                llvm_i32_type,   // user_id
                llvm_i32_type,   // sbt_offset
                llvm_i32_type,   // mask
                llvm_i32_type,   // flags
                llvm_i64_type,   // handle
                llvm_padding_type// padding
            },
            false);
        struct alignas(16) OptiXInstance : optix::Instance {};
        static_assert(sizeof(OptiXInstance) == sizeof(optix::Instance));
        detail::luisa_check_llvm_type_size_and_alignment(
            *_data_layout, _llvm_accel_instance_type,
            sizeof(OptiXInstance), alignof(OptiXInstance));
    }
    return _llvm_accel_instance_type;
}

llvm::Type *CUDACodegenLLVMImpl::_get_llvm_ray_type() noexcept {
    if (_llvm_ray_type == nullptr) {
        auto llvm_f32_type = llvm::Type::getFloatTy(_llvm_context);
        auto llvm_f32x3_type = llvm::ArrayType::get(llvm_f32_type, 3);
        _llvm_ray_type = llvm::StructType::get(llvm_f32x3_type /* origin */,
                                               llvm_f32_type /* t_min */,
                                               llvm_f32x3_type /* direction */,
                                               llvm_f32_type /* t_max */);
    }
    return _llvm_ray_type;
}

llvm::Type *CUDACodegenLLVMImpl::_get_llvm_surface_hit_type() noexcept {
    if (_llvm_surface_hit_type == nullptr) {
        auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
        auto llvm_f32_type = llvm::Type::getFloatTy(_llvm_context);
        auto llvm_f32x2_type = llvm::VectorType::get(llvm_f32_type, 2, false);
        _llvm_surface_hit_type = llvm::StructType::get(llvm_i32_type /* inst_id */,
                                                       llvm_i32_type /* prim_id */,
                                                       llvm_f32x2_type /* bary */,
                                                       llvm_f32_type /* t */);
    }
    return _llvm_surface_hit_type;
}

llvm::Type *CUDACodegenLLVMImpl::_get_llvm_procedural_hit_type() noexcept {
    if (_llvm_procedural_hit_type == nullptr) {
        auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
        _llvm_procedural_hit_type = llvm::StructType::get(llvm_i32_type /* inst_id */,
                                                          llvm_i32_type /* prim_id */);
    }
    return _llvm_procedural_hit_type;
}

llvm::Type *CUDACodegenLLVMImpl::_get_llvm_committed_hit_type() noexcept {
    if (_llvm_committed_hit_type == nullptr) {
        auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
        auto llvm_f32_type = llvm::Type::getFloatTy(_llvm_context);
        auto llvm_f32x2_type = llvm::VectorType::get(llvm_f32_type, 2, false);
        _llvm_committed_hit_type = llvm::StructType::get(llvm_i32_type /* inst_id */,
                                                         llvm_i32_type /* prim_id */,
                                                         llvm_f32x2_type /* bary */,
                                                         llvm_i32_type /* hit_kind */,
                                                         llvm_f32_type /* t */);
    }
    return _llvm_committed_hit_type;
}

llvm::Type *CUDACodegenLLVMImpl::_get_llvm_intersection_result_type() noexcept {
    if (_llvm_intersection_result_type == nullptr) {
        auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
        _llvm_intersection_result_type = llvm::StructType::get(llvm_i8_type /* committed */,
                                                               llvm_i8_type /* terminated */);
    }
    return _llvm_intersection_result_type;
}

llvm::Type *CUDACodegenLLVMImpl::_get_llvm_ray_query_type() noexcept {
    return llvm::Type::getInt8Ty(_llvm_context);
}

std::pair<llvm::Value *, const Type *>
CUDACodegenLLVMImpl::_lower_access_chain_address(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_ptr,
                                                 const Type *type, luisa::span<const xir::Use *const> index_uses) noexcept {
    // FIXME: we directly calculate the address here, which might hinder LLVM optimizations
    LUISA_DEBUG_ASSERT(llvm_ptr->getType()->isPointerTy());
    for (auto index_use : index_uses) {
        auto llvm_index = _get_llvm_value(b, func_ctx, index_use->value());
        switch (type->tag()) {
            case Type::Tag::VECTOR: {
                type = type->element();
                auto llvm_elem_type = _get_llvm_type(type)->mem_type;
                llvm_ptr = b.CreateInBoundsGEP(llvm_elem_type, llvm_ptr, {llvm_index});
                break;
            }
            case Type::Tag::MATRIX: {
                type = Type::vector(type->element(), type->dimension());
                auto llvm_col_type = _get_llvm_type(type)->mem_type;
                llvm_ptr = b.CreateInBoundsGEP(llvm_col_type, llvm_ptr, {llvm_index});
                break;
            }
            case Type::Tag::ARRAY: {
                type = type->element();
                auto llvm_elem_type = _get_llvm_type(type)->mem_type;
                llvm_ptr = b.CreateInBoundsGEP(llvm_elem_type, llvm_ptr, {llvm_index});
                break;
            }
            case Type::Tag::STRUCTURE: {
                LUISA_DEBUG_ASSERT(llvm::isa<llvm::ConstantInt>(llvm_index));
                auto member_index = llvm::cast<llvm::ConstantInt>(llvm_index)->getZExtValue();
                LUISA_DEBUG_ASSERT(member_index < type->members().size());
                auto llvm_struct_info = _get_llvm_type(type);
                auto llvm_member_offset = llvm_struct_info->member_offsets[member_index];
                type = type->members()[member_index];
                auto llvm_i8_type = b.getInt8Ty();
                llvm_ptr = b.CreateConstInBoundsGEP1_64(llvm_i8_type, llvm_ptr, llvm_member_offset);
                break;
            }
            default: LUISA_ERROR("Invalid GEP base type: {}.", type->description());
        }
    }
    return std::make_pair(llvm_ptr, type);
}

}// namespace luisa::compute::cuda
