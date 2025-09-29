//
// Created by mike on 9/21/25.
//

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
    LUISA_ASSERT(llvm_size.getFixedValue() == expected_size &&
                     llvm_align.value() <= expected_alignment,
                 "Type '{}' size or alignment mismatch: "
                 "expected size {}, alignment {}; "
                 "actual size {}, alignment {}.",
                 type_str(), expected_size, expected_alignment,
                 llvm_size.getFixedValue(), llvm_align.value());
}
}// namespace detail

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
                auto llvm_reg_type = llvm::VectorType::get(elem->reg_type, dim, false);
                auto llvm_mem_type = llvm::ArrayType::get(elem->mem_type, dim == 3 ? 4 : dim);
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
                auto llvm_elem_type = _get_llvm_type(type->element())->mem_type;
                auto llvm_type = llvm::ArrayType::get(llvm_elem_type, type->dimension());
                return make_llvm_type_info(llvm_type, llvm_type, type->size(), type->alignment());
            }
            case Type::Tag::STRUCTURE: {
                luisa::vector<size_t> member_indices;
                luisa::vector<size_t> member_offsets;
                auto member_count = type->members().size();
                member_indices.reserve(member_count);
                member_offsets.reserve(member_count);
                llvm::SmallVector<llvm::Type *> llvm_member_types;
                auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
                auto current_offset = static_cast<size_t>(0u);
                for (auto member : type->members()) {
                    auto next_offset = luisa::align(current_offset, member->alignment());
                    if (next_offset > current_offset) {
                        llvm_member_types.emplace_back(
                            llvm::ArrayType::get(llvm_i8_type, next_offset - current_offset));
                    }
                    member_indices.emplace_back(llvm_member_types.size());
                    member_offsets.emplace_back(next_offset);
                    llvm_member_types.emplace_back(_get_llvm_type(member)->mem_type);
                    current_offset = next_offset + member->size();
                }
                if (current_offset < type->size()) {
                    llvm_member_types.emplace_back(
                        llvm::ArrayType::get(llvm_i8_type, type->size() - current_offset));
                }
                auto llvm_type = llvm::StructType::get(_llvm_context, llvm_member_types, false);
                return make_llvm_type_info(llvm_type, llvm_type, type->size(), type->alignment(),
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
            case Type::Tag::CUSTOM: LUISA_NOT_IMPLEMENTED("Custom type: {}.", type->description());
        }
        LUISA_ERROR_WITH_LOCATION("Invalid type: {}.", type->description());
    }();
    auto [iter, success] = _xir_to_llvm_type.try_emplace(type, std::move(llvm_type_info));
    LUISA_ASSERT(success, "Failed to insert LLVM type.");
    return iter->second.get();
}

llvm::Type *CUDACodegenLLVMImpl::_get_llvm_buffer_type() noexcept {
    if (_llvm_buffer_type == nullptr) {
        auto llvm_ptr_type = llvm::PointerType::get(_llvm_context, nvptx_address_space_global);
        auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
        _llvm_buffer_type = llvm::StructType::get(_llvm_context, {llvm_ptr_type, llvm_i32_type, llvm_i32_type}, false);
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
        auto llvm_affine_type = llvm::ArrayType::get(llvm::Type::getFloatTy(_llvm_context), 12);
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
        detail::luisa_check_llvm_type_size_and_alignment(
            *_data_layout, _llvm_accel_instance_type,
            sizeof(optix::Instance), alignof(optix::Instance));
    }
    return _llvm_accel_instance_type;
}

}// namespace luisa::compute::cuda
