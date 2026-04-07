//
// Created by mike on 3/18/26.
//

#include <luisa/dsl/rtx/ray_query.h>

#include "hip_codegen_llvm_impl.h"
#include "../hip_bindless_array.h"
#include "../hip_buffer.h"

namespace luisa::compute::hip {

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

size_t HIPCodegenLLVMImpl::_get_type_alignment(const Type *type) noexcept {
    if (type->is_basic() || type->is_array() || type->is_structure()) {
        return type->alignment();
    }
    if (type->is_resource() || type->is_custom() || type->is_cooperative_vector_ref() || type->is_cooperative_matrix_ref()) {
        return 16;
    }
    return 16;
}

const HIPCodegenLLVMImpl::LLVMTypeInfo *HIPCodegenLLVMImpl::_get_llvm_type(const Type *type) noexcept {
    if (auto iter = _xir_to_llvm_type.find(type); iter != _xir_to_llvm_type.end()) {
        return iter->second.get();
    }
    auto llvm_type_info = [this, type]() noexcept -> luisa::unique_ptr<LLVMTypeInfo> {
        auto make_info = [this](llvm::Type *mem_t, llvm::Type *reg_t, size_t s, size_t a,
                                luisa::vector<size_t> member_indices = {},
                                luisa::vector<size_t> member_offsets = {}) noexcept {
            auto info = luisa::make_unique<LLVMTypeInfo>();
            info->mem_type = mem_t;
            info->reg_type = reg_t;
            info->member_indices = std::move(member_indices);
            info->member_offsets = std::move(member_offsets);
            return info;
        };
        switch (type->tag()) {
            case Type::Tag::BOOL: {
                auto t = llvm::Type::getInt1Ty(_llvm_context);
                return make_info(t, t, sizeof(bool), alignof(bool));
            }
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::UINT8: {
                auto t = llvm::Type::getInt8Ty(_llvm_context);
                return make_info(t, t, sizeof(int8_t), alignof(int8_t));
            }
            case Type::Tag::INT16: [[fallthrough]];
            case Type::Tag::UINT16: {
                auto t = llvm::Type::getInt16Ty(_llvm_context);
                return make_info(t, t, sizeof(int16_t), alignof(int16_t));
            }
            case Type::Tag::INT32: [[fallthrough]];
            case Type::Tag::UINT32: {
                auto t = llvm::Type::getInt32Ty(_llvm_context);
                return make_info(t, t, sizeof(int32_t), alignof(int32_t));
            }
            case Type::Tag::INT64: [[fallthrough]];
            case Type::Tag::UINT64: {
                auto t = llvm::Type::getInt64Ty(_llvm_context);
                return make_info(t, t, sizeof(int64_t), alignof(int64_t));
            }
            case Type::Tag::FLOAT16: {
                auto t = llvm::Type::getHalfTy(_llvm_context);
                return make_info(t, t, sizeof(luisa::half), alignof(luisa::half));
            }
            case Type::Tag::FLOAT32: {
                auto t = llvm::Type::getFloatTy(_llvm_context);
                return make_info(t, t, sizeof(float), alignof(float));
            }
            case Type::Tag::FLOAT64: {
                auto t = llvm::Type::getDoubleTy(_llvm_context);
                return make_info(t, t, sizeof(double), alignof(double));
            }
            case Type::Tag::VECTOR: {
                auto elem = _get_llvm_type(type->element());
                auto dim = type->dimension();
                auto llvm_reg_type = llvm::FixedVectorType::get(elem->reg_type, dim);
                auto llvm_mem_type = llvm::ArrayType::get(elem->mem_type, luisa::align(dim, 2));
                return make_info(llvm_mem_type, llvm_reg_type, type->size(), type->alignment());
            }
            case Type::Tag::MATRIX: {
                auto dim = type->dimension();
                auto col_type = Type::vector(type->element(), dim);
                auto col_llvm = _get_llvm_type(col_type);
                auto llvm_reg_type = llvm::ArrayType::get(col_llvm->reg_type, dim);
                auto llvm_mem_type = llvm::ArrayType::get(col_llvm->mem_type, dim);
                return make_info(llvm_mem_type, llvm_reg_type, type->size(), type->alignment());
            }
            case Type::Tag::ARRAY: {
                auto elem = _get_llvm_type(type->element());
                auto llvm_reg_type = llvm::ArrayType::get(elem->reg_type, type->dimension());
                auto llvm_mem_type = llvm::ArrayType::get(elem->mem_type, type->dimension());
                return make_info(llvm_mem_type, llvm_reg_type, type->size(), type->alignment());
            }
            case Type::Tag::STRUCTURE: {
                luisa::vector<size_t> member_indices;
                luisa::vector<size_t> member_offsets;
                llvm::SmallVector<llvm::Type *> llvm_member_reg_types;
                llvm::SmallVector<llvm::Type *> llvm_member_mem_types;
                auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
                auto current_offset = static_cast<size_t>(0u);
                for (auto member : type->members()) {
                    auto llvm_member_type = _get_llvm_type(member);
                    llvm_member_reg_types.emplace_back(llvm_member_type->reg_type);
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
                return make_info(llvm_mem_type, llvm_reg_type, type->size(), type->alignment(),
                                 std::move(member_indices), std::move(member_offsets));
            }
            case Type::Tag::BUFFER: {
                auto llvm_type = _get_llvm_buffer_type();
                return make_info(llvm_type, llvm_type, 16u, 16u);
            }
            case Type::Tag::TEXTURE: {
                auto llvm_type = _get_llvm_texture_type();
                return make_info(llvm_type, llvm_type, 16u, 16u);
            }
            case Type::Tag::BINDLESS_ARRAY: {
                auto llvm_type = _get_llvm_bindless_array_type();
                return make_info(llvm_type, llvm_type, 16u, 16u);
            }
            case Type::Tag::ACCEL: {
                auto llvm_type = _get_llvm_accel_type();
                return make_info(llvm_type, llvm_type, 16u, 16u);
            }
            case Type::Tag::CUSTOM: {
                if (type == Type::of<RayQueryAll>() || type == Type::of<RayQueryAny>()) {
                    auto llvm_type = _get_llvm_ray_query_type();
                    return make_info(llvm_type, llvm_type, sizeof(uint8_t), alignof(uint8_t));
                }
                LUISA_NOT_IMPLEMENTED("Custom type: {}.", type->description());
            }
            default: LUISA_ERROR_WITH_LOCATION("Unsupported type with tag {}.", static_cast<uint32_t>(type->tag()));
        }
    }();
    auto [iter, _] = _xir_to_llvm_type.try_emplace(type, std::move(llvm_type_info));
    return iter->second.get();
}

const HIPCodegenLLVMImpl::KernelArgumentStruct *HIPCodegenLLVMImpl::_get_kernel_argument_struct(const xir::KernelFunction *func) noexcept {
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
    if (current_offset % alignment != 0u) {
        auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
        auto count = luisa::align(current_offset, alignment) - current_offset;
        llvm_arg_members.emplace_back(llvm::ArrayType::get(llvm_i8_type, count));
    }
    auto dispatch_size_and_kernel_id_index = llvm_arg_members.size();
    auto llvm_i32x4_type = llvm::ArrayType::get(llvm::Type::getInt32Ty(_llvm_context), 4);
    llvm_arg_members.emplace_back(llvm_i32x4_type);
    auto rt_global_stack_buffer_index = static_cast<size_t>(0u);
    // TEMPORARILY DISABLED for firefly debugging: do NOT add RT stack fields to the kernel arg struct.
    // The RT stack data will be passed as constants instead.
    auto has_rt_global_stack_buffer = _rt_analysis.uses_ray_tracing;
    if (has_rt_global_stack_buffer) {
        rt_global_stack_buffer_index = llvm_arg_members.size();
        auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
        auto generic_ptr = llvm::PointerType::get(_llvm_context, 0);
        llvm_arg_members.emplace_back(llvm_i32_type);// stack_size
        llvm_arg_members.emplace_back(llvm_i32_type);// stack_count
        llvm_arg_members.emplace_back(generic_ptr);  // stack_data
    }
    auto llvm_arg_struct_type = llvm::StructType::create(_llvm_context, llvm_arg_members, "kernel.params.struct");
    auto kernel_arg_struct = luisa::make_unique<KernelArgumentStruct>(KernelArgumentStruct{
        .llvm_type = llvm_arg_struct_type,
        .argument_indices = std::move(llvm_arg_member_indices),
        .dispatch_size_and_kernel_id_index = dispatch_size_and_kernel_id_index,
        .rt_global_stack_buffer_index = rt_global_stack_buffer_index,
        .has_rt_global_stack_buffer = has_rt_global_stack_buffer,
    });
    auto [iter, success] = _kernel_arg_struct_types.try_emplace(func, std::move(kernel_arg_struct));
    LUISA_ASSERT(success, "Failed to insert kernel argument struct.");
    return iter->second.get();
}

llvm::Type *HIPCodegenLLVMImpl::_get_llvm_buffer_type() noexcept {
    if (_llvm_buffer_type == nullptr) {
        auto ptr_type = llvm::PointerType::get(_llvm_context, amdgpu_address_space_global);
        auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
        _llvm_buffer_type = llvm::StructType::get(_llvm_context, {ptr_type, llvm_i64_type}, false);
        detail::luisa_check_llvm_type_size_and_alignment(
            *_data_layout, _llvm_buffer_type,
            sizeof(HIPBuffer::Binding),
            alignof(HIPBuffer::Binding));
    }
    return _llvm_buffer_type;
}

llvm::Type *HIPCodegenLLVMImpl::_get_llvm_texture_type() noexcept {
    if (_llvm_texture_type == nullptr) {
        auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
        _llvm_texture_type = llvm::StructType::get(_llvm_context, {llvm_i64_type, llvm_i64_type}, false);
    }
    return _llvm_texture_type;
}

llvm::Type *HIPCodegenLLVMImpl::_get_llvm_bindless_array_type() noexcept {
    if (_llvm_bindless_array_type == nullptr) {
        auto ptr_type = llvm::PointerType::get(_llvm_context, amdgpu_address_space_global);
        auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
        _llvm_bindless_array_type = llvm::StructType::get(_llvm_context, {ptr_type, llvm_i64_type}, false);
    }
    return _llvm_bindless_array_type;
}

llvm::Type *HIPCodegenLLVMImpl::_get_llvm_bindless_array_slot_type() noexcept {
    if (_llvm_bindless_array_slot_type == nullptr) {
        auto llvm_ptr_type = llvm::PointerType::get(_llvm_context, amdgpu_address_space_global);
        auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
        _llvm_bindless_array_slot_type = llvm::StructType::create(
            _llvm_context,
            {
                llvm_ptr_type,// buffer
                llvm_i64_type,// size
                llvm_i64_type,// tex2d
                llvm_i64_type // tex3d
            },
            "luisa.bindless.slot");
        detail::luisa_check_llvm_type_size_and_alignment(
            _llvm_module->getDataLayout(), _llvm_bindless_array_slot_type,
            sizeof(HIPBindlessArray::Slot),
            alignof(HIPBindlessArray::Slot));
    }
    return _llvm_bindless_array_slot_type;
}

llvm::Type *HIPCodegenLLVMImpl::_get_llvm_accel_type() noexcept {
    if (_llvm_accel_type == nullptr) {
        auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
        auto ptr_type = llvm::PointerType::get(_llvm_context, amdgpu_address_space_global);
        _llvm_accel_type = llvm::StructType::get(_llvm_context, {llvm_i64_type, ptr_type}, false);
    }
    return _llvm_accel_type;
}

llvm::Type *HIPCodegenLLVMImpl::_get_llvm_accel_instance_type() noexcept {
    if (_llvm_accel_instance_type == nullptr) {
        auto float4x3_type = llvm::ArrayType::get(llvm::FixedVectorType::get(llvm::Type::getFloatTy(_llvm_context), 4), 3);
        _llvm_accel_instance_type = llvm::StructType::get(_llvm_context,
                                                          {float4x3_type,
                                                           llvm::Type::getInt32Ty(_llvm_context),
                                                           llvm::Type::getInt32Ty(_llvm_context),
                                                           llvm::Type::getInt32Ty(_llvm_context),
                                                           llvm::Type::getInt32Ty(_llvm_context),
                                                           llvm::Type::getInt64Ty(_llvm_context)},
                                                          false);
    }
    return _llvm_accel_instance_type;
}

llvm::Type *HIPCodegenLLVMImpl::_get_llvm_ray_type() noexcept {
    if (_llvm_ray_type == nullptr) {
        _llvm_ray_type = llvm::StructType::get(_llvm_context,
                                               {llvm::ArrayType::get(llvm::Type::getFloatTy(_llvm_context), 3),
                                                llvm::Type::getFloatTy(_llvm_context),
                                                llvm::ArrayType::get(llvm::Type::getFloatTy(_llvm_context), 3),
                                                llvm::Type::getFloatTy(_llvm_context)},
                                               false);
    }
    return _llvm_ray_type;
}

llvm::Type *HIPCodegenLLVMImpl::_get_llvm_surface_hit_type() noexcept {
    if (_llvm_surface_hit_type == nullptr) {
        auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
        auto llvm_f32_type = llvm::Type::getFloatTy(_llvm_context);
        auto llvm_f32x2_type = llvm::FixedVectorType::get(llvm_f32_type, 2);
        _llvm_surface_hit_type = llvm::StructType::get(llvm_i32_type,
                                                       llvm_i32_type,
                                                       llvm_f32x2_type,
                                                       llvm_f32_type);
    }
    return _llvm_surface_hit_type;
}

llvm::Type *HIPCodegenLLVMImpl::_get_llvm_procedural_hit_type() noexcept {
    if (_llvm_procedural_hit_type == nullptr) {
        auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
        _llvm_procedural_hit_type = llvm::StructType::get(llvm_i32_type,
                                                          llvm_i32_type);
    }
    return _llvm_procedural_hit_type;
}

llvm::Type *HIPCodegenLLVMImpl::_get_llvm_committed_hit_type() noexcept {
    if (_llvm_committed_hit_type == nullptr) {
        auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
        auto llvm_f32_type = llvm::Type::getFloatTy(_llvm_context);
        auto llvm_f32x2_type = llvm::FixedVectorType::get(llvm_f32_type, 2);
        _llvm_committed_hit_type = llvm::StructType::get(llvm_i32_type,
                                                         llvm_i32_type,
                                                         llvm_f32x2_type,
                                                         llvm_i32_type,
                                                         llvm_f32_type);
    }
    return _llvm_committed_hit_type;
}

llvm::Type *HIPCodegenLLVMImpl::_get_llvm_ray_query_type() noexcept {
    if (_llvm_ray_query_type == nullptr) {
        auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
        _llvm_ray_query_type = llvm::StructType::get(llvm_i64_type,
                                                     llvm_i64_type,
                                                     llvm_i64_type,
                                                     llvm_i64_type,
                                                     llvm_i64_type);
    }
    return _llvm_ray_query_type;
}

std::pair<llvm::Value *, const Type *> HIPCodegenLLVMImpl::_lower_access_chain_address(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_ptr, const Type *type, luisa::span<const xir::Use *const> index_uses) noexcept {
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

}// namespace luisa::compute::hip
