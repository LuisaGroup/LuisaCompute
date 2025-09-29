//
// Created by mike on 9/25/25.
//

#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

llvm::Value *CUDACodegenLLVMImpl::_get_llvm_literal(IB &b, const Type *type, const void *data) noexcept {
    LUISA_DEBUG_ASSERT(type != nullptr);
    if (data == nullptr) [[unlikely]] {
        auto llvm_type = _get_llvm_type(type)->reg_type;
        return llvm::Constant::getNullValue(llvm_type);
    }
    auto decode_scalar = [](auto p, auto x) noexcept {
        std::memcpy(&x, p, sizeof(x));
        return x;
    };
    switch (type->tag()) {
        case Type::Tag::BOOL: return b.getInt1(decode_scalar(data, bool{}));
        case Type::Tag::INT8: [[fallthrough]];
        case Type::Tag::UINT8: return b.getInt8(decode_scalar(data, uint8_t{}));
        case Type::Tag::INT16: [[fallthrough]];
        case Type::Tag::UINT16: return b.getInt16(decode_scalar(data, uint16_t{}));
        case Type::Tag::INT32: [[fallthrough]];
        case Type::Tag::UINT32: return b.getInt32(decode_scalar(data, uint32_t{}));
        case Type::Tag::INT64: [[fallthrough]];
        case Type::Tag::UINT64: return b.getInt64(decode_scalar(data, uint64_t{}));
        case Type::Tag::FLOAT16: return llvm::ConstantFP::get(b.getHalfTy(), decode_scalar(data, luisa::half{}));
        case Type::Tag::FLOAT32: return llvm::ConstantFP::get(b.getFloatTy(), decode_scalar(data, float{}));
        case Type::Tag::FLOAT64: return llvm::ConstantFP::get(b.getDoubleTy(), decode_scalar(data, double{}));
        case Type::Tag::VECTOR: {
            auto elem_type = type->element();
            // for i8/i16/i32/i64/f16/f32/f64, we can use ConstantDataVector for fast creation
            if (elem_type->is_scalar() && !elem_type->is_bool()) {
                auto llvm_elem_type = _get_llvm_type(elem_type);
                LUISA_DEBUG_ASSERT(llvm_elem_type->reg_type == llvm_elem_type->mem_type);
                llvm::StringRef const_data{static_cast<const char *>(data), elem_type->size() * type->dimension()};
                return llvm::ConstantDataVector::getRaw(const_data, type->dimension(), llvm_elem_type->reg_type);
            }
            // otherwise, create element by element
            auto elem_stride = elem_type->size();
            auto dim = type->dimension();
            llvm::SmallVector<llvm::Constant *, 4> llvm_elems;
            for (auto i = 0u; i < dim; i++) {
                auto p = static_cast<const std::byte *>(data) + i * elem_stride;
                auto llvm_elem = _get_llvm_literal(b, elem_type, p);
                LUISA_DEBUG_ASSERT(llvm::isa<llvm::Constant>(llvm_elem));
                llvm_elems.emplace_back(llvm::cast<llvm::Constant>(llvm_elem));
            }
            llvm::ConstantVector::get(llvm_elems);
        }
        case Type::Tag::MATRIX: {
            auto dim = type->dimension();
            LUISA_DEBUG_ASSERT(dim > 0u);
            auto elem_type = type->element();
            LUISA_DEBUG_ASSERT(elem_type->is_float16() || elem_type->is_float32() || elem_type->is_float64());
            auto col_type = Type::vector(elem_type, dim);
            auto col_stride = col_type->size();
            llvm::SmallVector<llvm::Constant *, 4> llvm_cols;
            for (auto col = 0; col < dim; col++) {
                auto p_col = static_cast<const char *>(data) + col * col_stride;
                auto llvm_col = _get_llvm_literal(b, col_type, p_col);
                LUISA_DEBUG_ASSERT(llvm::isa<llvm::Constant>(llvm_col));
                llvm_cols.emplace_back(llvm::cast<llvm::Constant>(llvm_col));
            }
            auto llvm_col_type = llvm_cols.front()->getType();
            auto llvm_matrix_type = llvm::ArrayType::get(llvm_col_type, dim);
            return llvm::ConstantArray::get(llvm_matrix_type, llvm_cols);
        }
        case Type::Tag::ARRAY: {
            auto elem_type = type->element();
            // for i8/i16/i32/i64/f16/f32/f64, we can use ConstantDataArray for fast creation
            if (elem_type->is_scalar() && !elem_type->is_bool()) {
                auto llvm_elem_type = _get_llvm_type(elem_type);
                LUISA_DEBUG_ASSERT(llvm_elem_type->reg_type == llvm_elem_type->mem_type);
                llvm::StringRef const_data{static_cast<const char *>(data), type->size()};
                return llvm::ConstantDataArray::getRaw(const_data, type->dimension(), llvm_elem_type->reg_type);
            }
            // otherwise, create element by element
            auto elem_stride = elem_type->size();
            auto dim = type->dimension();
            LUISA_DEBUG_ASSERT(dim > 0u);
            llvm::SmallVector<llvm::Constant *> llvm_elems;
            llvm_elems.reserve(dim);
            for (auto i = 0u; i < dim; i++) {
                auto p = static_cast<const std::byte *>(data) + i * elem_stride;
                auto llvm_elem = _get_llvm_literal(b, elem_type, p);
                LUISA_DEBUG_ASSERT(llvm::isa<llvm::Constant>(llvm_elem));
                llvm_elems.emplace_back(llvm::cast<llvm::Constant>(llvm_elem));
            }
            auto llvm_elem_type = llvm_elems.front()->getType();
            auto llvm_array_type = llvm::ArrayType::get(llvm_elem_type, dim);
            return llvm::ConstantArray::get(llvm_array_type, llvm_elems);
        }
        case Type::Tag::STRUCTURE: {
            auto llvm_type_info = _get_llvm_type(type);
            llvm::SmallVector<llvm::Constant *> llvm_elems;
            LUISA_DEBUG_ASSERT(llvm::isa<llvm::StructType>(llvm_type_info->reg_type));
            auto llvm_struct_type = llvm::cast<llvm::StructType>(llvm_type_info->reg_type);
            llvm_elems.reserve(llvm_struct_type->getNumElements());
            auto output_index = 0u;
            for (auto i = 0u; i < llvm_type_info->member_indices.size(); i++) {
                auto member_index = llvm_type_info->member_indices[i];
                while (output_index < member_index) {
                    // padding
                    auto llvm_padding_type = llvm_struct_type->getElementType(output_index);
                    auto llvm_padding = llvm::PoisonValue::get(llvm_padding_type);
                    llvm_elems.emplace_back(llvm_padding);
                    output_index++;
                }
                auto member_type = type->members()[i];
                auto member_offset = llvm_type_info->member_offsets[i];
                auto p_member = static_cast<const std::byte *>(data) + member_offset;
                auto llvm_member = _get_llvm_literal(b, member_type, p_member);
                LUISA_DEBUG_ASSERT(llvm::isa<llvm::Constant>(llvm_member));
                llvm_elems.emplace_back(llvm::cast<llvm::Constant>(llvm_member));
                output_index++;
            }
            if (output_index < llvm_struct_type->getNumElements()) {
                // tail padding
                LUISA_DEBUG_ASSERT(output_index + 1 == llvm_struct_type->getNumElements());
                auto llvm_padding_type = llvm_struct_type->getElementType(output_index);
                auto llvm_padding = llvm::PoisonValue::get(llvm_padding_type);
                llvm_elems.emplace_back(llvm_padding);
            }
            return llvm::ConstantStruct::get(llvm_struct_type, llvm_elems);
        }
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Cannot create LLVM literal for type: {}.", type->description());
}

llvm::Value *CUDACodegenLLVMImpl::_get_llvm_constant(IB &b, const xir::Constant *c) noexcept {
    auto type = c->type();
    if (type->is_basic()) { return _get_llvm_literal(b, type, c->data()); }
    // find global constant
    llvm::Align alignment{type->alignment()};
    auto iter = _xir_to_llvm_global.try_emplace(c, nullptr).first;
    if (iter->second == nullptr) {
        // create global constant
        llvm::ArrayRef const_data{static_cast<const uint8_t *>(c->data()), type->size()};
        auto llvm_init = llvm::ConstantDataArray::get(_llvm_context, const_data);
        auto global = new llvm::GlobalVariable(
            *_llvm_module, b.getInt8Ty(), true, llvm::GlobalValue::PrivateLinkage,
            llvm_init, "const", nullptr, llvm::GlobalVariable::NotThreadLocal,
            nvptx_address_space_constant, false);
        global->setAlignment(alignment);
        global->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
        iter->second = global;
    }
    auto llvm_type = _get_llvm_type(type);
    LUISA_DEBUG_ASSERT(llvm_type->reg_type == llvm_type->mem_type,
                       "Global constant must have same reg and mem type.");
    // don't worry about the huge load, we will delay it in extract(constant)
    return b.CreateAlignedLoad(llvm_type->reg_type, iter->second, alignment, false);
}

}// namespace luisa::compute::cuda
