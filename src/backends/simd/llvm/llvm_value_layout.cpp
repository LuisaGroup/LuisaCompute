#include "llvm_value_layout.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>

#include <luisa/ast/type.h>

namespace luisa::compute::simd {

LLVMValueLayout::LLVMValueLayout(::llvm::LLVMContext &context,
                                 uint32_t width) noexcept
    : _context{context}, _width{width} {
    if (width == 0u || width > 128u) {
        _error = "LLVM SIMD specialization width must be in [1, 128]";
    }
}

::llvm::Type *LLVMValueLayout::_fail(const Type *type,
                                     const char *reason) noexcept {
    if (_error.empty()) {
        _error = reason;
        if (type != nullptr) {
            _error += ": ";
            _error += type->description();
        }
    }
    return nullptr;
}

::llvm::Type *LLVMValueLayout::_uniform_type(const Type *type) noexcept {
    if (!_error.empty()) { return nullptr; }
    if (type == nullptr) {
        return _fail(type, "missing Luisa type for LLVM value");
    }
    if (auto iter = _uniform_types.find(type);
        iter != _uniform_types.end()) {
        return iter->second;
    }

    ::llvm::Type *result = nullptr;
    using Tag = Type::Tag;
    switch (type->tag()) {
        case Tag::BOOL: result = ::llvm::Type::getInt1Ty(_context); break;
        case Tag::INT8:
        case Tag::UINT8: result = ::llvm::Type::getInt8Ty(_context); break;
        case Tag::INT16:
        case Tag::UINT16: result = ::llvm::Type::getInt16Ty(_context); break;
        case Tag::INT32:
        case Tag::UINT32: result = ::llvm::Type::getInt32Ty(_context); break;
        case Tag::INT64:
        case Tag::UINT64: result = ::llvm::Type::getInt64Ty(_context); break;
        case Tag::FLOAT16: result = ::llvm::Type::getHalfTy(_context); break;
        case Tag::FLOAT32: result = ::llvm::Type::getFloatTy(_context); break;
        case Tag::FLOAT64: result = ::llvm::Type::getDoubleTy(_context); break;
        case Tag::VECTOR: {
            auto *element = _uniform_type(type->element());
            if (element != nullptr && element->isSingleValueType()) {
                result = ::llvm::FixedVectorType::get(
                    element, type->dimension());
            }
            break;
        }
        case Tag::MATRIX: {
            auto *element = _uniform_type(type->element());
            if (element != nullptr && element->isSingleValueType()) {
                auto *column = ::llvm::FixedVectorType::get(
                    element, type->dimension());
                result = ::llvm::ArrayType::get(
                    column, type->dimension());
            }
            break;
        }
        case Tag::ARRAY: {
            if (auto *element = _uniform_type(type->element())) {
                result = ::llvm::ArrayType::get(
                    element, type->dimension());
            }
            break;
        }
        case Tag::STRUCTURE: {
            ::llvm::SmallVector<::llvm::Type *, 8u> members;
            members.reserve(type->members().size());
            for (auto *member : type->members()) {
                auto *llvm_member = _uniform_type(member);
                if (llvm_member == nullptr) { return nullptr; }
                members.emplace_back(llvm_member);
            }
            result = ::llvm::StructType::get(_context, members);
            break;
        }
        case Tag::BUFFER: {
            result = ::llvm::StructType::get(
                _context,
                {::llvm::PointerType::getUnqual(_context),
                 ::llvm::Type::getInt64Ty(_context)});
            break;
        }
        case Tag::FLOAT8_E4M3:
        case Tag::FLOAT8_E5M2:
        case Tag::TEXTURE:
        case Tag::BINDLESS_ARRAY:
        case Tag::ACCEL:
        case Tag::COOPERATIVE_VECTOR:
        case Tag::COOPERATIVE_VECTOR_REF:
        case Tag::COOPERATIVE_MATRIX_REF:
        case Tag::CUSTOM: break;
    }
    if (result == nullptr) {
        return _fail(type, "unsupported Phase 2 LLVM value type");
    }
    _uniform_types.emplace(type, result);
    return result;
}

::llvm::Type *LLVMValueLayout::_varying_type(const Type *type) noexcept {
    if (!_error.empty()) { return nullptr; }
    if (type == nullptr) {
        return _fail(type, "missing Luisa type for varying LLVM value");
    }
    if (auto iter = _varying_types.find(type);
        iter != _varying_types.end()) {
        return iter->second;
    }

    ::llvm::Type *result = nullptr;
    if (type->is_scalar()) {
        if (auto *element = _uniform_type(type)) {
            result = ::llvm::FixedVectorType::get(element, _width);
        }
    } else {
        using Tag = Type::Tag;
        switch (type->tag()) {
            case Tag::VECTOR:
            case Tag::MATRIX:
            case Tag::ARRAY: {
                if (auto *element = _varying_type(type->element())) {
                    auto dimension = type->dimension();
                    if (type->is_matrix()) {
                        auto *column = ::llvm::ArrayType::get(
                            element, dimension);
                        result = ::llvm::ArrayType::get(
                            column, dimension);
                    } else {
                        result = ::llvm::ArrayType::get(
                            element, dimension);
                    }
                }
                break;
            }
            case Tag::STRUCTURE: {
                ::llvm::SmallVector<::llvm::Type *, 8u> members;
                members.reserve(type->members().size());
                for (auto *member : type->members()) {
                    auto *llvm_member = _varying_type(member);
                    if (llvm_member == nullptr) { return nullptr; }
                    members.emplace_back(llvm_member);
                }
                result = ::llvm::StructType::get(_context, members);
                break;
            }
            default: break;
        }
    }
    if (result == nullptr) {
        return _fail(type, "unsupported Phase 2 varying LLVM value type");
    }
    _varying_types.emplace(type, result);
    return result;
}

::llvm::VectorType *LLVMValueLayout::mask_type() noexcept {
    if (!_error.empty()) { return nullptr; }
    return ::llvm::FixedVectorType::get(
        ::llvm::Type::getInt1Ty(_context), _width);
}

::llvm::Type *LLVMValueLayout::expression_type(
    const schedule::Value &value) noexcept {
    using Class = schedule::ValueClass;
    switch (value.value_class) {
        case Class::warp_uniform:
        case Class::cohort_uniform: return _uniform_type(value.type);
        case Class::varying: return _varying_type(value.type);
        case Class::mask: return mask_type();
        case Class::token: return ::llvm::Type::getInt32Ty(_context);
    }
    return _fail(value.type, "invalid Schedule IR value class");
}

::llvm::Type *LLVMValueLayout::state_type(
    const schedule::Value &value) noexcept {
    using Class = schedule::ValueClass;
    switch (value.value_class) {
        case Class::warp_uniform: return _uniform_type(value.type);
        case Class::cohort_uniform:
        case Class::varying: return _varying_type(value.type);
        case Class::mask: return mask_type();
        case Class::token: return ::llvm::Type::getInt32Ty(_context);
    }
    return _fail(value.type, "invalid Schedule IR value class");
}

}// namespace luisa::compute::simd
