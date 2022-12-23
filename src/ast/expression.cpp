//
// Created by Mike Smith on 2021/3/13.
//

#include <ast/variable.h>
#include <core/logging.h>
#include <ast/expression.h>
#include <ast/function_builder.h>

namespace luisa::compute {

void RefExpr::_mark(Usage usage) const noexcept {
    detail::FunctionBuilder::current()->mark_variable_usage(
        _variable.uid(), usage);
}

void CallExpr::_mark() const noexcept {
    if (is_builtin()) {
        switch (_op) {
            case CallOp::BUFFER_WRITE:
            case CallOp::TEXTURE_WRITE:
            case CallOp::SET_INSTANCE_TRANSFORM:
            case CallOp::SET_INSTANCE_VISIBILITY:
            case CallOp::SET_INSTANCE_OPAQUE:
            case CallOp::SET_AABB:
            case CallOp::ATOMIC_EXCHANGE:
            case CallOp::ATOMIC_COMPARE_EXCHANGE:
            case CallOp::ATOMIC_FETCH_ADD:
            case CallOp::ATOMIC_FETCH_SUB:
            case CallOp::ATOMIC_FETCH_AND:
            case CallOp::ATOMIC_FETCH_OR:
            case CallOp::ATOMIC_FETCH_XOR:
            case CallOp::ATOMIC_FETCH_MIN:
            case CallOp::ATOMIC_FETCH_MAX:
            case CallOp::CLEAR_DISPATCH_INDIRECT_BUFFER:
            case CallOp::EMPLACE_DISPATCH_INDIRECT_KERNEL: {
                _arguments[0]->mark(Usage::WRITE);
                for (auto i = 1u; i < _arguments.size(); i++) {
                    _arguments[i]->mark(Usage::READ);
                }
            } break;
            default:
                for (auto arg : _arguments) {
                    arg->mark(Usage::READ);
                }
                break;
        };
    } else {
        auto args = _custom.arguments();
        for (auto i = 0u; i < args.size(); i++) {
            auto arg = args[i];
            _arguments[i]->mark(
                arg.tag() == Variable::Tag::REFERENCE ||
                        arg.tag() == Variable::Tag::BUFFER ||
                        arg.tag() == Variable::Tag::ACCEL ||
                        arg.tag() == Variable::Tag::TEXTURE ?
                    _custom.variable_usage(arg.uid()) :
                    Usage::READ);
        }
    }
}

void Expression::mark(Usage usage) const noexcept {
    if (auto a = to_underlying(_usage), u = static_cast<uint8_t>(a | to_underlying(usage)); a != u) {
        _usage = static_cast<Usage>(u);
        _mark(usage);
    }
}

uint64_t Expression::hash() const noexcept {
    if (!_hash_computed) {
        using namespace std::string_view_literals;
        _hash = hash64(_tag, hash64(_compute_hash(), hash64("__hash_expression")));
        if (_type != nullptr) { _hash = hash64(_type->hash(), _hash); }
        _hash_computed = true;
    }
    return _hash;
}
MemberExpr::MemberExpr(const Type *type, const Expression *self, uint swizzle_size, uint swizzle_code) noexcept
    : Expression{Tag::MEMBER, type}, _self{self},
      _swizzle_size{swizzle_size}, _swizzle_code{swizzle_code} {
    LUISA_ASSERT(_swizzle_size != 0u && _swizzle_size <= 4u,
                 "Swizzle size must be in [1, 4]");
}


uint32_t MemberExpr::swizzle_size() const noexcept {
    LUISA_ASSERT(_swizzle_size != 0u && _swizzle_size <= 4u,
                 "Invalid swizzle size {}.", _swizzle_size);
    return _swizzle_size;
}

uint32_t MemberExpr::swizzle_code() const noexcept {
    LUISA_ASSERT(is_swizzle(), "MemberExpr is not swizzled.");
    return _swizzle_code;
}

uint32_t MemberExpr::swizzle_index(uint index) const noexcept {
    if (auto s = swizzle_size(); index >= s) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid swizzle index {} (count = {}).",
            index, s);
    }
    return (_swizzle_code >> (index * 4u)) & 0x0fu;
}

uint32_t MemberExpr::member_index() const noexcept {
    LUISA_ASSERT(!is_swizzle(), "MemberExpr is not a member");
    return _swizzle_code;
}

}// namespace luisa::compute
