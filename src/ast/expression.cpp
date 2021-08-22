//
// Created by Mike Smith on 2021/3/13.
//

#include "ast/variable.h"
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
        if (_op == CallOp::TEXTURE_WRITE
            || _op == CallOp::ATOMIC_STORE
            || _op == CallOp::ATOMIC_EXCHANGE
            || _op == CallOp::ATOMIC_COMPARE_EXCHANGE
            || _op == CallOp::ATOMIC_FETCH_ADD
            || _op == CallOp::ATOMIC_FETCH_SUB
            || _op == CallOp::ATOMIC_FETCH_AND
            || _op == CallOp::ATOMIC_FETCH_OR
            || _op == CallOp::ATOMIC_FETCH_XOR
            || _op == CallOp::ATOMIC_FETCH_MIN
            || _op == CallOp::ATOMIC_FETCH_MAX) {
            _arguments[0]->mark(Usage::WRITE);
            for (auto i = 1u; i < _arguments.size(); i++) {
                _arguments[i]->mark(Usage::READ);
            }
        } else {
            for (auto arg : _arguments) {
                arg->mark(Usage::READ);
            }
        }
    } else {
        auto args = _custom.arguments();
        for (auto i = 0u; i < args.size(); i++) {
            auto arg = args[i];
            _arguments[i]->mark(
                arg.tag() == Variable::Tag::REFERENCE
                        || arg.tag() == Variable::Tag::BUFFER
                        || arg.tag() == Variable::Tag::TEXTURE
                    ? _custom.variable_usage(arg.uid())
                    : Usage::READ);
        }
    }
}

void Expression::mark(Usage usage) const noexcept {
    if (auto a = to_underlying(_usage), b = to_underlying(usage); (a & b) == 0u) {
        _usage = static_cast<Usage>(a | b);
        _mark(usage);
    }
}

uint64_t Expression::hash() const noexcept {
    if (auto h = _hash.load(std::memory_order::consume); h != 0u) { return h; }
    auto h = hash64(_tag, hash64(_type->hash(), _compute_hash()));
    _hash.store(h, std::memory_order::release);
    return h;
}

}// namespace luisa::compute
