//
// Created by Mike Smith on 2021/3/13.
//

#include <ast/variable.h>
#include <core/logging.h>
#include <ast/expression.h>
#include <ast/statement.h>
#include <ast/function_builder.h>

namespace luisa::compute {

void Expression::mark(Usage usage) const noexcept {
    if (auto a = to_underlying(_usage), u = a | to_underlying(usage); a != u) {
        _usage = static_cast<Usage>(u);
        _mark(usage);
    }
}

uint64_t Expression::hash() const noexcept {
    if (!_hash_computed) {
        using namespace std::string_view_literals;
        static thread_local auto seed = hash_value("__hash_expression"sv);
        std::array a{static_cast<uint64_t>(_tag), _compute_hash(), 0ull};
        if (_type != nullptr) { a.back() = _type->hash(); }
        _hash = hash64(&a, sizeof(a), seed);
        _hash_computed = true;
    }
    return _hash;
}

void RefExpr::_mark(Usage usage) const noexcept {
    detail::FunctionBuilder::current()->mark_variable_usage(
        _variable.uid(), usage);
}

uint64_t RefExpr::_compute_hash() const noexcept {
    return hash_value(_variable);
}

void CallExpr::_mark() const noexcept {
    if (is_builtin()) {
        if (_op == CallOp::BUFFER_WRITE ||
            _op == CallOp::TEXTURE_WRITE ||
            _op == CallOp::SET_INSTANCE_TRANSFORM ||
            _op == CallOp::SET_INSTANCE_VISIBILITY ||
            _op == CallOp::ATOMIC_EXCHANGE ||
            _op == CallOp::ATOMIC_COMPARE_EXCHANGE ||
            _op == CallOp::ATOMIC_FETCH_ADD ||
            _op == CallOp::ATOMIC_FETCH_SUB ||
            _op == CallOp::ATOMIC_FETCH_AND ||
            _op == CallOp::ATOMIC_FETCH_OR ||
            _op == CallOp::ATOMIC_FETCH_XOR ||
            _op == CallOp::ATOMIC_FETCH_MIN ||
            _op == CallOp::ATOMIC_FETCH_MAX) {
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
                arg.tag() == Variable::Tag::REFERENCE ||
                        arg.tag() == Variable::Tag::BUFFER ||
                        arg.tag() == Variable::Tag::ACCEL ||
                        arg.tag() == Variable::Tag::TEXTURE ?
                    _custom.variable_usage(arg.uid()) :
                    Usage::READ);
        }
    }
}

uint64_t CallExpr::_compute_hash() const noexcept {
    auto hash = hash64(&_op, sizeof(_op), hash64_default_seed);
    for (auto &&a : _arguments) {
        auto h = a->hash();
        hash = hash64(&h, sizeof(h), hash);
    }
    if (_op == CallOp::CUSTOM) {
        auto h = _custom.hash();
        hash = hash64(&h, sizeof(h), hash);
    }
    return hash;
}

uint64_t UnaryExpr::_compute_hash() const noexcept {
    std::array a{static_cast<uint64_t>(_op), _operand->hash()};
    return hash64(&a, sizeof(a), hash64_default_seed);
}

uint64_t BinaryExpr::_compute_hash() const noexcept {
    auto hl = _lhs->hash();
    auto hr = _rhs->hash();
    std::array a{static_cast<uint64_t>(_op), hl, hr};
    return hash64(&a, sizeof(a), hash64_default_seed);
}

uint64_t AccessExpr::_compute_hash() const noexcept {
    std::array a{_index->hash(), _range->hash()};
    return hash64(&a, sizeof(a), hash64_default_seed);
}

uint64_t MemberExpr::_compute_hash() const noexcept {
    std::array a{(static_cast<uint64_t>(_swizzle_size) << 32u) | _swizzle_code, _self->hash()};
    return hash64(&a, sizeof(a), hash64_default_seed);
}

uint64_t CastExpr::_compute_hash() const noexcept {
    std::array a{static_cast<uint64_t>(_op), _source->hash()};
    return hash64(&a, sizeof(a), hash64_default_seed);
}

uint64_t LiteralExpr::_compute_hash() const noexcept {
    return luisa::visit([](auto &&v) noexcept { return hash_value(v); }, _value);
}

uint64_t ConstantExpr::_compute_hash() const noexcept {
    return hash_value(_data);
}

}// namespace luisa::compute
