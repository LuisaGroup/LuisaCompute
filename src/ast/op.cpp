//
// Created by Mike Smith on 2021/8/6.
//

#include <core/logging.h>
#include <ast/op.h>
#include <ast/type_registry.h>

namespace luisa::compute {

CallOpSet::Iterator::Iterator(const CallOpSet &set) noexcept : _set{set} {
    while (_index != call_op_count && !_set.test(static_cast<CallOp>(_index))) {
        _index++;
    }
}

CallOp CallOpSet::Iterator::operator*() const noexcept {
    return static_cast<CallOp>(_index);
}

CallOpSet::Iterator &CallOpSet::Iterator::operator++() noexcept {
    if (_index == call_op_count) {
        LUISA_ERROR_WITH_LOCATION(
            "Walking past the end of CallOpSet.");
    }
    _index++;
    while (_index != call_op_count && !_set.test(static_cast<CallOp>(_index))) {
        _index++;
    }
    return (*this);
}

CallOpSet::Iterator CallOpSet::Iterator::operator++(int) noexcept {
    auto self = *this;
    this->operator++();
    return self;
}

bool CallOpSet::Iterator::operator==(luisa::default_sentinel_t) const noexcept {
    return _index == call_op_count;
}

TypePromotion promote_types(BinaryOp op, const Type *lhs, const Type *rhs) noexcept {
    auto dimensions_compatible = [](auto a, auto b) noexcept {
        return a->dimension() == b->dimension() ||
               a->dimension() == 1u ||
               b->dimension() == 1u;
    };
    if (is_logical(op)) {
        LUISA_ASSERT((lhs->is_scalar() || lhs->is_vector()) &&
                         (rhs->is_scalar() || rhs->is_vector()) &&
                         dimensions_compatible(lhs, rhs),
                     "Invalid operand types '{}' and '{}' "
                     "for logical binary operation.",
                     lhs->description(), rhs->description());
        auto dim = std::max(lhs->dimension(), rhs->dimension());
        auto t = std::array{Type::of<bool>(),
                            Type::of<bool2>(),
                            Type::of<bool3>(),
                            Type::of<bool4>()}[dim - 1u];
        return {.lhs = t, .rhs = t, .result = t};
    }
    if (lhs->is_scalar() && rhs->is_scalar()) {
        auto lhs_and_rhs = [&] {
            switch (lhs->tag()) {
                case Type::Tag::BOOL: return rhs;
                case Type::Tag::FLOAT: return lhs;
                case Type::Tag::INT: return rhs->tag() == Type::Tag::BOOL ? lhs : rhs;
                case Type::Tag::UINT: return rhs->tag() == Type::Tag::FLOAT ? rhs : lhs;
                default: LUISA_ERROR_WITH_LOCATION(
                    "Invalid operand types '{}' and '{}'.",
                    lhs->description(), rhs->description());
            }
        }();
        return {.lhs = lhs_and_rhs,
                .rhs = lhs_and_rhs,
                .result = is_relational(op) ?
                              Type::of<bool>() :
                              lhs_and_rhs};
    }
    if ((lhs->is_scalar() && rhs->is_vector()) ||
        (lhs->is_vector() && rhs->is_scalar()) ||
        (lhs->is_vector() && rhs->is_vector())) {
        LUISA_ASSERT(dimensions_compatible(lhs, rhs),
                     "Invalid operand types '{}' and '{}' "
                     "for binary operation.",
                     lhs->description(), rhs->description());
        auto prom = promote_types(op, lhs->element(), rhs->element());
        auto dim = std::max(lhs->dimension(), rhs->dimension());
        return {.lhs = Type::from(luisa::format(
                    "vector<{},{}>", prom.lhs->description(), dim)),
                .rhs = Type::from(luisa::format(
                    "vector<{},{}>", prom.rhs->description(), dim)),
                .result = Type::from(luisa::format(
                    "vector<{},{}>", prom.result->description(), dim))};
    }
    if ((lhs->is_matrix() && rhs->is_vector()) ||
        (lhs->is_vector() && rhs->is_matrix())) {
        LUISA_ASSERT(lhs->dimension() == rhs->dimension() &&
                         lhs->element()->tag() == Type::Tag::FLOAT &&
                         rhs->element()->tag() == Type::Tag::FLOAT,
                     "Invalid operand types '{}' and '{}' "
                     "for binary operation.",
                     lhs->description(), rhs->description());
        return {.lhs = lhs,
                .rhs = rhs,
                .result = lhs->is_matrix() ? rhs : lhs};
    }
    LUISA_ASSERT(lhs->element()->tag() == Type::Tag::FLOAT &&
                     rhs->element()->tag() == Type::Tag::FLOAT &&
                     dimensions_compatible(lhs, rhs),
                 "Invalid operand types '{}' and '{}' "
                 "for binary operation.",
                 lhs->description(), rhs->description());
    auto t = lhs->is_matrix() ? lhs : rhs;
    return {.lhs = t,
            .rhs = t,
            .result = t};
}

}// namespace luisa::compute
