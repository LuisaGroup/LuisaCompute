#include <luisa/core/logging.h>
#include <luisa/ast/op.h>
#include <luisa/ast/type_registry.h>
#include <array>

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

LC_AST_API TypePromotion promote_types(BinaryOp op, const Type *lhs, const Type *rhs) noexcept {
    auto dimensions_compatible = [](auto a, auto b) noexcept {
        return a->dimension() == b->dimension() ||
               a->dimension() == 1u || b->dimension() == 1u;
    };
    // logical operator; cast both operands to bool or boolN
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
    // scalar op scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        auto lhs_and_rhs = [&] {
            static luisa::unordered_map<Type::Tag, uint> scalar_to_score{
                {Type::Tag::BOOL, 0u},
                {Type::Tag::INT16, 1u},
                {Type::Tag::UINT16, 2u},
                {Type::Tag::INT32, 3u},
                {Type::Tag::UINT32, 4u},
                {Type::Tag::INT64, 5u},
                {Type::Tag::UINT64, 6u},
                {Type::Tag::FLOAT16, 7u},
                {Type::Tag::FLOAT32, 8u},
                {Type::Tag::FLOAT64, 9u}};
            return scalar_to_score.at(lhs->tag()) > scalar_to_score.at(rhs->tag()) ?
                       lhs :
                       rhs;
        }();
        return {.lhs = lhs_and_rhs,
                .rhs = lhs_and_rhs,
                .result = is_relational(op) ?
                              Type::of<bool>() :
                              lhs_and_rhs};
    }
    // scalar op vector | vector op scalar | vector op vector
    if ((lhs->is_scalar() && rhs->is_vector()) ||
        (lhs->is_vector() && rhs->is_scalar()) ||
        (lhs->is_vector() && rhs->is_vector())) {
        LUISA_ASSERT(dimensions_compatible(lhs, rhs),
                     "Invalid operand types '{}' and '{}' "
                     "for binary operation.",
                     lhs->description(), rhs->description());
        auto prom = promote_types(op, lhs->element(), rhs->element());
        auto dim = std::max(lhs->dimension(), rhs->dimension());
        return {.lhs = Type::vector(prom.lhs, dim),
                .rhs = Type::vector(prom.rhs, dim),
                .result = Type::vector(prom.result, dim)};
    }
    // matrix op matrix
    if (lhs->is_matrix() && rhs->is_matrix()) {
        LUISA_ASSERT(lhs->dimension() == rhs->dimension(),
                     "Invalid operand types '{}' and '{}' "
                     "for binary operation.",
                     lhs->description(), rhs->description());
        return {.lhs = lhs,
                .rhs = rhs,
                .result = lhs};
    }
    // matrix op scalar
    if (lhs->is_matrix() && rhs->is_scalar()) {
        return {.lhs = lhs,
                .rhs = Type::of<float>(),
                .result = lhs};
    }
    // scalar op matrix
    if (lhs->is_scalar() && rhs->is_matrix()) {
        return {.lhs = Type::of<float>(),
                .rhs = rhs,
                .result = rhs};
    }
    // otherwise, must be matrix * vector
    LUISA_ASSERT(lhs->is_matrix() && rhs->is_vector() &&
                 lhs->dimension() == rhs->dimension(),
                 "Invalid operand types '{}' and '{}' "
                 "for binary operation.");
    auto v = Type::vector(Type::of<float>(), lhs->dimension());
    return {.lhs = lhs,
            .rhs = v,
            .result = v};
}

}// namespace luisa::compute
