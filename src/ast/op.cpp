#include <array>

#include <luisa/ast/op.h>
#include <luisa/ast/expression.h>
#include <luisa/ast/type_registry.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>

namespace {

[[nodiscard]] bool is_lvalue_expression(const luisa::compute::Expression *expr) noexcept {
    switch (expr->tag()) {
        case luisa::compute::Expression::Tag::REF: return true;
        case luisa::compute::Expression::Tag::ACCESS:
            return is_lvalue_expression(static_cast<const luisa::compute::AccessExpr *>(expr)->range());
        case luisa::compute::Expression::Tag::MEMBER: {
            auto member = static_cast<const luisa::compute::MemberExpr *>(expr);
            if (member->is_swizzle()) {
                return member->swizzle_size() == 1u &&
                       is_lvalue_expression(member->self());
            }
            return is_lvalue_expression(member->self());
        }
        default: return false;
    }
}

}// namespace

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

LUISA_AST_API TypePromotion promote_types(BinaryOp op, const Type *lhs, const Type *rhs) noexcept {
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
        auto lhs_and_rhs = [lhs, rhs] {
            static luisa::unordered_map<Type::Tag, uint> scalar_to_score{
                {Type::Tag::BOOL, 0u},
                {Type::Tag::INT8, 1u},
                {Type::Tag::UINT8, 2u},
                {Type::Tag::INT16, 3u},
                {Type::Tag::UINT16, 4u},
                {Type::Tag::INT32, 5u},
                {Type::Tag::UINT32, 6u},
                {Type::Tag::INT64, 7u},
                {Type::Tag::UINT64, 8u},
                {Type::Tag::FLOAT8_E4M3, 9u},
                {Type::Tag::FLOAT8_E5M2, 10u},
                {Type::Tag::FLOAT16, 11u},
                {Type::Tag::FLOAT32, 12u},
                {Type::Tag::FLOAT64, 13u}};
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
                 "for binary operation.",
                 lhs->description(), rhs->description());
    auto v = Type::vector(Type::of<float>(), lhs->dimension());
    return {.lhs = lhs,
            .rhs = v,
            .result = v};
}

LUISA_AST_API void check_builtin_call_valid(CallOp op, const Type *return_type, luisa::span<const Expression *const> args) noexcept {
    switch (op) {
        case CallOp::RAY_TRACING_TRACE_CLOSEST:
        case CallOp::RAY_TRACING_TRACE_ANY:
        case CallOp::RAY_TRACING_QUERY_ALL:
        case CallOp::RAY_TRACING_QUERY_ANY:
        case CallOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR:
        case CallOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR:
        case CallOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR:
        case CallOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: {
            if ((luisa::to_underlying(args[0]->usage()) & luisa::to_underlying(Usage::WRITE)) != 0) [[unlikely]] {
                LUISA_ERROR("Accel must not be writable when tracing.");
            }
            break;
        }
        case CallOp::COOPERATIVE_OUTER_PRODUCT_ACCUMULATE: {
            if (!(return_type == Type::of<void>() &&
                  args.size() == 4 &&
                  args[0]->type()->is_buffer() &&
                  args[1]->type()->is_cooperative_matrix_ref() &&
                  args[2]->type()->is_cooperative_vector() &&
                  args[3]->type()->is_cooperative_vector() &&
                  args[2]->type()->element() == args[3]->type()->element())) [[unlikely]] {
                LUISA_ERROR("Cooperative-Outer-Product-Accumulate call argument type mismatch.");
            }
            auto matrix_dimension = args[1]->type()->coop_matrix_dimension();// weight is KxN
            if (!(args[2]->type()->dimension() == matrix_dimension.x &&
                  args[3]->type()->dimension() == matrix_dimension.y)) [[unlikely]] {
                LUISA_ERROR("Cooperative-Outer-Product-Accumulate call dimension mismatch.");
            }
            break;
        }
        case CallOp::COOPERATIVE_VECTOR_ACCUMULATE: {
            if (!(return_type == Type::of<void>() &&
                  args.size() == 3 &&
                  args[0]->type()->is_buffer() &&
                  args[1]->type()->is_cooperative_vector_ref() &&
                  args[2]->type()->is_cooperative_vector())) [[unlikely]] {
                LUISA_ERROR("Cooperative-Vector-Accumulate call argument type mismatch.");
            }
            if (args[1]->type()->dimension() != args[2]->type()->dimension()) [[unlikely]] {
                LUISA_ERROR("Cooperative-Vector-Accumulate call dimension mismatch.");
            }
            break;
        }
        case CallOp::COOPERATIVE_VECTOR_LOAD: {
            if (!(return_type->is_cooperative_vector() &&
                  args.size() == 2 &&
                  args[0]->type()->is_buffer() &&
                  args[1]->type()->is_cooperative_vector_ref())) [[unlikely]] {
                LUISA_ERROR("Cooperative-Vector-Load call argument type mismatch.");
            }
            if (args[1]->type()->dimension() != return_type->dimension()) [[unlikely]] {
                LUISA_ERROR("Cooperative-Vector-Load call dimension mismatch.");
            }
            break;
        }
        case CallOp::COOPERATIVE_VECTOR_STORE: {
            if (!(return_type == Type::of<void>() &&
                  args.size() == 3 &&
                  args[0]->type()->is_buffer() &&
                  args[1]->type()->is_cooperative_vector_ref() &&
                  args[2]->type()->is_cooperative_vector())) [[unlikely]] {
                LUISA_ERROR("Cooperative-Vector-Store call argument type mismatch.");
            }
            if (args[1]->type()->dimension() != args[2]->type()->dimension()) [[unlikely]] {
                LUISA_ERROR("Cooperative-Vector-Store call dimension mismatch.");
            }
            break;
        }
        case CallOp::COOPERATIVE_VECTOR_SPLAT: {
            if (!(return_type->is_cooperative_vector() &&
                  args.size() == 1 &&
                  args[0]->type()->is_scalar() &&
                  args[0]->type() == return_type->element())) [[unlikely]] {
                LUISA_ERROR("Cooperative-Vector-Splat call argument type mismatch.");
            }
            break;
        }
        case CallOp::COOPERATIVE_VECTOR_CAST: {
            if (!(return_type->is_cooperative_vector() &&
                  args.size() == 1 &&
                  args[0]->type()->is_cooperative_vector() &&
                  args[0]->type()->dimension() == return_type->dimension())) [[unlikely]] {
                LUISA_ERROR("Cooperative-Vector-Cast call argument type mismatch.");
            }
            break;
        }
        case CallOp::BINDLESS_COOPERATIVE_VECTOR_LOAD:
        case CallOp::TYPED_BINDLESS_COOPERATIVE_VECTOR_LOAD: {
            if (!(return_type->is_cooperative_vector() &&
                  args.size() == 3 &&
                  args[0]->type()->is_bindless_array() &&
                  args[1]->type()->is_uint32() &&
                  args[2]->type()->is_cooperative_vector_ref())) [[unlikely]] {
                LUISA_ERROR("Bindless-Cooperative-Vector-Load call argument type mismatch.");
            }
            if (args[2]->type()->dimension() != return_type->dimension()) [[unlikely]] {
                LUISA_ERROR("Bindless-Cooperative-Vector-Load call dimension mismatch.");
            }
            break;
        }
        case CallOp::BINDLESS_COOPERATIVE_VECTOR_STORE:
        case CallOp::TYPED_BINDLESS_COOPERATIVE_VECTOR_STORE: {
            if (!(return_type == Type::of<void>() &&
                  args.size() == 4 &&
                  args[0]->type()->is_bindless_array() &&
                  args[1]->type()->is_uint32() &&
                  args[2]->type()->is_cooperative_vector_ref() &&
                  args[3]->type()->is_cooperative_vector())) [[unlikely]] {
                LUISA_ERROR("Bindless-Cooperative-Vector-Store call argument type mismatch.");
            }
            if (args[2]->type()->dimension() != args[3]->type()->dimension()) [[unlikely]] {
                LUISA_ERROR("Bindless-Cooperative-Vector-Store call dimension mismatch.");
            }
            break;
        }
        case CallOp::COOPERATIVE_VECTOR_WORKGROUP_LOAD: {
            if (!(return_type->is_cooperative_vector() &&
                  args.size() == 2 &&
                  args[0]->type()->is_array() &&
                  args[1]->type()->is_uint32())) [[unlikely]] {
                LUISA_ERROR("Cooperative-Vector-Workgroup-Load call argument type mismatch.");
            }
            break;
        }
        case CallOp::COOPERATIVE_VECTOR_WORKGROUP_STORE: {
            if (!(return_type == Type::of<void>() &&
                  args.size() == 3 &&
                  args[0]->type()->is_array() &&
                  args[1]->type()->is_uint32() &&
                  args[2]->type()->is_cooperative_vector())) [[unlikely]] {
                LUISA_ERROR("Cooperative-Vector-Workgroup-Store call argument type mismatch.");
            }
            break;
        }
        case CallOp::COOPERATIVE_MUL_ADD: {
            if ((luisa::to_underlying(args[0]->usage()) & luisa::to_underlying(Usage::WRITE)) != 0 &&
                (luisa::to_underlying(args[2]->usage()) & luisa::to_underlying(Usage::WRITE)) == 0) [[unlikely]] {
                LUISA_ERROR("Matrix-buffer and bias-buffer must not be writable.");
            }
            if (!(return_type->is_cooperative_vector() &&
                  args.size() == 5 &&
                  args[0]->type()->is_buffer() &&
                  args[1]->type()->is_cooperative_matrix_ref() &&
                  args[2]->type()->is_buffer() &&
                  args[3]->type()->is_cooperative_vector_ref() &&
                  args[4]->type()->is_cooperative_vector())) [[unlikely]] {
                LUISA_ERROR("Cooperative-Mul-Add call argument type mismatch.");
            }
            // https://developer.nvidia.com/blog/neural-rendering-in-nvidia-optix-using-cooperative-vectors/
            auto matrix_dimension = args[1]->type()->coop_matrix_dimension();// weight is KxN
            if (!(return_type->dimension() == matrix_dimension.y &&          // output is N
                  args[3]->type()->dimension() == matrix_dimension.y &&      // bias is N
                  args[4]->type()->dimension() == matrix_dimension.x         // input is K
                  )) [[unlikely]] {
                LUISA_ERROR("Cooperative-Mul-Add call dimension mismatch.");
            }
            break;
        }
        case CallOp::TYPED_BINDLESS_COOPERATIVE_MUL_ADD:
        case CallOp::BINDLESS_COOPERATIVE_MUL_ADD: {
            if (!(return_type->is_cooperative_vector() &&
                  args.size() == 6 &&
                  args[0]->type()->is_bindless_array() &&
                  args[1]->type()->is_uint32() &&
                  args[2]->type()->is_cooperative_matrix_ref() &&
                  args[3]->type()->is_uint32() &&
                  args[4]->type()->is_cooperative_vector_ref() &&
                  args[5]->type()->is_cooperative_vector())) [[unlikely]] {
                LUISA_ERROR("Cooperative-Mul-Add call argument type mismatch.");
            }
            // https://developer.nvidia.com/blog/neural-rendering-in-nvidia-optix-using-cooperative-vectors/
            auto matrix_dimension = args[2]->type()->coop_matrix_dimension();// weight is KxN
            if (!(return_type->dimension() == matrix_dimension.y &&          // output is N
                  args[4]->type()->dimension() == matrix_dimension.y &&      // bias is N
                  args[5]->type()->dimension() == matrix_dimension.x         // input is K
                  )) [[unlikely]] {
                LUISA_ERROR("Cooperative-Mul-Add call dimension mismatch.");
            }
            break;
        }
        case CallOp::COOPERATIVE_MUL: {
            if ((luisa::to_underlying(args[0]->usage()) & luisa::to_underlying(Usage::WRITE)) != 0) [[unlikely]] {
                LUISA_ERROR("Matrix-buffer must not be writable.");
            }
            if (!(return_type->is_cooperative_vector() &&
                  args.size() == 3 &&
                  args[0]->type()->is_buffer() &&
                  args[1]->type()->is_cooperative_matrix_ref() &&
                  args[2]->type()->is_cooperative_vector())) [[unlikely]] {
                LUISA_ERROR("Cooperative-Mul call argument type mismatch.");
            }
            auto matrix_dimension = args[1]->type()->coop_matrix_dimension();// weight is KxN
            if (!(return_type->dimension() == matrix_dimension.y &&          // output is N
                  args[2]->type()->dimension() == matrix_dimension.x         // input is K
                  )) [[unlikely]] {
                LUISA_ERROR("Cooperative-Mul call dimension mismatch.");
            }
            break;
        }
        case CallOp::TYPED_BINDLESS_COOPERATIVE_MUL:
        case CallOp::BINDLESS_COOPERATIVE_MUL: {
            if (!(return_type->is_cooperative_vector() &&
                  args.size() == 4 &&
                  args[0]->type()->is_bindless_array() &&
                  args[1]->type()->is_uint32() &&
                  args[2]->type()->is_cooperative_matrix_ref() &&
                  args[3]->type()->is_cooperative_vector())) [[unlikely]] {
                LUISA_ERROR("Cooperative-Mul call argument type mismatch.");
            }
            auto matrix_dimension = args[2]->type()->coop_matrix_dimension();// weight is KxN
            if (!(return_type->dimension() == matrix_dimension.y &&          // output is N
                  args[3]->type()->dimension() == matrix_dimension.x         // input is K
                  )) [[unlikely]] {
                LUISA_ERROR("Cooperative-Mul call dimension mismatch.");
            }
            break;
        }
        case CallOp::ASYNC_COPY: {
            if (!(return_type->is_uint32() &&
                  args.size() == 7 &&
                  args[0]->type()->is_uint32() &&
                  args[3]->type()->is_uint32() &&
                  args[4]->type()->is_uint32() &&
                  args[5]->type()->is_uint32() &&
                  args[6]->type()->is_uint32() &&
                  is_lvalue_expression(args[1]) &&
                  is_lvalue_expression(args[2]))) [[unlikely]] {
                LUISA_ERROR("ASYNC_COPY argument type mismatch.");
            }
            break;
        }
        default: break;
    }
}

}// namespace luisa::compute
