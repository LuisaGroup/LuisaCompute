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
    // otherwise, must be matrix * vector or vector * matrix
    LUISA_ASSERT(((lhs->is_matrix() && rhs->is_vector()) ||
                  (lhs->is_vector() && rhs->is_matrix())) &&
                     lhs->dimension() == rhs->dimension(),
                 "Invalid operand types '{}' and '{}' "
                 "for binary operation.",
                 lhs->description(), rhs->description());
    auto v = Type::vector(Type::of<float>(), lhs->dimension());
    return {.lhs = lhs->is_matrix() ? lhs : v,
            .rhs = rhs->is_matrix() ? rhs : v,
            .result = v};
}

LUISA_AST_API void check_builtin_call_valid(CallOp op, const Type *return_type, luisa::span<const Expression *const> args) noexcept {
    switch (op) {
        case CallOp::UNDEFINED: {
            LUISA_ASSERT(return_type != nullptr &&
                             return_type != Type::of<void>() &&
                             args.empty(),
                         "UNDEFINED requires a non-void result type and no arguments.");
            break;
        }
        case CallOp::PACK: {
            if (!(return_type == Type::of<void>() &&
                  args.size() == 3u &&
                  !args[0]->type()->is_resource() &&
                  !args[0]->type()->is_custom() &&
                  args[1]->type()->is_buffer() &&
                  args[1]->type()->element() == Type::of<uint32_t>() &&
                  args[2]->type() == Type::of<uint32_t>())) [[unlikely]] {
                LUISA_ERROR("PACK expects (packable value, buffer<uint>, uint offset).");
            }
            break;
        }
        case CallOp::UNPACK: {
            if (!(return_type != Type::of<void>() &&
                  !return_type->is_resource() &&
                  !return_type->is_custom() &&
                  args.size() == 2u &&
                  args[0]->type()->is_buffer() &&
                  args[0]->type()->element() == Type::of<uint32_t>() &&
                  args[1]->type() == Type::of<uint32_t>())) [[unlikely]] {
                LUISA_ERROR("UNPACK expects (buffer<uint>, uint offset) and a packable return type.");
            }
            break;
        }
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
                  args[3]->type()->dimension() == matrix_dimension.x         // input is K
                  )) [[unlikely]] {
                LUISA_ERROR("Cooperative-Mul call dimension mismatch.");
            }
            break;
        }
        // Future cooperative-vector element-wise operations. These validate only
        // the general operand shape; every backend currently rejects them with a
        // placeholder assertion until native support lands.
        case CallOp::COOPERATIVE_VECTOR_DOT: {
            if (!(return_type->is_scalar() &&
                  args.size() == 2u &&
                  args[0]->type()->is_cooperative_vector() &&
                  args[1]->type()->is_cooperative_vector() &&
                  args[0]->type()->dimension() == args[1]->type()->dimension())) [[unlikely]] {
                LUISA_ERROR("Cooperative-Vector-Dot call argument type mismatch.");
            }
            break;
        }
        case CallOp::COOPERATIVE_VECTOR_ABS:
        case CallOp::COOPERATIVE_VECTOR_SIGN:
        case CallOp::COOPERATIVE_VECTOR_FLOOR:
        case CallOp::COOPERATIVE_VECTOR_CEIL:
        case CallOp::COOPERATIVE_VECTOR_FRACT:
        case CallOp::COOPERATIVE_VECTOR_TRUNC:
        case CallOp::COOPERATIVE_VECTOR_ROUND:
        case CallOp::COOPERATIVE_VECTOR_RINT:
        case CallOp::COOPERATIVE_VECTOR_SQRT:
        case CallOp::COOPERATIVE_VECTOR_RSQRT:
        case CallOp::COOPERATIVE_VECTOR_EXP2:
        case CallOp::COOPERATIVE_VECTOR_EXP10:
        case CallOp::COOPERATIVE_VECTOR_LOG2:
        case CallOp::COOPERATIVE_VECTOR_LOG10:
        case CallOp::COOPERATIVE_VECTOR_SATURATE:
        case CallOp::COOPERATIVE_VECTOR_SIN:
        case CallOp::COOPERATIVE_VECTOR_COS:
        case CallOp::COOPERATIVE_VECTOR_TAN:
        case CallOp::COOPERATIVE_VECTOR_ASIN:
        case CallOp::COOPERATIVE_VECTOR_ACOS:
        case CallOp::COOPERATIVE_VECTOR_SINH:
        case CallOp::COOPERATIVE_VECTOR_COSH:
        case CallOp::COOPERATIVE_VECTOR_ASINH:
        case CallOp::COOPERATIVE_VECTOR_ACOSH:
        case CallOp::COOPERATIVE_VECTOR_ATANH: {
            if (!(return_type->is_cooperative_vector() &&
                  args.size() == 1u &&
                  args[0]->type()->is_cooperative_vector() &&
                  args[0]->type()->dimension() == return_type->dimension())) [[unlikely]] {
                LUISA_ERROR("Cooperative-Vector unary call argument type mismatch.");
            }
            break;
        }
        case CallOp::COOPERATIVE_VECTOR_ISINF:
        case CallOp::COOPERATIVE_VECTOR_ISNAN: {
            if (!(return_type->is_cooperative_vector() &&
                  return_type->element()->is_bool() &&
                  args.size() == 1u &&
                  args[0]->type()->is_cooperative_vector() &&
                  args[0]->type()->dimension() == return_type->dimension())) [[unlikely]] {
                LUISA_ERROR("Cooperative-Vector-IsInf/IsNan call argument type mismatch.");
            }
            break;
        }
        case CallOp::COOPERATIVE_VECTOR_POW:
        case CallOp::COOPERATIVE_VECTOR_STEP:
        case CallOp::COOPERATIVE_VECTOR_ADD:
        case CallOp::COOPERATIVE_VECTOR_SUB:
        case CallOp::COOPERATIVE_VECTOR_MUL:
        case CallOp::COOPERATIVE_VECTOR_DIV: {
            if (!(return_type->is_cooperative_vector() &&
                  args.size() == 2u &&
                  args[0]->type()->is_cooperative_vector() &&
                  args[1]->type()->is_cooperative_vector() &&
                  args[0]->type()->dimension() == args[1]->type()->dimension() &&
                  args[0]->type()->dimension() == return_type->dimension())) [[unlikely]] {
                LUISA_ERROR("Cooperative-Vector binary call argument type mismatch.");
            }
            break;
        }
        case CallOp::COOPERATIVE_VECTOR_LESS:
        case CallOp::COOPERATIVE_VECTOR_LESS_EQUAL:
        case CallOp::COOPERATIVE_VECTOR_GREATER:
        case CallOp::COOPERATIVE_VECTOR_GREATER_EQUAL:
        case CallOp::COOPERATIVE_VECTOR_EQUAL:
        case CallOp::COOPERATIVE_VECTOR_NOT_EQUAL: {
            if (!(return_type->is_cooperative_vector() &&
                  return_type->element()->is_bool() &&
                  args.size() == 2u &&
                  args[0]->type()->is_cooperative_vector() &&
                  args[1]->type()->is_cooperative_vector() &&
                  args[0]->type()->dimension() == args[1]->type()->dimension() &&
                  args[0]->type()->dimension() == return_type->dimension())) [[unlikely]] {
                LUISA_ERROR("Cooperative-Vector relational call argument type mismatch.");
            }
            break;
        }
        case CallOp::COOPERATIVE_VECTOR_MIX:
        case CallOp::COOPERATIVE_VECTOR_LERP:
        case CallOp::COOPERATIVE_VECTOR_SMOOTHSTEP: {
            if (!(return_type->is_cooperative_vector() &&
                  args.size() == 3u &&
                  args[0]->type()->is_cooperative_vector() &&
                  args[1]->type()->is_cooperative_vector() &&
                  args[2]->type()->is_cooperative_vector() &&
                  args[0]->type()->dimension() == args[1]->type()->dimension() &&
                  args[0]->type()->dimension() == args[2]->type()->dimension() &&
                  args[0]->type()->dimension() == return_type->dimension())) [[unlikely]] {
                LUISA_ERROR("Cooperative-Vector ternary call argument type mismatch.");
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
        case CallOp::TENSOR_COPY:
        case CallOp::TENSOR_FILL:
        case CallOp::TENSOR_CAST:
        case CallOp::TENSOR_PERMUTE:
        case CallOp::TENSOR_CONCAT:
        case CallOp::TENSOR_PAD:
        case CallOp::TENSOR_NEG:
        case CallOp::TENSOR_ABS:
        case CallOp::TENSOR_EXP:
        case CallOp::TENSOR_LOG:
        case CallOp::TENSOR_SQRT:
        case CallOp::TENSOR_RSQRT:
        case CallOp::TENSOR_SIN:
        case CallOp::TENSOR_COS:
        case CallOp::TENSOR_TAN:
        case CallOp::TENSOR_TANH:
        case CallOp::TENSOR_SIGMOID:
        case CallOp::TENSOR_GELU:
        case CallOp::TENSOR_RELU:
        case CallOp::TENSOR_LEAKY_RELU:
        case CallOp::TENSOR_ERF:
        case CallOp::TENSOR_CEIL:
        case CallOp::TENSOR_FLOOR:
        case CallOp::TENSOR_ROUND:
        case CallOp::TENSOR_ISNAN:
        case CallOp::TENSOR_ISINF:
        case CallOp::TENSOR_ADD:
        case CallOp::TENSOR_SUB:
        case CallOp::TENSOR_MUL:
        case CallOp::TENSOR_DIV:
        case CallOp::TENSOR_POW:
        case CallOp::TENSOR_MIN:
        case CallOp::TENSOR_MAX:
        case CallOp::TENSOR_CLAMP:
        case CallOp::TENSOR_FMA:
        case CallOp::TENSOR_REDUCE_SUM:
        case CallOp::TENSOR_REDUCE_MAX:
        case CallOp::TENSOR_REDUCE_MIN:
        case CallOp::TENSOR_CUMSUM:
        case CallOp::TENSOR_MATMUL:
        case CallOp::TENSOR_CONTRACT:
        case CallOp::TENSOR_BATCH_MATMUL: {
            // Runtime tensor operators (plan.md §1.5). All tensor ops are
            // side-effecting statement-like calls returning void. Each tensor
            // operand is encoded as six arguments: [dtype:uint32, rank:uint32,
            // extents:uint4, strides:uint4, offset:uint32, addr:uint64]. The
            // remaining arguments are op-specific scalar/vector constants
            // (count, dims, alpha/beta, modes, ...) carried verbatim to the
            // backend.
            auto is_uint32 = [](const Expression *e) noexcept { return e->type()->is_uint32(); };
            auto is_uint4 = [](const Expression *e) noexcept {
                return e->type()->is_vector() && e->type()->element()->is_uint32() &&
                       e->type()->dimension() == 4u;
            };
            auto is_float32 = [](const Expression *e) noexcept { return e->type()->is_float32(); };
            auto is_uint64 = [](const Expression *e) noexcept { return e->type()->is_uint64(); };
            auto check_desc = [&](const Expression *const *a) noexcept {
                return is_uint32(a[0]) && is_uint32(a[1]) && is_uint4(a[2]) &&
                       is_uint4(a[3]) && is_uint32(a[4]) && is_uint64(a[5]);
            };
            auto check_void = [&]() noexcept {
                return return_type == nullptr || return_type == Type::of<void>();
            };
            auto fail = [&] { LUISA_ERROR("Tensor call argument type mismatch ({}).", luisa::to_string(op)); };
            if (!check_void()) [[unlikely]] { fail(); }
            switch (op) {
                case CallOp::TENSOR_COPY:
                case CallOp::TENSOR_CAST:
                    if (!(args.size() == 13u && check_desc(args.data()) &&
                          check_desc(args.data() + 6u) && is_uint32(args[12]))) [[unlikely]] {
                        fail();
                    }
                    break;
                case CallOp::TENSOR_FILL:
                    if (!(args.size() == 8u && check_desc(args.data()) &&
                          is_uint32(args[6]) && is_uint32(args[7]))) [[unlikely]] {
                        fail();
                    }
                    break;
                case CallOp::TENSOR_PERMUTE:
                case CallOp::TENSOR_PAD:
                    if (!(args.size() == 13u && check_desc(args.data()) &&
                          check_desc(args.data() + 6u) && is_uint4(args[12]))) [[unlikely]] {
                        fail();
                    }
                    break;
                case CallOp::TENSOR_CONCAT:
                    if (!(args.size() == 56u && check_desc(args.data()) &&
                          is_uint32(args[6]) && is_uint32(args[7]))) [[unlikely]] {
                        fail();
                    }
                    for (auto i = 0u; i < 8u; i++) {
                        if (!check_desc(args.data() + 8u + 6u * i)) [[unlikely]] { fail(); }
                    }
                    break;
                case CallOp::TENSOR_NEG:
                case CallOp::TENSOR_ABS:
                case CallOp::TENSOR_EXP:
                case CallOp::TENSOR_LOG:
                case CallOp::TENSOR_SQRT:
                case CallOp::TENSOR_RSQRT:
                case CallOp::TENSOR_SIN:
                case CallOp::TENSOR_COS:
                case CallOp::TENSOR_TAN:
                case CallOp::TENSOR_TANH:
                case CallOp::TENSOR_SIGMOID:
                case CallOp::TENSOR_GELU:
                case CallOp::TENSOR_RELU:
                case CallOp::TENSOR_LEAKY_RELU:
                case CallOp::TENSOR_ERF:
                case CallOp::TENSOR_CEIL:
                case CallOp::TENSOR_FLOOR:
                case CallOp::TENSOR_ROUND:
                case CallOp::TENSOR_ISNAN:
                case CallOp::TENSOR_ISINF:
                    if (!(args.size() == 13u && check_desc(args.data()) &&
                          check_desc(args.data() + 6u) && is_uint32(args[12]))) [[unlikely]] {
                        fail();
                    }
                    break;
                case CallOp::TENSOR_ADD:
                case CallOp::TENSOR_SUB:
                case CallOp::TENSOR_MUL:
                case CallOp::TENSOR_DIV:
                case CallOp::TENSOR_POW:
                case CallOp::TENSOR_MIN:
                case CallOp::TENSOR_MAX:
                    if (!(args.size() == 19u && check_desc(args.data()) &&
                          check_desc(args.data() + 6u) && check_desc(args.data() + 12u) &&
                          is_uint32(args[18]))) [[unlikely]] {
                        fail();
                    }
                    break;
                case CallOp::TENSOR_CLAMP:
                    if (!(args.size() == 15u && check_desc(args.data()) &&
                          check_desc(args.data() + 6u) && is_uint32(args[12]) &&
                          is_uint32(args[13]) && is_uint32(args[14]))) [[unlikely]] {
                        fail();
                    }
                    break;
                case CallOp::TENSOR_FMA:
                    if (!(args.size() == 25u && check_desc(args.data()) &&
                          check_desc(args.data() + 6u) && check_desc(args.data() + 12u) &&
                          check_desc(args.data() + 18u) && is_uint32(args[24]))) [[unlikely]] {
                        fail();
                    }
                    break;
                case CallOp::TENSOR_REDUCE_SUM:
                case CallOp::TENSOR_REDUCE_MAX:
                case CallOp::TENSOR_REDUCE_MIN:
                    if (!(args.size() == 14u && check_desc(args.data()) &&
                          check_desc(args.data() + 6u) && is_uint32(args[12]) &&
                          is_uint4(args[13]))) [[unlikely]] {
                        fail();
                    }
                    break;
                case CallOp::TENSOR_CUMSUM:
                    if (!(args.size() == 13u && check_desc(args.data()) &&
                          check_desc(args.data() + 6u) && is_uint32(args[12]))) [[unlikely]] {
                        fail();
                    }
                    break;
                case CallOp::TENSOR_MATMUL:
                    if (!(args.size() == 24u && check_desc(args.data()) &&
                          check_desc(args.data() + 6u) && check_desc(args.data() + 12u) &&
                          is_uint32(args[18]) && is_uint32(args[19]) && is_uint32(args[20]) &&
                          is_float32(args[21]) && is_float32(args[22]) && is_uint32(args[23]))) [[unlikely]] {
                        fail();
                    }
                    break;
                case CallOp::TENSOR_BATCH_MATMUL:
                    if (!(args.size() == 25u && check_desc(args.data()) &&
                          check_desc(args.data() + 6u) && check_desc(args.data() + 12u) &&
                          is_uint32(args[18]) && is_uint32(args[19]) && is_uint32(args[20]) &&
                          is_float32(args[21]) && is_float32(args[22]) && is_uint32(args[23]) &&
                          is_uint32(args[24]))) [[unlikely]] {
                        fail();
                    }
                    break;
                case CallOp::TENSOR_CONTRACT:
                    if (!(args.size() == 22u && check_desc(args.data()) &&
                          check_desc(args.data() + 6u) && check_desc(args.data() + 12u) &&
                          is_uint4(args[18]) && is_uint4(args[19]) && is_uint4(args[20]) &&
                          is_uint32(args[21]))) [[unlikely]] {
                        fail();
                    }
                    break;
                default: break;
            }
            break;
        }
        case CallOp::ASYNC_COPY: {
            // The op is emitted as a statement (void) by the DSL; the uint return
            // type is kept for the SPIR-V event handle in the AST contract.
            // dst is an lvalue of the shared-memory destination; src is the
            // 64-bit device address of the global source.
            if (!((return_type == nullptr || return_type->is_uint32()) &&
                  args.size() == 7 &&
                  args[0]->type()->is_uint32() &&
                  is_lvalue_expression(args[1]) &&
                  args[2]->type()->is_uint64() &&
                  args[3]->type()->is_uint32() &&
                  args[4]->type()->is_uint32() &&
                  args[5]->type()->is_uint32() &&
                  args[6]->type()->is_uint32())) [[unlikely]] {
                LUISA_ERROR("ASYNC_COPY argument type mismatch.");
            }
            break;
        }
        case CallOp::PIPELINE_COMMIT: {
            LUISA_ASSERT(args.empty(), "PIPELINE_COMMIT takes no arguments.");
            break;
        }
        case CallOp::PIPELINE_WAIT_PRIOR: {
            LUISA_ASSERT(args.size() == 1 && args[0]->type()->is_uint32(),
                         "PIPELINE_WAIT_PRIOR: expected (uint prior_stages).");
            break;
        }
        case CallOp::CLUSTER_LAUNCH_CONTROL_TRY_CANCEL:
        case CallOp::CLUSTER_LAUNCH_CONTROL_TRY_CANCEL_MULTICAST: {
            LUISA_ASSERT(args.size() == 2 &&
                             args[0]->type()->is_uint32() &&
                             args[0]->type()->is_vector() &&
                             args[0]->type()->dimension() == 4 &&
                             args[1]->type()->is_uint64(),
                         "CLC try_cancel: expected (uint4 result, uint64 bar).");
            break;
        }
        case CallOp::CLUSTER_LAUNCH_CONTROL_QUERY_IS_CANCELED: {
            LUISA_ASSERT(args.size() == 1 &&
                             args[0]->type()->is_uint32() &&
                             args[0]->type()->dimension() == 4,
                         "CLC query_is_canceled: expected (uint4 result).");
            break;
        }
        case CallOp::CLUSTER_LAUNCH_CONTROL_QUERY_GET_CTAD_X:
        case CallOp::CLUSTER_LAUNCH_CONTROL_QUERY_GET_CTAD_Y:
        case CallOp::CLUSTER_LAUNCH_CONTROL_QUERY_GET_CTAD_Z: {
            LUISA_ASSERT(args.size() == 1 &&
                             args[0]->type()->is_uint32() &&
                             args[0]->type()->dimension() == 4,
                         "CLC query_get_ctaid: expected (uint4 result).");
            break;
        }
        case CallOp::MBARRIER_INIT: {
            LUISA_ASSERT(args.size() == 2 &&
                             args[0]->type()->is_uint64() &&
                             args[1]->type()->is_uint32(),
                         "MBARRIER_INIT: expected (uint64 bar, uint count).");
            break;
        }
        case CallOp::MBARRIER_ARRIVE_EXPECT_TX: {
            LUISA_ASSERT(args.size() == 2 &&
                             args[0]->type()->is_uint64() &&
                             args[1]->type()->is_uint32(),
                         "MBARRIER_ARRIVE_EXPECT_TX: expected (uint64 bar, uint tx_bytes).");
            break;
        }
        case CallOp::MBARRIER_TRY_WAIT_PARITY: {
            LUISA_ASSERT(args.size() == 2 &&
                             args[0]->type()->is_uint64() &&
                             args[1]->type()->is_int32(),
                         "MBARRIER_TRY_WAIT_PARITY: expected (uint64 bar, int phase).");
            break;
        }
        default: break;
    }
}

}// namespace luisa::compute
