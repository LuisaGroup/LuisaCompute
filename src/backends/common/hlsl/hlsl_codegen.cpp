#include "hlsl_codegen.h"
#include "struct_generator.h"
#include "codegen_stack_data.h"
#include "../env_flag.h"
#include "../shader_print_formatter.h"
#include <luisa/core/stl/format.h>
#include <algorithm>
#include <cmath>
#include <optional>
namespace lc::hlsl {

struct LocalEmitPlan {
    luisa::unordered_map<const IfStmt *, bool> const_if_taken;
    luisa::unordered_map<const SwitchStmt *, const Statement *> const_switch_branch;
    luisa::unordered_set<const SwitchStmt *> const_switch_needs_guard;
    luisa::unordered_set<const AssignStmt *> skip_assigns;
    luisa::unordered_map<const AssignStmt *, Variable> ssa_decl_assigns;
    luisa::unordered_map<uint32_t, const Expression *> inline_exprs;
    luisa::unordered_map<const ScopeStmt *, luisa::vector<Variable>> scoped_decls;
};

namespace {

struct ConstScalar {
    enum class Kind : uint8_t { BOOL, I32, U32, F32 };
    Kind kind{Kind::BOOL};
    uint32_t bits{0u};
};

[[nodiscard]] bool is_portable_f32(float v) noexcept {
    return std::isfinite(v) && std::fpclassify(v) != FP_SUBNORMAL;
}

[[nodiscard]] std::optional<ConstScalar> make_const_f32(float v) noexcept {
    if (!is_portable_f32(v)) { return std::nullopt; }
    return ConstScalar{ConstScalar::Kind::F32, luisa::bit_cast<uint32_t>(v)};
}

[[nodiscard]] float const_as_f32(ConstScalar c) noexcept {
    return luisa::bit_cast<float>(c.bits);
}

[[nodiscard]] bool is_pure_inlineable_call(CallOp op) noexcept {
    if (op >= CallOp::ALL && op <= CallOp::INVERSE) { return true; }
    if (op >= CallOp::MAKE_BOOL2 && op <= CallOp::MAKE_FLOAT4X4) { return true; }
    return op == CallOp::ZERO || op == CallOp::ONE;
}

[[nodiscard]] bool is_readonly_load_call(CallOp op) noexcept {
    return op == CallOp::BUFFER_READ || op == CallOp::BYTE_BUFFER_READ ||
           op == CallOp::TEXTURE_READ ||
           op == CallOp::BINDLESS_BUFFER_READ || op == CallOp::BINDLESS_BYTE_BUFFER_READ ||
           op == CallOp::BINDLESS_TEXTURE2D_READ || op == CallOp::BINDLESS_TEXTURE3D_READ ||
           op == CallOp::TYPED_BINDLESS_BUFFER_READ || op == CallOp::TYPED_BINDLESS_BYTE_BUFFER_READ ||
           op == CallOp::TYPED_BINDLESS_TEXTURE2D_READ || op == CallOp::TYPED_BINDLESS_TEXTURE3D_READ;
}

[[nodiscard]] bool call_arg_is_readonly(CallExpr const *call, size_t arg_idx) noexcept {
    if (call->is_custom()) {
        auto formal_args = call->custom().arguments();
        return arg_idx < formal_args.size() &&
               formal_args[arg_idx].tag() != Variable::Tag::REFERENCE;
    }
    if (call->is_external()) { return false; }
    auto op = call->op();
    if (is_pure_inlineable_call(op)) { return true; }
    if (op >= CallOp::BUFFER_READ && op <= CallOp::TYPED_BINDLESS_BUFFER_ADDRESS) { return true; }
    return op == CallOp::BUFFER_VOLATILE_READ || op == CallOp::BUFFER_VOLATILE_WRITE ||
           op == CallOp::ASSERT || op == CallOp::ASSUME;
}

[[nodiscard]] bool call_may_mutate_any_local(CallExpr const *call) noexcept {
    for (size_t i = 0u; i < call->arguments().size(); ++i) {
        if (!call_arg_is_readonly(call, i)) { return true; }
    }
    return false;
}

[[nodiscard]] bool expr_has_mutating_call(Expression const *expr) noexcept;
[[nodiscard]] bool expression_has_call(Expression const *expr) noexcept;

[[nodiscard]] std::optional<ConstScalar> eval_const_scalar(
    Expression const *expr,
    luisa::unordered_map<uint32_t, ConstScalar> const &env) noexcept;

[[nodiscard]] std::optional<ConstScalar> eval_const_vector_lane(
    Expression const *vec_expr,
    uint32_t lane,
    luisa::unordered_map<uint32_t, ConstScalar> const &env) noexcept {
    if (vec_expr == nullptr || vec_expr->type() == nullptr || !vec_expr->type()->is_vector() ||
        lane >= vec_expr->type()->dimension()) {
        return std::nullopt;
    }
    switch (vec_expr->tag()) {
        case Expression::Tag::LITERAL: {
            auto *lit = static_cast<LiteralExpr const *>(vec_expr);
            return luisa::visit(
                [lane]<typename T>(T const &v) noexcept -> std::optional<ConstScalar> {
                    if constexpr (is_vector_v<T>) {
                        using Elem = vector_element_t<T>;
                        if (lane >= vector_dimension_v<T>) { return std::nullopt; }
                        auto elem = v[lane];
                        if constexpr (std::is_same_v<Elem, bool>) {
                            return ConstScalar{ConstScalar::Kind::BOOL, elem ? 1u : 0u};
                        } else if constexpr (std::is_same_v<Elem, int32_t>) {
                            return ConstScalar{ConstScalar::Kind::I32, static_cast<uint32_t>(elem)};
                        } else if constexpr (std::is_same_v<Elem, uint32_t>) {
                            return ConstScalar{ConstScalar::Kind::U32, elem};
                        } else if constexpr (std::is_same_v<Elem, float>) {
                            return make_const_f32(elem);
                        }
                    }
                    return std::nullopt;
                },
                lit->value());
        }
        case Expression::Tag::CALL: {
            auto *c = static_cast<CallExpr const *>(vec_expr);
            if (!c->is_builtin()) { return std::nullopt; }
            auto op = c->op();
            if (op == CallOp::ZERO || op == CallOp::ONE) {
                auto *elem = vec_expr->type()->element();
                bool is_one = (op == CallOp::ONE);
                if (elem->is_bool()) { return ConstScalar{ConstScalar::Kind::BOOL, is_one ? 1u : 0u}; }
                if (elem->is_int32()) { return ConstScalar{ConstScalar::Kind::I32, is_one ? 1u : 0u}; }
                if (elem->is_uint32()) { return ConstScalar{ConstScalar::Kind::U32, is_one ? 1u : 0u}; }
                if (elem->is_float32()) { return make_const_f32(is_one ? 1.0f : 0.0f); }
                return std::nullopt;
            }
            if (op >= CallOp::MAKE_BOOL2 && op <= CallOp::MAKE_FLOAT4) {
                auto args = c->arguments();
                if (args.size() == 1u && args[0]->type()->is_scalar()) {
                    return eval_const_scalar(args[0], env);
                }
                uint32_t cursor = 0u;
                for (auto *arg : args) {
                    if (arg->type() == nullptr) { return std::nullopt; }
                    if (arg->type()->is_scalar()) {
                        if (cursor == lane) { return eval_const_scalar(arg, env); }
                        ++cursor;
                    } else if (arg->type()->is_vector()) {
                        uint32_t d = static_cast<uint32_t>(arg->type()->dimension());
                        if (lane < cursor + d) {
                            return eval_const_vector_lane(arg, lane - cursor, env);
                        }
                        cursor += d;
                    } else {
                        return std::nullopt;
                    }
                }
            }
            return std::nullopt;
        }
        default:
            return std::nullopt;
    }
}

[[nodiscard]] std::optional<ConstScalar> eval_const_scalar(
    Expression const *expr,
    luisa::unordered_map<uint32_t, ConstScalar> const &env) noexcept {
    if (expr == nullptr || expr->type() == nullptr) { return std::nullopt; }
    switch (expr->tag()) {
        case Expression::Tag::LITERAL: {
            auto *lit = static_cast<LiteralExpr const *>(expr);
            return luisa::visit(
                []<typename T>(T const &v) noexcept -> std::optional<ConstScalar> {
                    if constexpr (std::is_same_v<T, bool>) {
                        return ConstScalar{ConstScalar::Kind::BOOL, v ? 1u : 0u};
                    } else if constexpr (std::is_same_v<T, int32_t>) {
                        return ConstScalar{ConstScalar::Kind::I32, static_cast<uint32_t>(v)};
                    } else if constexpr (std::is_same_v<T, uint32_t>) {
                        return ConstScalar{ConstScalar::Kind::U32, v};
                    } else if constexpr (std::is_same_v<T, float>) {
                        return make_const_f32(v);
                    } else {
                        return std::nullopt;
                    }
                },
                lit->value());
        }
        case Expression::Tag::REF: {
            auto v = static_cast<RefExpr const *>(expr)->variable();
            if (v.tag() != Variable::Tag::LOCAL) { return std::nullopt; }
            auto it = env.find(v.uid());
            return it != env.end() ? std::optional<ConstScalar>{it->second} : std::nullopt;
        }
        case Expression::Tag::MEMBER: {
            auto *m = static_cast<MemberExpr const *>(expr);
            if (m->is_swizzle() && m->swizzle_size() == 1u) {
                return eval_const_vector_lane(m->self(), m->swizzle_index(0u), env);
            }
            return std::nullopt;
        }
        case Expression::Tag::ACCESS: {
            auto *a = static_cast<AccessExpr const *>(expr);
            if (a->range()->type() != nullptr && a->range()->type()->is_vector()) {
                auto idx = eval_const_scalar(a->index(), env);
                if (idx && (idx->kind == ConstScalar::Kind::U32 ||
                            (idx->kind == ConstScalar::Kind::I32 && static_cast<int32_t>(idx->bits) >= 0))) {
                    return eval_const_vector_lane(a->range(), idx->bits, env);
                }
            }
            return std::nullopt;
        }
        case Expression::Tag::UNARY: {
            auto *u = static_cast<UnaryExpr const *>(expr);
            auto inner = eval_const_scalar(u->operand(), env);
            if (!inner) { return std::nullopt; }
            switch (u->op()) {
                case UnaryOp::PLUS: return inner;
                case UnaryOp::MINUS:
                    if (inner->kind == ConstScalar::Kind::BOOL) { return std::nullopt; }
                    if (inner->kind == ConstScalar::Kind::F32) {
                        return make_const_f32(-const_as_f32(*inner));
                    }
                    return ConstScalar{inner->kind, static_cast<uint32_t>(0u - inner->bits)};
                case UnaryOp::NOT:
                    if (inner->kind != ConstScalar::Kind::BOOL) { return std::nullopt; }
                    return ConstScalar{ConstScalar::Kind::BOOL, inner->bits == 0u ? 1u : 0u};
                case UnaryOp::BIT_NOT:
                    if (inner->kind == ConstScalar::Kind::F32) { return std::nullopt; }
                    if (inner->kind == ConstScalar::Kind::BOOL) {
                        return ConstScalar{ConstScalar::Kind::BOOL, inner->bits == 0u ? 1u : 0u};
                    }
                    return ConstScalar{inner->kind, ~inner->bits};
            }
            return std::nullopt;
        }
        case Expression::Tag::CAST: {
            auto *c = static_cast<CastExpr const *>(expr);
            auto inner = eval_const_scalar(c->expression(), env);
            if (!inner) { return std::nullopt; }
            auto *dst = c->type();
            if (c->op() == CastOp::BITWISE) {
                if (inner->kind == ConstScalar::Kind::BOOL) { return std::nullopt; }
                if (dst->is_int32()) { return ConstScalar{ConstScalar::Kind::I32, inner->bits}; }
                if (dst->is_uint32()) { return ConstScalar{ConstScalar::Kind::U32, inner->bits}; }
                if (dst->is_float32()) { return make_const_f32(luisa::bit_cast<float>(inner->bits)); }
                return std::nullopt;
            }
            if (c->op() != CastOp::STATIC) { return std::nullopt; }
            if (dst->is_bool()) {
                if (inner->kind == ConstScalar::Kind::F32) {
                    return ConstScalar{ConstScalar::Kind::BOOL, const_as_f32(*inner) != 0.0f ? 1u : 0u};
                }
                return ConstScalar{ConstScalar::Kind::BOOL, inner->bits != 0u ? 1u : 0u};
            }
            if (dst->is_int32()) {
                if (inner->kind == ConstScalar::Kind::F32) {
                    float f = const_as_f32(*inner);
                    if (f >= -2147483648.0f && f < 2147483648.0f) {
                        return ConstScalar{ConstScalar::Kind::I32, static_cast<uint32_t>(static_cast<int32_t>(f))};
                    }
                    return std::nullopt;
                }
                return ConstScalar{ConstScalar::Kind::I32, inner->bits};
            }
            if (dst->is_uint32()) {
                if (inner->kind == ConstScalar::Kind::F32) {
                    float f = const_as_f32(*inner);
                    if (f >= 0.0f && f < 4294967296.0f) {
                        return ConstScalar{ConstScalar::Kind::U32, static_cast<uint32_t>(f)};
                    }
                    return std::nullopt;
                }
                return ConstScalar{ConstScalar::Kind::U32, inner->bits};
            }
            if (dst->is_float32()) {
                switch (inner->kind) {
                    case ConstScalar::Kind::BOOL: return make_const_f32(inner->bits != 0u ? 1.0f : 0.0f);
                    case ConstScalar::Kind::I32: return make_const_f32(static_cast<float>(static_cast<int32_t>(inner->bits)));
                    case ConstScalar::Kind::U32: return make_const_f32(static_cast<float>(inner->bits));
                    case ConstScalar::Kind::F32: return inner;
                }
            }
            return std::nullopt;
        }
        case Expression::Tag::BINARY: {
            auto *b = static_cast<BinaryExpr const *>(expr);
            auto op = b->op();
            auto lhs = eval_const_scalar(b->lhs(), env);
            // Short-circuit / absorbing boolean folding when pure.
            if (b->type()->is_bool() &&
                (op == BinaryOp::AND || op == BinaryOp::BIT_AND)) {
                if (lhs && lhs->kind == ConstScalar::Kind::BOOL && lhs->bits == 0u &&
                    !expr_has_mutating_call(b->rhs())) {
                    return ConstScalar{ConstScalar::Kind::BOOL, 0u};
                }
                auto rhs = eval_const_scalar(b->rhs(), env);
                if (rhs && rhs->kind == ConstScalar::Kind::BOOL && rhs->bits == 0u &&
                    !expr_has_mutating_call(b->lhs())) {
                    return ConstScalar{ConstScalar::Kind::BOOL, 0u};
                }
                if (lhs && rhs && lhs->kind == ConstScalar::Kind::BOOL && rhs->kind == ConstScalar::Kind::BOOL) {
                    return ConstScalar{ConstScalar::Kind::BOOL, (lhs->bits != 0u && rhs->bits != 0u) ? 1u : 0u};
                }
                return std::nullopt;
            }
            if (b->type()->is_bool() &&
                (op == BinaryOp::OR || op == BinaryOp::BIT_OR)) {
                if (lhs && lhs->kind == ConstScalar::Kind::BOOL && lhs->bits != 0u &&
                    !expr_has_mutating_call(b->rhs())) {
                    return ConstScalar{ConstScalar::Kind::BOOL, 1u};
                }
                auto rhs = eval_const_scalar(b->rhs(), env);
                if (rhs && rhs->kind == ConstScalar::Kind::BOOL && rhs->bits != 0u &&
                    !expr_has_mutating_call(b->lhs())) {
                    return ConstScalar{ConstScalar::Kind::BOOL, 1u};
                }
                if (lhs && rhs && lhs->kind == ConstScalar::Kind::BOOL && rhs->kind == ConstScalar::Kind::BOOL) {
                    return ConstScalar{ConstScalar::Kind::BOOL, (lhs->bits != 0u || rhs->bits != 0u) ? 1u : 0u};
                }
                return std::nullopt;
            }
            auto rhs = eval_const_scalar(b->rhs(), env);
            if (!lhs || !rhs || lhs->kind != rhs->kind) { return std::nullopt; }
            auto k = lhs->kind;
            if (k == ConstScalar::Kind::F32) {
                float fa = const_as_f32(*lhs);
                float fb = const_as_f32(*rhs);
                switch (op) {
                    case BinaryOp::EQUAL: return ConstScalar{ConstScalar::Kind::BOOL, fa == fb ? 1u : 0u};
                    case BinaryOp::NOT_EQUAL: return ConstScalar{ConstScalar::Kind::BOOL, fa != fb ? 1u : 0u};
                    case BinaryOp::LESS: return ConstScalar{ConstScalar::Kind::BOOL, fa < fb ? 1u : 0u};
                    case BinaryOp::LESS_EQUAL: return ConstScalar{ConstScalar::Kind::BOOL, fa <= fb ? 1u : 0u};
                    case BinaryOp::GREATER: return ConstScalar{ConstScalar::Kind::BOOL, fa > fb ? 1u : 0u};
                    case BinaryOp::GREATER_EQUAL: return ConstScalar{ConstScalar::Kind::BOOL, fa >= fb ? 1u : 0u};
                    case BinaryOp::ADD: return make_const_f32(fa + fb);
                    case BinaryOp::SUB: return make_const_f32(fa - fb);
                    case BinaryOp::MUL: return make_const_f32(fa * fb);
                    case BinaryOp::DIV: return fb != 0.0f ? make_const_f32(fa / fb) : std::nullopt;
                    default: return std::nullopt;
                }
            }
            switch (op) {
                case BinaryOp::EQUAL:
                    return ConstScalar{ConstScalar::Kind::BOOL, lhs->bits == rhs->bits ? 1u : 0u};
                case BinaryOp::NOT_EQUAL:
                    return ConstScalar{ConstScalar::Kind::BOOL, lhs->bits != rhs->bits ? 1u : 0u};
                case BinaryOp::BIT_AND:
                    return ConstScalar{k, lhs->bits & rhs->bits};
                case BinaryOp::BIT_OR:
                    return ConstScalar{k, lhs->bits | rhs->bits};
                case BinaryOp::BIT_XOR:
                    return ConstScalar{k, lhs->bits ^ rhs->bits};
                case BinaryOp::SHL:
                    if (k == ConstScalar::Kind::BOOL || rhs->bits >= 32u) { return std::nullopt; }
                    return ConstScalar{k, lhs->bits << rhs->bits};
                case BinaryOp::SHR:
                    if (k == ConstScalar::Kind::BOOL || rhs->bits >= 32u) { return std::nullopt; }
                    if (k == ConstScalar::Kind::I32) {
                        return ConstScalar{k, static_cast<uint32_t>(static_cast<int32_t>(lhs->bits) >> rhs->bits)};
                    }
                    return ConstScalar{k, lhs->bits >> rhs->bits};
                case BinaryOp::LESS:
                    if (k == ConstScalar::Kind::I32) {
                        return ConstScalar{ConstScalar::Kind::BOOL, static_cast<int32_t>(lhs->bits) < static_cast<int32_t>(rhs->bits) ? 1u : 0u};
                    }
                    if (k == ConstScalar::Kind::U32) {
                        return ConstScalar{ConstScalar::Kind::BOOL, lhs->bits < rhs->bits ? 1u : 0u};
                    }
                    return std::nullopt;
                case BinaryOp::LESS_EQUAL:
                    if (k == ConstScalar::Kind::I32) {
                        return ConstScalar{ConstScalar::Kind::BOOL, static_cast<int32_t>(lhs->bits) <= static_cast<int32_t>(rhs->bits) ? 1u : 0u};
                    }
                    if (k == ConstScalar::Kind::U32) {
                        return ConstScalar{ConstScalar::Kind::BOOL, lhs->bits <= rhs->bits ? 1u : 0u};
                    }
                    return std::nullopt;
                case BinaryOp::GREATER:
                    if (k == ConstScalar::Kind::I32) {
                        return ConstScalar{ConstScalar::Kind::BOOL, static_cast<int32_t>(lhs->bits) > static_cast<int32_t>(rhs->bits) ? 1u : 0u};
                    }
                    if (k == ConstScalar::Kind::U32) {
                        return ConstScalar{ConstScalar::Kind::BOOL, lhs->bits > rhs->bits ? 1u : 0u};
                    }
                    return std::nullopt;
                case BinaryOp::GREATER_EQUAL:
                    if (k == ConstScalar::Kind::I32) {
                        return ConstScalar{ConstScalar::Kind::BOOL, static_cast<int32_t>(lhs->bits) >= static_cast<int32_t>(rhs->bits) ? 1u : 0u};
                    }
                    if (k == ConstScalar::Kind::U32) {
                        return ConstScalar{ConstScalar::Kind::BOOL, lhs->bits >= rhs->bits ? 1u : 0u};
                    }
                    return std::nullopt;
                case BinaryOp::ADD:
                    if (k == ConstScalar::Kind::BOOL) { return std::nullopt; }
                    return ConstScalar{k, static_cast<uint32_t>(lhs->bits + rhs->bits)};
                case BinaryOp::SUB:
                    if (k == ConstScalar::Kind::BOOL) { return std::nullopt; }
                    return ConstScalar{k, static_cast<uint32_t>(lhs->bits - rhs->bits)};
                case BinaryOp::MUL:
                    if (k == ConstScalar::Kind::BOOL) { return std::nullopt; }
                    return ConstScalar{k, static_cast<uint32_t>(lhs->bits * rhs->bits)};
                case BinaryOp::DIV:
                    if (k == ConstScalar::Kind::U32 && rhs->bits != 0u) {
                        return ConstScalar{k, lhs->bits / rhs->bits};
                    }
                    if (k == ConstScalar::Kind::I32) {
                        auto a = static_cast<int32_t>(lhs->bits);
                        auto bv = static_cast<int32_t>(rhs->bits);
                        if (bv != 0 && !(a == INT32_MIN && bv == -1)) {
                            return ConstScalar{k, static_cast<uint32_t>(a / bv)};
                        }
                    }
                    return std::nullopt;
                case BinaryOp::MOD:
                    if (k == ConstScalar::Kind::U32 && rhs->bits != 0u) {
                        return ConstScalar{k, lhs->bits % rhs->bits};
                    }
                    if (k == ConstScalar::Kind::I32) {
                        auto a = static_cast<int32_t>(lhs->bits);
                        auto bv = static_cast<int32_t>(rhs->bits);
                        if (bv != 0 && !(a == INT32_MIN && bv == -1)) {
                            return ConstScalar{k, static_cast<uint32_t>(a % bv)};
                        }
                    }
                    return std::nullopt;
                default:
                    return std::nullopt;
            }
        }
        case Expression::Tag::CALL: {
            auto *c = static_cast<CallExpr const *>(expr);
            if (!c->is_builtin()) { return std::nullopt; }
            auto args = c->arguments();
            auto op = c->op();
            if (op == CallOp::ZERO || op == CallOp::ONE) {
                bool is_one = (op == CallOp::ONE);
                if (expr->type()->is_bool()) { return ConstScalar{ConstScalar::Kind::BOOL, is_one ? 1u : 0u}; }
                if (expr->type()->is_int32()) { return ConstScalar{ConstScalar::Kind::I32, is_one ? 1u : 0u}; }
                if (expr->type()->is_uint32()) { return ConstScalar{ConstScalar::Kind::U32, is_one ? 1u : 0u}; }
                if (expr->type()->is_float32()) { return make_const_f32(is_one ? 1.0f : 0.0f); }
                return std::nullopt;
            }
            if (op == CallOp::SELECT && args.size() == 3u) {
                auto pred = eval_const_scalar(args[2], env);
                if (pred && pred->kind == ConstScalar::Kind::BOOL) {
                    auto *taken = pred->bits != 0u ? args[1] : args[0];
                    auto *discarded = pred->bits != 0u ? args[0] : args[1];
                    if (!expr_has_mutating_call(discarded)) {
                        return eval_const_scalar(taken, env);
                    }
                }
                return std::nullopt;
            }
            if (op == CallOp::ABS && args.size() == 1u) {
                auto a = eval_const_scalar(args[0], env);
                if (!a) { return std::nullopt; }
                if (a->kind == ConstScalar::Kind::F32) { return make_const_f32(std::abs(const_as_f32(*a))); }
                if (a->kind == ConstScalar::Kind::I32) {
                    auto iv = static_cast<int32_t>(a->bits);
                    if (iv != INT32_MIN) { return ConstScalar{ConstScalar::Kind::I32, static_cast<uint32_t>(iv < 0 ? -iv : iv)}; }
                }
                return std::nullopt;
            }
            if ((op == CallOp::MIN || op == CallOp::MAX) && args.size() == 2u) {
                auto a = eval_const_scalar(args[0], env);
                auto b = eval_const_scalar(args[1], env);
                if (!a || !b || a->kind != b->kind) { return std::nullopt; }
                bool is_min = (op == CallOp::MIN);
                if (a->kind == ConstScalar::Kind::F32) {
                    float fa = const_as_f32(*a), fb = const_as_f32(*b);
                    return make_const_f32(is_min ? std::min(fa, fb) : std::max(fa, fb));
                }
                if (a->kind == ConstScalar::Kind::I32) {
                    auto ia = static_cast<int32_t>(a->bits), ib = static_cast<int32_t>(b->bits);
                    return ConstScalar{ConstScalar::Kind::I32, static_cast<uint32_t>(is_min ? std::min(ia, ib) : std::max(ia, ib))};
                }
                if (a->kind == ConstScalar::Kind::U32) {
                    return ConstScalar{ConstScalar::Kind::U32, is_min ? std::min(a->bits, b->bits) : std::max(a->bits, b->bits)};
                }
                return std::nullopt;
            }
            if (op == CallOp::CLAMP && args.size() == 3u) {
                auto v = eval_const_scalar(args[0], env);
                auto lo = eval_const_scalar(args[1], env);
                auto hi = eval_const_scalar(args[2], env);
                if (!v || !lo || !hi || v->kind != lo->kind || v->kind != hi->kind) { return std::nullopt; }
                if (v->kind == ConstScalar::Kind::F32) {
                    return make_const_f32(std::min(std::max(const_as_f32(*v), const_as_f32(*lo)), const_as_f32(*hi)));
                }
                if (v->kind == ConstScalar::Kind::I32) {
                    auto iv = static_cast<int32_t>(v->bits), ilo = static_cast<int32_t>(lo->bits), ihi = static_cast<int32_t>(hi->bits);
                    return ConstScalar{ConstScalar::Kind::I32, static_cast<uint32_t>(std::min(std::max(iv, ilo), ihi))};
                }
                if (v->kind == ConstScalar::Kind::U32) {
                    return ConstScalar{ConstScalar::Kind::U32, std::min(std::max(v->bits, lo->bits), hi->bits)};
                }
                return std::nullopt;
            }
            if (op == CallOp::SATURATE && args.size() == 1u) {
                auto v = eval_const_scalar(args[0], env);
                if (v && v->kind == ConstScalar::Kind::F32) {
                    return make_const_f32(std::min(std::max(const_as_f32(*v), 0.0f), 1.0f));
                }
                return std::nullopt;
            }
            if (op == CallOp::FMA && args.size() == 3u) {
                auto a = eval_const_scalar(args[0], env);
                auto b = eval_const_scalar(args[1], env);
                auto cc = eval_const_scalar(args[2], env);
                if (a && b && cc && a->kind == ConstScalar::Kind::F32 &&
                    b->kind == ConstScalar::Kind::F32 && cc->kind == ConstScalar::Kind::F32) {
                    return make_const_f32(std::fma(const_as_f32(*a), const_as_f32(*b), const_as_f32(*cc)));
                }
                return std::nullopt;
            }
            if ((op == CallOp::FLOOR || op == CallOp::CEIL || op == CallOp::TRUNC) && args.size() == 1u) {
                auto v = eval_const_scalar(args[0], env);
                if (v && v->kind == ConstScalar::Kind::F32) {
                    float fv = const_as_f32(*v);
                    if (op == CallOp::FLOOR) { return make_const_f32(std::floor(fv)); }
                    if (op == CallOp::CEIL) { return make_const_f32(std::ceil(fv)); }
                    return make_const_f32(std::trunc(fv));
                }
                return std::nullopt;
            }
            if (op == CallOp::COPYSIGN && args.size() == 2u) {
                auto a = eval_const_scalar(args[0], env);
                auto b = eval_const_scalar(args[1], env);
                if (a && b && a->kind == ConstScalar::Kind::F32 && b->kind == ConstScalar::Kind::F32) {
                    return make_const_f32(std::copysign(const_as_f32(*a), const_as_f32(*b)));
                }
                return std::nullopt;
            }
            return std::nullopt;
        }
        default:
            return std::nullopt;
    }
}

[[nodiscard]] bool scope_has_non_trailing_break(ScopeStmt const *body, bool allow_trailing_at_top) noexcept {
    auto stmts = body->statements();
    for (size_t i = 0u; i < stmts.size(); ++i) {
        auto *s = stmts[i];
        switch (s->tag()) {
            case Statement::Tag::BREAK:
                if (allow_trailing_at_top && i + 1u == stmts.size()) {
                    return false;
                }
                return true;
            case Statement::Tag::CONTINUE:
            case Statement::Tag::RETURN:
                return false;
            case Statement::Tag::IF: {
                auto *ifs = static_cast<IfStmt const *>(s);
                if (scope_has_non_trailing_break(ifs->true_branch(), false) ||
                    scope_has_non_trailing_break(ifs->false_branch(), false)) {
                    return true;
                }
                break;
            }
            case Statement::Tag::SCOPE:
                if (scope_has_non_trailing_break(static_cast<ScopeStmt const *>(s), false)) {
                    return true;
                }
                break;
            case Statement::Tag::AUTO_DIFF:
                if (scope_has_non_trailing_break(static_cast<AutoDiffStmt const *>(s)->body(), false)) {
                    return true;
                }
                break;
            default:
                // Nested LOOP / FOR / SWITCH / RAY_QUERY trap their own breaks.
                break;
        }
    }
    return false;
}

[[nodiscard]] std::optional<Variable> lvalue_root_variable(Expression const *expr) noexcept {
    while (expr != nullptr) {
        switch (expr->tag()) {
            case Expression::Tag::REF:
                return static_cast<RefExpr const *>(expr)->variable();
            case Expression::Tag::MEMBER:
                expr = static_cast<MemberExpr const *>(expr)->self();
                break;
            case Expression::Tag::ACCESS:
                expr = static_cast<AccessExpr const *>(expr)->range();
                break;
            default:
                return std::nullopt;
        }
    }
    return std::nullopt;
}

[[nodiscard]] bool is_pure_inlineable_expr(Expression const *expr) noexcept;

[[nodiscard]] std::optional<uint32_t> extract_const_uint_literal(Expression const *expr) noexcept {
    if (expr == nullptr || expr->tag() != Expression::Tag::LITERAL) { return std::nullopt; }
    auto *lit = static_cast<LiteralExpr const *>(expr);
    return luisa::visit(
        []<typename T>(T const &v) noexcept -> std::optional<uint32_t> {
            if constexpr (std::is_same_v<T, uint32_t>) {
                return v;
            } else if constexpr (std::is_same_v<T, int32_t>) {
                return v >= 0 ? std::optional<uint32_t>{static_cast<uint32_t>(v)} : std::nullopt;
            }
            return std::nullopt;
        },
        lit->value());
}

[[nodiscard]] bool is_pure_vector_ctor(Expression const *expr) noexcept {
    if (expr == nullptr || expr->tag() != Expression::Tag::CALL) { return false; }
    auto *c = static_cast<CallExpr const *>(expr);
    return c->is_builtin() && c->op() >= CallOp::MAKE_BOOL2 && c->op() <= CallOp::MAKE_FLOAT4;
}

[[nodiscard]] Expression const *extract_pure_vector_lane(
    Expression const *vec_expr, uint32_t lane, Type const *expected_type,
    luisa::unordered_map<uint32_t, Expression const *> const *inline_exprs = nullptr) noexcept {
    for (int hop = 0; hop < 8 && vec_expr != nullptr &&
                      vec_expr->tag() == Expression::Tag::REF && inline_exprs != nullptr;
         ++hop) {
        auto v = static_cast<RefExpr const *>(vec_expr)->variable();
        if (v.tag() != Variable::Tag::LOCAL) { break; }
        auto it = inline_exprs->find(v.uid());
        if (it == inline_exprs->end()) { break; }
        vec_expr = it->second;
    }
    if (vec_expr == nullptr || vec_expr->type() == nullptr || expected_type == nullptr ||
        !vec_expr->type()->is_vector() || lane >= vec_expr->type()->dimension() ||
        vec_expr->tag() != Expression::Tag::CALL) {
        return nullptr;
    }
    auto *c = static_cast<CallExpr const *>(vec_expr);
    if (!c->is_builtin()) { return nullptr; }
    auto op = c->op();
    if (op < CallOp::MAKE_BOOL2 || op > CallOp::MAKE_FLOAT4) { return nullptr; }
    auto args = c->arguments();
    for (auto *arg : args) {
        if (!is_pure_inlineable_expr(arg)) { return nullptr; }
    }
    if (args.size() == 1u && args[0]->type() != nullptr && args[0]->type()->is_scalar()) {
        return *args[0]->type() == *expected_type ? args[0] : nullptr;
    }
    uint32_t cursor = 0u;
    for (auto *arg : args) {
        if (arg->type() == nullptr) { return nullptr; }
        if (arg->type()->is_scalar()) {
            if (cursor == lane) {
                return *arg->type() == *expected_type ? arg : nullptr;
            }
            ++cursor;
        } else if (arg->type()->is_vector()) {
            uint32_t d = static_cast<uint32_t>(arg->type()->dimension());
            if (lane < cursor + d) {
                return extract_pure_vector_lane(arg, lane - cursor, expected_type, inline_exprs);
            }
            cursor += d;
        } else {
            return nullptr;
        }
    }
    return nullptr;
}

[[nodiscard]] Expression const *unwrap_pure_vector_lane(
    Expression const *expr,
    luisa::unordered_map<uint32_t, Expression const *> const *inline_exprs = nullptr) noexcept {
    for (int depth = 0; depth < 8 && expr != nullptr; ++depth) {
        Expression const *next = nullptr;
        if (expr->tag() == Expression::Tag::ACCESS) {
            auto *a = static_cast<AccessExpr const *>(expr);
            if (a->range()->type() != nullptr && a->range()->type()->is_vector()) {
                if (auto lane = extract_const_uint_literal(a->index())) {
                    next = extract_pure_vector_lane(a->range(), *lane, expr->type(), inline_exprs);
                }
            }
        } else if (expr->tag() == Expression::Tag::MEMBER) {
            auto *m = static_cast<MemberExpr const *>(expr);
            if (m->is_swizzle() && m->swizzle_size() == 1u) {
                next = extract_pure_vector_lane(m->self(), m->swizzle_index(0u), expr->type(), inline_exprs);
            }
        }
        if (next == nullptr) { break; }
        expr = next;
    }
    return expr;
}

[[nodiscard]] bool is_pure_inlineable_expr(Expression const *expr) noexcept {
    if (expr == nullptr) { return false; }
    switch (expr->tag()) {
        case Expression::Tag::LITERAL:
        case Expression::Tag::CONSTANT:
            return true;
        case Expression::Tag::REF: {
            auto tag = static_cast<RefExpr const *>(expr)->variable().tag();
            return tag == Variable::Tag::LOCAL || tag == Variable::Tag::REFERENCE ||
                   tag == Variable::Tag::THREAD_ID || tag == Variable::Tag::BLOCK_ID ||
                   tag == Variable::Tag::DISPATCH_ID || tag == Variable::Tag::DISPATCH_SIZE ||
                   tag == Variable::Tag::KERNEL_ID || tag == Variable::Tag::WARP_LANE_COUNT ||
                   tag == Variable::Tag::WARP_LANE_ID;
        }
        case Expression::Tag::UNARY:
            return is_pure_inlineable_expr(static_cast<UnaryExpr const *>(expr)->operand());
        case Expression::Tag::BINARY: {
            auto *b = static_cast<BinaryExpr const *>(expr);
            return is_pure_inlineable_expr(b->lhs()) && is_pure_inlineable_expr(b->rhs());
        }
        case Expression::Tag::CAST:
            return is_pure_inlineable_expr(static_cast<CastExpr const *>(expr)->expression());
        case Expression::Tag::MEMBER:
            return is_pure_inlineable_expr(static_cast<MemberExpr const *>(expr)->self());
        case Expression::Tag::ACCESS: {
            auto *a = static_cast<AccessExpr const *>(expr);
            return is_pure_inlineable_expr(a->range()) && is_pure_inlineable_expr(a->index());
        }
        case Expression::Tag::CALL: {
            auto *c = static_cast<CallExpr const *>(expr);
            if (!c->is_builtin() || !is_pure_inlineable_call(c->op())) { return false; }
            for (auto *arg : c->arguments()) {
                if (!is_pure_inlineable_expr(arg)) { return false; }
            }
            return true;
        }
        default:
            return false;
    }
}

[[nodiscard]] bool expr_has_mutating_call(Expression const *expr) noexcept {
    if (expr == nullptr) { return false; }
    switch (expr->tag()) {
        case Expression::Tag::UNARY:
            return expr_has_mutating_call(static_cast<UnaryExpr const *>(expr)->operand());
        case Expression::Tag::BINARY: {
            auto *b = static_cast<BinaryExpr const *>(expr);
            return expr_has_mutating_call(b->lhs()) || expr_has_mutating_call(b->rhs());
        }
        case Expression::Tag::CAST:
            return expr_has_mutating_call(static_cast<CastExpr const *>(expr)->expression());
        case Expression::Tag::MEMBER:
            return expr_has_mutating_call(static_cast<MemberExpr const *>(expr)->self());
        case Expression::Tag::ACCESS: {
            auto *a = static_cast<AccessExpr const *>(expr);
            return expr_has_mutating_call(a->range()) || expr_has_mutating_call(a->index());
        }
        case Expression::Tag::CALL: {
            auto *c = static_cast<CallExpr const *>(expr);
            if (call_may_mutate_any_local(c)) { return true; }
            for (auto *arg : c->arguments()) {
                if (expr_has_mutating_call(arg)) { return true; }
            }
            return false;
        }
        default:
            return false;
    }
}

[[nodiscard]] LocalEmitPlan build_local_emit_plan(
    Function func,
    CodegenUtility *util,
    luisa::unordered_set<uint32_t> const &lazy_decl_uids) noexcept {
    LocalEmitPlan plan;

    struct ScopeMeta {
        const ScopeStmt *parent{nullptr};
        uint32_t depth{0u};
        uint32_t loop_depth{0u};
        bool can_declare{true};
    };
    luisa::unordered_map<const ScopeStmt *, ScopeMeta> scope_meta;

    struct VarInfo {
        Variable var;
        bool eligible{false};
        bool partial_or_ref{false};
        bool used_as_member_or_access_base{false};
        bool used_as_non_lane_base{false};
        bool read_before_def{false};
        bool read_in_nested_loop{false};
        uint32_t write_count{0u};
        uint32_t read_count{0u};
        const AssignStmt *def_stmt{nullptr};
        const ScopeStmt *def_scope{nullptr};
        size_t def_stmt_idx{0u};
        size_t def_order_seq{SIZE_MAX};
        uint32_t def_loop_depth{0u};
        const ScopeStmt *single_read_scope{nullptr};
        size_t single_read_stmt_idx{0u};
        bool single_read_is_direct_top_expr{false};
        const ScopeStmt *lca_scope{nullptr};
    };
    luisa::unordered_map<uint32_t, VarInfo> vars;
    luisa::vector<uint32_t> local_order;
    local_order.reserve(func.local_variables().size());
    vars.reserve(func.local_variables().size());
    for (auto &&v : func.local_variables()) {
        if (func.variable_usage(v.uid()) == Usage::NONE) { continue; }
        bool eligible = !lazy_decl_uids.contains(v.uid()) &&
                        !(v.type()->is_resource() || v.type()->is_custom() ||
                          v.type()->is_cooperative_vector());
        vars.emplace(v.uid(), VarInfo{.var = v, .eligible = eligible});
        local_order.emplace_back(v.uid());
    }

    auto compute_lca = [&](const ScopeStmt *a, const ScopeStmt *b) noexcept -> const ScopeStmt * {
        if (a == nullptr) { return b; }
        if (b == nullptr) { return a; }
        auto da = scope_meta.at(a).depth;
        auto db = scope_meta.at(b).depth;
        while (da > db) {
            a = scope_meta.at(a).parent;
            --da;
        }
        while (db > da) {
            b = scope_meta.at(b).parent;
            --db;
        }
        while (a != b) {
            a = scope_meta.at(a).parent;
            b = scope_meta.at(b).parent;
        }
        while (a != nullptr && !scope_meta.at(a).can_declare) {
            a = scope_meta.at(a).parent;
        }
        return a != nullptr ? a : func.body();
    };

    const ScopeStmt *cur_scope = nullptr;
    size_t cur_stmt_idx = 0u;
    size_t next_def_order_seq = 0u;
    uint32_t cur_loop_depth = 0u;
    luisa::unordered_map<uint32_t, ConstScalar> const_env;
    luisa::vector<uint32_t> *branch_written_uids = nullptr;

    auto cur_decl_scope = [&]() noexcept -> const ScopeStmt * {
        auto *s = cur_scope;
        while (s != nullptr && !scope_meta.at(s).can_declare) {
            s = scope_meta.at(s).parent;
        }
        return s != nullptr ? s : func.body();
    };

    auto touch_lca = [&](VarInfo &vi) noexcept {
        vi.lca_scope = compute_lca(vi.lca_scope, cur_decl_scope());
    };

    auto record_env_write = [&](uint32_t uid, std::optional<ConstScalar> val) noexcept {
        if (branch_written_uids != nullptr) {
            branch_written_uids->emplace_back(uid);
        }
        if (val) {
            const_env[uid] = *val;
        } else {
            const_env.erase(uid);
        }
    };

    auto mark_read = [&](uint32_t uid, bool direct_top_expr, bool as_base, bool non_lane_base = false) noexcept {
        auto it = vars.find(uid);
        if (it == vars.end()) { return; }
        auto &vi = it->second;
        touch_lca(vi);
        if (as_base) {
            vi.used_as_member_or_access_base = true;
            if (non_lane_base) { vi.used_as_non_lane_base = true; }
        }
        if (vi.write_count == 0u) { vi.read_before_def = true; }
        if (vi.write_count == 1u && cur_loop_depth > vi.def_loop_depth) {
            vi.read_in_nested_loop = true;
        }
        ++vi.read_count;
        if (vi.read_count == 1u) {
            vi.single_read_scope = cur_scope;
            vi.single_read_stmt_idx = cur_stmt_idx;
            vi.single_read_is_direct_top_expr = direct_top_expr;
        } else {
            vi.single_read_scope = nullptr;
            vi.single_read_is_direct_top_expr = false;
        }
    };

    auto mark_partial_or_ref_write = [&](uint32_t uid) noexcept {
        record_env_write(uid, std::nullopt);
        auto it = vars.find(uid);
        if (it == vars.end()) { return; }
        auto &vi = it->second;
        touch_lca(vi);
        vi.partial_or_ref = true;
        ++vi.write_count;
    };

    auto scan_expr = [&](auto &&self, Expression const *expr, bool direct_top_expr, bool as_base, bool non_lane_base = false) noexcept -> void {
        if (expr == nullptr) { return; }
        if (auto *unwrapped = unwrap_pure_vector_lane(expr); unwrapped != expr) {
            self(self, unwrapped, direct_top_expr, as_base, non_lane_base);
            return;
        }
        switch (expr->tag()) {
            case Expression::Tag::REF: {
                auto v = static_cast<RefExpr const *>(expr)->variable();
                if (v.tag() == Variable::Tag::LOCAL) {
                    mark_read(v.uid(), direct_top_expr, as_base, non_lane_base);
                }
                break;
            }
            case Expression::Tag::UNARY:
                self(self, static_cast<UnaryExpr const *>(expr)->operand(), direct_top_expr, false);
                break;
            case Expression::Tag::BINARY: {
                auto *b = static_cast<BinaryExpr const *>(expr);
                self(self, b->lhs(), direct_top_expr, false);
                self(self, b->rhs(), direct_top_expr, false);
                break;
            }
            case Expression::Tag::CAST:
                self(self, static_cast<CastExpr const *>(expr)->expression(), direct_top_expr, false);
                break;
            case Expression::Tag::MEMBER: {
                auto *m = static_cast<MemberExpr const *>(expr);
                const bool is_const_vec_lane =
                    m->is_swizzle() && m->swizzle_size() == 1u &&
                    m->self()->type() != nullptr && m->self()->type()->is_vector();
                self(self, m->self(), direct_top_expr, true, !is_const_vec_lane);
                break;
            }
            case Expression::Tag::ACCESS: {
                auto *a = static_cast<AccessExpr const *>(expr);
                const bool is_const_vec_lane =
                    a->range()->type() != nullptr && a->range()->type()->is_vector() &&
                    extract_const_uint_literal(a->index()).has_value();
                self(self, a->range(), direct_top_expr, true, !is_const_vec_lane);
                self(self, a->index(), direct_top_expr, false);
                break;
            }
            case Expression::Tag::CALL: {
                auto *c = static_cast<CallExpr const *>(expr);
                auto args = c->arguments();
                for (size_t i = 0u; i < args.size(); ++i) {
                    if (call_arg_is_readonly(c, i)) {
                        self(self, args[i], direct_top_expr, false);
                    } else {
                        if (auto root = lvalue_root_variable(args[i]);
                            root && root->tag() == Variable::Tag::LOCAL) {
                            mark_partial_or_ref_write(root->uid());
                            mark_read(root->uid(), false, false);
                        }
                        self(self, args[i], false, false);
                    }
                }
                break;
            }
            default:
                break;
        }
    };

    auto scan_lvalue_indices = [&](auto &&self, Expression const *expr) noexcept -> void {
        if (expr == nullptr) { return; }
        switch (expr->tag()) {
            case Expression::Tag::MEMBER:
                self(self, static_cast<MemberExpr const *>(expr)->self());
                break;
            case Expression::Tag::ACCESS: {
                auto *a = static_cast<AccessExpr const *>(expr);
                self(self, a->range());
                scan_expr(scan_expr, a->index(), true, false);
                break;
            }
            default:
                break;
        }
    };

    auto scan_scope = [&](auto &&self, const ScopeStmt *scope, const ScopeStmt *parent,
                          uint32_t depth, uint32_t loop_depth, bool can_declare) noexcept -> void {
        scope_meta.emplace(scope, ScopeMeta{parent, depth, loop_depth, can_declare});
        auto prev_scope = cur_scope;
        auto prev_stmt_idx = cur_stmt_idx;
        auto prev_loop_depth = cur_loop_depth;
        cur_scope = scope;
        cur_loop_depth = loop_depth;

        auto stmts = scope->statements();
        for (size_t idx = 0u; idx < stmts.size(); ++idx) {
            cur_stmt_idx = idx;
            auto *s = stmts[idx];
            switch (s->tag()) {
                case Statement::Tag::BREAK:
                case Statement::Tag::CONTINUE:
                    goto done_scope;
                case Statement::Tag::RETURN: {
                    auto *ret = static_cast<ReturnStmt const *>(s);
                    scan_expr(scan_expr, ret->expression(), true, false);
                    goto done_scope;
                }
                case Statement::Tag::EXPR:
                    scan_expr(scan_expr, static_cast<ExprStmt const *>(s)->expression(), true, false);
                    break;
                case Statement::Tag::PRINT: {
                    auto *p = static_cast<PrintStmt const *>(s);
                    for (auto *arg : p->arguments()) {
                        scan_expr(scan_expr, arg, true, false);
                    }
                    break;
                }
                case Statement::Tag::ASSIGN: {
                    auto *assign = static_cast<AssignStmt const *>(s);
                    if (is_local_undefined_lifetime_seed(assign)) { break; }
                    // Scan RHS before marking LHS written so `x = x + 1` records read_before_def.
                    auto rhs_const = eval_const_scalar(assign->rhs(), const_env);
                    scan_expr(scan_expr, assign->rhs(), true, false);
                    if (assign->lhs()->tag() == Expression::Tag::REF) {
                        auto v = static_cast<RefExpr const *>(assign->lhs())->variable();
                        if (v.tag() == Variable::Tag::LOCAL) {
                            auto it = vars.find(v.uid());
                            if (it != vars.end()) {
                                auto &vi = it->second;
                                touch_lca(vi);
                                bool rhs_is_shared = assign->rhs()->tag() == Expression::Tag::REF &&
                                                     static_cast<RefExpr const *>(assign->rhs())->variable().tag() == Variable::Tag::SHARED;
                                bool rhs_needs_bf16 = util->opt->bf16_fallback &&
                                                      CodegenUtility::IsBf16(assign->lhs()->type()) &&
                                                      !CodegenUtility::IsBf16(assign->rhs()->type());
                                bool rhs_is_rq = assign->rhs()->tag() == Expression::Tag::CALL &&
                                                 (static_cast<CallExpr const *>(assign->rhs())->op() == CallOp::RAY_TRACING_QUERY_ANY ||
                                                  static_cast<CallExpr const *>(assign->rhs())->op() == CallOp::RAY_TRACING_QUERY_ALL);
                                ++vi.write_count;
                                if (rhs_is_shared || rhs_needs_bf16 || rhs_is_rq || !can_declare) {
                                    vi.partial_or_ref = true;
                                } else if (vi.write_count == 1u && !vi.partial_or_ref) {
                                    vi.def_stmt = assign;
                                    vi.def_scope = scope;
                                    vi.def_stmt_idx = idx;
                                    vi.def_order_seq = next_def_order_seq++;
                                    vi.def_loop_depth = loop_depth;
                                }
                            }
                            record_env_write(v.uid(), rhs_const);
                        }
                    } else {
                        scan_lvalue_indices(scan_lvalue_indices, assign->lhs());
                        if (auto root = lvalue_root_variable(assign->lhs());
                            root && root->tag() == Variable::Tag::LOCAL) {
                            mark_partial_or_ref_write(root->uid());
                        }
                    }
                    break;
                }
                case Statement::Tag::IF: {
                    auto *ifs = static_cast<IfStmt const *>(s);
                    auto cond_val = eval_const_scalar(ifs->condition(), const_env);
                    if (cond_val && cond_val->kind == ConstScalar::Kind::BOOL) {
                        bool taken = cond_val->bits != 0u;
                        plan.const_if_taken.emplace(ifs, taken);
                        if (taken) {
                            self(self, ifs->true_branch(), scope, depth + 1u, loop_depth, true);
                        } else if (!ifs->false_branch()->statements().empty()) {
                            self(self, ifs->false_branch(), scope, depth + 1u, loop_depth, true);
                        }
                    } else {
                        scan_expr(scan_expr, ifs->condition(), true, false);
                        auto saved_env = const_env;
                        luisa::vector<uint32_t> written;
                        auto *prev_written = branch_written_uids;
                        branch_written_uids = &written;
                        self(self, ifs->true_branch(), scope, depth + 1u, loop_depth, true);
                        const_env = saved_env;
                        if (!ifs->false_branch()->statements().empty()) {
                            self(self, ifs->false_branch(), scope, depth + 1u, loop_depth, true);
                        }
                        branch_written_uids = prev_written;
                        const_env = std::move(saved_env);
                        for (auto uid : written) {
                            record_env_write(uid, std::nullopt);
                        }
                    }
                    break;
                }
                case Statement::Tag::SWITCH: {
                    auto *sw = static_cast<SwitchStmt const *>(s);
                    auto sel_val = eval_const_scalar(sw->expression(), const_env);
                    if (sel_val && (sel_val->kind == ConstScalar::Kind::I32 ||
                                    sel_val->kind == ConstScalar::Kind::U32)) {
                        const Statement *matched = nullptr;
                        const Statement *default_stmt = nullptr;
                        for (auto *cs : sw->body()->statements()) {
                            if (cs->tag() == Statement::Tag::SWITCH_CASE ||
                                cs->tag() == Statement::Tag::SWITCH_CASE_GROUP) {
                                auto *cstmt = static_cast<SwitchCaseStmt const *>(cs);
                                for (auto *ce : cstmt->expressions()) {
                                    auto cv = eval_const_scalar(ce, const_env);
                                    if (cv && cv->bits == sel_val->bits) {
                                        matched = cstmt;
                                        break;
                                    }
                                }
                                if (matched != nullptr) { break; }
                            } else if (cs->tag() == Statement::Tag::SWITCH_DEFAULT) {
                                default_stmt = cs;
                            }
                        }
                        const Statement *chosen = matched != nullptr ? matched : default_stmt;
                        plan.const_switch_branch.emplace(sw, chosen);
                        if (chosen != nullptr) {
                            const ScopeStmt *chosen_body =
                                chosen->tag() == Statement::Tag::SWITCH_DEFAULT ?
                                    static_cast<SwitchDefaultStmt const *>(chosen)->body() :
                                    static_cast<SwitchCaseStmt const *>(chosen)->body();
                            if (!chosen_body->statements().empty()) {
                                if (scope_has_non_trailing_break(chosen_body, true)) {
                                    plan.const_switch_needs_guard.emplace(sw);
                                }
                                self(self, chosen_body, scope, depth + 1u, loop_depth, true);
                            }
                        }
                    } else {
                        scan_expr(scan_expr, sw->expression(), true, false);
                        auto saved_env = const_env;
                        luisa::vector<uint32_t> written;
                        auto *prev_written = branch_written_uids;
                        branch_written_uids = &written;
                        scope_meta.emplace(sw->body(), ScopeMeta{scope, depth + 1u, loop_depth, false});
                        for (auto *cs : sw->body()->statements()) {
                            if (cs->tag() == Statement::Tag::SWITCH_CASE ||
                                cs->tag() == Statement::Tag::SWITCH_CASE_GROUP) {
                                auto *cstmt = static_cast<SwitchCaseStmt const *>(cs);
                                if (!cstmt->body()->statements().empty()) {
                                    const_env = saved_env;
                                    self(self, cstmt->body(), sw->body(), depth + 2u, loop_depth, true);
                                }
                            } else if (cs->tag() == Statement::Tag::SWITCH_DEFAULT) {
                                auto *dstmt = static_cast<SwitchDefaultStmt const *>(cs);
                                if (!dstmt->body()->statements().empty()) {
                                    const_env = saved_env;
                                    self(self, dstmt->body(), sw->body(), depth + 2u, loop_depth, true);
                                }
                            }
                        }
                        branch_written_uids = prev_written;
                        const_env = std::move(saved_env);
                        for (auto uid : written) {
                            record_env_write(uid, std::nullopt);
                        }
                    }
                    break;
                }
                case Statement::Tag::LOOP: {
                    auto *loop = static_cast<LoopStmt const *>(s);
                    const_env.clear();
                    self(self, loop->body(), scope, depth + 1u, loop_depth + 1u, true);
                    const_env.clear();
                    break;
                }
                case Statement::Tag::FOR: {
                    auto *fs = static_cast<ForStmt const *>(s);
                    const_env.clear();
                    if (auto root = lvalue_root_variable(fs->variable());
                        root && root->tag() == Variable::Tag::LOCAL) {
                        mark_partial_or_ref_write(root->uid());
                        mark_read(root->uid(), false, false);
                    }
                    scan_expr(scan_expr, fs->condition(), false, false);
                    scan_expr(scan_expr, fs->step(), false, false);
                    self(self, fs->body(), scope, depth + 1u, loop_depth + 1u, true);
                    const_env.clear();
                    break;
                }
                case Statement::Tag::RAY_QUERY: {
                    auto *rq = static_cast<RayQueryStmt const *>(s);
                    const_env.clear();
                    scan_expr(scan_expr, rq->query(), false, false);
                    self(self, rq->on_triangle_candidate(), scope, depth + 1u, loop_depth + 1u, true);
                    self(self, rq->on_procedural_candidate(), scope, depth + 1u, loop_depth + 1u, true);
                    const_env.clear();
                    break;
                }
                case Statement::Tag::SCOPE:
                    self(self, static_cast<ScopeStmt const *>(s), scope, depth + 1u, loop_depth, false);
                    break;
                case Statement::Tag::AUTO_DIFF:
                    self(self, static_cast<AutoDiffStmt const *>(s)->body(), scope, depth + 1u, loop_depth, false);
                    break;
                default:
                    break;
            }
        }
    done_scope:
        cur_scope = prev_scope;
        cur_stmt_idx = prev_stmt_idx;
        cur_loop_depth = prev_loop_depth;
    };

    scan_scope(scan_scope, func.body(), nullptr, 0u, 0u, true);

    // Helper: compute expanded AST node cost and leaf variable dependencies for inlining.
    luisa::unordered_map<uint32_t, uint32_t> inlined_node_cost;
    luisa::unordered_map<uint32_t, luisa::fixed_vector<Variable, 8>> inlined_deps;
    luisa::unordered_map<uint32_t, const RefExpr *> ssa_alias_root;
    luisa::unordered_set<uint32_t> param_immutable;
    for (auto &&arg : func.arguments()) {
        if (arg.tag() == Variable::Tag::LOCAL &&
            (static_cast<uint32_t>(func.variable_usage(arg.uid())) &
             static_cast<uint32_t>(Usage::WRITE)) == 0u) {
            param_immutable.emplace(arg.uid());
        }
    }
    auto is_scope_ancestor = [&](const ScopeStmt *anc, const ScopeStmt *desc) noexcept -> bool {
        while (desc != nullptr) {
            if (desc == anc) { return true; }
            auto it = scope_meta.find(desc);
            if (it == scope_meta.end()) { return false; }
            desc = it->second.parent;
        }
        return false;
    };
    constexpr uint32_t kMaxInlineNodes = 16u;
    constexpr size_t kMaxInlineSpan = 32u;

    auto collect_expr_cost_and_deps = [&](auto &&self, Expression const *expr,
                                          uint32_t &cost,
                                          luisa::fixed_vector<Variable, 8> &deps) noexcept -> bool {
        if (expr == nullptr || cost > kMaxInlineNodes) { return false; }
        if (auto *unwrapped = unwrap_pure_vector_lane(expr, &plan.inline_exprs); unwrapped != expr) {
            return self(self, unwrapped, cost, deps);
        }
        ++cost;
        if (cost > kMaxInlineNodes) { return false; }
        switch (expr->tag()) {
            case Expression::Tag::LITERAL:
            case Expression::Tag::CONSTANT:
                return true;
            case Expression::Tag::REF: {
                auto v = static_cast<RefExpr const *>(expr)->variable();
                if (v.tag() == Variable::Tag::LOCAL) {
                    if (auto ait = ssa_alias_root.find(v.uid()); ait != ssa_alias_root.end()) {
                        v = ait->second->variable();
                    }
                    if (auto it = inlined_node_cost.find(v.uid()); it != inlined_node_cost.end()) {
                        cost += it->second - 1u;
                        if (cost > kMaxInlineNodes) { return false; }
                        for (auto &d : inlined_deps.at(v.uid())) {
                            if (deps.size() >= 16u) { return false; }
                            deps.emplace_back(d);
                        }
                        return true;
                    }
                    // Never transitively re-inline an adjacent-load temp past its consuming assign.
                    if (plan.inline_exprs.contains(v.uid())) {
                        return false;
                    }
                }
                if (v.tag() == Variable::Tag::LOCAL || v.tag() == Variable::Tag::REFERENCE) {
                    if (deps.size() >= 16u) { return false; }
                    deps.emplace_back(v);
                }
                return true;
            }
            case Expression::Tag::UNARY:
                return self(self, static_cast<UnaryExpr const *>(expr)->operand(), cost, deps);
            case Expression::Tag::BINARY: {
                auto *b = static_cast<BinaryExpr const *>(expr);
                return self(self, b->lhs(), cost, deps) && self(self, b->rhs(), cost, deps);
            }
            case Expression::Tag::CAST:
                return self(self, static_cast<CastExpr const *>(expr)->expression(), cost, deps);
            case Expression::Tag::MEMBER:
                return self(self, static_cast<MemberExpr const *>(expr)->self(), cost, deps);
            case Expression::Tag::ACCESS: {
                auto *a = static_cast<AccessExpr const *>(expr);
                return self(self, a->range(), cost, deps) && self(self, a->index(), cost, deps);
            }
            case Expression::Tag::CALL: {
                auto *c = static_cast<CallExpr const *>(expr);
                for (auto *arg : c->arguments()) {
                    if (!self(self, arg, cost, deps)) { return false; }
                }
                return true;
            }
            default:
                return false;
        }
    };

    // Process locals in definition order so transitive inlining always sees
    // expanded dependencies of earlier-defined inlined operands.
    luisa::vector<uint32_t> eval_order = local_order;
    std::stable_sort(eval_order.begin(), eval_order.end(),
                     [&](uint32_t lhs, uint32_t rhs) noexcept {
                         return vars.at(lhs).def_order_seq < vars.at(rhs).def_order_seq;
                     });
    luisa::unordered_set<uint32_t> handled_uids;
    handled_uids.reserve(eval_order.size());

    for (auto uid : eval_order) {
        auto &vi = vars.at(uid);
        if (!vi.eligible) { continue; }
        // Case 0: completely unused (or only referenced inside pruned dead branches).
        if (vi.write_count == 0u && vi.read_count == 0u) {
            handled_uids.emplace(uid);
            continue;
        }
        // Case 1: dead pure single write with zero reads.
        if (!vi.partial_or_ref && vi.write_count == 1u && vi.read_count == 0u &&
            vi.def_stmt != nullptr && is_pure_inlineable_expr(vi.def_stmt->rhs())) {
            plan.skip_assigns.emplace(vi.def_stmt);
            handled_uids.emplace(uid);
            continue;
        }
        // Case 1a: zero-cost SSA alias (`const T lA = lB;` or `const T lA = float3(lB, ...)[0];`)
        // when lB is an immutable parameter or dominating single-write local.
        auto *def_rhs_unwrapped =
            vi.def_stmt != nullptr ? unwrap_pure_vector_lane(vi.def_stmt->rhs(), &plan.inline_exprs) : nullptr;
        if (!vi.partial_or_ref && vi.write_count == 1u && !vi.read_before_def &&
            vi.def_stmt != nullptr && vi.def_scope != nullptr &&
            vi.lca_scope == vi.def_scope &&
            def_rhs_unwrapped != nullptr &&
            def_rhs_unwrapped->tag() == Expression::Tag::REF) {
            auto *ref_expr = static_cast<RefExpr const *>(def_rhs_unwrapped);
            auto src_var = ref_expr->variable();
            if (src_var.tag() == Variable::Tag::LOCAL && src_var.uid() != uid &&
                *vi.var.type() == *src_var.type()) {
                if (auto ait = ssa_alias_root.find(src_var.uid()); ait != ssa_alias_root.end()) {
                    ref_expr = ait->second;
                    src_var = ref_expr->variable();
                }
                if (src_var.uid() != uid && !plan.inline_exprs.contains(src_var.uid())) {
                    bool src_ok = param_immutable.contains(src_var.uid());
                    if (!src_ok) {
                        if (auto sit = vars.find(src_var.uid()); sit != vars.end()) {
                            auto &svi = sit->second;
                            src_ok = svi.eligible && !svi.partial_or_ref &&
                                     svi.write_count == 1u && !svi.read_before_def &&
                                     svi.def_stmt != nullptr && svi.def_scope != nullptr &&
                                     svi.def_order_seq < vi.def_order_seq &&
                                     is_scope_ancestor(svi.def_scope, vi.def_scope);
                        }
                    }
                    if (src_ok) {
                        plan.inline_exprs.emplace(uid, ref_expr);
                        ssa_alias_root.emplace(uid, ref_expr);
                        plan.skip_assigns.emplace(vi.def_stmt);
                        handled_uids.emplace(uid);
                        continue;
                    }
                }
            }
        }
        // Case 1a2: zero-cost scalar literal propagation (`const uint lX = 9u;`)
        // across any read count or nested scope dominated by def_scope.
        if (!vi.partial_or_ref && vi.write_count == 1u && !vi.read_before_def &&
            !vi.used_as_member_or_access_base &&
            vi.def_stmt != nullptr && vi.def_scope != nullptr &&
            vi.lca_scope == vi.def_scope &&
            vi.var.type()->is_scalar() &&
            !CodegenUtility::IsBf16(vi.var.type()) &&
            def_rhs_unwrapped != nullptr &&
            def_rhs_unwrapped->tag() == Expression::Tag::LITERAL &&
            def_rhs_unwrapped->type() != nullptr &&
            *vi.var.type() == *def_rhs_unwrapped->type()) {
            plan.inline_exprs.emplace(uid, def_rhs_unwrapped);
            plan.skip_assigns.emplace(vi.def_stmt);
            inlined_node_cost.emplace(uid, 1u);
            inlined_deps.emplace(uid, luisa::fixed_vector<Variable, 8>{});
            handled_uids.emplace(uid);
            continue;
        }
        // Case 1b: adjacent single-use copy temporary (`T lTmp = _bfread(...); lDst = lTmp;`
        // or struct/matrix pure copy) with zero non-skipped intervening statements.
        if (!vi.partial_or_ref && vi.write_count == 1u && vi.read_count == 1u &&
            !vi.read_before_def && !vi.read_in_nested_loop &&
            !vi.used_as_member_or_access_base &&
            vi.def_stmt != nullptr && vi.def_scope != nullptr &&
            vi.single_read_scope == vi.def_scope && vi.single_read_is_direct_top_expr &&
            vi.single_read_stmt_idx > vi.def_stmt_idx &&
            vi.def_stmt->rhs()->type() != nullptr &&
            *vi.var.type() == *vi.def_stmt->rhs()->type()) {
            auto *rhs = vi.def_stmt->rhs();
            bool rhs_ok = is_pure_inlineable_expr(rhs);
            if (!rhs_ok && rhs->tag() == Expression::Tag::CALL) {
                auto *c = static_cast<CallExpr const *>(rhs);
                if (c->is_builtin() && is_readonly_load_call(c->op()) &&
                    !expr_has_mutating_call(rhs)) {
                    rhs_ok = true;
                }
            }
            if (rhs_ok) {
                auto stmts = vi.def_scope->statements();
                bool adjacent = true;
                for (size_t k = vi.def_stmt_idx + 1u; k < vi.single_read_stmt_idx; ++k) {
                    auto *mid = stmts[k];
                    if (mid->tag() != Statement::Tag::ASSIGN) {
                        adjacent = false;
                        break;
                    }
                    auto *mid_assign = static_cast<AssignStmt const *>(mid);
                    if (!plan.skip_assigns.contains(mid_assign) &&
                        !is_local_undefined_lifetime_seed(mid_assign)) {
                        adjacent = false;
                        break;
                    }
                }
                if (adjacent && stmts[vi.single_read_stmt_idx]->tag() == Statement::Tag::ASSIGN) {
                    auto *next_assign = static_cast<AssignStmt const *>(stmts[vi.single_read_stmt_idx]);
                    if (next_assign->rhs()->tag() == Expression::Tag::REF &&
                        static_cast<RefExpr const *>(next_assign->rhs())->variable().uid() == uid &&
                        next_assign->lhs()->type() != nullptr &&
                        *next_assign->lhs()->type() == *vi.var.type() &&
                        !expression_has_call(next_assign->lhs())) {
                        plan.inline_exprs.emplace(uid, rhs);
                        plan.skip_assigns.emplace(vi.def_stmt);
                        handled_uids.emplace(uid);
                        continue;
                    }
                }
            }
        }
        // Case 2: pure single-use scalar/vector temp inlined at its unique use in same scope.
        bool inlined = false;
        if (!vi.partial_or_ref && vi.write_count == 1u && vi.read_count == 1u &&
            !vi.read_before_def && !vi.read_in_nested_loop &&
            vi.def_stmt != nullptr && vi.def_scope != nullptr &&
            vi.single_read_scope == vi.def_scope && vi.single_read_is_direct_top_expr &&
            vi.single_read_stmt_idx > vi.def_stmt_idx &&
            (vi.single_read_stmt_idx - vi.def_stmt_idx) <= kMaxInlineSpan &&
            (vi.var.type()->is_scalar() || vi.var.type()->is_vector()) &&
            vi.def_stmt->rhs()->type() != nullptr &&
            *vi.var.type() == *vi.def_stmt->rhs()->type() &&
            (!vi.used_as_member_or_access_base ||
             vi.def_stmt->rhs()->tag() == Expression::Tag::REF ||
             (!vi.used_as_non_lane_base && is_pure_vector_ctor(vi.def_stmt->rhs()))) &&
            is_pure_inlineable_expr(vi.def_stmt->rhs())) {
            uint32_t cost = 0u;
            luisa::fixed_vector<Variable, 8> deps;
            if (collect_expr_cost_and_deps(collect_expr_cost_and_deps, vi.def_stmt->rhs(), cost, deps)) {
                bool has_ref_dep = false;
                for (auto &d : deps) {
                    if (d.tag() == Variable::Tag::REFERENCE) { has_ref_dep = true; }
                }
                bool interference = false;
                auto stmts = vi.def_scope->statements();
                for (size_t k = vi.def_stmt_idx + 1u; k < vi.single_read_stmt_idx; ++k) {
                    auto *mid = stmts[k];
                    if (mid->tag() != Statement::Tag::ASSIGN) {
                        interference = true;
                        break;
                    }
                    auto *mid_assign = static_cast<AssignStmt const *>(mid);
                    if (plan.skip_assigns.contains(mid_assign) ||
                        is_local_undefined_lifetime_seed(mid_assign)) {
                        continue;
                    }
                    if (expr_has_mutating_call(mid_assign->rhs())) {
                        interference = true;
                        break;
                    }
                    auto root = lvalue_root_variable(mid_assign->lhs());
                    if (!root) {
                        interference = true;
                        break;
                    }
                    if (has_ref_dep && root->tag() == Variable::Tag::REFERENCE) {
                        interference = true;
                        break;
                    }
                    for (auto &d : deps) {
                        if (d.uid() == root->uid()) {
                            interference = true;
                            break;
                        }
                    }
                    if (interference) { break; }
                }
                if (!interference) {
                    plan.inline_exprs.emplace(uid, vi.def_stmt->rhs());
                    plan.skip_assigns.emplace(vi.def_stmt);
                    inlined_node_cost.emplace(uid, cost);
                    inlined_deps.emplace(uid, std::move(deps));
                    handled_uids.emplace(uid);
                    inlined = true;
                }
            }
        }
        if (inlined) { continue; }
        // Case 3: single-write dominating local -> declare at definition site.
        if (!vi.partial_or_ref && vi.write_count == 1u && !vi.read_before_def &&
            vi.def_stmt != nullptr && vi.def_scope != nullptr &&
            vi.lca_scope == vi.def_scope) {
            plan.ssa_decl_assigns.emplace(vi.def_stmt, vi.var);
            handled_uids.emplace(uid);
            continue;
        }
    }

    // Emit Case 4 scoped declarations in deterministic original local_order.
    for (auto uid : local_order) {
        auto &vi = vars.at(uid);
        if (!vi.eligible || handled_uids.contains(uid)) { continue; }
        auto *decl_scope = vi.lca_scope != nullptr ? vi.lca_scope : func.body();
        while (decl_scope != nullptr && decl_scope != func.body()) {
            auto &meta = scope_meta.at(decl_scope);
            if (meta.can_declare && meta.loop_depth == 0u) { break; }
            decl_scope = meta.parent;
        }
        if (decl_scope == nullptr) { decl_scope = func.body(); }
        plan.scoped_decls[decl_scope].emplace_back(vi.var);
    }

    return plan;
}

// Software BF16 (builtin/bfloat16_fallback.bytes): `bfloat16_t` is a raw 16-bit
// pattern, so HLSL operators and intrinsics do not apply to it.  Every operator
// is emitted as a `_lc_bf16_*` helper call instead, and these tables name that
// helper.  An empty name means the plain HLSL spelling is already correct for
// the emulated type (the bitwise operators, for example, work directly on the
// pattern).
[[nodiscard]] vstd::string_view bf16_binary_helper_name(BinaryOp op) noexcept {
    switch (op) {
        case BinaryOp::ADD: return "add"sv;
        case BinaryOp::SUB: return "sub"sv;
        case BinaryOp::MUL: return "mul"sv;
        case BinaryOp::DIV: return "div"sv;
        case BinaryOp::MOD: return "mod"sv;
        case BinaryOp::LESS: return "lt"sv;
        case BinaryOp::GREATER: return "gt"sv;
        case BinaryOp::LESS_EQUAL: return "le"sv;
        case BinaryOp::GREATER_EQUAL: return "ge"sv;
        case BinaryOp::EQUAL: return "eq"sv;
        case BinaryOp::NOT_EQUAL: return "ne"sv;
        default: return {};
    }
}

// Whether an expression evaluates a call.  A packed struct member is written
// through a read-modify-write helper, which renders the member's owner twice; that
// is only sound for an owner without side effects.
[[nodiscard]] bool expression_has_call(Expression const *expr) noexcept {
    if (expr == nullptr) { return false; }
    switch (expr->tag()) {
        case Expression::Tag::CALL:
            return true;
        case Expression::Tag::UNARY:
            return expression_has_call(static_cast<UnaryExpr const *>(expr)->operand());
        case Expression::Tag::BINARY: {
            auto e = static_cast<BinaryExpr const *>(expr);
            return expression_has_call(e->lhs()) || expression_has_call(e->rhs());
        }
        case Expression::Tag::MEMBER:
            return expression_has_call(static_cast<MemberExpr const *>(expr)->self());
        case Expression::Tag::ACCESS: {
            auto e = static_cast<AccessExpr const *>(expr);
            return expression_has_call(e->range()) || expression_has_call(e->index());
        }
        case Expression::Tag::CAST:
            return expression_has_call(static_cast<CastExpr const *>(expr)->expression());
        default:
            return false;
    }
}

[[nodiscard]] vstd::string_view bf16_unary_helper_name(UnaryOp op) noexcept {
    switch (op) {
        case UnaryOp::PLUS: return "pos"sv;
        case UnaryOp::MINUS: return "neg"sv;
        case UnaryOp::NOT: return "not"sv;
        default: return {};
    }
}

[[nodiscard]] bool is_emulated_i8(Type const *t) noexcept {
    if (t == nullptr) { return false; }
    auto elem = t->is_vector() ? t->element() : t;
    return elem != nullptr && elem->is_int8();
}

[[nodiscard]] bool is_emulated_u8(Type const *t) noexcept {
    if (t == nullptr) { return false; }
    auto elem = t->is_vector() ? t->element() : t;
    return elem != nullptr && elem->is_uint8();
}

}// namespace

void StringStateVisitor::visit(const UnaryExpr *expr) {
    literalBrace = true;
    auto fallBackLiteral = vstd::scope_exit([this]() { literalBrace = false; });
    if (util->opt->bf16_fallback &&
        CodegenUtility::IsBf16(expr->operand()->type())) {
        if (auto helper = bf16_unary_helper_name(expr->op());
            !helper.empty()) {
            str << "_lc_bf16_"sv << helper << '(';
            expr->operand()->accept(*this);
            str << ')';
            return;
        }
    }
    const bool wrap_i8 = util->opt->emulate_8bit &&
                         is_emulated_i8(expr->type()) &&
                         (expr->op() == UnaryOp::MINUS || expr->op() == UnaryOp::BIT_NOT);
    const bool wrap_u8 = util->opt->emulate_8bit &&
                         is_emulated_u8(expr->type()) &&
                         (expr->op() == UnaryOp::MINUS || expr->op() == UnaryOp::BIT_NOT);
    if (wrap_i8) { str << "_lc_wrap_i8("sv; }
    else if (wrap_u8) { str << "_lc_wrap_u8("sv; }
    str << "(";
    switch (expr->op()) {
        case UnaryOp::PLUS://+x
            str << '+';
            break;
        case UnaryOp::MINUS://-x
            str << '-';
            break;
        case UnaryOp::NOT://!x
            str << '!';
            break;
        case UnaryOp::BIT_NOT://~x
            str << '~';
            break;
    }
    expr->operand()->accept(*this);
    str << ")";
    if (wrap_i8 || wrap_u8) { str << ')'; }
}
void StringStateVisitor::visit(const BinaryExpr *expr) {
    literalBrace = true;
    auto fallBackLiteral = vstd::scope_exit([this]() { literalBrace = false; });
    auto op = expr->op();
    if (util->opt->bf16_fallback &&
        CodegenUtility::IsBf16(expr->lhs()->type())) {
        if (auto helper = bf16_binary_helper_name(op);
            !helper.empty()) {
            str << "_lc_bf16_"sv << helper << '(';
            expr->lhs()->accept(*this);
            str << ',';
            expr->rhs()->accept(*this);
            str << ')';
            return;
        }
    }
    auto IsMulFuncCall = [&]() -> bool {
        if (op == BinaryOp::MUL) [[unlikely]] {
            if ((expr->lhs()->type()->is_matrix() &&
                 (!expr->rhs()->type()->is_scalar())) ||
                (expr->rhs()->type()->is_matrix() &&
                 (!expr->lhs()->type()->is_scalar()))) {
                return true;
            }
        }
        return false;
    };
    if (IsMulFuncCall()) {
        str << "_Mul("sv;
        expr->lhs()->accept(*this);
        str << ',';
        expr->rhs()->accept(*this);//Reverse matrix
        str << ')';

    } else if (op == BinaryOp::MOD &&
               (expr->type()->is_float() ||
                (expr->type()->is_vector() &&
                 expr->type()->element()->is_float()))) {
        auto element = expr->type()->is_vector() ?
                           expr->type()->element() :
                           expr->type();
        // In fast-math mode, keep native HLSL `fmod(...)`; under strict math
        // (!enable_fast_math), use `_lc_fmod(...)` for binary32 exact remainder.
        str << ((!util->opt->enable_fast_math && element->is_float32()) ?
                    "_lc_fmod("sv :
                    "fmod("sv);
        expr->lhs()->accept(*this);
        str << ',';
        expr->rhs()->accept(*this);
        str << ')';
    } else if (op == BinaryOp::AND) {
        if (expr->type()->is_scalar() && !expression_has_call(expr->rhs())) {
            str << "(";
            expr->lhs()->accept(*this);
            str << "&&";
            expr->rhs()->accept(*this);
            str << ")";
        } else {
            str << "and(";
            expr->lhs()->accept(*this);
            str << ",";
            expr->rhs()->accept(*this);
            str << ")";
        }
    } else if (op == BinaryOp::OR) {
        if (expr->type()->is_scalar() && !expression_has_call(expr->rhs())) {
            str << "(";
            expr->lhs()->accept(*this);
            str << "||";
            expr->rhs()->accept(*this);
            str << ")";
        } else {
            str << "or(";
            expr->lhs()->accept(*this);
            str << ",";
            expr->rhs()->accept(*this);
            str << ")";
        }
    } else {
        const bool wrap_op = op == BinaryOp::ADD || op == BinaryOp::SUB ||
                             op == BinaryOp::MUL || op == BinaryOp::SHL;
        const bool wrap_i8 = util->opt->emulate_8bit && wrap_op && is_emulated_i8(expr->type());
        const bool wrap_u8 = util->opt->emulate_8bit && wrap_op && is_emulated_u8(expr->type());
        if (wrap_i8) { str << "_lc_wrap_i8("sv; }
        else if (wrap_u8) { str << "_lc_wrap_u8("sv; }
        str << "(";
        expr->lhs()->accept(*this);
        switch (op) {
            case BinaryOp::ADD:
                str << '+';
                break;
            case BinaryOp::SUB:
                str << '-';
                break;
            case BinaryOp::MUL: {
                str << '*';
            } break;
            case BinaryOp::DIV:
                str << '/';
                break;
            case BinaryOp::MOD:
                str << '%';
                break;
            case BinaryOp::BIT_AND:
                if (expr->type()->is_bool() && !expression_has_call(expr->rhs())) {
                    str << "&&"sv;
                } else {
                    str << '&';
                }
                break;
            case BinaryOp::BIT_OR:
                if (expr->type()->is_bool() && !expression_has_call(expr->rhs())) {
                    str << "||"sv;
                } else {
                    str << '|';
                }
                break;
            case BinaryOp::BIT_XOR:
                str << '^';
                break;
            case BinaryOp::SHL:
                str << "<<"sv;
                break;
            case BinaryOp::SHR:
                str << ">>"sv;
                break;
            case BinaryOp::LESS:
                str << '<';
                break;
            case BinaryOp::GREATER:
                str << '>';
                break;
            case BinaryOp::LESS_EQUAL:
                str << "<="sv;
                break;
            case BinaryOp::GREATER_EQUAL:
                str << ">="sv;
                break;
            case BinaryOp::EQUAL:
                str << "=="sv;
                break;
            case BinaryOp::NOT_EQUAL:
                str << "!="sv;
                break;
            default:
                LUISA_ERROR_WITH_LOCATION(
                    "Not implemented.");
        }
        expr->rhs()->accept(*this);
        str << ")";
        if (wrap_i8 || wrap_u8) { str << ')'; }
    }
}
void StringStateVisitor::visit(const MemberExpr *expr) {
    char const *xyzw = "xyzw";
    if (expr->is_swizzle()) {
        if (emit_plan != nullptr) {
            if (auto *unwrapped = unwrap_pure_vector_lane(expr, &emit_plan->inline_exprs); unwrapped != expr) {
                const bool need_parens =
                    unwrapped->tag() != Expression::Tag::REF &&
                    unwrapped->tag() != Expression::Tag::CALL &&
                    unwrapped->tag() != Expression::Tag::CONSTANT &&
                    unwrapped->tag() != Expression::Tag::LITERAL;
                if (need_parens) { str << '('; }
                unwrapped->accept(*this);
                if (need_parens) { str << ')'; }
                return;
            }
        }
        accessCount++;
        expr->self()->accept(*this);
        accessCount--;
        str << '.';
        for (auto i : vstd::range(expr->swizzle_size())) {
            str << xyzw[expr->swizzle_index(i)];
        }

    } else {
        vstd::StringBuilder curStr;
        StringStateVisitor vis(f, curStr, util);
        vis.emit_plan = emit_plan;
        expr->self()->accept(vis);
        // A member of a word that mixes 16-bit scalars with one-byte bools lives
        // in the packed `uint` of that word (see `PackedStructMember`).
        if (auto packed = util->GetPackedStructMember(
                expr->self()->type(), expr->member_index())) {
            str << "_lc_pack_get_"sv << packed_kind_suffix(packed->kind) << '('
                << curStr << ".v"sv << vstd::to_string(packed->word_index) << ','
                << vstd::to_string(packed->byte_offset) << ')';
            return;
        }
        str << curStr << ".v"sv << vstd::to_string(expr->member_index());
        auto t = expr->type();
        // Not a special vec3

        if (t->is_vector() && t->dimension() == 3 && !t->element()->is_bool()) {
            auto self_type = expr->self()->type();
            // Internal raster structs (_mesh, v2p) emit raw vectors — no .v wrapper
            bool is_internal = (util->opt->internalStruct.find(self_type) != util->opt->internalStruct.end());
            if (!is_internal && !(self_type->is_structure() && !self_type->member_attributes().empty())) [[likely]] {
                str << ".v"sv;
            }
        }
    }
}
void StringStateVisitor::visit(const AccessExpr *expr) {
    if (emit_plan != nullptr) {
        if (auto *unwrapped = unwrap_pure_vector_lane(expr, &emit_plan->inline_exprs); unwrapped != expr) {
            const bool need_parens =
                unwrapped->tag() != Expression::Tag::REF &&
                unwrapped->tag() != Expression::Tag::CALL &&
                unwrapped->tag() != Expression::Tag::CONSTANT &&
                unwrapped->tag() != Expression::Tag::LITERAL;
            if (need_parens) { str << '('; }
            unwrapped->accept(*this);
            if (need_parens) { str << ')'; }
            return;
        }
    }
    auto t = expr->range()->type();
    auto basicAccess = [&]() {
        accessCount++;
        expr->range()->accept(*this);
        accessCount--;
        str << '[';
        expr->index()->accept(*this);
        str << ']';
    };
    if (expr->range()->tag() == Expression::Tag::REF) {
        auto variable = static_cast<RefExpr const *>(expr->range())->variable();
        if (variable.tag() == Variable::Tag::SHARED) {
            accessCount++;
            expr->range()->accept(*this);
            accessCount--;
            str << '[';
            expr->index()->accept(*this);
            str << ']';
            if (accessCount == 0 && t->is_matrix() && t->dimension() == 3u) {
                str << ".xyz"sv;
            } else if (t->is_array() && t->element()->is_vector() && t->element()->dimension() == 3) {
                str << ".v"sv;
            }
            return;
        }
    }
    //    else if (expr->range()->tag() == Expression::Tag::CONSTANT) {
    //        basicAccess();
    //        return;
    //    }
    switch (t->tag()) {
        case Type::Tag::BUFFER:
        case Type::Tag::VECTOR: {
            basicAccess();
        } break;
        case Type::Tag::COOPERATIVE_VECTOR: {
            accessCount++;
            expr->range()->accept(*this);
            accessCount--;
            str << ".Get(";
            expr->index()->accept(*this);
            str << ')';
        } break;
        case Type::Tag::MATRIX: {
            accessCount++;
            expr->range()->accept(*this);
            accessCount--;
            str << '[';
            expr->index()->accept(*this);
            str << ']';
            if (accessCount == 0 && t->dimension() == 3u) {
                str << ".xyz"sv;
            }
        } break;
        default: {
            accessCount++;
            expr->range()->accept(*this);
            accessCount--;
            str << ".v[";
            expr->index()->accept(*this);
            str << ']';
            if (t->element()->is_vector() && t->element()->dimension() == 3) {
                str << ".v"sv;
            }
        } break;
    }
}
void StringStateVisitor::visit(const RefExpr *expr) {
    Variable v = expr->variable();
    if (emit_plan != nullptr && v.tag() == Variable::Tag::LOCAL) {
        if (auto it = emit_plan->inline_exprs.find(v.uid());
            it != emit_plan->inline_exprs.end()) {
            auto *inlined = it->second;
            const bool need_parens =
                inlined->tag() != Expression::Tag::REF &&
                inlined->tag() != Expression::Tag::CALL &&
                inlined->tag() != Expression::Tag::CONSTANT;
            if (need_parens) { str << '('; }
            inlined->accept(*this);
            if (need_parens) { str << ')'; }
            return;
        }
    }
    vstd::StringBuilder tempStr;
    if (util->opt->funcType == CodegenStackData::FuncType::Callable) {
        if (auto const *promoted = util->opt->FindPromotedResourceArg(f, v)) {
            util->GetVariableName(util->opt->kernel, *promoted, tempStr);
            util->RegistStructType(promoted->type());
            str << tempStr;
            return;
        }
    }
    util->GetVariableName(f, v, tempStr);
    util->RegistStructType(v.type());
    str << tempStr;
}

void StringStateVisitor::visit(const LiteralExpr *expr) {
    if (literalBrace) {
        str << '(';
    }
    luisa::visit(
        [&]<typename T>(T const &value) -> void {
            PrintValue<T> prt;
            if constexpr (std::is_same_v<T, bfloat16> ||
                          std::is_same_v<T, bfloat162> ||
                          std::is_same_v<T, bfloat163> ||
                          std::is_same_v<T, bfloat164>) {
                prt(value, str, util->opt->bf16_fallback);
            } else if constexpr (std::is_same_v<T, luisa::byte> ||
                                 std::is_same_v<T, luisa::ubyte>) {
                prt(value, str, util->opt->emulate_8bit);
            } else if constexpr (std::is_same_v<T, byte2> ||
                                 std::is_same_v<T, byte3> ||
                                 std::is_same_v<T, byte4> ||
                                 std::is_same_v<T, ubyte2> ||
                                 std::is_same_v<T, ubyte3> ||
                                 std::is_same_v<T, ubyte4>) {
                prt(value, str, false, util->opt->emulate_8bit);
            } else {
                prt(value, str);
            }
        },
        expr->value());
    if (literalBrace) {
        str << ')';
    }
}
void StringStateVisitor::visit(const CallExpr *expr) {
    util->GetFunctionName(expr, str, *this);
}

void StringStateVisitor::visit(const StringIDExpr *expr) {
    str << "((";
    util->GetTypeName(*expr->type(), str, Usage::READ);
    str << ")" << luisa::hash_value(expr->data()) << "ull)";
}

void StringStateVisitor::visit(const CastExpr *expr) {
    if (expr->type() == expr->expression()->type()) [[unlikely]] {
        expr->expression()->accept(*this);
        return;
    }
    // Software BF16 (builtin/bfloat16_fallback.bytes): the value lives in a raw
    // 16-bit pattern, so a value conversion is a `_lc_bf16_*` helper call and a
    // bit cast is a plain cast between two 16-bit patterns.
    if (util->opt->bf16_fallback &&
        (CodegenUtility::IsBf16(expr->type()) ||
         CodegenUtility::IsBf16(expr->expression()->type()))) {
        auto dst_bf16 = CodegenUtility::IsBf16(expr->type());
        auto src_bf16 = CodegenUtility::IsBf16(expr->expression()->type());
        auto emit_dst_cast = [&]() noexcept {
            // BF16 scalars and vectors share one storage type, so the outer
            // cast performs the broadcast/truncation a scalar-to-vector
            // conversion needs.
            str << '(';
            util->GetTypeName(*expr->type(), str, Usage::READ);
            str << ')';
        };
        if (expr->op() == CastOp::BITWISE || (dst_bf16 && src_bf16)) {
            emit_dst_cast();
            expr->expression()->accept(*this);
            return;
        }
        auto element_tag = [](Type const *t) noexcept {
            return t->is_vector() ? t->element()->tag() : t->tag();
        };
        if (dst_bf16) {
            vstd::string_view helper;
            switch (element_tag(expr->expression()->type())) {
                case Type::Tag::BOOL: helper = "from_bool"sv; break;
                case Type::Tag::FLOAT16: helper = "from_f16"sv; break;
                case Type::Tag::FLOAT32:
                case Type::Tag::FLOAT64: helper = "from_f32"sv; break;
                case Type::Tag::INT8:
                case Type::Tag::INT16:
                case Type::Tag::INT32:
                case Type::Tag::INT64: helper = "from_i32"sv; break;
                case Type::Tag::UINT8:
                case Type::Tag::UINT16:
                case Type::Tag::UINT32:
                case Type::Tag::UINT64: helper = "from_u32"sv; break;
                default:
                    LUISA_ERROR_WITH_LOCATION(
                        "Cannot convert '{}' to BF16: unsupported source type '{}'.",
                        expr->type()->description(),
                        expr->expression()->type()->description());
            }
            emit_dst_cast();
            str << "_lc_bf16_"sv << helper << '(';
            expr->expression()->accept(*this);
            str << ')';
            return;
        }
        // BF16 source, ordinary destination.
        vstd::string_view helper;
        switch (element_tag(expr->type())) {
            case Type::Tag::BOOL: helper = "to_bool"sv; break;
            case Type::Tag::FLOAT16: helper = "to_f16"sv; break;
            case Type::Tag::FLOAT32:
            case Type::Tag::FLOAT64: helper = "to_f32"sv; break;
            case Type::Tag::INT8:
            case Type::Tag::INT16:
            case Type::Tag::INT32:
            case Type::Tag::INT64: helper = "to_i32"sv; break;
            case Type::Tag::UINT8:
            case Type::Tag::UINT16:
            case Type::Tag::UINT32:
            case Type::Tag::UINT64: helper = "to_u32"sv; break;
            default:
                LUISA_ERROR_WITH_LOCATION(
                    "Cannot convert BF16 to '{}': unsupported destination type.",
                    expr->type()->description());
        }
        emit_dst_cast();
        str << "_lc_bf16_"sv << helper << '(';
        expr->expression()->accept(*this);
        str << ')';
        return;
    }
    str << "(";
    switch (expr->op()) {
        case CastOp::STATIC: {
            const bool wrap_i8 = util->opt->emulate_8bit && is_emulated_i8(expr->type());
            const bool wrap_u8 = util->opt->emulate_8bit && is_emulated_u8(expr->type());
            if (wrap_i8) { str << "_lc_wrap_i8("sv; }
            else if (wrap_u8) { str << "_lc_wrap_u8("sv; }
            str << '(';
            util->GetTypeName(*expr->type(), str, Usage::READ);
            str << ")("sv;
            expr->expression()->accept(*this);
            str << ')';
            if (wrap_i8 || wrap_u8) { str << ')'; }
        } break;
        case CastOp::BITWISE: {
            auto dst_type = expr->type();
            auto type = dst_type;
            while (type->is_vector()) {
                type = type->element();
            }
            auto arg_type = expr->expression()->type();
            // Vector to scalar
            if (dst_type->dimension() < arg_type->dimension()) {
                LUISA_ASSERT(dst_type->is_scalar(), "Only support vector to scalar");
                auto elem = arg_type->element();
                const auto dim = arg_type->dimension();
                const auto elem_size = elem->size();
                // Wrap unsigned bit pattern to final scalar destination type.
                auto emit_scalar_from_bits = [&](size_t bit_bytes, auto &&emit_bits) {
                    if (bit_bytes == 2u) {
                        if (dst_type->is_float16()) {
                            str << "asfloat16("sv;
                            emit_bits();
                            str << ')';
                        } else if (dst_type->is_int16()) {
                            str << "(int16_t)("sv;
                            emit_bits();
                            str << ')';
                        } else {
                            emit_bits();
                        }
                    } else if (bit_bytes == 4u) {
                        if (dst_type->is_float32()) {
                            str << "asfloat("sv;
                            emit_bits();
                            str << ')';
                        } else if (dst_type->is_int32()) {
                            str << "asint("sv;
                            emit_bits();
                            str << ')';
                        } else {
                            emit_bits();
                        }
                    } else {
                        if (dst_type->is_float64()) {
                            str << "_lc_bitcast_u64_f64("sv;
                            emit_bits();
                            str << ')';
                        } else if (dst_type->is_int64()) {
                            str << "(int64_t)("sv;
                            emit_bits();
                            str << ')';
                        } else {
                            emit_bits();
                        }
                    }
                };
                if (dim == 2u && elem_size == 1u) {
                    emit_scalar_from_bits(2u, [&]() {
                        str << (elem->is_bool() ? "_lc_bitcast_bool2_u16("sv : "_lc_bitcast_b8x2_u16("sv);
                        expr->expression()->accept(*this);
                        str << ')';
                    });
                } else if (dim == 4u && elem_size == 1u) {
                    emit_scalar_from_bits(4u, [&]() {
                        str << (elem->is_bool() ? "Bool4AsInt<uint>("sv : "_lc_bitcast_b8x4_u32("sv);
                        expr->expression()->accept(*this);
                        str << ')';
                    });
                } else if (dim == 2u && elem_size == 2u) {
                    emit_scalar_from_bits(4u, [&]() {
                        str << (elem->is_float16() ? "_lc_bitcast_f16x2_u32("sv : "_lc_bitcast_i16x2_u32("sv);
                        expr->expression()->accept(*this);
                        str << ')';
                    });
                } else if (dim == 2u && elem_size == 4u) {
                    emit_scalar_from_bits(8u, [&]() {
                        str << "_lc_bitcast_u32x2_u64<uint64_t>("sv;
                        expr->expression()->accept(*this);
                        str << ')';
                    });
                } else if (dim == 4u && elem_size == 2u) {
                    emit_scalar_from_bits(8u, [&]() {
                        str << (elem->is_float16() ? "_lc_bitcast_f16x4_u64<uint64_t>("sv : "_lc_bitcast_i16x4_u64<uint64_t>("sv);
                        expr->expression()->accept(*this);
                        str << ')';
                    });
                } else {
                    LUISA_ERROR("Vector to scalar bitwise cast unsupported: {} -> {}.",
                                arg_type->description(), dst_type->description());
                }
            }
            // Scalar to vector
            else if (dst_type->dimension() > arg_type->dimension()) {
                LUISA_ASSERT(arg_type->is_scalar(), "Only support scalar to vector");
                auto elem = dst_type->element();
                const auto dim = dst_type->dimension();
                const auto elem_size = elem->size();
                auto emit_src_bits = [&](size_t bit_bytes) {
                    if (bit_bytes == 2u) {
                        if (arg_type->is_float16()) {
                            str << "asuint16("sv;
                            expr->expression()->accept(*this);
                            str << ')';
                        } else {
                            str << "(uint16_t)("sv;
                            expr->expression()->accept(*this);
                            str << ')';
                        }
                    } else if (bit_bytes == 4u) {
                        str << "asuint("sv;
                        expr->expression()->accept(*this);
                        str << ')';
                    } else {
                        if (arg_type->is_float64()) {
                            str << "_lc_bitcast_f64_u64("sv;
                            expr->expression()->accept(*this);
                            str << ')';
                        } else {
                            str << "(uint64_t)("sv;
                            expr->expression()->accept(*this);
                            str << ')';
                        }
                    }
                };
                if (dim == 2u && elem_size == 1u) {
                    if (elem->is_bool()) {
                        str << "_lc_bitcast_u16_bool2("sv;
                    } else {
                        str << (elem->is_int8() ? "_lc_bitcast_u16_i8x2<"sv : "_lc_bitcast_u16_u8x2<"sv);
                        util->GetTypeName(*dst_type, str, Usage::READ, true);
                        str << ">("sv;
                    }
                    emit_src_bits(2u);
                    str << ')';
                } else if (dim == 4u && elem_size == 1u) {
                    if (elem->is_bool()) {
                        str << "IntAsBool4("sv;
                    } else {
                        str << (elem->is_int8() ? "_lc_bitcast_u32_i8x4<"sv : "_lc_bitcast_u32_u8x4<"sv);
                        util->GetTypeName(*dst_type, str, Usage::READ, true);
                        str << ">("sv;
                    }
                    emit_src_bits(4u);
                    str << ')';
                } else if (dim == 2u && elem_size == 2u) {
                    str << (elem->is_float16() ? "_lc_bitcast_u32_f16x2<"sv : "_lc_bitcast_u32_i16x2<"sv);
                    util->GetTypeName(*dst_type, str, Usage::READ, true);
                    str << ">("sv;
                    emit_src_bits(4u);
                    str << ')';
                } else if (dim == 2u && elem_size == 4u) {
                    if (elem->is_float32()) {
                        str << "asfloat(_lc_bitcast_u64_u32x2("sv;
                        emit_src_bits(8u);
                        str << "))"sv;
                    } else if (elem->is_int32()) {
                        str << "asint(_lc_bitcast_u64_u32x2("sv;
                        emit_src_bits(8u);
                        str << "))"sv;
                    } else {
                        str << "_lc_bitcast_u64_u32x2("sv;
                        emit_src_bits(8u);
                        str << ')';
                    }
                } else if (dim == 4u && elem_size == 2u) {
                    str << (elem->is_float16() ? "_lc_bitcast_u64_f16x4<"sv : "_lc_bitcast_u64_i16x4<"sv);
                    util->GetTypeName(*dst_type, str, Usage::READ, true);
                    str << ">("sv;
                    emit_src_bits(8u);
                    str << ')';
                } else {
                    LUISA_ERROR("Scalar to vector bitwise cast unsupported: {} -> {}.",
                                arg_type->description(), dst_type->description());
                }
            } else {
                switch (type->tag()) {
                    case Type::Tag::FLOAT16:
                    case Type::Tag::FLOAT32:
                        str << "asfloat"sv;
                        break;
                    case Type::Tag::INT8:
                    case Type::Tag::INT16:
                    case Type::Tag::INT32:
                    case Type::Tag::INT64:
                        str << "asint"sv;
                        break;
                    case Type::Tag::UINT8:
                    case Type::Tag::UINT16:
                    case Type::Tag::UINT32:
                    case Type::Tag::UINT64:
                        str << "asuint"sv;
                        break;
                    default:
                        LUISA_ERROR_WITH_LOCATION(
                            "Bitwise cast not implemented for type '{}'.",
                            type->description());
                }
                str << '(';
                expr->expression()->accept(*this);
                str << ')';
            }
        } break;
    }
    str << ")";
}

void StringStateVisitor::visit(const ConstantExpr *expr) {
    util->GetConstName(expr->data().hash(), expr->data(), str);
}

void StringStateVisitor::visit(const BreakStmt *state) {
    str << "break;\n";
}
void StringStateVisitor::visit(const ContinueStmt *state) {

    str << "continue;\n";
}
void StringStateVisitor::visit(const ReturnStmt *state) {
    if (state->expression()) {
        str << "return ";
        state->expression()->accept(*this);
        str << ";\n";
    } else {
        str << "return;\n";
    }
}
void StringStateVisitor::_emit_local_zero_decl(Variable const &v) {
    Usage usage = f.variable_usage(v.uid());
    if ((static_cast<uint32_t>(usage) & static_cast<uint32_t>(Usage::WRITE)) == 0) {
        str << "const "sv;
    }
    vstd::StringBuilder typeName;
    util->GetTypeName(*v.type(), typeName, usage);
    vstd::StringBuilder varName;
    util->GetVariableName(f, v, varName);
    str << typeName << ' ' << varName;
    if (!(v.type()->is_resource() || v.type()->is_custom() || v.type()->is_cooperative_vector())) [[likely]] {
        str << "=("sv << typeName << ")0"sv;
    }
    str << ";\n";
}

void StringStateVisitor::_emit_folded_scope(const ScopeStmt *body, bool wrap_in_switch_guard) {
    auto emit_body_stmts = [&](bool strip_trailing_break) {
        if (emit_plan != nullptr) {
            if (auto it = emit_plan->scoped_decls.find(body);
                it != emit_plan->scoped_decls.end()) {
                for (auto &&v : it->second) {
                    _emit_local_zero_decl(v);
                }
            }
        }
        auto stmts = body->statements();
        for (size_t i = 0u; i < stmts.size(); ++i) {
            auto *s = stmts[i];
            if (strip_trailing_break && i + 1u == stmts.size() && s->tag() == Statement::Tag::BREAK) {
                return;
            }
            s->accept(*this);
            switch (s->tag()) {
                case Statement::Tag::BREAK:
                case Statement::Tag::CONTINUE:
                case Statement::Tag::RETURN:
                    return;
                default: break;
            }
        }
    };
    if (wrap_in_switch_guard) {
        str << "switch(0u){\ndefault:{\n"sv;
        emit_body_stmts(false);
        str << "break;\n}\n}\n"sv;
    } else {
        Scope scope{this};
        emit_body_stmts(true);
    }
}

void StringStateVisitor::visit(const ScopeStmt *state) {
    if (emit_plan != nullptr && state != f.body()) {
        if (auto it = emit_plan->scoped_decls.find(state);
            it != emit_plan->scoped_decls.end()) {
            for (auto &&v : it->second) {
                _emit_local_zero_decl(v);
            }
        }
    }
    for (auto &&i : state->statements()) {
        i->accept(*this);
        switch (i->tag()) {
            case Statement::Tag::BREAK:
            case Statement::Tag::CONTINUE:
            case Statement::Tag::RETURN:
                return;
            default: break;
        }
    }
}
void StringStateVisitor::visit(const AutoDiffStmt *stmt) {
    visit(stmt->body());
}
void StringStateVisitor::visit(const PrintStmt *stmt) {
    vstd::fixed_vector<Type const *, 16> types;
    vstd::push_back_func(
        types,
        stmt->arguments().size(),
        [&](size_t i) {
            return stmt->arguments()[i]->type();
        });
    auto structType = Type::structure(types);
    vstd::string counterName = "_p";
    vstd::to_string(printCount, counterName);
    str << "uint "sv << counterName << ";\nInterlockedAdd(_printCounter[0],"sv;
    size_t align = std::max<size_t>(structType->alignment(), 4);
    size_t ele_size = structType->size() + align;
    ele_size = ((ele_size + 15ull) & (~15ull));
    vstd::to_string(ele_size, str);
    str << ',' << counterName << ");\n"sv
        << "if("sv << counterName << "<1048576){\n"sv;
    {
        vstd::StringBuilder typeName;
        util->GetTypeName(*structType, typeName, Usage::READ_WRITE);
        vstd::string dataName = "_pv";
        vstd::to_string(printCount, dataName);
        str << typeName << ' ' << dataName << "=("sv << typeName << ")0;\n"sv;
        for (auto i : vstd::range(static_cast<int64>(types.size()))) {
            str << dataName << ".v"sv;
            vstd::to_string(i, str);
            auto t = types[i];
            if (t->is_vector() && t->dimension() == 3) {
                str << ".v"sv;
            }
            str << '=';
            stmt->arguments()[i]->accept(*this);
            str << ";\n"sv;
        }
        str << "_printBuffer.Store("sv
            << counterName << ',';
        auto printerIdx = util->AddPrinter(stmt->format(), structType);
        vstd::to_string(printerIdx, str);
        str << ");\n"sv
            << "_printBuffer.Store("sv
            << counterName << '+' + vstd::to_string(align) + ',' << dataName << ");\n"sv;
        // The whole-struct store above uses the compiler's natural aggregate
        // layout, which is target-dependent for matrix members: DXC->DXIL
        // stores matrices row-major with 16B-major-vector padding (the Luisa
        // layout), while DXC->SPIR-V stores them column-major with tightly
        // packed columns. The printer ABI is the Luisa layout (columns back to
        // back, each padded to its vector size), so every matrix field is
        // rewritten column by column at its exact offsets. A matrix row in the
        // HLSL representation is exactly one Luisa column, and its byte width
        // (float2/float4/half2/half4/double2/...) is exactly the Luisa column
        // size, so each store is width-exact and layout-independent.
        for (auto &&field : compute::detail::shader_print_matrix_fields(types)) {
            vstd::string access = dataName;
            access += ".v";
            vstd::to_string(field.argument_index, access);
            for (auto &&step : field.path) {
                if (step.container->is_structure()) {
                    access += ".v";
                    vstd::to_string(step.index, access);
                } else {
                    access += ".v[";
                    vstd::to_string(step.index, access);
                    access += ']';
                }
            }
            for (auto i = 0u; i < field.type->dimension(); i++) {
                str << "_printBuffer.Store("sv << counterName << '+';
                vstd::to_string(align + field.offset + i * field.column_stride, str);
                str << ',' << access << '[';
                vstd::to_string(i, str);
                str << "]);\n"sv;
            }
        }
        printCount++;
    }
    str << "}\n"sv;
}
void StringStateVisitor::visit(const CommentStmt *state) {
#ifndef NDEBUG
    str << "/* " << state->comment() << " */\n";
#endif
}
void StringStateVisitor::visit(const IfStmt *state) {
    if (emit_plan != nullptr) {
        if (auto it = emit_plan->const_if_taken.find(state);
            it != emit_plan->const_if_taken.end()) {
            util->opt->cond_opt_value = CodegenStackData::CondOptValue::None;
            const ScopeStmt *taken = it->second ? state->true_branch() : state->false_branch();
            if (!taken->statements().empty()) {
                Scope scope{this};
                taken->accept(*this);
            }
            return;
        }
    }
    if ((luisa::to_underlying(util->opt->cond_opt_value) & luisa::to_underlying(CodegenStackData::CondOptValue::Branch)) != 0) {
        str << "[[branch]] "sv;
    } else if ((luisa::to_underlying(util->opt->cond_opt_value) & luisa::to_underlying(CodegenStackData::CondOptValue::Flatten)) != 0) {
        str << "[[flatten]] "sv;
    }
    util->opt->cond_opt_value = CodegenStackData::CondOptValue::None;
    str << "if(";
    state->condition()->accept(*this);
    str << ")";
    {
        Scope scope{this};
        state->true_branch()->accept(*this);
    }
    if (!state->false_branch()->statements().empty()) {
        str << "else";
        {
            Scope scope{this};
            state->false_branch()->accept(*this);
        }
    }
}
void StringStateVisitor::visit(const LoopStmt *state) {
    str << "while(true)";
    {
        Scope scope{this};
        state->body()->accept(*this);
    }
}
void StringStateVisitor::visit(const ExprStmt *state) {
    state->expression()->accept(*this);
    str << ";\n";
}
void StringStateVisitor::visit(const SwitchStmt *state) {
    if (emit_plan != nullptr) {
        if (auto it = emit_plan->const_switch_branch.find(state);
            it != emit_plan->const_switch_branch.end()) {
            util->opt->cond_opt_value = CodegenStackData::CondOptValue::None;
            const Statement *chosen = it->second;
            if (chosen != nullptr) {
                const ScopeStmt *chosen_body =
                    chosen->tag() == Statement::Tag::SWITCH_DEFAULT ?
                        static_cast<SwitchDefaultStmt const *>(chosen)->body() :
                        static_cast<SwitchCaseStmt const *>(chosen)->body();
                if (!chosen_body->statements().empty()) {
                    _emit_folded_scope(chosen_body, emit_plan->const_switch_needs_guard.contains(state));
                }
            }
            return;
        }
    }
    if ((luisa::to_underlying(util->opt->cond_opt_value) & luisa::to_underlying(CodegenStackData::CondOptValue::Branch)) != 0) {
        str << "[[branch]] "sv;
    } else if ((luisa::to_underlying(util->opt->cond_opt_value) & luisa::to_underlying(CodegenStackData::CondOptValue::Flatten)) != 0) {
        str << "[[flatten]] "sv;
    }
    if ((luisa::to_underlying(util->opt->cond_opt_value) & luisa::to_underlying(CodegenStackData::CondOptValue::ForceCase)) != 0) {
        str << "[[forcecase]] "sv;
    }
    switchCount.emplace_back(0);
    auto dsp = vstd::scope_exit([&]() {
        switchCount.pop_back();
    });
    util->opt->cond_opt_value = CodegenStackData::CondOptValue::None;
    str << "switch(";
    state->expression()->accept(*this);
    str << ")";
    {
        Scope scope{this};
        state->body()->accept(*this);
        if (switchCount.back() == 0) {
            str << "default: break;\n";
        }
    }
}
void StringStateVisitor::visit(const SwitchCaseStmt *state) {
    if (state->body()->statements().empty()) {
        return;
    }
    switchCount.back()++;
    for (auto expression : state->expressions()) {
        str << "case ";
        expression->accept(*this);
        str << ":";
    }
    {
        Scope scope{this};
        state->body()->accept(*this);
    }
    if (std::none_of(state->body()->statements().begin(),
                     state->body()->statements().end(),
                     [](const auto &stmt) {
                         return stmt->tag() == Statement::Tag::BREAK;
                     })) {
        str << "break;\n";
    }
}
void StringStateVisitor::visit(const SwitchDefaultStmt *state) {
    if (state->body()->statements().empty()) {
        return;
    }
    switchCount.back()++;
    str << "default:";
    {
        Scope scope{this};
        state->body()->accept(*this);
    }
    if (std::none_of(state->body()->statements().begin(),
                     state->body()->statements().end(),
                     [](const auto &stmt) {
                         return stmt->tag() == Statement::Tag::BREAK;
                     })) {
        str << "break;\n";
    }
}

void StringStateVisitor::visit(const AssignStmt *state) {
    if (is_local_undefined_lifetime_seed(state)) { return; }
    if (emit_plan != nullptr) {
        if (emit_plan->skip_assigns.contains(state)) { return; }
        if (auto it = emit_plan->ssa_decl_assigns.find(state);
            it != emit_plan->ssa_decl_assigns.end()) {
            auto const &v = it->second;
            if (v.type()->is_scalar() || v.type()->is_vector()) {
                str << "const "sv;
            }
            vstd::StringBuilder typeName;
            util->GetTypeName(*v.type(), typeName, f.variable_usage(v.uid()));
            vstd::StringBuilder varName;
            util->GetVariableName(f, v, varName);
            util->RegistStructType(v.type());
            str << typeName << ' ' << varName << '=';
            state->rhs()->accept(*this);
            str << ";\n";
            return;
        }
    }
    // A member that lives in a packed 32-bit word (a 1-byte bool mixed with 16-bit
    // scalars, see `PackedStructMember`) is part of the word's single `uint`
    // member, so writing it is a read-modify-write through the matching
    // `_lc_pack_set_*` helper.
    if (state->lhs()->tag() == Expression::Tag::MEMBER &&
        !static_cast<MemberExpr const *>(state->lhs())->is_swizzle()) {
        auto member = static_cast<MemberExpr const *>(state->lhs());
        if (auto packed = util->GetPackedStructMember(
                member->self()->type(), member->member_index())) {
            if (expression_has_call(member->self())) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION(
                    "Cannot assign to the packed struct member '{}': its owner "
                    "calls a function, and writing the member needs the owner "
                    "twice.",
                    member->type()->description());
            }
            vstd::StringBuilder owner;
            StringStateVisitor vis(f, owner, util);
            vis.emit_plan = emit_plan;
            member->self()->accept(vis);
            str << owner << ".v"sv << vstd::to_string(packed->word_index)
                << "=_lc_pack_set_"sv << packed_kind_suffix(packed->kind) << '('
                << owner << ".v"sv << vstd::to_string(packed->word_index)
                << ',' << vstd::to_string(packed->byte_offset) << ',';
            state->rhs()->accept(*this);
            str << ");\n";
            return;
        }
    }
    auto is_shared = [&](const Expression *x) noexcept {
        if (x->tag() == Expression::Tag::REF) {
            auto v = static_cast<RefExpr const *>(x)->variable();
            if (v.tag() == Variable::Tag::SHARED) {
                return true;
            }
        }
        return false;
    };
    auto is_custom = [&](const Expression *x, Variable &v) noexcept {
        if (x->tag() == Expression::Tag::REF) {
            v = static_cast<RefExpr const *>(x)->variable();
            if (v.type()->is_custom()) {
                return true;
            }
        }
        return false;
    };
    // shared variables are not wrapped in array
    // structs, so some hack is necessary
    bool isLazyDecl = false;
    Variable rqVar;
    bool lhs_is_shared{false};
    auto is_rayquery = [&]() {
        if (state->rhs()->tag() != Expression::Tag::CALL) return false;
        auto s = static_cast<CallExpr const *>(state->rhs());
        return (s->op() == CallOp::RAY_TRACING_QUERY_ANY || s->op() == CallOp::RAY_TRACING_QUERY_ALL);
    };
    if (is_custom(state->lhs(), rqVar)) {
        auto iter = lazyDeclVars.find(rqVar);
        vstd::StringBuilder var_name;
        util->GetVariableName(f, rqVar, var_name);
        if (iter != lazyDeclVars.end()) {
            util->GetTypeName(*rqVar.type(), str, Usage::READ);
            str << ' '
                << var_name;
            lazyDeclVars.erase(iter);
            isLazyDecl = true;
        }
        if (is_rayquery()) {
            if (isLazyDecl) {
                str << ";\n";
                if (!util->opt->isSpirv) {
                    str << "float "sv << var_name << "_tmin=0.0;\n"sv;
                }
            }
            state->rhs()->accept(*this);
            str << ',';
            state->lhs()->accept(*this);
            if (!util->opt->isSpirv) {
                str << ',';
                state->lhs()->accept(*this);
                str << "_tmin"sv;
            }
            str << ");\n";
            return;
        }
    }
    // Detect assignment to cooperative vector element: v[i] = x → v.Set(x, i)
    auto lhs_access = state->lhs()->tag() == Expression::Tag::ACCESS ?
                          static_cast<AccessExpr const *>(state->lhs()) :
                          nullptr;
    bool lhs_is_coopvec_element = lhs_access &&
                                  lhs_access->range()->type()->is_cooperative_vector();
    if (lhs_is_coopvec_element) {
        // Generate: range.Set(rhs, index)
        lhs_access->range()->accept(*this);
        str << ".Set(";
        state->rhs()->accept(*this);
        str << ',';
        lhs_access->index()->accept(*this);
        str << ");\n";
        return;
    }
    // DXC can miscompile `m[col] = floatN(...)` (and later mat-vec products) when
    // columns share components. Store each element instead of a vector write.
    bool lhs_is_matrix_column = lhs_access &&
                                lhs_access->range()->type()->is_matrix() &&
                                state->rhs()->type() &&
                                state->rhs()->type()->is_vector();
    if (lhs_is_matrix_column) {
        auto dim = lhs_access->range()->type()->dimension();
        auto tmp = util->GetNewTempVarName();
        util->GetTypeName(*state->rhs()->type(), str, Usage::READ);
        str << ' ' << tmp << ";\n"sv;
        str << tmp << '=';
        state->rhs()->accept(*this);
        str << ";\n"sv;
        static constexpr char xyzw[] = "xyzw";
        for (auto i = 0u; i < dim; i++) {
            accessCount++;
            lhs_access->range()->accept(*this);
            accessCount--;
            str << '[';
            lhs_access->index()->accept(*this);
            str << "]["sv;
            vstd::to_string(i, str);
            str << "]="sv;
            str << tmp << '.' << xyzw[i] << ";\n"sv;
        }
        return;
    }
    auto rhs_is_shared = is_shared(state->rhs());
    if (!isLazyDecl) {
        lhs_is_shared = is_shared(state->lhs());
        if (!lhs_is_shared && rhs_is_shared) {
            str << "(";
            state->lhs()->accept(*this);
            str << ").v";
        } else {
            state->lhs()->accept(*this);
        }
    }
    str << '=';
    // The AST keeps a plain literal on the right-hand side of a BF16
    // declaration ("BFloat16 zero = 0.0f;") and relies on the target type's
    // implicit conversion.  Software BF16 has no such conversion (its storage is
    // a raw 16-bit pattern), so the narrowing is explicit here.
    auto rhs_needs_bf16_narrow =
        util->opt->bf16_fallback &&
        CodegenUtility::IsBf16(state->lhs()->type()) &&
        !CodegenUtility::IsBf16(state->rhs()->type());
    if (rhs_needs_bf16_narrow) {
        str << "_lc_bf16_from_f32("sv;
    }
    if (lhs_is_shared && !rhs_is_shared) {
        str << "(";
        state->rhs()->accept(*this);
        str << ").v";
    } else {
        state->rhs()->accept(*this);
    }
    if (rhs_needs_bf16_narrow) {
        str << ')';
    }
    str << ";\n";
}
void StringStateVisitor::visit(const ForStmt *state) {
    str << "for(";
    //    state->variable()->accept(*this);
    str << ';';
    state->condition()->accept(*this);
    str << ';';
    state->variable()->accept(*this);
    str << "+=";
    state->step()->accept(*this);
    str << ")";
    {
        Scope scope{this};
        state->body()->accept(*this);
    }
}
void StringStateVisitor::visit(const RayQueryStmt *stmt) {
    // Ray queries (inline or the RayQuery object form) are not implemented by
    // the software ray-tracing fallback: it has no candidate/committed-hit
    // machinery to drive this loop.  Report it instead of emitting a loop over
    // a resource the fallback never declares.
    if (util->opt->fallback_rtx) {
        LUISA_ERROR("RayQuery (RAY_TRACING_QUERY_*) is not available on the "
                    "software ray-tracing fallback "
                    "(DeviceConfigExt::use_fallback_rtx()).  The fallback answers "
                    "closest-hit and any-hit tracing only; run on a device with "
                    "hardware ray tracing and leave use_fallback_rtx() false, or "
                    "rewrite the shader with Accel::intersect()/intersect_any().");
    }
    str << "{\n"sv;
    str << "while("sv;
    stmt->query()->accept(*this);
    str << ".Proceed()){\n"sv
        << "if("sv;
    stmt->query()->accept(*this);
    str << ".CandidateType()==CANDIDATE_NON_OPAQUE_TRIANGLE){\n"sv;
    stmt->on_triangle_candidate()->accept(*this);
    str << "}else{\n"sv;
    stmt->on_procedural_candidate()->accept(*this);
    str << "}}}\n"sv;
}
void StringStateVisitor::visit(const SuspendStmt *) {
}

void StringStateVisitor::visit(const DebugBreakStmt *stmt) {
    if (util->opt->isSpirv && util->opt->spirv_capabilities.abort) {
        // VK_KHR_shader_abort: OpAbortKHR ends the invocation that executes it and
        // hands one 32-bit word to the client API (vkGetDeviceFaultDebugInfoKHR).
        // The AST carries no text for a break point - `watches` are debugger
        // expressions that never reach the device - so the word is the low half of
        // the statement's hash: the host enumerates the debug breaks of the same
        // Function and matches the value it receives to one of them.
        str << "vk::Abort("sv
            << luisa::format("0x{:08x}u", static_cast<uint32_t>(stmt->hash()))
            << ");\n"sv;
        return;
    }
    LUISA_ERROR(
        "Debug break is not supported for the target of this shader: it lowers to "
        "OpAbortKHR, which needs a SPIR-V device with VK_KHR_shader_abort "
        "(src/backends/vk/vk_shader_capabilities.h), and DXIL has no equivalent "
        "instruction. Enable the extension on the device, or remove $debug_break.");
}
StringStateVisitor::StringStateVisitor(
    Function f,
    vstd::StringBuilder &str,
    CodegenUtility *util)
    : f(f), util(util), str(str), switchCount{}, lazyDeclVars{} {
}
void StringStateVisitor::VisitFunction(Function func) {
    lazyDeclVars.clear();
    luisa::unordered_set<uint32_t> lazy_decl_uids;
    for (auto &&v : func.local_variables()) {
        Usage usage = func.variable_usage(v.uid());
        if (usage == Usage::NONE) [[unlikely]] {
            continue;
        }
        if (v.type()->tag() == Type::Tag::CUSTOM) {
            auto desc = v.type()->description();
            // rayquery need specialization to workaround DXC's bug
            if (desc == "LC_RayQueryAll"sv || desc == "LC_RayQueryAny"sv) {
                lazyDeclVars.emplace(v);
                lazy_decl_uids.emplace(v.uid());
                continue;
            }
        }
    }
    const bool enable_local_opt =
        !luisa::compute::detail::env_flag("LUISA_HLSL_DISABLE_LOCAL_OPT");
    LocalEmitPlan local_plan;
    if (enable_local_opt) {
        local_plan = build_local_emit_plan(func, util, lazy_decl_uids);
        emit_plan = &local_plan;
        // Emit non-eligible locals (resources, custom types, coop vectors) at function top.
        for (auto &&v : func.local_variables()) {
            Usage usage = func.variable_usage(v.uid());
            if (usage == Usage::NONE || lazy_decl_uids.contains(v.uid())) {
                continue;
            }
            if (v.type()->is_resource() || v.type()->is_custom() || v.type()->is_cooperative_vector()) {
                _emit_local_zero_decl(v);
            }
        }
        // Emit function-body-scoped mutable locals at function top.
        if (auto it = local_plan.scoped_decls.find(func.body());
            it != local_plan.scoped_decls.end()) {
            for (auto &&v : it->second) {
                _emit_local_zero_decl(v);
            }
        }
    } else {
        emit_plan = nullptr;
        for (auto &&v : func.local_variables()) {
            Usage usage = func.variable_usage(v.uid());
            if (usage == Usage::NONE || lazy_decl_uids.contains(v.uid())) {
                continue;
            }
            _emit_local_zero_decl(v);
        }
    }
    auto clear_plan = vstd::scope_exit([this]() { emit_plan = nullptr; });
    if (sharedVariables) {
        size_t shared_size{};
        for (auto &&v : func.shared_variables()) {
            // FIXME: redundant creation of string
            vstd::StringBuilder typeName;
            util->GetTypeName(*v.type(), typeName, f.variable_usage(v.uid()));
            sharedVariables->emplace(f, v);
            shared_size += v.type()->size();
        }

        // SM 6.10 will increase to 64kb in the future
        LUISA_ASSERT(shared_size <= 32768, "Shared memory size must be less than 32kb.");
    }
    func.body()->accept(*this);
}

StringStateVisitor::Scope::Scope(StringStateVisitor *self)
    : self(self) {
    self->str << "{\n"sv;
}
}// namespace lc::hlsl
