#include <ast/ast_evaluator.h>
#include <core/logging.h>
#include <algorithm>
#include <core/mathematics.h>

namespace luisa::compute {

ASTEvaluator::Result ASTEvaluator::try_eval(Expression const *expr) {
    switch (expr->tag()) {
        case Expression::Tag::UNARY:
            return try_eval(static_cast<UnaryExpr const *>(expr));
        case Expression::Tag::BINARY:
            return try_eval(static_cast<BinaryExpr const *>(expr));
        case Expression::Tag::MEMBER:
            return try_eval(static_cast<MemberExpr const *>(expr));
        case Expression::Tag::ACCESS:
            return try_eval(static_cast<AccessExpr const *>(expr));
        case Expression::Tag::LITERAL:
            return try_eval(static_cast<LiteralExpr const *>(expr));
        case Expression::Tag::REF:
            return try_eval(static_cast<RefExpr const *>(expr));
        case Expression::Tag::CALL:
            return try_eval(static_cast<CallExpr const *>(expr));
        case Expression::Tag::CAST:
            return try_eval(static_cast<CastExpr const *>(expr));
    }
    return monostate{};
}

namespace analyzer_detail {

template<typename T>
constexpr size_t TypeImportance() {
    if constexpr (std::is_same_v<T, bool>) {
        return 10;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return 20;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return 20;
    } else if constexpr (std::is_same_v<T, float>) {
        return 30;
    } else {
        static_assert(luisa::always_false_v<T>, "illegal type");
    }
}

template<typename A, typename B>
constexpr decltype(auto) TypeCast() {
    if constexpr (TypeImportance<A>() >= TypeImportance<B>()) {
        return std::type_identity<A>{};
    } else {
        return std::type_identity<B>{};
    }
}

template<typename T>
struct ScalarType {
    using type = T;
    static constexpr size_t size = 1;
    static constexpr bool is_scalar = true;
    static constexpr bool is_matrix = false;
    static constexpr bool is_vector = false;
};

template<typename T, size_t N>
struct ScalarType<Vector<T, N>> {
    using type = T;
    static constexpr size_t size = N;
    static constexpr bool is_scalar = false;
    static constexpr bool is_matrix = false;
    static constexpr bool is_vector = true;
};

template<size_t N>
struct ScalarType<Matrix<N>> {
    using type = float;
    static constexpr size_t size = N;
    static constexpr bool is_scalar = false;
    static constexpr bool is_matrix = true;
    static constexpr bool is_vector = false;
};

template<typename T>
using ScalarType_t = typename ScalarType<T>::type;

}// namespace analyzer_detail

ASTEvaluator::Result ASTEvaluator::try_eval(UnaryExpr const *expr) {
    auto result = try_eval(expr->operand());
    if (holds_alternative<monostate>(result)) [[likely]] {
        return monostate{};
    }
    using namespace analyzer_detail;
    Result r;
    return visit(
        [&]<typename T>(T const &v) -> Result {
            using TScalar = ScalarType_t<T>;

            switch (expr->op()) {
                case UnaryOp::PLUS:
                    return Result{v};
                case UnaryOp::NOT:
                case UnaryOp::MINUS: {
                    if constexpr (std::is_same_v<TScalar, bool>) {
                        return Result{!v};
                    } else if constexpr (ScalarType<T>::is_matrix) {
                        return Result{v * -1.0f};
                    } else if constexpr (!std::is_same_v<T, monostate>) {
                        return Result{-v};
                    }
                }
                case UnaryOp::BIT_NOT:
                    if constexpr (std::is_same_v<TScalar, int32_t> || std::is_same_v<TScalar, uint32_t>) {
                        return Result{~v};
                    }
            }
            return monostate{};
        },
        result);
}

ASTEvaluator::Result ASTEvaluator::try_eval(BinaryExpr const *expr) {
    auto lr = try_eval(expr->lhs());
    if (holds_alternative<monostate>(lr)) [[likely]]
        return monostate{};
    auto rr = try_eval(expr->rhs());
    using namespace analyzer_detail;
    if (rr.index() != 0) [[unlikely]] {
        if (expr->op() == BinaryOp::MUL) {
            if (auto [lhs, rhs] = std::make_pair(get_if<float2x2>(&lr), get_if<float2>(&rr));
                lhs != nullptr && rhs != nullptr) { return (*lhs) * (*rhs); }
            if (auto [lhs, rhs] = std::make_pair(get_if<float3x3>(&lr), get_if<float3>(&rr));
                lhs != nullptr && rhs != nullptr) { return (*lhs) * (*rhs); }
            if (auto [lhs, rhs] = std::make_pair(get_if<float4x4>(&lr), get_if<float4>(&rr));
                lhs != nullptr && rhs != nullptr) { return (*lhs) * (*rhs); }
        }
        // Transform rr
        bool trans_success = false;
        if (lr.index() != rr.index()) {
            visit(
                [&]<typename A, typename B>(A a, B b) {
                    if constexpr (!std::is_same_v<A, monostate> && !std::is_same_v<B, monostate>) {
                        using ScalarA = ScalarType_t<A>;
                        using ScalarB = ScalarType_t<B>;
                        using TTA = ScalarType<A>;
                        using TTB = ScalarType<B>;
                        using DstScalar = typename decltype(TypeCast<ScalarA, ScalarB>())::type;
                        // vec + scalar
                        if constexpr (TTA::is_vector && TTB::is_scalar) {
                            using VecType = Vector<DstScalar, TTA::size>;
                            VecType a_result, b_result;
                            for (auto &&i : range(TTA::size)) {
                                a_result[i] = static_cast<DstScalar>(a[i]);
                                b_result[i] = static_cast<DstScalar>(b);
                            }
                            lr = a_result;
                            rr = b_result;
                            trans_success = true;
                        }
                        // scalar + vec
                        else if constexpr (TTA::is_scalar && TTB::is_vector) {
                            using VecType = Vector<DstScalar, TTB::size>;
                            VecType a_result, b_result;
                            for (auto &&i : range(TTB::size)) {
                                a_result[i] = static_cast<DstScalar>(a);
                                b_result[i] = static_cast<DstScalar>(b[i]);
                            }
                            lr = a_result;
                            rr = b_result;
                            trans_success = true;
                        }
                        // scalar + scalar
                        else if constexpr (TTA::is_scalar && TTB::is_scalar) {
                            DstScalar a_result, b_result;
                            a_result = static_cast<DstScalar>(a);
                            b_result = static_cast<DstScalar>(b);
                            lr = a_result;
                            rr = b_result;
                            trans_success = true;
                        } else if constexpr (TTA::is_matrix && TTB::is_scalar) {
                            using VecType = Matrix<TTA::size>;
                            VecType b_result;
                            for (auto x : range(TTA::size))
                                for (auto y : range(TTA::size)) {
                                    b_result[x][y] = static_cast<float>(b);
                                }
                            rr = b_result;
                            trans_success = true;
                        }
                        // scalar + vec
                        else if constexpr (TTA::is_scalar && TTB::is_matrix) {
                            using VecType = Matrix<TTB::size>;
                            VecType a_result;
                            for (auto x : range(TTA::size))
                                for (auto y : range(TTA::size)) {
                                    a_result[x][y] = static_cast<float>(a);
                                }
                            lr = a_result;
                            trans_success = true;
                        }
                    }
                },
                lr, rr);
        }
        if (!trans_success) return monostate{};
        return visit(
            [&]<typename A>(A const &a) -> Result {
                if constexpr (!std::is_same_v<A, monostate>) {
                    using TT = ScalarType<A>;
                    using TScalar = ScalarType_t<A>;
                    auto b = get<A>(rr);
                    // TODO
                    switch (expr->op()) {
                        case BinaryOp::ADD:
                            if constexpr (std::is_same_v<TScalar, bool>) {
                                return monostate{};
                            } else if constexpr (!std::is_same_v<A, monostate>) {
                                return Result{a + b};
                            }
                            break;
                        case BinaryOp::SUB:
                            if constexpr (std::is_same_v<TScalar, bool>) {
                                return monostate{};
                            } else if constexpr (!std::is_same_v<A, monostate>) {
                                return Result{a - b};
                            }
                            break;
                        case BinaryOp::MUL:
                            if constexpr (std::is_same_v<TScalar, bool>) {
                                return monostate{};
                            } else if constexpr (!std::is_same_v<A, monostate>) {
                                return Result{a * b};
                            }
                            break;
                        case BinaryOp::DIV:
                            if constexpr (std::is_same_v<TScalar, bool>) {
                                return monostate{};
                            } else if constexpr (TT::is_matrix) {
                                return monostate{};
                            } else if constexpr (!std::is_same_v<A, monostate>) {
                                return Result{a / b};
                            }
                            break;
                        case BinaryOp::MOD:
                            if constexpr (std::is_same_v<TScalar, int32_t> || std::is_same_v<TScalar, uint32_t>) {
                                return Result{a % b};
                            }
                            break;
                        case BinaryOp::BIT_AND:
                            if constexpr (std::is_same_v<TScalar, int32_t> || std::is_same_v<TScalar, uint32_t> || (std::is_same_v<TScalar, bool> && TT::is_scalar)) {
                                return Result{a & b};
                            }
                            break;
                        case BinaryOp::BIT_OR:
                            if constexpr (std::is_same_v<TScalar, int32_t> || std::is_same_v<TScalar, uint32_t> || (std::is_same_v<TScalar, bool> && TT::is_scalar)) {
                                return Result{a | b};
                            }
                            break;
                        case BinaryOp::BIT_XOR:
                            if constexpr (std::is_same_v<TScalar, int32_t> || std::is_same_v<TScalar, uint32_t> || (std::is_same_v<TScalar, bool> && TT::is_scalar)) {
                                return Result{a ^ b};
                            }
                            break;
                        case BinaryOp::SHL:
                            if constexpr (std::is_same_v<TScalar, int32_t> || std::is_same_v<TScalar, uint32_t>) {
                                return Result{a << b};
                            }
                            break;
                        case BinaryOp::SHR:
                            if constexpr (std::is_same_v<TScalar, int32_t> || std::is_same_v<TScalar, uint32_t>) {
                                return Result{a >> b};
                            }
                            break;
                        case BinaryOp::AND:
                            if constexpr (std::is_same_v<TScalar, bool>) {
                                return Result{a && b};
                            }
                            break;
                        case BinaryOp::OR:
                            if constexpr (std::is_same_v<TScalar, bool>) {
                                return Result{a || b};
                            }
                            break;
                        case BinaryOp::LESS:
                            if constexpr (!std::is_same_v<TScalar, bool> && !TT::is_matrix) {
                                return Result{a < b};
                            }
                            break;
                        case BinaryOp::GREATER:
                            if constexpr (!std::is_same_v<TScalar, bool> && !TT::is_matrix) {
                                return Result{a > b};
                            }
                            break;
                        case BinaryOp::LESS_EQUAL:
                            if constexpr (!std::is_same_v<TScalar, bool> && !TT::is_matrix) {
                                return Result{a <= b};
                            }
                            break;
                        case BinaryOp::GREATER_EQUAL:
                            if constexpr (!std::is_same_v<TScalar, bool> && !TT::is_matrix) {
                                return Result{a >= b};
                            }
                            break;
                        case BinaryOp::EQUAL:
                            if constexpr (!TT::is_matrix) {
                                return Result{a == b};
                            }
                            break;
                        case BinaryOp::NOT_EQUAL:
                            if constexpr (!TT::is_matrix) {
                                return Result{a != b};
                            }
                            break;
                    }
                }
                return monostate{};
            },
            lr);
    }
    return monostate{};
}

ASTEvaluator::Result ASTEvaluator::try_eval(MemberExpr const *expr) {
    if (expr->is_swizzle()) {
        auto self = try_eval(expr->self());
        if (holds_alternative<monostate>(self)) [[likely]]
            return monostate{};
        using namespace analyzer_detail;
        return visit(
            [&]<typename T>(T const &t) -> Result {
                if constexpr (ScalarType<T>::is_vector) {
                    T newT;
                    for (auto i : range(expr->swizzle_size())) {
                        newT[i] = t[expr->swizzle_index(i)];
                    }
                    return Result{newT};
                } else {
                    return monostate{};
                }
            },
            self);
    }
    return monostate{};
}

ASTEvaluator::Result ASTEvaluator::try_eval(AccessExpr const *expr) {
    if (expr->range()->tag() != Expression::Tag::CONSTANT) [[likely]]
        return monostate{};
    auto index_result = try_eval(expr->index());
    if (holds_alternative<monostate>(index_result)) [[likely]]
        return monostate{};
    int64_t index = visit(
        [&]<typename T>(T const &t) -> int64_t {
            if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>) {
                return t;
            } else {
                return 0;
            }
        },
        index_result);
    return visit(
        [&]<typename T>(span<T const> const &t) -> Result {
            return t[index];
        },
        static_cast<ConstantExpr const *>(expr->range())->data().view());
}

ASTEvaluator::Result ASTEvaluator::try_eval(LiteralExpr const *expr) {
    Result r;
    visit([&](auto &&a) { r = a; }, expr->value());
    return r;
}

ASTEvaluator::ASTEvaluator() noexcept
    : branch_scope{} {
    var_values.emplace_back(false);
}

ASTEvaluator::Result ASTEvaluator::try_eval(RefExpr const *expr) {
    auto uid = expr->variable().uid();
    for (int32_t idx = branch_scope; idx >= 0; --idx) {
        auto &&map = var_values[idx];
        if (map.is_loop) return monostate{};
        auto ite = map.variables.find(uid);
        if (ite == map.variables.end()) continue;
        return ite->second;
    }
    return monostate{};
}

ASTEvaluator::Result ASTEvaluator::assign(Expression const *lhs, Expression const *rhs) {
    if (lhs->tag() != Expression::Tag::REF) [[unlikely]]
        return monostate{};
    auto var_index = static_cast<RefExpr const *>(lhs)->variable().uid();
    auto &&map = var_values[branch_scope];
    if (map.is_loop) {
        map.variables.insert_or_assign(var_index, monostate{});
        return monostate{};
    } else {
        auto result = try_eval(rhs);
        map.variables.insert_or_assign(var_index, result);
        return result;
    }
}

void ASTEvaluator::ref_var(Variable var) {
    auto var_index = var.uid();
    auto &&map = var_values[branch_scope];
    map.variables.insert_or_assign(var_index, monostate{});
}

void ASTEvaluator::assign(AssignStmt const *stmt) {
    assign(stmt->lhs(), stmt->rhs());
}

Statement const *ASTEvaluator::map_if(IfStmt const *stmt) {
    auto result = try_eval(stmt->condition());
    if (holds_alternative<monostate>(result)) [[likely]]
        return nullptr;
    return visit(
        [&]<typename T>(T const &t) -> Statement const * {
            if constexpr (std::is_same_v<T, bool>) {
                if (t) {
                    return stmt->true_branch();
                } else {
                    return stmt->false_branch();
                }
            } else {
                return nullptr;
            }
        },
        result);
}

void ASTEvaluator::execute_for(ForStmt const *stmt) {
    if (stmt->variable()->tag() != Expression::Tag::REF) return;
    ref_var(static_cast<RefExpr const *>(stmt->variable())->variable());
}

void ASTEvaluator::begin_switch(SwitchStmt const *stmt) {
    switch_scopes.emplace_back(try_eval(stmt->expression()));
}

bool ASTEvaluator::check_switch_case(SwitchCaseStmt const *stmt) {
    auto &&result = switch_scopes.back();
    if (holds_alternative<monostate>(result)) [[likely]]
        return true;
    auto case_result = try_eval(stmt->expression());
    if (case_result.index() != result.index()) [[likely]]
        return true;
    return visit(
        [&]<typename A>(A const &a) -> bool {
            auto b = get<A>(case_result);
            if constexpr (std::is_same_v<A, int32_t> || std::is_same_v<A, uint32_t> || std::is_same_v<A, bool> || std::is_same_v<A, float>) {
                return a == b;
            } else {
                return true;
            }
        },
        result);
}

void ASTEvaluator::end_switch() {
    if (switch_scopes.empty()) [[unlikely]] {
        LUISA_ERROR("Switch scope empty!");
    }
    switch_scopes.pop_back();
}

void ASTEvaluator::begin_branch_scope(bool is_loop) {
    auto &branch = var_values.emplace_back(is_loop);
    branch_scope++;
}

void ASTEvaluator::end_branch_scope() {
    auto last_map = std::move(var_values.back());
    var_values.pop_back();
    auto &&next_map = var_values.back();
    for (auto &&ele : last_map.variables) {
        next_map.variables.insert_or_assign(ele.first, monostate{});
    }
    branch_scope--;
}

void ASTEvaluator::check_call_ref(Function func, luisa::span<Expression const *const> args_var) {
    auto argTypes = func.arguments();
    for (auto i : range(args_var.size())) {
        if (args_var[i]->tag() == Expression::Tag::REF && argTypes[i].tag() == Variable::Tag::REFERENCE) {
            ref_var(static_cast<RefExpr const *>(args_var[i])->variable());
        }
    }
}

ASTEvaluator::Result ASTEvaluator::try_eval(CallExpr const *expr) {
    using namespace analyzer_detail;
    switch (expr->op()) {
        case CallOp::CUSTOM: {
            check_call_ref(expr->custom(), expr->arguments());
            return monostate{};
        }
        case CallOp::ALL: {
            auto result = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(result)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, bool>) {
                        if constexpr (std::is_same_v<T, bool>) {
                            return t;
                        } else {
                            bool result = true;
                            for (auto i : range(ScalarType<T>::size)) {
                                result &= t[i];
                            }
                            return result;
                        }
                    } else {
                        return monostate{};
                    }
                },
                result);
        } break;
        case CallOp::ANY: {
            auto result = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(result)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, bool>) {
                        if constexpr (std::is_same_v<T, bool>) {
                            return t;
                        } else {
                            bool result = false;
                            for (auto i : range(ScalarType<T>::size)) {
                                result |= t[i];
                            }
                            return result;
                        }
                    } else {
                        return monostate{};
                    }
                },
                result);
        } break;
        case CallOp::SELECT: {
            auto result_a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(result_a)) [[likely]] { return monostate{}; }
            auto result_b = try_eval(expr->arguments()[1]);
            if (holds_alternative<monostate>(result_b)) [[likely]] { return monostate{}; }
            auto result_t = try_eval(expr->arguments()[2]);
            if (holds_alternative<monostate>(result_t)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, bool>) {
                        return visit(
                            [&]<typename A>(A const &a) -> Result {
                                if constexpr (ScalarType<A>::size == ScalarType<T>::size && !ScalarType<A>::is_matrix && !std::is_same_v<A, monostate>) {
                                    auto b = get<A>(result_b);
                                    return select(a, b, t);
                                } else {
                                    return monostate{};
                                }
                            },
                            result_a);
                    } else {
                        return monostate{};
                    }
                },
                result_t);
            break;
        }
        case CallOp::CLAMP: {
            auto value = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(value)) [[likely]] { return monostate{}; }
            auto min_value = try_eval(expr->arguments()[1]);
            if (holds_alternative<monostate>(min_value)) [[likely]] { return monostate{}; }
            auto max_value = try_eval(expr->arguments()[2]);
            if (holds_alternative<monostate>(max_value)) [[likely]] { return monostate{}; }
            if (value.index() != min_value.index() || value.index() != max_value.index()) return monostate{};
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (!std::is_same_v<T, monostate> && !ScalarType<T>::is_matrix) {
                        auto min_v = get<T>(min_value);
                        auto max_v = get<T>(max_value);
                        return clamp(t, min_v, max_v);
                    } else {
                        return monostate{};
                    }
                },
                value);
        } break;
        case CallOp::SATURATE: {
            auto value = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(value)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (!std::is_same_v<T, monostate> && !ScalarType<T>::is_matrix) {
                        if constexpr (ScalarType<T>::is_scalar) {
                            return std::clamp(t, static_cast<T>(0), static_cast<T>(1));
                        } else {
                            T result;
                            for (auto i : range(ScalarType<T>::size)) {
                                result[i] = std::clamp(t[i], static_cast<ScalarType_t<T>>(0), static_cast<ScalarType_t<T>>(1));
                            }
                            return result;
                        }

                    } else {
                        return monostate{};
                    }
                },
                value);
        };
        case CallOp::LERP: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            auto b = try_eval(expr->arguments()[1]);
            if (holds_alternative<monostate>(b)) [[likely]] { return monostate{}; }
            auto t = try_eval(expr->arguments()[2]);
            if (holds_alternative<monostate>(t)) [[likely]] { return monostate{}; }
            if (a.index() != t.index() || a.index() != b.index()) return monostate{};
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (!std::is_same_v<T, monostate> && !ScalarType<T>::is_matrix && std::is_same_v<ScalarType_t<float>, bool>) {
                        auto a_value = get<T>(a);
                        auto b_value = get<T>(b);
                        return lerp(a_value, b_value, t);
                    } else {
                        return monostate{};
                    }
                },
                t);
        } break;
        case CallOp::STEP: {
            auto y = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(y)) [[likely]] { return monostate{}; }
            auto x = try_eval(expr->arguments()[1]);
            if (holds_alternative<monostate>(x)) [[likely]] { return monostate{}; }
            if (y.index() != x.index()) return monostate{};
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (!std::is_same_v<T, monostate> && !ScalarType<T>::is_matrix) {
                        auto x_value = get<T>(x);
                        if constexpr (ScalarType<T>::is_scalar) {
                            return (x_value >= t) ? static_cast<T>(1) : static_cast<T>(0);
                        } else {
                            T result;
                            for (auto i : range(ScalarType<T>::size)) {
                                result[i] = (x_value[i] >= t[i]) ? static_cast<ScalarType_t<T>>(0) : static_cast<ScalarType_t<T>>(1);
                            }
                            return result;
                        }
                    } else {
                        return monostate{};
                    }
                },
                y);
        }
        case CallOp::ABS: {
            auto v = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(v)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType<T>, float> && !ScalarType<T>::is_matrix) {
                        return abs(t);
                    } else {
                        return monostate{};
                    }
                },
                v);
        } break;
        case CallOp::MIN: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            auto b = try_eval(expr->arguments()[1]);
            if (holds_alternative<monostate>(b)) [[likely]] { return monostate{}; }
            if (a.index() != b.index()) return monostate{};
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (!std::is_same_v<T, monostate> && !ScalarType<T>::is_matrix) {
                        auto b_value = get<T>(b);
                        return min(t, b_value);
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::MAX: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            auto b = try_eval(expr->arguments()[1]);
            if (holds_alternative<monostate>(b)) [[likely]] { return monostate{}; }
            if (a.index() != b.index()) return monostate{};
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (!std::is_same_v<T, monostate> && !ScalarType<T>::is_matrix) {
                        auto b_value = get<T>(b);
                        return max(t, b_value);

                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::CLZ: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            auto clz = [](uint32_t v) -> uint32_t {
                constexpr uint32_t mask = 1u << 31u;
                for (auto i : range(32)) {
                    if (v & mask) return i;
                    v <<= 1;
                }
                return 32;
            };
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, int32_t> || std::is_same_v<ScalarType_t<T>, uint32_t>) {
                        if constexpr (ScalarType<T>::is_scalar) {
                            return clz(t);
                        } else {
                            T result;
                            for (auto i : range(ScalarType<T>::size)) {
                                result[i] = clz(t[i]);
                            }
                            return result;
                        }
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::CTZ: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            auto ctz = [](uint32_t v) -> uint32_t {
                constexpr uint32_t mask = 1u;
                for (auto i : range(32)) {
                    if (v & mask) return i;
                    v >>= 1;
                }
                return 32;
            };
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, int32_t> || std::is_same_v<ScalarType_t<T>, uint32_t>) {
                        if constexpr (ScalarType<T>::is_scalar) {
                            return ctz(t);
                        } else {
                            T result;
                            for (auto i : range(ScalarType<T>::size)) {
                                result[i] = ctz(t[i]);
                            }
                            return result;
                        }
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::POPCOUNT: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            auto popcount = [](uint32_t v) -> uint32_t {
                constexpr uint32_t mask = 1u;
                uint32_t r = 0;
                for (auto i : range(32)) {
                    if (v & mask) r += 1;
                    v >>= 1;
                }
                return r;
            };
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, int32_t> || std::is_same_v<ScalarType_t<T>, uint32_t>) {
                        if constexpr (ScalarType<T>::is_scalar) {
                            return popcount(t);
                        } else {
                            T result;
                            for (auto i : range(ScalarType<T>::size)) {
                                result[i] = popcount(t[i]);
                            }
                            return result;
                        }
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::REVERSE: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            auto reverse = [](uint32_t v) -> uint32_t {
                uint32_t result = 0;
                constexpr uint32_t mask = 1u;
                for (auto i : range(32)) {
                    result <<= 1;
                    result |= (v & mask);
                    v >>= 1;
                }
                return result;
            };
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, int32_t> || std::is_same_v<ScalarType_t<T>, uint32_t>) {
                        if constexpr (ScalarType<T>::is_scalar) {
                            return reverse(t);
                        } else {
                            T result;
                            for (auto i : range(ScalarType<T>::size)) {
                                result[i] = reverse(t[i]);
                            }
                            return result;
                        }
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::ISNAN: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        return isnan(t);
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::ISINF: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        return isinf(t);
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::ACOS: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        return acos(t);
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::ACOSH: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        if constexpr (ScalarType<T>::is_scalar) {
                            return std::acosh(t);
                        } else {
                            T result;
                            for (auto i : range(ScalarType<T>::size)) {
                                result[i] = std::acosh(t[i]);
                            }
                            return result;
                        }
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::ASIN: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        return asin(t);
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::ASINH: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        if constexpr (ScalarType<T>::is_scalar) {
                            return std::asinh(t);
                        } else {
                            T result;
                            for (auto i : range(ScalarType<T>::size)) {
                                result[i] = std::asinh(t[i]);
                            }
                            return result;
                        }
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::ATAN: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        return atan(t);
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::ATANH: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        if constexpr (ScalarType<T>::is_scalar) {
                            return std::atanh(t);
                        } else {
                            T result;
                            for (auto i : range(ScalarType<T>::size)) {
                                result[i] = std::atanh(t[i]);
                            }
                            return result;
                        }
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::ATAN2: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            auto b = try_eval(expr->arguments()[1]);
            if (holds_alternative<monostate>(b)) [[likely]] { return monostate{}; }
            if (a.index() != b.index()) return monostate{};
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        auto b_value = get<T>(b);
                        return atan2(t, b_value);
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::COS: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        return cos(t);
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::COSH: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        if constexpr (ScalarType<T>::is_scalar) {
                            return std::cosh(t);
                        } else {
                            T result;
                            for (auto i : range(ScalarType<T>::size)) {
                                result[i] = std::cosh(t[i]);
                            }
                            return result;
                        }
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::SIN: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        return sin(t);
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::SINH: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        if constexpr (ScalarType<T>::is_scalar) {
                            return std::sinh(t);
                        } else {
                            T result;
                            for (auto i : range(ScalarType<T>::size)) {
                                result[i] = std::sinh(t[i]);
                            }
                            return result;
                        }
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::TAN: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        return tan(t);
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::TANH: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        if constexpr (ScalarType<T>::is_scalar) {
                            return std::tanh(t);
                        } else {
                            T result;
                            for (auto i : range(ScalarType<T>::size)) {
                                result[i] = std::tanh(t[i]);
                            }
                            return result;
                        }
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::EXP: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        return exp(t);
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::EXP2: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        if constexpr (ScalarType<T>::is_scalar) {
                            return std::exp2(t);
                        } else {
                            T result;
                            for (auto i : range(ScalarType<T>::size)) {
                                result[i] = std::exp2(t[i]);
                            }
                            return result;
                        }
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::EXP10: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        if constexpr (ScalarType<T>::is_scalar) {
                            return std::pow(10.0f, t);
                        } else {
                            T result;
                            for (auto i : range(ScalarType<T>::size)) {
                                result[i] = std::pow(10.0f, t[i]);
                            }
                            return result;
                        }
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::LOG: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        return log(t);
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::LOG2: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        return log2(t);

                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::LOG10: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        return log10(t);

                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::SQRT: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        return sqrt(t);
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::RSQRT: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        if constexpr (ScalarType<T>::is_scalar) {
                            return 1.0f / std::sqrt(t);
                        } else {
                            T result;
                            for (auto i : range(ScalarType<T>::size)) {
                                result[i] = 1.0f / std::sqrt(t[i]);
                            }
                            return result;
                        }
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::CEIL: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        return ceil(t);
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::FLOOR: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        return floor(t);
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::FRACT: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        return fract(t);
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::TRUNC: {
            auto trunc = [](float a) -> float {
                return (int32_t)a;
            };
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        if constexpr (ScalarType<T>::is_scalar) {
                            return trunc(t);
                        } else {
                            T result;
                            for (auto i : range(ScalarType<T>::size)) {
                                result[i] = trunc(t[i]);
                            }
                            return result;
                        }
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::ROUND: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        return round(t);
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::FMA: {
            auto result_a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(result_a)) [[likely]] { return monostate{}; }
            auto result_b = try_eval(expr->arguments()[1]);
            if (holds_alternative<monostate>(result_b)) [[likely]] { return monostate{}; }
            auto result_c = try_eval(expr->arguments()[2]);
            if (holds_alternative<monostate>(result_c)) [[likely]] { return monostate{}; }
            if (result_a.index() != result_b.index() || result_a.index() != result_c.index()) return monostate{};
            return visit(
                [&]<typename T>(T const &a) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        auto b = get<T>(result_b);
                        auto c = get<T>(result_c);
                        if constexpr (ScalarType<T>::is_scalar) {
                            return a * b + c;
                        } else {
                            T result;
                            for (auto i : range(ScalarType<T>::size)) {
                                result[i] = a[i] * b[i] + c[i];
                            }
                            return result;
                        }
                    } else {
                        return monostate{};
                    }
                },
                result_a);
        } break;
        case CallOp::COPYSIGN: {
            auto copysign = []<typename T>(T a, T b) -> T {
                auto v = ((reinterpret_cast<uint32_t &>(a) & 0x7fffffffu) | (reinterpret_cast<uint32_t &>(b) & 0x80000000u));
                return reinterpret_cast<float &>(v);
            };
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            auto b = try_eval(expr->arguments()[1]);
            if (holds_alternative<monostate>(b)) [[likely]] { return monostate{}; }
            if (a.index() != b.index()) { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<ScalarType_t<T>, float> && !ScalarType<T>::is_matrix) {
                        auto b_v = get<T>(b);
                        if constexpr (ScalarType<T>::is_scalar) {
                            return copysign(t, b_v);
                        } else {
                            T result;
                            for (auto i : range(ScalarType<T>::size)) {
                                result[i] = copysign(t[i], b_v[i]);
                            }
                            return result;
                        }
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::CROSS: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            auto b = try_eval(expr->arguments()[1]);
            if (holds_alternative<monostate>(b)) [[likely]] { return monostate{}; }
            if (a.index() != b.index()) { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<T, float3>) {
                        auto b_v = get<T>(b);
                        return cross(t, b_v);
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::DOT: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            auto b = try_eval(expr->arguments()[1]);
            if (holds_alternative<monostate>(b)) [[likely]] { return monostate{}; }
            if (a.index() != b.index()) { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<T, float2> || std::is_same_v<T, float3> || std::is_same_v<T, float4>) {
                        auto b_v = get<T>(b);
                        return dot(t, b_v);
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::LENGTH: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<T, float2> || std::is_same_v<T, float3> || std::is_same_v<T, float4>) {
                        return length(t);
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::LENGTH_SQUARED: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<T, float2> || std::is_same_v<T, float3> || std::is_same_v<T, float4>) {
                        return dot(t, t);
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::NORMALIZE: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (std::is_same_v<T, float2> || std::is_same_v<T, float3> || std::is_same_v<T, float4>) {
                        return normalize(t);
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::DETERMINANT: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (ScalarType<T>::is_matrix) {
                        return determinant(t);
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::TRANSPOSE: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (ScalarType<T>::is_matrix) {
                        return transpose(t);
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
        case CallOp::INVERSE: {
            auto a = try_eval(expr->arguments()[0]);
            if (holds_alternative<monostate>(a)) [[likely]] { return monostate{}; }
            return visit(
                [&]<typename T>(T const &t) -> Result {
                    if constexpr (ScalarType<T>::is_matrix) {
                        return inverse(t);
                    } else {
                        return monostate{};
                    }
                },
                a);
        } break;
    }
    return monostate{};
}

ASTEvaluator::Result ASTEvaluator::try_eval(CastExpr const *expr) {
    auto result = try_eval(expr->expression());
    using namespace analyzer_detail;
    auto cast_scalar = [&]<typename T>(T const &t) -> Result {
        if (expr->type()->is_scalar()) {
            if (expr->op() == CastOp::STATIC) {
                switch (expr->type()->tag()) {
                    case Type::Tag::BOOL:
                        return static_cast<bool>(t);
                    case Type::Tag::FLOAT:
                        return static_cast<float>(t);
                    case Type::Tag::INT:
                        return static_cast<int>(t);
                    case Type::Tag::UINT:
                        return static_cast<uint>(t);
                    default: return monostate{};
                }
            } else {
                auto cast_ele = [&]<typename Dst>() -> Result {
                    if constexpr (sizeof(Dst) == sizeof(T)) {
                        return reinterpret_cast<Dst const &>(t);
                    } else {
                        return monostate{};
                    }
                };
                switch (expr->type()->tag()) {
                    case Type::Tag::BOOL:
                        return cast_ele.template operator()<bool>();
                    case Type::Tag::FLOAT:
                        return cast_ele.template operator()<float>();
                    case Type::Tag::INT:
                        return cast_ele.template operator()<int>();
                    case Type::Tag::UINT:
                        return cast_ele.template operator()<uint>();
                    default: return monostate{};
                }
            }
        } else {
            if (expr->op() == CastOp::STATIC) {
                auto temp_func = [&]<size_t dim, typename Dst>() -> Result {
                    Vector<Dst, dim> vec;
                    for (auto idx : range(dim)) {
                        vec[idx] = static_cast<Dst>(t);
                    }
                    return vec;
                };
                auto temp_func_idx = [&]<size_t dim>() -> Result {
                    switch (expr->type()->tag()) {
                        case Type::Tag::BOOL:
                            return temp_func.template operator()<dim, bool>();
                        case Type::Tag::FLOAT:
                            return temp_func.template operator()<dim, float>();
                        case Type::Tag::INT:
                            return temp_func.template operator()<dim, int>();
                        case Type::Tag::UINT:
                            return temp_func.template operator()<dim, uint>();
                        default: return monostate{};
                    }
                };

                switch (expr->type()->dimension()) {
                    case 2:
                        return temp_func_idx.template operator()<2>();
                    case 3:
                        return temp_func_idx.template operator()<3>();
                    case 4:
                        return temp_func_idx.template operator()<4>();
                }
            } else {
                return monostate{};
            }
        }
    };

    auto cast_vector = [&]<typename T>(T const &t) -> Result {
        if (!expr->type()->is_vector() || expr->type()->dimension() != ScalarType<T>::size) return monostate{};
        if (expr->op() == CastOp::STATIC) {
            auto cast_func = [&]<size_t dim, typename Dst>() -> Result {
                Vector<Dst, dim> vec;
                for (auto idx : range(dim)) {
                    vec[idx] = static_cast<Dst>(t[idx]);
                }
                return vec;
            };
            auto cast_func_idx = [&]<size_t dim>() -> Result {
                switch (expr->type()->element()->tag()) {
                    case Type::Tag::BOOL:
                        return cast_func.template operator()<dim, bool>();
                    case Type::Tag::FLOAT:
                        return cast_func.template operator()<dim, float>();
                    case Type::Tag::INT:
                        return cast_func.template operator()<dim, int>();
                    case Type::Tag::UINT:
                        return cast_func.template operator()<dim, uint>();
                    default: return monostate{};
                }
            };
            switch (expr->type()->dimension()) {
                case 2:
                    return cast_func_idx.template operator()<2>();
                case 3:
                    return cast_func_idx.template operator()<3>();
                case 4:
                    return cast_func_idx.template operator()<4>();
            }
        } else {
            auto cast_func = [&]<size_t dim, typename Dst>() -> Result {
                if constexpr (sizeof(Dst) == sizeof(T)) {
                    Vector<Dst, dim> vec;
                    for (auto idx : range(dim)) {
                        vec[idx] = reinterpret_cast<Dst const &>(t[idx]);
                    }
                    return vec;
                } else {
                    return monostate{};
                }
            };
            auto cast_func_idx = [&]<size_t dim>() -> Result {
                switch (expr->type()->element()->tag()) {
                    case Type::Tag::BOOL:
                        return cast_func.template operator()<dim, bool>();
                    case Type::Tag::FLOAT:
                        return cast_func.template operator()<dim, float>();
                    case Type::Tag::INT:
                        return cast_func.template operator()<dim, int>();
                    case Type::Tag::UINT:
                        return cast_func.template operator()<dim, uint>();
                    default: return monostate{};
                }
            };
            switch (expr->type()->dimension()) {
                case 2:
                    return cast_func_idx.template operator()<2>();
                case 3:
                    return cast_func_idx.template operator()<3>();
                case 4:
                    return cast_func_idx.template operator()<4>();
            }
        }
        return monostate{};
    };

    return visit(
        [&]<typename T>(T const &t) -> Result {
            if constexpr (std::is_same_v<T, monostate> || ScalarType<T>::is_matrix) {
                return monostate{};
            } else if constexpr (ScalarType<T>::is_vector) {
                return cast_vector(t);
            } else {
                return cast_scalar(t);
            }
        },
        result);
}

}// namespace luisa::compute
