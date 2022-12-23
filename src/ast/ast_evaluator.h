#pragma once

#include <core/stl/variant.h>
#include <core/stl/unordered_map.h>
#include <ast/expression.h>
#include <ast/statement.h>

namespace luisa::compute {

namespace detail {

template<typename T>
struct make_optional_literal_value {
    static_assert(always_false_v<T>, "Unsupported literal type.");
};

template<typename... T>
struct make_optional_literal_value<std::tuple<T...>> {
    using type = luisa::variant<luisa::monostate, T...>;
};

template<typename T>
using make_optional_literal_value_t = typename make_optional_literal_value<T>::type;

}// namespace detail

class LC_AST_API ASTEvaluator {

public:
    using Result = detail::make_optional_literal_value_t<basic_types>;

    struct Branch {
        luisa::unordered_map<uint, Result> variables;
        bool is_loop;
        explicit Branch(bool is_loop) noexcept : is_loop{is_loop} {}
    };

private:
    luisa::vector<Branch> var_values;
    luisa::vector<Result> switch_scopes;
    size_t branch_scope;

public:
    ASTEvaluator() noexcept;
    ASTEvaluator(ASTEvaluator const &) = delete;
    ASTEvaluator(ASTEvaluator &&) = default;
    ~ASTEvaluator() = default;
    [[nodiscard]] Result try_eval(UnaryExpr const *expr);
    [[nodiscard]] Result try_eval(BinaryExpr const *expr);
    [[nodiscard]] Result try_eval(MemberExpr const *expr);
    [[nodiscard]] Result try_eval(AccessExpr const *expr);
    [[nodiscard]] Result try_eval(LiteralExpr const *expr);
    [[nodiscard]] Result try_eval(RefExpr const *expr);
    [[nodiscard]] Result try_eval(CallExpr const *expr);
    [[nodiscard]] Result try_eval(CastExpr const *expr);
    [[nodiscard]] Result try_eval(Expression const *expr);
    void check_call_ref(Function func, luisa::span<Expression const *const> args_var);
    void assign(AssignStmt const *stmt);
    ASTEvaluator::Result assign(Expression const *lhs, Expression const *rhs);
    void ref_var(Variable var);
    Statement const *map_if(IfStmt const *stmt);
    void execute_for(ForStmt const *stmt);
    void begin_switch(SwitchStmt const *stmt);
    bool check_switch_case(SwitchCaseStmt const *stmt);
    void end_switch();
    void begin_branch_scope(bool is_loop);
    void end_branch_scope();
};

}// namespace luisa::compute
