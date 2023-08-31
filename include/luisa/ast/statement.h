#pragma once

#include <luisa/core/concepts.h>
#include <luisa/ast/variable.h>
#include <luisa/ast/expression.h>

namespace luisa::compute {
class CallableLibrary;
struct StmtVisitor;

/**
 * @brief Base statement class
 * 
 */
class LC_AST_API Statement : public concepts::Noncopyable {
    friend class CallableLibrary;

public:
    /// Statement types
    enum struct Tag : uint32_t {
        BREAK,
        CONTINUE,
        RETURN,
        SCOPE,
        IF,
        LOOP,
        EXPR,
        SWITCH,
        SWITCH_CASE,
        SWITCH_DEFAULT,
        ASSIGN,
        FOR,
        COMMENT,
        RAY_QUERY,
        AUTO_DIFF
    };

private:
    mutable uint64_t _hash{0u};
    mutable bool _hash_computed{false};
    Tag _tag;

protected:
    Statement() noexcept = default;

private:
    [[nodiscard]] virtual uint64_t _compute_hash() const noexcept = 0;

public:
    explicit Statement(Tag tag) noexcept : _tag{tag} {}
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    virtual void accept(StmtVisitor &) const = 0;
    virtual ~Statement() noexcept = default;
    [[nodiscard]] uint64_t hash() const noexcept;
};

class BreakStmt;
class ContinueStmt;

class ReturnStmt;

class ScopeStmt;
class IfStmt;
class LoopStmt;
class ExprStmt;
class SwitchStmt;
class SwitchCaseStmt;
class SwitchDefaultStmt;
class AssignStmt;
class ForStmt;
class CommentStmt;
class RayQueryStmt;
class AutoDiffStmt;

struct LC_AST_API StmtVisitor {
    virtual void visit(const BreakStmt *) = 0;
    virtual void visit(const ContinueStmt *) = 0;
    virtual void visit(const ReturnStmt *) = 0;
    virtual void visit(const ScopeStmt *) = 0;
    virtual void visit(const IfStmt *) = 0;
    virtual void visit(const LoopStmt *) = 0;
    virtual void visit(const ExprStmt *) = 0;
    virtual void visit(const SwitchStmt *) = 0;
    virtual void visit(const SwitchCaseStmt *) = 0;
    virtual void visit(const SwitchDefaultStmt *) = 0;
    virtual void visit(const AssignStmt *) = 0;
    virtual void visit(const ForStmt *) = 0;
    virtual void visit(const CommentStmt *) = 0;
    virtual void visit(const RayQueryStmt *) = 0;
    virtual void visit(const AutoDiffStmt *stmt);
    virtual ~StmtVisitor() noexcept = default;
};

#define LUISA_STATEMENT_COMMON() \
    void accept(StmtVisitor &visitor) const override { visitor.visit(this); }

/// Break statement
class BreakStmt final : public Statement {

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    BreakStmt() noexcept : Statement{Tag::BREAK} {}
    LUISA_STATEMENT_COMMON()
};

/// Continue statement
class ContinueStmt : public Statement {

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    ContinueStmt() noexcept : Statement{Tag::CONTINUE} {}
    LUISA_STATEMENT_COMMON()
};

/// Return statement
class ReturnStmt : public Statement {
    friend class CallableLibrary;

private:
    const Expression *_expr;
    ReturnStmt() noexcept = default;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    /**
     * @brief Construct a new ReturnStmt object
     * 
     * @param expr return expression, will be marked as Usage::READ
     */
    explicit ReturnStmt(const Expression *expr) noexcept
        : Statement{Tag::RETURN}, _expr{expr} {
        if (_expr != nullptr) { _expr->mark(Usage::READ); }
    }
    [[nodiscard]] auto expression() const noexcept { return _expr; }
    LUISA_STATEMENT_COMMON()
};

/// Scope statement
class ScopeStmt : public Statement {
    friend class CallableLibrary;

private:
    vector<const Statement *> _statements;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    ScopeStmt() noexcept : Statement{Tag::SCOPE} {}
    [[nodiscard]] auto statements() const noexcept { return luisa::span{_statements}; }
    void append(const Statement *stmt) noexcept { _statements.emplace_back(stmt); }
    const Statement *pop() noexcept;
    LUISA_STATEMENT_COMMON()
};

/// Assign statement
class AssignStmt : public Statement {
    friend class CallableLibrary;

private:
    const Expression *_lhs;
    const Expression *_rhs;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;
    AssignStmt() noexcept = default;

public:
    /**
     * @brief Construct a new AssignStmt object
     * 
     * @param lhs will be marked as Usage::WRITE
     * @param rhs will be marked as Usage::READ
     */
    AssignStmt(const Expression *lhs, const Expression *rhs) noexcept
        : Statement{Tag::ASSIGN}, _lhs{lhs}, _rhs{rhs} {
        _lhs->mark(Usage::WRITE);
        _rhs->mark(Usage::READ);
    }

    [[nodiscard]] auto lhs() const noexcept { return _lhs; }
    [[nodiscard]] auto rhs() const noexcept { return _rhs; }
    LUISA_STATEMENT_COMMON()
};

/// If statement
class IfStmt : public Statement {
    friend class CallableLibrary;

private:
    const Expression *_condition;
    ScopeStmt _true_branch;
    ScopeStmt _false_branch;
    IfStmt() noexcept = default;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    /**
     * @brief Construct a new IfStmt object
     * 
     * @param cond condition expression, will be marked as Usage::READ
     */
    explicit IfStmt(const Expression *cond) noexcept
        : Statement{Tag::IF},
          _condition{cond} {
        _condition->mark(Usage::READ);
    }
    [[nodiscard]] auto condition() const noexcept { return _condition; }
    [[nodiscard]] auto true_branch() noexcept { return &_true_branch; }
    [[nodiscard]] auto false_branch() noexcept { return &_false_branch; }
    [[nodiscard]] auto true_branch() const noexcept { return &_true_branch; }
    [[nodiscard]] auto false_branch() const noexcept { return &_false_branch; }
    LUISA_STATEMENT_COMMON()
};

/// Loop statement
class LoopStmt : public Statement {
    friend class CallableLibrary;

private:
    ScopeStmt _body;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    LoopStmt() noexcept : Statement{Tag::LOOP} {}
    [[nodiscard]] auto body() noexcept { return &_body; }
    [[nodiscard]] auto body() const noexcept { return &_body; }
    LUISA_STATEMENT_COMMON()
};

/// Expression statement
class ExprStmt : public Statement {
    friend class CallableLibrary;

private:
    const Expression *_expr;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;
    ExprStmt() noexcept = default;

public:
    /**
     * @brief Construct a new ExprStmt object
     * 
     * @param expr will be marked as Usage::READ
     */
    explicit ExprStmt(const Expression *expr) noexcept
        : Statement{Tag::EXPR}, _expr{expr} {
        _expr->mark(Usage::READ);
    }
    [[nodiscard]] auto expression() const noexcept { return _expr; }
    LUISA_STATEMENT_COMMON()
};

/// Switch statement
class SwitchStmt : public Statement {
    friend class CallableLibrary;

private:
    const Expression *_expr;
    ScopeStmt _body;
    SwitchStmt() noexcept = default;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    /**
     * @brief Construct a new SwitchStmt object
     * 
     * @param expr expression, will be marked as Usage::READ
     */
    explicit SwitchStmt(const Expression *expr) noexcept
        : Statement{Tag::SWITCH}, _expr{expr} {
        _expr->mark(Usage::READ);
    }
    [[nodiscard]] auto expression() const noexcept { return _expr; }
    [[nodiscard]] auto body() noexcept { return &_body; }
    [[nodiscard]] auto body() const noexcept { return &_body; }
    LUISA_STATEMENT_COMMON()
};

/// Case statement of switch
class SwitchCaseStmt : public Statement {
    friend class CallableLibrary;

private:
    const Expression *_expr;
    ScopeStmt _body;
    SwitchCaseStmt() noexcept = default;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    /**
     * @brief Construct a new SwitchCaseStmt object
     * 
     * @param expr expression, will be marked as Usage::READ
     */
    explicit SwitchCaseStmt(const Expression *expr) noexcept
        : Statement{Tag::SWITCH_CASE}, _expr{expr} {
        _expr->mark(Usage::READ);
    }
    [[nodiscard]] auto expression() const noexcept { return _expr; }
    [[nodiscard]] auto body() noexcept { return &_body; }
    [[nodiscard]] auto body() const noexcept { return &_body; }
    LUISA_STATEMENT_COMMON()
};

/// Default statement of switch
class SwitchDefaultStmt : public Statement {
    friend class CallableLibrary;

private:
    ScopeStmt _body;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    SwitchDefaultStmt() noexcept : Statement{Tag::SWITCH_DEFAULT} {}
    [[nodiscard]] auto body() noexcept { return &_body; }
    [[nodiscard]] auto body() const noexcept { return &_body; }
    LUISA_STATEMENT_COMMON()
};

/// For statement
class ForStmt : public Statement {
    friend class CallableLibrary;

private:
    const Expression *_var;
    const Expression *_cond;
    const Expression *_step;
    ScopeStmt _body;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;
    ForStmt() noexcept = default;

public:
    /**
     * @brief Construct a new ForStmt object
     * 
     * @param var variable expression, will be marked as Usage::READ_WRITE
     * @param cond condition expression, will be marked as Usage::READ
     * @param step step expression, will be marked as Usage::READ
     */
    ForStmt(const Expression *var,
            const Expression *cond,
            const Expression *step) noexcept
        : Statement{Tag::FOR},
          _var{var}, _cond{cond}, _step{step} {
        _var->mark(Usage::READ_WRITE);
        _cond->mark(Usage::READ);
        _step->mark(Usage::READ);
    }
    [[nodiscard]] auto variable() const noexcept { return _var; }
    [[nodiscard]] auto condition() const noexcept { return _cond; }
    [[nodiscard]] auto step() const noexcept { return _step; }
    [[nodiscard]] auto body() noexcept { return &_body; }
    [[nodiscard]] auto body() const noexcept { return &_body; }
    LUISA_STATEMENT_COMMON()
};

/// Comment statement
class CommentStmt : public Statement {
    friend class CallableLibrary;

private:
    luisa::string _comment;
    CommentStmt() noexcept = default;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    /**
     * @brief Construct a new CommentStmt object
     * 
     * @param comment comment content
     */
    explicit CommentStmt(luisa::string comment) noexcept
        : Statement{Tag::COMMENT},
          _comment{std::move(comment)} {}
    [[nodiscard]] auto comment() const noexcept { return std::string_view{_comment}; }
    LUISA_STATEMENT_COMMON()
};

// Example:
// auto q = accel->trace_all(ray, mask)
//   .on_triangle_candidate([](auto &candidate) {
//     << triangle candidate handling >>
//   })
//   .on_procedural_candidate([](auto &candidate) {
//     << procedural candidate handling >>
//   })
//   .query();
// auto hit = q.committed_hit();
//
// On inline RT, translates to
// auto q = lc_accel_trace_all(accel, ray, mask);
// while (lc_ray_query_next(q)) {
//   if (lc_ray_query_is_triangle(q)) {
//     auto candidate = lc_ray_query_triangle_candidate(q);
//     << triangle candidate handling >>
//   } else {
//     auto candidate = lc_ray_query_procedural_candidate(q);
//     << procedural candidate handling >>
//   }
// }
// auto hit = lc_ray_query_committed_hit(q);
//
// On RT-pipelines, translates to
// in caller:
// auto q = lc_accel_trace_all(accel, ray, mask);
// auto hit = lc_ray_query_committed_hit(q);
//
// anyhit shader for triangle:
// __global__ void__ __anyhit__ray_query() {
//   auto committed = false;
//   auto candidate = lc_ray_query_triangle_candidate();
//   << triangle candidate handling >>
//   if (!committed) { lc_ignore_intersection(); }
// }
//
// intersection shader for procedural geometry:
// __global__ void __intersection__ray_query() {
//   auto candidate = lc_ray_query_procedural_candidate();
//   << procedural candidate handling >>
// }
//
// closest hit shader
// __global__ void __closesthit__ray_query() {
//   ...
// }

class RayQueryStmt : public Statement {
    friend class CallableLibrary;

private:
    const RefExpr *_query;
    ScopeStmt _on_triangle_candidate;
    ScopeStmt _on_procedural_candidate;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;
    RayQueryStmt() noexcept = default;

public:
    explicit RayQueryStmt(const RefExpr *query) noexcept
        : Statement{Tag::RAY_QUERY}, _query{query} {
        _query->mark(Usage::READ_WRITE);
    }
    [[nodiscard]] auto query() const noexcept { return _query; }
    [[nodiscard]] auto on_triangle_candidate() noexcept { return &_on_triangle_candidate; }
    [[nodiscard]] auto on_triangle_candidate() const noexcept { return &_on_triangle_candidate; }
    [[nodiscard]] auto on_procedural_candidate() noexcept { return &_on_procedural_candidate; }
    [[nodiscard]] auto on_procedural_candidate() const noexcept { return &_on_procedural_candidate; }
    LUISA_STATEMENT_COMMON()
};

class AutoDiffStmt : public Statement {
    friend class CallableLibrary;

private:
    ScopeStmt _body;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    explicit AutoDiffStmt() noexcept : Statement{Tag::AUTO_DIFF} {}
    [[nodiscard]] auto body() noexcept { return &_body; }
    [[nodiscard]] auto body() const noexcept { return &_body; }
    LUISA_STATEMENT_COMMON()
};

#undef LUISA_STATEMENT_COMMON

// helper function for easy traversal over the ASTs
template<bool recurse_subexpr,
         typename F,
         typename EnterStmt,
         typename ExitStmt>
void traverse_expressions(
    const Statement *stmt, const F &visit,
    const EnterStmt &enter_stmt,
    const ExitStmt &exit_stmt) noexcept {

    auto do_visit = [&visit](auto expr) noexcept {
        if constexpr (recurse_subexpr) {
            traverse_subexpressions(expr, visit, [](auto) noexcept {});
        } else {
            visit(expr);
        }
    };

    enter_stmt(stmt);
    switch (stmt->tag()) {
        case Statement::Tag::BREAK:
        case Statement::Tag::CONTINUE: break;
        case Statement::Tag::RETURN: {
            auto return_stmt = static_cast<const ReturnStmt *>(stmt);
            if (auto value = return_stmt->expression()) { do_visit(value); }
            break;
        }
        case Statement::Tag::SCOPE: {
            auto scope_stmt = static_cast<const ScopeStmt *>(stmt);
            for (auto s : scope_stmt->statements()) {
                traverse_expressions<recurse_subexpr>(
                    s, visit, enter_stmt, exit_stmt);
            }
            break;
        }
        case Statement::Tag::IF: {
            auto if_stmt = static_cast<const IfStmt *>(stmt);
            do_visit(if_stmt->condition());
            traverse_expressions<recurse_subexpr>(
                if_stmt->true_branch(), visit, enter_stmt, exit_stmt);
            traverse_expressions<recurse_subexpr>(
                if_stmt->false_branch(), visit, enter_stmt, exit_stmt);
            break;
        }
        case Statement::Tag::LOOP: {
            auto loop_stmt = static_cast<const LoopStmt *>(stmt);
            traverse_expressions<recurse_subexpr>(
                loop_stmt->body(), visit, enter_stmt, exit_stmt);
            break;
        }
        case Statement::Tag::EXPR: {
            auto expr_stmt = static_cast<const ExprStmt *>(stmt);
            do_visit(expr_stmt->expression());
            break;
        }
        case Statement::Tag::SWITCH: {
            auto switch_stmt = static_cast<const SwitchStmt *>(stmt);
            do_visit(switch_stmt->expression());
            traverse_expressions<recurse_subexpr>(
                switch_stmt->body(), visit, enter_stmt, exit_stmt);
            break;
        }
        case Statement::Tag::SWITCH_CASE: {
            auto case_stmt = static_cast<const SwitchCaseStmt *>(stmt);
            traverse_expressions<recurse_subexpr>(
                case_stmt->body(), visit, enter_stmt, exit_stmt);
            break;
        }
        case Statement::Tag::SWITCH_DEFAULT: {
            auto default_stmt = static_cast<const SwitchDefaultStmt *>(stmt);
            traverse_expressions<recurse_subexpr>(
                default_stmt->body(), visit, enter_stmt, exit_stmt);
            break;
        }
        case Statement::Tag::ASSIGN: {
            auto assign_stmt = static_cast<const AssignStmt *>(stmt);
            do_visit(assign_stmt->lhs());
            do_visit(assign_stmt->rhs());
            break;
        }
        case Statement::Tag::FOR: {
            auto for_stmt = static_cast<const ForStmt *>(stmt);
            do_visit(for_stmt->variable());
            do_visit(for_stmt->condition());
            do_visit(for_stmt->step());
            traverse_expressions<recurse_subexpr>(
                for_stmt->body(), visit, enter_stmt, exit_stmt);
            break;
        }
        case Statement::Tag::COMMENT: break;
        case Statement::Tag::RAY_QUERY: {
            auto rq_stmt = static_cast<const RayQueryStmt *>(stmt);
            do_visit(rq_stmt->query());
            traverse_expressions<recurse_subexpr>(
                rq_stmt->on_triangle_candidate(), visit, enter_stmt, exit_stmt);
            traverse_expressions<recurse_subexpr>(
                rq_stmt->on_procedural_candidate(), visit, enter_stmt, exit_stmt);
            break;
        }
        case Statement::Tag::AUTO_DIFF: {
            auto ad_stmt = static_cast<const AutoDiffStmt *>(stmt);
            traverse_expressions<recurse_subexpr>(
                ad_stmt->body(), visit, enter_stmt, exit_stmt);
            break;
        }
    }
    exit_stmt(stmt);
}

}// namespace luisa::compute
