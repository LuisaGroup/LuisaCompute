//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <core/concepts.h>
#include <core/stl.h>
#include <ast/variable.h>
#include <ast/expression.h>

namespace luisa::compute {

class AstSerializer;
struct StmtVisitor;

/**
 * @brief Base statement class
 * 
 */
class LC_AST_API Statement : public concepts::Noncopyable {

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
        COMMENT
    };

private:
    mutable uint64_t _hash{0u};
    mutable bool _hash_computed{false};
    Tag _tag;

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

struct StmtVisitor {
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
};

#define LUISA_STATEMENT_COMMON() \
    friend class AstSerializer;  \
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

private:
    const Expression *_expr;

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

private:
    vector<const Statement *> _statements;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    ScopeStmt() noexcept : Statement{Tag::SCOPE} {}
    [[nodiscard]] auto statements() const noexcept { return luisa::span{_statements}; }
    void append(const Statement *stmt) noexcept { _statements.emplace_back(stmt); }
    LUISA_STATEMENT_COMMON()
};

/// Assign statement
class AssignStmt : public Statement {

private:
    const Expression *_lhs;
    const Expression *_rhs;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

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

private:
    const Expression *_condition;
    ScopeStmt _true_branch;
    ScopeStmt _false_branch;

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

private:
    const Expression *_expr;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

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

private:
    const Expression *_expr;
    ScopeStmt _body;

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

private:
    const Expression *_expr;
    ScopeStmt _body;

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

private:
    const Expression *_var;
    const Expression *_cond;
    const Expression *_step;
    ScopeStmt _body;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

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

private:
    luisa::string _comment;

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

#undef LUISA_STATEMENT_COMMON

}// namespace luisa::compute
