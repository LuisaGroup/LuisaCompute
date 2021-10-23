//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <core/concepts.h>
#include <core/arena.h>
#include <ast/variable.h>
#include <ast/expression.h>

namespace luisa::compute {

struct StmtVisitor;

class Statement : public concepts::Noncopyable {

protected:
    ~Statement() noexcept = default;

public:
    virtual void accept(StmtVisitor &) const = 0;
};

struct BreakStmt;
struct ContinueStmt;

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

#define LUISA_MAKE_STATEMENT_ACCEPT_VISITOR() \
    void accept(StmtVisitor &visitor) const override { visitor.visit(this); }

struct BreakStmt : public Statement {
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

struct ContinueStmt : public Statement {
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class ReturnStmt : public Statement {

private:
    const Expression *_expr;

public:
    explicit ReturnStmt(const Expression *expr) noexcept : _expr{expr} {
        if (_expr != nullptr) { _expr->mark(Usage::READ); }
    }
    [[nodiscard]] auto expression() const noexcept { return _expr; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class ScopeStmt : public Statement {

private:
    ArenaVector<const Statement *> _statements;

public:
    explicit ScopeStmt(ArenaVector<const Statement *> stmts) noexcept : _statements{std::move(stmts)} {}
    [[nodiscard]] auto statements() const noexcept { return std::span{_statements.data(), _statements.size()}; }
    void append(const Statement *stmt) noexcept { _statements.emplace_back(stmt); }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

enum struct AssignOp {
    ASSIGN,
    ADD_ASSIGN,
    SUB_ASSIGN,
    MUL_ASSIGN,
    DIV_ASSIGN,
    MOD_ASSIGN,
    BIT_AND_ASSIGN,
    BIT_OR_ASSIGN,
    BIT_XOR_ASSIGN,
    SHL_ASSIGN,
    SHR_ASSIGN
};

class AssignStmt : public Statement {

private:
    const Expression *_lhs;
    const Expression *_rhs;
    AssignOp _op;

public:
    AssignStmt(AssignOp op, const Expression *lhs, const Expression *rhs) noexcept
        : _lhs{lhs}, _rhs{rhs}, _op{op} {
        _lhs->mark(Usage::WRITE);
        _rhs->mark(Usage::READ);
    }

    [[nodiscard]] auto lhs() const noexcept { return _lhs; }
    [[nodiscard]] auto rhs() const noexcept { return _rhs; }
    [[nodiscard]] auto op() const noexcept { return _op; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class IfStmt : public Statement {

private:
    const Expression *_condition;
    const ScopeStmt *_true_branch;
    const ScopeStmt *_false_branch;

public:
    IfStmt(const Expression *cond, const ScopeStmt *true_branch, const ScopeStmt *false_branch) noexcept
        : _condition{cond}, _true_branch{true_branch}, _false_branch{false_branch} {
        _condition->mark(Usage::READ);
    }
    [[nodiscard]] auto condition() const noexcept { return _condition; }
    [[nodiscard]] auto true_branch() const noexcept { return _true_branch; }
    [[nodiscard]] auto false_branch() const noexcept { return _false_branch; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class LoopStmt : public Statement {

private:
    const ScopeStmt *_body;

public:
    LoopStmt(const ScopeStmt *body) : _body{body} {}
    [[nodiscard]] auto body() const noexcept { return _body; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class ExprStmt : public Statement {

private:
    const Expression *_expr;

public:
    explicit ExprStmt(const Expression *expr) noexcept : _expr{expr} {
        _expr->mark(Usage::READ);
    }
    [[nodiscard]] auto expression() const noexcept { return _expr; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class SwitchStmt : public Statement {

private:
    const Expression *_expr;
    const ScopeStmt *_body;

public:
    SwitchStmt(const Expression *expr, const ScopeStmt *body) noexcept
        : _expr{expr}, _body{body} { _expr->mark(Usage::READ); }
    [[nodiscard]] auto expression() const noexcept { return _expr; }
    [[nodiscard]] auto body() const noexcept { return _body; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class SwitchCaseStmt : public Statement {

private:
    const Expression *_expr;
    const ScopeStmt *_body;

public:
    SwitchCaseStmt(const Expression *expr, const ScopeStmt *body) noexcept
        : _expr{expr}, _body{body} { _expr->mark(Usage::READ); }
    [[nodiscard]] auto expression() const noexcept { return _expr; }
    [[nodiscard]] auto body() const noexcept { return _body; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class SwitchDefaultStmt : public Statement {

private:
    const ScopeStmt *_body;

public:
    explicit SwitchDefaultStmt(const ScopeStmt *body) noexcept : _body{body} {}
    [[nodiscard]] auto body() const noexcept { return _body; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class ForStmt : public Statement {

private:
    const Expression *_var;
    const Expression *_cond;
    const Expression *_step;
    const ScopeStmt *_body;

public:
    ForStmt(const Expression *var,
            const Expression *cond,
            const Expression *step,
            const ScopeStmt *body) noexcept
        : _var{var}, _cond{cond}, _step{step}, _body{body} {
        _var->mark(Usage::READ_WRITE);
        _cond->mark(Usage::READ);
        _step->mark(Usage::READ);
    }
    [[nodiscard]] auto variable() const noexcept { return _var; }
    [[nodiscard]] auto condition() const noexcept { return _cond; }
    [[nodiscard]] auto step() const noexcept { return _step; }
    [[nodiscard]] auto body() const noexcept { return _body; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class CommentStmt : public Statement {

private:
    std::string_view _comment;

public:
    explicit CommentStmt(std::string_view comment) noexcept
        : _comment{comment} {}
    [[nodiscard]] auto comment() const noexcept { return _comment; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

#undef LUISA_MAKE_STATEMENT_ACCEPT_VISITOR

// checks for working with Arena
static_assert(std::is_trivially_destructible_v<BreakStmt>);
static_assert(std::is_trivially_destructible_v<ContinueStmt>);
static_assert(std::is_trivially_destructible_v<ReturnStmt>);
static_assert(std::is_trivially_destructible_v<ScopeStmt>);
static_assert(std::is_trivially_destructible_v<IfStmt>);
static_assert(std::is_trivially_destructible_v<LoopStmt>);
static_assert(std::is_trivially_destructible_v<ExprStmt>);
static_assert(std::is_trivially_destructible_v<SwitchStmt>);
static_assert(std::is_trivially_destructible_v<SwitchCaseStmt>);
static_assert(std::is_trivially_destructible_v<SwitchDefaultStmt>);
static_assert(std::is_trivially_destructible_v<AssignStmt>);
static_assert(std::is_trivially_destructible_v<ForStmt>);
static_assert(std::is_trivially_destructible_v<CommentStmt>);

}// namespace luisa::compute
