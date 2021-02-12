//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <core/arena.h>

namespace luisa {

class Variable;
class Expression;

struct StmtVisitor;

class Statement {

protected:
    ~Statement() noexcept = default;

public:
    virtual void accept(StmtVisitor &visitor) const = 0;
};

struct EmptyStmt;

struct BreakStmt;
struct ContinueStmt;
struct ReturnStmt;

class ScopeStmt;
class DeclareStmt;
class IfStmt;
class WhileStmt;
class ExprStmt;
class SwitchStmt;
class SwitchCaseStmt;
class SwitchDefaultStmt;
class AssignStmt;

struct StmtVisitor {
    virtual void visit(const EmptyStmt *) = 0;
    virtual void visit(const BreakStmt *) = 0;
    virtual void visit(const ContinueStmt *) = 0;
    virtual void visit(const ReturnStmt *) = 0;
    virtual void visit(const ScopeStmt *scope_stmt) = 0;
    virtual void visit(const DeclareStmt *declare_stmt) = 0;
    virtual void visit(const IfStmt *if_stmt) = 0;
    virtual void visit(const WhileStmt *while_stmt) = 0;
    virtual void visit(const ExprStmt *expr_stmt) = 0;
    virtual void visit(const SwitchStmt *switch_stmt) = 0;
    virtual void visit(const SwitchCaseStmt *case_stmt) = 0;
    virtual void visit(const SwitchDefaultStmt *default_stmt) = 0;
    virtual void visit(const AssignStmt *assign_stmt) = 0;
};

#define LUISA_MAKE_STATEMENT_ACCEPT_VISITOR() \
    void accept(StmtVisitor &visitor) const override { visitor.visit(this); }

struct EmptyStmt : public Statement {
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

struct BreakStmt : public Statement {
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

struct ContinueStmt : public Statement {
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

struct ReturnStmt : public Statement {
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class ScopeStmt : public Statement {

public:
    using StmtList = ArenaVector<const Statement *>;

private:
    StmtList _statements;

public:
    explicit ScopeStmt(StmtList stmts) noexcept : _statements{std::move(stmts)} {}
    [[nodiscard]] const auto &statements() const noexcept { return _statements; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class DeclareStmt : public Statement {

public:
    using InitializerList = ArenaVector<const Expression *>;

private:
    const Variable *_var;
    InitializerList _initializer_list;

public:
    DeclareStmt(const Variable *var, InitializerList init) noexcept
        : _var{var}, _initializer_list{std::move(init)} {}
    [[nodiscard]] auto variable() const noexcept { return _var; }
    [[nodiscard]] const auto &initializer() const noexcept { return _initializer_list; }
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
        : _lhs{lhs}, _rhs{rhs}, _op{op} {}

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
        : _condition{cond}, _true_branch{true_branch}, _false_branch{false_branch} {}

    [[nodiscard]] auto condition() const noexcept { return _condition; }
    [[nodiscard]] auto true_branch() const noexcept { return _true_branch; }
    [[nodiscard]] auto false_branch() const noexcept { return _false_branch; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class WhileStmt : public Statement {

private:
    const Expression *_condition;
    const ScopeStmt *_body;

public:
    WhileStmt(const Expression *cond, const ScopeStmt *body) : _condition{cond}, _body{body} {}
    [[nodiscard]] auto condition() const noexcept { return _condition; }
    [[nodiscard]] auto body() const noexcept { return _body; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class ExprStmt : public Statement {

private:
    const Expression *_expr;

public:
    explicit ExprStmt(const Expression *expr) noexcept : _expr{expr} {}
    [[nodiscard]] auto expression() const noexcept { return _expr; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class SwitchStmt : public Statement {

private:
    const Expression *_expr;
    const ScopeStmt *_body;

public:
    SwitchStmt(const Expression *expr, const ScopeStmt *body) noexcept : _expr{expr}, _body{body} {}
    [[nodiscard]] auto expression() const noexcept { return _expr; }
    [[nodiscard]] auto body() const noexcept { return _body; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class SwitchCaseStmt : public Statement {

private:
    const Expression *_expr;
    const ScopeStmt *_body;

public:
    SwitchCaseStmt(const Expression *expr, const ScopeStmt *body) noexcept : _expr{expr}, _body{body} {}
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

#undef LUISA_MAKE_STATEMENT_ACCEPT_VISITOR

// checks for working with Arena
static_assert(std::is_trivially_destructible_v<EmptyStmt>);
static_assert(std::is_trivially_destructible_v<BreakStmt>);
static_assert(std::is_trivially_destructible_v<ContinueStmt>);
static_assert(std::is_trivially_destructible_v<ReturnStmt>);
static_assert(std::is_trivially_destructible_v<ScopeStmt>);
static_assert(std::is_trivially_destructible_v<DeclareStmt>);
static_assert(std::is_trivially_destructible_v<IfStmt>);
static_assert(std::is_trivially_destructible_v<WhileStmt>);
static_assert(std::is_trivially_destructible_v<ExprStmt>);
static_assert(std::is_trivially_destructible_v<SwitchStmt>);
static_assert(std::is_trivially_destructible_v<SwitchCaseStmt>);
static_assert(std::is_trivially_destructible_v<SwitchDefaultStmt>);
static_assert(std::is_trivially_destructible_v<AssignStmt>);

}// namespace luisa
