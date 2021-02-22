//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <core/concepts.h>
#include <ast/variable.h>

namespace luisa::compute {

class Expression;

struct StmtVisitor;

class Statement : public Noncopyable {

protected:
    ~Statement() noexcept = default;

public:
    virtual void accept(StmtVisitor &) const = 0;
};

struct BreakStmt;
struct ContinueStmt;

class ReturnStmt;

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
    virtual void visit(const BreakStmt *) = 0;
    virtual void visit(const ContinueStmt *) = 0;
    virtual void visit(const ReturnStmt *) = 0;
    virtual void visit(const ScopeStmt *) = 0;
    virtual void visit(const DeclareStmt *) = 0;
    virtual void visit(const IfStmt *) = 0;
    virtual void visit(const WhileStmt *) = 0;
    virtual void visit(const ExprStmt *) = 0;
    virtual void visit(const SwitchStmt *) = 0;
    virtual void visit(const SwitchCaseStmt *) = 0;
    virtual void visit(const SwitchDefaultStmt *) = 0;
    virtual void visit(const AssignStmt *) = 0;
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
    explicit ReturnStmt(const Expression *expr) noexcept : _expr{expr} {}
    [[nodiscard]] auto expression() const noexcept { return _expr; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class ScopeStmt : public Statement {

private:
    std::span<const Statement *> _statements;

public:
    explicit ScopeStmt(std::span<const Statement *> stmts) noexcept : _statements{stmts} {}
    [[nodiscard]] auto statements() noexcept { return _statements; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class DeclareStmt : public Statement {

private:
    Variable _var;
    std::span<const Expression *> _initializer;

public:
    DeclareStmt(Variable var, std::span<const Expression *> init) noexcept
        : _var{var}, _initializer{init} {}
    [[nodiscard]] auto variable() const noexcept { return _var; }
    [[nodiscard]] const auto &initializer() const noexcept { return _initializer; }
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
    const Statement *_true_branch;
    const Statement *_false_branch;

public:
    IfStmt(const Expression *cond, const Statement *true_branch, const Statement *false_branch = nullptr) noexcept
        : _condition{cond}, _true_branch{true_branch}, _false_branch{false_branch} {}

    [[nodiscard]] auto condition() const noexcept { return _condition; }
    [[nodiscard]] auto true_branch() const noexcept { return _true_branch; }
    [[nodiscard]] auto false_branch() const noexcept { return _false_branch; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class WhileStmt : public Statement {

private:
    const Expression *_condition;
    const Statement *_body;

public:
    WhileStmt(const Expression *cond, const Statement *body) : _condition{cond}, _body{body} {}
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
    const Statement *_body;

public:
    SwitchStmt(const Expression *expr, const Statement *body) noexcept : _expr{expr}, _body{body} {}
    [[nodiscard]] auto expression() const noexcept { return _expr; }
    [[nodiscard]] auto body() const noexcept { return _body; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class SwitchCaseStmt : public Statement {

private:
    const Expression *_expr;
    const Statement *_body;

public:
    SwitchCaseStmt(const Expression *expr, const Statement *body) noexcept : _expr{expr}, _body{body} {}
    [[nodiscard]] auto expression() const noexcept { return _expr; }
    [[nodiscard]] auto body() const noexcept { return _body; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class SwitchDefaultStmt : public Statement {

private:
    const Statement *_body;

public:
    explicit SwitchDefaultStmt(const Statement *body) noexcept : _body{body} {}
    [[nodiscard]] auto body() const noexcept { return _body; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

#undef LUISA_MAKE_STATEMENT_ACCEPT_VISITOR

// checks for working with Arena
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
