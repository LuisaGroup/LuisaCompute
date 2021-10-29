//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <core/concepts.h>
#include <core/allocator.h>
#include <ast/variable.h>
#include <ast/expression.h>

namespace luisa::compute {

struct StmtVisitor;

class Statement : public concepts::Noncopyable {

public:
    virtual void accept(StmtVisitor &) const = 0;
    virtual ~Statement() noexcept = default;
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
class MetaStmt;

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
    virtual void visit(const MetaStmt *) = 0;
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
    vector<const Statement *> _statements;

public:
    [[nodiscard]] auto statements() const noexcept { return std::span{_statements}; }
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
    ScopeStmt _true_branch;
    ScopeStmt _false_branch;

public:
    IfStmt(const Expression *cond) noexcept
        : _condition{cond} { _condition->mark(Usage::READ); }
    [[nodiscard]] auto condition() const noexcept { return _condition; }
    [[nodiscard]] auto true_branch() noexcept { return &_true_branch; }
    [[nodiscard]] auto false_branch() noexcept { return &_false_branch; }
    [[nodiscard]] auto true_branch() const noexcept { return &_true_branch; }
    [[nodiscard]] auto false_branch() const noexcept { return &_false_branch; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class LoopStmt : public Statement {

private:
    ScopeStmt _body;

public:
    [[nodiscard]] auto body() noexcept { return &_body; }
    [[nodiscard]] auto body() const noexcept { return &_body; }
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
    ScopeStmt _body;

public:
    explicit SwitchStmt(const Expression *expr) noexcept
        : _expr{expr} { _expr->mark(Usage::READ); }
    [[nodiscard]] auto expression() const noexcept { return _expr; }
    [[nodiscard]] auto body() noexcept { return &_body; }
    [[nodiscard]] auto body() const noexcept { return &_body; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class SwitchCaseStmt : public Statement {

private:
    const Expression *_expr;
    ScopeStmt _body;

public:
    explicit SwitchCaseStmt(const Expression *expr) noexcept
        : _expr{expr} { _expr->mark(Usage::READ); }
    [[nodiscard]] auto expression() const noexcept { return _expr; }
    [[nodiscard]] auto body() noexcept { return &_body; }
    [[nodiscard]] auto body() const noexcept { return &_body; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class SwitchDefaultStmt : public Statement {

private:
    ScopeStmt _body;

public:
    [[nodiscard]] auto body() noexcept { return &_body; }
    [[nodiscard]] auto body() const noexcept { return &_body; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class ForStmt : public Statement {

private:
    const Expression *_var;
    const Expression *_cond;
    const Expression *_step;
    ScopeStmt _body;

public:
    ForStmt(const Expression *var,
            const Expression *cond,
            const Expression *step) noexcept
        : _var{var}, _cond{cond}, _step{step} {
        _var->mark(Usage::READ_WRITE);
        _cond->mark(Usage::READ);
        _step->mark(Usage::READ);
    }
    [[nodiscard]] auto variable() const noexcept { return _var; }
    [[nodiscard]] auto condition() const noexcept { return _cond; }
    [[nodiscard]] auto step() const noexcept { return _step; }
    [[nodiscard]] auto body() noexcept { return &_body; }
    [[nodiscard]] auto body() const noexcept { return &_body; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class CommentStmt : public Statement {

private:
    luisa::string _comment;

public:
    explicit CommentStmt(luisa::string comment) noexcept
        : _comment{std::move(comment)} {}
    [[nodiscard]] auto comment() const noexcept { return std::string_view{_comment}; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

class MetaStmt : public Statement {

private:
    luisa::string _info;
    ScopeStmt _scope;
    vector<const MetaStmt *> _children;
    vector<Variable> _variables;

public:
    MetaStmt(luisa::string info) noexcept
        : _info{std::move(info)} {}
    [[nodiscard]] auto info() const noexcept { return std::string_view{_info}; }
    [[nodiscard]] auto scope() noexcept { return &_scope; }
    [[nodiscard]] auto scope() const noexcept { return &_scope; }
    [[nodiscard]] auto add(const MetaStmt *child) noexcept { _children.emplace_back(child); }
    [[nodiscard]] auto add(Variable v) noexcept { _variables.emplace_back(v); }
    [[nodiscard]] auto children() const noexcept { return std::span{_children}; }
    [[nodiscard]] auto variables() const noexcept { return std::span{_variables}; }
    LUISA_MAKE_STATEMENT_ACCEPT_VISITOR()
};

#undef LUISA_MAKE_STATEMENT_ACCEPT_VISITOR

}// namespace luisa::compute
