//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <string>
#include <vector>

#include <core/concepts.h>
#include <core/data_types.h>
#include <core/arena.h>

namespace luisa::compute {

struct ExprVisitor;

class Expression : public Noncopyable {

protected:
    ~Expression() noexcept = default;

public:
    virtual void accept(ExprVisitor &) const = 0;
};

class UnaryExpr;
class BinaryExpr;
class MemberExpr;
class AccessExpr;
class LiteralExpr;
class CallExpr;
class CastExpr;

struct ExprVisitor {
    virtual void visit(const UnaryExpr *unary_expr) = 0;
    virtual void visit(const BinaryExpr *binary_expr) = 0;
    virtual void visit(const MemberExpr *member_expr) = 0;
    virtual void visit(const AccessExpr *access_expr) = 0;
    virtual void visit(const LiteralExpr *literal_expr) = 0;
    virtual void visit(const CallExpr *func_expr) = 0;
    virtual void visit(const CastExpr *cast_expr) = 0;
};

#define LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR() \
    void accept(ExprVisitor &visitor) const override { visitor.visit(this); }

class Variable;

enum struct UnaryOp {
    PLUS,
    MINUS,  // +x, -x
    NOT,    // !x
    BIT_NOT,// ~x
    // Note: We deliberately support *NO* pre and postfix inc/dec operators to avoid possible abuse
};

class UnaryExpr : public Expression {
private:
    const Variable *_operand;
    UnaryOp _op;

public:
    UnaryExpr(UnaryOp op, const Variable *operand) noexcept : _operand{operand}, _op{op} {}
    [[nodiscard]] const Variable *operand() const noexcept { return _operand; }
    [[nodiscard]] UnaryOp op() const noexcept { return _op; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

enum struct BinaryOp {

    // arithmetic
    ADD,
    SUB,
    MUL,
    DIV,
    MOD,
    BIT_AND,
    BIT_OR,
    BIT_XOR,
    SHL,
    SHR,
    AND,
    OR,

    // relational
    LESS,
    GREATER,
    LESS_EQUAL,
    GREATER_EQUAL,
    EQUAL,
    NOT_EQUAL
};

class BinaryExpr : public Expression {

private:
    const Variable *_lhs;
    const Variable *_rhs;
    BinaryOp _op;

public:
    BinaryExpr(BinaryOp op, const Variable *lhs, const Variable *rhs) noexcept
        : _op{op}, _lhs{lhs}, _rhs{rhs} {}

    [[nodiscard]] const Variable *lhs() const noexcept { return _lhs; }
    [[nodiscard]] const Variable *rhs() const noexcept { return _rhs; }
    [[nodiscard]] BinaryOp op() const noexcept { return _op; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

class MemberExpr : public Expression {

private:
    const Variable *_self;
    size_t _member;

public:
    MemberExpr(const Variable *self, size_t member_index) noexcept : _self{self}, _member{member_index} {}
    [[nodiscard]] const Variable *self() const noexcept { return _self; }
    [[nodiscard]] size_t member_index() const noexcept { return _member; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

class LiteralExpr : public Expression {};

class CallExpr : public Expression {

private:
    std::string _name;
    std::vector<const Variable *> _arguments;

public:
    CallExpr(std::string name, std::vector<const Variable *> args) noexcept
        : _name{std::move(name)}, _arguments{std::move(args)} {}
    
    [[nodiscard]] const std::string &name() const noexcept { return _name; }
    [[nodiscard]] const std::vector<const Variable *> &arguments() const noexcept { return _arguments; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};


#undef LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR

}// namespace luisa::compute
