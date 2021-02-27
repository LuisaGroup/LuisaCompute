//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <variant>

#include <core/concepts.h>
#include <core/data_types.h>
#include <ast/variable.h>

namespace luisa::compute {

struct ExprVisitor;

class Expression : public concepts::Noncopyable {

private:
    const Type *_type;

protected:
    ~Expression() noexcept = default;

public:
    explicit Expression(const Type *type) noexcept : _type{type} {}
    [[nodiscard]] auto type() const noexcept { return _type; }
    virtual void accept(ExprVisitor &) const = 0;
};

class UnaryExpr;
class BinaryExpr;
class MemberExpr;
class AccessExpr;
class ValueExpr;
class CallExpr;
class CastExpr;

struct ExprVisitor {
    virtual void visit(const UnaryExpr *) = 0;
    virtual void visit(const BinaryExpr *) = 0;
    virtual void visit(const MemberExpr *) = 0;
    virtual void visit(const AccessExpr *) = 0;
    virtual void visit(const ValueExpr *) = 0;
    virtual void visit(const CallExpr *) = 0;
    virtual void visit(const CastExpr *) = 0;
};

#define LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR() \
    void accept(ExprVisitor &visitor) const override { visitor.visit(this); }

enum struct UnaryOp {
    PLUS,
    MINUS,  // +x, -x
    NOT,    // !x
    BIT_NOT,// ~x
    // Note: We deliberately support *NO* pre and postfix inc/dec operators to avoid possible abuse
};

class UnaryExpr : public Expression {
private:
    const Expression *_operand;
    UnaryOp _op;

public:
    UnaryExpr(const Type *type, UnaryOp op, const Expression *operand) noexcept : Expression{type}, _operand{operand}, _op{op} {}
    [[nodiscard]] auto operand() const noexcept { return _operand; }
    [[nodiscard]] auto op() const noexcept { return _op; }
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
    const Expression *_lhs;
    const Expression *_rhs;
    BinaryOp _op;

public:
    BinaryExpr(const Type *type, BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept
        : Expression{type}, _op{op}, _lhs{lhs}, _rhs{rhs} {}

    [[nodiscard]] auto lhs() const noexcept { return _lhs; }
    [[nodiscard]] auto rhs() const noexcept { return _rhs; }
    [[nodiscard]] auto op() const noexcept { return _op; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

class AccessExpr : public Expression {

private:
    const Expression *_range;
    const Expression *_index;

public:
    AccessExpr(const Type *type, const Expression *range, const Expression *index) noexcept
        : Expression{type}, _range{range}, _index{index} {}

    [[nodiscard]] auto range() const noexcept { return _range; }
    [[nodiscard]] auto index() const noexcept { return _index; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

class MemberExpr : public Expression {

private:
    const Expression *_self;
    size_t _member;

public:
    MemberExpr(const Type *type, const Expression *self, size_t member_index) noexcept
        : Expression{type}, _self{self}, _member{member_index} {}
    [[nodiscard]] auto self() const noexcept { return _self; }
    [[nodiscard]] auto member_index() const noexcept { return _member; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

class ValueExpr : public Expression {

public:
    using Value = std::variant<
        Variable,
        bool, float, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
        bool2, float2, char2, uchar2, short2, ushort2, int2, uint2,
        bool3, float3, char3, uchar3, short3, ushort3, int3, uint3,
        bool4, float4, char4, uchar4, short4, ushort4, int4, uint4,
        float3x3, float4x4>;

private:
    Value _value;

public:
    ValueExpr(const Type *type, Value v) noexcept
        : Expression{type}, _value{std::move(v)} {}
    [[nodiscard]] const Value &value() const noexcept { return _value; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

class CallExpr : public Expression {

public:
    using ArgumentList = std::span<const Expression *>;

private:
    std::string_view _name;
    ArgumentList _arguments;

public:
    CallExpr(const Type *type, std::string_view name, ArgumentList args) noexcept
        : Expression{type}, _name{name}, _arguments{args} {}
    [[nodiscard]] auto name() const noexcept { return _name; }
    [[nodiscard]] auto arguments() const noexcept { return _arguments; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

enum struct CastOp {
    STATIC,
    REINTERPRET,
    BITWISE
};

class CastExpr : public Expression {

private:
    const Expression *_source;
    CastOp _op;

public:
    CastExpr(const Type *type, CastOp op, const Expression *src) noexcept
        : Expression{type}, _source{src}, _op{op} {}
    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] auto expression() const noexcept { return _source; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

#undef LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR

// make sure we can allocate them using arena...
static_assert(std::is_trivially_destructible_v<UnaryExpr>);
static_assert(std::is_trivially_destructible_v<BinaryExpr>);
static_assert(std::is_trivially_destructible_v<MemberExpr>);
static_assert(std::is_trivially_destructible_v<AccessExpr>);
static_assert(std::is_trivially_destructible_v<ValueExpr>);
static_assert(std::is_trivially_destructible_v<CallExpr>);
static_assert(std::is_trivially_destructible_v<CastExpr>);

}// namespace luisa::compute
