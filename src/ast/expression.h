//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <string>
#include <vector>

#include <core/concepts.h>
#include <core/data_types.h>
#include <core/type_info.h>
#include <core/arena.h>
#include <core/union.h>

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

class AccessExpr : public Expression {

private:
    const Variable *_range;
    const Variable *_index;

public:
    AccessExpr(const Variable *range, const Variable *index) noexcept
        : _range{range}, _index{index} {}

    [[nodiscard]] auto range() const noexcept { return _range; }
    [[nodiscard]] auto index() const noexcept { return _index; }
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

class LiteralExpr : public Expression {

public:
    using Value = Union<
        bool, float, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
        bool2, float2, char2, uchar2, short2, ushort2, int2, uint2,
        bool3, float3, char3, uchar3, short3, ushort3, int3, uint3,
        bool4, float4, char4, uchar4, short4, ushort4, int4, uint4,
        float3x3, float4x4>;

private:
    Value _value;

public:
    template<typename T, std::enable_if_t<Value::contains<T>, int> = 0>
    explicit LiteralExpr(T value) noexcept : _value{value} {}

    [[nodiscard]] const Value &value() const noexcept { return _value; }

    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

class CallExpr : public Expression {

public:
    using ArgumentList = ArenaVector<const Variable *>;

private:
    ArenaString _name;
    ArgumentList _arguments;

public:
    CallExpr(ArenaString name, ArgumentList args) noexcept
        : _name{name}, _arguments{std::move(args)} {}

    [[nodiscard]] std::string_view name() const noexcept { return _name; }
    [[nodiscard]] const auto &arguments() const noexcept { return _arguments; }

    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

enum struct CastOp {
    STATIC,
    REINTERPRET,
    BITWISE
};

class CastExpr : public Expression {

private:
    const Variable *_source;
    const TypeInfo *_dest_type;
    CastOp _op;

public:
    CastExpr(CastOp op, const Variable *src, const TypeInfo *dest) noexcept
        : _source{src}, _dest_type{dest}, _op{op} {}
    [[nodiscard]] CastOp op() const noexcept { return _op; }
    [[nodiscard]] auto source() const noexcept { return _source; }
    [[nodiscard]] auto dest_type() const noexcept { return _dest_type; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

#undef LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR

// make sure we can allocate them using arena...
static_assert(std::is_trivially_destructible_v<UnaryExpr>);
static_assert(std::is_trivially_destructible_v<BinaryExpr>);
static_assert(std::is_trivially_destructible_v<MemberExpr>);
static_assert(std::is_trivially_destructible_v<AccessExpr>);
static_assert(std::is_trivially_destructible_v<LiteralExpr>);
static_assert(std::is_trivially_destructible_v<CallExpr>);
static_assert(std::is_trivially_destructible_v<CastExpr>);

}// namespace luisa::compute
