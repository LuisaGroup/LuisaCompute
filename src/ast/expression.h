//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <variant>
#ifndef LC_BACKEND
#include <core/concepts.h>
#include <core/data_types.h>
#include <core/memory.h>
#include <core/union.h>
#include <ast/type.h>
#include <ast/variable.h>
#else
#include "variable.h"
#include "../core/concepts.h"
#include "../core/data_types.h"
#endif

namespace luisa::compute {
class IType;
struct ExprVisitor;

class Expression : public Noncopyable {

private:
    const IType *_type;

protected:
    ~Expression() noexcept = default;

public:
    explicit Expression(const IType *type) noexcept : _type{type} {}
#ifndef LC_BACKEND
    [[nodiscard]] auto type() const noexcept { return static_cast<Type const *>(_type); }
#endif
    [[nodiscard]] auto itype() const noexcept { return _type; }
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
    UnaryExpr(const IType *type, UnaryOp op, const Expression *operand) noexcept : Expression{type}, _operand{operand}, _op{op} {}
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
    BinaryExpr(const IType *type, BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept
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
    AccessExpr(const IType *type, const Expression *range, const Expression *index) noexcept
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
    MemberExpr(const IType *type, const Expression *self, size_t member_index) noexcept
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
    // TODO...
    ValueExpr(const IType *type, Value v) noexcept
        : Expression{type}, _value{std::move(v)} {}
    [[nodiscard]] const Value &value() const noexcept { return _value; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};
class CallExpr {
public:
    virtual std::string_view name() const noexcept = 0;
    virtual Expression const *const *arguments_ptr() const noexcept = 0;
    virtual size_t arguments_count() const noexcept = 0;
};
#ifndef LC_BACKEND
class CallExpr_Impl : public CallExpr, public Expression {

public:
    using ArgumentList = ArenaVector<const Expression *>;

private:
    ArenaString _name;
    ArgumentList _arguments;

public:
    CallExpr_Impl(const IType *type, ArenaString name, ArgumentList args) noexcept
        : Expression{type}, _name{name}, _arguments{std::move(args)} {}
    [[nodiscard]] std::string_view name() const noexcept override { return _name; }
    [[nodiscard]] const auto &arguments() const noexcept { return _arguments; }
    [[nodiscard]] Expression const *const *arguments_ptr() const noexcept override {
        return _arguments.data();
    }
    [[nodiscard]] size_t arguments_count() const noexcept {
        return _arguments.size();
    }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};
#endif
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
    CastExpr(const IType *type, CastOp op, const Expression *src) noexcept
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
#ifndef LC_BACKEND
static_assert(std::is_trivially_destructible_v<CallExpr_Impl>);
#endif
static_assert(std::is_trivially_destructible_v<CastExpr>);

}// namespace luisa::compute
