//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <utility>
#include <variant>
#include <charconv>

#include <core/concepts.h>
#include <core/basic_types.h>
#include <ast/variable.h>

namespace luisa::compute {

struct ExprVisitor;

class Expression : public concepts::Noncopyable {

private:
    const Type *_type;
    mutable Variable::Usage _usage{Variable::Usage::NONE};

private:
    virtual void _mark(Variable::Usage usage) const noexcept = 0;

protected:
    ~Expression() noexcept = default;

public:
    explicit Expression(const Type *type) noexcept : _type{type} {}
    [[nodiscard]] auto type() const noexcept { return _type; }
    [[nodiscard]] auto usage() const noexcept { return _usage; }
    virtual void accept(ExprVisitor &) const = 0;
    void mark(Variable::Usage usage) const noexcept;
};

class UnaryExpr;
class BinaryExpr;
class MemberExpr;
class AccessExpr;
class LiteralExpr;
class RefExpr;
class ConstantExpr;
class CallExpr;
class CastExpr;

struct ExprVisitor {
    virtual void visit(const UnaryExpr *) = 0;
    virtual void visit(const BinaryExpr *) = 0;
    virtual void visit(const MemberExpr *) = 0;
    virtual void visit(const AccessExpr *) = 0;
    virtual void visit(const LiteralExpr *) = 0;
    virtual void visit(const RefExpr *) = 0;
    virtual void visit(const ConstantExpr *) = 0;
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

    void _mark(Variable::Usage) const noexcept override { _operand->mark(Variable::Usage::READ); }

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

    void _mark(Variable::Usage) const noexcept override {
        _lhs->mark(Variable::Usage::READ);
        _rhs->mark(Variable::Usage::READ);
    }

public:
    BinaryExpr(const Type *type, BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept
        : Expression{type}, _lhs{lhs}, _rhs{rhs}, _op{op} {}

    [[nodiscard]] auto lhs() const noexcept { return _lhs; }
    [[nodiscard]] auto rhs() const noexcept { return _rhs; }
    [[nodiscard]] auto op() const noexcept { return _op; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

class AccessExpr : public Expression {

private:
    const Expression *_range;
    const Expression *_index;

    void _mark(Variable::Usage usage) const noexcept override {
        _range->mark(usage);
        _index->mark(Variable::Usage::READ);
    }

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

    void _mark(Variable::Usage usage) const noexcept override { _self->mark(usage); }

public:
    MemberExpr(const Type *type, const Expression *self, size_t member_index) noexcept
        : Expression{type}, _self{self}, _member{member_index} {}
    [[nodiscard]] auto self() const noexcept { return _self; }
    [[nodiscard]] auto member_index() const noexcept { return _member; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

namespace detail {

template<typename T>
struct make_literal_value {
    static_assert(always_false_v<T>);
};

template<typename... T>
struct make_literal_value<std::tuple<T...>> {
    using type = std::variant<T...>;
};

}// namespace detail

class LiteralExpr : public Expression {

public:
    using Value = typename detail::make_literal_value<basic_types>::type;

private:
    Value _value;
    void _mark(Variable::Usage) const noexcept override {}

public:
    LiteralExpr(const Type *type, Value v) noexcept
        : Expression{type}, _value{std::move(v)} {}
    [[nodiscard]] decltype(auto) value() const noexcept { return _value; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

class RefExpr : public Expression {

private:
    Variable _variable;
    void _mark(Variable::Usage usage) const noexcept override;

public:
    explicit RefExpr(Variable v) noexcept
        : Expression{v.type()}, _variable{v} {}
    [[nodiscard]] auto variable() const noexcept { return _variable; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

class ConstantExpr : public Expression {

private:
    uint64_t _hash;
    void _mark(Variable::Usage) const noexcept override {}

public:
    explicit ConstantExpr(const Type *type, uint64_t hash) noexcept
        : Expression{type}, _hash{hash} {}
    [[nodiscard]] auto hash() const noexcept { return _hash; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

enum struct CallOp {
    
    CUSTOM,
    
    ALL,
    ANY,
    NONE,
    
    SELECT,
    
    CLAMP,
    LERP,
    SATURATE,
    SIGN,
    
    STEP,
    SMOOTHSTEP,
    
    ABS,
    MIN,
    MIN3,
    MAX,
    MAX3,
    
    CLZ,
    CTZ,
    POPCOUNT,
    REVERSEBITS,
    ROTATE,
    
    ISINF,
    ISNAN,
    
    ACOS,
    ACOSH,
    ASIN,
    ASINH,
    ATAN,
    ATAN2,
    ATANH,
    
    COS,
    COSH,
    SIN,
    SINH,
    TAN,
    TANH,
    
    EXP,
    EXP2,
    EXP10,
    LOG,
    LOG2,
    LOG10,
    
    SQRT,
    RSQRT,
    
    CEIL,
    FLOOR,
    FRACT,
    TRUNC,
    ROUND,
    FMOD,
    
    DEGREES,
    RADIANS,
    
    FMA,
    COPYSIGN,
    
    CROSS,
    DOT,
    DISTANCE,
    DISTANCE_SQUARED,
    LENGTH,
    LENGTH_SQUARED,
    NORMALIZE,
    FACEFORWARD,

    DETERMINANT,
    TRANSPOSE,
    INVERSE,
    
    GROUP_MEMORY_BARRIER,
    DEVICE_MEMORY_BARRIER,
    ALL_MEMORY_BARRIER,
    
    ATOMIC_LOAD,
    ATOMIC_STORE,
    ATOMIC_EXCHANGE,
    ATOMIC_COMPARE_EXCHANGE,
    ATMOIC_FETCH_ADD,
    ATMOIC_FETCH_SUB,
    ATMOIC_FETCH_AND,
    ATMOIC_FETCH_OR,
    ATMOIC_FETCH_XOR,
    ATMOIC_FETCH_MIN,
    ATMOIC_FETCH_MAX,

    TEXTURE_READ,
    TEXTURE_WRITE,
    TEXTURE_SAMPLE,
};

class CallExpr : public Expression {

public:
    using ArgumentList = std::span<const Expression *>;

private:
    ArgumentList _arguments;
    CallOp _op;
    uint32_t _uid;
    void _mark(Variable::Usage) const noexcept override;

public:
    CallExpr(const Type *type, uint32_t uid, ArgumentList args) noexcept
        : Expression{type}, _arguments{args}, _op{CallOp::CUSTOM}, _uid{uid} {}
    CallExpr(const Type *type, CallOp op, ArgumentList args) noexcept
        : Expression{type}, _arguments{args}, _op{op}, _uid{} {}
    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] auto arguments() const noexcept { return _arguments; }
    [[nodiscard]] auto uid() const noexcept { return _uid; }
    [[nodiscard]] auto is_builtin() const noexcept { return _op != CallOp::CUSTOM; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

enum struct CastOp {
    STATIC,
    BITWISE
};

class CastExpr : public Expression {

private:
    const Expression *_source;
    CastOp _op;
    void _mark(Variable::Usage) const noexcept override { _source->mark(Variable::Usage::READ); }

public:
    CastExpr(const Type *type, CastOp op, const Expression *src) noexcept
        : Expression{type}, _source{src}, _op{op} {}
    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] auto expression() const noexcept { return _source; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

#undef LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR

}// namespace luisa::compute
