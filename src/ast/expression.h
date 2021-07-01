//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <utility>
#include <variant>
#include <charconv>

#include <core/concepts.h>
#include <core/basic_types.h>
#include <core/logging.h>
#include <ast/variable.h>
#include <ast/function.h>
#include <ast/op.h>

namespace luisa::compute {

struct ExprVisitor;

namespace detail {
class FunctionBuilder;
}

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

public:
    static constexpr auto swizzle_mask = 0xff00000000ull;
    static constexpr auto swizzle_shift = 32u;

private:
    const Expression *_self;
    uint64_t _member;

    void _mark(Variable::Usage usage) const noexcept override { _self->mark(usage); }

public:
    MemberExpr(const Type *type, const Expression *self, size_t member_index) noexcept
        : Expression{type}, _self{self}, _member{member_index} {}
    MemberExpr(const Type *type, const Expression *self, size_t swizzle_size, uint64_t swizzle_code) noexcept
        : Expression{type}, _self{self},
          _member{(static_cast<uint64_t>(swizzle_size) << swizzle_shift) | swizzle_code} {}

    [[nodiscard]] auto is_swizzle() const noexcept { return (_member & swizzle_mask) != 0u; }
    [[nodiscard]] auto self() const noexcept { return _self; }

    [[nodiscard]] auto member_index() const noexcept {
        if (is_swizzle()) {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid member index in swizzled MemberExpr.");
        }
        return static_cast<size_t>(_member);
    }

    [[nodiscard]] auto swizzle_size() const noexcept {
        auto s = (_member & swizzle_mask) >> swizzle_shift;
        if (s == 0u || s > 4u) { LUISA_ERROR_WITH_LOCATION("Invalid swizzle size {}.", s); }
        return static_cast<size_t>(s);
    }

    [[nodiscard]] auto swizzle_index(size_t index) const noexcept {
        if (auto s = swizzle_size(); index >= s) {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid swizzle index {} (count = {}).",
                index, s);
        }
        return static_cast<size_t>((_member >> (index * 4u)) & 0x0fu);
    }

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
        : Expression{type}, _value{v} {}
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

class CallExpr : public Expression {

public:
    using ArgumentList = std::span<const Expression *>;

private:
    ArgumentList _arguments;
    Function _custom;
    CallOp _op;
    void _mark(Variable::Usage) const noexcept override;

public:
    CallExpr(const Type *type, Function callable, ArgumentList args) noexcept
        : Expression{type}, _arguments{args}, _custom{callable}, _op{CallOp::CUSTOM} {}
    CallExpr(const Type *type, CallOp builtin, ArgumentList args) noexcept
        : Expression{type}, _arguments{args}, _custom{}, _op{builtin} {}
    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] auto arguments() const noexcept { return _arguments; }
    [[nodiscard]] auto custom() const noexcept { return _custom; }
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
