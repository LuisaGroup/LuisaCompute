//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <variant>
#include <charconv>

#include <core/concepts.h>
#include <core/basic_types.h>
#include <ast/variable.h>
#include <core/logging.h>

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

namespace detail {

template<typename T>
struct make_literal_value {
    static_assert(always_false_v<T>);
};

template<typename... T>
struct make_literal_value<std::tuple<T...>> {
    using type = std::variant<T...>;
};

}

class LiteralExpr : public Expression {

public:
    using Value = typename detail::make_literal_value<basic_types>::type;

private:
    Value _value;

public:
    LiteralExpr(const Type *type, Value v) noexcept
        : Expression{type}, _value{v} {}
    [[nodiscard]] const Value &value() const noexcept { return _value; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

class RefExpr : public Expression {

private:
    Variable _variable;

public:
    explicit RefExpr(Variable v) noexcept
        : Expression{v.type()}, _variable{v} {}
    [[nodiscard]] auto variable() const noexcept { return _variable; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

class ConstantExpr : public Expression {

private:
    uint64_t _hash;

public:
    explicit ConstantExpr(const Type *type, uint64_t hash) noexcept
        : Expression{type}, _hash{hash} {}
    [[nodiscard]] auto hash() const noexcept { return _hash; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

class CallExpr : public Expression {

public:
    using ArgumentList = std::span<const Expression *>;
    static constexpr auto uid_npos = std::numeric_limits<uint32_t>::max();

private:
    std::string_view _name;
    ArgumentList _arguments;
    uint32_t _uid{uid_npos};

public:
    CallExpr(const Type *type, std::string_view name, ArgumentList args) noexcept
        : Expression{type}, _name{name}, _arguments{args} {
        using namespace std::string_view_literals;
        if (auto prefix = "custom_"sv; _name.starts_with(prefix)) {
            auto uid_str = _name.substr(prefix.size());
            if (auto [p, ec] = std::from_chars(uid_str.data(), uid_str.data() + uid_str.size(), _uid);
                ec != std::errc{}) {
                LUISA_ERROR_WITH_LOCATION("Invalid custom callable function: {}.", _name);
            }
        }
    }
    [[nodiscard]] auto name() const noexcept { return _name; }
    [[nodiscard]] auto arguments() const noexcept { return _arguments; }
    [[nodiscard]] auto is_builtin() const noexcept { return _uid == uid_npos; }
    [[nodiscard]] auto uid() const noexcept { return _uid; }
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

public:
    CastExpr(const Type *type, CastOp op, const Expression *src) noexcept
        : Expression{type}, _source{src}, _op{op} {}
    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] auto expression() const noexcept { return _source; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

#undef LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR

}// namespace luisa::compute
