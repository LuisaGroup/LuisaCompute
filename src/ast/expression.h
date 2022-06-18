//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <utility>
#include <variant>
#include <charconv>
#include <algorithm>
#include <numeric>

#include <core/concepts.h>
#include <core/basic_types.h>
#include <core/logging.h>
#include <core/hash.h>
#include <core/stl.h>
#include <ast/variable.h>
#include <ast/function.h>
#include <ast/op.h>
#include <ast/constant_data.h>

namespace luisa::compute {

struct ExprVisitor;

namespace detail {
class FunctionBuilder;
}

/**
 * @brief Base expression class
 * 
 */
class LC_AST_API Expression : public concepts::Noncopyable {

public:
    /// Expression type
    enum struct Tag : uint32_t {
        UNARY,
        BINARY,
        MEMBER,
        ACCESS,
        LITERAL,
        REF,
        CONSTANT,
        CALL,
        CAST
    };

private:
    const Type *_type;
    mutable uint64_t _hash{0u};
    mutable bool _hash_computed{false};
    Tag _tag;

protected:
    mutable Usage _usage{Usage::NONE};
    virtual void _mark(Usage usage) const noexcept = 0;
    [[nodiscard]] virtual uint64_t _compute_hash() const noexcept = 0;

public:
    /**
     * @brief Construct a new Expression object
     * 
     * @param tag type of expression
     * @param type result type of expression
     */
    explicit Expression(Tag tag, const Type *type) noexcept : _type{type}, _tag{tag} {}
    virtual ~Expression() noexcept = default;
    [[nodiscard]] auto type() const noexcept { return _type; }
    [[nodiscard]] auto usage() const noexcept { return _usage; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    virtual void accept(ExprVisitor &) const = 0;
    void mark(Usage usage) const noexcept;
    [[nodiscard]] uint64_t hash() const noexcept;
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

/// Unary expression
class UnaryExpr final : public Expression {

private:
    const Expression *_operand;
    UnaryOp _op;

protected:
    void _mark(Usage) const noexcept override {}
    uint64_t _compute_hash() const noexcept override {
        return hash64(_op, _operand->hash());
    }

public:
    /**
     * @brief Construct a new UnaryExpr object
     * 
     * @param type type
     * @param op UnaryOp
     * @param operand operand will be mark as Usage::READ
     */
    UnaryExpr(const Type *type, UnaryOp op, const Expression *operand) noexcept
        : Expression{Tag::UNARY, type}, _operand{operand}, _op{op} { _operand->mark(Usage::READ); }
    [[nodiscard]] auto operand() const noexcept { return _operand; }
    [[nodiscard]] auto op() const noexcept { return _op; }

public:
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

/// Binary expression
class BinaryExpr final : public Expression {

private:
    const Expression *_lhs;
    const Expression *_rhs;
    BinaryOp _op;

protected:
    void _mark(Usage) const noexcept override {}
    uint64_t _compute_hash() const noexcept override {
        auto hl = _lhs->hash();
        auto hr = _rhs->hash();
        return hash64(_op, hash64(hl, hr));
    }

public:
    /**
     * @brief Construct a new BinaryExpr object
     * 
     * @param type type
     * @param op BinaryOp
     * @param lhs lhs will be marked as Usage::READ
     * @param rhs rhs will be marked as Usage::READ
     */
    BinaryExpr(const Type *type, BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept
        : Expression{Tag::BINARY, type}, _lhs{lhs}, _rhs{rhs}, _op{op} {
        _lhs->mark(Usage::READ);
        _rhs->mark(Usage::READ);
    }

    [[nodiscard]] auto lhs() const noexcept { return _lhs; }
    [[nodiscard]] auto rhs() const noexcept { return _rhs; }
    [[nodiscard]] auto op() const noexcept { return _op; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

/// Access expression
class AccessExpr final : public Expression {

private:
    const Expression *_range;
    const Expression *_index;

protected:
    void _mark(Usage usage) const noexcept override { _range->mark(usage); }
    uint64_t _compute_hash() const noexcept override {
        return hash64(_index->hash(), _range->hash());
    }

public:
    /**
     * @brief Construct a new AccessExpr object
     * 
     * @param type type
     * @param range range will be marked as Usage::READ
     * @param index index will be marked as Usage::READ
     */
    AccessExpr(const Type *type, const Expression *range, const Expression *index) noexcept
        : Expression{Tag::ACCESS, type}, _range{range}, _index{index} {
        _range->mark(Usage::READ);
        _index->mark(Usage::READ);
    }

    [[nodiscard]] auto range() const noexcept { return _range; }
    [[nodiscard]] auto index() const noexcept { return _index; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

/// Member expression
class MemberExpr final : public Expression {

public:
    static constexpr auto swizzle_mask = 0xff000000u;
    static constexpr auto swizzle_shift = 24u;

private:
    const Expression *_self;
    uint32_t _swizzle_size;
    uint32_t _swizzle_code;

protected:
    void _mark(Usage usage) const noexcept override { _self->mark(usage); }
    uint64_t _compute_hash() const noexcept override {
        return hash64(make_uint2(_swizzle_size, _swizzle_code), _self->hash());
    }

public:
    /**
     * @brief Construct a new MemberExpr object accessing by index
     * 
     * @param type type
     * @param self where to get member
     * @param member_index index of member
     */
    MemberExpr(const Type *type, const Expression *self, uint member_index) noexcept
        : Expression{Tag::MEMBER, type}, _self{self},
          _swizzle_size{0u}, _swizzle_code{member_index} {}

    /**
     * @brief Construct a new Member Expr object accessing by swizzling
     * 
     * Swizzle size must be in [1, 4]. Swizzle code represents orders and indexes to fetch.
     * 
     * For example, consider a float4 object whose members named x, y, z and w, its member indexes are 0, 1, 2 and 3.
     * 
     * If you need to get float4.xyzw(which returns a float4 (x, y, z, w)), the swizzle code is coded as 0x3210u and swizzle size is 4.
     * 
     * Another example is float4.yyw(which returns a float3 (y, y, w)), thw swizzle code is codes as 0x0311u and swizzle size is 3.
     * 
     * @param type type
     * @param self where to get member
     * @param swizzle_size swizzle size
     * @param swizzle_code swizzle code
     */
    MemberExpr(const Type *type, const Expression *self, uint swizzle_size, uint swizzle_code) noexcept
        : Expression{Tag::MEMBER, type}, _self{self},
          _swizzle_size{swizzle_size}, _swizzle_code{swizzle_code} {
        LUISA_ASSERT(_swizzle_size != 0u && _swizzle_size <= 4u,
                     "Swizzle size must be in [1, 4]");
    }

    [[nodiscard]] auto self() const noexcept { return _self; }

    [[nodiscard]] auto swizzle_size() const noexcept {
        LUISA_ASSERT(_swizzle_size != 0u && _swizzle_size <= 4u,
                     "Invalid swizzle size {}.", _swizzle_size);
        return _swizzle_size;
    }

    [[nodiscard]] auto is_swizzle() const noexcept { return _swizzle_size != 0u; }

    [[nodiscard]] auto swizzle_code() const noexcept {
        LUISA_ASSERT(is_swizzle(), "MemberExpr is not swizzled.");
        return _swizzle_code;
    }

    [[nodiscard]] auto swizzle_index(uint index) const noexcept {
        if (auto s = swizzle_size(); index >= s) {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid swizzle index {} (count = {}).",
                index, s);
        }
        return (_swizzle_code >> (index * 4u)) & 0x0fu;
    }

    [[nodiscard]] auto member_index() const noexcept {
        LUISA_ASSERT(!is_swizzle(), "MemberExpr is not a member");
        return _swizzle_code;
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
    using type = luisa::variant<T...>;
};

template<typename T>
using make_literal_value_t = typename make_literal_value<T>::type;

}// namespace detail

/// TODO
class LiteralExpr final : public Expression {

public:
    using Value = detail::make_literal_value_t<basic_types>;

private:
    Value _value;

protected:
    void _mark(Usage) const noexcept override {}
    uint64_t _compute_hash() const noexcept override {
        return luisa::visit([](auto &&v) noexcept { return hash64(v); }, _value);
    }

public:
    /**
     * @brief Construct a new LiteralExpr object
     * 
     * @param type type
     * @param v value
     */
    LiteralExpr(const Type *type, Value v) noexcept
        : Expression{Tag::LITERAL, type}, _value{v} {}
    [[nodiscard]] decltype(auto) value() const noexcept { return _value; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

/// Reference expression
class RefExpr final : public Expression {

private:
    Variable _variable;

protected:
    void _mark(Usage usage) const noexcept override;
    uint64_t _compute_hash() const noexcept override {
        return hash64(_variable.hash());
    }

public:
    /**
     * @brief Construct a new RefExpr object
     * 
     * @param v variable referenced
     */
    explicit RefExpr(Variable v) noexcept
        : Expression{Tag::REF, v.type()}, _variable{v} {}
    [[nodiscard]] auto variable() const noexcept { return _variable; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

/// Constant expression
class ConstantExpr final : public Expression {

private:
    ConstantData _data;

protected:
    void _mark(Usage) const noexcept override {}
    uint64_t _compute_hash() const noexcept override {
        return hash64(_data.hash());
    }

public:
    /**
     * @brief Construct a new ConstantExpr object
     * 
     * @param type type
     * @param data const data
     */
    explicit ConstantExpr(const Type *type, ConstantData data) noexcept
        : Expression{Tag::CONSTANT, type}, _data{data} {}
    [[nodiscard]] auto data() const noexcept { return _data; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

/// Call expression
class CallExpr final : public Expression {

public:
    using ArgumentList = luisa::vector<const Expression *>;

private:
    ArgumentList _arguments;
    Function _custom;
    CallOp _op;

protected:
    void _mark(Usage) const noexcept override {}
    void _mark() const noexcept;
    uint64_t _compute_hash() const noexcept override {
        auto hash = hash64(_op);
        for (auto &&a : _arguments) { hash = hash64(a->hash(), hash); }
        if (_op == CallOp::CUSTOM) { hash = hash64(_custom.hash(), hash); }
        return hash;
    }

public:
    /**
     * @brief Construct a new CallExpr object calling custom function
     * 
     * @param type type
     * @param callable function to call
     * @param args arguments of function
     */
    CallExpr(const Type *type, Function callable, ArgumentList args) noexcept
        : Expression{Tag::CALL, type},
          _arguments{std::move(args)},
          _custom{callable},
          _op{CallOp::CUSTOM} { _mark(); }
    /**
     * @brief Construct a new CallExpr object calling builtin function
     * 
     * @param type type
     * @param builtin builtin function tag
     * @param args arguments of function
     */
    CallExpr(const Type *type, CallOp builtin, ArgumentList args) noexcept
        : Expression{Tag::CALL, type},
          _arguments{std::move(args)},
          _custom{},
          _op{builtin} { _mark(); }
    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] auto arguments() const noexcept { return luisa::span{_arguments}; }
    [[nodiscard]] auto custom() const noexcept { return _custom; }
    [[nodiscard]] auto is_builtin() const noexcept { return _op != CallOp::CUSTOM; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

enum struct CastOp {
    STATIC,
    BITWISE
};

/// Cast expression
class CastExpr final : public Expression {

private:
    const Expression *_source;
    CastOp _op;

protected:
    void _mark(Usage) const noexcept override {}
    uint64_t _compute_hash() const noexcept override {
        return hash64(_op, _source->hash());
    }

public:
    /**
     * @brief Construct a new CastExpr object
     * 
     * @param type type
     * @param op cast type(static, bitwise)
     * @param src source expression
     */
    CastExpr(const Type *type, CastOp op, const Expression *src) noexcept
        : Expression{Tag::CAST, type}, _source{src}, _op{op} { _source->mark(Usage::READ); }
    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] auto expression() const noexcept { return _source; }
    LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR()
};

#undef LUISA_MAKE_EXPRESSION_ACCEPT_VISITOR

}// namespace luisa::compute
