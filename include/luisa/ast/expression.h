#pragma once

#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/concepts.h>
#include <luisa/core/basic_types.h>
#include <luisa/ast/variable.h>
#include <luisa/ast/op.h>
#include <luisa/ast/constant_data.h>

#include <utility>

namespace luisa::compute {

class Statement;
class Function;
struct ExprVisitor;

class ExternalFunction;

namespace detail {
class FunctionBuilder;
}// namespace detail

/**
 * @brief Base expression class
 * 
 */
class LC_AST_API Expression : public concepts::Noncopyable {
    friend class CallableLibrary;

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
        CAST,
        TYPE_ID,
        STRING_ID,
        FUNC_REF,
        CPUCUSTOM,
        GPUCUSTOM
    };

private:
    const Type *_type{nullptr};
    mutable uint64_t _hash{0u};
    const detail::FunctionBuilder *_builder{nullptr};
    mutable bool _hash_computed{false};
    Tag _tag{};

protected:
    mutable Usage _usage{Usage::NONE};
    virtual void _mark(Usage usage) const noexcept = 0;
    [[nodiscard]] virtual uint64_t _compute_hash() const noexcept = 0;
    Expression() noexcept = default;

public:
    /**
     * @brief Construct a new Expression object
     * 
     * @param tag type of expression
     * @param type result type of expression
     */
    Expression(Tag tag, const Type *type) noexcept;
    virtual ~Expression() noexcept = default;
    [[nodiscard]] auto builder() const noexcept { return _builder; }
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
class TypeIDExpr;
class StringIDExpr;
class FuncRefExpr;
class CpuCustomOpExpr;
class GpuCustomOpExpr;

struct LC_AST_API ExprVisitor {
    virtual void visit(const UnaryExpr *) = 0;
    virtual void visit(const BinaryExpr *) = 0;
    virtual void visit(const MemberExpr *) = 0;
    virtual void visit(const AccessExpr *) = 0;
    virtual void visit(const LiteralExpr *) = 0;
    virtual void visit(const RefExpr *) = 0;
    virtual void visit(const ConstantExpr *) = 0;
    virtual void visit(const CallExpr *) = 0;
    virtual void visit(const CastExpr *) = 0;
    virtual void visit(const TypeIDExpr *) = 0;
    virtual void visit(const StringIDExpr *) = 0;
    virtual void visit(const FuncRefExpr *);
    virtual void visit(const CpuCustomOpExpr *);
    virtual void visit(const GpuCustomOpExpr *);
    virtual ~ExprVisitor() noexcept = default;
};

#define LUISA_EXPRESSION_COMMON() \
    void accept(ExprVisitor &visitor) const override { visitor.visit(this); }

/// Unary expression
class LC_AST_API UnaryExpr final : public Expression {
    friend class CallableLibrary;

private:
    const Expression *_operand;
    UnaryOp _op;
    UnaryExpr() noexcept = default;

protected:
    void _mark(Usage) const noexcept override {}
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

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
    LUISA_EXPRESSION_COMMON()
};

/// Binary expression
class LC_AST_API BinaryExpr final : public Expression {
    friend class CallableLibrary;

private:
    const Expression *_lhs;
    const Expression *_rhs;
    BinaryOp _op;
    BinaryExpr() noexcept = default;
protected:
    void _mark(Usage) const noexcept override {}
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    /**
     * @brief Construct a new BinaryExpr object
     * 
     * @param type type
     * @param op BinaryOp
     * @param lhs lhs will be marked as Usage::READ
     * @param rhs rhs will be marked as Usage::READ
     */
    BinaryExpr(const Type *type, BinaryOp op,
               const Expression *lhs,
               const Expression *rhs) noexcept
        : Expression{Tag::BINARY, type}, _lhs{lhs}, _rhs{rhs}, _op{op} {
        _lhs->mark(Usage::READ);
        _rhs->mark(Usage::READ);
    }

    [[nodiscard]] auto lhs() const noexcept { return _lhs; }
    [[nodiscard]] auto rhs() const noexcept { return _rhs; }
    [[nodiscard]] auto op() const noexcept { return _op; }
    LUISA_EXPRESSION_COMMON()
};

/// Access expression
class LC_AST_API AccessExpr final : public Expression {
    friend class CallableLibrary;

private:
    const Expression *_range;
    const Expression *_index;
    AccessExpr() noexcept = default;
protected:
    void _mark(Usage usage) const noexcept override { _range->mark(usage); }
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

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
    LUISA_EXPRESSION_COMMON()
};

/// Member expression
class LC_AST_API MemberExpr final : public Expression {
    friend class CallableLibrary;

public:
    static constexpr auto swizzle_mask = 0xff000000u;
    static constexpr auto swizzle_shift = 24u;

private:
    const Expression *_self;
    uint32_t _swizzle_size;
    uint32_t _swizzle_code;
    MemberExpr() noexcept = default;

protected:
    void _mark(Usage usage) const noexcept override { _self->mark(usage); }
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    /**
     * @brief Construct a new MemberExpr object accessing by index
     * 
     * @param type type
     * @param self where to get member
     * @param member_index index of member
     */
    MemberExpr(const Type *type,
               const Expression *self,
               uint member_index) noexcept;

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
    MemberExpr(const Type *type,
               const Expression *self,
               uint swizzle_size,
               uint swizzle_code) noexcept;

    [[nodiscard]] auto self() const noexcept { return _self; }
    [[nodiscard]] auto is_swizzle() const noexcept { return _swizzle_size != 0u; }

    [[nodiscard]] uint swizzle_size() const noexcept;
    [[nodiscard]] uint swizzle_code() const noexcept;
    [[nodiscard]] uint swizzle_index(uint index) const noexcept;
    [[nodiscard]] uint member_index() const noexcept;

    LUISA_EXPRESSION_COMMON()
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

using LiteralValueVariant = make_literal_value_t<basic_types>;

struct LiteralValue : LiteralValueVariant {

    using variant_type = LiteralValueVariant;

    LiteralValue() noexcept = default;

    template<typename T>
    LiteralValue(T &&x) noexcept : variant_type{std::forward<T>(x)} {}

    template<typename T>
        requires std::same_as<std::remove_cvref_t<T>, long>
    LiteralValue(T &&x) noexcept
        : variant_type{static_cast<canonical_c_long>(x)} {}

    template<typename T>
        requires std::same_as<std::remove_cvref_t<T>, unsigned long>
    LiteralValue(T &&x) noexcept
        : variant_type{static_cast<canonical_c_ulong>(x)} {}

    [[nodiscard]] auto to_variant() const noexcept {
        return static_cast<const variant_type &>(*this);
    }
};

}// namespace detail

class LC_AST_API LiteralExpr final : public Expression {
    friend class CallableLibrary;

public:
    using Value = detail::LiteralValue;

private:
    Value _value;
    LiteralExpr() noexcept = default;

protected:
    void _mark(Usage) const noexcept override {}
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    /**
     * @brief Construct a new LiteralExpr object
     * 
     * @param type type
     * @param v value
     */
    LiteralExpr(const Type *type, Value v) noexcept
        : Expression{Tag::LITERAL, type}, _value{std::move(v)} {}
    [[nodiscard]] decltype(auto) value() const noexcept { return _value; }
    LUISA_EXPRESSION_COMMON()
};

/// Reference expression
class LC_AST_API RefExpr final : public Expression {
    friend class CallableLibrary;

private:
    Variable _variable;
    RefExpr() noexcept = default;

protected:
    void _mark(Usage usage) const noexcept override;
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    /**
     * @brief Construct a new RefExpr object
     * 
     * @param v variable referenced
     */
    explicit RefExpr(Variable v) noexcept
        : Expression{Tag::REF, v.type()}, _variable{v} {}
    [[nodiscard]] auto variable() const noexcept { return _variable; }
    LUISA_EXPRESSION_COMMON()
};

/// Constant expression
class LC_AST_API ConstantExpr final : public Expression {
    friend class CallableLibrary;

private:
    ConstantData _data;
    ConstantExpr() noexcept = default;

protected:
    void _mark(Usage) const noexcept override {}
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    /**
     * @brief Construct a new ConstantExpr object
     * 
     * @param type type
     * @param data const data
     */
    explicit ConstantExpr(ConstantData data) noexcept
        : Expression{Tag::CONSTANT, data.type()}, _data{data} {}
    [[nodiscard]] auto data() const noexcept { return _data; }
    LUISA_EXPRESSION_COMMON()
};

/// Call expression
class LC_AST_API CallExpr final : public Expression {
    friend class CallableLibrary;

public:
    using ArgumentList = luisa::vector<const Expression *>;

    using CustomCallee = const detail::FunctionBuilder *;
    using ExternalCallee = const ExternalFunction *;

    using Callee = luisa::variant<
        luisa::monostate,
        CustomCallee,
        ExternalCallee>;

private:
    ArgumentList _arguments;
    CallOp _op;
    Callee _func;
    CallExpr() noexcept = default;

protected:
    void _mark(Usage) const noexcept override {}
    void _mark() const noexcept;
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    // FIXME: too hacky
    void _unsafe_set_custom(CustomCallee callee) const noexcept;

public:
    /**
     * @brief Construct a new CallExpr object calling custom function
     * 
     * @param type type
     * @param callable function to call
     * @param args arguments of function
     */
    CallExpr(const Type *type, Function callable, ArgumentList args) noexcept;
    /**
     * @brief Construct a new CallExpr object calling builtin function
     * 
     * @param type type
     * @param builtin builtin function tag
     * @param args arguments of function
     */
    CallExpr(const Type *type, CallOp builtin, ArgumentList args) noexcept;
    /**
     * @brief Construct a new CallExpr object calling external function
     * 
     * @param type type
     * @param external_name external function name
     * @param args arguments of function
     */
    CallExpr(const Type *type, const ExternalFunction *external, ArgumentList args) noexcept;
    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] auto arguments() const noexcept { return luisa::span{_arguments}; }
    [[nodiscard]] auto is_builtin() const noexcept { return _op > CallOp::EXTERNAL; }
    [[nodiscard]] auto is_custom() const noexcept { return _op == CallOp::CUSTOM; }
    [[nodiscard]] auto is_external() const noexcept { return _op == CallOp::EXTERNAL; }
    [[nodiscard]] Function custom() const noexcept;
    [[nodiscard]] const ExternalFunction *external() const noexcept;
    LUISA_EXPRESSION_COMMON()
};

enum struct CastOp {
    STATIC,
    BITWISE
};

/// Cast expression
class LC_AST_API CastExpr final : public Expression {
    friend class CallableLibrary;

private:
    const Expression *_source;
    CastOp _op;
    CastExpr() noexcept = default;

protected:
    void _mark(Usage) const noexcept override {}
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

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
    LUISA_EXPRESSION_COMMON()
};

class LC_AST_API TypeIDExpr final : public Expression {
    friend class CallableLibrary;

private:
    // Note: `data_type` is the argument of the expression,
    //   not the result type. The result type is always uint64.
    const Type *_data_type;
    TypeIDExpr() noexcept = default;

protected:
    void _mark(Usage) const noexcept override {}
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    explicit TypeIDExpr(const Type *type) noexcept
        : Expression{Tag::TYPE_ID, Type::of<ulong>()}, _data_type{type} {}
    [[nodiscard]] auto data_type() const noexcept { return _data_type; }
    LUISA_EXPRESSION_COMMON()
};

class LC_AST_API StringIDExpr final : public Expression {
    friend class CallableLibrary;

private:
    luisa::string _data;
    StringIDExpr() noexcept = default;

protected:
    void _mark(Usage) const noexcept override {}
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    explicit StringIDExpr(luisa::string data) noexcept
        : Expression{Tag::STRING_ID, Type::of<ulong>()}, _data{std::move(data)} {}
    [[nodiscard]] auto data() const noexcept { return luisa::string_view{_data}; }
    LUISA_EXPRESSION_COMMON()
};

class LC_AST_API FuncRefExpr final : public Expression {
    detail::FunctionBuilder const *_func;
    FuncRefExpr() = default;

protected:
    void _mark(Usage) const noexcept override {}
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;
public:
    [[nodiscard]] auto func() const noexcept { return _func; }
    FuncRefExpr(detail::FunctionBuilder const *func) noexcept : Expression(Tag::FUNC_REF, Type::of<uint64_t>()), _func{func} {}
    LUISA_EXPRESSION_COMMON()
};

class CpuCustomOpExpr final : public Expression {

public:
    using Func = void (*)(void *userdata, void *arg);
    using Dtor = void (*)(void *userdata);
    CpuCustomOpExpr(const Type *type, Func Func, Dtor dtor, void *user_data, const Expression *arg) noexcept
        : Expression{Tag::CPUCUSTOM, type}, _callback{Func}, _dtor(dtor), _arg(arg), _user_data{user_data} {}
    [[nodiscard]] auto user_data() const noexcept { return _user_data; }
    LUISA_EXPRESSION_COMMON()

private:
    Func _callback;
    Dtor _dtor;
    const Expression *_arg;
    void *_user_data;

protected:
    // TODO
    void _mark(Usage usage) const noexcept override {}
    [[nodiscard]] uint64_t _compute_hash() const noexcept override { return 0; }

public:
    [[nodiscard]] Func func() const noexcept { return _callback; }
    [[nodiscard]] Dtor dtor() const noexcept { return _dtor; }
    [[nodiscard]] auto arg() const noexcept { return _arg; }
};

class GpuCustomOpExpr final : public Expression {

public:
    GpuCustomOpExpr(const Type *type, luisa::string source, const Expression *arg) noexcept
        : Expression{Tag::GPUCUSTOM, type}, _source{std::move(source)}, _arg(arg) {}
    [[nodiscard]] auto source() const noexcept { return _source; }
    LUISA_EXPRESSION_COMMON()

private:
    luisa::string _source;
    const Expression *_arg;

protected:
    // TODO
    void _mark(Usage usage) const noexcept override {}
    [[nodiscard]] uint64_t _compute_hash() const noexcept override { return 0; }

public:
    [[nodiscard]] auto arg() const noexcept { return _arg; }
};

#undef LUISA_EXPRESSION_COMMON

// helper function for easy traversal over the ASTs
template<typename Enter, typename Exit>
void traverse_subexpressions(const Expression *expr,
                             const Enter &enter,
                             const Exit &exit) noexcept {
    enter(expr);
    switch (expr->tag()) {
        case Expression::Tag::UNARY: {
            auto unary_expr = static_cast<const UnaryExpr *>(expr);
            traverse_subexpressions(unary_expr->operand(), enter, exit);
            break;
        }
        case Expression::Tag::BINARY: {
            auto binary_expr = static_cast<const BinaryExpr *>(expr);
            traverse_subexpressions(binary_expr->lhs(), enter, exit);
            traverse_subexpressions(binary_expr->rhs(), enter, exit);
            break;
        }
        case Expression::Tag::MEMBER: {
            auto member_expr = static_cast<const MemberExpr *>(expr);
            traverse_subexpressions(member_expr->self(), enter, exit);
            break;
        }
        case Expression::Tag::ACCESS: {
            auto access_expr = static_cast<const AccessExpr *>(expr);
            traverse_subexpressions(access_expr->range(), enter, exit);
            traverse_subexpressions(access_expr->index(), enter, exit);
            break;
        }
        case Expression::Tag::LITERAL:
        case Expression::Tag::REF:
        case Expression::Tag::CONSTANT: break;
        case Expression::Tag::CALL: {
            auto call_expr = static_cast<const CallExpr *>(expr);
            for (auto arg : call_expr->arguments()) {
                traverse_subexpressions(arg, enter, exit);
            }
            break;
        }
        case Expression::Tag::CAST: {
            auto cast_expr = static_cast<const CastExpr *>(expr);
            traverse_subexpressions(cast_expr->expression(), enter, exit);
            break;
        }
        case Expression::Tag::FUNC_REF:
        case Expression::Tag::TYPE_ID:
        case Expression::Tag::STRING_ID:
        case Expression::Tag::CPUCUSTOM:
        case Expression::Tag::GPUCUSTOM: break;
    }
    exit(expr);
}

}// namespace luisa::compute

namespace eastl {
template<>
struct variant_size<luisa::compute::detail::LiteralValue>
    : variant_size<luisa::compute::detail::LiteralValueVariant> {};
}// namespace eastl
