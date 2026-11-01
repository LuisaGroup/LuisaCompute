#pragma once

#include <luisa/dsl/ref.h>
#include <luisa/dsl/arg.h>

// X11 macro
#ifdef Bool
#undef Bool
#endif

namespace luisa::compute {

namespace detail {

template<typename T>
void apply_default_initializer(const Ref<T> &var) noexcept {
    if constexpr (luisa::is_basic_v<T>) {
        dsl::assign(var, T{});
    } else {
        auto impl = [&var]<size_t... i>(std::index_sequence<i...>) noexcept {
            (apply_default_initializer(var.template get<i>()), ...);
        };
        impl(std::make_index_sequence<std::tuple_size_v<struct_member_tuple_t<T>>>{});
    }
}

}// namespace detail

/**
 * @brief Represents a variable in the embedded DSL.
 *
 * Var<T> is the core type for representing data on the device within the DSL.
 * It inherits from Ref<T>, providing reference semantics.
 * 
 * Logic: When a Var<T> is constructed, it records a local variable declaration
 * into the current FunctionBuilder's AST. Subsequent operations on the Var
 * are recorded as expressions targeting this local variable.
 *
 * @tparam T The C++ type being represented in the DSL.
 */
template<typename T>
struct Var : public detail::Ref<T> {

private:
    struct LocalCreationTag {};
    explicit Var(LocalCreationTag) noexcept
        : detail::Ref<T>{static_cast<const Expression *>(
              detail::FunctionBuilder::current()->local(Type::of<T>()))} {}

public:
    static_assert(std::is_trivially_destructible_v<T>);

    /**
     * @brief Construct Var from an existing AST expression.
     * @param expr Pointer to the AST expression.
     */
    explicit Var(const Expression *expr) noexcept
        : detail::Ref<T>{expr} {}

    /**
     * @brief Construct a new local variable.
     * 
     * Logic: This allocates a new local variable slot in the current AST FunctionBuilder
     * and applies default initialization.
     */
    Var() noexcept : Var{LocalCreationTag{}} {
        detail::apply_default_initializer(*this);
    }

    void set_name(luisa::string_view name) const noexcept {
        detail::FunctionBuilder::current()->set_variable_name(
            static_cast<RefExpr const *>(this->expression())->variable().uid(),
            name);
    }

    /**
     * @brief Construct and initialize from a tuple of arguments.
     */
    template<typename... Args, size_t... i>
    Var(std::tuple<Args...> args, std::index_sequence<i...>) noexcept : Var{LocalCreationTag{}} {
        (dsl::assign(this->template get<i>(), std::get<i>(args)), ...);
    }

    /**
     * @brief Construct and initialize from a tuple.
     */
    template<typename... Args>
    Var(std::tuple<Args...> args) noexcept
        : Var{std::move(args), std::index_sequence_for<Args...>{}} {}

    /**
     * @brief Construct and initialize from a single argument.
     * @tparam Arg Type of the initializer.
     * 
     * Logic: If T is a structure and Arg is not, it tries to initialize the first member.
     * If Arg is another DSL expression, it records an assignment.
     */
    template<typename Arg>
        requires concepts::different<std::remove_cvref_t<Arg>, Var> &&
                 std::negation_v<std::is_pointer<std::remove_cvref_t<Arg>>>
    Var(Arg &&arg) noexcept : Var{LocalCreationTag{}} {
        using member_tuple = struct_member_tuple_t<T>;
        if constexpr (std::tuple_size_v<member_tuple> > 1u ||
                      std::is_same_v<expr_value_t<Arg>, T>) {
            dsl::assign(*this, std::forward<Arg>(arg));
        } else {
            dsl::assign(this->template get<0u>(), std::forward<Arg>(arg));
        }
    }

    /**
     * @brief Construct and initialize from multiple arguments (list-init).
     */
    template<typename First, typename Second, typename... Other>
    Var(First &&first, Second &&second, Other &&...other) noexcept
        : Var{std::make_tuple(
              Expr{std::forward<First>(first)},
              Expr{std::forward<Second>(second)},
              Expr{std::forward<Other>(other)}...)} {}

    // Internal use: create as function argument
    explicit Var(detail::ArgumentCreation) noexcept
        : detail::Ref<T>{static_cast<const Expression *>(
              detail::FunctionBuilder::current()->argument(Type::of<T>()))} {}

    // Internal use: create as reference function argument
    explicit Var(detail::ReferenceArgumentCreation) noexcept
        : detail::Ref<T>{static_cast<const Expression *>(
              detail::FunctionBuilder::current()->reference(Type::of<T>()))} {}

    Var(Var &&) noexcept = default;

    /**
     * @brief Copy constructor (DSL assignment).
     * 
     * Logic: This records an assignment from `another` to the newly created local variable.
     */
    Var(const Var &another) noexcept : Var{LocalCreationTag{}} {
        detail::FunctionBuilder::current()->assign(this->expression(), another.expression());
    }

    const Var &operator=(Var &&rhs) & noexcept {
        detail::Ref<T>::operator=(std::move(rhs));
        return *this;
    }
    const Var &operator=(const Var &rhs) & noexcept {
        detail::Ref<T>::operator=(rhs);
        return *this;
    }
};

template<typename T>
Var(T &&) -> Var<expr_value_t<T>>;

template<typename... T>
Var(std::tuple<T...>) -> Var<std::tuple<expr_value_t<T>...>>;

template<typename T, size_t N>
using ArrayVar = Var<std::array<expr_value_t<T>, N>>;

using Int = Var<int>;
using Int2 = Var<int2>;
using Int3 = Var<int3>;
using Int4 = Var<int4>;
using UInt = Var<uint>;
using UInt2 = Var<uint2>;
using UInt3 = Var<uint3>;
using UInt4 = Var<uint4>;
using Float = Var<float>;
using Float2 = Var<float2>;
using Float3 = Var<float3>;
using Float4 = Var<float4>;
using Double2 = Var<double2>;
using Double3 = Var<double3>;
using Double4 = Var<double4>;
using Bool = Var<bool>;
using Bool2 = Var<bool2>;
using Bool3 = Var<bool3>;
using Bool4 = Var<bool4>;
using Float2x2 = Var<float2x2>;
using Float3x3 = Var<float3x3>;
using Float4x4 = Var<float4x4>;
using Double2x2 = Var<double2x2>;
using Double3x3 = Var<double3x3>;
using Double4x4 = Var<double4x4>;
using Half2x2 = Var<half2x2>;
using Half3x3 = Var<half3x3>;
using Half4x4 = Var<half4x4>;
using Short = Var<short>;
using Short2 = Var<short2>;
using Short3 = Var<short3>;
using Short4 = Var<short4>;
using UShort = Var<ushort>;
using UShort2 = Var<ushort2>;
using UShort3 = Var<ushort3>;
using UShort4 = Var<ushort4>;
using SLong = Var<slong>;
using SLong2 = Var<slong2>;
using SLong3 = Var<slong3>;
using SLong4 = Var<slong4>;
using ULong = Var<ulong>;
using ULong2 = Var<ulong2>;
using ULong3 = Var<ulong3>;
using ULong4 = Var<ulong4>;
using Half = Var<half>;
using Half2 = Var<half2>;
using Half3 = Var<half3>;
using Half4 = Var<half4>;
using BFloat16 = Var<bfloat16>;
using BFloat162 = Var<bfloat162>;
using BFloat163 = Var<bfloat163>;
using BFloat164 = Var<bfloat164>;

template<size_t N>
using ArrayInt = ArrayVar<int, N>;
template<size_t N>
using ArrayInt2 = ArrayVar<int2, N>;
template<size_t N>
using ArrayInt3 = ArrayVar<int3, N>;
template<size_t N>
using ArrayInt4 = ArrayVar<int4, N>;
template<size_t N>
using ArrayUInt = ArrayVar<uint, N>;
template<size_t N>
using ArrayUInt2 = ArrayVar<uint2, N>;
template<size_t N>
using ArrayUInt3 = ArrayVar<uint3, N>;
template<size_t N>
using ArrayUInt4 = ArrayVar<uint4, N>;
template<size_t N>
using ArrayFloat = ArrayVar<float, N>;
template<size_t N>
using ArrayFloat2 = ArrayVar<float2, N>;
template<size_t N>
using ArrayFloat3 = ArrayVar<float3, N>;
template<size_t N>
using ArrayFloat4 = ArrayVar<float4, N>;
template<size_t N>
using ArrayBool = ArrayVar<bool, N>;
template<size_t N>
using ArrayBool2 = ArrayVar<bool2, N>;
template<size_t N>
using ArrayBool3 = ArrayVar<bool3, N>;
template<size_t N>
using ArrayBool4 = ArrayVar<bool4, N>;
template<size_t N>
using ArrayFloat2x2 = ArrayVar<float2x2, N>;
template<size_t N>
using ArrayFloat3x3 = ArrayVar<float3x3, N>;
template<size_t N>
using ArrayFloat4x4 = ArrayVar<float4x4, N>;
template<size_t N>
using ArrayShort = ArrayVar<short, N>;
template<size_t N>
using ArrayShort2 = ArrayVar<short2, N>;
template<size_t N>
using ArrayShort3 = ArrayVar<short3, N>;
template<size_t N>
using ArrayShort4 = ArrayVar<short4, N>;
template<size_t N>
using ArrayUShort = ArrayVar<ushort, N>;
template<size_t N>
using ArrayUShort2 = ArrayVar<ushort2, N>;
template<size_t N>
using ArrayUShort3 = ArrayVar<ushort3, N>;
template<size_t N>
using ArrayUShort4 = ArrayVar<ushort4, N>;
template<size_t N>
using ArraySLong = ArrayVar<slong, N>;
template<size_t N>
using ArraySLong2 = ArrayVar<slong2, N>;
template<size_t N>
using ArraySLong3 = ArrayVar<slong3, N>;
template<size_t N>
using ArraySLong4 = ArrayVar<slong4, N>;
template<size_t N>
using ArrayULong = ArrayVar<ulong, N>;
template<size_t N>
using ArrayULong2 = ArrayVar<ulong2, N>;
template<size_t N>
using ArrayULong3 = ArrayVar<ulong3, N>;
template<size_t N>
using ArrayULong4 = ArrayVar<ulong4, N>;
template<size_t N>
using ArrayHalf = ArrayVar<half, N>;
template<size_t N>
using ArrayHalf2 = ArrayVar<half2, N>;
template<size_t N>
using ArrayHalf3 = ArrayVar<half3, N>;
template<size_t N>
using ArrayHalf4 = ArrayVar<half4, N>;

namespace detail {

template<typename>
struct is_var_impl : std::false_type {};

template<typename T>
struct is_var_impl<Var<T>> : std::true_type {};

// Now Var<ulong> is defined and we can use it here.
template<typename T>
Var<ulong> RefEnableGetAddress<T>::address() const noexcept {
    auto self = static_cast<const T *>(this);
    return def<ulong>(FunctionBuilder::current()->call(
        Type::of<ulong>(), CallOp::ADDRESS_OF, {self->expression()}));
}

}// namespace detail

template<typename T>
struct is_var : detail::is_var_impl<std::remove_cvref_t<T>> {};

template<typename T>
inline constexpr bool is_var_v = is_var<T>::value;

namespace dsl_literals {

#define LUISA_MAKE_DSL_INTEGRAL_LITERAL(suffix, type)                               \
    [[nodiscard]] inline auto operator"" suffix(unsigned long long v) noexcept {    \
        return def<type>(static_cast<type>(v));                                     \
    }                                                                               \
    [[nodiscard]] inline auto operator"" suffix##2(unsigned long long v) noexcept { \
        return def<type##2>(luisa::make_##type##2(static_cast<type>(v)));           \
    }                                                                               \
    [[nodiscard]] inline auto operator"" suffix##3(unsigned long long v) noexcept { \
        return def<type##3>(luisa::make_##type##3(static_cast<type>(v)));           \
    }                                                                               \
    [[nodiscard]] inline auto operator"" suffix##4(unsigned long long v) noexcept { \
        return def<type##4>(luisa::make_##type##4(static_cast<type>(v)));           \
    }

#define LUISA_MAKE_DSL_FLOATING_POINT_LITERAL(suffix, type)                  \
    [[nodiscard]] inline auto operator"" suffix(long double v) noexcept {    \
        return def<type>(static_cast<type>(v));                              \
    }                                                                        \
    [[nodiscard]] inline auto operator"" suffix##2(long double v) noexcept { \
        return def<type##2>(luisa::make_##type##2(static_cast<type>(v)));    \
    }                                                                        \
    [[nodiscard]] inline auto operator"" suffix##3(long double v) noexcept { \
        return def<type##3>(luisa::make_##type##3(static_cast<type>(v)));    \
    }                                                                        \
    [[nodiscard]] inline auto operator"" suffix##4(long double v) noexcept { \
        return def<type##4>(luisa::make_##type##4(static_cast<type>(v)));    \
    }

LUISA_MAKE_DSL_INTEGRAL_LITERAL(_short, short)
LUISA_MAKE_DSL_INTEGRAL_LITERAL(_ushort, ushort)
LUISA_MAKE_DSL_INTEGRAL_LITERAL(_int, int)
LUISA_MAKE_DSL_INTEGRAL_LITERAL(_uint, uint)
LUISA_MAKE_DSL_INTEGRAL_LITERAL(_slong, slong)
LUISA_MAKE_DSL_INTEGRAL_LITERAL(_ulong, ulong)

LUISA_MAKE_DSL_FLOATING_POINT_LITERAL(_half, half)
LUISA_MAKE_DSL_FLOATING_POINT_LITERAL(_bfloat16, bfloat16)
LUISA_MAKE_DSL_FLOATING_POINT_LITERAL(_float, float)
LUISA_MAKE_DSL_FLOATING_POINT_LITERAL(_double, double)

#undef LUISA_MAKE_DSL_INTEGRAL_LITERAL
#undef LUISA_MAKE_DSL_FLOATING_POINT_LITERAL

}// namespace dsl_literals

}// namespace luisa::compute
