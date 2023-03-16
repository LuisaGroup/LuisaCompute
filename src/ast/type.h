//
// Created by Mike Smith on 2021/2/6.
//

#pragma once

#include <core/stl/vector.h>
#include <core/stl/string.h>
#include <core/stl/functional.h>
#include <core/concepts.h>

namespace luisa::compute {

class AstSerializer;

template<typename T>
struct array_dimension {
    static constexpr size_t value = 0u;
};

template<typename T, size_t N>
struct array_dimension<T[N]> {
    static constexpr auto value = N;
};

template<typename T, size_t N>
struct array_dimension<std::array<T, N>> {
    static constexpr auto value = N;
};

template<typename T>
constexpr auto array_dimension_v = array_dimension<T>::value;

template<typename T>
struct array_element {
    using type = T;
};

template<typename T, size_t N>
struct array_element<T[N]> {
    using type = T;
};

template<typename T, size_t N>
struct array_element<std::array<T, N>> {
    using type = T;
};

template<typename T>
using array_element_t = typename array_element<T>::type;

template<typename T>
struct is_array : std::false_type {};

template<typename T, size_t N>
struct is_array<T[N]> : std::true_type {};

template<typename T, size_t N>
struct is_array<std::array<T, N>> : std::true_type {};

template<typename T>
constexpr auto is_array_v = is_array<T>::value;

template<typename T>
struct is_tuple : std::false_type {};

template<typename... T>
struct is_tuple<std::tuple<T...>> : std::true_type {};

template<typename T>
constexpr auto is_tuple_v = is_tuple<T>::value;

template<typename T>
struct is_custom_struct : std::false_type {};

template<typename T>
constexpr auto is_custom_struct_v = is_custom_struct<T>::value;

namespace detail {

template<typename T, size_t>
using array_to_tuple_element_t = T;

template<typename T, size_t... i>
[[nodiscard]] constexpr auto array_to_tuple_impl(std::index_sequence<i...>) noexcept {
    return static_cast<std::tuple<array_to_tuple_element_t<T, i>...> *>(nullptr);
}

}// namespace detail

template<typename T>
struct struct_member_tuple {
    using type = std::tuple<T>;
};

template<typename... T>
struct struct_member_tuple<std::tuple<T...>> {
    using type = std::tuple<T...>;
};

template<typename T, size_t N>
struct struct_member_tuple<std::array<T, N>> {
    using type = std::remove_pointer_t<
        decltype(detail::array_to_tuple_impl<T>(std::make_index_sequence<N>{}))>;
};

template<typename T, size_t N>
struct struct_member_tuple<T[N]> {
    using type = typename struct_member_tuple<std::array<T, N>>::type;
};

template<typename T, size_t N>
struct struct_member_tuple<Vector<T, N>> {
    using type = typename struct_member_tuple<std::array<T, N>>::type;
};

template<size_t N>
struct struct_member_tuple<Matrix<N>> {
    using type = typename struct_member_tuple<std::array<Vector<float, N>, N>>::type;
};

template<typename T>
using struct_member_tuple_t = typename struct_member_tuple<T>::type;

template<typename T>
struct canonical_layout {
    using type = typename canonical_layout<struct_member_tuple_t<T>>::type;
};

template<>
struct canonical_layout<float> {
    using type = std::tuple<float>;
};

template<>
struct canonical_layout<bool> {
    using type = std::tuple<bool>;
};

template<>
struct canonical_layout<int> {
    using type = std::tuple<int>;
};

template<>
struct canonical_layout<uint> {
    using type = std::tuple<uint>;
};

template<typename T>
struct canonical_layout<std::tuple<T>> {
    using type = typename canonical_layout<T>::type;
};

template<typename... T>
struct canonical_layout<std::tuple<T...>> {
    using type = std::tuple<typename canonical_layout<T>::type...>;
};

template<typename T>
using canonical_layout_t = typename canonical_layout<T>::type;

template<typename... T>
struct tuple_join {
    static_assert(always_false_v<T...>);
};

template<typename... A, typename... B, typename... C>
struct tuple_join<std::tuple<A...>, std::tuple<B...>, C...> {
    using type = typename tuple_join<std::tuple<A..., B...>, C...>::type;
};

template<typename... A>
struct tuple_join<std::tuple<A...>> {
    using type = std::tuple<A...>;
};

template<typename... T>
using tuple_join_t = typename tuple_join<T...>::type;

namespace detail {

template<typename L, typename T>
struct linear_layout_impl {
    using type = std::tuple<T>;
};

template<typename... L, typename... T>
struct linear_layout_impl<std::tuple<L...>, std::tuple<T...>> {
    using type = tuple_join_t<std::tuple<L...>, typename linear_layout_impl<std::tuple<>, T>::type...>;
};

}// namespace detail

template<typename T>
using linear_layout = detail::linear_layout_impl<std::tuple<>, canonical_layout_t<T>>;

template<typename T>
using linear_layout_t = typename linear_layout<T>::type;

namespace detail {

template<typename T>
struct dimension_impl {
    static constexpr auto value = dimension_impl<canonical_layout_t<T>>::value;
};

template<typename T, size_t N>
struct dimension_impl<T[N]> {
    static constexpr auto value = N;
};

template<typename T, size_t N>
struct dimension_impl<std::array<T, N>> {
    static constexpr auto value = N;
};

template<typename T, size_t N>
struct dimension_impl<Vector<T, N>> {
    static constexpr auto value = N;
};

template<size_t N>
struct dimension_impl<Matrix<N>> {
    static constexpr auto value = N;
};

template<typename... T>
struct dimension_impl<std::tuple<T...>> {
    static constexpr auto value = sizeof...(T);
};

}// namespace detail

template<typename T>
using dimension = detail::dimension_impl<std::remove_cvref_t<T>>;

template<typename T>
constexpr auto dimension_v = dimension<T>::value;

class Type;

struct TypeVisitor {
    virtual void visit(const Type *) noexcept = 0;
};

/// Type class
class LC_AST_API Type {

public:
    /// Type tags
    enum struct Tag : uint32_t {
        BOOL,
        FLOAT32,
        INT32,
        UINT32,

        VECTOR,
        MATRIX,

        ARRAY,
        STRUCTURE,

        BUFFER,
        TEXTURE,
        BINDLESS_ARRAY,
        ACCEL,

        CUSTOM
    };

public:
    static constexpr auto custom_struct_size = static_cast<size_t>(4u);
    static constexpr auto custom_struct_alignment = static_cast<size_t>(4u);

protected:
    Type() noexcept = default;
    ~Type() noexcept = default;

public:
    // disable copy & move
    Type(Type &&) noexcept = delete;
    Type(const Type &) noexcept = delete;
    Type &operator=(Type &&) noexcept = delete;
    Type &operator=(const Type &) noexcept = delete;

public:
    /// Return Type object of type T
    template<typename T>
    [[nodiscard]] static const Type *of() noexcept;
    /// Return Type object of type T
    template<typename T>
    [[nodiscard]] static auto of(T &&) noexcept { return of<std::remove_cvref_t<T>>(); }
    /// Return array type of type T
    [[nodiscard]] static const Type *array(const Type *elem, size_t n) noexcept;
    /// Return vector type of type T
    [[nodiscard]] static const Type *vector(const Type *elem, size_t n) noexcept;
    /// Return matrix type of type T
    [[nodiscard]] static const Type *matrix(size_t n) noexcept;
    /// Return buffer type of type T
    [[nodiscard]] static const Type *buffer(const Type *elem) noexcept;
    /// Return texture type of type T
    [[nodiscard]] static const Type *texture(const Type *elem, size_t dimension) noexcept;
    /// Return struct type of type T
    [[nodiscard]] static const Type *structure(luisa::span<const Type *> members) noexcept;
    /// Return struct type of type T
    [[nodiscard]] static const Type *structure(size_t alignment, luisa::span<const Type *> members) noexcept;
    /// Return struct type of type T
    [[nodiscard]] static const Type *structure(std::initializer_list<const Type *> members) noexcept;
    /// Return struct type of type T
    [[nodiscard]] static const Type *structure(size_t alignment, std::initializer_list<const Type *> members) noexcept;

    /// Return struct type of type T
    template<typename... T>
        requires std::conjunction_v<std::is_convertible<T, const Type *const>...>
    [[nodiscard]] static const Type *structure(size_t alignment, T &&...members) noexcept {
        return structure(alignment, {std::forward<T>(members)...});
    }

    /// Return struct type of type T
    template<typename... T>
        requires std::conjunction_v<std::is_convertible<T, const Type *const>...>
    [[nodiscard]] static const Type *structure(T &&...members) noexcept {
        return structure({std::forward<T>(members)...});
    }

    /// Return custom type with the specified name
    [[nodiscard]] static const Type *custom(luisa::string_view name) noexcept;

    /// Construct Type object from description
    /// @param description Type description in the following syntax: \n
    ///   TYPE := DATA | RESOURCE | CUSTOM \n
    ///   DATA := BASIC | ARRAY | VECTOR | MATRIX | STRUCT \n
    ///   BASIC := int | uint | bool | float \n
    ///   ARRAY := array\<BASIC,N\> \n
    ///   VECTOR := vector\<BASIC,VEC_MAT_DIM\> \n
    ///   MATRIX := matrix\<VEC_MAT_DIM\> | matrix\<VEC_MAT_DIM\> | matrix\<VEC_MAT_DIM\> \n
    ///   VEC_MAT_DIM := 2 | 3 | 4 \n
    ///   STRUCT := struct\<STRUCT_ALIGNMENT,DATA+\> \n
    ///   STRUCT_ALIGNMENT := 4 | 8 | 16 \n
    ///   RESOURCE := BUFFER | TEXTURE | BINDLESS_ARRAY | ACCEL \n
    ///   BUFFER := buffer\<DATA | CUSTOM\> \n
    ///   TEXTURE := texture\<TEXTURE_DIM,TEXTURE_ELEM\> \n
    ///   TEXTURE_DIM := 2 | 3 \n
    ///   TEXTURE_ELEM := float | int | uint \n
    ///   BINDLESS_ARRAY := bindless_array \n
    ///   ACCEL := accel \n
    ///   CUSTOM := [a-zA-Z_][a-zA-Z0-9_]* \n
    /// @example Type::from("array\<struct\<16,float,int,int,uint\>,233\>")
    /// @note Spaces are not allowed between tokens.
    [[nodiscard]] static const Type *from(std::string_view description) noexcept;

    /// Return type count
    [[nodiscard]] static size_t count() noexcept;

    /// Traverse TypeVisitor
    static void traverse(TypeVisitor &visitor) noexcept;
    static void traverse(const luisa::function<void(const Type *)> &visitor) noexcept;

    /// Compare by description
    [[nodiscard]] bool operator==(const Type &rhs) const noexcept;
    /// Compare by index
    /// @note The indices ensure the topological order of types (e.g., `uint` always goes before `array<uint,n>`).
    [[nodiscard]] bool operator<(const Type &rhs) const noexcept;
    [[nodiscard]] uint index() const noexcept;
    [[nodiscard]] uint64_t hash() const noexcept;
    [[nodiscard]] size_t size() const noexcept;
    [[nodiscard]] size_t alignment() const noexcept;
    [[nodiscard]] Tag tag() const noexcept;
    [[nodiscard]] luisa::string_view description() const noexcept;
    [[nodiscard]] uint dimension() const noexcept;
    [[nodiscard]] luisa::span<const Type *const> members() const noexcept;
    [[nodiscard]] const Type *element() const noexcept;

    /// Scalar = bool || float || int || uint
    [[nodiscard]] bool is_scalar() const noexcept;
    [[nodiscard]] bool is_bool() const noexcept;
    [[nodiscard]] bool is_int32() const noexcept;
    [[nodiscard]] bool is_uint32() const noexcept;
    [[nodiscard]] bool is_float32() const noexcept;
    /// Arithmetic = float || int || uint
    [[nodiscard]] bool is_arithmetic() const noexcept;

    /// Basic = scalar || vector || matrix
    [[nodiscard]] bool is_basic() const noexcept;
    [[nodiscard]] bool is_array() const noexcept;
    [[nodiscard]] bool is_vector() const noexcept;
    [[nodiscard]] bool is_bool_vector() const noexcept;
    [[nodiscard]] bool is_int32_vector() const noexcept;
    [[nodiscard]] bool is_uint32_vector() const noexcept;
    [[nodiscard]] bool is_float32_vector() const noexcept;
    [[nodiscard]] bool is_matrix() const noexcept;
    [[nodiscard]] bool is_structure() const noexcept;
    [[nodiscard]] bool is_buffer() const noexcept;
    [[nodiscard]] bool is_texture() const noexcept;
    [[nodiscard]] bool is_bindless_array() const noexcept;
    [[nodiscard]] bool is_accel() const noexcept;
    [[nodiscard]] bool is_custom() const noexcept;
};

}// namespace luisa::compute
