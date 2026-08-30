#pragma once

#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/functional.h>
#include <luisa/core/concepts.h>
#include <luisa/ast/attribute.h>

namespace luisa {
class MemorySanitizer;
}// namespace luisa

namespace luisa::compute {

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

template<typename T, size_t N>
struct struct_member_tuple<Matrix<T, N>> {
    using type = typename struct_member_tuple<std::array<Vector<T, N>, N>>::type;
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
struct canonical_layout<half> {
    using type = std::tuple<half>;
};

template<>
struct canonical_layout<double> {
    using type = std::tuple<double>;
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

template<>
struct canonical_layout<short> {
    using type = std::tuple<short>;
};

template<>
struct canonical_layout<ushort> {
    using type = std::tuple<ushort>;
};

template<>
struct canonical_layout<int8_t> {
    using type = std::tuple<int8_t>;
};

template<>
struct canonical_layout<uint8_t> {
    using type = std::tuple<uint8_t>;
};

template<>
struct canonical_layout<slong> {
    using type = std::tuple<slong>;
};

template<>
struct canonical_layout<ulong> {
    using type = std::tuple<ulong>;
};

template<>
struct canonical_layout<long>
    : canonical_layout<canonical_c_long> {};

template<>
struct canonical_layout<unsigned long>
    : canonical_layout<canonical_c_ulong> {};

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

template<typename T, size_t N>
struct dimension_impl<Matrix<T, N>> {
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

namespace detail {
class TypeRegistry;
}// namespace detail

struct TypeVisitor {
    virtual void visit(const Type *) noexcept = 0;
    virtual ~TypeVisitor() noexcept = default;
};
enum struct CoopRefVecType : uint32_t {
    UINT8,
    INT8,
    UINT32,
    INT32,
    FLOAT16,
    FLOAT32,
    FLOAT8_E4M3,
    FLOAT8_E5M2
};
constexpr size_t coop_ref_vec_type_size(CoopRefVecType type) {
    switch (type) {
        case CoopRefVecType::FLOAT8_E4M3: [[fallthrough]];
        case CoopRefVecType::FLOAT8_E5M2: [[fallthrough]];
        case CoopRefVecType::UINT8: [[fallthrough]];
        case CoopRefVecType::INT8:
            return 1;
        case CoopRefVecType::FLOAT16:
            return 2;
        case CoopRefVecType::FLOAT32: [[fallthrough]];
        case CoopRefVecType::INT32: [[fallthrough]];
        case CoopRefVecType::UINT32:
            return 4;
        default:
            return 0;
    }
}
/// Type class
class LUISA_AST_API Type {
    friend class ::luisa::MemorySanitizer;
    friend class detail::TypeRegistry;
    static void reset_type_registry() noexcept;

public:
    /// Type tags
    /// !!!DO NOT CHANGE THE ORDER OF THESE ENUMS!!!
    enum struct Tag : uint32_t {
        BOOL,
        INT8,
        UINT8,
        INT16,
        UINT16,
        INT32,
        UINT32,
        INT64,
        UINT64,
        FLOAT16,
        FLOAT32,
        FLOAT64,
        FLOAT8_E4M3,
        FLOAT8_E5M2,
        INT4,
        FP4_E2M1,

        VECTOR,
        MATRIX,

        ARRAY,
        STRUCTURE,

        // resource types, not valid type for IR
        BUFFER,
        TEXTURE,
        BINDLESS_ARRAY,
        ACCEL,

        COOPERATIVE_VECTOR,
        COOPERATIVE_VECTOR_REF,// should be uint32 for backend, only for metadata
        COOPERATIVE_MATRIX_REF,// should be uint32 for backend, only for metadata
        CUSTOM
    };

private:
    static_assert(static_cast<uint32_t>(Tag::INT8) ==
                  static_cast<uint32_t>(Tag::BOOL) + 1u);
    static_assert(static_cast<uint32_t>(Tag::FLOAT8_E5M2) ==
                  static_cast<uint32_t>(Tag::INT8) + 12u);
    // INT4 / FP4_E2M1 are 4-bit sub-byte scalar types (stored as 1 byte per
    // element on the host/device; the lower nibble holds the value, the upper
    // nibble is zero/unused).  They are appended after the 8-bit scalars so the
    // pre-existing scalar range invariants (is_scalar / is_arithmetic) extend
    // naturally; VECTOR follows the 4-bit group.
    static_assert(static_cast<uint32_t>(Tag::INT4) ==
                  static_cast<uint32_t>(Tag::FLOAT8_E5M2) + 1u);
    static_assert(static_cast<uint32_t>(Tag::FP4_E2M1) ==
                  static_cast<uint32_t>(Tag::INT4) + 1u);
    static_assert(static_cast<uint32_t>(Tag::VECTOR) ==
                  static_cast<uint32_t>(Tag::FP4_E2M1) + 1u);
    static_assert(static_cast<uint32_t>(Tag::FLOAT8_E5M2) ==
                  static_cast<uint32_t>(Tag::FLOAT16) + 4u);
    static_assert(static_cast<uint32_t>(Tag::ACCEL) ==
                  static_cast<uint32_t>(Tag::BUFFER) + 3u);

    // Types are interned and immutable after registry publication. Keep the
    // primary discriminator in the public base representation so exact tag
    // queries and tag-only predicates do not cross the shared-library
    // boundary merely to downcast to the private TypeImpl.
    Tag _tag{};

public:
    static constexpr auto custom_struct_size = static_cast<size_t>(4u);
    static constexpr auto custom_struct_alignment = static_cast<size_t>(4u);
    static constexpr auto coop_ref_type_size = static_cast<size_t>(CoopRefVecType::FLOAT8_E5M2) + 1;

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
    /// Return cooperative_vector type of type T
    [[nodiscard]] static const Type *cooperative_vector(const Type *elem, size_t n) noexcept;
    /// Return cooperative_vector type of type T
    [[nodiscard]] static const Type *cooperative_vector_ref(CoopRefVecType type, size_t n) noexcept;
    /// Return cooperative_vector type of type T
    [[nodiscard]] static const Type *cooperative_matrix_ref(CoopRefVecType type, size_t n, size_t m) noexcept;
    /// Return vector type of type T
    [[nodiscard]] static const Type *vector(const Type *elem, size_t n) noexcept;
    /// Return matrix type of type T
    [[nodiscard]] static const Type *matrix(size_t n) noexcept;
    /// Return buffer type of type T
    [[nodiscard]] static const Type *buffer(const Type *elem, luisa::span<const Attribute> attributes = {}) noexcept;
    /// Return texture type of type T
    [[nodiscard]] static const Type *texture(const Type *elem, size_t dimension, luisa::span<const Attribute> attributes = {}) noexcept;
    /// Return struct type of type T
    [[nodiscard]] static const Type *structure(luisa::span<Type const *const> members, luisa::span<const Attribute> attributes = {}) noexcept;
    /// Return struct type of type T
    [[nodiscard]] static const Type *structure(size_t alignment, luisa::span<Type const *const> members, luisa::span<const Attribute> attributes = {}) noexcept;
    /// Return struct type of type T
    [[nodiscard]] static const Type *structure(std::initializer_list<const Type *> members, luisa::span<const Attribute> attributes = {}) noexcept;
    /// Return struct type of type T
    [[nodiscard]] static const Type *structure(size_t alignment, std::initializer_list<const Type *> members, luisa::span<const Attribute> attributes = {}) noexcept;

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
    ///   STRUCT_ALIGNMENT := 1 | 4 | 8 | 16 \n
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
    [[nodiscard]] Tag tag() const noexcept { return _tag; }
    [[nodiscard]] luisa::string_view description() const noexcept;
    [[nodiscard]] uint dimension() const noexcept;
    [[nodiscard]] luisa::span<const Type *const> members() const noexcept;
    [[nodiscard]] luisa::span<const Attribute> member_attributes() const noexcept;
    [[nodiscard]] const Type *element() const noexcept;
    [[nodiscard]] CoopRefVecType coop_vec_ref_type() const noexcept;
    [[nodiscard]] uint2 coop_matrix_dimension() const noexcept;

    /// Scalar = bool || float || int || uint || quantized (int4/fp4)
    [[nodiscard]] bool is_scalar() const noexcept {
        return _tag <= Tag::FP4_E2M1;
    }
    [[nodiscard]] bool is_bool() const noexcept { return _tag == Tag::BOOL; }
    [[nodiscard]] bool is_int32() const noexcept { return _tag == Tag::INT32; }
    [[nodiscard]] bool is_uint32() const noexcept { return _tag == Tag::UINT32; }
    [[nodiscard]] bool is_int64() const noexcept { return _tag == Tag::INT64; }
    [[nodiscard]] bool is_uint64() const noexcept { return _tag == Tag::UINT64; }
    [[nodiscard]] bool is_float() const noexcept {
        return _tag >= Tag::FLOAT16 && _tag <= Tag::FLOAT8_E5M2;
    }
    [[nodiscard]] bool is_int() const noexcept {
        return _tag == Tag::INT8 || _tag == Tag::INT16 ||
               _tag == Tag::INT32 || _tag == Tag::INT64 ||
               _tag == Tag::INT4;
    }
    [[nodiscard]] bool is_uint() const noexcept {
        return _tag == Tag::UINT8 || _tag == Tag::UINT16 ||
               _tag == Tag::UINT32 || _tag == Tag::UINT64;
    }
    [[nodiscard]] bool is_float16() const noexcept { return _tag == Tag::FLOAT16; }
    [[nodiscard]] bool is_float32() const noexcept { return _tag == Tag::FLOAT32; }
    [[nodiscard]] bool is_float64() const noexcept { return _tag == Tag::FLOAT64; }
    [[nodiscard]] bool is_float8() const noexcept {
        return _tag == Tag::FLOAT8_E4M3 || _tag == Tag::FLOAT8_E5M2;
    }
    [[nodiscard]] bool is_float8_e4m3() const noexcept { return _tag == Tag::FLOAT8_E4M3; }
    [[nodiscard]] bool is_float8_e5m2() const noexcept { return _tag == Tag::FLOAT8_E5M2; }
    [[nodiscard]] bool is_int8() const noexcept { return _tag == Tag::INT8; }
    [[nodiscard]] bool is_uint8() const noexcept { return _tag == Tag::UINT8; }
    [[nodiscard]] bool is_int4() const noexcept { return _tag == Tag::INT4; }
    [[nodiscard]] bool is_fp4() const noexcept { return _tag == Tag::FP4_E2M1; }
    [[nodiscard]] bool is_quantized() const noexcept {
        return _tag == Tag::INT4 || _tag == Tag::FP4_E2M1;
    }
    [[nodiscard]] bool is_int16() const noexcept { return _tag == Tag::INT16; }
    [[nodiscard]] bool is_uint16() const noexcept { return _tag == Tag::UINT16; }

    [[nodiscard]] bool is_scalar_or_vector() const noexcept {
        return is_scalar() || _tag == Tag::VECTOR;
    }
    [[nodiscard]] bool is_bool_or_bool_vector() const noexcept;
    [[nodiscard]] bool is_int_or_int_vector() const noexcept;
    [[nodiscard]] bool is_uint_or_uint_vector() const noexcept;
    [[nodiscard]] bool is_float_or_float_vector() const noexcept;

    /// Arithmetic = float || int || uint
    /// Arithmetic = float || int || uint || quantized (int4/fp4)
    [[nodiscard]] bool is_arithmetic() const noexcept {
        return _tag >= Tag::INT8 && _tag <= Tag::FP4_E2M1;
    }

    /// Basic = scalar || vector || matrix
    [[nodiscard]] bool is_basic() const noexcept {
        return is_scalar() || _tag == Tag::VECTOR || _tag == Tag::MATRIX;
    }
    [[nodiscard]] bool is_cooperative_vector() const noexcept { return _tag == Tag::COOPERATIVE_VECTOR; }
    [[nodiscard]] bool is_cooperative_matrix_ref() const noexcept { return _tag == Tag::COOPERATIVE_MATRIX_REF; }
    [[nodiscard]] bool is_cooperative_vector_ref() const noexcept { return _tag == Tag::COOPERATIVE_VECTOR_REF; }
    [[nodiscard]] bool is_array() const noexcept { return _tag == Tag::ARRAY; }
    [[nodiscard]] bool is_vector() const noexcept { return _tag == Tag::VECTOR; }
    [[nodiscard]] bool is_bool_vector() const noexcept;
    [[nodiscard]] bool is_int32_vector() const noexcept;
    [[nodiscard]] bool is_uint32_vector() const noexcept;
    [[nodiscard]] bool is_float16_vector() const noexcept;
    [[nodiscard]] bool is_float32_vector() const noexcept;
    [[nodiscard]] bool is_float64_vector() const noexcept;
    [[nodiscard]] bool is_int8_vector() const noexcept;
    [[nodiscard]] bool is_uint8_vector() const noexcept;
    [[nodiscard]] bool is_int16_vector() const noexcept;
    [[nodiscard]] bool is_uint16_vector() const noexcept;
    [[nodiscard]] bool is_int64_vector() const noexcept;
    [[nodiscard]] bool is_uint64_vector() const noexcept;
    [[nodiscard]] bool is_int_vector() const noexcept;
    [[nodiscard]] bool is_uint_vector() const noexcept;
    [[nodiscard]] bool is_float_vector() const noexcept;
    [[nodiscard]] bool is_matrix() const noexcept { return _tag == Tag::MATRIX; }
    [[nodiscard]] bool is_structure() const noexcept { return _tag == Tag::STRUCTURE; }
    [[nodiscard]] bool is_buffer() const noexcept { return _tag == Tag::BUFFER; }
    [[nodiscard]] bool is_texture() const noexcept { return _tag == Tag::TEXTURE; }
    [[nodiscard]] bool is_bindless_array() const noexcept { return _tag == Tag::BINDLESS_ARRAY; }
    [[nodiscard]] bool is_accel() const noexcept { return _tag == Tag::ACCEL; }
    [[nodiscard]] bool is_custom() const noexcept { return _tag == Tag::CUSTOM; }
    [[nodiscard]] bool is_resource() const noexcept {
        return _tag >= Tag::BUFFER && _tag <= Tag::ACCEL;
    }
};

static_assert(sizeof(Type) == sizeof(Type::Tag));
static_assert(alignof(Type) == alignof(Type::Tag));

}// namespace luisa::compute
