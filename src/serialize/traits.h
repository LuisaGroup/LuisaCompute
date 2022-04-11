#pragma once

#include <concepts>
#include <string_view>
#include <string>
#include <core/stl.h>

namespace luisa::compute{

class Serializer;
class Deserializer;

template<typename T>
concept has_member_function_serialize = requires(T t){
    { t.serialize(std::declval<Serializer&>()) } -> std::same_as<void>;
    { t.serialize(std::declval<Deserializer&>()) } -> std::same_as<void>;
};

template<typename T>
concept has_member_function_save_load = requires(T t){
    { t.save(std::declval<Serializer&>()) } -> std::same_as<void>;
    { t.load(std::declval<Deserializer&>()) } -> std::same_as<void>;
};

template<typename T>
concept is_string = std::convertible_to<T, std::string_view>;

template<typename T>
concept can_directly_serialize = std::integral<std::remove_cvref_t<T>> ||
                                 is_string<std::remove_cvref_t<T>> ||
                                 std::floating_point<std::remove_cvref_t<T>> ||
                                 std::is_enum_v<std::remove_cvref_t<T>>;

template<typename... T>
concept all_serializable = (can_directly_serialize<T> && ...);


// template<typename Variant>
// struct variant_serializable_impl : std::false_type {};

// template<typename... T>
// struct variant_serializable_impl<std::variant<T...>>
//     : std::bool_constant<all_serializable<T...>> {};

// template<typename T>
// concept variant_serializable = variant_serializable_impl<T>::value;

template<typename T>
concept has_serialization_function = 
    (has_member_function_save_load<T> != 
    has_member_function_serialize<T>);

template<typename T>
struct is_variant : std::false_type {};

template<typename... T>
struct is_variant<luisa::variant<T...>> : std::true_type {};

template<typename T>
constexpr auto is_variant_v = is_variant<T>::value;

template<typename T>
concept enable_polymorphic_serialization = requires(T t) {
    typename T::is_polymorphically_serialized;
    typename T::polymorphic_tag_type;
    { T::create(std::declval<typename T::polymorphic_tag_type>()) } -> std::same_as<luisa::unique_ptr<T>>;
};

};