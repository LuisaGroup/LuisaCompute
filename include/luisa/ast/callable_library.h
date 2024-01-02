#pragma once

#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/ast/external_function.h>
#include <luisa/ast/function.h>
#include <luisa/ast/function_builder.h>

namespace luisa::compute {

template<typename T>
class Callable;

template<size_t dim, typename... T>
class Kernel;

class LC_AST_API CallableLibrary {
public:
    using CallableMap = luisa::unordered_map<luisa::string, luisa::shared_ptr<const detail::FunctionBuilder>>;

private:
    struct DeserPackage {
        detail::FunctionBuilder *builder;
        luisa::unordered_map<uint64_t, luisa::shared_ptr<detail::FunctionBuilder>> callable_map;
    };
    CallableMap _callables;
    static void serialize_func_builder(detail::FunctionBuilder const &builder, luisa::vector<std::byte> &vec) noexcept;
    static void deserialize_func_builder(detail::FunctionBuilder &builder, std::byte const *&ptr, DeserPackage &pack) noexcept;
    template<typename T>
    static void ser_value(T const &t, luisa::vector<std::byte> &vec) noexcept;
    template<typename T>
    static T deser_value(std::byte const *&ptr, DeserPackage &pack) noexcept;
    template<typename T>
    static void deser_ptr(T obj, std::byte const *&ptr, DeserPackage &pack) noexcept;

public:
    [[nodiscard]] bool empty() const { return _callables.empty(); }
    [[nodiscard]] auto const &callable_map() const { return _callables; }
    template<typename T>
    [[nodiscard]] Callable<T> get_callable(luisa::string_view name) const noexcept;
    template<size_t dim, typename... T>
    [[nodiscard]] Kernel<dim, T...> get_kernel(luisa::string_view name) const noexcept;
    [[nodiscard]] Function get_function(luisa::string_view name) const noexcept;
    [[nodiscard]] luisa::shared_ptr<const detail::FunctionBuilder> get_function_builder(luisa::string_view name) const noexcept;
    CallableLibrary() noexcept;
    void add_callable(luisa::string_view name, luisa::shared_ptr<const detail::FunctionBuilder> callable) noexcept;
    void load(luisa::span<const std::byte> binary) noexcept;
    [[nodiscard]] luisa::vector<std::byte> serialize() const noexcept;
    CallableLibrary(CallableLibrary const &) = delete;
    CallableLibrary(CallableLibrary &&) noexcept;
    ~CallableLibrary() noexcept;
};

}// namespace luisa::compute
