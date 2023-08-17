#pragma once
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/ast/external_function.h>
#include <luisa/ast/function_builder.h>
namespace luisa::compute {
class LC_AST_API CallableLibrary {
private:
    using CallableMap = luisa::unordered_map<luisa::string, luisa::shared_ptr<const detail::FunctionBuilder>>;
    CallableMap _callables;
    static void serialize_func_builder(detail::FunctionBuilder const &builder, luisa::vector<std::byte> &vec) noexcept;
    template <typename T>
    static void ser_value(T const& t, luisa::vector<std::byte> &vec) noexcept;
public:
    CallableLibrary() noexcept;
    void add_callable(luisa::string_view name, luisa::shared_ptr<const detail::FunctionBuilder> callable) noexcept;
    static CallableLibrary load(luisa::span<const std::byte> binary) noexcept;
    [[nodiscard]] luisa::vector<std::byte> serialize() const noexcept;
    CallableLibrary(CallableLibrary const &) = delete;
    CallableLibrary(CallableLibrary &&) noexcept;
    ~CallableLibrary() noexcept;
};
};// namespace luisa::compute