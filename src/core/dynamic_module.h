#pragma once

#include <core/platform.h>
#include <core/concepts.h>

namespace luisa {

class DynamicModule : concepts::Noncopyable {

private:
    void *_handle;

public:
    DynamicModule(const std::filesystem::path &folder, std::string_view name) noexcept;
    DynamicModule(DynamicModule &&another) noexcept;
    DynamicModule &operator=(DynamicModule &&rhs) noexcept;
    ~DynamicModule() noexcept;

    template<concepts::function F>
    [[nodiscard]] auto function(std::string_view name) const noexcept {
        return reinterpret_cast<std::add_pointer_t<F>>(
            dynamic_module_find_symbol(_handle, name));
    }

    template<concepts::function F, typename... Args>
    decltype(auto) invoke(std::string_view name, Args &&...args) const noexcept {
        return std::invoke(function<F>(name), std::forward<Args>(args)...);
    }

    template<concepts::function F, typename Tuple>
    decltype(auto) apply(std::string_view name, Tuple &&t) const noexcept {
        return std::apply(function<F>(name), std::forward<Tuple>(t));
    }
};

}// namespace luisa
