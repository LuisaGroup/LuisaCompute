#pragma once

#include <core/platform.h>
#include <core/concepts.h>

namespace luisa {

class DynamicModule : concepts::Noncopyable {

private:
    void *_handle;

public:
    explicit DynamicModule(const std::filesystem::path &path) noexcept;
    DynamicModule(DynamicModule &&another) noexcept;
    DynamicModule &operator=(DynamicModule &&rhs) noexcept;
    ~DynamicModule() noexcept;

    template<concepts::function F>
    [[nodiscard]] auto function(std::string_view name) const noexcept {
        return reinterpret_cast<std::add_pointer_t<F>>(
            dynamic_module_find_symbol(_handle, name));
    }
};

}// namespace luisa
