#pragma once
#include <core/stl/filesystem.h>
namespace luisa::compute {
class Context;
class LC_RUNTIME_API ContextPaths {
    friend class Context;
    void *_impl;
    ContextPaths(void *impl) : _impl{impl} {}
    ContextPaths(ContextPaths const &) = default;
    ContextPaths(ContextPaths &&) = default;

public:
    [[nodiscard]] const std::filesystem::path &runtime_directory() const noexcept;
    [[nodiscard]] const std::filesystem::path &cache_directory() const noexcept;
    [[nodiscard]] const std::filesystem::path &data_directory() const noexcept;
    ~ContextPaths() = default;
};
}// namespace luisa::compute