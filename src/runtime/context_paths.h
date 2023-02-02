#pragma once

#include <core/stl/filesystem.h>

namespace luisa::compute {

class Context;

class LC_RUNTIME_API ContextPaths {
    friend class Context;
    void *_impl;
    explicit ContextPaths(void *impl) noexcept : _impl{impl} {}
    ContextPaths(ContextPaths const &) noexcept = default;
    ContextPaths(ContextPaths &&) noexcept = default;

public:
    [[nodiscard]] const std::filesystem::path &runtime_directory() const noexcept;
    [[nodiscard]] const std::filesystem::path &cache_directory() const noexcept;
    [[nodiscard]] const std::filesystem::path &data_directory() const noexcept;
    ~ContextPaths() noexcept = default;
};

}// namespace luisa::compute
