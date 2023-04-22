#pragma once

#include <core/stl/filesystem.h>
#include <core/stl/unordered_map.h>
namespace luisa::compute {

class Context;

class LC_RUNTIME_API ContextPaths {
    friend class Context;
    void *_impl;
    mutable luisa::unordered_map<luisa::string, std::filesystem::path> _cached_paths;
    mutable std::mutex _path_mtx;

    explicit ContextPaths(void *impl) noexcept : _impl{impl} {}
    ContextPaths(ContextPaths const &) noexcept = default;
    ContextPaths(ContextPaths &&) noexcept = default;

public:
    [[nodiscard]] const std::filesystem::path &runtime_directory() const noexcept;
    [[nodiscard]] std::filesystem::path get_local_dir(luisa::string_view folder_name) const noexcept;
    ~ContextPaths() noexcept = default;
};

}// namespace luisa::compute
