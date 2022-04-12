//
// Created by Mike Smith on 2021/2/2.
//

#pragma once

#include <vector>
#include <filesystem>
#include <unordered_map>

#include <core/dynamic_module.h>

namespace luisa::compute {

class Device;

class LC_RUNTIME_API Context {

private:
    struct Impl;
    luisa::shared_ptr<Impl> _impl;

public:
    explicit Context(const std::filesystem::path &program) noexcept;
    Context(Context &&) noexcept = default;
    Context(const Context &) noexcept = default;
    Context &operator=(Context &&) noexcept = default;
    Context &operator=(const Context &) noexcept = default;
    ~Context() noexcept;
    [[nodiscard]] const std::filesystem::path &runtime_directory() const noexcept;
    [[nodiscard]] const std::filesystem::path &cache_directory() const noexcept;
    [[nodiscard]] Device create_device(luisa::string_view backend_name, luisa::string_view property_json = "{}") noexcept;
    [[nodiscard]] luisa::span<const luisa::string> installed_backends() const noexcept;
    [[nodiscard]] Device create_default_device() noexcept;
};

}// namespace luisa::compute
