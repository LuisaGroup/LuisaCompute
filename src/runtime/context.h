//
// Created by Mike Smith on 2021/2/2.
//

#pragma once

#include <vector>
#include <filesystem>
#include <unordered_map>

#include <nlohmann/json.hpp>

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
    [[nodiscard]] Device create_device(std::string_view backend_name, const nlohmann::json &properties = nlohmann::json::object()) noexcept;
};

}// namespace luisa::compute
