//
// Created by Mike Smith on 2021/2/2.
//

#pragma once

#include <vector>
#include <filesystem>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include <core/dynamic_module.h>
#include <runtime/device.h>

namespace luisa::compute {

class Context {

private:
    std::filesystem::path _runtime_directory;
    std::filesystem::path _cache_directory;
    luisa::vector<DynamicModule> _loaded_modules;
    luisa::vector<luisa::string> _device_identifiers;
    luisa::vector<Device::Creator *> _device_creators;
    luisa::vector<Device::Deleter *> _device_deleters;

public:
    explicit Context(const std::filesystem::path &program) noexcept;
    Context(Context &&) noexcept = default;
    Context(const Context &) noexcept = delete;
    Context &operator=(Context &&) noexcept = default;
    Context &operator=(const Context &) noexcept = delete;
    ~Context() noexcept;
    [[nodiscard]] const std::filesystem::path &runtime_directory() const noexcept;
    [[nodiscard]] const std::filesystem::path &cache_directory() const noexcept;
    [[nodiscard]] Device create_device(std::string_view backend_name, const nlohmann::json &properties = nlohmann::json::object()) noexcept;
};

}// namespace luisa::compute
