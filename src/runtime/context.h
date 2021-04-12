//
// Created by Mike Smith on 2021/2/2.
//

#pragma once

#include <vector>
#include <filesystem>
#include <unordered_map>

#include <core/dynamic_module.h>
#include <runtime/device.h>

namespace luisa::compute {

class Context {

private:
    std::filesystem::path _runtime_directory;
    std::filesystem::path _cache_directory;
    std::vector<DynamicModule> _loaded_modules;
    std::vector<std::string> _device_identifiers;
    std::vector<DeviceCreator *> _device_creators;
    std::vector<DeviceDeleter *> _device_deleters;

public:
    explicit Context(const std::filesystem::path &program) noexcept;
    [[nodiscard]] const std::filesystem::path &runtime_directory() const noexcept;
    [[nodiscard]] const std::filesystem::path &cache_directory() const noexcept;
    [[nodiscard]] DeviceHandle create_device(std::string_view backend_name, uint32_t index = 0u) noexcept;
};

}// namespace luisa::compute
