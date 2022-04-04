//
// Created by Mike Smith on 2021/2/2.
//

#include <core/logging.h>
#include <core/platform.h>
#include <runtime/context.h>
#include <runtime/device.h>
#include <nlohmann/json.hpp>

#ifdef LUISA_PLATFORM_WINDOWS
#include <windows.h>
#endif

namespace luisa::compute {

struct Context::Impl {
    std::filesystem::path runtime_directory;
    std::filesystem::path cache_directory;
    luisa::vector<DynamicModule> loaded_modules;
    luisa::vector<luisa::string> device_identifiers;
    luisa::vector<Device::Creator *> device_creators;
    luisa::vector<Device::Deleter *> device_deleters;
};

namespace detail {
[[nodiscard]] auto runtime_directory(const std::filesystem::path &p) noexcept {
    auto cp = std::filesystem::canonical(p);
    if (std::filesystem::is_directory(cp)) { return cp; }
    return std::filesystem::canonical(cp.parent_path());
}
}// namespace detail

Context::Context(const std::filesystem::path &program) noexcept
    : _impl{luisa::make_shared<Impl>()} {
    _impl->runtime_directory = detail::runtime_directory(program);
    LUISA_INFO("Created context for program: {}.", program.filename().string<char>());
    LUISA_INFO("Runtime directory: {}.", _impl->runtime_directory.string<char>());
    _impl->cache_directory = _impl->runtime_directory / ".cache";
    LUISA_INFO("Cache directory: {}.", _impl->cache_directory.string<char>());
    if (!std::filesystem::exists(_impl->cache_directory)) {
        LUISA_INFO("Created cache directory.");
        std::filesystem::create_directories(_impl->cache_directory);
    }
    DynamicModule::add_search_path(_impl->runtime_directory);
#ifdef LUISA_PLATFORM_WINDOWS
    SetDllDirectoryW((_impl->runtime_directory / "backends").wstring().c_str());
#endif
}

const std::filesystem::path &Context::runtime_directory() const noexcept {
    return _impl->runtime_directory;
}

const std::filesystem::path &Context::cache_directory() const noexcept {
    return _impl->cache_directory;
}

Device Context::create_device(std::string_view backend_name, const nlohmann::json &properties) noexcept {
    if (!properties.is_object()) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid device properties: {}.",
            properties.dump());
    }
    auto [create, destroy] = [backend_name, this] {
        if (auto iter = std::find(_impl->device_identifiers.cbegin(),
                                  _impl->device_identifiers.cend(),
                                  backend_name);
            iter != _impl->device_identifiers.cend()) {
            auto i = iter - _impl->device_identifiers.cbegin();
            auto c = _impl->device_creators[i];
            auto d = _impl->device_deleters[i];
            return std::make_pair(c, d);
        }
        auto &&m = _impl->loaded_modules.emplace_back(
            _impl->runtime_directory / "backends",
            fmt::format("luisa-compute-backend-{}", backend_name));
        auto c = m.function<Device::Creator>("create");
        auto d = m.function<Device::Deleter>("destroy");
        _impl->device_identifiers.emplace_back(backend_name);
        _impl->device_creators.emplace_back(c);
        _impl->device_deleters.emplace_back(d);
        return std::make_pair(c, d);
    }();
    return Device{Device::Handle{create(*this, properties.dump()), destroy}};
}

Context::~Context() noexcept {
    if (_impl != nullptr) {
        DynamicModule::remove_search_path(
            _impl->runtime_directory);
    }
}

}// namespace luisa::compute
