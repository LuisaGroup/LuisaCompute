//
// Created by Mike Smith on 2021/2/2.
//

#include <core/logging.h>
#include <core/platform.h>
#include <runtime/context.h>
#include <runtime/device.h>

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
    luisa::vector<luisa::string> installed_backends;
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
#ifdef LUISA_PLATFORM_WINDOWS
    SetDllDirectoryW(_impl->runtime_directory.c_str());
#endif
    LUISA_INFO("Created context for program '{}'.", program.filename().string<char>());
    LUISA_INFO("Runtime directory: {}.", _impl->runtime_directory.string<char>());
    _impl->cache_directory = _impl->runtime_directory / ".cache";
    LUISA_INFO("Cache directory: {}.", _impl->cache_directory.string<char>());
    if (!std::filesystem::exists(_impl->cache_directory)) {
        LUISA_INFO("Created cache directory.");
        std::filesystem::create_directories(_impl->cache_directory);
    }
    DynamicModule::add_search_path(_impl->runtime_directory);
    for (auto &&p : std::filesystem::directory_iterator{_impl->runtime_directory}) {
        if (auto path = p.path();
            p.is_regular_file() &&
            (path.extension() == ".so" ||
             path.extension() == ".dll" ||
             path.extension() == ".dylib")) {
            using namespace std::string_view_literals;
            constexpr std::array possible_prefixes{
                "luisa-compute-backend-"sv,
                "libluisa-compute-backend-"sv};
            auto filename = path.stem().string();
            for (auto prefix : possible_prefixes) {
                if (filename.starts_with(prefix)) {
                    auto name = filename.substr(prefix.size());
                    for (auto &c : name) { c = static_cast<char>(std::tolower(c)); }
                    LUISA_VERBOSE_WITH_LOCATION("Found backend: {}.", name);
                    _impl->installed_backends.emplace_back(std::move(name));
                    break;
                }
            }
        }
    }
    std::sort(_impl->installed_backends.begin(), _impl->installed_backends.end());
    _impl->installed_backends.erase(
        std::unique(_impl->installed_backends.begin(), _impl->installed_backends.end()),
        _impl->installed_backends.end());
}

const std::filesystem::path &Context::runtime_directory() const noexcept {
    return _impl->runtime_directory;
}

const std::filesystem::path &Context::cache_directory() const noexcept {
    return _impl->cache_directory;
}

Device Context::create_device(std::string_view backend_name_in, luisa::string_view property_json) noexcept {
    luisa::string backend_name{backend_name_in};
    for (auto &c : backend_name) { c = static_cast<char>(std::tolower(c)); }
    if (std::find(_impl->installed_backends.cbegin(),
                  _impl->installed_backends.cend(),
                  backend_name) == _impl->installed_backends.cend()) {
        LUISA_ERROR_WITH_LOCATION("Backend '{}' is not installed.", backend_name);
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
            _impl->runtime_directory,
            fmt::format("luisa-compute-backend-{}", backend_name));
        auto c = m.function<Device::Creator>("create");
        auto d = m.function<Device::Deleter>("destroy");
        _impl->device_identifiers.emplace_back(backend_name);
        _impl->device_creators.emplace_back(c);
        _impl->device_deleters.emplace_back(d);
        return std::make_pair(c, d);
    }();
    return Device{Device::Handle{create(*this, property_json), destroy}};
}

Context::~Context() noexcept {
    if (_impl != nullptr) {
        DynamicModule::remove_search_path(
            _impl->runtime_directory);
    }
}

luisa::span<const luisa::string> Context::installed_backends() const noexcept {
    return _impl->installed_backends;
}

Device Context::create_default_device() noexcept {
    LUISA_ASSERT(!installed_backends().empty(), "No backends installed.");
    return create_device(installed_backends().front());
}

}// namespace luisa::compute
