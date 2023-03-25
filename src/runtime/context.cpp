//
// Created by Mike Smith on 2021/2/2.
//

#include <core/dynamic_module.h>
#include <core/logging.h>
#include <core/platform.h>
#include <runtime/context.h>
#include <runtime/context_paths.h>
#include <runtime/device.h>
#include <core/binary_io.h>
#include <vstl/pdqsort.h>
#include <core/stl/filesystem.h>
#include <core/stl/unordered_map.h>

namespace luisa::compute {
struct BackendModule {
    using BackendDeviceNames = void(luisa::vector<luisa::string> &);
    DynamicModule module;
    Device::Creator *creator;
    Device::Deleter *deleter;
    BackendDeviceNames *backend_device_names;
};
// Make context global, so dynamic modules cannot be over-loaded
struct Context::Impl {
    std::filesystem::path runtime_directory;
    std::filesystem::path cache_directory;
    std::filesystem::path data_directory;
    luisa::unordered_map<luisa::string, BackendModule> loaded_backends;
    luisa::vector<luisa::string> installed_backends;
    const BackendModule &create_module(luisa::string_view backend_name_in) noexcept {
        luisa::string backend_name{backend_name_in};
        for (auto &c : backend_name) { c = static_cast<char>(std::tolower(c)); }
        if (std::find(installed_backends.cbegin(),
                      installed_backends.cend(),
                      backend_name) == installed_backends.cend()) {
            LUISA_ERROR_WITH_LOCATION("Backend '{}' is not installed.", backend_name);
        }
        auto create_new = [&]() {
            BackendModule m{
                .module = DynamicModule::load(
                    runtime_directory,
                    luisa::format("lc-backend-{}", backend_name))};
            m.creator = m.module.function<Device::Creator>("create");
            m.deleter = m.module.function<Device::Deleter>("destroy");
            m.backend_device_names = m.module.function<BackendModule::BackendDeviceNames>("backend_device_names");
            return m;
        };
        return loaded_backends
            .try_emplace(backend_name, luisa::lazy_construct(std::move(create_new)))
            .first->second;
    }
};

namespace detail {

auto runtime_directory(const std::filesystem::path &p) noexcept {
    auto cp = std::filesystem::canonical(p);
    if (std::filesystem::is_directory(cp)) { return cp; }
    return std::filesystem::canonical(cp.parent_path());
}

}// namespace detail

Context::Context(string_view program_path) noexcept
    : _impl{luisa::make_shared<Impl>()} {
    std::filesystem::path program{program_path};
    _impl->runtime_directory = detail::runtime_directory(program);
    LUISA_INFO("Created context for program '{}'.", to_string(program.filename()));
    LUISA_INFO("Runtime directory: {}.", to_string(_impl->runtime_directory));
    _impl->cache_directory = _impl->runtime_directory / ".cache";
    _impl->data_directory = _impl->runtime_directory / ".data";
    LUISA_INFO("Cache directory: {}.", to_string(_impl->cache_directory));
    LUISA_INFO("Data directory: {}.", to_string(_impl->data_directory));
    if (!std::filesystem::exists(_impl->cache_directory)) {
        LUISA_INFO("Created cache directory.");
        std::filesystem::create_directories(_impl->cache_directory);
    }
    DynamicModule::add_search_path(_impl->runtime_directory);
    for (auto &&p : std::filesystem::directory_iterator{_impl->runtime_directory}) {
        if (auto &&path = p.path();
            p.is_regular_file() &&
            (path.extension() == ".so" ||
             path.extension() == ".dll" ||
             path.extension() == ".dylib")) {
            using namespace std::string_view_literals;
            constexpr std::array possible_prefixes{
                "lc-backend-"sv,
                // Make Mingw happy
                "liblc-backend-"sv};
            auto filename = to_string(path.stem());
            for (auto prefix : possible_prefixes) {
                if (filename.starts_with(prefix)) {
                    auto name = filename.substr(prefix.size());
                    for (auto &c : name) { c = static_cast<char>(std::tolower(c)); }
                    LUISA_INFO_WITH_LOCATION("Found backend: {}.", name);
                    _impl->installed_backends.emplace_back(std::move(name));
                    break;
                }
            }
        }
    }
    pdqsort(_impl->installed_backends.begin(), _impl->installed_backends.end());
    _impl->installed_backends.erase(
        std::unique(_impl->installed_backends.begin(), _impl->installed_backends.end()),
        _impl->installed_backends.end());
}

const std::filesystem::path &ContextPaths::runtime_directory() const noexcept {
    return static_cast<const Context::Impl *>(_impl)->runtime_directory;
}

const std::filesystem::path &ContextPaths::cache_directory() const noexcept {
    return static_cast<const Context::Impl *>(_impl)->cache_directory;
}

const std::filesystem::path &ContextPaths::data_directory() const noexcept {
    return static_cast<const Context::Impl *>(_impl)->data_directory;
}

ContextPaths Context::paths() const noexcept {
    return ContextPaths{_impl.get()};
}

Device Context::create_device(std::string_view backend_name_in, const DeviceConfig *settings) noexcept {
    auto impl = _impl.get();
    auto &&m = impl->create_module(backend_name_in);
    return Device{Device::Handle{m.creator(Context{_impl}, settings), m.deleter}};
}

Context::Context(luisa::shared_ptr<Impl> impl) noexcept
    : _impl{std::move(impl)} {}

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
luisa::vector<luisa::string> Context::backend_device_names(luisa::string_view backend_name_in) const noexcept {
    auto impl = _impl.get();
    auto &&m = impl->create_module(backend_name_in);
    luisa::vector<luisa::string> names;
    m.backend_device_names(names);
    return names;
}
}// namespace luisa::compute
