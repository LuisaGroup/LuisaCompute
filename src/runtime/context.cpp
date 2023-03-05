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
namespace luisa::compute {

// Make context global, so dynamic modules cannot be over-loaded
struct Context::Impl {
    std::filesystem::path runtime_directory;
    std::filesystem::path cache_directory;
    std::filesystem::path data_directory;
    luisa::vector<DynamicModule> loaded_modules;
    luisa::vector<luisa::string> device_identifiers;
    luisa::vector<Device::Creator *> device_creators;
    luisa::vector<Device::Deleter *> device_deleters;
    luisa::vector<luisa::string> installed_backends;
    BinaryIO *file_io{nullptr};
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
    luisa::string backend_name{backend_name_in};
    for (auto &c : backend_name) { c = static_cast<char>(std::tolower(c)); }
    auto impl = _impl.get();
    if (std::find(impl->installed_backends.cbegin(),
                  impl->installed_backends.cend(),
                  backend_name) == impl->installed_backends.cend()) {
        LUISA_ERROR_WITH_LOCATION("Backend '{}' is not installed.", backend_name);
    }
    LUISA_ASSERT(impl->device_identifiers.size() == impl->device_creators.size() &&
                     impl->device_identifiers.size() == impl->device_deleters.size(),
                 "Internal error.");
    Device::Creator *creator;
    Device::Deleter *deleter;
    if (auto iter = std::find(impl->device_identifiers.cbegin(),
                              impl->device_identifiers.cend(),
                              backend_name);
        iter != impl->device_identifiers.cend()) {
        auto index = std::distance(impl->device_identifiers.cbegin(), iter);
        creator = impl->device_creators[index];
        deleter = impl->device_deleters[index];
    } else {
        auto &&m = impl->loaded_modules.emplace_back(
            DynamicModule::load(
                impl->runtime_directory,
                fmt::format("lc-backend-{}", backend_name)));
        creator = m.function<Device::Creator>("create");
        deleter = m.function<Device::Deleter>("destroy");
        impl->device_identifiers.emplace_back(backend_name);
        impl->device_creators.emplace_back(creator);
        impl->device_deleters.emplace_back(deleter);
    }
    return Device{Device::Handle{creator(Context{_impl}, settings), deleter}};
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
luisa::span<const DynamicModule> Context::loaded_modules() const noexcept {
    return _impl->loaded_modules;
}

Device Context::create_default_device() noexcept {
    LUISA_ASSERT(!installed_backends().empty(), "No backends installed.");
    return create_device(installed_backends().front());
}

BinaryIO *Context::file_io() const noexcept {
    return _impl->file_io;
}

void Context::set_file_io(BinaryIO *file_io) noexcept {
    _impl->file_io = file_io;
}

}// namespace luisa::compute
