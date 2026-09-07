#pragma once

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/hash.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/filesystem.h>

namespace luisa {
class DynamicModule;
class BinaryIO;
}// namespace luisa

namespace luisa::compute {

class Device;
class DeviceInterface;
struct DeviceConfig;

namespace detail {
class ContextImpl;
}// namespace detail

class LUISA_RUNTIME_API Context {

private:
    luisa::shared_ptr<detail::ContextImpl> _impl;

public:
    using StaticBackendCreator = DeviceInterface *(
        Context &&context, const DeviceConfig *config);
    using StaticBackendDeleter = void(DeviceInterface *device);
    using StaticBackendDeviceNames = void(
        luisa::vector<luisa::string> &names);

    /// Registers a backend linked into the process image. This is primarily
    /// used by signed iOS applications, where runtime-loadable backend modules
    /// are not available. Registration affects Context instances created
    /// afterwards and preserves the normal create_device() API.
    static void register_static_backend(
        luisa::string_view backend_name,
        StaticBackendCreator *creator,
        StaticBackendDeleter *deleter,
        StaticBackendDeviceNames *device_names) noexcept;

    explicit Context(luisa::shared_ptr<detail::ContextImpl> impl) noexcept;
    // program_path can be first arg from main entry
    explicit Context(luisa::string_view program_path) noexcept;
    explicit Context(luisa::string_view program_path, luisa::string_view data_dir) noexcept;
    explicit Context(const char *program_path) noexcept
        : Context{luisa::string_view{program_path}} {}
    ~Context() noexcept;
    Context(Context &&) noexcept = default;
    Context(const Context &) noexcept = default;
    Context &operator=(Context &&) noexcept = default;
    Context &operator=(const Context &) noexcept = default;
    [[nodiscard]] const auto &impl() const & noexcept { return _impl; }
    [[nodiscard]] auto impl() && noexcept { return std::move(_impl); }
    // runtime directory
    [[nodiscard]] const luisa::filesystem::path &runtime_directory() const noexcept;
    // data(cache, builtin-data, etc.) directory
    [[nodiscard]] const luisa::filesystem::path &data_directory() const noexcept;
    // create subdirectories under the runtime directory
    [[nodiscard]] const luisa::filesystem::path &create_runtime_subdir(luisa::string_view folder_name) const noexcept;
    // create subdirectories under the data directory
    [[nodiscard]] const luisa::filesystem::path &create_data_subdir(luisa::string_view folder_name) const noexcept;
    // Create a backend device
    [[nodiscard]] Device create_device(luisa::string_view backend_name,
                                       const DeviceConfig *settings,
                                       bool enable_validation) noexcept;
    // Create a beackend device with validation mode determined by environment variable `LUISA_ENABLE_VALIDATION`
    [[nodiscard]] Device create_device(luisa::string_view backend_name,
                                       const DeviceConfig *settings = nullptr) noexcept;
    // installed backends automatically detacted
    // The compiled backends' name is returned
    [[nodiscard]] luisa::span<const luisa::string> installed_backends() const noexcept;
    // choose one backend randomly when multiple installed backends compiled
    // program panic when no installed backends compiled
    [[nodiscard]] Device create_default_device() noexcept;
    [[nodiscard]] luisa::vector<luisa::string> backend_device_names(luisa::string_view backend_name) const noexcept;
    [[nodiscard]] const DynamicModule &load_backend(luisa::string_view backend_name) const noexcept;
};

}// namespace luisa::compute
