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
struct DeviceConfig;

namespace detail {
class ContextImpl;
}// namespace detail

class LC_RUNTIME_API Context {

private:
    luisa::shared_ptr<detail::ContextImpl> _impl;

public:
    explicit Context(luisa::shared_ptr<luisa::compute::detail::ContextImpl> impl) noexcept;
    // program_path can be first arg from main entry
    explicit Context(luisa::string_view program_path) noexcept;
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
    // create subdirectories under the runtime directory
    [[nodiscard]] const luisa::filesystem::path &create_runtime_subdir(luisa::string_view folder_name) const noexcept;
    // Create a virtual device
    // backend "metal", "dx", "cuda" is supported currently
    [[nodiscard]] Device create_device(
        luisa::string_view backend_name,
        const DeviceConfig *settings = nullptr,
        bool enable_validation = false) noexcept;
    // installed backends automatically detacted
    // The compiled backends' name is returned
    [[nodiscard]] luisa::span<const luisa::string> installed_backends() const noexcept;
    // choose one backend randomly when multiple installed backends compiled
    // program panic when no installed backends compiled
    [[nodiscard]] Device create_default_device() noexcept;
    [[nodiscard]] luisa::vector<luisa::string> backend_device_names(luisa::string_view backend_name) const noexcept;
};

}// namespace luisa::compute
