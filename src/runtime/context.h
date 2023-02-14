//
// Created by Mike Smith on 2021/2/2.
//

#pragma once

#include <core/stl/memory.h>
#include <core/stl/string.h>
#include <core/stl/hash.h>

namespace luisa {
class DynamicModule;
}

namespace luisa::compute {

class Device;
class BinaryIO;
class ContextPaths;
class DeviceConfigExt {
public:
    virtual ~DeviceConfigExt() = default;
};
struct DeviceConfig {
    mutable luisa::unique_ptr<DeviceConfigExt> extension;
    size_t device_index{0ull};
    bool inqueue_buffer_limit{true};
    bool headless{false};
};

class LC_RUNTIME_API Context {

    friend class ContextPaths;

private:
    struct Impl;
    luisa::shared_ptr<Impl> _impl;
    explicit Context(luisa::shared_ptr<Impl> impl) noexcept;

public:
    // program_path can be first arg from main entry
    explicit Context(string_view program_path) noexcept;
    explicit Context(const char *program_path) noexcept
        : Context{string_view{program_path}} {}
    ~Context() noexcept;
    Context(Context &&) noexcept = default;
    Context(const Context &) noexcept = default;
    Context &operator=(Context &&) noexcept = default;
    Context &operator=(const Context &) noexcept = default;
    // relative paths
    [[nodiscard]] ContextPaths paths() const noexcept;
    // Create a virtual device
    // backend "metal", "dx", "cuda" is supported currently
    [[nodiscard]] Device create_device(luisa::string_view backend_name,
                                       const DeviceConfig *settings = nullptr) noexcept;
    // installed backends automatically detacted
    // The compiled backends' name is returned
    [[nodiscard]] luisa::span<const luisa::string> installed_backends() const noexcept;
    // loaded backends' modules
    [[nodiscard]] luisa::span<const DynamicModule> loaded_modules() const noexcept;
    // choose one backend randomly when multiple installed backends compiled
    // program panic when no installed backends compiled
    [[nodiscard]] Device create_default_device() noexcept;
    // Interface for binary input and output
    // A default BinaryIO instance is provided by backend, usually the behavior is: read and write binary data from disk
    [[nodiscard]] BinaryIO *file_io() const noexcept;
    void set_file_io(BinaryIO *file_io) noexcept;
};

}// namespace luisa::compute
