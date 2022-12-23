//
// Created by Mike Smith on 2021/2/2.
//

#pragma once

#include <core/stl/memory.h>
#include <core/stl/string.h>
#include <core/hash_128.h>
namespace luisa {
class DynamicModule;
namespace compute {
class Device;
class BinaryIO;
class ContextPaths;
struct DeviceConfig {
    Hash128 hash;
    size_t device_index{0ull};
    bool inqueue_buffer_limit{true};
    bool headless{false};
};
class LC_RUNTIME_API Context {
    friend class ContextPaths;
private:
    struct Impl;
    luisa::shared_ptr<Impl> _impl;
    size_t _index = ~0ull;
    Context(
        luisa::shared_ptr<Impl> const &impl,
        size_t index);

public:
    explicit Context(string_view program_path) noexcept;
    explicit Context(const char *program_path) noexcept
        : Context(string_view{program_path}) {}
    Context(Context &&) noexcept = default;
    Context(const Context &) noexcept = default;
    Context &operator=(Context &&) noexcept = default;
    Context &operator=(const Context &) noexcept = default;
    ~Context() noexcept;
    ContextPaths paths() const noexcept;
    [[nodiscard]] auto index() const noexcept { return _index; }
    [[nodiscard]] Device create_device(luisa::string_view backend_name, DeviceConfig const *settings = nullptr) noexcept;
    [[nodiscard]] luisa::span<const luisa::string> installed_backends() const noexcept;
    [[nodiscard]] luisa::span<const DynamicModule> loaded_modules() const noexcept;
    [[nodiscard]] Device create_default_device() noexcept;
    [[nodiscard]] BinaryIO *get_fileio_visitor() const noexcept;
    void set_fileio_visitor(BinaryIO *file_io) noexcept;
};
}// namespace compute
}// namespace luisa
