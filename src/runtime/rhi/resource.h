//
// Created by Mike Smith on 2021/7/30.
//

#pragma once

#include <core/dll_export.h>
#include <core/stl/memory.h>
#include <core/stl/string.h>
#include <core/stl/hash.h>
#include <runtime/rhi/pixel.h>

namespace luisa::compute {

class DeviceInterface;
class Device;

constexpr auto invalid_resource_handle = ~0ull;

struct ResourceCreationInfo {
    uint64_t handle;
    void *native_handle;

    [[nodiscard]] constexpr auto valid() const noexcept { return handle != invalid_resource_handle; }

    void invalidate() noexcept {
        handle = invalid_resource_handle;
        native_handle = nullptr;
    }

    [[nodiscard]] static constexpr auto make_invalid() noexcept {
        return ResourceCreationInfo{invalid_resource_handle, nullptr};
    }
};

struct BufferCreationInfo : public ResourceCreationInfo {
    size_t element_stride;
    size_t total_size_bytes;
    [[nodiscard]] static constexpr auto make_invalid() noexcept {
        return BufferCreationInfo{invalid_resource_handle, nullptr, 0, 0};
    }
};

struct SwapChainCreationInfo : public ResourceCreationInfo {
    PixelStorage storage;
};

struct ShaderCreationInfo : public ResourceCreationInfo {
    // luisa::string name;
    uint3 block_size;
};

struct AccelOption {

    enum struct UsageHint : uint8_t {
        FAST_TRACE,// build with best quality
        FAST_BUILD // optimize for frequent rebuild
    };

    UsageHint hint{UsageHint::FAST_TRACE};
    bool allow_compaction{true};
    bool allow_update{false};
};

struct ShaderOption {
    bool enable_cache{true};
    bool enable_fast_math{true};
    bool enable_debug_info{false};
    bool compile_only{false};// compile the shader but not create a shader object
    luisa::string name;      // will generate a default name if empty
};

class LC_RUNTIME_API Resource {

    friend class Device;

public:
    enum struct Tag : uint8_t {
        BUFFER,
        TEXTURE,
        BINDLESS_ARRAY,
        MESH,
        PROCEDURAL_PRIMITIVE,
        ACCEL,
        STREAM,
        EVENT,
        SHADER,
        RASTER_SHADER,
        SWAP_CHAIN,
        DEPTH_BUFFER
    };

private:
    luisa::shared_ptr<DeviceInterface> _device{nullptr};
    ResourceCreationInfo _info{};
    Tag _tag{};

protected:
    void _destroy() noexcept;

public:
    Resource() noexcept {
        _info.invalidate();
    }
    Resource(DeviceInterface *device, Tag tag, const ResourceCreationInfo &info) noexcept;
    virtual ~Resource() noexcept {
        if (*this) { _destroy(); }
    }
    Resource(Resource &&) noexcept;
    Resource(const Resource &) noexcept = delete;
    Resource &operator=(Resource &&) noexcept;
    Resource &operator=(const Resource &) noexcept = delete;
    [[nodiscard]] auto device() const noexcept { return _device.get(); }
    [[nodiscard]] auto handle() const noexcept { return _info.handle; }
    [[nodiscard]] void *native_handle() const noexcept { return _info.native_handle; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] explicit operator bool() const noexcept { return _info.valid(); }
};

}// namespace luisa::compute

namespace luisa {

template<>
struct hash<compute::ShaderOption> {
    [[nodiscard]] auto operator()(const compute::ShaderOption &option,
                                  uint64_t seed = hash64_default_seed) const noexcept {
        constexpr auto enable_cache_shift = 0u;
        constexpr auto enable_fast_math_shift = 1u;
        constexpr auto enable_debug_info_shift = 2u;
        constexpr auto compile_only_shift = 3u;
        auto opt_hash = hash_value((static_cast<uint>(option.enable_cache) << enable_cache_shift) |
                                       (static_cast<uint>(option.enable_fast_math) << enable_fast_math_shift) |
                                       (static_cast<uint>(option.enable_debug_info) << enable_debug_info_shift) |
                                       (static_cast<uint>(option.compile_only) << compile_only_shift),
                                   seed);
        auto name_hash = hash_value(option.name, seed);
        return hash_combine({opt_hash, name_hash}, seed);
    }
};

template<>
struct hash<compute::AccelOption> {
    [[nodiscard]] auto operator()(const compute::AccelOption &option,
                                  uint64_t seed = hash64_default_seed) const noexcept {
        constexpr auto hint_shift = 0u;
        constexpr auto allow_compaction_shift = 8u;
        constexpr auto allow_update_shift = 9u;
        return hash_value((static_cast<uint>(option.hint) << hint_shift) |
                              (static_cast<uint>(option.allow_compaction) << allow_compaction_shift) |
                              (static_cast<uint>(option.allow_update) << allow_update_shift),
                          seed);
    }
};

}// namespace luisa
