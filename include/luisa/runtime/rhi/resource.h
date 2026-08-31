#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/hash.h>
#include <luisa/runtime/rhi/pixel.h>

namespace luisa::compute {

class DeviceInterface;
class Device;

namespace detail {
class ShaderInvokeBase;
}// namespace detail

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
        BufferCreationInfo info{
            .element_stride = 0,
            .total_size_bytes = 0};
        info.handle = invalid_resource_handle;
        info.native_handle = nullptr;
        return info;
    }
};

struct SwapchainOption {
    uint64_t display;
    uint64_t window;
    uint2 size;
    bool wants_hdr = false;
    bool wants_vsync = true;
    bool wants_transparent = false;
    uint back_buffer_count = 2;
};

struct SwapchainCreationInfo : public ResourceCreationInfo {
    PixelStorage storage;
};

struct ShaderCreationInfo : public ResourceCreationInfo {
    uint3 block_size;

    [[nodiscard]] static auto make_invalid() noexcept {
        ShaderCreationInfo info{};
        info.invalidate();
        return info;
    }
};

struct TileShaderCreationInfo : public ShaderCreationInfo {
    uint2 dispatch_size_xy;

    [[nodiscard]] static TileShaderCreationInfo make_invalid() noexcept {
        TileShaderCreationInfo info{};
        info.invalidate();
        return info;
    }
};

struct SparseTextureCreationInfo : public ResourceCreationInfo {
    size_t tile_size_bytes;
    uint3 tile_size;

    [[nodiscard]] static auto make_invalid() noexcept {
        SparseTextureCreationInfo info{};
        info.invalidate();
        return info;
    }
};

struct SparseBufferCreationInfo : public BufferCreationInfo {
    size_t tile_size_bytes;

    [[nodiscard]] static auto make_invalid() noexcept {
        SparseBufferCreationInfo info{};
        info.invalidate();
        return info;
    }
};

struct AccelOption {

    enum struct UsageHint : uint32_t {
        FAST_TRACE,// build with best quality
        FAST_BUILD // optimize for frequent rebuild
    };

    enum struct MotionMode : uint8_t {
        MATRIX,
        SRT,
    };

    struct Motion {
        uint keyframe_count{0};         // <= 1 means no motion blur, otherwise the number of keyframes in [time_start, time_end]
        float time_start{0.f};          // the start time of the motion blur effect
        float time_end{1.f};            // the end time of the motion blur effect
        bool should_vanish_start{false};// whether the object should vanish before time_start
        bool should_vanish_end{false};  // whether the object should vanish after time_end

        using Mode = MotionMode;
        Mode mode{};// only valid for motion blur geometry

        [[nodiscard]] constexpr auto is_enabled() const noexcept { return keyframe_count > 1; }
        [[nodiscard]] constexpr explicit operator bool() const noexcept { return is_enabled(); }
    };

    UsageHint hint{UsageHint::FAST_TRACE};
    bool allow_compaction{false};
    bool allow_update{false};

    // motion blur
    Motion motion;
};

using AccelUsageHint = AccelOption::UsageHint;
using AccelMotionOption = AccelOption::Motion;
using AccelMotionMode = AccelMotionOption::Mode;

/// \brief Options for shader creation.
struct ShaderOption {
    /// \brief Whether to enable shader cache.
    /// \details LuisaCompute uses shader cache to avoid redundant shader
    ///   compilation. Cache read/write behaviors are controlled by the
    ///   `read_shader_cache` and `write_shader_cache` methods in `BinaryIO`
    ///   passed via `class DeviceConfig` to backends on device creation.
    ///   This field has no effects if a user-defined `name` is provided.
    /// \sa DeviceConfig
    /// \sa BinaryIO
    bool enable_cache{true};
    /// \brief Whether to enable fast math.
    bool enable_fast_math{true};
    /// \brief Whether to enable debug info.
    bool enable_debug_info{false};
    /// \brief Whether to create the shader object.
    /// \details No shader object will be created if this field is set to
    ///   `true`. This field is useful for AOT compilation.
    bool compile_only{false};
    /// @brief The maximum number of registers used by the shader.
    /// \details If set to a positive value, the shader will be compiled with
    ///   the specified number of registers. This field has no effect on CPU
    ///   backend.
    uint32_t max_registers{0};
    /// \brief Whether to measure time spent on each compilation phase.
    bool time_trace{false};
    /// \brief Whether to enable extended acceleration structure limits.
    /// \details If set to true, the shader will be compiled with support for
    ///   massive instance counts (>2^24) in acceleration structures. Only has
    ///   effect on the Metal backend; other backends ignore this option.
    bool enable_extended_accel_limits{false};
    /// \brief Whether to enable the XIR scalarizer in the SPIR-V optimization
    ///   pipeline.
    /// \details The scalarizer decomposes vector operations into scalar
    ///   components. It is disabled by default: measurements on the Vulkan
    ///   backend show that keeping vectors intact preserves GVN/CSE
    ///   effectiveness and produces ~10% smaller SPIR-V modules. Only has
    ///   effect on backends compiling through the XIR-to-SPIR-V pipeline.
    ///   The `LUISA_XIR_ENABLE_SCALARIZER` environment variable, when set,
    ///   overrides this field.
    bool enable_scalarizer{false};
    /// \brief Whether the native driver may run its full optimization
    ///   pipeline while creating the shader.
    /// \details Disabling this option provides a bounded-compilation escape
    ///   hatch for unusually large shaders whose driver optimizer would
    ///   otherwise consume excessive time or memory. It can reduce execution
    ///   performance and currently only affects Vulkan compute pipelines.
    bool enable_driver_optimization{true};
    /// \brief A user-defined name for the shader.
    /// \details If provided, the shader will be read from or written to disk
    ///   via the `BinaryIO` object (passed to backends on device creation)
    ///   through the `read_shader_bytecode` and `write_shader_bytecode` methods.
    ///   The `enable_cache` field will be ignored if this field is not empty.
    /// \sa DeviceConfig
    /// \sa BinaryIO
    luisa::string name;
    /// \brief Include code in the backend's native shader representation.
    /// \details If provided, the backend incorporates this string into the generated
    ///   shader module. The accepted representation is backend-specific (for example,
    ///   source code for source-generating backends and LLVM IR/bitcode for the direct
    ///   LLVM HIP backend). This field is useful for interoperation with external callables.
    /// \sa ExternalCallable
    luisa::string native_include;
};

struct TileShaderOption : ShaderOption {
    bool use_cooperative : 1 {false};
    // Tensor-op fast path (see TileToKernelConfig::use_tensor): forwarded to
    // tile_to_kernel on the non-native-tile fallback.  Default false — only
    // the CUDA backend implements the TENSOR_* ops.
    bool use_tensor : 1 {false};
    // Dynamic batching: when enabled (min != 1 || max != 1), each thread
    // group computes `block_size().z` batch items at once — one per z-thread —
    // and the z axis of the dispatch carries the runtime batch count
    // (batch_count must lie in [min_batching_size, max_batching_size]).
    //   * min_batching_size / max_batching_size are >= 1 with min <= max;
    //   * (1, 1) disables batching and adds zero lowering overhead;
    //   * when enabled the z block size is chosen by the lowering heuristic as
    //     clamp(ceil(target_threads / threads), 1,
    //           min(min_batching_size, 64, max(1, 1024 / threads))), so
    //     B_z <= min_batching_size and B_z <= 64 always hold.
    uint32_t min_batching_size{1};
    uint32_t max_batching_size{1};
};

class LUISA_RUNTIME_API Resource {

    friend class Device;
    friend class detail::ShaderInvokeBase;

public:
    enum struct Tag : uint32_t {
        BUFFER,
        TEXTURE,
        BINDLESS_ARRAY,
        MESH,
        CURVE,
        PROCEDURAL_PRIMITIVE,
        MOTION_INSTANCE,
        ACCEL,
        STREAM,
        EVENT,
        SHADER,
        RASTER_SHADER,
        SWAP_CHAIN,
        DEPTH_BUFFER,
        DSTORAGE_FILE,
        DSTORAGE_PINNED_MEMORY,
        SPARSE_BUFFER,
        SPARSE_TEXTURE,
        SPARSE_BUFFER_HEAP,
        SPARSE_TEXTURE_HEAP,
        TENSOR_GRAPH
    };

private:
    luisa::shared_ptr<DeviceInterface> _device{nullptr};
    ResourceCreationInfo _info{};
    Tag _tag{};
    uint64_t _uid{};

private:
    [[noreturn]] static void _error_invalid() noexcept;

protected:
    static void _check_same_derived_types(const Resource &lhs,
                                          const Resource &rhs) noexcept;

    // helper method for derived classes to implement move assignment
    template<typename Derived>
    void _move_from(Derived &&rhs) noexcept {
        if (this != &rhs) [[likely]] {
            // check if the two resources are compatible if both are valid
            _check_same_derived_types(*this, rhs);
            using Self = std::remove_cvref_t<Derived>;
            static_assert(std::is_base_of_v<Resource, Self> &&
                              !std::is_same_v<Resource, Self>,
                          "Resource::_move_from can only be used in derived classes");
            auto self = static_cast<Self *>(this);
            // destroy the old resource
            std::destroy_at(self);
            std::construct_at(self, std::move(rhs));
        }
    }

    void _check_is_valid() const noexcept {
#ifndef NDEBUG
        if (!*this) [[unlikely]] { _error_invalid(); }
#endif
    }

protected:
    // protected constructors for derived classes
    Resource() noexcept { _info.invalidate(); }
    Resource(DeviceInterface *device, Tag tag, const ResourceCreationInfo &info) noexcept;
    Resource(Resource &&) noexcept;
    // protected destructor for derived classes
    // give out the ownership of the resource without destroying it
    [[nodiscard]] ResourceCreationInfo release() noexcept;
public:
    virtual ~Resource() noexcept;
    Resource(const Resource &) noexcept = delete;
    Resource &operator=(Resource &&) noexcept = delete;// use _move_from in derived classes
    Resource &operator=(const Resource &) noexcept = delete;
    [[nodiscard]] auto device() const noexcept { return _device.get(); }
    [[nodiscard]] auto handle() const noexcept { return _info.handle; }
    [[nodiscard]] auto native_handle() const noexcept { return _info.native_handle; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] auto uid() const noexcept { return _uid; }
    [[nodiscard]] auto valid() const noexcept { return _info.valid(); }
    [[nodiscard]] explicit operator bool() const noexcept { return valid(); }
    void set_name(luisa::string_view name) const noexcept;
};

}// namespace luisa::compute

namespace luisa {

template<>
struct hash<compute::ShaderOption> {
    using is_avalanching = void;
    [[nodiscard]] auto operator()(const compute::ShaderOption &option,
                                  uint64_t seed = hash64_default_seed) const noexcept {
        constexpr auto enable_cache_shift = 0u;
        constexpr auto enable_fast_math_shift = 1u;
        constexpr auto enable_debug_info_shift = 2u;
        constexpr auto compile_only_shift = 3u;
        constexpr auto enable_extended_accel_limits_shift = 4u;
        constexpr auto enable_driver_optimization_shift = 5u;
        constexpr auto enable_scalarizer_shift = 6u;
        auto opt_hash = hash_value((static_cast<uint>(option.enable_cache) << enable_cache_shift) |
                                       (static_cast<uint>(option.enable_fast_math) << enable_fast_math_shift) |
                                       (static_cast<uint>(option.enable_debug_info) << enable_debug_info_shift) |
                                       (static_cast<uint>(option.compile_only) << compile_only_shift) |
                                       (static_cast<uint>(option.enable_extended_accel_limits) << enable_extended_accel_limits_shift) |
                                       (static_cast<uint>(option.enable_driver_optimization) << enable_driver_optimization_shift) |
                                       (static_cast<uint>(option.enable_scalarizer) << enable_scalarizer_shift),
                                   seed);
        auto name_hash = hash_value(option.name, seed);
        auto native_include_hash = hash_value(option.native_include, seed);
        auto max_registers_hash = hash_value(option.max_registers, seed);
        return hash_combine(
            {opt_hash, name_hash, native_include_hash, max_registers_hash},
            seed);
    }
};

template<>
struct hash<compute::AccelOption> {
    using is_avalanching = void;
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
