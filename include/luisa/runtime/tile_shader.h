#pragma once

#include <luisa/core/basic_types.h>
#include <luisa/core/logging.h>
#include <luisa/ast/function_builder.h>
#include <luisa/ast/tile_function_builder.h>
#include <luisa/ast/tile_to_kernel.h>
#include <luisa/runtime/shader.h>

namespace luisa::compute {

namespace detail {

// TileShaderInvoke is the dispatch handle returned by TileShader::operator().
// The tile grid is baked into TileShaderCreationInfo::dispatch_size_xy at
// compile time; the user only supplies the runtime batch count (z axis):
//   stream << tile_shader(buf_a, buf_b, buf_c).dispatch(batch_count);
class TileShaderInvoke final : public ShaderInvokeBase {

private:
    uint2 _dispatch_size_xy;

public:
    TileShaderInvoke(uint64_t handle,
                     size_t arg_count,
                     size_t uniform_size,
                     uint2 dispatch_size_xy) noexcept
        : ShaderInvokeBase{handle, arg_count, uniform_size},
          _dispatch_size_xy{dispatch_size_xy} {}
    TileShaderInvoke(TileShaderInvoke &&) noexcept = default;
    TileShaderInvoke(const TileShaderInvoke &) noexcept = delete;
    TileShaderInvoke &operator=(TileShaderInvoke &&) noexcept = default;
    TileShaderInvoke &operator=(const TileShaderInvoke &) noexcept = delete;

    [[nodiscard]] auto dispatch(uint size_z) && noexcept {
        return this->_parallelize(
                       uint3{_dispatch_size_xy.x, _dispatch_size_xy.y, size_z})
            .build();
    }
};

}// namespace detail

/// TileShader is the device-side handle of a compiled tile kernel
/// (TileFunctionBuilder). It mirrors Shader<N, Args...>:
///   auto tile_kernel = tile::jit(elementwise_add).compile();
///   auto tile_shader = device.compile_tile(tile_kernel);
///   stream << tile_shader(buf_a, buf_b, buf_c).dispatch(1u) << synchronize();
///
/// When the backend reports support_tile_compiling() the TileShader owns a
/// native tile shader created through DeviceInterface::create_tile_shader and
/// destroys it with destroy_tile_shader. Otherwise the traced tile function is
/// lowered to a regular Luisa kernel with tile_to_kernel and managed exactly
/// like a regular Shader (create_shader/destroy_shader); the compile-time tile
/// grid (dispatch_size_xy) is carried by TileShaderCreationInfo and the user
/// supplied z is the runtime batch count:
///   dispatch(dispatch_size_xy.x, dispatch_size_xy.y, size_z)
template<typename... Args>
class TileShader final : public Resource {

    friend class Device;

private:
    struct Init {
        TileShaderCreationInfo info;
        bool is_tile_shader;
    };

    TileShaderCreationInfo _info;
    bool _is_tile_shader{};

    [[nodiscard]] static Init _initialize(
        DeviceInterface *device,
        luisa::shared_ptr<const detail::TileFunctionBuilder> const &tile_kernel,
        const TileShaderOption &option) noexcept {
        if (device->support_tile_compiling()) {
            auto info = device->create_tile_shader(option, tile_kernel.get());
#ifdef LUISA_ENABLE_SAFE_MODE
            if (!info.valid() && !option.compile_only) {
                LUISA_ERROR("Failed to create tile shader.");
            }
#endif
            return {info, true};
        }
        // Backend without native tile support: lower the traced tile function
        // to a regular Luisa kernel and manage it like a normal shader.
        TileToKernelConfig config{
            .use_cooperative = option.use_cooperative,
            .min_batching_size = option.min_batching_size,
            .max_batching_size = option.max_batching_size};
        auto lowered = tile_to_kernel(tile_kernel, config);
        auto info = device->create_shader(option, Function{lowered.function.get()});
#ifdef LUISA_ENABLE_SAFE_MODE
        if (!info.valid() && !option.compile_only) {
            LUISA_ERROR("Failed to create shader.");
        }
#endif
        TileShaderCreationInfo tile_info{};
        tile_info.handle = info.handle;
        tile_info.native_handle = info.native_handle;
        tile_info.block_size = info.block_size;
        tile_info.dispatch_size_xy = lowered.dispatch_size;
        return {tile_info, false};
    }

    // base constructor
    TileShader(DeviceInterface *device, Init init) noexcept
        : Resource{device, Tag::SHADER, init.info},
          _info{init.info},
          _is_tile_shader{init.is_tile_shader} {}

public:
    // JIT tile shader
    TileShader(DeviceInterface *device,
               luisa::shared_ptr<const detail::TileFunctionBuilder> const &tile_kernel,
               const TileShaderOption &option = {}) noexcept
        : TileShader{device, _initialize(device, tile_kernel, option)} {}

    TileShader() noexcept = default;
    ~TileShader() noexcept override {
        if (*this) [[likely]] {
            if (_is_tile_shader) {
                device()->destroy_tile_shader(handle());
            } else {
                device()->destroy_shader(handle());
            }
        }
    }
    TileShader(TileShader &&) noexcept = default;
    TileShader(TileShader const &) noexcept = delete;
    TileShader &operator=(TileShader &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    TileShader &operator=(TileShader const &) noexcept = delete;
    using Resource::operator bool;

    [[nodiscard]] auto block_size() const noexcept {
        _check_is_valid();
        return _info.block_size;
    }

    [[nodiscard]] auto dispatch_size_xy() const noexcept {
        _check_is_valid();
        return _info.dispatch_size_xy;
    }

    /// True when the TileShader wraps a native backend tile shader
    /// (support_tile_compiling); false when it falls back to a regular kernel.
    [[nodiscard]] auto is_native_tile_shader() const noexcept {
        _check_is_valid();
        return _is_tile_shader;
    }

    [[nodiscard]] auto uniform_size() const noexcept {
        _check_is_valid();
        return 0u;
    }

    TileShaderCreationInfo release() noexcept {
        auto info = _info;
        static_cast<void>(Resource::release());
        _info.invalidate();
        return info;
    }

    [[nodiscard]] static constexpr uint arg_count() noexcept {
        return (0u + ... + detail::shader_argument_encode_count<Args>::value);
    }

    [[nodiscard]] auto operator()(
        detail::prototype_to_shader_invocation_t<Args>... args) const noexcept {
        _check_is_valid();
        detail::TileShaderInvoke invoke{
            handle(), arg_count(), 0u, _info.dispatch_size_xy};
        static_cast<void>((invoke << ... << args));
        return invoke;
    }
};

}// namespace luisa::compute
