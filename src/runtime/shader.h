//
// Created by Mike Smith on 2021/7/4.
//

#pragma once

#include <core/basic_types.h>
#include <ast/function_builder.h>
#include <runtime/resource.h>
#include <runtime/device.h>
#include <runtime/bindless_array.h>
#include <runtime/custom_struct.h>
#include <runtime/cmd_encoder.h>
namespace luisa::compute {

class Accel;
class BindlessArray;

namespace detail {

template<typename T>
struct prototype_to_shader_invocation {
    using type = const T &;
};

template<typename T>
struct prototype_to_shader_invocation<Buffer<T>> {
    using type = BufferView<T>;
};

template<typename T>
struct prototype_to_shader_invocation<Image<T>> {
    using type = ImageView<T>;
};

template<typename T>
struct prototype_to_shader_invocation<Volume<T>> {
    using type = VolumeView<T>;
};

template<typename T>
using prototype_to_shader_invocation_t = typename prototype_to_shader_invocation<T>::type;

class LC_RUNTIME_API ShaderInvokeBase {

private:
    luisa::unique_ptr<ComputeDispatchCmdEncoder> _command;
    Function _kernel;

public:
    explicit ShaderInvokeBase(uint64_t handle, Function kernel) noexcept
        : _command{luisa::make_unique<ComputeDispatchCmdEncoder>(handle, kernel)},
          _kernel{kernel} {}
    ShaderInvokeBase(ShaderInvokeBase &&) noexcept = default;
    ShaderInvokeBase(const ShaderInvokeBase &) noexcept = delete;
    ShaderInvokeBase &operator=(ShaderInvokeBase &&) noexcept = default;
    ShaderInvokeBase &operator=(const ShaderInvokeBase &) noexcept = delete;
    template<typename T>
    ShaderInvokeBase &operator<<(BufferView<T> buffer) noexcept {
        _command->encode_buffer(buffer.handle(), buffer.offset_bytes(), buffer.size_bytes());
        return *this;
    }

    template<typename T>
    ShaderInvokeBase &operator<<(ImageView<T> image) noexcept {
        _command->encode_texture(image.handle(), image.level());
        return *this;
    }

    template<typename T>
    ShaderInvokeBase &operator<<(VolumeView<T> volume) noexcept {
        _command->encode_texture(volume.handle(), volume.level());
        return *this;
    }

    template<typename T>
    ShaderInvokeBase &operator<<(const Buffer<T> &buffer) noexcept {
        return *this << buffer.view();
    }

    template<typename T>
    ShaderInvokeBase &operator<<(const Image<T> &image) noexcept {
        return *this << image.view();
    }

    template<typename T>
    ShaderInvokeBase &operator<<(const Volume<T> &volume) noexcept {
        return *this << volume.view();
    }

    template<typename T>
    ShaderInvokeBase &operator<<(T data) noexcept {
        _command->encode_uniform(&data, sizeof(T));
        return *this;
    }

    // see definition in rtx/accel.cpp
    ShaderInvokeBase &operator<<(const Accel &accel) noexcept;

    // see definition in runtime/bindless_array.cpp
    ShaderInvokeBase &operator<<(const BindlessArray &array) noexcept;

protected:
    [[nodiscard]] auto _parallelize(uint3 dispatch_size) &&noexcept {
        _command->set_dispatch_size(dispatch_size);
        return std::move(_command);
    }
    [[nodiscard]] auto _parallelize(Buffer<DispatchArgs1D> const &indirect_buffer) &&noexcept {
        _command->set_dispatch_size(IndirectDispatchArg{indirect_buffer.handle()});
        return std::move(_command);
    }
    [[nodiscard]] auto _parallelize(Buffer<DispatchArgs2D> const &indirect_buffer) &&noexcept {
        _command->set_dispatch_size(IndirectDispatchArg{indirect_buffer.handle()});
        return std::move(_command);
    }
    [[nodiscard]] auto _parallelize(Buffer<DispatchArgs3D> const &indirect_buffer) &&noexcept {
        _command->set_dispatch_size(IndirectDispatchArg{indirect_buffer.handle()});
        return std::move(_command);
    }
};

template<size_t dim>
struct ShaderInvoke {
    static_assert(always_false_v<std::index_sequence<dim>>);
};

template<>
struct ShaderInvoke<1> : public ShaderInvokeBase {
    explicit ShaderInvoke(uint64_t handle, Function kernel) noexcept : ShaderInvokeBase{handle, kernel} {}
    [[nodiscard]] auto dispatch(uint size_x) &&noexcept {
        return std::move(*std::move(*this)._parallelize(uint3{size_x, 1u, 1u})).build();
    }
    [[nodiscard]] auto dispatch(Buffer<DispatchArgs1D> const &indirect_buffer) &&noexcept {
        return std::move(*std::move(*this)._parallelize(indirect_buffer)).build();
    }
};

template<>
struct ShaderInvoke<2> : public ShaderInvokeBase {
    explicit ShaderInvoke(uint64_t handle, Function kernel) noexcept : ShaderInvokeBase{handle, kernel} {}
    [[nodiscard]] auto dispatch(uint size_x, uint size_y) &&noexcept {
        return std::move(*std::move(*this)._parallelize(uint3{size_x, size_y, 1u})).build();
    }
    [[nodiscard]] auto dispatch(uint2 size) &&noexcept {
        return std::move(*this).dispatch(size.x, size.y);
    }
    [[nodiscard]] auto dispatch(Buffer<DispatchArgs2D> const &indirect_buffer) &&noexcept {
        return std::move(*std::move(*this)._parallelize(indirect_buffer)).build();
    }
};

template<>
struct ShaderInvoke<3> : public ShaderInvokeBase {
    explicit ShaderInvoke(uint64_t handle, Function kernel) noexcept : ShaderInvokeBase{handle, kernel} {}
    [[nodiscard]] auto dispatch(uint size_x, uint size_y, uint size_z) &&noexcept {
        return std::move(*std::move(*this)._parallelize(uint3{size_x, size_y, size_z})).build();
    }
    [[nodiscard]] auto dispatch(Buffer<DispatchArgs3D> const &indirect_buffer) &&noexcept {
        return std::move(*std::move(*this)._parallelize(indirect_buffer)).build();
    }
    [[nodiscard]] auto dispatch(uint3 size) &&noexcept {
        std::move(*this).dispatch(size.x, size.y, size.z);
    }
};

}// namespace detail

template<size_t dimension, typename... Args>
class Shader final : public Resource {

    static_assert(dimension == 1u || dimension == 2u || dimension == 3u);

private:
    friend class Device;
    luisa::shared_ptr<const detail::FunctionBuilder> _kernel;

private:
    // JIT shader
    Shader(DeviceInterface *device,
           luisa::shared_ptr<const detail::FunctionBuilder> kernel,
           luisa::string_view name,
           bool enable_debug_info,
           bool enable_fast_math) noexcept
        : Resource{device, Tag::SHADER, device->create_shader(kernel->function(), DeviceInterface::ShaderOption{.enable_cache = true, .enable_debug_info = enable_debug_info, .enable_fast_math = enable_fast_math, .name = name})},
          _kernel{std::move(kernel)} {}
    Shader(DeviceInterface *device,
           luisa::shared_ptr<const detail::FunctionBuilder> kernel,
           bool enable_cache,
           bool enable_debug_info,
           bool enable_fast_math) noexcept
        : Resource{device, Tag::SHADER, device->create_shader(kernel->function(), DeviceInterface::ShaderOption{.enable_cache = enable_cache, .enable_debug_info = enable_debug_info, .enable_fast_math = enable_fast_math})},
          _kernel{std::move(kernel)} {}

private:
    // AOT shader
    Shader(DeviceInterface *device,
           string_view file_path) noexcept
        : Resource{[device, &file_path]() noexcept {
              auto handle = device->load_shader(file_path, luisa::span<Type const *const>{{Type::of<Args>()...}});
              return handle == Resource::invalid_handle ?
                         Resource{} :
                         Resource{device, Tag::SHADER, handle};
          }()} {}

public:
    Shader() noexcept = default;
    using Resource::operator bool;
    [[nodiscard]] auto operator()(detail::prototype_to_shader_invocation_t<Args>... args) const noexcept {
        using invoke_type = detail::ShaderInvoke<dimension>;
        invoke_type invoke{handle(), _kernel->function()};
        return static_cast<invoke_type &&>((invoke << ... << args));
    }
    [[nodiscard]] uint3 block_size() const noexcept {
        return device()->shader_block_size(handle());
    }
};

template<typename... Args>
using Shader1D = Shader<1, Args...>;

template<typename... Args>
using Shader2D = Shader<2, Args...>;

template<typename... Args>
using Shader3D = Shader<3, Args...>;

}// namespace luisa::compute
