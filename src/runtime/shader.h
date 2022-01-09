//
// Created by Mike Smith on 2021/7/4.
//

#pragma once

#include <core/basic_types.h>
#include <ast/function_builder.h>
#include <runtime/resource.h>
#include <runtime/bindless_array.h>

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

class ShaderInvokeBase {

private:
    Command *_command;
    Function _kernel;
    size_t _argument_index{0u};

private:
    [[nodiscard]] auto _dispatch_command() noexcept {
        return static_cast<ShaderDispatchCommand *>(_command);
    }
    void _encode_pending_bindings() noexcept {
        auto bindings = _kernel.builder()->argument_bindings();
        for (; _argument_index < _kernel.arguments().size() &&
               !luisa::holds_alternative<luisa::monostate>(// has binding
                   bindings[_argument_index]);
             _argument_index++) {
            luisa::visit(
                [&, arg = _kernel.arguments()[_argument_index]]<typename T>(T binding) noexcept {
                    if constexpr (std::is_same_v<T, FunctionBuilder::BufferBinding>) {
                        _dispatch_command()->encode_buffer(
                            arg.uid(), binding.handle, binding.offset_bytes,
                            _kernel.variable_usage(arg.uid()));
                    } else if constexpr (std::is_same_v<T, FunctionBuilder::TextureBinding>) {
                        _dispatch_command()->encode_texture(
                            arg.uid(), binding.handle, binding.level,
                            _kernel.variable_usage(arg.uid()));
                    } else if constexpr (std::is_same_v<T, FunctionBuilder::BindlessArrayBinding>) {
                        _dispatch_command()->encode_bindless_array(arg.uid(), binding.handle);
                    } else if constexpr (std::is_same_v<T, FunctionBuilder::AccelBinding>) {
                        _dispatch_command()->encode_accel(arg.uid(), binding.handle);
                    } else {
                        LUISA_ERROR_WITH_LOCATION("Invalid argument binding type.");
                    }
                },
                bindings[_argument_index]);
        }
    }

public:
    explicit ShaderInvokeBase(uint64_t handle, Function kernel) noexcept
        : _command{ShaderDispatchCommand::create(handle, kernel)},
          _kernel{kernel} {}
    ShaderInvokeBase(ShaderInvokeBase &&) noexcept = default;
    ShaderInvokeBase(const ShaderInvokeBase &) noexcept = delete;
    ShaderInvokeBase &operator=(ShaderInvokeBase &&) noexcept = default;
    ShaderInvokeBase &operator=(const ShaderInvokeBase &) noexcept = delete;

    template<typename T>
    ShaderInvokeBase &operator<<(BufferView<T> buffer) noexcept {
        _encode_pending_bindings();
        if (auto t = _kernel.arguments()[_argument_index].type();
            !t->is_buffer()) {
            LUISA_ERROR_WITH_LOCATION(
                "Expected {} but got buffer for argument {}.",
                t->description(), _argument_index);
        }
        auto variable_uid = _kernel.arguments()[_argument_index++].uid();
        auto usage = _kernel.variable_usage(variable_uid);
        _dispatch_command()->encode_buffer(
            variable_uid, buffer.handle(), buffer.offset_bytes(), usage);
        return *this;
    }

    template<typename T>
    ShaderInvokeBase &operator<<(ImageView<T> image) noexcept {
        _encode_pending_bindings();
        if (auto t = _kernel.arguments()[_argument_index].type();
            !t->is_texture()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Expected {} but got image for argument {}.",
                t->description(), _argument_index);
        }
        auto variable_uid = _kernel.arguments()[_argument_index++].uid();
        auto usage = _kernel.variable_usage(variable_uid);
        _dispatch_command()->encode_texture(variable_uid, image.handle(), image.level(), usage);
        return *this << image.offset();
    }

    template<typename T>
    ShaderInvokeBase &operator<<(VolumeView<T> volume) noexcept {
        _encode_pending_bindings();
        if (auto t = _kernel.arguments()[_argument_index].type();
            !t->is_texture()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Expected {} but got volume for argument {}.",
                t->description(), _argument_index);
        }
        auto variable_uid = _kernel.arguments()[_argument_index++].uid();
        auto usage = _kernel.variable_usage(variable_uid);
        _dispatch_command()->encode_texture(variable_uid, volume.handle(), volume.level(), usage);
        return *this << volume.offset();
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
        _encode_pending_bindings();
        if (auto t = _kernel.arguments()[_argument_index].type();
            (!t->is_basic() && !t->is_structure()) ||
            t->size() != sizeof(T)) {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid uniform (size = {}) at index {}, "
                "expected {} (size = {}).",
                sizeof(T), _argument_index,
                t->description(), t->size());
        }
        auto variable_uid = _kernel.arguments()[_argument_index++].uid();
        _dispatch_command()->encode_uniform(variable_uid, &data, sizeof(T), alignof(T));
        return *this;
    }

    // see definition in rtx/accel.cpp
    ShaderInvokeBase &operator<<(const Accel &accel) noexcept;

    // see definition in runtime/bindless_array.cpp
    ShaderInvokeBase &operator<<(const BindlessArray &array) noexcept;

protected:
    [[nodiscard]] auto _parallelize(uint3 dispatch_size) &&noexcept {
        _encode_pending_bindings();
        _dispatch_command()->set_dispatch_size(dispatch_size);
        Command *command{nullptr};
        std::swap(command, _command);
        return command;
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
        return std::move(*this)._parallelize(uint3{size_x, 1u, 1u});
    }
};

template<>
struct ShaderInvoke<2> : public ShaderInvokeBase {
    explicit ShaderInvoke(uint64_t handle, Function kernel) noexcept : ShaderInvokeBase{handle, kernel} {}
    [[nodiscard]] auto dispatch(uint size_x, uint size_y) &&noexcept {
        return std::move(*this)._parallelize(uint3{size_x, size_y, 1u});
    }
    [[nodiscard]] auto dispatch(uint2 size) &&noexcept {
        return std::move(*this).dispatch(size.x, size.y);
    }
};

template<>
struct ShaderInvoke<3> : public ShaderInvokeBase {
    explicit ShaderInvoke(uint64_t handle, Function kernel) noexcept : ShaderInvokeBase{handle, kernel} {}
    [[nodiscard]] auto dispatch(uint size_x, uint size_y, uint size_z) &&noexcept {
        return std::move(*this)._parallelize(uint3{size_x, size_y, size_z});
    }
    [[nodiscard]] auto dispatch(uint3 size) &&noexcept {
        return std::move(*this).dispatch(size.x, size.y, size.z);
    }
};

}// namespace detail

template<size_t dimension, typename... Args>
class Shader final : public Resource {

    static_assert(dimension == 1u || dimension == 2u || dimension == 3u);

private:
    luisa::shared_ptr<const detail::FunctionBuilder> _kernel;

private:
    friend class Device;
    Shader(Device::Interface *device, luisa::shared_ptr<const detail::FunctionBuilder> kernel, std::string_view meta_options) noexcept
        : Resource{
              device,
              Tag::SHADER,
              device->create_shader(kernel->function(), meta_options)},
          _kernel{std::move(kernel)} {}

public:
    Shader() noexcept = default;
    using Resource::operator bool;
    [[nodiscard]] auto operator()(detail::prototype_to_shader_invocation_t<Args>... args) const noexcept {
        using invoke_type = detail::ShaderInvoke<dimension>;
        invoke_type invoke{handle(), _kernel->function()};
        return static_cast<invoke_type &&>((invoke << ... << args));
    }
};

template<typename... Args>
using Shader1D = Shader<1, Args...>;

template<typename... Args>
using Shader2D = Shader<2, Args...>;

template<typename... Args>
using Shader3D = Shader<3, Args...>;

}// namespace luisa::compute
