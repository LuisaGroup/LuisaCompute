#pragma once

#ifdef LUISA_ENABLE_IR
#include <luisa/ir/ir2ast.h>
#endif

#include <luisa/core/basic_types.h>
#include <luisa/ast/function_builder.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/runtime/dispatch_buffer.h>
#include <luisa/runtime/rhi/command_encoder.h>

namespace luisa::compute {

class Accel;
class BindlessArray;
class IndirectDispatchBuffer;

template<typename>
class SOA;

template<typename T>
class SOAView;

namespace detail {

template<typename... Args>
[[nodiscard]] static auto shader_argument_types() noexcept {
    if constexpr (sizeof...(Args) == 0) {
        return luisa::span<const Type *const>{};
    } else {
        static const std::array args{Type::of<Args>()...};
        return luisa::span{args};
    }
}

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
struct prototype_to_shader_invocation<SOA<T>> {
    using type = SOAView<T>;
};

template<typename T>
using prototype_to_shader_invocation_t = typename prototype_to_shader_invocation<T>::type;

class LC_RUNTIME_API ShaderInvokeBase {

private:
    ComputeDispatchCmdEncoder _encoder;

public:
    explicit ShaderInvokeBase(uint64_t handle, size_t arg_count, size_t uniform_size) noexcept
        : _encoder{handle, arg_count, uniform_size} {}

    ShaderInvokeBase(ShaderInvokeBase &&) noexcept = default;
    ShaderInvokeBase(const ShaderInvokeBase &) noexcept = delete;
    ShaderInvokeBase &operator=(ShaderInvokeBase &&) noexcept = default;
    ShaderInvokeBase &operator=(const ShaderInvokeBase &) noexcept = delete;

    template<typename T>
    ShaderInvokeBase &operator<<(BufferView<T> buffer) noexcept {
        _encoder.encode_buffer(buffer.handle(), buffer.offset_bytes(), buffer.size_bytes());
        return *this;
    }

    template<typename T>
    ShaderInvokeBase &operator<<(ImageView<T> image) noexcept {
        _encoder.encode_texture(image.handle(), image.level());
        return *this;
    }

    template<typename T>
    ShaderInvokeBase &operator<<(VolumeView<T> volume) noexcept {
        _encoder.encode_texture(volume.handle(), volume.level());
        return *this;
    }

    template<typename T>
    ShaderInvokeBase &operator<<(const Buffer<T> &buffer) noexcept {
        buffer._check_is_valid();
        return *this << buffer.view();
    }

    ShaderInvokeBase &operator<<(const ByteBuffer &buffer) noexcept;

    template<typename T>
    ShaderInvokeBase &operator<<(const Image<T> &image) noexcept {
        image._check_is_valid();
        return *this << image.view();
    }

    template<typename T>
    ShaderInvokeBase &operator<<(const Volume<T> &volume) noexcept {
        volume._check_is_valid();
        return *this << volume.view();
    }

    template<typename T>
    ShaderInvokeBase &operator<<(const SOA<T> &soa) noexcept;

    template<typename T>
    ShaderInvokeBase &operator<<(SOAView<T> soa) noexcept;

    template<typename T>
    ShaderInvokeBase &operator<<(T data) noexcept {
        _encoder.encode_uniform(&data, sizeof(T));
        return *this;
    }

    // see definition in rtx/accel.cpp
    ShaderInvokeBase &operator<<(const Accel &accel) noexcept;

    // see definition in runtime/bindless_array.cpp
    ShaderInvokeBase &operator<<(const BindlessArray &array) noexcept;

    // see definition in runtime/dispatch_buffer.cpp
    ShaderInvokeBase &operator<<(const IndirectDispatchBuffer &array) noexcept;

protected:
    [[nodiscard]] auto _parallelize(uint3 dispatch_size) noexcept {
        _encoder.set_dispatch_size(dispatch_size);
        return std::move(_encoder);
    }
    [[nodiscard]] auto _parallelize(luisa::span<const uint3> dispatch_sizes) noexcept {
        _encoder.set_dispatch_sizes(dispatch_sizes);
        return std::move(_encoder);
    }
    [[nodiscard]] auto _parallelize(const IndirectDispatchBuffer &indirect_buffer,
                                    uint32_t offset = 0,
                                    uint32_t max_dispatch_size = std::numeric_limits<uint32_t>::max()) noexcept {
        _encoder.set_dispatch_size(IndirectDispatchArg{indirect_buffer.handle(), offset, max_dispatch_size});
        return std::move(_encoder);
    }
};

template<size_t dim>
struct ShaderInvoke {
    static_assert(always_false_v<std::index_sequence<dim>>);
};

template<>
struct ShaderInvoke<1> : public ShaderInvokeBase {
    explicit ShaderInvoke(uint64_t handle, size_t arg_count, size_t uniform_size) noexcept
        : ShaderInvokeBase{handle, arg_count, uniform_size} {}
    [[nodiscard]] auto dispatch(uint size_x) && noexcept {
        return this->_parallelize(uint3{size_x, 1u, 1u}).build();
    }
    [[nodiscard]] auto dispatch(luisa::span<const uint3> dispatch_sizes) && noexcept {
        return this->_parallelize(dispatch_sizes).build();
    }
    [[nodiscard]] auto dispatch(const IndirectDispatchBuffer &indirect_buffer,
                                uint32_t offset = 0,
                                uint32_t max_dispatch_size = std::numeric_limits<uint32_t>::max()) && noexcept {
        return this->_parallelize(indirect_buffer, offset, max_dispatch_size).build();
    }
};

template<>
struct ShaderInvoke<2> : public ShaderInvokeBase {
    explicit ShaderInvoke(uint64_t handle, size_t arg_count, size_t uniform_size) noexcept
        : ShaderInvokeBase{handle, arg_count, uniform_size} {}
    [[nodiscard]] auto dispatch(uint size_x, uint size_y) && noexcept {
        return this->_parallelize(uint3{size_x, size_y, 1u}).build();
    }
    [[nodiscard]] auto dispatch(uint2 size) && noexcept {
        return this->_parallelize(uint3{size.x, size.y, 1u}).build();
    }
    [[nodiscard]] auto dispatch(luisa::span<const uint3> dispatch_sizes) && noexcept {
        return this->_parallelize(dispatch_sizes).build();
    }
    [[nodiscard]] auto dispatch(const IndirectDispatchBuffer &indirect_buffer,
                                uint32_t offset = 0,
                                uint32_t max_dispatch_size = std::numeric_limits<uint32_t>::max()) && noexcept {
        return std::move(*this)._parallelize(indirect_buffer, offset, max_dispatch_size).build();
    }
};

template<>
struct ShaderInvoke<3> : public ShaderInvokeBase {
    explicit ShaderInvoke(uint64_t handle, size_t arg_count, size_t uniform_size) noexcept
        : ShaderInvokeBase{handle, arg_count, uniform_size} {}
    [[nodiscard]] auto dispatch(uint size_x, uint size_y, uint size_z) && noexcept {
        return this->_parallelize(uint3{size_x, size_y, size_z}).build();
    }
    [[nodiscard]] auto dispatch(luisa::span<const uint3> dispatch_sizes) && noexcept {
        return this->_parallelize(dispatch_sizes).build();
    }
    [[nodiscard]] auto dispatch(const IndirectDispatchBuffer &indirect_buffer,
                                uint32_t offset = 0,
                                uint32_t max_dispatch_size = std::numeric_limits<uint32_t>::max()) && noexcept {
        return this->_parallelize(indirect_buffer, offset, max_dispatch_size).build();
    }
    [[nodiscard]] auto dispatch(uint3 size) && noexcept {
        return std::move(*this).dispatch(size.x, size.y, size.z);
    }
};

template<typename T>
struct shader_argument_encode_count {
    static constexpr uint value = 1u;
};

}// namespace detail

class ShaderBase : public Resource {
protected:
    size_t _uniform_size{};
    uint3 _block_size;

public:
    explicit ShaderBase(DeviceInterface *device,
                        const ShaderCreationInfo &info,
                        size_t uniform_size) noexcept
        : Resource{device, Tag::SHADER, info},
          _block_size{info.block_size},
          _uniform_size{uniform_size} {}
    explicit ShaderBase() = default;
    ~ShaderBase() noexcept override {
        if (*this) { device()->destroy_shader(handle()); }
    }
    ShaderBase(ShaderBase &&) noexcept = default;
    ShaderBase(ShaderBase const &) noexcept = delete;
    ShaderBase &operator=(ShaderBase &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    ShaderBase &operator=(ShaderBase const &) noexcept = delete;
    using Resource::operator bool;
    [[nodiscard]] auto block_size() const noexcept {
        _check_is_valid();
        return _block_size;
    }
    [[nodiscard]] auto uniform_size() const noexcept {
        _check_is_valid();
        return _uniform_size;
    }
};

template<size_t dimension, typename... Args>
class Shader final : public ShaderBase {

    static_assert(dimension == 1u || dimension == 2u || dimension == 3u);

    friend class Device;

private:
    // base ctor
    Shader(DeviceInterface *device,
           const ShaderCreationInfo &info,
           size_t uniform_size) noexcept
        : ShaderBase{device, info, uniform_size} {}

private:
    // JIT shader
    Shader(DeviceInterface *device,
           Function kernel,
           const ShaderOption &option) noexcept
        : ShaderBase{device, device->create_shader(option, kernel),
                     ShaderDispatchCmdEncoder::compute_uniform_size(kernel.unbound_arguments())} {}

#ifdef LUISA_ENABLE_IR
    // JIT shader from IR module
    Shader(DeviceInterface *device,
           const ir::KernelModule *const module,
           const ShaderOption &option) noexcept
        : Shader{device, IR2AST::build(module)->function(), option} {}
#endif

    // AOT shader
    Shader(DeviceInterface *device, string_view file_path) noexcept
        : ShaderBase{device,
                     device->load_shader(file_path, detail::shader_argument_types<Args...>()),
                     ShaderDispatchCmdEncoder::compute_uniform_size(detail::shader_argument_types<Args...>())} {}

public:
    Shader() noexcept = default;
    ~Shader() noexcept = default;
    Shader(Shader &&) noexcept = default;
    Shader(Shader const &) noexcept = delete;
    Shader &operator=(Shader &&rhs) noexcept = default;
    Shader &operator=(Shader const &) noexcept = delete;
    using Resource::operator bool;

    [[nodiscard]] auto operator()(detail::prototype_to_shader_invocation_t<Args>... args) const noexcept {
        _check_is_valid();
        using invoke_type = detail::ShaderInvoke<dimension>;
        auto arg_count = (0u + ... + detail::shader_argument_encode_count<Args>::value);
        invoke_type invoke{handle(), arg_count, _uniform_size};
        static_cast<void>((invoke << ... << args));
        return invoke;
    }
};

template<typename... Args>
using Shader1D = Shader<1, Args...>;

template<typename... Args>
using Shader2D = Shader<2, Args...>;

template<typename... Args>
using Shader3D = Shader<3, Args...>;

}// namespace luisa::compute
