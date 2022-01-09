//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <functional>

#include <core/concepts.h>
#include <ast/function.h>
#include <meta/property.h>
#include <runtime/context.h>
#include <runtime/pixel.h>
#include <runtime/sampler.h>
#include <runtime/command_list.h>

namespace luisa::compute {

class Context;

class Event;
class Stream;
class Mesh;
class Accel;
class BindlessArray;

template<typename T>
class Buffer;

template<typename T>
class Image;

template<typename T>
class Volume;

template<size_t dim, typename... Args>
class Shader;

template<size_t N, typename... Args>
class Kernel;

namespace detail {
class FunctionBuilder;
}

class Device {

public:
    class Interface : public luisa::enable_shared_from_this<Interface> {

    private:
        Context _ctx;

    public:
        explicit Interface(Context ctx) noexcept : _ctx{std::move(ctx)} {}
        virtual ~Interface() noexcept = default;

        [[nodiscard]] const Context &context() const noexcept { return _ctx; }

        // native handle
        [[nodiscard]] virtual void *native_handle() const noexcept = 0;

        // buffer
        [[nodiscard]] virtual uint64_t create_buffer(size_t size_bytes) noexcept = 0;
        virtual void destroy_buffer(uint64_t handle) noexcept = 0;
        [[nodiscard]] virtual void *buffer_native_handle(uint64_t handle) const noexcept = 0;

        // texture
        [[nodiscard]] virtual uint64_t create_texture(
            PixelFormat format, uint dimension,
            uint width, uint height, uint depth,
            uint mipmap_levels) noexcept = 0;
        virtual void destroy_texture(uint64_t handle) noexcept = 0;
        [[nodiscard]] virtual void *texture_native_handle(uint64_t handle) const noexcept = 0;

        // bindless array
        [[nodiscard]] virtual uint64_t create_bindless_array(size_t size) noexcept = 0;
        virtual void destroy_bindless_array(uint64_t handle) noexcept = 0;
        virtual void emplace_buffer_in_bindless_array(uint64_t array, size_t index, uint64_t handle, size_t offset_bytes) noexcept = 0;
        virtual void emplace_tex2d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept = 0;
        virtual void emplace_tex3d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept = 0;
        virtual bool is_buffer_in_bindless_array(uint64_t array, uint64_t handle) const noexcept = 0;
        virtual bool is_texture_in_bindless_array(uint64_t array, uint64_t handle) const noexcept = 0;
        virtual void remove_buffer_in_bindless_array(uint64_t array, size_t index) noexcept = 0;
        virtual void remove_tex2d_in_bindless_array(uint64_t array, size_t index) noexcept = 0;
        virtual void remove_tex3d_in_bindless_array(uint64_t array, size_t index) noexcept = 0;

        // stream
        [[nodiscard]] virtual uint64_t create_stream() noexcept = 0;
        virtual void destroy_stream(uint64_t handle) noexcept = 0;
        virtual void synchronize_stream(uint64_t stream_handle) noexcept = 0;
        virtual void dispatch(uint64_t stream_handle, CommandList) noexcept = 0;
        [[nodiscard]] virtual void *stream_native_handle(uint64_t handle) const noexcept = 0;

        // kernel
        [[nodiscard]] virtual uint64_t create_shader(Function kernel, std::string_view meta_options) noexcept = 0;
        virtual void destroy_shader(uint64_t handle) noexcept = 0;

        // event
        [[nodiscard]] virtual uint64_t create_event() noexcept = 0;
        virtual void destroy_event(uint64_t handle) noexcept = 0;
        virtual void signal_event(uint64_t handle, uint64_t stream_handle) noexcept = 0;
        virtual void wait_event(uint64_t handle, uint64_t stream_handle) noexcept = 0;
        virtual void synchronize_event(uint64_t handle) noexcept = 0;

        // accel
        [[nodiscard]] virtual uint64_t create_mesh(
            uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
            uint64_t t_buffer, size_t t_offset, size_t t_count, AccelBuildHint hint) noexcept = 0;
        virtual void destroy_mesh(uint64_t handle) noexcept = 0;

        [[nodiscard]] virtual uint64_t create_accel(AccelBuildHint hint) noexcept = 0;
        virtual void emplace_back_instance_in_accel(uint64_t accel, uint64_t mesh, float4x4 transform, bool visible) noexcept = 0;
        virtual void pop_back_instance_from_accel(uint64_t accel) noexcept = 0;
        virtual void set_instance_in_accel(uint64_t accel, size_t index, uint64_t mesh, float4x4 transform, bool visible) noexcept = 0;
        virtual void set_instance_transform_in_accel(uint64_t accel, size_t index, float4x4 transform) noexcept = 0;
        virtual void set_instance_visibility_in_accel(uint64_t accel, size_t index, bool visible) noexcept = 0;
        [[nodiscard]] virtual bool is_buffer_in_accel(uint64_t accel, uint64_t buffer) const noexcept = 0;
        [[nodiscard]] virtual bool is_mesh_in_accel(uint64_t accel, uint64_t mesh) const noexcept = 0;
        [[nodiscard]] virtual uint64_t get_vertex_buffer_from_mesh(uint64_t mesh_handle) const noexcept = 0;
        [[nodiscard]] virtual uint64_t get_triangle_buffer_from_mesh(uint64_t mesh_handle) const noexcept = 0;
        virtual void destroy_accel(uint64_t handle) noexcept = 0;

        [[nodiscard]] virtual luisa::string query(std::string_view meta_expr) noexcept { return {}; }
    };

    using Deleter = void(Interface *);
    using Creator = Interface *(const Context & /* context */, std::string_view /* properties */);
    using Handle = luisa::shared_ptr<Interface>;

private:
    Handle _impl;

    template<typename T, typename... Args>
    [[nodiscard]] auto _create(Args &&...args) noexcept {
        return T{this->_impl.get(), std::forward<Args>(args)...};
    }

public:
    explicit Device(Handle handle) noexcept
        : _impl{std::move(handle)} {}

    [[nodiscard]] decltype(auto) context() const noexcept { return _impl->context(); }
    [[nodiscard]] auto impl() const noexcept { return _impl.get(); }

    [[nodiscard]] Stream create_stream() noexcept;// see definition in runtime/stream.cpp
    [[nodiscard]] Event create_event() noexcept;  // see definition in runtime/event.cpp

    template<typename VBuffer, typename TBuffer>
    [[nodiscard]] Mesh create_mesh(
        VBuffer &&vertices, TBuffer &&triangles,
        AccelBuildHint hint = AccelBuildHint::FAST_TRACE) noexcept;                             // see definition in rtx/mesh.h
    [[nodiscard]] Accel create_accel(AccelBuildHint hint = AccelBuildHint::FAST_TRACE) noexcept;// see definition in rtx/accel.cpp
    [[nodiscard]] BindlessArray create_bindless_array(size_t slots = 65536u) noexcept;          // see definition in runtime/bindless_array.cpp

    template<typename T>
    [[nodiscard]] auto create_image(PixelStorage pixel, uint width, uint height, uint mip_levels = 1u) noexcept {
        return _create<Image<T>>(pixel, make_uint2(width, height), mip_levels);
    }

    template<typename T>
    [[nodiscard]] auto create_image(PixelStorage pixel, uint2 size, uint mip_levels = 1u) noexcept {
        return _create<Image<T>>(pixel, size, mip_levels);
    }

    template<typename T>
    [[nodiscard]] auto create_volume(PixelStorage pixel, uint width, uint height, uint depth, uint mip_levels = 1u) noexcept {
        return _create<Volume<T>>(pixel, make_uint3(width, height, depth), mip_levels);
    }

    template<typename T>
    [[nodiscard]] auto create_volume(PixelStorage pixel, uint3 size, uint mip_levels = 1u) noexcept {
        return _create<Volume<T>>(pixel, size, mip_levels);
    }

    template<typename T>
    [[nodiscard]] auto create_buffer(size_t size) noexcept {
        return _create<Buffer<T>>(size);
    }

    template<size_t N, typename... Args>
    [[nodiscard]] auto compile(const Kernel<N, Args...> &kernel, std::string_view meta_options = {}) noexcept {
        return _create<Shader<N, Args...>>(kernel.function(), meta_options);
    }

    [[nodiscard]] auto query(std::string_view meta_expr) const noexcept {
        return _impl->query(meta_expr);
    }

    template<typename T, typename... Args>
    [[nodiscard]] auto create(Args &&...args) noexcept {
        return _create<T>(std::forward<Args>(args)...);
    }
};

}// namespace luisa::compute
