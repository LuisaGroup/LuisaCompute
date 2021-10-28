//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <functional>

#include <core/arena.h>
#include <core/concepts.h>
#include <ast/function.h>
#include <runtime/pixel.h>
#include <runtime/command_list.h>

namespace luisa::compute {

class Context;

class Event;
class Stream;
class Heap;
class Sampler;
class Mesh;
class Accel;

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
    class Interface : public std::enable_shared_from_this<Interface> {

    private:
        const Context &_ctx;

    public:
        explicit Interface(const Context &ctx) noexcept : _ctx{ctx} {}
        virtual ~Interface() noexcept = default;

        [[nodiscard]] const Context &context() const noexcept { return _ctx; }

        // native handle
        [[nodiscard]] virtual void *native_handle() const noexcept = 0;

        // buffer
        [[nodiscard]] virtual uint64_t create_buffer(
            size_t size_bytes,
            uint64_t heap_handle,// == uint64(-1) when not from heap
            uint32_t index_in_heap) noexcept = 0;
        virtual void destroy_buffer(uint64_t handle) noexcept = 0;
        [[nodiscard]] virtual void *buffer_native_handle(uint64_t handle) const noexcept = 0;

        // texture
        [[nodiscard]] virtual uint64_t create_texture(
            PixelFormat format, uint dimension,
            uint width, uint height, uint depth,
            uint mipmap_levels,
            Sampler sampler,
            uint64_t heap_handle,// == uint64(-1) when not from heap
            uint32_t index_in_heap) = 0;
        virtual void destroy_texture(uint64_t handle) noexcept = 0;
        [[nodiscard]] virtual void *texture_native_handle(uint64_t handle) const noexcept = 0;

        // texture heap
        [[nodiscard]] virtual uint64_t create_heap(size_t size) noexcept = 0;
        [[nodiscard]] virtual size_t query_heap_memory_usage(uint64_t handle) noexcept = 0;
        virtual void destroy_heap(uint64_t handle) noexcept = 0;

        // stream
        [[nodiscard]] virtual uint64_t create_stream() noexcept = 0;
        virtual void destroy_stream(uint64_t handle) noexcept = 0;
        virtual void synchronize_stream(uint64_t stream_handle) noexcept = 0;
        virtual void dispatch(uint64_t stream_handle, CommandList) noexcept = 0;
        [[nodiscard]] virtual void *stream_native_handle(uint64_t handle) const noexcept = 0;

        // kernel
        [[nodiscard]] virtual uint64_t create_shader(Function kernel) noexcept = 0;
        virtual void destroy_shader(uint64_t handle) noexcept = 0;

        // event
        [[nodiscard]] virtual uint64_t create_event() noexcept = 0;
        virtual void destroy_event(uint64_t handle) noexcept = 0;
        virtual void signal_event(uint64_t handle, uint64_t stream_handle) noexcept = 0;
        virtual void wait_event(uint64_t handle, uint64_t stream_handle) noexcept = 0;
        virtual void synchronize_event(uint64_t handle) noexcept = 0;

        // accel
        [[nodiscard]] virtual uint64_t create_mesh() noexcept = 0;
        virtual void destroy_mesh(uint64_t handle) noexcept = 0;
        [[nodiscard]] virtual uint64_t create_accel() noexcept = 0;
        virtual void destroy_accel(uint64_t handle) noexcept = 0;
    };

    using Deleter = void(Interface *);
    using Creator = Interface *(const Context &ctx, uint32_t index);
    using Handle = std::shared_ptr<Interface>;

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

    [[nodiscard]] Stream create_stream() noexcept;                // see definition in runtime/stream.cpp
    [[nodiscard]] Event create_event() noexcept;                  // see definition in runtime/event.cpp
    [[nodiscard]] Mesh create_mesh() noexcept;                    // see definition in rtx/mesh.cpp
    [[nodiscard]] Accel create_accel() noexcept;                  // see definition in rtx/accel.cpp
    [[nodiscard]] Heap create_heap(size_t size = 128_mb) noexcept;// see definition in runtime/heap.cpp

    template<typename T>
    [[nodiscard]] auto create_image(PixelStorage pixel, uint width, uint height, uint mip_levels = 1u) noexcept {
        return _create<Image<T>>(pixel, width, height, mip_levels);
    }

    template<typename T>
    [[nodiscard]] auto create_image(PixelStorage pixel, uint2 size, uint mip_levels = 1u) noexcept {
        return _create<Image<T>>(pixel, size, mip_levels);
    }

    template<typename T>
    [[nodiscard]] auto create_volume(PixelStorage pixel, uint width, uint height, uint depth, uint mip_levels = 1u) noexcept {
        return _create<Volume<T>>(pixel, width, height, depth, mip_levels);
    }

    template<typename T>
    [[nodiscard]] auto create_volume(PixelStorage pixel, uint3 size, uint mip_levels = 1u) noexcept {
        return _create<Volume<T>>(pixel, size, mip_levels);
    }

    template<typename T>
    [[nodiscard]] auto create_buffer(size_t size) noexcept {
        return _create<Buffer<T>>(size);
    }

    // see definitions in dsl/func.h
    template<size_t N, typename... Args>
    [[nodiscard]] auto compile(const Kernel<N, Args...> &kernel) noexcept {
        return _create<Shader<N, Args...>>(kernel.function());
    }
};

}// namespace luisa::compute
