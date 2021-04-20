//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <functional>

#include <core/memory.h>
#include <core/concepts.h>
#include <runtime/pixel.h>
#include <runtime/command_buffer.h>

namespace luisa::compute {

class Context;

class Event;
class Stream;

template<typename T>
class Buffer;

template<typename T>
class Image;

template<typename T>
class Volume;

class Device {

public:
    class Interface {
    private:
        const Context &_ctx;

    public:
        explicit Interface(const Context &ctx) noexcept : _ctx{ctx} {}
        virtual ~Interface() noexcept = default;

        [[nodiscard]] const Context &context() const noexcept { return _ctx; }

        // buffer
        [[nodiscard]] virtual uint64_t create_buffer(size_t size_bytes) noexcept = 0;
        virtual void dispose_buffer(uint64_t handle) noexcept = 0;

        // texture
        [[nodiscard]] virtual uint64_t create_texture(
            PixelFormat format, uint dimension, uint width, uint height, uint depth,
            uint mipmap_levels, bool is_bindless) = 0;
        virtual void dispose_texture(uint64_t handle) noexcept = 0;

        // stream
        [[nodiscard]] virtual uint64_t create_stream() noexcept = 0;
        virtual void dispose_stream(uint64_t handle) noexcept = 0;
        virtual void synchronize_stream(uint64_t stream_handle) noexcept = 0;
        virtual void dispatch(uint64_t stream_handle, CommandBuffer) noexcept = 0;

        // kernel
        virtual void compile_kernel(uint32_t uid) noexcept = 0;

        // event
        [[nodiscard]] virtual uint64_t create_event() noexcept = 0;
        virtual void dispose_event(uint64_t handle) noexcept = 0;
        virtual void signal_event(uint64_t handle, uint64_t stream_handle) noexcept = 0;
        virtual void wait_event(uint64_t handle, uint64_t stream_handle) noexcept = 0;
        virtual void synchronize_event(uint64_t handle) noexcept = 0;
    };

    using Deleter = void(Interface *);
    using Creator = Interface *(const Context &ctx, uint32_t index);
    using Handle = std::unique_ptr<Interface, Deleter *>;

private:
    Handle _impl;

public:
    explicit Device(Handle handle) noexcept
        : _impl{std::move(handle)} {}

    [[nodiscard]] decltype(auto) context() const noexcept { return _impl->context(); }
    [[nodiscard]] auto impl() const noexcept { return _impl.get(); }

    template<typename T, typename... Args>
    [[nodiscard]] auto create(Args &&...args) noexcept {
        return T{*this, std::forward<Args>(args)...};
    }

    [[nodiscard]] Stream create_stream() noexcept;
    [[nodiscard]] Event create_event() noexcept;

    template<typename T>
    [[nodiscard]] auto create_image(PixelStorage pixel, uint width, uint height) noexcept {
        return create<Image<T>>(pixel, width, height);
    }

    template<typename T>
    [[nodiscard]] auto create_image(PixelStorage pixel, uint2 size) noexcept {
        return create<Image<T>>(pixel, size);
    }
    
    template<typename T>
    [[nodiscard]] auto create_volume(PixelStorage pixel, uint width, uint height, uint depth) noexcept {
        return create<Volume<T>>(pixel, width, height, depth);
    }

    template<typename T>
    [[nodiscard]] auto create_volume(PixelStorage pixel, uint3 size) noexcept {
        return create<Volume<T>>(pixel, size);
    }

    template<typename T>
    [[nodiscard]] auto create_buffer(size_t size) noexcept {
        return create<Buffer<T>>(size);
    }

    template<typename Kernel>
    void compile(Kernel &&kernel) noexcept {
        kernel.wait_for_compilation(*this);
    }
};

}// namespace luisa::compute
