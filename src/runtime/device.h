//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <functional>

#include <core/concepts.h>
#include <core/thread_pool.h>
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
class SwapChain;
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

template<typename... Args>
class Kernel1D;

template<typename... Args>
class Kernel2D;

template<typename... Args>
class Kernel3D;

namespace detail {

class FunctionBuilder;

template<typename T>
struct is_dsl_kernel : std::false_type {};

template<size_t N, typename... Args>
struct is_dsl_kernel<Kernel<N, Args...>> : std::true_type {};

template<typename... Args>
struct is_dsl_kernel<Kernel1D<Args...>> : std::true_type {};

template<typename... Args>
struct is_dsl_kernel<Kernel2D<Args...>> : std::true_type {};

template<typename... Args>
struct is_dsl_kernel<Kernel3D<Args...>> : std::true_type {};

}// namespace detail

class LC_RUNTIME_API Device {

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
        virtual bool is_resource_in_bindless_array(uint64_t array, uint64_t handle) const noexcept = 0;
        virtual void remove_buffer_in_bindless_array(uint64_t array, size_t index) noexcept = 0;
        virtual void remove_tex2d_in_bindless_array(uint64_t array, size_t index) noexcept = 0;
        virtual void remove_tex3d_in_bindless_array(uint64_t array, size_t index) noexcept = 0;

        // stream
        [[nodiscard]] virtual uint64_t create_stream(bool for_present) noexcept = 0;
        virtual void destroy_stream(uint64_t handle) noexcept = 0;
        virtual void synchronize_stream(uint64_t stream_handle) noexcept = 0;
        virtual void dispatch(uint64_t stream_handle, const CommandList &list) noexcept = 0;
        virtual void dispatch(uint64_t stream_handle, luisa::span<const CommandList> lists) noexcept {
            for (auto &&list : lists) { dispatch(stream_handle, list); }
        }
        virtual void dispatch(uint64_t stream_handle, luisa::move_only_function<void()> &&func) noexcept = 0;
        [[nodiscard]] virtual void *stream_native_handle(uint64_t handle) const noexcept = 0;
        // swap chain
        [[nodiscard]] virtual uint64_t create_swap_chain(
            uint64_t window_handle, uint64_t stream_handle, uint width, uint height,
            bool allow_hdr, uint back_buffer_size) noexcept = 0;
        virtual void destroy_swap_chain(uint64_t handle) noexcept = 0;
        virtual PixelStorage swap_chain_pixel_storage(uint64_t handle) noexcept = 0;
        virtual void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept = 0;
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
            uint64_t t_buffer, size_t t_offset, size_t t_count, AccelUsageHint hint) noexcept = 0;
        virtual void destroy_mesh(uint64_t handle) noexcept = 0;
        [[nodiscard]] virtual uint64_t create_accel(AccelUsageHint hint) noexcept = 0;
        virtual void destroy_accel(uint64_t handle) noexcept = 0;

        // query
        [[nodiscard]] virtual luisa::string query(std::string_view meta_expr) noexcept { return {}; }
        [[nodiscard]] virtual bool requires_command_reordering() const noexcept { return true; }
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
    explicit Device(Handle handle) noexcept : _impl{std::move(handle)} {}

    [[nodiscard]] decltype(auto) context() const noexcept { return _impl->context(); }
    [[nodiscard]] auto impl() const noexcept { return _impl.get(); }

    [[nodiscard]] Stream create_stream(bool for_present = false) noexcept;// see definition in runtime/stream.cpp
    [[nodiscard]] Event create_event() noexcept;                          // see definition in runtime/event.cpp

    [[nodiscard]] SwapChain create_swapchain(
        uint64_t window_handle, const Stream &stream, uint2 resolution,
        bool allow_hdr = true, uint back_buffer_count = 1) noexcept;

    template<typename VBuffer, typename TBuffer>
    [[nodiscard]] Mesh create_mesh(
        VBuffer &&vertices, TBuffer &&triangles,
        AccelUsageHint hint = AccelUsageHint::FAST_TRACE) noexcept;                             // see definition in rtx/mesh.h
                                                                                                // see definition in rtx/mesh.h
    [[nodiscard]] Accel create_accel(AccelUsageHint hint = AccelUsageHint::FAST_TRACE) noexcept;// see definition in rtx/accel.cpp
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
    [[nodiscard]] auto compile(const Kernel<N, Args...> &kernel, luisa::string_view meta_options = {}) noexcept {
        return _create<Shader<N, Args...>>(kernel.function(), meta_options);
    }

    template<size_t N, typename... Args>
    [[nodiscard]] auto compile_async(const Kernel<N, Args...> &kernel, luisa::string_view meta_options = {}) noexcept {
        return ThreadPool::global().async([this, f = kernel.function(), opt = luisa::string{meta_options}] {
            return _create<Shader<N, Args...>>(f, opt);
        });
    }

    // clang-format off
    template<size_t N, typename Func>
        requires std::negation_v<detail::is_dsl_kernel<std::remove_cvref_t<Func>>>
    [[nodiscard]] auto compile(Func &&f, std::string_view meta_options = {}) noexcept {
        if constexpr (N == 1u) {
            return compile(Kernel1D{std::forward<Func>(f)});
        } else if constexpr (N == 2u) {
            return compile(Kernel2D{std::forward<Func>(f)});
        } else if constexpr (N == 3u) {
            return compile(Kernel3D{std::forward<Func>(f)});
        } else {
            static_assert(always_false_v<Func>, "Invalid kernel dimension.");
        }
    }
    template<size_t N, typename Func>
        requires std::negation_v<detail::is_dsl_kernel<std::remove_cvref_t<Func>>>
    [[nodiscard]] auto compile_async(Func &&f, std::string_view meta_options = {}) noexcept {
        if constexpr (N == 1u) {
            return compile_async(Kernel1D{std::forward<Func>(f)});
        } else if constexpr (N == 2u) {
            return compile_async(Kernel2D{std::forward<Func>(f)});
        } else if constexpr (N == 3u) {
            return compile_async(Kernel3D{std::forward<Func>(f)});
        } else {
            static_assert(always_false_v<Func>, "Invalid kernel dimension.");
        }
    }
    // clang-format on

    [[nodiscard]] auto query(std::string_view meta_expr) const noexcept {
        return _impl->query(meta_expr);
    }

    template<typename T, typename... Args>
    [[nodiscard]] auto create(Args &&...args) noexcept {
        return _create<T>(std::forward<Args>(args)...);
    }
};

}// namespace luisa::compute
