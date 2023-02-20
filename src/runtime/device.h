//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <ast/type_registry.h>
#include <runtime/device_interface.h>

namespace luisa::compute {

class MeshFormat;
struct RasterState;
class Context;
class Event;
class Stream;
class Mesh;
class ProceduralPrimitive;
class Accel;
class SwapChain;
class BinaryIO;
class BindlessArray;
class IndirectDispatchBuffer;

template<typename T>
class Buffer;

template<typename T>
class Image;

template<typename T>
class Volume;

template<size_t dim, typename... Args>
class Shader;

template<size_t dim, typename... Args>
class AOTShader;

template<size_t N, typename... Args>
class Kernel;

template<typename... Args>
class RasterShader;

template<typename VertCallable, typename PixelCallable>
class RasterKernel;

template<typename... Args>
struct Kernel1D;

template<typename... Args>
struct Kernel2D;

template<typename... Args>
struct Kernel3D;
class DepthBuffer;

namespace detail {

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
    using Deleter = void(DeviceInterface *);
    using Creator = DeviceInterface *(Context && /* context */, const DeviceConfig * /* properties */);
    using Handle = luisa::shared_ptr<DeviceInterface>;

private:
    Handle _impl;

    template<typename T, typename... Args>
    [[nodiscard]] auto _create(Args &&...args) noexcept {
        return T{this->_impl.get(), std::forward<Args>(args)...};
    }
    static void _check_no_implicit_binding(Function func, luisa::string_view shader_path) noexcept;

public:
    // Device construct from backend handle, use Context::create_device for convenient usage
    explicit Device(Handle handle) noexcept : _impl{std::move(handle)} {}
    // Return a 128-bit device local unique identity hash-code, this code transformed after backend device's configure(hardware like GPU, drivers, etc.) changed
    [[nodiscard]] decltype(auto) device_hash() const noexcept { return _impl->device_hash(); }
    // Shader may generate cache file for runtime loading performance, cache_name return this name from backend
    [[nodiscard]] decltype(auto) cache_name(luisa::string_view file_name) const noexcept { return _impl->cache_name(file_name); }
    // see definition in runtime/context.h
    [[nodiscard]] decltype(auto) context() const noexcept { return _impl->context(); }
    // The backend implementation, can be used by other frontend language
    [[nodiscard]] auto impl() const noexcept { return _impl.get(); }
    // backend native plugins & extensions interface
    template<typename Ext = DeviceExtension>
        requires std::derived_from<Ext, DeviceExtension>
    [[nodiscard]] auto extension(luisa::string_view name) const noexcept {
        return dynamic_cast<Ext *>(_impl->extension(name));
    }
    // see definition in runtime/stream.cpp
    [[nodiscard]] Stream create_stream(StreamTag stream_tag = StreamTag::COMPUTE) noexcept;
    // see definition in runtime/event.cpp
    [[nodiscard]] Event create_event() noexcept;
    // see definition in runtime/swap_chain.cpp
    [[nodiscard]] SwapChain create_swapchain(
        uint64_t window_handle, const Stream &stream, uint2 resolution,
        bool allow_hdr = true, bool vsync = true, uint back_buffer_count = 1) noexcept;
    // see definition in runtime/dispatch_buffer.cpp
    [[nodiscard]] IndirectDispatchBuffer create_indirect_dispatch_buffer(size_t capacity) noexcept;
    // see definition in rtx/mesh.h
    template<typename VBuffer, typename TBuffer>
    [[nodiscard]] Mesh create_mesh(VBuffer &&vertices,
                                   TBuffer &&triangles,
                                   const AccelOption &option = {}) noexcept;
    // see definition in rtx/procedural_primitive.h
    template<typename AABBBuffer>
    [[nodiscard]] ProceduralPrimitive create_procedural_primitive(AABBBuffer &&aabb_buffer,
                                                                  const AccelOption &option = {}) noexcept;
    // see definition in rtx/accel.cpp
    [[nodiscard]] Accel create_accel(const AccelOption &option = {}) noexcept;
    // see definition in runtime/bindless_array.cpp
    [[nodiscard]] BindlessArray create_bindless_array(size_t slots = 65536u) noexcept;

    template<typename T>
    [[nodiscard]] auto create_image(PixelStorage pixel, uint width, uint height, uint mip_levels = 1u) noexcept {
        return _create<Image<T>>(pixel, make_uint2(width, height), mip_levels);
    }

    template<typename T>
    [[nodiscard]] auto create_image(PixelStorage pixel, uint2 size, uint mip_levels = 1u) noexcept {
        return _create<Image<T>>(pixel, size, mip_levels);
    }

    [[nodiscard]] DepthBuffer create_depth_buffer(DepthFormat depth_format, uint2 size) noexcept;

    template<typename T>
    [[nodiscard]] auto create_volume(PixelStorage pixel, uint width, uint height, uint depth, uint mip_levels = 1u) noexcept {
        return _create<Volume<T>>(pixel, make_uint3(width, height, depth), mip_levels);
    }

    template<typename T>
    [[nodiscard]] auto create_volume(PixelStorage pixel, uint3 size, uint mip_levels = 1u) noexcept {
        return _create<Volume<T>>(pixel, size, mip_levels);
    }

    template<typename T>
        requires(!is_custom_struct_v<T>)//backend-specific type not allowed
    [[nodiscard]] auto create_buffer(size_t size) noexcept {
        return _create<Buffer<T>>(size);
    }

    template<typename T>
        requires(!is_custom_struct_v<T>)//backend-specific type not allowed
    [[nodiscard]] auto create_buffer(void *ptr, size_t size) noexcept {
        return _create<Buffer<T>>(ptr, size);
    }

    // [[nodiscard]] Buffer<DrawIndirectArgs> create_draw_buffer(const MeshFormat &mesh_format, size_t capacity) noexcept;
    // [[nodiscard]] Buffer<DrawIndexedIndirectArgs> create_indexed_draw_buffer(const MeshFormat &mesh_format, size_t capacity) noexcept;

    void set_io(BinaryIO *visitor) noexcept { _impl->set_io(visitor); }

    template<size_t N, typename... Args>
    [[nodiscard]] auto compile(const Kernel<N, Args...> &kernel,
                               const ShaderOption &option = {}) noexcept {
        return _create<Shader<N, Args...>>(kernel.function()->function(), option);
    }

    template<typename Kernel>
    void compile_to(Kernel &&kernel,
                    luisa::string_view name,
                    bool enable_fast_math = true,
                    bool enable_debug_info = false) noexcept {
        ShaderOption option{
            .enable_cache = false,
            .enable_fast_math = enable_fast_math,
            .enable_debug_info = enable_debug_info,
            .compile_only = true,
            .name = name};
        static_cast<void>(this->compile(std::forward<Kernel>(kernel), option));
    }

    template<size_t N, typename Func>
        requires(std::negation_v<detail::is_dsl_kernel<std::remove_cvref_t<Func>>> && N >= 1 && N <= 3)
    [[nodiscard]] auto compile(Func &&f, const ShaderOption &option = {}) noexcept {
        if constexpr (N == 1u) {
            return compile(Kernel1D{std::forward<Func>(f)}, option);
        } else if constexpr (N == 2u) {
            return compile(Kernel2D{std::forward<Func>(f)}, option);
        } else {
            return compile(Kernel3D{std::forward<Func>(f)}, option);
        }
    }

    template<size_t N, typename Kernel>
    void compile_to(Kernel &&kernel,
                    luisa::string_view name,
                    bool enable_fast_math = true,
                    bool enable_debug_info = false) noexcept {
        ShaderOption option{
            .enable_cache = false,
            .enable_fast_math = enable_fast_math,
            .enable_debug_info = enable_debug_info,
            .compile_only = true,
            .name = name};
        static_cast<void>(this->compile<N>(std::forward<Kernel>(kernel), option));
    }

    template<typename... Args>
    [[nodiscard]] auto compile(
        const RasterKernel<Args...> &kernel,
        const MeshFormat &mesh_format,
        const RasterState &raster_state,
        luisa::span<PixelFormat const> rtv_format,
        DepthFormat dsv_format,
        const ShaderOption &option = {}) noexcept {
        return _create<typename RasterKernel<Args...>::RasterShaderType>(mesh_format, raster_state, rtv_format, dsv_format, kernel.vert(), kernel.pixel(), option);
    }
    template<typename V, typename P>
    void compile_to(
        const RasterKernel<V, P> &kernel,
        const MeshFormat &format,
        luisa::string_view serialization_path,
        bool enable_debug_info,
        bool enable_fast_math) {
        _check_no_implicit_binding(kernel.vert().get(), serialization_path);
        _check_no_implicit_binding(kernel.pixel().get(), serialization_path);
        _impl->save_raster_shader(format, Function(kernel.vert().get()), Function(kernel.pixel().get()), serialization_path, enable_debug_info, enable_fast_math);
    }

    template<typename... Args>
    RasterShader<Args...> load_raster_shader(
        const MeshFormat &mesh_format,
        const RasterState &raster_state,
        luisa::span<PixelFormat const> rtv_format,
        DepthFormat dsv_format,
        luisa::string_view shader_name) {
        return _create<RasterShader<Args...>>(mesh_format, raster_state, rtv_format, dsv_format, shader_name);
    }

    template<size_t N, typename... Args>
    [[nodiscard]] auto load_shader(luisa::string_view shader_name) noexcept {
        return _create<Shader<N, Args...>>(shader_name);
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
