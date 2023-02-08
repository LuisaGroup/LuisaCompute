//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <runtime/device_interface.h>
#include <runtime/custom_struct.h>
#include <runtime/dynamic_struct.h>

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
    using Creator = DeviceInterface *(Context && /* context */, DeviceConfig const * /* properties */);
    using Handle = luisa::shared_ptr<DeviceInterface>;

private:
    Handle _impl;

    template<typename T, typename... Args>
    [[nodiscard]] auto _create(Args &&...args) noexcept {
        return T{this->_impl.get(), std::forward<Args>(args)...};
    }

<<<<<<< HEAD
#ifndef NDEBUG
    static void _check_no_implicit_binding(Function func, luisa::string_view shader_name) noexcept;
#endif
=======
    static void _check_no_implicit_binding(Function func, luisa::string_view shader_path) noexcept;
>>>>>>> f9bd719a (merge)

public:
    explicit Device(Handle handle) noexcept : _impl{std::move(handle)} {}
    [[nodiscard]] decltype(auto) device_hash() const noexcept { return _impl->device_hash(); }
    [[nodiscard]] decltype(auto) cache_name(luisa::string_view file_name) const noexcept { return _impl->cache_name(file_name); }
    [[nodiscard]] decltype(auto) context() const noexcept { return _impl->context(); }
    [[nodiscard]] auto impl() const noexcept { return _impl.get(); }

    template<typename Ext = DeviceExtension>
        requires std::derived_from<Ext, DeviceExtension>
    [[nodiscard]] auto extension(luisa::string_view name) const noexcept {
        return dynamic_cast<Ext *>(_impl->extension(name));
    }

    [[nodiscard]] Stream create_stream(StreamTag stream_tag = StreamTag::COMPUTE) noexcept;// see definition in runtime/stream.cpp
    [[nodiscard]] Event create_event() noexcept;                                           // see definition in runtime/event.cpp

    [[nodiscard]] SwapChain create_swapchain(
        uint64_t window_handle, const Stream &stream, uint2 resolution,
        bool allow_hdr = true, bool vsync = true, uint back_buffer_count = 1) noexcept;

    template<size_t i, typename T>
    [[nodiscard]] Buffer<T> create_dispatch_buffer(size_t capacity) noexcept;

    template<typename VBuffer, typename TBuffer>
    [[nodiscard]] Mesh create_mesh(VBuffer &&vertices,
                                   TBuffer &&triangles,
                                   const AccelOption &option = {}) noexcept;// see definition in rtx/mesh.h

    [[nodiscard]] ProceduralPrimitive create_procedural_primitive(BufferView<AABB> aabb_buffer,
                                                                  const AccelOption &option = {}) noexcept;// see definition in rtx/procedural_primitive.h

    [[nodiscard]] Accel create_accel(const AccelOption &option = {}) noexcept;        // see definition in rtx/accel.cpp
    [[nodiscard]] BindlessArray create_bindless_array(size_t slots = 65536u) noexcept;// see definition in runtime/bindless_array.cpp

    template<typename T>
    [[nodiscard]] auto create_image(PixelStorage pixel, uint width, uint height, uint mip_levels = 1u) noexcept {
        return _create<Image<T>>(pixel, make_uint2(width, height), mip_levels);
    }

    template<typename T>
    [[nodiscard]] auto create_image(PixelStorage pixel, uint2 size, uint mip_levels = 1u) noexcept {
        return _create<Image<T>>(pixel, size, mip_levels);
    }
    DepthBuffer create_depth_buffer(DepthFormat depth_format, uint2 size) noexcept;

    template<typename T>
    [[nodiscard]] auto create_volume(PixelStorage pixel, uint width, uint height, uint depth, uint mip_levels = 1u) noexcept {
        return _create<Volume<T>>(pixel, make_uint3(width, height, depth), mip_levels);
    }

    template<typename T>
    [[nodiscard]] auto create_volume(PixelStorage pixel, uint3 size, uint mip_levels = 1u) noexcept {
        return _create<Volume<T>>(pixel, size, mip_levels);
    }

    template<typename T>
        requires(!std::is_base_of_v<CustomStructBase, T>)//backend-specific type not allowed
    [[nodiscard]] auto create_buffer(size_t size) noexcept {
        return _create<Buffer<T>>(size);
    }

    template<typename T>
        requires(!std::is_base_of_v<CustomStructBase, T>)//backend-specific type not allowed
    [[nodiscard]] auto create_buffer(void *ptr, size_t size) noexcept {
        return _create<Buffer<T>>(ptr, size);
    }

    [[nodiscard]] Buffer<DynamicStruct> create_buffer(const DynamicStruct &type, size_t size) noexcept;
    // TODO
    //    [[nodiscard]] Buffer<DispatchArgs1D> create_1d_dispatch_buffer(size_t capacity) noexcept;
    //    [[nodiscard]] Buffer<DispatchArgs2D> create_2d_dispatch_buffer(size_t capacity) noexcept;
    //    [[nodiscard]] Buffer<DispatchArgs3D> create_3d_dispatch_buffer(size_t capacity) noexcept;

    // [[nodiscard]] Buffer<DrawIndirectArgs> create_draw_buffer(const MeshFormat &mesh_format, size_t capacity) noexcept;
    // [[nodiscard]] Buffer<DrawIndexedIndirectArgs> create_indexed_draw_buffer(const MeshFormat &mesh_format, size_t capacity) noexcept;

    void set_io(BinaryIO *visitor) noexcept { _impl->set_io(visitor); }

    template<size_t N, typename... Args>
<<<<<<< HEAD
    [[nodiscard]] auto compile_to(
        const Kernel<N, Args...> &kernel,
        luisa::string_view shader_name,
        bool enable_debug_info = false,
        bool enable_fast_math = true) noexcept {
#ifndef NDEBUG
        _check_no_implicit_binding(kernel.function().get(), shader_name);
#endif
        return _create<Shader<N, Args...>>(kernel.function(), shader_name, enable_debug_info, enable_fast_math);
    }
    template<size_t N, typename... Args>
    [[nodiscard]] auto compile(
        const Kernel<N, Args...> &kernel,
        bool use_cache = true,
        bool enable_debug_info = false,
        bool enable_fast_math = true) noexcept {
        return _create<Shader<N, Args...>>(kernel.function(), use_cache, enable_debug_info, enable_fast_math);
=======
    [[nodiscard]] auto compile(const Kernel<N, Args...> &kernel,
                               const ShaderOption &option = {}) noexcept {
        return _create<Shader<N, Args...>>(kernel.function(), option);
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
            .name = name};
        static_cast<void>(this->compile(std::forward<Kernel>(kernel), option));
>>>>>>> f9bd719a (merge)
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
            .name = name};
        static_cast<void>(this->compile<N>(std::forward<Kernel>(kernel), option));
    }

    template<typename... Args>
    [[nodiscard]] auto compile_to(
        const RasterKernel<Args...> &kernel,
        const MeshFormat &mesh_format,
        const RasterState &raster_state,
        luisa::span<PixelFormat const> rtv_format,
        DepthFormat dsv_format,
<<<<<<< HEAD
        luisa::string_view shader_name,
        bool enable_debug_info = false,
        bool enable_fast_math = true) noexcept {
#ifndef NDEBUG
        _check_no_implicit_binding(kernel.vert().get(), shader_name);
        _check_no_implicit_binding(kernel.pixel().get(), shader_name);
#endif
        return _create<typename RasterKernel<Args...>::RasterShaderType>(mesh_format, raster_state, rtv_format, dsv_format, kernel.vert(), kernel.pixel(), shader_name, enable_debug_info, enable_fast_math);
=======
        luisa::string_view shader_path) noexcept {
        _check_no_implicit_binding(kernel.vert().get(), shader_path);
        _check_no_implicit_binding(kernel.pixel().get(), shader_path);
        return _create<typename RasterKernel<Args...>::RasterShaderType>(mesh_format, raster_state,
                                                                         rtv_format, dsv_format,
                                                                         kernel.vert(), kernel.pixel(),
                                                                         shader_path);
>>>>>>> f9bd719a (merge)
    }

    template<typename... Args>
    [[nodiscard]] auto compile(
        const RasterKernel<Args...> &kernel,
        const MeshFormat &mesh_format,
        const RasterState &raster_state,
        luisa::span<PixelFormat const> rtv_format,
        DepthFormat dsv_format,
        bool use_cache = true,
        bool enable_debug_info = false,
        bool enable_fast_math = true) noexcept {
        return _create<typename RasterKernel<Args...>::RasterShaderType>(mesh_format, raster_state, rtv_format, dsv_format, kernel.vert(), kernel.pixel(), use_cache, enable_debug_info, enable_fast_math);
    }
<<<<<<< HEAD
    template<size_t N, typename... Args>
    void save(
        const Kernel<N, Args...> &kernel,
        luisa::string_view shader_name,
        bool enable_debug_info = false,
        bool enable_fast_math = true) noexcept {
#ifndef NDEBUG
        _check_no_implicit_binding(kernel.function().get(), shader_name);
#endif
        _impl->create_shader(
            Function(kernel.function().get()),
            DeviceInterface::ShaderOption{
                .enable_debug_info = enable_debug_info,
                .enable_fast_math = enable_fast_math,
                .compile_only = true,
                .name = shader_name});
    }
    template<typename V, typename P>
    void save_raster_shader(
        const RasterKernel<V, P> &kernel,
        const MeshFormat &format,
        luisa::string_view shader_name,
        bool enable_debug_info = false,
        bool enable_fast_math = true) {
#ifndef NDEBUG
        _check_no_implicit_binding(kernel.vert().get(), shader_name);
        _check_no_implicit_binding(kernel.pixel().get(), shader_name);
#endif
        _impl->save_raster_shader(
            format,
            Function(kernel.vert().get()), Function(kernel.pixel().get()),
            shader_name, enable_debug_info, enable_fast_math);
=======

    template<typename V, typename P>
    void save_raster_shader(const RasterKernel<V, P> &kernel, const MeshFormat &format, luisa::string_view serialization_path) {
        _check_no_implicit_binding(kernel.vert().get(), serialization_path);
        _check_no_implicit_binding(kernel.pixel().get(), serialization_path);
        _impl->save_raster_shader(format, Function(kernel.vert().get()), Function(kernel.pixel().get()), serialization_path);
>>>>>>> f9bd719a (merge)
    }

    template<typename... Args>
    RasterShader<Args...> load_raster_shader(
        const MeshFormat &mesh_format,
        const RasterState &raster_state,
        luisa::span<PixelFormat const> rtv_format,
        DepthFormat dsv_format,
        luisa::string_view shader_name,
        bool enable_debug_info = false,
        bool enable_fast_math = true) {
        return _create<RasterShader<Args...>>(mesh_format, raster_state, rtv_format, dsv_format, shader_name, enable_debug_info, enable_fast_math);
    }

    template<size_t N, typename... Args>
    [[nodiscard]] auto load_shader(luisa::string_view shader_name) noexcept {
        return _create<Shader<N, Args...>>(shader_name);
    }

<<<<<<< HEAD
    template<size_t N, typename Func>
        requires(
            std::negation_v<detail::is_dsl_kernel<std::remove_cvref_t<Func>>> && N >= 1 && N <= 3)
    [[nodiscard]] auto compile_to(Func &&f, std::string_view shader_name) noexcept {
        if constexpr (N == 1u) {
            return compile_to(Kernel1D{std::forward<Func>(f)}, shader_name);
        } else if constexpr (N == 2u) {
            return compile_to(Kernel2D{std::forward<Func>(f)}, shader_name);
        } else {
            return compile_to(Kernel3D{std::forward<Func>(f)}, shader_name);
        }
    }
    template<size_t N, typename Func>
        requires(
            std::negation_v<detail::is_dsl_kernel<std::remove_cvref_t<Func>>> && N >= 1 && N <= 3)
    [[nodiscard]] auto compile(Func &&f, bool use_cache = true) noexcept {
        if constexpr (N == 1u) {
            return compile(Kernel1D{std::forward<Func>(f)}, use_cache);
        } else if constexpr (N == 2u) {
            return compile(Kernel2D{std::forward<Func>(f)}, use_cache);
        } else {
            return compile(Kernel3D{std::forward<Func>(f)}, use_cache);
        }
    }

=======
>>>>>>> f9bd719a (merge)
    [[nodiscard]] auto query(std::string_view meta_expr) const noexcept {
        return _impl->query(meta_expr);
    }

    template<typename T, typename... Args>
    [[nodiscard]] auto create(Args &&...args) noexcept {
        return _create<T>(std::forward<Args>(args)...);
    }
};

}// namespace luisa::compute
