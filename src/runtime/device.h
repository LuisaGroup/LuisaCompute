//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#ifdef LUISA_ENABLE_IR
#include "ir/ir2ast.h"
#endif
#include <ast/type_registry.h>
#include <runtime/rhi/device_interface.h>

namespace luisa {
class BinaryIO;
}

namespace luisa::compute {

class Context;
class Event;
class Stream;
class Mesh;
class MeshFormat;
class ProceduralPrimitive;
class Accel;
class Swapchain;
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
template<typename Ext>
concept ExtClass =
    requires {
        requires(std::is_base_of_v<DeviceExtension, Ext>);
        requires(std::is_same_v<const luisa::string_view, decltype(Ext::name)>);
    };
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
    // The backend name in lower case, can be used to recognize the corresponding backend
    [[nodiscard]] auto backend_name() const noexcept { return _impl->backend_name(); }
    // The backend implementation, can be used by other frontend language
    [[nodiscard]] auto impl() const noexcept { return _impl.get(); }
    // backend native plugins & extensions interface
    template<ExtClass Ext>
    [[nodiscard]] auto extension() const noexcept {
        return static_cast<Ext *>(_impl->extension(Ext::name));
    }
    // see definition in runtime/stream.cpp
    [[nodiscard]] Stream create_stream(StreamTag stream_tag = StreamTag::COMPUTE) noexcept;
    // see definition in runtime/event.cpp
    [[nodiscard]] Event create_event() noexcept;
    // see definition in runtime/swap_chain.cpp
    [[nodiscard]] Swapchain create_swapchain(
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
            .name = luisa::string{name}};
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
            .name = luisa::string{name}};
        static_cast<void>(this->compile<N>(std::forward<Kernel>(kernel), option));
    }

#ifdef LUISA_ENABLE_IR
    template<size_t N, typename... Args>
    [[nodiscard]] auto compile(const ir::KernelModule *const module,
                               const ShaderOption &option = {}) noexcept {
        return _create<Shader<N, Args...>>(module, option);
    }
#endif

    template<typename V, typename P>
    [[nodiscard]] typename RasterKernel<V, P>::RasterShaderType compile(
        const RasterKernel<V, P> &kernel,
        const MeshFormat &mesh_format,
        const ShaderOption &option = {}) noexcept;

    template<typename V, typename P>
    void compile_to(
        const RasterKernel<V, P> &kernel,
        const MeshFormat &mesh_format,
        luisa::string_view serialization_path,
        const ShaderOption& option = {}) noexcept;

    template<typename... Args>
    RasterShader<Args...> load_raster_shader(
        luisa::string_view shader_name) noexcept;

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
