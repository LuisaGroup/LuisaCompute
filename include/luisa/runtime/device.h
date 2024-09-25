#pragma once

#ifdef LUISA_ENABLE_IR
#include <luisa/ir/ir2ast.h>
#endif
#include <luisa/ast/type_registry.h>
#include <luisa/runtime/rhi/device_interface.h>

namespace luisa {
class BinaryIO;
}// namespace luisa

namespace luisa::compute {

class Context;
class Event;
class TimelineEvent;
class Stream;
class Mesh;
class Curve;
class MotionInstance;
class MeshFormat;
class ProceduralPrimitive;
class Accel;
class Swapchain;
class BindlessArray;
class IndirectDispatchBuffer;
class SparseBufferHeap;
class SparseTextureHeap;
class ByteBuffer;

template<typename T>
class SOA;

template<typename T>
class Buffer;

template<typename T>
class SparseBuffer;

template<typename T>
class Image;

template<typename T>
class Volume;

template<size_t dimension, concepts::non_cvref... Args>
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

template<typename T>
class SparseImage;

template<typename T>
class SparseVolume;

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

template<typename T>
concept device_extension = std::is_base_of_v<DeviceExtension, T> &&
                           std::is_same_v<const luisa::string_view, decltype(T::name)>;

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
    Device() noexcept = default;
    // Device construct from backend handle, use Context::create_device for convenience
    explicit Device(Handle handle) noexcept : _impl{std::move(handle)} {}
    // The backend name in lower case, can be used to recognize the corresponding backend
    [[nodiscard]] auto backend_name() const noexcept { return _impl->backend_name(); }
    // The native handle, can be used by other frontend language
    [[nodiscard]] auto native_handle() const noexcept { return _impl->native_handle(); }
    // The backend implementation, can be used by other frontend language
    [[nodiscard]] auto impl() const noexcept { return _impl.get(); }
    [[nodiscard]] auto const &impl_shared() const & noexcept { return _impl; }
    [[nodiscard]] auto &&impl_shared() && noexcept { return std::move(_impl); }
    [[nodiscard]] auto compute_warp_size() const noexcept { return _impl->compute_warp_size(); }
    // Is device initialized
    [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(_impl); }
    // backend native plugins & extensions interface
    template<device_extension Ext>
    [[nodiscard]] auto extension() const noexcept {
        return static_cast<Ext *>(_impl->extension(Ext::name));
    }
    // see definition in runtime/stream.cpp
    [[nodiscard]] Stream create_stream(StreamTag stream_tag = StreamTag::COMPUTE) noexcept;
    // see definition in runtime/event.cpp
    [[nodiscard]] Event create_event() noexcept;
    // see definition in runtime/event.cpp
    [[nodiscard]] TimelineEvent create_timeline_event() noexcept;
    // see definition in runtime/swap_chain.cpp
    [[nodiscard]] Swapchain create_swapchain(const Stream &stream, const SwapchainOption &option) noexcept;
    // see definition in runtime/dispatch_buffer.cpp
    [[nodiscard]] IndirectDispatchBuffer create_indirect_dispatch_buffer(size_t capacity) noexcept;
    // see definition in rtx/mesh.h
    template<typename VBuffer, typename TBuffer>
    [[nodiscard]] Mesh create_mesh(VBuffer &&vertices,
                                   TBuffer &&triangles,
                                   const AccelOption &option = {}) noexcept;

    template<typename VBuffer, typename TBuffer>
    [[nodiscard]] Mesh create_mesh(VBuffer &&vertices,
                                   size_t vertex_stride,
                                   TBuffer &&triangles,
                                   const AccelOption &option = {}) noexcept;
    // see definition in rtx/curve.h
    template<typename CPBuffer, typename SegmentBuffer>
    [[nodiscard]] Curve create_curve(CurveBasis basis,
                                     CPBuffer &&control_points,
                                     SegmentBuffer &&segments,
                                     const AccelOption &option = {}) noexcept;

    // see definition in rtx/procedural_primitive.h
    template<typename AABBBuffer>
    [[nodiscard]] ProceduralPrimitive create_procedural_primitive(AABBBuffer &&aabb_buffer,
                                                                  const AccelOption &option = {}) noexcept;

    // see definition in rtx/motion_instance.h
    [[nodiscard]] MotionInstance create_motion_instance(const Mesh &mesh, const AccelMotionOption &option) noexcept;
    [[nodiscard]] MotionInstance create_motion_instance(const Curve &curve, const AccelMotionOption &option) noexcept;
    [[nodiscard]] MotionInstance create_motion_instance(const ProceduralPrimitive &primitive, const AccelMotionOption &option) noexcept;

    // see definition in rtx/accel.cpp
    [[nodiscard]] Accel create_accel(const AccelOption &option = {}) noexcept;
    // see definition in runtime/bindless_array.cpp
    [[nodiscard]] BindlessArray create_bindless_array(size_t slots = 65536u) noexcept;

    template<typename T>
    [[nodiscard]] auto create_image(PixelStorage pixel, uint width, uint height, uint mip_levels = 1u, bool simultaneous_access = false, bool allow_raster_target = false) noexcept {
        return _create<Image<T>>(pixel, make_uint2(width, height), mip_levels, simultaneous_access, allow_raster_target);
    }

    template<typename T>
    [[nodiscard]] auto create_image(PixelStorage pixel, uint2 size, uint mip_levels = 1u, bool simultaneous_access = false, bool allow_raster_target = false) noexcept {
        return _create<Image<T>>(pixel, size, mip_levels, simultaneous_access, allow_raster_target);
    }

    template<typename T>
    [[nodiscard]] auto create_sparse_image(PixelStorage pixel, uint width, uint height, uint mip_levels = 1u, bool simultaneous_access = true) noexcept {
        return _create<SparseImage<T>>(pixel, make_uint2(width, height), mip_levels, simultaneous_access);
    }

    template<typename T>
    [[nodiscard]] auto create_sparse_image(PixelStorage pixel, uint2 size, uint mip_levels = 1u, bool simultaneous_access = true) noexcept {
        return _create<SparseImage<T>>(pixel, size, mip_levels, simultaneous_access);
    }

    [[nodiscard]] DepthBuffer create_depth_buffer(DepthFormat depth_format, uint2 size) noexcept;

    template<typename T>
    [[nodiscard]] auto create_volume(PixelStorage pixel, uint width, uint height, uint depth, uint mip_levels = 1u, bool simultaneous_access = false, bool allow_raster_target = false) noexcept {
        return _create<Volume<T>>(pixel, make_uint3(width, height, depth), mip_levels, simultaneous_access, allow_raster_target);
    }

    template<typename T>
    [[nodiscard]] auto create_volume(PixelStorage pixel, uint3 size, uint mip_levels = 1u, bool simultaneous_access = false, bool allow_raster_target = false) noexcept {
        return _create<Volume<T>>(pixel, size, mip_levels, simultaneous_access, allow_raster_target);
    }

    template<typename T>
    [[nodiscard]] auto create_sparse_volume(PixelStorage pixel, uint width, uint height, uint depth, uint mip_levels = 1u, bool simultaneous_access = true) noexcept {
        return _create<SparseVolume<T>>(pixel, make_uint3(width, height, depth), mip_levels, simultaneous_access);
    }

    template<typename T>
    [[nodiscard]] auto create_sparse_volume(PixelStorage pixel, uint3 size, uint mip_levels = 1u, bool simultaneous_access = true) noexcept {
        return _create<SparseVolume<T>>(pixel, size, mip_levels, simultaneous_access);
    }

    [[nodiscard]] SparseBufferHeap allocate_sparse_buffer_heap(size_t byte_size) noexcept;

    [[nodiscard]] SparseTextureHeap allocate_sparse_texture_heap(size_t byte_size, bool is_compressed_type) noexcept;

    [[nodiscard]] ByteBuffer create_byte_buffer(size_t byte_size) noexcept;

    [[nodiscard]] ByteBuffer import_external_byte_buffer(void *external_memory, size_t byte_size) noexcept;

    template<typename T>
        requires(!is_custom_struct_v<T>)//backend-specific type not allowed
    [[nodiscard]] auto create_buffer(size_t size) noexcept {
        return _create<Buffer<T>>(size);
    }

    template<typename T>
        requires(!is_custom_struct_v<T>)
    [[nodiscard]] auto import_external_buffer(void *external_memory, size_t elem_count) noexcept {
        return _create<Buffer<T>>(impl()->create_buffer(Type::of<T>(), elem_count, external_memory));
    }

    template<typename T>
    [[nodiscard]] auto create_soa(size_t size) noexcept {
        return SOA<T>{*this, size};
    }

    template<typename T>
        requires(!is_custom_struct_v<T>)//backend-specific type not allowed
    [[nodiscard]] auto create_sparse_buffer(size_t size) noexcept {
        return _create<SparseBuffer<T>>(size);
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
        const ShaderOption &option = {}) noexcept;

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
