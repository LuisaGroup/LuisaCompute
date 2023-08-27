#pragma once
#include <luisa/vstl/common.h>
#include <luisa/vstl/spin_mutex.h>
#include <luisa/backends/ext/tex_compress_ext.h>
#include <luisa/backends/ext/native_resource_ext_interface.h>
#include <luisa/backends/ext/raster_ext_interface.h>
#include <luisa/backends/ext/dx_cuda_interop.h>
#include <luisa/backends/ext/dstorage_ext_interface.h>
#include <luisa/core/dynamic_module.h>
#include <dstorage/dstorage.h>
#include "../d3dx12.h"
using Microsoft::WRL::ComPtr;

namespace lc::dx {
class LCDevice;
using namespace luisa::compute;
class Device;
class DxTexCompressExt final : public TexCompressExt, public vstd::IOperatorNewBase {
public:
    static constexpr size_t BLOCK_SIZE = 16;
    Device *device;
    DxTexCompressExt(Device *device);
    ~DxTexCompressExt();
    Result compress_bc6h(Stream &stream, Image<float> const &src, luisa::compute::BufferView<uint> const &result) noexcept override;
    Result compress_bc7(Stream &stream, Image<float> const &src, luisa::compute::BufferView<uint> const &result, float alphaImportance) noexcept override;
    Result check_builtin_shader() noexcept override;
};
struct NativeTextureDesc {
    D3D12_RESOURCE_STATES initState;
    // custom_format only for LC's non-supported format, like DXGI_FORMAT_R10G10B10A2_UNORM
    // DXGI_FORMAT_UNKNOWN for default
    DXGI_FORMAT custom_format;
    bool allowUav;
};
class DxNativeResourceExt final : public NativeResourceExt, public vstd::IOperatorNewBase {
public:
    Device *dx_device;

    DxNativeResourceExt(DeviceInterface *lc_device, Device *dx_device);
    ~DxNativeResourceExt() = default;
    BufferCreationInfo register_external_buffer(
        void *external_ptr,
        const Type *element,
        size_t elem_count,
        // D3D12_RESOURCE_STATES const*
        void *custom_data) noexcept override;
    ResourceCreationInfo register_external_image(
        void *external_ptr,
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels,
        // NativeTextureDesc const*
        void *custom_data) noexcept override;
    ResourceCreationInfo register_external_depth_buffer(
        void *external_ptr,
        DepthFormat format,
        uint width,
        uint height,
        // D3D12_RESOURCE_STATES const*
        void *custom_data) noexcept override;
    uint64_t get_native_resource_device_address(
        void *native_handle) noexcept override;
    static PixelFormat ToPixelFormat(GFXFormat f) {
        switch (f) {
            case GFXFormat_R8_SInt:
                return PixelFormat::R8SInt;
            case GFXFormat_R8_UInt:
                return PixelFormat::R8UInt;
            case GFXFormat_R8_UNorm:
                return PixelFormat::R8UNorm;
            case GFXFormat_R8G8_SInt:
                return PixelFormat::RG8SInt;
            case GFXFormat_R8G8_UInt:
                return PixelFormat::RG8UInt;
            case GFXFormat_R8G8_UNorm:
                return PixelFormat::RG8UNorm;
            case GFXFormat_R8G8B8A8_SInt:
                return PixelFormat::RGBA8SInt;
            case GFXFormat_R8G8B8A8_UInt:
                return PixelFormat::RGBA8UInt;
            case GFXFormat_R8G8B8A8_UNorm:
                return PixelFormat::RGBA8UNorm;
            case GFXFormat_R16_SInt:
                return PixelFormat::R16SInt;
            case GFXFormat_R16_UInt:
                return PixelFormat::R16UInt;
            case GFXFormat_R16_UNorm:
                return PixelFormat::R16UNorm;
            case GFXFormat_R16G16_SInt:
                return PixelFormat::RG16SInt;
            case GFXFormat_R16G16_UInt:
                return PixelFormat::RG16UInt;
            case GFXFormat_R16G16_UNorm:
                return PixelFormat::RG16UNorm;
            case GFXFormat_R16G16B16A16_SInt:
                return PixelFormat::RGBA16SInt;
            case GFXFormat_R16G16B16A16_UInt:
                return PixelFormat::RGBA16UInt;
            case GFXFormat_R16G16B16A16_UNorm:
                return PixelFormat::RGBA16UNorm;
            case GFXFormat_R32_SInt:
                return PixelFormat::R32SInt;
            case GFXFormat_R32_UInt:
                return PixelFormat::R32UInt;
            case GFXFormat_R32G32_SInt:
                return PixelFormat::RG32SInt;
            case GFXFormat_R32G32_UInt:
                return PixelFormat::RG32UInt;
            case GFXFormat_R32G32B32A32_SInt:
                return PixelFormat::RGBA32SInt;
            case GFXFormat_R32G32B32A32_UInt:
                return PixelFormat::RGBA32UInt;
            case GFXFormat_R16_Float:
                return PixelFormat::R16F;
            case GFXFormat_R16G16_Float:
                return PixelFormat::RG16F;
            case GFXFormat_R16G16B16A16_Float:
                return PixelFormat::RGBA16F;
            case GFXFormat_R32_Float:
                return PixelFormat::R32F;
            case GFXFormat_R32G32_Float:
                return PixelFormat::RG32F;
            case GFXFormat_R32G32B32A32_Float:
                return PixelFormat::RGBA32F;
            case GFXFormat_BC6H_UF16:
                return PixelFormat::BC6HUF16;
            case GFXFormat_BC7_UNorm:
                return PixelFormat::BC7UNorm;
            case GFXFormat_BC5_UNorm:
                return PixelFormat::BC5UNorm;
            case GFXFormat_BC4_UNorm:
                return PixelFormat::BC4UNorm;
            default:
                return static_cast<PixelFormat>(-1);
        }
    }
};
class Device;
class DxRasterExt final : public RasterExt, public vstd::IOperatorNewBase {
    Device &nativeDevice;

public:
    DxRasterExt(Device &nativeDevice) noexcept : nativeDevice{nativeDevice} {}
    ResourceCreationInfo create_raster_shader(
        const MeshFormat &mesh_format,
        Function vert,
        Function pixel,
        const ShaderOption &cache_option) noexcept override;
    [[nodiscard]] ResourceCreationInfo load_raster_shader(
        const MeshFormat &mesh_format,
        luisa::span<Type const *const> types,
        luisa::string_view ser_path) noexcept override;
    void destroy_raster_shader(uint64_t handle) noexcept override;
    void warm_up_pipeline_cache(
        uint64_t shader_handle,
        luisa::span<PixelFormat const> render_target_formats,
        DepthFormat depth_format,
        const RasterState &state) noexcept override;

    ResourceCreationInfo create_depth_buffer(DepthFormat format, uint width, uint height) noexcept override;
    void destroy_depth_buffer(uint64_t handle) noexcept override;
};
class DxCudaInteropImpl : public luisa::compute::DxCudaInterop {
    Device &_device;

public:
    uint64_t cuda_buffer(uint64_t dx_buffer) noexcept override;
    uint64_t cuda_texture(uint64_t dx_texture) noexcept override;
    uint64_t cuda_event(uint64_t dx_event) noexcept override;
    DxCudaInteropImpl(Device &device) : _device{device} {}
};

class DStorageExtImpl final : public DStorageExt, public vstd::IOperatorNewBase {
    luisa::DynamicModule dstorage_core_module;
    luisa::DynamicModule dstorage_module;
    ComPtr<IDStorageFactory> factory;
    ComPtr<IDStorageCompressionCodec> compression_codec;
    vstd::spin_mutex spin_mtx;
    std::mutex mtx;
    std::atomic_size_t staging_size;
    LCDevice *mdevice;
    bool is_hdd = false;
    void init_factory();
    void init_factory_nolock();
    void set_config(bool hdd) noexcept;

public:
    auto Factory() const { return factory.Get(); }
    DeviceInterface *device() const noexcept override;
    DStorageExtImpl(std::filesystem::path const &runtime_dir, LCDevice *device) noexcept;
    ResourceCreationInfo create_stream_handle(const DStorageStreamOption &option) noexcept override;
    FileCreationInfo open_file_handle(luisa::string_view path) noexcept override;
    void close_file_handle(uint64_t handle) noexcept override;
    PinnedMemoryInfo pin_host_memory(void *ptr, size_t size_bytes) noexcept override {
        // no pin memory in dx yet
        PinnedMemoryInfo info;
        info.handle = reinterpret_cast<uint64_t>(ptr);
        info.native_handle = ptr;
        info.size_bytes = size_bytes;
        return info;
    }
    void unpin_host_memory(uint64_t handle) noexcept override {}
    void compress(
        const void *data, size_t size_bytes,
        Compression algorithm, CompressionQuality quality,
        luisa::vector<std::byte> &result) noexcept override;
};
}// namespace lc::dx
