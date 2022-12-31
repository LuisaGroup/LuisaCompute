#include <filesystem>
#include <Api/LCDevice.h>
#include <DXRuntime/Device.h>
#include <Resource/DefaultBuffer.h>
#include <Shader/ShaderCompiler.h>
#include <Resource/RenderTexture.h>
#include <Resource/DepthBuffer.h>
#include <Resource/BindlessArray.h>
#include <Shader/ComputeShader.h>
#include <Shader/RasterShader.h>
#include <Api/LCCmdBuffer.h>
#include <Api/LCEvent.h>
#include <vstl/md5.h>
#include <Shader/ShaderSerializer.h>
#include <Resource/BottomAccel.h>
#include <Resource/TopAccel.h>
#include <vstl/binary_reader.h>
#include <Api/LCSwapChain.h>
#include <Api/ext.h>
#include "HLSL/dx_codegen.h"
#include <ast/function_builder.h>
#include <Resource/DepthBuffer.h>
#include <core/stl/filesystem.h>
#include <Resource/ExternalBuffer.h>
#include <runtime/context_paths.h>
using namespace toolhub::directx;
namespace toolhub::directx {
static constexpr uint kShaderModel = 65u;
LCDevice::LCDevice(Context &&ctx, DeviceConfig const *settings)
    : DeviceInterface(std::move(ctx)),
      shaderPaths{_ctx.paths().cache_directory(), _ctx.paths().data_directory() / "dx_builtin", _ctx.paths().runtime_directory()},
      nativeDevice(_ctx, shaderPaths, settings) {
    exts.try_emplace("tex_compress"sv, [](LCDevice *device) -> DeviceExtension * {
        return new DxTexCompressExt(&device->nativeDevice);
    });
}
LCDevice::~LCDevice() {
}
Hash128 LCDevice::device_hash() const noexcept {
    vstd::MD5::MD5Data const &md5 = nativeDevice.adapterID.ToBinary();
    Hash128 r;
    static_assert(sizeof(Hash128) == sizeof(vstd::MD5::MD5Data));
    memcpy(&r, &md5, sizeof(Hash128));
    return r;
}
void *LCDevice::native_handle() const noexcept {
    return nativeDevice.device;
}
uint64 LCDevice::create_buffer(void *ptr) noexcept {
    return reinterpret_cast<uint64>(
        new ExternalBuffer(
            &nativeDevice,
            reinterpret_cast<ID3D12Resource *>(ptr)));
}
uint64 LCDevice::create_buffer(size_t size_bytes) noexcept {
    return reinterpret_cast<uint64>(
        static_cast<Buffer *>(
            new DefaultBuffer(
                &nativeDevice,
                size_bytes,
                nativeDevice.defaultAllocator.get())));
}
void LCDevice::destroy_buffer(uint64 handle) noexcept {
    delete reinterpret_cast<Buffer *>(handle);
}
void *LCDevice::buffer_native_handle(uint64 handle) const noexcept {
    return reinterpret_cast<Buffer *>(handle)->GetResource();
}
uint64 LCDevice::create_texture(
    PixelFormat format,
    uint dimension,
    uint width,
    uint height,
    uint depth,
    uint mipmap_levels) noexcept {
    bool allowUAV = true;
    switch (format) {
        case PixelFormat::BC4UNorm:
        case PixelFormat::BC5UNorm:
        case PixelFormat::BC6HUF16:
        case PixelFormat::BC7UNorm:
            allowUAV = false;
            break;
        default: break;
    }
    return reinterpret_cast<uint64>(
        static_cast<TextureBase *>(
            new RenderTexture(
                &nativeDevice,
                width,
                height,
                TextureBase::ToGFXFormat(format),
                (TextureDimension)dimension,
                depth,
                mipmap_levels,
                allowUAV,
                nativeDevice.defaultAllocator.get())));
}
string LCDevice::cache_name(string_view file_name) const noexcept {
    return Shader::PSOName(&nativeDevice, file_name);
}
void LCDevice::destroy_texture(uint64 handle) noexcept {
    delete reinterpret_cast<TextureBase *>(handle);
}
void *LCDevice::texture_native_handle(uint64 handle) const noexcept {
    return reinterpret_cast<TextureBase *>(handle)->GetResource();
}
uint64 LCDevice::create_bindless_array(size_t size) noexcept {
    return reinterpret_cast<uint64>(
        new BindlessArray(&nativeDevice, size));
}
void LCDevice::destroy_bindless_array(uint64 handle) noexcept {
    delete reinterpret_cast<BindlessArray *>(handle);
}
void LCDevice::emplace_buffer_in_bindless_array(uint64 array, size_t index, uint64 handle, size_t offset_bytes) noexcept {
    auto buffer = reinterpret_cast<Buffer *>(handle);
    reinterpret_cast<BindlessArray *>(array)
        ->Bind(handle, BufferView(buffer, offset_bytes), index);
}
void LCDevice::emplace_tex2d_in_bindless_array(uint64 array, size_t index, uint64 handle, Sampler sampler) noexcept {
    auto tex = reinterpret_cast<TextureBase *>(handle);
    reinterpret_cast<BindlessArray *>(array)
        ->Bind(handle, std::pair<TextureBase const *, Sampler>(tex, sampler), index);
}
void LCDevice::emplace_tex3d_in_bindless_array(uint64 array, size_t index, uint64 handle, Sampler sampler) noexcept {
    emplace_tex2d_in_bindless_array(array, index, handle, sampler);
}
/*
bool LCDevice::is_resource_in_bindless_array(uint64 array, uint64 handle) const noexcept {

}*/
void LCDevice::remove_buffer_from_bindless_array(uint64 array, size_t index) noexcept {
    reinterpret_cast<BindlessArray *>(array)
        ->UnBind(BindlessArray::BindTag::Buffer, index);
}
void LCDevice::remove_tex2d_from_bindless_array(uint64 array, size_t index) noexcept {
    reinterpret_cast<BindlessArray *>(array)
        ->UnBind(BindlessArray::BindTag::Tex2D, index);
}
void LCDevice::remove_tex3d_from_bindless_array(uint64 array, size_t index) noexcept {
    reinterpret_cast<BindlessArray *>(array)
        ->UnBind(BindlessArray::BindTag::Tex3D, index);
}
uint64 LCDevice::create_stream(StreamTag type) noexcept {
    return reinterpret_cast<uint64>(
        new LCCmdBuffer(
            &nativeDevice,
            nativeDevice.defaultAllocator.get(),
            [&] {
                switch (type) {
                    case compute::StreamTag::COMPUTE:
                        return D3D12_COMMAND_LIST_TYPE_COMPUTE;
                    case compute::StreamTag::GRAPHICS:
                        return D3D12_COMMAND_LIST_TYPE_DIRECT;
                    case compute::StreamTag::COPY:
                        return D3D12_COMMAND_LIST_TYPE_COPY;
                }
            }()));
}

void LCDevice::destroy_stream(uint64 handle) noexcept {
    delete reinterpret_cast<LCCmdBuffer *>(handle);
}
void LCDevice::synchronize_stream(uint64 stream_handle) noexcept {
    reinterpret_cast<LCCmdBuffer *>(stream_handle)->Sync();
}
void LCDevice::dispatch(uint64 stream_handle, CommandList &&list, fixed_vector<move_only_function<void()>, 1> &&func) noexcept {
    reinterpret_cast<LCCmdBuffer *>(stream_handle)
        ->Execute(std::move(list), nativeDevice.maxAllocatorCount, &func);
}
void LCDevice::dispatch(uint64 stream_handle, CommandList &&list) noexcept {
    reinterpret_cast<LCCmdBuffer *>(stream_handle)
        ->Execute(std::move(list), nativeDevice.maxAllocatorCount, nullptr);
}

void *LCDevice::stream_native_handle(uint64 handle) const noexcept {
    return reinterpret_cast<LCCmdBuffer *>(handle)
        ->queue.Queue();
}
void LCDevice::set_io_visitor(BinaryIO *visitor) noexcept {
    if (visitor) {
        nativeDevice.fileIo = visitor;
    } else {
        nativeDevice.fileIo = &nativeDevice.serVisitor;
    }
}

uint64 LCDevice::create_shader(Function kernel, std::string_view file_name) noexcept {
    auto code = CodegenUtility::Codegen(kernel, nativeDevice.fileIo);
    vstd::MD5 checkMD5({reinterpret_cast<uint8_t const *>(code.result.data() + code.immutableHeaderSize), code.result.size() - code.immutableHeaderSize});
    return reinterpret_cast<uint64>(
        ComputeShader::CompileCompute(
            nativeDevice.fileIo,
            &nativeDevice,
            kernel,
            [&]() { return std::move(code); },
            checkMD5,
            kernel.block_size(),
            kShaderModel,
            file_name,
            FileType::ByteCode));
}
uint64 LCDevice::create_shader(Function kernel, bool use_cache) noexcept {
    auto code = CodegenUtility::Codegen(kernel, nativeDevice.fileIo);
    vstd::string file_name;
    vstd::MD5 checkMD5({reinterpret_cast<uint8_t const *>(code.result.data() + code.immutableHeaderSize), code.result.size() - code.immutableHeaderSize});
    if (use_cache) {
        file_name << checkMD5.ToString(false) << ".dxil"sv;
    }
    return reinterpret_cast<uint64>(
        ComputeShader::CompileCompute(
            nativeDevice.fileIo,
            &nativeDevice,
            kernel,
            [&]() { return std::move(code); },
            checkMD5,
            kernel.block_size(),
            kShaderModel,
            file_name,
            FileType::Cache));
}
uint64 LCDevice::load_shader(
    vstd::string_view file_name,
    vstd::span<Type const *const> types) noexcept {
    auto cs = ComputeShader::LoadPresetCompute(
        nativeDevice.fileIo,
        &nativeDevice,
        types,
        file_name);
    if (cs)
        return reinterpret_cast<uint64>(cs);
    else
        return compute::Resource::invalid_handle;
}
void LCDevice::save_shader(Function kernel, string_view file_name) noexcept {
    auto code = CodegenUtility::Codegen(kernel, nativeDevice.fileIo);
    ComputeShader::SaveCompute(
        nativeDevice.fileIo,
        kernel,
        code,
        kernel.block_size(),
        kShaderModel,
        file_name);
}
void LCDevice::destroy_shader(uint64 handle) noexcept {
    auto shader = reinterpret_cast<Shader *>(handle);
    delete shader;
}
uint64 LCDevice::create_event() noexcept {
    return reinterpret_cast<uint64>(
        new LCEvent(&nativeDevice));
}
void LCDevice::destroy_event(uint64 handle) noexcept {
    delete reinterpret_cast<LCEvent *>(handle);
}
void LCDevice::signal_event(uint64 handle, uint64 stream_handle) noexcept {
    reinterpret_cast<LCEvent *>(handle)->Signal(
        &reinterpret_cast<LCCmdBuffer *>(stream_handle)->queue);
}

void LCDevice::wait_event(uint64 handle, uint64 stream_handle) noexcept {
    reinterpret_cast<LCEvent *>(handle)->Wait(
        &reinterpret_cast<LCCmdBuffer *>(stream_handle)->queue);
}
void LCDevice::synchronize_event(uint64 handle) noexcept {
    reinterpret_cast<LCEvent *>(handle)->Sync();
}

uint64 LCDevice::create_mesh(
    AccelUsageHint hint,
    MeshType type,
    bool allow_compact, bool allow_update) noexcept {
    return reinterpret_cast<uint64>((new BottomAccel(
        &nativeDevice,
        hint,
        allow_compact,
        allow_update)));
}
void LCDevice::destroy_mesh(uint64 handle) noexcept {
    delete reinterpret_cast<BottomAccel *>(handle);
}
uint64 LCDevice::create_accel(AccelUsageHint hint, bool allow_compact, bool allow_update) noexcept {
    return reinterpret_cast<uint64>(new TopAccel(
        &nativeDevice,
        hint,
        allow_compact,
        allow_update));
}
void LCDevice::destroy_accel(uint64 handle) noexcept {
    delete reinterpret_cast<TopAccel *>(handle);
}
uint64 LCDevice::create_swap_chain(
    uint64 window_handle,
    uint64 stream_handle,
    uint width,
    uint height,
    bool allow_hdr,
    bool vsync,
    uint back_buffer_size) noexcept {
    return reinterpret_cast<uint64>(
        new LCSwapChain(
            &nativeDevice,
            &reinterpret_cast<LCCmdBuffer *>(stream_handle)->queue,
            nativeDevice.defaultAllocator.get(),
            reinterpret_cast<HWND>(window_handle),
            width,
            height,
            allow_hdr,
            vsync,
            back_buffer_size));
}
void LCDevice::destroy_swap_chain(uint64 handle) noexcept {
    delete reinterpret_cast<LCSwapChain *>(handle);
}
PixelStorage LCDevice::swap_chain_pixel_storage(uint64 handle) noexcept {
    return PixelStorage::BYTE4;
}
void LCDevice::present_display_in_stream(uint64 stream_handle, uint64 swapchain_handle, uint64 image_handle) noexcept {
    reinterpret_cast<LCCmdBuffer *>(stream_handle)
        ->Present(
            reinterpret_cast<LCSwapChain *>(swapchain_handle),
            reinterpret_cast<TextureBase *>(image_handle), nativeDevice.maxAllocatorCount);
}

uint64_t LCDevice::create_raster_shader(
    const MeshFormat &mesh_format,
    const RasterState &raster_state,
    span<const PixelFormat> rtv_format,
    DepthFormat dsv_format,
    Function vert,
    Function pixel,
    string_view file_name) noexcept {
    auto code = CodegenUtility::RasterCodegen(mesh_format, vert, pixel, nativeDevice.fileIo);
    vstd::MD5 checkMD5({reinterpret_cast<uint8_t const *>(code.result.data() + code.immutableHeaderSize), code.result.size() - code.immutableHeaderSize});
    auto shader = RasterShader::CompileRaster(
        nativeDevice.fileIo,
        &nativeDevice,
        vert,
        pixel,
        [&] { return std::move(code); },
        checkMD5,
        kShaderModel,
        mesh_format,
        raster_state,
        rtv_format,
        dsv_format,
        file_name,
        FileType::Cache);
    return reinterpret_cast<uint64>(shader);
}

uint64_t LCDevice::load_raster_shader(
    const MeshFormat &mesh_format,
    const RasterState &raster_state,
    span<const PixelFormat> rtv_format,
    DepthFormat dsv_format,
    span<Type const *const> types,
    string_view ser_path) noexcept {
    auto ptr = RasterShader::LoadRaster(
        nativeDevice.fileIo,
        &nativeDevice,
        mesh_format,
        raster_state,
        rtv_format,
        dsv_format,
        types,
        ser_path);
    if (ptr) return reinterpret_cast<uint64>(ptr);
    return compute::Resource::invalid_handle;
}
void LCDevice::save_raster_shader(
    const MeshFormat &mesh_format,
    Function vert,
    Function pixel,
    string_view file_name) noexcept {
    auto code = CodegenUtility::RasterCodegen(mesh_format, vert, pixel, nativeDevice.fileIo);
    vstd::MD5 checkMD5({reinterpret_cast<uint8_t const *>(code.result.data() + code.immutableHeaderSize), code.result.size() - code.immutableHeaderSize});
    RasterShader::SaveRaster(
        nativeDevice.fileIo,
        &nativeDevice,
        code,
        checkMD5,
        file_name,
        vert,
        pixel,
        kShaderModel);
}

uint64_t LCDevice::create_raster_shader(
    const MeshFormat &mesh_format,
    const RasterState &raster_state,
    span<const PixelFormat> rtv_format,
    DepthFormat dsv_format,
    Function vert,
    Function pixel,
    bool use_cache) noexcept {
    auto code = CodegenUtility::RasterCodegen(mesh_format, vert, pixel, nativeDevice.fileIo);
    vstd::string file_name;
    vstd::MD5 checkMD5({reinterpret_cast<uint8_t const *>(code.result.data() + code.immutableHeaderSize), code.result.size() - code.immutableHeaderSize});
    if (use_cache) {
        file_name << checkMD5.ToString(false) << ".dxil"sv;
    }
    auto shader = RasterShader::CompileRaster(
        nativeDevice.fileIo,
        &nativeDevice,
        vert,
        pixel,
        [&] { return std::move(code); },
        checkMD5,
        kShaderModel,
        mesh_format,
        raster_state,
        rtv_format,
        dsv_format,
        file_name,
        FileType::Cache);
    return reinterpret_cast<uint64>(shader);
}
uint64_t LCDevice::create_depth_buffer(DepthFormat format, uint width, uint height) noexcept {
    return reinterpret_cast<uint64_t>(
        static_cast<TextureBase *>(
            new DepthBuffer(
                &nativeDevice,
                width, height,
                format, nativeDevice.defaultAllocator.get())));
}
DeviceInterface::BuiltinBuffer LCDevice::create_dispatch_buffer(uint32_t dimension, size_t capacity) noexcept {
    auto size = 4 + 28 * capacity;
    return {reinterpret_cast<uint64>(static_cast<Buffer *>(new DefaultBuffer(&nativeDevice, size, nativeDevice.defaultAllocator.get()))), size};
}
DeviceInterface::BuiltinBuffer LCDevice::create_aabb_buffer(size_t capacity) noexcept {
    auto size = capacity * 32;
    return {reinterpret_cast<uint64>(static_cast<Buffer *>(new DefaultBuffer(&nativeDevice, size, nativeDevice.defaultAllocator.get()))), size};
}
void LCDevice::destroy_depth_buffer(uint64_t handle) noexcept {
    delete reinterpret_cast<TextureBase *>(handle);
}
DeviceExtension *LCDevice::extension(vstd::string_view name) noexcept {
    auto ite = exts.find(name);
    if (ite == exts.end()) return nullptr;
    auto &v = ite->second;
    {
        std::lock_guard lck{extMtx};
        if (v.ext == nullptr) {
            v.ext = vstd::create_unique(v.get_ext(this));
        }
    }
    return v.ext.get();
}
void *LCDevice::swapchain_native_handle(uint64_t handle) const noexcept {
    auto swapchain = reinterpret_cast<LCSwapChain *>(handle);
    return swapchain->swapChain.Get();
}
void *LCDevice::bindless_native_handle(uint64_t handle) const noexcept {
    auto bindless = reinterpret_cast<BindlessArray *>(handle);
    return bindless->Buffer()->GetResource();
}
void *LCDevice::depth_native_handle(uint64_t handle) const noexcept {
    auto db = static_cast<DepthBuffer *>(reinterpret_cast<TextureBase *>(handle));
    return db->GetResource();
}
void *LCDevice::event_native_handle(uint64_t handle) const noexcept {
    auto evt = reinterpret_cast<LCEvent *>(handle);
    return evt->Fence();
}
void *LCDevice::mesh_native_handle(uint64_t handle) const noexcept {
    auto mesh = reinterpret_cast<BottomAccel *>(handle);
    return mesh->GetAccelBuffer()->GetResource();
}
void *LCDevice::accel_native_handle(uint64_t handle) const noexcept {
    auto accel = reinterpret_cast<TopAccel *>(handle);
    return accel->GetAccelBuffer()->GetResource();
}

VSTL_EXPORT_C DeviceInterface *create(Context &&c, DeviceConfig const *settings) {
    return new LCDevice(std::move(c), settings);
}
VSTL_EXPORT_C void destroy(DeviceInterface *device) {
    delete static_cast<LCDevice *>(device);
}
}// namespace toolhub::directx