#pragma vengine_package vengine_directx
#include <Api/LCDevice.h>
#include <DXRuntime/Device.h>
#include <Resource/DefaultBuffer.h>
#include <Codegen/DxCodegen.h>
#include <Shader/ShaderCompiler.h>
#include <Codegen/ShaderHeader.h>
#include <Resource/RenderTexture.h>
#include <Resource/BindlessArray.h>
#include <Shader/ComputeShader.h>
#include <Api/LCCmdBuffer.h>
#include <Api/LCEvent.h>
#include <vstl/MD5.h>
#include <Shader/ShaderSerializer.h>
#include <Resource/BottomAccel.h>
#include <Shader/PipelineLibrary.h>
#include <Resource/TopAccel.h>
#include <vstl/BinaryReader.h>
using namespace toolhub::directx;
namespace toolhub::directx {
LCDevice::LCDevice(const Context &ctx)
    : LCDeviceInterface(ctx) {
}
void *LCDevice::native_handle() const noexcept {
    return nativeDevice.device.Get();
}
uint64_t LCDevice::create_buffer(size_t size_bytes) noexcept {
    return reinterpret_cast<uint64>(
        static_cast<Buffer *>(
            new DefaultBuffer(
                &nativeDevice,
                size_bytes,
                nativeDevice.defaultAllocator)));
}
void LCDevice::destroy_buffer(uint64_t handle) noexcept {
    delete reinterpret_cast<Buffer *>(handle);
}
void *LCDevice::buffer_native_handle(uint64_t handle) const noexcept {
    return reinterpret_cast<Buffer *>(handle)->GetResource();
}
uint64_t LCDevice::create_texture(
    PixelFormat format,
    uint dimension,
    uint width,
    uint height,
    uint depth,
    uint mipmap_levels) noexcept {
    return reinterpret_cast<uint64>(
        new RenderTexture(
            &nativeDevice,
            width,
            height,
            TextureBase::ToGFXFormat(format),
            (TextureDimension)dimension,
            depth,
            mipmap_levels,
            true,
            nativeDevice.defaultAllocator));
}
void LCDevice::destroy_texture(uint64_t handle) noexcept {
    delete reinterpret_cast<RenderTexture *>(handle);
}
void *LCDevice::texture_native_handle(uint64_t handle) const noexcept {
    return reinterpret_cast<RenderTexture *>(handle)->GetResource();
}
uint64_t LCDevice::create_bindless_array(size_t size) noexcept {
    return reinterpret_cast<uint64>(
        new BindlessArray(&nativeDevice, size));
}
void LCDevice::destroy_bindless_array(uint64_t handle) noexcept {
    delete reinterpret_cast<BindlessArray *>(handle);
}
void LCDevice::emplace_buffer_in_bindless_array(uint64_t array, size_t index, uint64_t handle, size_t offset_bytes) noexcept {
    auto buffer = reinterpret_cast<Buffer *>(handle);
    reinterpret_cast<BindlessArray *>(array)
        ->Bind(BufferView(buffer, offset_bytes), index);
}
void LCDevice::emplace_tex2d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept {
    auto tex = reinterpret_cast<RenderTexture *>(handle);
    reinterpret_cast<BindlessArray *>(array)
        ->Bind(std::pair<TextureBase const *, Sampler>(tex, sampler), index);
}
void LCDevice::emplace_tex3d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept {
    emplace_tex2d_in_bindless_array(array, index, handle, sampler);
}
bool LCDevice::is_buffer_in_bindless_array(uint64_t array, uint64_t handle) const noexcept {
    return reinterpret_cast<BindlessArray *>(array)
        ->IsPtrInBindless(handle);
}
bool LCDevice::is_texture_in_bindless_array(uint64_t array, uint64_t handle) const noexcept {
    return reinterpret_cast<BindlessArray *>(array)
        ->IsPtrInBindless(handle);
}
void LCDevice::remove_buffer_in_bindless_array(uint64_t array, size_t index) noexcept {
    reinterpret_cast<BindlessArray *>(array)
        ->UnBind(BindlessArray::BindTag::Buffer, index);
}
void LCDevice::remove_tex2d_in_bindless_array(uint64_t array, size_t index) noexcept {
    reinterpret_cast<BindlessArray *>(array)
        ->UnBind(BindlessArray::BindTag::Tex2D, index);
}
void LCDevice::remove_tex3d_in_bindless_array(uint64_t array, size_t index) noexcept {
    reinterpret_cast<BindlessArray *>(array)
        ->UnBind(BindlessArray::BindTag::Tex3D, index);
}
uint64_t LCDevice::create_stream() noexcept {
    return reinterpret_cast<uint64>(
        new LCCmdBuffer(
            &nativeDevice,
            nativeDevice.defaultAllocator,
            D3D12_COMMAND_LIST_TYPE_COMPUTE));
}
void LCDevice::destroy_stream(uint64_t handle) noexcept {
    delete reinterpret_cast<LCCmdBuffer *>(handle);
}
void LCDevice::synchronize_stream(uint64_t stream_handle) noexcept {
    reinterpret_cast<LCCmdBuffer *>(stream_handle)->Sync();
}
void LCDevice::dispatch(uint64_t stream_handle, CommandList const &v) noexcept {
    reinterpret_cast<LCCmdBuffer *>(stream_handle)
        ->Execute({&v, 1}, maxAllocatorCount);
}
void LCDevice::dispatch(uint64_t stream_handle, luisa::span<const CommandList> lists) noexcept {
    reinterpret_cast<LCCmdBuffer *>(stream_handle)
        ->Execute(lists, maxAllocatorCount);
}

void *LCDevice::stream_native_handle(uint64_t handle) const noexcept {
    return reinterpret_cast<LCCmdBuffer *>(handle)
        ->queue.Queue();
}
uint64_t LCDevice::create_shader(Function kernel, std::string_view meta_options) noexcept {
    return create_shader(kernel, meta_options, 0);
}

uint64_t LCDevice::create_shader(Function kernel, std::string_view meta_options, uint64_t psolib) noexcept {

    auto str = CodegenUtility::Codegen(kernel);
    if (str) {
        return reinterpret_cast<uint64_t>(
            ComputeShader::CompileCompute(
                &nativeDevice,
                *str,
                kernel.block_size(),
                kernel.raytracing() ? 65u : 60u));
    }
    return 0;
}
void LCDevice::destroy_shader(uint64_t handle) noexcept {
    auto shader = reinterpret_cast<Shader *>(handle);
    delete shader;
}
uint64_t LCDevice::create_event() noexcept {
    return reinterpret_cast<uint64>(
        new LCEvent(&nativeDevice));
}
void LCDevice::destroy_event(uint64_t handle) noexcept {
    delete reinterpret_cast<LCEvent *>(handle);
}
void LCDevice::signal_event(uint64_t handle, uint64_t stream_handle) noexcept {
    reinterpret_cast<LCEvent *>(handle)->Signal(
        &reinterpret_cast<LCCmdBuffer *>(stream_handle)->queue);
}
void LCDevice::wait_event(uint64_t handle, uint64_t stream_handle) noexcept {
    reinterpret_cast<LCEvent *>(handle)->Wait(
        &reinterpret_cast<LCCmdBuffer *>(stream_handle)->queue);
}
void LCDevice::synchronize_event(uint64_t handle) noexcept {
    reinterpret_cast<LCEvent *>(handle)->Sync();
}
uint64_t LCDevice::create_mesh(
    uint64_t v_buffer,
    size_t v_offset,
    size_t v_stride,
    size_t v_count,
    uint64_t t_buffer,
    size_t t_offset,
    size_t t_count,
    AccelBuildHint hint) noexcept {
    return reinterpret_cast<uint64>(
        new BottomAccel(
            &nativeDevice,
            reinterpret_cast<Buffer *>(v_buffer),
            v_offset * v_stride,
            v_stride,
            v_count,
            reinterpret_cast<Buffer *>(t_buffer),
            t_offset * 3 * sizeof(uint),
            t_count * 3));
}
void LCDevice::destroy_mesh(uint64_t handle) noexcept {
    delete reinterpret_cast<BottomAccel *>(handle);
}
uint64_t LCDevice::create_accel(AccelBuildHint hint) noexcept {
    return reinterpret_cast<uint64>(new TopAccel(
        &nativeDevice,
        hint));
}
void LCDevice::emplace_back_instance_in_accel(uint64_t accel, uint64_t mesh, luisa::float4x4 transform, bool visible) noexcept {
    auto topAccel = reinterpret_cast<TopAccel *>(accel);
    auto bottomAccel = reinterpret_cast<BottomAccel *>(mesh);
    topAccel->Emplace(
        bottomAccel,
        visible ? std::numeric_limits<uint>::max() : 0,
        transform);
}
void LCDevice::pop_back_instance_from_accel(uint64_t accel) noexcept {
    auto topAccel = reinterpret_cast<TopAccel *>(accel);
    topAccel->PopBack();
}
void LCDevice::set_instance_in_accel(uint64_t accel, size_t index, uint64_t mesh, luisa::float4x4 transform, bool visible) noexcept {
    auto topAccel = reinterpret_cast<TopAccel *>(accel);
    topAccel->Update(
        index,
        reinterpret_cast<BottomAccel *>(mesh),
        visible ? std::numeric_limits<uint>::max() : 0,
        transform);
}
void LCDevice::set_instance_transform_in_accel(uint64_t accel, size_t index, luisa::float4x4 transform) noexcept {
    auto topAccel = reinterpret_cast<TopAccel *>(accel);
    topAccel->Update(
        index,
        transform);
}
void LCDevice::set_instance_visibility_in_accel(uint64_t accel, size_t index, bool visible) noexcept {
    auto topAccel = reinterpret_cast<TopAccel *>(accel);
    topAccel->Update(
        index,
        visible ? std::numeric_limits<uint>::max() : 0);
}
bool LCDevice::is_buffer_in_accel(uint64_t accel, uint64_t buffer) const noexcept {
    auto topAccel = reinterpret_cast<TopAccel *>(accel);
    return topAccel->IsBufferInAccel(reinterpret_cast<Buffer *>(buffer));
}
bool LCDevice::is_mesh_in_accel(uint64_t accel, uint64_t mesh) const noexcept {
    auto topAccel = reinterpret_cast<TopAccel *>(accel);
    return topAccel->IsMeshInAccel(reinterpret_cast<BottomAccel *>(mesh)->GetMesh());
}
uint64_t LCDevice::get_vertex_buffer_from_mesh(uint64_t mesh_handle) const noexcept {
    return reinterpret_cast<uint64>(reinterpret_cast<BottomAccel *>(mesh_handle)->GetMesh()->vHandle);
}
uint64_t LCDevice::get_triangle_buffer_from_mesh(uint64_t mesh_handle) const noexcept {
    return reinterpret_cast<uint64>(reinterpret_cast<BottomAccel *>(mesh_handle)->GetMesh()->iHandle);
}
void LCDevice::destroy_accel(uint64_t handle) noexcept {
    delete reinterpret_cast<TopAccel *>(handle);
}

uint64_t LCDevice::create_psolib(eastl::span<uint64_t> shaders) noexcept {
    auto sp = vstd::span<ComputeShader const *>(reinterpret_cast<ComputeShader const **>(shaders.data()), shaders.size());
    return reinterpret_cast<uint64>(
        new PipelineLibrary(&nativeDevice, sp));
}
void LCDevice::destroy_psolib(uint64_t lib_handle) noexcept {
    delete reinterpret_cast<PipelineLibrary *>(lib_handle);
}
bool LCDevice::deser_psolib(uint64_t lib_handle, eastl::span<std::byte const> data) noexcept {
    auto psoLib = reinterpret_cast<PipelineLibrary *>(lib_handle);
    return psoLib->Deserialize(vstd::span<vbyte const>(
        reinterpret_cast<vbyte const *>(data.data()),
        data.size()));
}
size_t LCDevice::ser_psolib(uint64_t lib_handle, eastl::vector<std::byte> &result) noexcept {
    auto psoLib = reinterpret_cast<PipelineLibrary *>(lib_handle);
    auto retSize = 0;
    psoLib->Serialize(
        [&](size_t sz) -> void * {
            retSize = sz;
            auto lastSize = result.size();
            result.resize(lastSize + sz);
            return result.data() + lastSize;
        });
    return retSize;
}

VSTL_EXPORT_C LCDeviceInterface *create(Context const &c, std::string_view) {
    return new LCDevice(c);
}
VSTL_EXPORT_C void destroy(LCDeviceInterface *device) {
    delete static_cast<LCDevice *>(device);
}
}// namespace toolhub::directx