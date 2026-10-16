#pragma once
#include <d3dx12.h>
#include <Resource/BufferView.h>
#include <luisa/vstl/v_guid.h>
#include <luisa/vstl/md5.h>
#include <dxgi1_3.h>
#include <luisa/core/binary_io.h>
#include <luisa/runtime/device.h>
#include <DXRuntime/DxPtr.h>
#include "../../common/default_binary_io.h"
#include "../../common/command_reorder_switch.h"
#include <luisa/backends/ext/dx_config_ext.h>
#include <Resource/FeatureCheck.h>

namespace luisa {
class BinaryIO;
}// namespace luisa
namespace lc::hlsl {
class ShaderCompiler;
}// namespace lc::hlsl
namespace luisa::compute {
class Context;
}// namespace luisa::compute
class ElementAllocator;
using Microsoft::WRL::ComPtr;
namespace lc::dx {
class GpuAllocator;
class DescriptorHeap;
class ComputeShader;
class PipelineLibrary;
class Device;
class DXAllocatorImpl : public luisa::compute::DirectXFuncTable {
public:
    Device *device;
    luisa::compute::DirectXHeap allocate_buffer_heap(
        luisa::string_view name,
        uint64_t target_size_in_bytes,
        D3D12_HEAP_TYPE heap_type,
        D3D12_HEAP_FLAGS extra_flags) const noexcept override;
    luisa::compute::DirectXHeap allocate_texture_heap(
        vstd::string_view name,
        size_t size_bytes,
        bool is_render_texture,
        D3D12_HEAP_FLAGS extra_flags) const noexcept override;
    void deallocate_heap(uint64_t handle) const noexcept override;
};
class Device {
public:
    enum class GpuType {
        OTHER,
        AMD,
        INTEL,
        NVIDIA
    };
    GpuType gpu_type = GpuType::OTHER;
    size_t max_allocator_count = 2;
    luisa::BinaryIO const *file_io = nullptr;
    luisa::compute::Profiler *profiler = nullptr;
    struct LazyLoadShader {
    public:
        using LoadFunc = vstd::func_ptr_t<ComputeShader *(Device *)>;

    private:
        vstd::unique_ptr<ComputeShader> _shader;
        LoadFunc _load_func;

    public:
        LazyLoadShader(LoadFunc load_func);
        ComputeShader *get(Device *self);
        bool check(Device *self);
        ~LazyLoadShader();
    };
    vstd::unique_ptr<luisa::compute::DefaultBinaryIO> ser_visitor;
    vstd::unique_ptr<luisa::compute::DirectXDeviceConfigExt> device_settings;
    bool support_mesh_shader() const;
    vstd::MD5 adapter_id;
    DxPtr<IDXGIAdapter1> adapter;
    DxPtr<ID3D12Device5> device;
    DxPtr<IDXGIFactory2> dxgi_factory;
    vstd::unique_ptr<GpuAllocator> default_allocator;
    DXAllocatorImpl allocator_interface;

    vstd::unique_ptr<DescriptorHeap> global_heap;
    vstd::unique_ptr<DescriptorHeap> sampler_heap;
    LazyLoadShader set_bindless_kernel;
    LazyLoadShader set_accel_kernel;

    LazyLoadShader bc6_try_mode_g10;
    LazyLoadShader bc6_try_mode_le10;
    LazyLoadShader bc6_encode_block;

    LazyLoadShader bc7_try_mode_456;
    LazyLoadShader bc7_try_mode_137;
    LazyLoadShader bc7_try_mode_02;
    LazyLoadShader bc7_encode_block;

    /*vstd::unique_ptr<ComputeShader> bc6_0;
    vstd::unique_ptr<ComputeShader> bc6_1;
    vstd::unique_ptr<ComputeShader> bc6_2;
    vstd::unique_ptr<ComputeShader> bc7_0;
    vstd::unique_ptr<ComputeShader> bc7_1;
    vstd::unique_ptr<ComputeShader> bc7_2;
    vstd::unique_ptr<ComputeShader> bc7_3;*/
    FeatureCheck feature_check;
    Device(luisa::compute::Context &&ctx, luisa::compute::DeviceConfig const *settings);
    Device(Device const &) = delete;
    Device(Device &&) = delete;
    ~Device();
    void wait_fence(ID3D12Fence *fence, uint64 fenceIndex);
    static hlsl::ShaderCompiler *compiler();
    uint wave_size() const;
    // Command-reorder switch sampled when a batch starts (see
    // CommandReorderSwitch for the precedence of the device settings extension,
    // the runtime CommandReorderExt override and LUISA_DISABLE_COMMAND_REORDER).
    // LCCmdBuffer forwards it to its CommandReorderVisitor once per batch.
    [[nodiscard]] bool command_reorder_enabled() const noexcept {
        return _command_reorder.enabled();
    }
    [[nodiscard]] luisa::compute::CommandReorderSwitch &command_reorder_switch() noexcept {
        return _command_reorder;
    }

private:
    luisa::compute::CommandReorderSwitch _command_reorder;
};
}// namespace lc::dx
