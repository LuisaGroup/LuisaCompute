#pragma once
#include <d3dx12.h>
#include <Resource/BufferView.h>
#include <vstl/v_guid.h>
#include <vstl/md5.h>
#include <dxgi1_3.h>
#include <core/binary_io.h>
#include <runtime/device.h>
#include <DXRuntime/DxPtr.h>
#include <ext_settings.h>
#include <backends/common/default_binary_io.h>

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
class Device {
public:
    size_t maxAllocatorCount = 2;
    luisa::BinaryIO const *fileIo = nullptr;
    struct LazyLoadShader {
    public:
        using LoadFunc = vstd::func_ptr_t<ComputeShader *(Device *, luisa::BinaryIO const *)>;

    private:
        vstd::unique_ptr<ComputeShader> shader;
        LoadFunc loadFunc;

    public:
        LazyLoadShader(LoadFunc loadFunc);
        ComputeShader *Get(Device *self);
        bool Check(Device *self);
        ~LazyLoadShader();
    };
    vstd::unique_ptr<luisa::compute::DefaultBinaryIO> serVisitor;
    bool SupportMeshShader() const;
    vstd::MD5 adapterID;
    DxPtr<IDXGIAdapter1> adapter;
    DxPtr<ID3D12Device5> device;
    DxPtr<IDXGIFactory2> dxgiFactory;
    vstd::unique_ptr<GpuAllocator> defaultAllocator;
    vstd::unique_ptr<DescriptorHeap> globalHeap;
    vstd::unique_ptr<DescriptorHeap> samplerHeap;
    vstd::unique_ptr<luisa::compute::DirectXDeviceConfigExt> deviceSettings;
    LazyLoadShader setAccelKernel;

    LazyLoadShader bc6TryModeG10;
    LazyLoadShader bc6TryModeLE10;
    LazyLoadShader bc6EncodeBlock;

    LazyLoadShader bc7TryMode456;
    LazyLoadShader bc7TryMode137;
    LazyLoadShader bc7TryMode02;
    LazyLoadShader bc7EncodeBlock;

    /*vstd::unique_ptr<ComputeShader> bc6_0;
    vstd::unique_ptr<ComputeShader> bc6_1;
    vstd::unique_ptr<ComputeShader> bc6_2;
    vstd::unique_ptr<ComputeShader> bc7_0;
    vstd::unique_ptr<ComputeShader> bc7_1;
    vstd::unique_ptr<ComputeShader> bc7_2;
    vstd::unique_ptr<ComputeShader> bc7_3;*/
    Device(luisa::compute::Context &&ctx, luisa::compute::DeviceConfig const *settings);
    Device(Device const &) = delete;
    Device(Device &&) = delete;
    ~Device();
    void WaitFence(ID3D12Fence *fence, uint64 fenceIndex);
    static hlsl::ShaderCompiler *Compiler();
};
}// namespace lc::dx