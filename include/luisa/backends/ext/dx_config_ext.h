#pragma once

#include <luisa/runtime/context.h>
#ifdef byte
#undef byte
#endif
#ifdef LUISA_DX_SDK
#include <LCAgilitySDK/d3d12.h>
#else
#include <d3d12.h>
#endif
#include <dxgi1_2.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/backends/ext/dx_custom_cmd.h>

struct IDxcCompiler3;
struct IDxcLibrary;
struct IDxcUtils;
namespace luisa::compute {
struct DirectXHeap {
    uint64_t handle{};
    ID3D12Heap *heap{};
    size_t offset{};
};
class DirectXFuncTable {
public:
    [[nodiscard]] virtual DirectXHeap allocate_buffer_heap(
        luisa::string_view name,
        uint64_t sizeBytes,
        D3D12_HEAP_TYPE heapType,
        D3D12_HEAP_FLAGS extraFlags) const noexcept = 0;
    [[nodiscard]] virtual DirectXHeap allocate_texture_heap(
        luisa::string_view name,
        size_t sizeBytes,
        bool isRenderTexture,
        D3D12_HEAP_FLAGS extraFlags) const noexcept = 0;
    virtual void deallocate_heap(uint64_t handle) const noexcept = 0;
};
struct DirectXDeviceConfigExt : public DeviceConfigExt {

    struct ExternalDevice {
        ID3D12Device *device;
        IDXGIAdapter1 *adapter;
        IDXGIFactory2 *factory;
    };

    struct GPUAllocatorSettings {
        size_t preferred_block_size;
        size_t sparse_buffer_block_size;
        size_t sparse_image_block_size;
    };

    [[nodiscard]] virtual luisa::optional<ExternalDevice> CreateExternalDevice() noexcept { return {}; }
    [[nodiscard]] virtual luisa::optional<GPUAllocatorSettings> GetGPUAllocatorSettings() noexcept { return {}; }
    virtual bool UseDRED() const noexcept { return false; }
    virtual bool LoadDXC() const noexcept { return true; }
    virtual bool UseEnhancedBarrier() const noexcept { return false; }
    virtual bool UseExperimental() const noexcept { return false; }

    // Return a custom Agility SDK version (0 = use default / D3D12_PREVIEW_SDK_VERSION)
    [[nodiscard]] virtual uint32_t GetSDKVersion() const noexcept { return 0u; }

    // Return a custom Agility SDK DLL path (empty = use default ".\\D3D12\\")
    // The path should point to the directory containing D3D12Core.dll
    [[nodiscard]] virtual luisa::string_view GetSDKPath() const noexcept { return {}; }

    // Return true to use system <d3d12.h> instead of bundled LCAgilitySDK headers
    [[nodiscard]] virtual bool UseSystemD3D12Headers() const noexcept { return false; }

    // Called during create_device
    virtual void ReadbackDX12Device(
        ID3D12Device *device,
        IDXGIAdapter1 *adapter,
        IDXGIFactory2 *factory,
        DirectXFuncTable const *funcTable,
        luisa::BinaryIO const *shaderIo,
        IDxcCompiler3 *dxcCompiler,
        IDxcLibrary *dxcLibrary,
        IDxcUtils *dxcUtils,
        ID3D12DescriptorHeap *shaderDescriptor,
        ID3D12DescriptorHeap *samplerDescriptor) noexcept {}
    virtual void GetDefragmentFunction(luisa::move_only_function<void()> &&defragment_func) {}
    // plugin resources
    virtual ID3D12CommandQueue *CreateQueue(D3D12_COMMAND_LIST_TYPE type) noexcept { return nullptr; }

    virtual ID3D12GraphicsCommandList *BorrowCommandList(D3D12_COMMAND_LIST_TYPE type) noexcept { return nullptr; }

    // Custom callback
    // return true if this callback is implemented
    virtual bool ExecuteCommandList(
        ID3D12CommandQueue *queue,
        ID3D12GraphicsCommandList *cmdList) noexcept { return false; }

    virtual bool SignalFence(
        ID3D12CommandQueue *queue,
        ID3D12Fence *fence, uint64_t fenceIndex) noexcept { return false; }
    virtual bool WaitFence(
        ID3D12CommandQueue *queue,
        ID3D12Fence *fence, uint64_t fenceIndex) noexcept { return false; }
    virtual bool SyncFence(ID3D12Fence *fence, uint64_t fenceIndex) noexcept { return false; }
    virtual ~DirectXDeviceConfigExt() noexcept override = default;
    [[nodiscard]] virtual luisa::span<DXCustomCmd::EnhancedResourceUsage const> before_states(uint64_t stream_handle) noexcept { return {}; }
    [[nodiscard]] virtual luisa::span<DXCustomCmd::EnhancedResourceUsage const> after_states(uint64_t stream_handle) noexcept { return {}; }

    // Optional feedback from Device creation about whether the requested
    // experimental features (e.g. cooperative vectors) could be enabled.
    virtual void SetExperimentalFeaturesEnabled(bool value) noexcept {
        _experimental_features_enabled = value;
    }
    [[nodiscard]] bool ExperimentalFeaturesEnabled() const noexcept {
        return _experimental_features_enabled;
    }

protected:
    bool _experimental_features_enabled = false;
};

}// namespace luisa::compute
#ifdef byte
#undef byte
#endif