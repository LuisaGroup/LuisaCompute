#include <DXRuntime/Device.h>
#include <Resource/DescriptorHeap.h>
#include <Resource/DefaultBuffer.h>
#include <DXRuntime/GlobalSamplers.h>
#include <Resource/GpuAllocator.h>
#include <Shader/BuiltinKernel.h>
#include "../../common/hlsl/shader_compiler.h"
#include <Shader/ComputeShader.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/backends/ext/dx_config_ext.h>

namespace lc::dx {
static std::mutex gDxcMutex;
static vstd::optional<hlsl::ShaderCompiler> gDxcCompiler;
static int32 gDxcRefCount = 0;

Device::LazyLoadShader::~LazyLoadShader() {}
VSTL_EXPORT_C void backend_device_names(luisa::vector<luisa::string> &r);
Device::LazyLoadShader::LazyLoadShader(LoadFunc loadFunc) : loadFunc(loadFunc) {}
Device::~Device() {
    //lcmdSig.destroy();
    std::lock_guard lck(gDxcMutex);
    if (--gDxcRefCount == 0) {
        gDxcCompiler.destroy();
    }
}

void Device::WaitFence(ID3D12Fence *fence, uint64 fenceIndex) {
    if (fenceIndex <= 0) return;
    HANDLE eventHandle = CreateEventEx(nullptr, nullptr, false, EVENT_ALL_ACCESS);
    auto d = vstd::scope_exit([&] {
        CloseHandle(eventHandle);
    });
    if (fence->GetCompletedValue() < fenceIndex) {
        ThrowIfFailed(fence->SetEventOnCompletion(fenceIndex, eventHandle));
        WaitForSingleObject(eventHandle, INFINITE);
    }
}
ComputeShader *Device::LazyLoadShader::Get(Device *self) {
    if (!shader) {
        shader = vstd::create_unique(loadFunc(self, self->fileIo));
    }
    return shader.get();
}
bool Device::LazyLoadShader::Check(Device *self) {
    if (shader) return true;
    shader = vstd::create_unique(loadFunc(self, self->fileIo));
    if (shader) {
        auto afterExit = vstd::scope_exit([&] { shader = nullptr; });
        return true;
    }
    return false;
}

hlsl::ShaderCompiler *Device::Compiler() {
    return gDxcCompiler;
}
Device::Device(Context &&ctx, DeviceConfig const *settings)
    : setBindlessKernel(BuiltinKernel::LoadBindlessSetKernel),
      setAccelKernel(BuiltinKernel::LoadAccelSetKernel),
      bc6TryModeG10(BuiltinKernel::LoadBC6TryModeG10CSKernel),
      bc6TryModeLE10(BuiltinKernel::LoadBC6TryModeLE10CSKernel),
      bc6EncodeBlock(BuiltinKernel::LoadBC6EncodeBlockCSKernel),
      bc7TryMode456(BuiltinKernel::LoadBC7TryMode456CSKernel),
      bc7TryMode137(BuiltinKernel::LoadBC7TryMode137CSKernel),
      bc7TryMode02(BuiltinKernel::LoadBC7TryMode02CSKernel),
      bc7EncodeBlock(BuiltinKernel::LoadBC7EncodeBlockCSKernel) {
    using Microsoft::WRL::ComPtr;
    size_t index{std::numeric_limits<size_t>::max()};
    bool useRuntime = true;
    {
        std::lock_guard lck(gDxcMutex);
        if (gDxcRefCount == 0)
            gDxcCompiler.create(ctx.runtime_directory());
        gDxcRefCount++;
    }
    if (settings) {
        index = settings->device_index;
        // auto select
        useRuntime = !settings->headless;
        maxAllocatorCount = settings->inqueue_buffer_limit ? 2 : std::numeric_limits<size_t>::max();
        fileIo = settings->binary_io;
    }
    if (fileIo == nullptr) {
        serVisitor = vstd::make_unique<DefaultBinaryIO>(std::move(ctx), device.Get());
        fileIo = serVisitor.get();
    }
    if (useRuntime) {
        auto GenAdapterGUID = [](DXGI_ADAPTER_DESC1 const &desc) {
            struct AdapterInfo {
                WCHAR Description[128];
                UINT VendorId;
                UINT DeviceId;
                UINT SubSysId;
                UINT Revision;
            };
            AdapterInfo info;
            memcpy(info.Description, desc.Description, sizeof(WCHAR) * 128);
            info.VendorId = desc.VendorId;
            info.DeviceId = desc.DeviceId;
            info.SubSysId = desc.SubSysId;
            info.Revision = desc.Revision;
            return vstd::MD5{vstd::span<uint8_t const>{reinterpret_cast<uint8_t const *>(&info), sizeof(AdapterInfo)}};
        };
        if (settings && settings->extension) {
            deviceSettings = vstd::create_unique(static_cast<DirectXDeviceConfigExt *>(settings->extension.release()));
        }
        vstd::optional<DirectXDeviceConfigExt::ExternalDevice> extDevice;
        if (deviceSettings) {
            extDevice = deviceSettings->CreateExternalDevice();
        }
        if (extDevice) {
            device = {static_cast<ID3D12Device5 *>(extDevice->device), false};
            adapter = {extDevice->adapter, false};
            dxgiFactory = {extDevice->factory, false};
            DXGI_ADAPTER_DESC1 desc;
            adapter->GetDesc1(&desc);
            adapterID = GenAdapterGUID(desc);
        } else {
            uint32_t dxgiFactoryFlags = 0;
#ifndef NDEBUG
            // Enable the debug layer (requires the Graphics Tools "optional feature").
            // NOTE: Enabling the debug layer after device creation will invalidate the active device.
            {
                ComPtr<ID3D12Debug> debugController;
                if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
                    debugController->EnableDebugLayer();

                    // Enable additional debug layers.
                    dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
                }
            }
#endif
            ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(dxgiFactory.GetAddressOf())));
            if (index == std::numeric_limits<size_t>::max()) {
                luisa::vector<luisa::string> device_names;
                backend_device_names(device_names);
                index = 0;
                for (size_t i = 0; i < device_names.size(); ++i) {
                    luisa::string &device_name = device_names[i];
                    if (device_name.find("GeForce") != luisa::string::npos ||
                        device_name.find("Radeon RX") != luisa::string::npos ||
                        device_name.find("Arc") != luisa::string::npos) {
                        LUISA_INFO("Select device: {}", device_name);
                        index = i;
                        break;
                    }
                }
            }
            auto capableAdapterIndex = 0u;
            for (auto adapterIndex = 0u; dxgiFactory->EnumAdapters1(adapterIndex, adapter.GetAddressOf()) != DXGI_ERROR_NOT_FOUND; adapterIndex++) {
                DXGI_ADAPTER_DESC1 desc;
                adapter->GetDesc1(&desc);
                if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
                    if (capableAdapterIndex++ == index) {
                        ThrowIfFailed(D3D12CreateDevice(
                            adapter.Get(), D3D_FEATURE_LEVEL_12_1,
                            IID_PPV_ARGS(device.GetAddressOf())));
                        adapterID = GenAdapterGUID(desc);
                        break;
                    }
                }
                device.Clear();
                adapter.Clear();
            }
            if (adapter == nullptr) { LUISA_ERROR_WITH_LOCATION("Failed to create DirectX device at index {}.", index); }
        }
        defaultAllocator = vstd::make_unique<GpuAllocator>(this, settings ? settings->memory_profiler : nullptr);
        globalHeap = vstd::create_unique(
            new DescriptorHeap(
                this,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                524288,
                true));
        samplerHeap = vstd::create_unique(
            new DescriptorHeap(
                this,
                D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                16,
                true));
        hdr.create(dxgiFactory.Get(), adapter.Get());
        auto samplers = GlobalSamplers::GetSamplers();
        for (auto i : vstd::range(samplers.size())) {
            samplerHeap->CreateSampler(
                samplers[i], i);
        }
        if (deviceSettings) {
            deviceSettings->ReadbackDX12Device(
                device,
                adapter,
                dxgiFactory);
        }
    }
}
bool Device::SupportMeshShader() const {
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 featureData = {};
    device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &featureData, sizeof(featureData));
    return (featureData.MeshShaderTier >= D3D12_MESH_SHADER_TIER_1);
}
VSTL_EXPORT_C void backend_device_names(luisa::vector<luisa::string> &r) {
    r.clear();
    ComPtr<IDXGIFactory2> dxgiFactory;
    ComPtr<IDXGIAdapter1> adapter;
    ThrowIfFailed(CreateDXGIFactory2(0, IID_PPV_ARGS(dxgiFactory.GetAddressOf())));
    for (auto adapterIndex = 0u; dxgiFactory->EnumAdapters1(adapterIndex, adapter.GetAddressOf()) != DXGI_ERROR_NOT_FOUND; adapterIndex++) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
            vstd::wstring s{desc.Description};
            auto &ss = r.emplace_back(s.size(), '\0');
            std::transform(s.cbegin(), s.cend(), ss.begin(), [](auto c) noexcept { return static_cast<char>(c); });
        }
    }
}
uint Device::waveSize() const {
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 waveOption;
    ThrowIfFailed(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &waveOption, sizeof(waveOption)));
    return waveOption.WaveLaneCountMin;
}
}// namespace lc::dx
