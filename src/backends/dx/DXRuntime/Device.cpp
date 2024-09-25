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

namespace lc::dx {
DirectXHeap DXAllocatorImpl::AllocateBufferHeap(
    luisa::string_view name,
    uint64_t targetSizeInBytes,
    D3D12_HEAP_TYPE heapType,
    D3D12_HEAP_FLAGS extraFlags) const noexcept {
    DirectXHeap heap;
    heap.handle = device->defaultAllocator->AllocateBufferHeap(device, name, targetSizeInBytes, heapType, &heap.heap, &heap.offset, extraFlags);
    return heap;
}
DirectXHeap DXAllocatorImpl::AllocateTextureHeap(
    vstd::string_view name,
    size_t sizeBytes,
    bool isRenderTexture,
    D3D12_HEAP_FLAGS extraFlags) const noexcept {
    DirectXHeap heap;
    heap.handle = device->defaultAllocator->AllocateTextureHeap(device, name, sizeBytes, &heap.heap, &heap.offset, extraFlags);
    return heap;
}
void DXAllocatorImpl::DeAllocateHeap(uint64_t handle) const noexcept {
    device->defaultAllocator->Release(handle);
}
static std::mutex gDxcMutex;
static vstd::StackObject<hlsl::ShaderCompiler, false> gDxcCompiler;
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
        shader = vstd::create_unique(loadFunc(self));
    }
    return shader.get();
}
bool Device::LazyLoadShader::Check(Device *self) {
    if (shader) return true;
    shader = vstd::create_unique(loadFunc(self));
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
        profiler = settings->profiler;
        if (settings->extension) {
            deviceSettings = vstd::create_unique(static_cast<DirectXDeviceConfigExt *>(settings->extension.release()));
        }
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
            std::memcpy(info.Description, desc.Description, sizeof(WCHAR) * 128);
            info.VendorId = desc.VendorId;
            info.DeviceId = desc.DeviceId;
            info.SubSysId = desc.SubSysId;
            info.Revision = desc.Revision;
            return vstd::MD5{vstd::span<uint8_t const>{reinterpret_cast<uint8_t const *>(&info), sizeof(AdapterInfo)}};
        };

        luisa::optional<DirectXDeviceConfigExt::ExternalDevice> extDevice;
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
            luisa::vector<luisa::string> device_names;
            backend_device_names(device_names);
            for (auto &name : device_names) {
                for (auto &i : name) {
                    if (i >= 'A' && i <= 'Z') {
                        i += 'a' - 'A';
                    }
                }
            }
            if (index == std::numeric_limits<size_t>::max()) {
                index = 0;
                size_t max_score = 0;
                for (size_t i = 0; i < device_names.size(); ++i) {
                    luisa::string &device_name = device_names[i];
                    size_t score = 0;
                    if (device_name.find("geforce") != luisa::string::npos ||
                        device_name.find("radeon") != luisa::string::npos) {
                        score += 1;
                    }
                    if (device_name.find("gtx") != luisa::string::npos ||
                        device_name.find("rtx") != luisa::string::npos ||
                        device_name.find("arc") != luisa::string::npos ||
                        device_name.find("rx") != luisa::string::npos) {
                        score += 10;
                    }
                    if (score > max_score) {
                        index = i;
                        max_score = score;
                    }
                }
                LUISA_INFO("Select device: {}", device_names[index]);
            }
            auto &device_name = device_names[index];

            if (device_name.find("nvidia") != luisa::string::npos) {
                gpuType = GpuType::NVIDIA;
            } else if (device_name.find("amd") != luisa::string::npos) {
                gpuType = GpuType::AMD;
            } else if (device_name.find("intel") != luisa::string::npos) {
                gpuType = GpuType::INTEL;
            }
            auto capableAdapterIndex = 0u;
            for (auto adapterIndex = 0u; dxgiFactory->EnumAdapters1(adapterIndex, adapter.GetAddressOf()) != DXGI_ERROR_NOT_FOUND; adapterIndex++) {
                DXGI_ADAPTER_DESC1 desc;
                adapter->GetDesc1(&desc);
                if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
                    if (capableAdapterIndex++ == index) {
                        ThrowIfFailed(D3D12CreateDevice(
                            adapter.Get(), D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(device.GetAddressOf())));
                        adapterID = GenAdapterGUID(desc);
                        break;
                    }
                }
                device.Clear();
                adapter.Clear();
            }
            if (adapter == nullptr) { LUISA_ERROR_WITH_LOCATION("Failed to create DirectX device at index {}.", index); }
        }
        {
            auto adapterIdStream = fileIo->read_shader_cache("dx_adapterid");
            bool sameAdaptor = false;
            if (adapterIdStream) {
                auto blob = adapterIdStream->read(~0ull);
                sameAdaptor = blob.size() == sizeof(vstd::MD5) && std::memcmp(blob.data(), &adapterID, sizeof(vstd::MD5)) == 0;
            }
            if (!sameAdaptor) {
                LUISA_INFO("Adapter mismatch, shader cache cleared.");
                fileIo->clear_shader_cache();
            }
            fileIo->write_shader_cache("dx_adapterid", {reinterpret_cast<std::byte const *>(&adapterID), sizeof(vstd::MD5)});
        }
        defaultAllocator = vstd::make_unique<GpuAllocator>(this, profiler);
        allocatorInterface.device = this;
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
                dxgiFactory,
                &allocatorInterface,
                fileIo,
                gDxcCompiler->compiler(),
                gDxcCompiler->library(),
                gDxcCompiler->utils(),
                globalHeap->GetHeap(),
                samplerHeap->GetHeap());
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
