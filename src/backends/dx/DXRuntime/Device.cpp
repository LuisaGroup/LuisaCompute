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
static ID3D12Device *last_device_handle = nullptr;

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
static luisa::spin_mutex gDxcMutex;
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
    if (deviceSettings && deviceSettings->SyncFence(fence, fenceIndex)) return;
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
    return gDxcCompiler ? gDxcCompiler.ptr() : nullptr;
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
    bool use_lmdb = false;
    if (settings) {
        index = settings->device_index;
        // auto select
        useRuntime = !settings->headless;
        use_lmdb = settings->use_lmdb;
        maxAllocatorCount = settings->inqueue_buffer_limit ? 2 : std::numeric_limits<size_t>::max();
        fileIo = settings->binary_io;
        profiler = settings->profiler;
        if (settings->extension) {
            deviceSettings = vstd::create_unique(static_cast<DirectXDeviceConfigExt *>(settings->extension.release()));
        }
    }
    if (!deviceSettings || deviceSettings->LoadDXC()) {
        std::lock_guard lck(gDxcMutex);
        if (gDxcRefCount == 0) {
            gDxcCompiler.create(ctx.runtime_directory());
        }
        gDxcRefCount++;
    }
    if (fileIo == nullptr) {
        serVisitor = vstd::make_unique<DefaultBinaryIO>(std::move(ctx), !useRuntime, use_lmdb);
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
        bool use_dred = false;
        luisa::optional<DirectXDeviceConfigExt::ExternalDevice> extDevice;
        luisa::optional<DirectXDeviceConfigExt::GPUAllocatorSettings> allocSettings;
        if (deviceSettings) {
            extDevice = deviceSettings->CreateExternalDevice();
            allocSettings = deviceSettings->GetGPUAllocatorSettings();
            use_dred = deviceSettings->UseDRED();
        }
        if (extDevice && extDevice->device) {
            device = {static_cast<ID3D12Device5 *>(extDevice->device), false};
            if (extDevice->factory) {
                dxgiFactory = {extDevice->factory, false};
            } else {
                ThrowIfFailed(CreateDXGIFactory2(0, IID_PPV_ARGS(dxgiFactory.GetAddressOf())));
                // IDXGIFactory *pDxgiFactory;
                // ThrowIfFailed(adapter->GetParent(IID_PPV_ARGS(&pDxgiFactory)));
                // dxgiFactory = {static_cast<IDXGIFactory2 *>(pDxgiFactory), true};
            }
            if (extDevice->adapter) {
                adapter = {extDevice->adapter, false};
            } else {
                DxPtr<IDXGIAdapter1> local_adapter;
                auto device_id = device->GetAdapterLuid();
                for (auto adapterIndex = 0u; dxgiFactory->EnumAdapters1(adapterIndex, local_adapter.GetAddressOf()) != DXGI_ERROR_NOT_FOUND; adapterIndex++) {
                    DXGI_ADAPTER_DESC1 desc;
                    local_adapter->GetDesc1(&desc);
                    if (std::memcmp(&desc.AdapterLuid, &device_id, sizeof(LUID)) == 0) {
                        adapter = std::move(local_adapter);
                        break;
                    }
                }
                if (!adapter) {
                    LUISA_ERROR("Adapter not found.");
                }
                // IDXGIDevice *pDXGIDevice = nullptr;
                // auto dispose_pDXGIDevice = vstd::scope_exit([&] {
                //     if (pDXGIDevice)
                //         pDXGIDevice->Release();
                // });
                // ThrowIfFailed(device->QueryInterface(IID_PPV_ARGS(&pDXGIDevice)));
                // // Query for IDXGIAdapter from IDXGIDevice
                // IDXGIAdapter *pAdapter = nullptr;
                // ThrowIfFailed(pDXGIDevice->GetAdapter(&pAdapter));
                // adapter = {static_cast<IDXGIAdapter1 *>(pAdapter), true};
            }
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
            ComPtr<ID3D12DeviceRemovedExtendedDataSettings> pDredSettings;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&pDredSettings)))) {
                // Turn on AutoBreadcrumbs and Page Fault reporting
                pDredSettings->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                pDredSettings->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
#ifdef __ID3D12DeviceRemovedExtendedDataSettings1_INTERFACE_DEFINED__
                ComPtr<ID3D12DeviceRemovedExtendedDataSettings1> pDredSettings1;
                if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&pDredSettings1)))) {
                    pDredSettings1->SetBreadcrumbContextEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                }
#endif
                LUISA_WARNING("DRED settings enable");
            } else {
                LUISA_WARNING("DRED settings disable");
            }
#endif
            if (use_dred) {
#ifdef __ID3D12DeviceRemovedExtendedDataSettings2_INTERFACE_DEFINED__
                ComPtr<ID3D12DeviceRemovedExtendedDataSettings2> pDredSettings2;
                if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&pDredSettings2)))) {
                    pDredSettings2->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                    pDredSettings2->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                    pDredSettings2->UseMarkersOnlyAutoBreadcrumbs(true);// LightweightDRED
                    LUISA_WARNING("LightweightDRED settings enable");
                } else {
                    LUISA_WARNING("LightweightDRED settings disable");
                }
#endif
            }
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
        }
        static_cast<void>(fileIo->write_shader_cache("dx_adapterid", {reinterpret_cast<std::byte const *>(&adapterID), sizeof(vstd::MD5)}));
        if (allocSettings)
            defaultAllocator = vstd::make_unique<GpuAllocator>(
                this,
                profiler,
                allocSettings->preferred_block_size,
                allocSettings->sparse_buffer_block_size,
                allocSettings->sparse_image_block_size);
        else
            defaultAllocator = vstd::make_unique<GpuAllocator>(this, profiler, 0, 0, 0);
        allocatorInterface.device = this;
        globalHeap = vstd::create_unique(
            new DescriptorHeap(
                this,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                1000000ull,// Max allowed in Tier 3
                true));
        samplerHeap = vstd::create_unique(
            new DescriptorHeap(
                this,
                D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                16,
                true));
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
        // Test device
        D3D12_FEATURE_DATA_D3D12_OPTIONS12 options12 = {};
        if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &options12, sizeof(options12)))) {
            use_enhanced_barrier = (!deviceSettings || deviceSettings->UseEnhancedBarrier()) && options12.EnhancedBarriersSupported;
            // use_enhanced_barrier = false;
        }
        // TODO: currently there are lots of BUGS in NVIDIA's driver while using Enhanced barrier, disable it temporarily

        // if (!use_enhanced_barrier) [[unlikely]] {
        //     LUISA_WARNING("Enhanced barrier not supported, please update your Windows or GPU driver");
        // }
    } else {
        if (deviceSettings) {
            if (gDxcCompiler) {
                deviceSettings->ReadbackDX12Device(
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    fileIo,
                    gDxcCompiler->compiler(),
                    gDxcCompiler->library(),
                    gDxcCompiler->utils(),
                    nullptr,
                    nullptr);
            } else {
                deviceSettings->ReadbackDX12Device(
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    fileIo,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr);
            }
        }
    }
    last_device_handle = device.Get();
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
    return waveOption.WaveLaneCountMax;
}
void process_dxgi_error(HRESULT hr) {
    // Should match all values from D3D12_AUTO_BREADCRUMB_OP
    static const wchar_t *OpNames[]{
        L"SetMarker",
        L"BeginEvent",
        L"EndEvent",
        L"DrawInstanced",
        L"DrawIndexedInstanced",
        L"ExecuteIndirect",
        L"Dispatch",
        L"CopyBufferRegion",
        L"CopyTextureRegion",
        L"CopyResource",
        L"CopyTiles",
        L"ResolveSubresource",
        L"ClearRenderTargetView",
        L"ClearUnorderedAccessView",
        L"ClearDepthStencilView",
        L"ResourceBarrier",
        L"ExecuteBundle",
        L"Present",
        L"ResolveQueryData",
        L"BeginSubmission",
        L"EndSubmission",
        L"DecodeFrame",
        L"ProcessFrames",
        L"AtomicCopyBufferUint",
        L"AtomicCopyBufferUint64",
        L"ResolveSubresourceRegion",
        L"WriteBufferImmediate",
        L"DecodeFrame1",
        L"SetProtectedResourceSession",
        L"DecodeFrame2",
        L"ProcessFrames1",
        L"BuildRaytracingAccelerationStructure",
        L"EmitRaytracingAccelerationStructurePostBuildInfo",
        L"CopyRaytracingAccelerationStructure",
        L"DispatchRays",
        L"InitializeMetaCommand",
        L"ExecuteMetaCommand",
        L"EstimateMotion",
        L"ResolveMotionVectorHeap",
        L"SetPipelineState1",
        L"InitializeExtensionCommand",
        L"ExecuteExtensionCommand",
        L"DispatchMesh",
        L"EncodeFrame",
        L"ResolveEncoderOutputMetadata"};
    static_assert(std::size(OpNames) == D3D12_AUTO_BREADCRUMB_OP_RESOLVEENCODEROUTPUTMETADATA + 1, "OpNames array length mismatch");
    // Should match all valid values from D3D12_DRED_ALLOCATION_TYPE
    static const wchar_t *AllocTypesNames[]{
        L"CommandQueue",
        L"CommandAllocator",
        L"PipelineState",
        L"CommandList",
        L"Fence",
        L"DescriptorHeap",
        L"Heap",
        L"Unknown",// Unknown type - missing enum value in D3D12_DRED_ALLOCATION_TYPE
        L"QueryHeap",
        L"CommandSignature",
        L"PipelineLibrary",
        L"VideoDecoder",
        L"Unknown",// Unknown type - missing enum value in D3D12_DRED_ALLOCATION_TYPE
        L"VideoProcessor",
        L"Unknown",// Unknown type - missing enum value in D3D12_DRED_ALLOCATION_TYPE
        L"Resource",
        L"Pass",
        L"CryptoSession",
        L"CryptoSessionPolicy",
        L"ProtectedResourceSession",
        L"VideoDecoderHeap",
        L"CommandPool",
        L"CommandRecorder",
        L"StateObjectr",
        L"MetaCommand",
        L"SchedulingGroup",
        L"VideoMotionEstimator",
        L"VideoMotionVectorHeap",
        L"VideoExtensionCommand",
    };
    static_assert(std::size(AllocTypesNames) == D3D12_DRED_ALLOCATION_TYPE_VIDEO_EXTENSION_COMMAND - D3D12_DRED_ALLOCATION_TYPE_COMMAND_QUEUE + 1, "AllocTypes array length mismatch");
    auto GetBreadcrumbContexts = [](const D3D12_AUTO_BREADCRUMB_NODE1 *Node) {
        return luisa::span<D3D12_DRED_BREADCRUMB_CONTEXT>{Node->pBreadcrumbContexts, Node->BreadcrumbContextsCount};
    };

    //if (hr != DXGI_ERROR_DEVICE_REMOVED && hr != DXGI_ERROR_DEVICE_HUNG && hr != DXGI_ERROR_DEVICE_RESET) {
    //    return;
    //}
    if (!last_device_handle) {
        return;
    }
    auto pDevice = last_device_handle;
    last_device_handle = nullptr;
    ComPtr<ID3D12DeviceRemovedExtendedData1> pDred;
    if (FAILED(pDevice->QueryInterface(IID_PPV_ARGS(&pDred)))) {
        return;
    }

    D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1 DredAutoBreadcrumbsOutput;
    D3D12_DRED_PAGE_FAULT_OUTPUT DredPageFaultOutput;
    if (FAILED(pDred->GetAutoBreadcrumbsOutput1(&DredAutoBreadcrumbsOutput))) {
        return;
    }
    if (FAILED(pDred->GetPageFaultAllocationOutput(&DredPageFaultOutput))) {
        return;
    }
    luisa::wstring result;
    result += L"DRED: Last tracked GPU operations:\n";

    luisa::wstring ContextStr;
    luisa::unordered_map<int32, const wchar_t *> ContextStrings;
    int TracedCommandLists = 0;
    auto node = DredAutoBreadcrumbsOutput.pHeadAutoBreadcrumbNode;
    while (node && node->pLastBreadcrumbValue) {
        int32 LastCompletedOp = *node->pLastBreadcrumbValue;
        if (LastCompletedOp != node->BreadcrumbCount && LastCompletedOp != 0) {
            if (node->pCommandListDebugNameW) {
                luisa::format_to(std::back_inserter(result), L"Command list debug name: {}\n", node->pCommandListDebugNameW);
            }
            if (node->pCommandQueueDebugNameW) {
                luisa::format_to(std::back_inserter(result), L"Command queue debug name: {}\n", node->pCommandQueueDebugNameW);
            }
            luisa::format_to(std::back_inserter(result), L"DRED: {} completed of {}\n", LastCompletedOp, node->BreadcrumbCount);
            TracedCommandLists++;
            int32 FirstOp = std::max(LastCompletedOp - 100, 0);
            int32 LastOp = std::min(LastCompletedOp + 20, int32(node->BreadcrumbCount) - 1);
            ContextStrings.clear();
            for (const D3D12_DRED_BREADCRUMB_CONTEXT &Context : GetBreadcrumbContexts(node)) {
                ContextStrings.emplace(Context.BreadcrumbIndex, Context.pContextString);
            }
            for (int32 Op = FirstOp; Op <= LastOp; ++Op) {
                D3D12_AUTO_BREADCRUMB_OP BreadcrumbOp = node->pCommandHistory[Op];
                auto OpContextStr = ContextStrings.find(Op);
                if (OpContextStr != ContextStrings.end()) {
                    ContextStr += OpContextStr->second;
                } else {
                    ContextStr.clear();
                }
                luisa::wstring_view OpName = (BreadcrumbOp < std::size(OpNames)) ? OpNames[BreadcrumbOp] : L"Unknown Op";
                luisa::wstring_view State = Op < LastCompletedOp ? L"[ok]" : (Op == LastCompletedOp ? L"[Active]" : L"[ ]");
                luisa::format_to(std::back_inserter(result), L"\t{} Op: {}, {} {} {}\n", State, Op, OpName, ContextStr, (Op + 1 == LastCompletedOp) ? L" - LAST COMPLETED" : L"");
            }
        }
        node = node->pNext;
    }
    if (TracedCommandLists == 0) {
        result += L"DRED: No command list found with active outstanding operations (all finished or not started yet)\n";
    }
    luisa::format_to(std::back_inserter(result), L"page fault VA: {}\n", DredPageFaultOutput.PageFaultVA);

    for (auto node = DredPageFaultOutput.pHeadExistingAllocationNode; node != nullptr; node = node->pNext) {
        if (node->ObjectNameW) {
            luisa::format_to(std::back_inserter(result), L"Exists object name {}\n", node->ObjectNameW);
        }
    }
    for (auto node = DredPageFaultOutput.pHeadRecentFreedAllocationNode; node != nullptr; node = node->pNext) {
        if (node->ObjectNameW) {
            luisa::format_to(std::back_inserter(result), L"Freed object name {}\n", node->ObjectNameW);
        }
    }
    LUISA_WARNING(luisa::string(result.begin(), result.end()));
}

}// namespace lc::dx
