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
#ifdef LUISA_DX_SDK
extern "C" extern const uint32_t D3D12SDKVersion = D3D12_PREVIEW_SDK_VERSION;

extern "C" extern LPCSTR D3D12SDKPath = ".\\D3D12\\";
#endif

namespace lc::dx {
static ID3D12Device *last_device_handle = nullptr;

DirectXHeap DXAllocatorImpl::allocate_buffer_heap(
    luisa::string_view name,
    uint64_t target_size_in_bytes,
    D3D12_HEAP_TYPE heap_type,
    D3D12_HEAP_FLAGS extra_flags) const noexcept {
    DirectXHeap heap{};
    heap.handle = device->default_allocator->AllocateBufferHeap(device, name, target_size_in_bytes, heap_type, &heap.heap, &heap.offset, extra_flags);
    return heap;
}
DirectXHeap DXAllocatorImpl::allocate_texture_heap(
    vstd::string_view name,
    size_t size_bytes,
    bool is_render_texture,
    D3D12_HEAP_FLAGS extra_flags) const noexcept {
    DirectXHeap heap{};
    heap.handle = device->default_allocator->AllocateTextureHeap(device, name, size_bytes, &heap.heap, &heap.offset, extra_flags);
    return heap;
}
void DXAllocatorImpl::deallocate_heap(uint64_t handle) const noexcept {
    device->default_allocator->Release(handle);
}
static luisa::spin_mutex g_dxc_mutex;
static vstd::StackObject<hlsl::ShaderCompiler, false> g_dxc_compiler;
static int32 g_dxc_ref_count = 0;

Device::LazyLoadShader::~LazyLoadShader() {}
LUISA_EXPORT_API void backend_device_names(luisa::vector<luisa::string> &r);
Device::LazyLoadShader::LazyLoadShader(LoadFunc load_func) : _load_func(load_func) {}
Device::~Device() {
    //lcmdSig.destroy();
#ifndef LC_NO_HLSL_BUILTIN
    std::lock_guard lck(g_dxc_mutex);
    if (--g_dxc_ref_count == 0) {
        g_dxc_compiler.destroy();
    }
#endif
}

void Device::wait_fence(ID3D12Fence *fence, uint64 fenceIndex) {
    if (fenceIndex <= 0) return;
    if (device_settings && device_settings->SyncFence(fence, fenceIndex)) return;
    HANDLE eventHandle = CreateEventEx(nullptr, nullptr, false, EVENT_ALL_ACCESS);
    auto d = vstd::scope_exit([&] {
        CloseHandle(eventHandle);
    });
    if (fence->GetCompletedValue() < fenceIndex) {
        ThrowIfFailed(fence->SetEventOnCompletion(fenceIndex, eventHandle));
        WaitForSingleObject(eventHandle, INFINITE);
    }
}
ComputeShader *Device::LazyLoadShader::get(Device *self) {
    if (!_shader) {
        _shader = vstd::create_unique(_load_func(self));
    }
    return _shader.get();
}
bool Device::LazyLoadShader::check(Device *self) {
    if (_shader) return true;
    _shader = vstd::create_unique(_load_func(self));
    if (_shader) {
        auto afterExit = vstd::scope_exit([&] { _shader = nullptr; });
        return true;
    }
    return false;
}

hlsl::ShaderCompiler *Device::compiler() {
    return g_dxc_compiler ? g_dxc_compiler.ptr() : nullptr;
}
Device::Device(Context &&ctx, DeviceConfig const *settings)
    : set_bindless_kernel(BuiltinKernel::load_bindless_set_kernel),
      set_accel_kernel(BuiltinKernel::load_accel_set_kernel),
      bc6_try_mode_g10(BuiltinKernel::load_bc6_try_mode_g10cs_kernel),
      bc6_try_mode_le10(BuiltinKernel::load_bc6_try_mode_le10cs_kernel),
      bc6_encode_block(BuiltinKernel::load_bc6_encode_block_cs_kernel),
      bc7_try_mode_456(BuiltinKernel::load_bc7_try_mode_456cs_kernel),
      bc7_try_mode_137(BuiltinKernel::load_bc7_try_mode_137cs_kernel),
      bc7_try_mode_02(BuiltinKernel::load_bc7_try_mode_02cs_kernel),
      bc7_encode_block(BuiltinKernel::load_bc7_encode_block_cs_kernel) {
    using Microsoft::WRL::ComPtr;
    size_t index{std::numeric_limits<size_t>::max()};
    bool use_runtime = true;
    bool use_lmdb = false;
    bool use_experimental = false;
    if (settings) {
        index = settings->device_index;
        // auto select
        use_runtime = !settings->headless;
        use_lmdb = settings->use_lmdb;
        max_allocator_count = settings->inqueue_buffer_limit ? 2 : std::numeric_limits<size_t>::max();
        file_io = settings->binary_io;
        profiler = settings->profiler;
        if (settings->extension) {
            device_settings = vstd::create_unique(static_cast<DirectXDeviceConfigExt *>(settings->extension.release()));
            use_experimental = device_settings->UseExperimental();
        }
    }
#ifndef LC_NO_HLSL_BUILTIN
    if (!device_settings || device_settings->LoadDXC()) {
        std::lock_guard lck(g_dxc_mutex);
        if (g_dxc_ref_count == 0) {
            g_dxc_compiler.create(ctx.runtime_directory(), false);
        }
        g_dxc_ref_count++;
    }
#endif
    if (file_io == nullptr) {
        ser_visitor = vstd::make_unique<DefaultBinaryIO>(std::move(ctx), !use_runtime, use_lmdb);
        file_io = ser_visitor.get();
    }
    if (use_runtime) {
        bool experimental_features_enabled = false;
        if (use_experimental) {
            // Enable experimental features individually.  If one is not supported
            // by the current OS/runtime we log a warning and continue without it
            // instead of aborting device creation.
            auto try_enable = [](UUID const &feature, const char *name) {
                HRESULT hr = D3D12EnableExperimentalFeatures(1, &feature, nullptr, nullptr);
                if (FAILED(hr)) {
                    LUISA_WARNING(
                        "Failed to enable D3D12 experimental feature '{}': {:#08x}. "
                        "Continuing without experimental features.",
                        name, static_cast<uint32_t>(hr));
                    return false;
                }
                return true;
            };
            experimental_features_enabled =
                try_enable(D3D12ExperimentalShaderModels, "ExperimentalShaderModels");
        }
        if (device_settings) {
            device_settings->SetExperimentalFeaturesEnabled(experimental_features_enabled);
        }
        auto gen_adapter_guid = [](DXGI_ADAPTER_DESC1 const &desc) {
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
        luisa::optional<DirectXDeviceConfigExt::ExternalDevice> ext_device;
        luisa::optional<DirectXDeviceConfigExt::GPUAllocatorSettings> alloc_settings;
        if (device_settings) {
            ext_device = device_settings->CreateExternalDevice();
            alloc_settings = device_settings->GetGPUAllocatorSettings();
            use_dred = device_settings->UseDRED();
        }
        if (ext_device && ext_device->device) {
            device = {static_cast<ID3D12Device5 *>(ext_device->device), false};
            if (ext_device->factory) {
                dxgi_factory = {ext_device->factory, false};
            } else {
                ThrowIfFailed(CreateDXGIFactory2(0, IID_PPV_ARGS(dxgi_factory.GetAddressOf())));
                // IDXGIFactory *pDxgiFactory;
                // ThrowIfFailed(adapter->GetParent(IID_PPV_ARGS(&pDxgiFactory)));
                // dxgi_factory = {static_cast<IDXGIFactory2 *>(pDxgiFactory), true};
            }
            if (ext_device->adapter) {
                adapter = {ext_device->adapter, false};
            } else {
                DxPtr<IDXGIAdapter1> local_adapter;
                auto device_id = device->GetAdapterLuid();
                for (auto adapter_index = 0u; dxgi_factory->EnumAdapters1(adapter_index, local_adapter.GetAddressOf()) != DXGI_ERROR_NOT_FOUND; adapter_index++) {
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
            adapter_id = gen_adapter_guid(desc);
        } else {
            uint32_t dxgi_factory_flags = 0;
#ifndef NDEBUG
            // Enable the debug layer (requires the Graphics Tools "optional feature").
            // NOTE: Enabling the debug layer after device creation will invalidate the active device.
            {
                ComPtr<ID3D12Debug> debugController;
                if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
                    debugController->EnableDebugLayer();

                    // Enable additional debug layers.
                    dxgi_factory_flags |= DXGI_CREATE_FACTORY_DEBUG;
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
            ThrowIfFailed(CreateDXGIFactory2(dxgi_factory_flags, IID_PPV_ARGS(dxgi_factory.GetAddressOf())));
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
                gpu_type = GpuType::NVIDIA;
            } else if (device_name.find("amd") != luisa::string::npos) {
                gpu_type = GpuType::AMD;
            } else if (device_name.find("intel") != luisa::string::npos) {
                gpu_type = GpuType::INTEL;
            }
            auto capable_adapter_index = 0u;
            for (auto adapter_index = 0u; dxgi_factory->EnumAdapters1(adapter_index, adapter.GetAddressOf()) != DXGI_ERROR_NOT_FOUND; adapter_index++) {
                DXGI_ADAPTER_DESC1 desc;
                adapter->GetDesc1(&desc);
                if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
                    if (capable_adapter_index++ == index) {
                        ThrowIfFailed(D3D12CreateDevice(
                            adapter.Get(), D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(device.GetAddressOf())));
                        adapter_id = gen_adapter_guid(desc);
                        break;
                    }
                }
                device.Clear();
                adapter.Clear();
            }
            if (adapter == nullptr) { LUISA_ERROR_WITH_LOCATION("Failed to create DirectX device at index {}.", index); }
        }
        {
            auto adapter_id_stream = file_io->read_shader_cache("dx_adapterid");
            bool same_adaptor = false;
            if (adapter_id_stream) {
                auto blob = adapter_id_stream->read(~0ull);
                same_adaptor = blob.size() == sizeof(vstd::MD5) && std::memcmp(blob.data(), &adapter_id, sizeof(vstd::MD5)) == 0;
            }
            if (!same_adaptor) {
                LUISA_INFO("Adapter mismatch, shader cache cleared.");
                file_io->clear_shader_cache();
            }
        }
        static_cast<void>(file_io->write_shader_cache("dx_adapterid", {reinterpret_cast<std::byte const *>(&adapter_id), sizeof(vstd::MD5)}));
        if (alloc_settings)
            default_allocator = vstd::make_unique<GpuAllocator>(
                this,
                profiler,
                alloc_settings->preferred_block_size,
                alloc_settings->sparse_buffer_block_size,
                alloc_settings->sparse_image_block_size);
        else
            default_allocator = vstd::make_unique<GpuAllocator>(this, profiler, 0, 0, 0);
        if (device_settings) {
            device_settings->GetDefragmentFunction([ptr = default_allocator.get()] {
                ptr->Defragment();
            });
        }
        allocator_interface.device = this;
        global_heap = vstd::create_unique(
            new DescriptorHeap(
                this,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                1000000ull,// Max allowed in Tier 3
                true));
        sampler_heap = vstd::create_unique(
            new DescriptorHeap(
                this,
                D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                16,
                true));
        auto samplers = GlobalSamplers::GetSamplers();
        for (auto i : vstd::range(samplers.size())) {
            sampler_heap->CreateSampler(
                samplers[i], i);
        }
        if (device_settings) {
            device_settings->ReadbackDX12Device(
                device,
                adapter,
                dxgi_factory,
                &allocator_interface,
                file_io,
                g_dxc_compiler->compiler(),
                g_dxc_compiler->library(),
                g_dxc_compiler->utils(),
                global_heap->GetHeap(),
                sampler_heap->GetHeap());
        }
        feature_check.check(this);
        {
            feature_check.flags().enhanced_barriers_supported = (device_settings && device_settings->UseEnhancedBarrier()) && feature_check.flags().enhanced_barriers_supported;
        }
    } else {
        if (device_settings) {
            if (g_dxc_compiler) {
                device_settings->ReadbackDX12Device(
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    file_io,
                    g_dxc_compiler->compiler(),
                    g_dxc_compiler->library(),
                    g_dxc_compiler->utils(),
                    nullptr,
                    nullptr);
            } else {
                device_settings->ReadbackDX12Device(
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    file_io,
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
bool Device::support_mesh_shader() const {
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 feature_data = {};
    device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &feature_data, sizeof(feature_data));
    return (feature_data.MeshShaderTier >= D3D12_MESH_SHADER_TIER_1);
}

LUISA_EXPORT_API void backend_device_names(luisa::vector<luisa::string> &r) {
    r.clear();
    ComPtr<IDXGIFactory2> dxgi_factory;
    ComPtr<IDXGIAdapter1> adapter;
    ThrowIfFailed(CreateDXGIFactory2(0, IID_PPV_ARGS(dxgi_factory.GetAddressOf())));
    for (auto adapter_index = 0u; dxgi_factory->EnumAdapters1(adapter_index, adapter.GetAddressOf()) != DXGI_ERROR_NOT_FOUND; adapter_index++) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
            vstd::wstring s{desc.Description};
            auto &ss = r.emplace_back(s.size(), '\0');
            std::transform(s.cbegin(), s.cend(), ss.begin(), [](auto c) noexcept { return static_cast<char>(c); });
        }
    }
}
uint Device::wave_size() const {
    return feature_check.wave_lane_count_max();
}
void process_dxgi_error(HRESULT hr) {
    // Should match all values from D3D12_AUTO_BREADCRUMB_OP
    static const wchar_t *kOpNames[]{
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
    static_assert(std::size(kOpNames) == D3D12_AUTO_BREADCRUMB_OP_RESOLVEENCODEROUTPUTMETADATA + 1, "kOpNames array length mismatch");
    // Should match all valid values from D3D12_DRED_ALLOCATION_TYPE
    static const wchar_t *kAllocTypesNames[]{
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
    static_assert(std::size(kAllocTypesNames) == D3D12_DRED_ALLOCATION_TYPE_VIDEO_EXTENSION_COMMAND - D3D12_DRED_ALLOCATION_TYPE_COMMAND_QUEUE + 1, "kAllocTypesNames array length mismatch");
    auto get_breadcrumb_contexts = [](const D3D12_AUTO_BREADCRUMB_NODE1 *Node) {
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
    ComPtr<ID3D12DeviceRemovedExtendedData1> dred;
    if (FAILED(pDevice->QueryInterface(IID_PPV_ARGS(&dred)))) {
        return;
    }

    D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1 dred_auto_breadcrumbs_output;
    D3D12_DRED_PAGE_FAULT_OUTPUT dred_page_fault_output;
    if (FAILED(dred->GetAutoBreadcrumbsOutput1(&dred_auto_breadcrumbs_output))) {
        return;
    }
    if (FAILED(dred->GetPageFaultAllocationOutput(&dred_page_fault_output))) {
        return;
    }
    luisa::wstring result;
    result += L"DRED: Last tracked GPU operations:\n";

    luisa::wstring context_str;
    luisa::unordered_map<int32, const wchar_t *> context_strings;
    int traced_command_lists = 0;
    auto node = dred_auto_breadcrumbs_output.pHeadAutoBreadcrumbNode;
    while (node && node->pLastBreadcrumbValue) {
        int32 last_completed_op = *node->pLastBreadcrumbValue;
        if (last_completed_op != node->BreadcrumbCount && last_completed_op != 0) {
            if (node->pCommandListDebugNameW) {
                luisa::format_to(std::back_inserter(result), L"Command list debug name: {}\n", node->pCommandListDebugNameW);
            }
            if (node->pCommandQueueDebugNameW) {
                luisa::format_to(std::back_inserter(result), L"Command queue debug name: {}\n", node->pCommandQueueDebugNameW);
            }
            luisa::format_to(std::back_inserter(result), L"DRED: {} completed of {}\n", last_completed_op, node->BreadcrumbCount);
            traced_command_lists++;
            int32 first_op = std::max(last_completed_op - 100, 0);
            int32 last_op = std::min(last_completed_op + 20, int32(node->BreadcrumbCount) - 1);
            context_strings.clear();
            for (const D3D12_DRED_BREADCRUMB_CONTEXT &Context : get_breadcrumb_contexts(node)) {
                context_strings.emplace(Context.BreadcrumbIndex, Context.pContextString);
            }
            for (int32 op = first_op; op <= last_op; ++op) {
                D3D12_AUTO_BREADCRUMB_OP breadcrumb_op = node->pCommandHistory[op];
                auto op_context_str = context_strings.find(op);
                if (op_context_str != context_strings.end()) {
                    context_str += op_context_str->second;
                } else {
                    context_str.clear();
                }
                luisa::wstring_view op_name = (breadcrumb_op < std::size(kOpNames)) ? kOpNames[breadcrumb_op] : L"Unknown Op";
                luisa::wstring_view state = op < last_completed_op ? L"[ok]" : (op == last_completed_op ? L"[Active]" : L"[ ]");
                luisa::format_to(std::back_inserter(result), L"\t{} Op: {}, {} {} {}\n", state, op, op_name, context_str, (op + 1 == last_completed_op) ? L" - LAST COMPLETED" : L"");
            }
        }
        node = node->pNext;
    }
    if (traced_command_lists == 0) {
        result += L"DRED: No command list found with active outstanding operations (all finished or not started yet)\n";
    }
    luisa::format_to(std::back_inserter(result), L"page fault VA: {}\n", dred_page_fault_output.PageFaultVA);

    for (auto node = dred_page_fault_output.pHeadExistingAllocationNode; node != nullptr; node = node->pNext) {
        if (node->ObjectNameW) {
            luisa::format_to(std::back_inserter(result), L"Exists object name {}\n", node->ObjectNameW);
        }
    }
    for (auto node = dred_page_fault_output.pHeadRecentFreedAllocationNode; node != nullptr; node = node->pNext) {
        if (node->ObjectNameW) {
            luisa::format_to(std::back_inserter(result), L"Freed object name {}\n", node->ObjectNameW);
        }
    }
    LUISA_WARNING(luisa::string(result.begin(), result.end()));
}

}// namespace lc::dx
