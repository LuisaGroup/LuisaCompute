#include <DXRuntime/Device.h>
#include <DXRuntime/ShaderPaths.h>
#include <Resource/DescriptorHeap.h>
#include <Resource/DefaultBuffer.h>
#include <DXRuntime/GlobalSamplers.h>
#include <Resource/GpuAllocator.h>
#include <Shader/BuiltinKernel.h>
#include <dxgi1_3.h>
#include <Shader/ShaderCompiler.h>
#include <Shader/ComputeShader.h>
#include <vstl/binary_reader.h>
#include <core/logging.h>
#include <runtime/context.h>
#include <ext_settings.h>
#include <runtime/context_paths.h>
namespace toolhub::directx {
static std::mutex gDxcMutex;
static vstd::optional<ShaderCompiler> gDxcCompiler;
static int32 gDxcRefCount = 0;

SerializeVisitor::SerializeVisitor(
    ShaderPaths const &path) noexcept
    : path(path) {
    eastl::make_shared<int>(5);
}
luisa::unique_ptr<luisa::compute::IBinaryStream> SerializeVisitor::Read(vstd::string const &filePath) noexcept {
    return luisa::make_unique<BinaryStream>(filePath);
}
void SerializeVisitor::Write(vstd::string const &filePath, luisa::span<std::byte const> data) noexcept {

    auto f = fopen(filePath.c_str(), "wb");
    if (f) {
        fwrite(data.data(), data.size(), 1, f);
        fclose(f);
    }
}
luisa::unique_ptr<luisa::compute::IBinaryStream> SerializeVisitor::read_bytecode(luisa::string_view name) noexcept {
    std::filesystem::path localPath{name};
    if (localPath.is_absolute()) {
        return Read(luisa::to_string(name));
    }
    auto filePath = luisa::to_string(this->path.runtimeFolder / name);
    return Read(filePath);
}
luisa::unique_ptr<luisa::compute::IBinaryStream> SerializeVisitor::read_internal(luisa::string_view name) noexcept {
    auto filePath = luisa::to_string(this->path.dataFolder / name);
    return Read(filePath);
}
luisa::unique_ptr<luisa::compute::IBinaryStream> SerializeVisitor::read_cache(luisa::string_view name) noexcept {
    auto filePath = luisa::to_string(this->path.shaderCacheFolder / name);
    return Read(filePath);
}
void SerializeVisitor::write_bytecode(luisa::string_view name, luisa::span<std::byte const> data) noexcept {
    std::filesystem::path localPath{name};
    if (localPath.is_absolute()) {
        Write(luisa::to_string(name), data);
        return;
    }
    auto filePath = luisa::to_string(this->path.runtimeFolder / name);
    Write(filePath, data);
}
void SerializeVisitor::write_cache(luisa::string_view name, luisa::span<std::byte const> data) noexcept {
    auto filePath = luisa::to_string(this->path.shaderCacheFolder / name);
    Write(filePath, data);
}
void SerializeVisitor::write_internal(luisa::string_view name, luisa::span<std::byte const> data) noexcept {
    auto filePath = luisa::to_string(this->path.dataFolder / name);
    Write(filePath, data);
}

Device::LazyLoadShader::~LazyLoadShader() {}

Device::LazyLoadShader::LazyLoadShader(LoadFunc loadFunc) : loadFunc(loadFunc) {}
Device::~Device() {
    //lcmdSig.Delete();
    std::lock_guard lck(gDxcMutex);
    if (--gDxcRefCount == 0) {
        gDxcCompiler.Delete();
    }
}

void Device::WaitFence(ID3D12Fence *fence, uint64 fenceIndex) {
    if (fenceIndex <= 0) return;
    HANDLE eventHandle = CreateEventEx(nullptr, (LPCWSTR) nullptr, false, EVENT_ALL_ACCESS);
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

ShaderCompiler *Device::Compiler() {
    return gDxcCompiler;
}
Device::Device(Context &ctx, ShaderPaths const &path, DeviceConfig const *settings)
    : path(path),
      serVisitor(path),
      setAccelKernel(BuiltinKernel::LoadAccelSetKernel),
      bc6TryModeG10(BuiltinKernel::LoadBC6TryModeG10CSKernel),
      bc6TryModeLE10(BuiltinKernel::LoadBC6TryModeLE10CSKernel),
      bc6EncodeBlock(BuiltinKernel::LoadBC6EncodeBlockCSKernel),
      bc7TryMode456(BuiltinKernel::LoadBC7TryMode456CSKernel),
      bc7TryMode137(BuiltinKernel::LoadBC7TryMode137CSKernel),
      bc7TryMode02(BuiltinKernel::LoadBC7TryMode02CSKernel),
      bc7EncodeBlock(BuiltinKernel::LoadBC7EncodeBlockCSKernel) {
    using Microsoft::WRL::ComPtr;
    fileIo = &serVisitor;
    size_t index = 0;
    bool useRuntime = true;
    if (settings) {
        index = settings->device_index;
        useRuntime = !settings->headless;
        maxAllocatorCount = settings->inqueue_buffer_limit ? 2 : std::numeric_limits<size_t>::max();
    }
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
    if (useRuntime) {
        if (settings && settings->hash == DirectXDeviceSettings::kHash) {
            deviceSettings = static_cast<DirectXDeviceSettings const *>(settings)->CreateExternalRuntime();
        }
        if (deviceSettings) {
            device = static_cast<ID3D12Device5 *>(deviceSettings->GetDevice());
            adapter = deviceSettings->GetAdapter();
            dxgiFactory = deviceSettings->GetDXGIFactory();
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
            ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&mDxgiFactory)));
            dxgiFactory = mDxgiFactory.Get();
            auto capableAdapterIndex = 0u;
            for (auto adapterIndex = 0u; mDxgiFactory->EnumAdapters1(adapterIndex, &mAdapter) != DXGI_ERROR_NOT_FOUND; adapterIndex++) {
                DXGI_ADAPTER_DESC1 desc;
                mAdapter->GetDesc1(&desc);
                if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
                    if (capableAdapterIndex++ == index) {
                        ThrowIfFailed(D3D12CreateDevice(
                            mAdapter.Get(), D3D_FEATURE_LEVEL_12_1,
                            IID_PPV_ARGS(&mDevice)));
                        vstd::wstring s{desc.Description};
                        vstd::string ss(s.size(), ' ');
                        std::transform(s.cbegin(), s.cend(), ss.begin(), [](auto c) noexcept { return static_cast<char>(c); });
                        LUISA_INFO("Found capable DirectX device at index {}: {}.", index, ss);
                        adapterID = GenAdapterGUID(desc);
                        break;
                    }
                }
                mDevice = nullptr;
                mAdapter = nullptr;
            }
            device = mDevice.Get();
            adapter = mAdapter.Get();
            if (mAdapter == nullptr) { LUISA_ERROR_WITH_LOCATION("Failed to create DirectX device at index {}.", index); }
        }
        defaultAllocator = vstd::make_unique<GpuAllocator>(this);
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
        auto samplers = GlobalSamplers::GetSamplers();
        for (auto i : vstd::range(samplers.size())) {
            samplerHeap->CreateSampler(
                samplers[i], i);
        }
    }
    {
        std::lock_guard lck(gDxcMutex);
        if (gDxcRefCount == 0)
            gDxcCompiler.New(ctx.paths().runtime_directory());
        gDxcRefCount++;
    }
}
bool Device::SupportMeshShader() const {
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 featureData = {};
    device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &featureData, sizeof(featureData));
    return (featureData.MeshShaderTier >= D3D12_MESH_SHADER_TIER_1);
}

BinaryStream::BinaryStream(vstd::string const &path)
    : reader(path) {}
size_t BinaryStream::length() const {
    return reader.GetLength();
}
size_t BinaryStream::pos() const {
    return reader.GetPos();
}
void BinaryStream::read(vstd::span<std::byte> dst) {
    reader.Read(dst.data(), dst.size_bytes());
}
BinaryStream::~BinaryStream() {}
}// namespace toolhub::directx