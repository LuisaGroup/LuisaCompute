#include "ext.h"
#include <DXApi/LCDevice.h>
#include <DXRuntime/Device.h>
#include <Resource/RenderTexture.h>
#include <DXApi/LCCmdBuffer.h>
#include <luisa/runtime/stream.h>
#include <Resource/ExternalBuffer.h>
#include <Resource/ExternalTexture.h>
#include <Resource/ExternalDepth.h>
#include <Resource/UploadBuffer.h>
#include <Resource/ReadbackBuffer.h>
#include <DXApi/LCEvent.h>
#include <DXApi/LCDevice.h>
#include <DXApi/LCSwapChain.h>
#include <DXRuntime/DStorageCommandQueue.h>
#include <DXApi/TypeCheck.h>
#include <luisa/runtime/image.h>
namespace lc::dx {
// IUtil *LCDevice::get_util() noexcept {
//     if (!util) {
//         util = vstd::create_unique(new DxTexCompressExt(&nativeDevice));
//     }
//     return util.get();
// }
DxTexCompressExt::DxTexCompressExt(Device *device)
    : device(device) {
}
DxTexCompressExt::~DxTexCompressExt() {
}
TexCompressExt::Result DxTexCompressExt::compress_bc6h(Stream &stream, ImageView<float> const &src, luisa::compute::BufferView<uint> const &result) noexcept {
    LCCmdBuffer *cmdBuffer = reinterpret_cast<LCCmdBuffer *>(stream.handle());

    TextureBase *srcTex = reinterpret_cast<TextureBase *>(src.handle());
    cmdBuffer->CompressBC(
        srcTex,
        result,
        true,
        0,
        device->defaultAllocator.get(),
        device->maxAllocatorCount);
    return Result::Success;
}

TexCompressExt::Result DxTexCompressExt::compress_bc7(Stream &stream, ImageView<float> const &src, luisa::compute::BufferView<uint> const &result, float alphaImportance) noexcept {
    LCCmdBuffer *cmdBuffer = reinterpret_cast<LCCmdBuffer *>(stream.handle());
    cmdBuffer->CompressBC(
        reinterpret_cast<TextureBase *>(src.handle()),
        result,
        false,
        alphaImportance,
        device->defaultAllocator.get(),
        device->maxAllocatorCount);
    return Result::Success;
}
TexCompressExt::Result DxTexCompressExt::check_builtin_shader() noexcept {
    LUISA_VERBOSE("start try compile setAccelKernel");
    if (!device->setAccelKernel.Check(device)) return Result::Failed;
    LUISA_VERBOSE("start try compile bc6TryModeG10");
    if (!device->bc6TryModeG10.Check(device)) return Result::Failed;
    LUISA_VERBOSE("start try compile bc6TryModeLE10");
    if (!device->bc6TryModeLE10.Check(device)) return Result::Failed;
    LUISA_VERBOSE("start try compile bc6EncodeBlock");
    if (!device->bc6EncodeBlock.Check(device)) return Result::Failed;
    LUISA_VERBOSE("start try compile bc7TryMode456");
    if (!device->bc7TryMode456.Check(device)) return Result::Failed;
    LUISA_VERBOSE("start try compile bc7TryMode137");
    if (!device->bc7TryMode137.Check(device)) return Result::Failed;
    LUISA_VERBOSE("start try compile bc7TryMode02");
    if (!device->bc7TryMode02.Check(device)) return Result::Failed;
    LUISA_VERBOSE("start try compile bc7EncodeBlock");
    if (!device->bc7EncodeBlock.Check(device)) return Result::Failed;
    return Result::Success;
}
DxNativeResourceExt::DxNativeResourceExt(DeviceInterface *lc_device, Device *dx_device)
    : NativeResourceExt{lc_device}, dx_device{dx_device} {
}
uint64_t DxNativeResourceExt::get_native_resource_device_address(
    void *native_handle) noexcept {
    return reinterpret_cast<ID3D12Resource *>(native_handle)->GetGPUVirtualAddress();
}
BufferCreationInfo DxNativeResourceExt::register_external_buffer(
    void *external_ptr,
    const Type *element,
    size_t elem_count,
    void *custom_data) noexcept {
    auto res = static_cast<Buffer *>(new ExternalBuffer(
        dx_device,
        reinterpret_cast<ID3D12Resource *>(external_ptr),
        *reinterpret_cast<D3D12_RESOURCE_STATES const *>(custom_data)));
    BufferCreationInfo info;
    info.handle = resource_to_handle(res);
    info.native_handle = res->GetResource();
    info.element_stride = element->size();
    info.total_size_bytes = element->size() * elem_count;
    return info;
}
ResourceCreationInfo DxNativeResourceExt::register_external_image(
    void *external_ptr,
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels,
    void *custom_data) noexcept {
    auto desc = reinterpret_cast<NativeTextureDesc const *>(custom_data);
    GFXFormat gfxFormat;
    if (desc->custom_format == DXGI_FORMAT_UNKNOWN) {
        gfxFormat = TextureBase::ToGFXFormat(format);
    } else {
        gfxFormat = static_cast<GFXFormat>(desc->custom_format);
    }
    auto res = static_cast<TextureBase *>(new ExternalTexture(
        dx_device,
        reinterpret_cast<ID3D12Resource *>(external_ptr),
        desc->initState,
        width,
        height,
        gfxFormat,
        (TextureDimension)dimension,
        depth,
        mipmap_levels,
        desc->allowUav));
    return {
        reinterpret_cast<uint64_t>(res),
        external_ptr};
}
ResourceCreationInfo DxNativeResourceExt::register_external_depth_buffer(
    void *external_ptr,
    DepthFormat format,
    uint width,
    uint height,
    // custom data see backends' header
    void *custom_data) noexcept {
    auto res = static_cast<TextureBase *>(new ExternalDepth(
        reinterpret_cast<ID3D12Resource *>(external_ptr),
        dx_device,
        width,
        height,
        format,
        *reinterpret_cast<D3D12_RESOURCE_STATES const *>(custom_data)));
    return {
        reinterpret_cast<uint64_t>(res),
        external_ptr};
}
SwapchainCreationInfo DxNativeResourceExt::register_external_swapchain(
    void *swapchain_ptr,
    bool vsync) noexcept {
    SwapchainCreationInfo info;
    auto res = new LCSwapChain(
        info.storage,
        dx_device,
        reinterpret_cast<IDXGISwapChain3 *>(swapchain_ptr),
        vsync);
    info.handle = reinterpret_cast<uint64_t>(res);
    info.native_handle = swapchain_ptr;
    return info;
}
void DStorageExtImpl::init_factory_nolock() {
    HRESULT(WINAPI * DStorageGetFactory)
    (REFIID riid, _COM_Outptr_ void **ppv);
    if (!dstorage_module || !dstorage_core_module) {
        LUISA_WARNING("Direct-Storage DLL not found.");
        return;
    }
    DStorageGetFactory = dstorage_module.function<std::remove_pointer_t<decltype(DStorageGetFactory)>>("DStorageGetFactory");
    DStorageGetFactory(IID_PPV_ARGS(factory.GetAddressOf()));
}
void DStorageExtImpl::init_factory() {
    {
        std::lock_guard lck{spin_mtx};
        if (factory) [[likely]] {
            return;
        }
    }
    std::lock_guard lck{mtx};
    if (factory) [[unlikely]] {
        return;
    }
    init_factory_nolock();
}
DStorageExtImpl::DStorageExtImpl(std::filesystem::path const &runtime_dir, LCDevice *device) noexcept
    : dstorage_core_module{DynamicModule::load(runtime_dir, "dstoragecore")},
      dstorage_module{DynamicModule::load(runtime_dir, "dstorage")},
      mdevice{device} {
}
ResourceCreationInfo DStorageExtImpl::create_stream_handle(const DStorageStreamOption &option) noexcept {
    set_config(option.supports_hdd);
    if (option.staging_buffer_size != staging_buffer_size) {
        if (!staging.exchange(true)) {
            factory->SetStagingBufferSize(option.staging_buffer_size);
            staging_buffer_size = option.staging_buffer_size;
        } else {
            LUISA_WARNING("Staging buffer already setted, staging set failed.");
        }
    }
    ResourceCreationInfo r;
    auto ptr = new DStorageCommandQueue{factory.Get(), &mdevice->nativeDevice, option.source};
    ptr->staging_buffer_size = staging_buffer_size;
    r.handle = reinterpret_cast<uint64_t>(ptr);
    r.native_handle = nullptr;
    return r;
}
DStorageExtImpl::FileCreationInfo DStorageExtImpl::open_file_handle(luisa::string_view path) noexcept {
    init_factory();
    ComPtr<IDStorageFile> file;
    luisa::vector<wchar_t> wstr;
    wstr.push_back_uninitialized(path.size() + 1);
    wstr[path.size()] = 0;
    for (size_t i = 0; i < path.size(); ++i) {
        wstr[i] = path[i];
    }
    HRESULT hr = factory->OpenFile(wstr.data(), IID_PPV_ARGS(file.GetAddressOf()));
    DStorageExtImpl::FileCreationInfo f;
    if (FAILED(hr)) {
        f.invalidate();
        return f;
    }
    size_t length;
    BY_HANDLE_FILE_INFORMATION info{};
    ThrowIfFailed(file->GetFileInformation(&info));
    if constexpr (sizeof(size_t) > sizeof(DWORD)) {
        length = info.nFileSizeHigh;
        length <<= (sizeof(DWORD) * 8);
        length |= info.nFileSizeLow;
    } else {
        length = info.nFileSizeLow;
    }
    if (length == 0) {
        f.invalidate();
        return f;
    }
    f.native_handle = file.Get();
    f.handle = reinterpret_cast<uint64_t>(new DStorageFileImpl{std::move(file), length});
    f.size_bytes = length;
    return f;
}
DeviceInterface *DStorageExtImpl::device() const noexcept {
    return mdevice;
}
void DStorageExtImpl::close_file_handle(uint64_t handle) noexcept {
    delete reinterpret_cast<DStorageFileImpl *>(handle);
}
void DStorageExtImpl::compress(
    const void *data, size_t size_bytes,
    Compression algorithm, CompressionQuality quality,
    luisa::vector<std::byte> &result) noexcept {
    constexpr DSTORAGE_COMPRESSION qua[] = {
        DSTORAGE_COMPRESSION_FASTEST,
        DSTORAGE_COMPRESSION_DEFAULT,
        DSTORAGE_COMPRESSION_BEST_RATIO};

    result.clear();
    size_t out_size{};
    [&]() {
        {
            std::lock_guard lck{spin_mtx};
            if (compression_codec) [[likely]] {
                return;
            }
        }
        std::lock_guard lck{mtx};
        if (compression_codec) [[unlikely]] {
            return;
        }
        HRESULT(WINAPI * DStorageCreateCompressionCodec)
        (DSTORAGE_COMPRESSION_FORMAT format, UINT32 numThreads, REFIID riid, _COM_Outptr_ void **ppv);
        DStorageCreateCompressionCodec = dstorage_module.function<std::remove_pointer_t<decltype(DStorageCreateCompressionCodec)>>("DStorageCreateCompressionCodec");
        DStorageCreateCompressionCodec(DSTORAGE_COMPRESSION_FORMAT_GDEFLATE, std::thread::hardware_concurrency(), IID_PPV_ARGS(compression_codec.GetAddressOf()));
    }();
    result.push_back_uninitialized(compression_codec->CompressBufferBound(size_bytes));
    ThrowIfFailed(compression_codec->CompressBuffer(
        data,
        size_bytes,
        qua[luisa::to_underlying(quality)],
        result.data(),
        result.size(),
        &out_size));
    result.resize(out_size);
}
void DStorageExtImpl::set_config(bool hdd) noexcept {
    std::lock_guard lck{mtx};
    if (hdd == is_hdd) {
        if (!factory) {
            init_factory_nolock();
        }
        return;
    }
    is_hdd = hdd;
    if (factory) [[unlikely]] {
        LUISA_ERROR("set_config can only be called before first open_file and create_stream");
    }
    HRESULT(WINAPI * DStorageSetConfiguration1)
    (DSTORAGE_CONFIGURATION1 const *configuration);
    DStorageSetConfiguration1 = dstorage_module.function<std::remove_pointer_t<decltype(DStorageSetConfiguration1)>>("DStorageSetConfiguration1");
    if (hdd) {
        DSTORAGE_CONFIGURATION1 cfg{
            .DisableBypassIO = true,
            .ForceFileBuffering = true};
        DStorageSetConfiguration1(&cfg);
    } else {
        DSTORAGE_CONFIGURATION1 cfg{};
        DStorageSetConfiguration1(&cfg);
    }
    init_factory_nolock();
}
BufferCreationInfo DxPinnedMemoryExt::_pin_host_memory(
    const Type *elem_type, size_t elem_count,
    void *host_ptr, const PinnedMemoryOption &option) noexcept {
    LUISA_ERROR("DX backend can not pin host memory.");
    return BufferCreationInfo::make_invalid();
}

DeviceInterface *DxPinnedMemoryExt::device() const noexcept {
    return _device;
}

BufferCreationInfo DxPinnedMemoryExt::_allocate_pinned_memory(
    const Type *elem_type, size_t elem_count,
    const PinnedMemoryOption &option) noexcept {
    BufferCreationInfo info{};
    if (elem_type == Type::of<void>()) {
        info.total_size_bytes = elem_count;
        info.element_stride = 1u;
    } else {
        LUISA_ASSERT(!elem_type->is_custom(), "Custom type not allowed.");
        info.element_stride = elem_type->size();
        info.total_size_bytes = info.element_stride * elem_count;
    }
    if (option.write_combined) {
        auto res = new UploadBuffer(
            &_device->nativeDevice,
            info.total_size_bytes,
            _device->nativeDevice.defaultAllocator.get());
        info.handle = resource_to_handle(res);
        info.native_handle = res->MappedPtr();
    } else {
        auto res = new ReadbackBuffer(
            &_device->nativeDevice,
            info.total_size_bytes,
            _device->nativeDevice.defaultAllocator.get());
        info.handle = resource_to_handle(res);
        info.native_handle = res->MappedPtr();
    }
    return info;
}

}// namespace lc::dx
#ifdef LUISA_BACKEND_ENABLE_OIDN
#include <DXApi/dx_oidn_denoiser_ext.h>
namespace lc::dx {
auto DXOidnDenoiser::get_buffer(const DenoiserExt::Image &img, bool read) noexcept -> oidn::BufferRef {
    // TODO: fix this
    // TODO: don't create shared buffer if given buffer is already shared
    auto interop_buffer = _interop->create_interop_buffer(nullptr, img.size_bytes);
    auto buffer = static_cast<DefaultBuffer *>(reinterpret_cast<Buffer *>(interop_buffer.handle));
    uint64_t cuda_device_ptr, cuda_handle;
    _interop->cuda_buffer(interop_buffer.handle, &cuda_device_ptr, &cuda_handle);
    auto oidn_buffer = _oidn_device.newBuffer(
        (void *)cuda_device_ptr,
        buffer->GetByteSize());
    LUISA_ASSERT(oidn_buffer, "OIDN buffer creation failed.");
    _interop_images.push_back(InteropImage{.img = img, .shared_buffer = interop_buffer, .read = read});
    return oidn_buffer;
}
void DXOidnDenoiser::reset() noexcept {
    OidnDenoiser::reset();
    for (auto &&img : _interop_images) {
        _device->destroy_buffer(img.shared_buffer.handle);
    }
    _interop_images.clear();
}

void DXOidnDenoiser::prepare() noexcept {
    auto cmd_list = CommandList{};
    for (auto &&img : _interop_images) {
        if (img.read) {
            cmd_list.append(std::move(luisa::make_unique<BufferCopyCommand>(
                img.img.buffer_handle,
                img.shared_buffer.handle,
                img.img.offset,
                0ull,
                img.img.size_bytes)));
        }
    }

    _device->dispatch(_stream, std::move(cmd_list.commit()).command_list());
}
void DXOidnDenoiser::post_sync() noexcept {
    auto cmd_list = CommandList{};
    for (auto &&img : _interop_images) {
        if (!img.read) {
            cmd_list.append(std::move(luisa::make_unique<BufferCopyCommand>(
                img.shared_buffer.handle,
                img.img.buffer_handle,
                0ull,
                img.img.offset,
                img.img.size_bytes)));
        }
    }

    _device->dispatch(_stream, std::move(cmd_list.commit()).command_list());
}
void DXOidnDenoiser::execute(bool async) noexcept {
    if (async) {
        LUISA_WARNING_WITH_LOCATION("Async execution not implemented due to lacking cuda/dx event interop");
    }
    prepare();
    _device->synchronize_stream(_stream);
    exec_filters();
    _oidn_device.sync();
    post_sync();
    _device->synchronize_stream(_stream);
}
DXOidnDenoiser::DXOidnDenoiser(LCDevice *_device, oidn::DeviceRef &&oidn_device, uint64_t stream)
    : OidnDenoiser(static_cast<DeviceInterface *>(_device), std::move(oidn_device), stream) {
    _interop = static_cast<DxCudaInterop *>(_device->extension(DxCudaInterop::name));
    if (_interop == nullptr) {
        LUISA_ERROR_WITH_LOCATION("DxCudaInterop not found. Cannot use OIDN denoiser.");
    }
}
DXOidnDenoiserExt::DXOidnDenoiserExt(LCDevice *device) noexcept
    : _device{device} {}
luisa::shared_ptr<DenoiserExt::Denoiser> DXOidnDenoiserExt::create(uint64_t stream) noexcept {
    DXGI_ADAPTER_DESC1 desc;
    _device->nativeDevice.adapter->GetDesc1(&desc);
    auto device_id = desc.DeviceId;
    LUISA_ASSERT(device_id != -1, "device_id should not be -1.");
    return luisa::make_shared<DXOidnDenoiser>(_device, oidn::newCUDADevice(device_id, 0), stream);
}
luisa::shared_ptr<DenoiserExt::Denoiser> DXOidnDenoiserExt::create(Stream &stream) noexcept {
    return create(stream.handle());
}
}// namespace lc::dx
#endif