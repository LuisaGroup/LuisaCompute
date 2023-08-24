#include "ext.h"
#include <DXApi/LCDevice.h>
#include <DXRuntime/Device.h>
#include <Resource/RenderTexture.h>
#include <DXApi/LCCmdBuffer.h>
#include <luisa/runtime/stream.h>
#include <Resource/ExternalBuffer.h>
#include <Resource/ExternalTexture.h>
#include <Resource/ExternalDepth.h>
#include <Resource/Buffer.h>
#include <DXApi/LCEvent.h>
#include <DXApi/LCDevice.h>
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
TexCompressExt::Result DxTexCompressExt::compress_bc6h(Stream &stream, Image<float> const &src, luisa::compute::BufferView<uint> const &result) noexcept {
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

TexCompressExt::Result DxTexCompressExt::compress_bc7(Stream &stream, Image<float> const &src, luisa::compute::BufferView<uint> const &result, float alphaImportance) noexcept {
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
void DStorageExtImpl::init_factory_nolock() {
    HRESULT(WINAPI * DStorageGetFactory)
    (REFIID riid, _COM_Outptr_ void **ppv);
    if (!dstorage_module || !dstorage_core_module) {
        LUISA_WARNING("Direct-Storage DLL not found.");
        return;
    }
    DStorageGetFactory = dstorage_module.function<std::remove_pointer_t<decltype(DStorageGetFactory)>>("DStorageGetFactory");
    DStorageGetFactory(IID_PPV_ARGS(factory.GetAddressOf()));
    factory->SetStagingBufferSize(staging_buffer_size);
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
    staging_size = DSTORAGE_STAGING_BUFFER_SIZE_32MB;
}
ResourceCreationInfo DStorageExtImpl::create_stream_handle(const DStorageStreamOption &option) noexcept {
    set_config(option.supports_hdd);
    ResourceCreationInfo r;
    auto ptr = new DStorageCommandQueue{factory.Get(), &mdevice->nativeDevice, option.source};
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

}// namespace lc::dx
