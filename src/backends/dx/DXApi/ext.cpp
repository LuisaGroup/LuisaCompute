#include "../ext.h"
#include <DXApi/LCDevice.h>
#include <DXRuntime/Device.h>
#include <Resource/RenderTexture.h>
#include <DXApi/LCCmdBuffer.h>
#include <runtime/stream.h>
#include <Resource/ExternalBuffer.h>
#include <Resource/ExternalTexture.h>
#include <Resource/ExternalDepth.h>
#include <Resource/Buffer.h>
#include <DXApi/LCEvent.h>
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
    LUISA_INFO("start try compile setAccelKernel");
    if (!device->setAccelKernel.Check(device)) return Result::Failed;
    LUISA_INFO("start try compile bc6TryModeG10");
    if (!device->bc6TryModeG10.Check(device)) return Result::Failed;
    LUISA_INFO("start try compile bc6TryModeLE10");
    if (!device->bc6TryModeLE10.Check(device)) return Result::Failed;
    LUISA_INFO("start try compile bc6EncodeBlock");
    if (!device->bc6EncodeBlock.Check(device)) return Result::Failed;
    LUISA_INFO("start try compile bc7TryMode456");
    if (!device->bc7TryMode456.Check(device)) return Result::Failed;
    LUISA_INFO("start try compile bc7TryMode137");
    if (!device->bc7TryMode137.Check(device)) return Result::Failed;
    LUISA_INFO("start try compile bc7TryMode02");
    if (!device->bc7TryMode02.Check(device)) return Result::Failed;
    LUISA_INFO("start try compile bc7EncodeBlock");
    if (!device->bc7EncodeBlock.Check(device)) return Result::Failed;
    return Result::Success;
}
DxNativeResourceExt::DxNativeResourceExt(DeviceInterface *lc_device, Device *dx_device)
    : NativeResourceExt{lc_device}, dx_device{dx_device} {
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
    info.handle = reinterpret_cast<uint64>(res);
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
DStorageExtImpl::DStorageExtImpl(std::filesystem::path const &runtime_dir, ID3D12Device *device) noexcept
    : dstorage_core_module{DynamicModule::load(runtime_dir, "dstoragecore")},
      dstorage_module{DynamicModule::load(runtime_dir, "dstorage")},
      device{device} {
    HRESULT(WINAPI * DStorageGetFactory)
    (REFIID riid, _COM_Outptr_ void **ppv);
    if (!dstorage_module || !dstorage_core_module) {
        LUISA_WARNING("Direct-Storage DLL not found.");
        return;
    }
    DStorageGetFactory = reinterpret_cast<decltype(DStorageGetFactory)>(GetProcAddress(reinterpret_cast<HMODULE>(dstorage_module.handle()), "DStorageGetFactory"));
    ThrowIfFailed(DStorageGetFactory(IID_PPV_ARGS(factory.GetAddressOf())));
}
class DStorageFileImpl : public vstd::IOperatorNewBase {
public:
    ComPtr<IDStorageFile> file;
    size_t size_bytes;
    DStorageFileImpl(ComPtr<IDStorageFile> &&file, size_t size_bytes) : file{std::move(file)}, size_bytes{size_bytes} {}
};
class DStorageQueueImpl : public vstd::IOperatorNewBase {
public:
    ComPtr<IDStorageQueue> queue;
    DStorageQueueImpl(IDStorageFactory *factory, ID3D12Device *device) {
        DSTORAGE_QUEUE_DESC queue_desc{
            .SourceType = DSTORAGE_REQUEST_SOURCE_FILE,
            .Capacity = DSTORAGE_MAX_QUEUE_CAPACITY,
            .Priority = DSTORAGE_PRIORITY_NORMAL,
            .Device = device};
        ThrowIfFailed(factory->CreateQueue(&queue_desc, IID_PPV_ARGS(queue.GetAddressOf())));
    }
};
ResourceCreationInfo DStorageExtImpl::create_stream_handle() noexcept {
    ResourceCreationInfo r;
    auto ptr = new DStorageQueueImpl{factory.Get(), device};
    r.handle = reinterpret_cast<uint64_t>(ptr);
    r.native_handle = ptr->queue.Get();
    return r;
}
DStorageExtImpl::File DStorageExtImpl::open_file_handle(luisa::string_view path) noexcept {
    ComPtr<IDStorageFile> file;
    luisa::vector<wchar_t> wstr;
    wstr.push_back_uninitialized(path.size() + 1);
    wstr[path.size()] = 0;
    for (size_t i = 0; i < path.size(); ++i) {
        wstr[i] = path[i];
    }
    HRESULT hr = factory->OpenFile(wstr.data(), IID_PPV_ARGS(file.GetAddressOf()));
    DStorageExtImpl::File f;
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
void DStorageExtImpl::close_file_handle(uint64_t handle) noexcept {
    delete reinterpret_cast<DStorageFileImpl *>(handle);
}
void DStorageExtImpl::destroy_stream_handle(uint64_t handle) noexcept {
    delete reinterpret_cast<DStorageQueueImpl *>(handle);
}
void DStorageExtImpl::enqueue_buffer(uint64_t stream_handle, uint64_t file_handle, size_t file_offset, uint64_t buffer_handle, size_t buffer_offset, size_t size_bytes) noexcept {
    auto queue = reinterpret_cast<DStorageQueueImpl *>(stream_handle)->queue.Get();
    auto file = reinterpret_cast<DStorageFileImpl *>(file_handle);
    if (file_offset + size_bytes > file->size_bytes) {
        LUISA_ERROR("Direct-Storage enqueue_buffer out of bound, required size: {}, file size: {}.", file_offset + size_bytes, file->size_bytes);
    }
    DSTORAGE_REQUEST request = {};
    request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
    request.Source.File.Source = file->file.Get();
    request.Source.File.Offset = file_offset;
    request.Source.File.Size = size_bytes;
    request.UncompressedSize = 0;
    request.Destination.Buffer.Offset = buffer_offset;
    request.Destination.Buffer.Size = size_bytes;
    request.Destination.Buffer.Resource = reinterpret_cast<Buffer *>(buffer_handle)->GetResource();
    queue->EnqueueRequest(&request);
}
void DStorageExtImpl::enqueue_memory(uint64_t stream_handle, uint64_t file_handle, size_t file_offset, void *dst_ptr, size_t size_bytes) noexcept {
    auto queue = reinterpret_cast<DStorageQueueImpl *>(stream_handle)->queue.Get();
    auto file = reinterpret_cast<DStorageFileImpl *>(file_handle);
    if (file_offset + size_bytes > file->size_bytes) {
        LUISA_ERROR("Direct-Storage enqueue_buffer out of bound, required size: {}, file size: {}.", file_offset + size_bytes, file->size_bytes);
    }
    DSTORAGE_REQUEST request = {};
    request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_MEMORY;
    request.Source.File.Source = file->file.Get();
    request.Source.File.Offset = file_offset;
    request.Source.File.Size = size_bytes;
    request.UncompressedSize = 0;
    request.Destination.Memory.Buffer = dst_ptr;
    request.Destination.Memory.Size = size_bytes;
    queue->EnqueueRequest(&request);
}
void DStorageExtImpl::enqueue_image(uint64_t stream_handle, uint64_t file_handle, size_t file_offset, uint64_t image_handle, size_t size_bytes, uint32_t mip) noexcept {
    auto queue = reinterpret_cast<DStorageQueueImpl *>(stream_handle)->queue.Get();
    auto file = reinterpret_cast<DStorageFileImpl *>(file_handle);
    if (file_offset + size_bytes > file->size_bytes) {
        LUISA_ERROR("Direct-Storage enquue_image out of bound, required size: {}, file size: {}.", file_offset + size_bytes, file->size_bytes);
    }
    DSTORAGE_REQUEST request = {};
    request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_TEXTURE_REGION;
    request.Source.File.Source = file->file.Get();
    request.Source.File.Offset = file_offset;
    request.Source.File.Size = size_bytes;
    request.UncompressedSize = 0;
    auto tex = reinterpret_cast<TextureBase *>(image_handle);
    request.Destination.Texture.SubresourceIndex = mip;
    request.Destination.Texture.Resource = tex->GetResource();
    request.Destination.Texture.Region = D3D12_BOX{
        0u, 0u, 0u,
        tex->Width(), tex->Height(), tex->Depth()};
    queue->EnqueueRequest(&request);
}
void DStorageExtImpl::signal(uint64_t stream_handle, uint64_t event_handle) noexcept {
    auto queue = reinterpret_cast<DStorageQueueImpl *>(stream_handle)->queue.Get();
    auto event = reinterpret_cast<LCEvent *>(event_handle);
    auto idx = ++event->fenceIndex;
    {
        std::lock_guard lck(event->eventMtx);
        event->currentThreadSync = true;
        queue->EnqueueSignal(event->Fence(), ++event->fenceIndex);
    }
}
void DStorageExtImpl::commit(uint64_t stream_handle) noexcept {
    auto queue = reinterpret_cast<DStorageQueueImpl *>(stream_handle)->queue.Get();
    queue->Submit();
}
}// namespace lc::dx
