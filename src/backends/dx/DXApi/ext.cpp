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
#include <DXApi/LCSwapChain.h>
#include <DXRuntime/DStorageCommandQueue.h>
#include <DXApi/TypeCheck.h>
#include <luisa/runtime/image.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/magic_enum.h>
#include <algorithm>
#include <cstring>
namespace lc::dx {
// IUtil *LCDevice::get_util() noexcept {
//     if (!util) {
//         util = vstd::create_unique(new DxTexCompressExt(&native_device));
//     }
//     return util.get();
// }
DxTexCompressExt::DxTexCompressExt(Device *device)
    : device(device) {
}

TexCompressExt::Result DxTexCompressExt::compress_bc6h(Stream &stream, ImageView<float> const &src, luisa::compute::BufferView<uint> const &result) noexcept {
    auto cmdBuffer = reinterpret_cast<LCCmdBuffer *>(stream.handle());

    auto srcTex = reinterpret_cast<TextureBase *>(src.handle());
    cmdBuffer->CompressBC(
        srcTex,
        src.level(),
        result,
        true,
        0,
        device->default_allocator.get(),
        2);
    return Result::Success;
}

TexCompressExt::Result DxTexCompressExt::compress_bc7(Stream &stream, ImageView<float> const &src, luisa::compute::BufferView<uint> const &result, float alphaImportance) noexcept {
    auto cmdBuffer = reinterpret_cast<LCCmdBuffer *>(stream.handle());
    cmdBuffer->CompressBC(
        reinterpret_cast<TextureBase *>(src.handle()),
        src.level(),
        result,
        false,
        alphaImportance,
        device->default_allocator.get(),
        2);
    return Result::Success;
}
TexCompressExt::Result DxTexCompressExt::check_builtin_shader() noexcept {
    LUISA_VERBOSE("start try compile set_accel_kernel");
    if (!device->set_accel_kernel.check(device)) return Result::Failed;
    LUISA_VERBOSE("start try compile bc6_try_mode_g10");
    if (!device->bc6_try_mode_g10.check(device)) return Result::Failed;
    LUISA_VERBOSE("start try compile bc6_try_mode_le10");
    if (!device->bc6_try_mode_le10.check(device)) return Result::Failed;
    LUISA_VERBOSE("start try compile bc6_encode_block");
    if (!device->bc6_encode_block.check(device)) return Result::Failed;
    LUISA_VERBOSE("start try compile bc7_try_mode_456");
    if (!device->bc7_try_mode_456.check(device)) return Result::Failed;
    LUISA_VERBOSE("start try compile bc7_try_mode_137");
    if (!device->bc7_try_mode_137.check(device)) return Result::Failed;
    LUISA_VERBOSE("start try compile bc7_try_mode_02");
    if (!device->bc7_try_mode_02.check(device)) return Result::Failed;
    LUISA_VERBOSE("start try compile bc7_encode_block");
    if (!device->bc7_encode_block.check(device)) return Result::Failed;
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
        custom_data ? *reinterpret_cast<D3D12_RESOURCE_STATES const *>(custom_data) : D3D12_RESOURCE_STATE_COMMON));
    BufferCreationInfo info{};
    info.handle = resource_to_handle(res);
    info.native_handle = res->GetResource();
    info.element_stride = element->size();
    info.total_size_bytes = element->size() * elem_count;
    return info;
}
ResourceCreationInfo DxNativeResourceExt::register_external_texture(
    void *external_ptr,
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels,
    void *custom_data) noexcept {
    auto desc = reinterpret_cast<NativeTextureDesc const *>(custom_data);
    GFXFormat gfxFormat;
    if (!desc || desc->custom_format == DXGI_FORMAT_UNKNOWN) {
        gfxFormat = TextureBase::ToGFXFormat(format);
    } else {
        gfxFormat = static_cast<GFXFormat>(desc->custom_format);
    }
    auto res = static_cast<TextureBase *>(new ExternalTexture(
        dx_device,
        reinterpret_cast<ID3D12Resource *>(external_ptr),
        desc ? desc->initState : D3D12_RESOURCE_STATE_COMMON,
        width,
        height,
        gfxFormat,
        (TextureDimension)dimension,
        depth,
        mipmap_levels,
        desc ? desc->allowUav : true));
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
    SwapchainCreationInfo info{};
    auto res = new LCSwapChain(
        info.storage,
        dx_device,
        reinterpret_cast<IDXGISwapChain1 *>(swapchain_ptr),
        vsync);
    info.handle = reinterpret_cast<uint64_t>(res);
    info.native_handle = swapchain_ptr;
    return info;
}
bool DStorageExtImpl::_init_factory_nolock() noexcept {
    if (_factory) [[likely]] {
        return true;
    }
    if (!_dstorage_module || !_dstorage_core_module) {
        if (!_dll_warning_issued) {
            _dll_warning_issued = true;
            LUISA_WARNING("DirectStorage runtime (dstorage.dll / dstoragecore.dll) was "
                          "not found next to the application; DirectStorage is "
                          "unavailable on this machine.");
        }
        return false;
    }
    // Take the exact function type (including the calling convention) from the
    // in-tree DirectStorage header instead of re-spelling it.
    using DStorageGetFactoryFn = std::remove_pointer_t<decltype(&DStorageGetFactory)>;
    auto get_factory = _dstorage_module.function<DStorageGetFactoryFn>("DStorageGetFactory");
    if (get_factory == nullptr) [[unlikely]] {
        LUISA_WARNING("dstorage.dll does not export DStorageGetFactory; DirectStorage is unavailable.");
        return false;
    }
    if (FAILED(get_factory(IID_PPV_ARGS(_factory.GetAddressOf()))) || !_factory) [[unlikely]] {
        LUISA_WARNING("DStorageGetFactory failed; DirectStorage is unavailable.");
        return false;
    }
    return true;
}
bool DStorageExtImpl::_init_factory() {
    {
        std::lock_guard lck{_spin_mtx};
        if (_factory) [[likely]] {
            return true;
        }
    }
    std::lock_guard lck{_mtx};
    if (_factory) [[unlikely]] {
        return true;
    }
    return _init_factory_nolock();
}
DStorageExtImpl::DStorageExtImpl(luisa::filesystem::path const &runtime_dir, LCDevice *device) noexcept
    : _dstorage_core_module{DynamicModule::load(runtime_dir, "dstoragecore")},
      _dstorage_module{DynamicModule::load(runtime_dir, "dstorage")},
      _mdevice{device} {
}
DStorageExtImpl::~DStorageExtImpl() noexcept {
    // Streams own the native queues and release them through
    // `LCDevice::destroy_stream`, which must have happened before the device
    // (and therefore this extension) is destroyed.  Release the codec before
    // the factory so a late `close_file_handle` can never touch a dead factory.
    std::lock_guard lck{_mtx};
    _compression_codec.Reset();
    _pinned_ranges.clear();
    _factory.Reset();
}
ResourceCreationInfo DStorageExtImpl::create_stream_handle(const DStorageStreamOption &option) noexcept {
    _set_config(option.supports_hdd);
    if (!_init_factory()) {
        return ResourceCreationInfo::make_invalid();
    }
    // `SetStagingBufferSize` is process-global and must be applied before the
    // first queue is created; later requests must be consistent.
    if (!_staging_applied) {
        _factory->SetStagingBufferSize(static_cast<UINT32>(option.staging_buffer_size));
        _staging_buffer_size = option.staging_buffer_size;
        _staging_applied = true;
    } else if (option.staging_buffer_size != _staging_buffer_size) {
        LUISA_WARNING(
            "DirectStorage's staging buffer size is process-global and was already "
            "set to {} byte(s); ignoring the request for {} byte(s).",
            _staging_buffer_size, option.staging_buffer_size);
    }
    if (option.source == DStorageStreamSource::AnySource) {
        LUISA_INFO("DirectStorage stream created with AnySource: one native queue "
                   "per source type (file + memory).");
    }
    ResourceCreationInfo r{};
    auto ptr = new DStorageCommandQueue{
        this, _factory.Get(), &_mdevice->native_device,
        option.source, _staging_buffer_size};
    r.handle = reinterpret_cast<uint64_t>(ptr);
    r.native_handle = nullptr;
    return r;
}
DStorageExtImpl::FileCreationInfo DStorageExtImpl::open_file_handle(luisa::string_view path) noexcept {
    if (!_init_factory()) {
        return FileCreationInfo::make_invalid();
    }
    ComPtr<IDStorageFile> file;
    luisa::vector<wchar_t> wstr;
    luisa::enlarge_by(wstr, path.size() + 1);
    wstr[path.size()] = 0;
    for (size_t i = 0; i < path.size(); ++i) {
        wstr[i] = path[i];
    }
    HRESULT hr = _factory->OpenFile(wstr.data(), IID_PPV_ARGS(file.GetAddressOf()));
    DStorageExtImpl::FileCreationInfo f{};
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
    return _mdevice;
}
void DStorageExtImpl::close_file_handle(uint64_t handle) noexcept {
    delete reinterpret_cast<DStorageFileImpl *>(handle);
}
size_t DStorageExtImpl::pinned_memory_size(uint64_t handle) noexcept {
    std::lock_guard lck{_mtx};
    auto iter = _pinned_ranges.find(handle);
    return iter == _pinned_ranges.end() ? 0u : iter->second;
}
DStorageExtImpl::PinnedMemoryInfo DStorageExtImpl::pin_host_memory(void *ptr, size_t size_bytes) noexcept {
    // There is no device-side pinning in DX yet: the raw host pointer *is* the
    // handle (see the handle contract in dstorage_ext_interface.h).  The range
    // is recorded so memory-sourced requests can be validated.
    if (ptr == nullptr || size_bytes == 0u) [[unlikely]] {
        LUISA_WARNING("Cannot pin a null or empty host range "
                      "(pointer = {}, size = {} byte(s)).",
                      ptr, size_bytes);
        return PinnedMemoryInfo::make_invalid();
    }
    auto handle = reinterpret_cast<uint64_t>(ptr);
    {
        std::lock_guard lck{_mtx};
        auto iter = _pinned_ranges.try_emplace(handle, size_bytes);
        if (!iter.second) {
            // Overlapping/re-pinned range: keep the widest window so a range
            // check never rejects valid requests.
            iter.first->second = std::max(iter.first->second, size_bytes);
        }
    }
    PinnedMemoryInfo info;
    info.handle = handle;
    info.native_handle = ptr;
    info.size_bytes = size_bytes;
    return info;
}
void DStorageExtImpl::unpin_host_memory(uint64_t handle) noexcept {
    std::lock_guard lck{_mtx};
    _pinned_ranges.erase(handle);
}
void DStorageExtImpl::compress(
    const void *data, size_t size_bytes,
    Compression algorithm, CompressionQuality quality,
    luisa::vector<std::byte> &result) noexcept {
    if (algorithm == Compression::None) {
        result.resize(size_bytes);
        if (size_bytes != 0u) {
            std::memcpy(result.data(), data, size_bytes);
        }
        return;
    }
    if (algorithm != Compression::GDeflate) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Unsupported DirectStorage compression format {}: the DX backend "
            "only provides GDeflate (use DStorageCompression::None for a plain copy).",
            to_string(algorithm));
    }
    if (!_dstorage_module) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "DirectStorage runtime is unavailable; cannot GDeflate-compress. "
            "Use DStorageCompression::None or compress offline.");
    }
    constexpr DSTORAGE_COMPRESSION qua[] = {
        DSTORAGE_COMPRESSION_FASTEST,
        DSTORAGE_COMPRESSION_DEFAULT,
        DSTORAGE_COMPRESSION_BEST_RATIO};

    result.clear();
    size_t out_size{};
    [&]() {
        {
            std::lock_guard lck{_spin_mtx};
            if (_compression_codec) [[likely]] {
                return;
            }
        }
        std::lock_guard lck{_mtx};
        if (_compression_codec) [[unlikely]] {
            return;
        }
        using DStorageCreateCompressionCodecFn =
            std::remove_pointer_t<decltype(&DStorageCreateCompressionCodec)>;
        auto create_codec = _dstorage_module.function<DStorageCreateCompressionCodecFn>(
            "DStorageCreateCompressionCodec");
        if (create_codec == nullptr) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "dstorage.dll does not export DStorageCreateCompressionCodec; "
                "cannot GDeflate-compress. Use DStorageCompression::None or "
                "compress offline.");
        }
        if (FAILED(create_codec(DSTORAGE_COMPRESSION_FORMAT_GDEFLATE,
                                std::thread::hardware_concurrency(),
                                IID_PPV_ARGS(_compression_codec.GetAddressOf()))) ||
            !_compression_codec) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Failed to create the GDeflate compression codec.");
        }
    }();
    luisa::enlarge_by(result, _compression_codec->CompressBufferBound(size_bytes));
    ThrowIfFailed(_compression_codec->CompressBuffer(
        data,
        size_bytes,
        qua[luisa::to_underlying(quality)],
        result.data(),
        result.size(),
        &out_size));
    result.resize(out_size);
}
void DStorageExtImpl::_set_config(bool hdd) noexcept {
    std::lock_guard lck{_mtx};
    if (!_config_set) {
        // First use decides the process-global configuration.  DirectStorage
        // requires it to be set before the factory exists, so an app that
        // called `open_file()` first cannot influence it any more.
        _config_set = true;
        _is_hdd = hdd;
        if (_factory) [[unlikely]] {
            LUISA_WARNING(
                "DirectStorage configuration can no longer be changed: the "
                "factory already exists (open_file() was called before "
                "create_stream()). Ignoring supports_hdd={}.",
                hdd);
        } else if (_dstorage_module) {
            using DStorageSetConfiguration1Fn =
                std::remove_pointer_t<decltype(&DStorageSetConfiguration1)>;
            auto set_configuration = _dstorage_module.function<DStorageSetConfiguration1Fn>(
                "DStorageSetConfiguration1");
            if (set_configuration != nullptr) {
                if (hdd) {
                    DSTORAGE_CONFIGURATION1 cfg{
                        .DisableBypassIO = true,
                        .ForceFileBuffering = true};
                    set_configuration(&cfg);
                } else {
                    DSTORAGE_CONFIGURATION1 cfg{};
                    set_configuration(&cfg);
                }
            } else {
                LUISA_WARNING("dstorage.dll does not export DStorageSetConfiguration1; "
                              "ignoring supports_hdd={}.",
                              hdd);
            }
        }
        _init_factory_nolock();
        return;
    }
    if (hdd != _is_hdd) [[unlikely]] {
        // Fixes the "first stream silently pins the process to non-HDD" trap:
        // the mismatch is reported instead of being fatal or silent.
        LUISA_WARNING(
            "DirectStorage configuration is process-global and was already "
            "latched with supports_hdd={}; ignoring supports_hdd={}.",
            _is_hdd, hdd);
    }
    _init_factory_nolock();
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
            &_device->native_device,
            info.total_size_bytes,
            _device->native_device.default_allocator.get());
        info.handle = resource_to_handle(res);
        info.native_handle = res->MappedPtr();
    } else {
        auto res = new ReadbackBuffer(
            &_device->native_device,
            info.total_size_bytes,
            _device->native_device.default_allocator.get());
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
        reinterpret_cast<void *>(cuda_device_ptr),
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
            cmd_list.append(
                luisa::make_unique<BufferCopyCommand>(
                    img.img.buffer_handle,
                    img.shared_buffer.handle,
                    img.img.offset,
                    0ull,
                    img.img.size_bytes
                )
            );
        }
    }

    _device->dispatch(_stream, std::move(cmd_list.commit()).command_list());
}
void DXOidnDenoiser::post_sync() noexcept {
    auto cmd_list = CommandList{};
    for (auto &&img : _interop_images) {
        if (!img.read) {
            cmd_list.append(
                luisa::make_unique<BufferCopyCommand>(
                    img.shared_buffer.handle,
                    img.img.buffer_handle,
                    0ull,
                    img.img.offset,
                    img.img.size_bytes
                )
            );
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
    auto d3d12_device = _device->native_device.device.Get();
    auto cuda_device = get_cuda_device_for_d3d12_device(d3d12_device);
    return luisa::make_shared<DXOidnDenoiser>(_device, oidn::newCUDADevice(cuda_device, nullptr), stream);
}
luisa::shared_ptr<DenoiserExt::Denoiser> DXOidnDenoiserExt::create(Stream &stream) noexcept {
    return create(stream.handle());
}
}// namespace lc::dx
#endif