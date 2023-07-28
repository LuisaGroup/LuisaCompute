#include "dstorage_ext_impl.h"
#include "rw_resource.h"
#include "device.h"
#include "stream.h"
#include <luisa/runtime/rhi/command.h>
#include <luisa/backends/ext/registry.h>

namespace lc::validation {

DStorageExtImpl::FileCreationInfo DStorageExtImpl::open_file_handle(luisa::string_view path) noexcept {
    auto file = _impl->open_file_handle(path);
    if (file.valid()) {
        new RWResource(file.handle, RWResource::Tag::DSTORAGE_FILE, false);
    }
    return file;
}
void DStorageExtImpl::close_file_handle(uint64_t handle) noexcept {
    _impl->close_file_handle(handle);
    RWResource::dispose(handle);
}
DeviceInterface *DStorageExtImpl::device() const noexcept {
    return _self;
}
ResourceCreationInfo DStorageExtImpl::create_stream_handle(const DStorageStreamOption &option) noexcept {
    auto p = _impl->create_stream_handle(option);
    if (!p.valid()) return p;
    new Stream(p.handle, StreamTag::CUSTOM);
    StreamOption opt;
    opt.func = static_cast<StreamFunc>(
        luisa::to_underlying(StreamFunc::Custom) |
        luisa::to_underlying(StreamFunc::Signal) |
        luisa::to_underlying(StreamFunc::Sync));
    opt.supported_custom.emplace(to_underlying(CustomCommandUUID::DSTORAGE_READ));
    Device::add_custom_stream(p.handle, std::move(opt));
    return p;
}
DStorageExtImpl::DStorageExtImpl(DStorageExt *ext, DeviceInterface *self) : _impl{ext}, _self{self} {}

DStorageExt::PinnedMemoryInfo DStorageExtImpl::pin_host_memory(void *ptr, size_t size_bytes) noexcept {
    auto p = _impl->pin_host_memory(ptr, size_bytes);
    if (p.valid()) {
        new RWResource(p.handle, RWResource::Tag::DSTORAGE_PINNED_MEMORY, false);
    }
    return p;
}

void DStorageExtImpl::unpin_host_memory(uint64_t handle) noexcept {
    _impl->unpin_host_memory(handle);
    RWResource::dispose(handle);
}
void DStorageExtImpl::compress(const void *data, size_t size_bytes,
                               DStorageExt::Compression algorithm,
                               DStorageExt::CompressionQuality quality,
                               vector<std::byte> &result) noexcept {
    _impl->compress(data, size_bytes, algorithm, quality, result);
}

}// namespace lc::validation

