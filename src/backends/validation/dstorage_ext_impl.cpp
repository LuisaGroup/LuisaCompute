#include "dstorage_ext_impl.h"
#include "rw_resource.h"
#include "device.h"
#include "stream.h"
#include <runtime/rhi/command.h>

namespace lc::validation {
void DStorageExtImpl::gdeflate_compress(
    luisa::span<std::byte const> input,
    CompressQuality quality,
    luisa::vector<std::byte> &result) noexcept {
    _impl->gdeflate_compress(input, quality, result);
}
DStorageExtImpl::File DStorageExtImpl::open_file_handle(luisa::string_view path) noexcept {
    auto file = _impl->open_file_handle(path);
    if (file.valid())
        new RWResource(file.handle, RWResource::Tag::DSTORAGE_FILE, false);
    return file;
}
void DStorageExtImpl::close_file_handle(uint64_t handle) noexcept {
    _impl->close_file_handle(handle);
    RWResource::dispose(handle);
}
DeviceInterface *DStorageExtImpl::device() const noexcept {
    return _impl->device();
}
ResourceCreationInfo DStorageExtImpl::create_stream_handle() noexcept {
    auto p = _impl->create_stream_handle();
    if (!p.valid()) return p;
    new Stream(p.handle, StreamTag::CUSTOM);
    StreamOption opt;
    opt.func = static_cast<StreamFunc>(
        luisa::to_underlying(StreamFunc::Custom) |
        luisa::to_underlying(StreamFunc::Signal) |
        luisa::to_underlying(StreamFunc::Sync));
    opt.supported_custom.emplace(dstorage_command_uuid);
    Device::add_custom_stream(p.handle, std::move(opt));
    return p;
}
DStorageExtImpl::DStorageExtImpl(DStorageExt *ext) : _impl{ext} {}
}// namespace lc::validation