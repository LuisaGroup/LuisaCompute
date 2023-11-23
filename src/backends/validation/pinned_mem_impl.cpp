#include "pinned_mem_impl.h"
#include "rw_resource.h"
#include <luisa/core/logging.h>
namespace lc::validation {

BufferCreationInfo PinnedMemoryExtImpl::_pin_host_memory(
    const Type *elem_type, size_t elem_count,
    void *host_ptr, const PinnedMemoryOption &option) noexcept {
    auto res = _impl->_pin_host_memory(elem_type, elem_count, host_ptr, option);
    if (res.valid())
        new RWResource(res.handle, RWResource::Tag::BUFFER, false);
    return res;
}
BufferCreationInfo PinnedMemoryExtImpl::_allocate_pinned_memory(
    const Type *elem_type, size_t elem_count,
    const PinnedMemoryOption &option) noexcept {
    auto res = _impl->_allocate_pinned_memory(elem_type, elem_count, option);
    if (res.valid())
        new RWResource(res.handle, RWResource::Tag::BUFFER, false);
    return res;
}
}// namespace lc::validation