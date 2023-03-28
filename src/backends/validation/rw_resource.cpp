#include "rw_resource.h"
#include "stream.h"
#include <core/basic_traits.h>
namespace lc::validation {
RWResource::RWResource(uint64_t handle, bool non_simultaneous)
    : Resource{handle}, _non_simultaneous{non_simultaneous} {
}
void RWResource::_set(Stream const *stream, Usage usage) {
    auto iter = _info.try_emplace(stream);
    auto &info = iter.first->second;
    if (stream->executed_layer() > info.last_frame) {
        info.last_frame = stream->executed_layer();
        info.usage = usage;
    } else {
        info.usage = static_cast<Usage>(luisa::to_underlying(info.usage) | luisa::to_underlying(usage));
    }
}

void RWResource::read(Stream const *stream) {
    _set(stream, Usage::READ);
}
void RWResource::write(Stream const *stream) {
    _set(stream, Usage::WRITE);
}
}// namespace lc::validation