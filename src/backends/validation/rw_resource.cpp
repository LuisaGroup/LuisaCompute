#include "rw_resource.h"
#include "stream.h"
#include <core/basic_traits.h>
#include <core/logging.h>
namespace lc::validation {
RWResource::RWResource(uint64_t handle, Tag tag, bool non_simultaneous)
    : Resource{handle, tag}, _non_simultaneous{non_simultaneous} {
}
void RWResource::set_usage(Stream *stream, RWResource *res, Usage usage) {
    LUISA_ASSERT(usage == Usage::READ || usage == Usage::WRITE, "Usage need to be read or write.");
    {
        auto iter = stream->res_usages.try_emplace(res, Usage::NONE);
        iter.first->second = static_cast<Usage>(luisa::to_underlying(iter.first->second) | luisa::to_underlying(usage));
    }
    {
        auto iter = res->_info.try_emplace(stream);
        auto &info = iter.first->second;
        if (stream->executed_layer() > info.last_frame) {
            info.last_frame = stream->executed_layer();
            info.usage = usage;
        } else {
            info.usage = static_cast<Usage>(luisa::to_underlying(info.usage) | luisa::to_underlying(usage));
        }
    }
}
RWResource::~RWResource() {
    for (auto &&i : _info) {
        if (i.second.last_frame > i.first->synced_layer()) {
            LUISA_ERROR("Resource \"{}\" destroyed when {} is still using it.", get_name(), i.first->get_name());
        }
    }
}
}// namespace lc::validation