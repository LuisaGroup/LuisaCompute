#include "rw_resource.h"
#include "stream.h"
#include <core/basic_traits.h>
#include <core/logging.h>
namespace lc::validation {
RWResource::RWResource(uint64_t handle, Tag tag, bool non_simultaneous)
    : Resource{handle, tag}, _non_simultaneous{non_simultaneous} {
}
void RWResource::set_usage(Stream *stream, RWResource *res, Usage usage, Range range) {
    if (usage == Usage::NONE) [[likely]]
        return;
    {
        auto iter = stream->res_usages.try_emplace(res);
        auto &ite_usage = iter.first->second;
        ite_usage.usage = static_cast<Usage>(luisa::to_underlying(ite_usage.usage) | luisa::to_underlying(usage));
        [&] {
            for (auto &&i : ite_usage.ranges) {
                if (i == range) return;
            }
            ite_usage.ranges.emplace_back(range);
        }();
    }
    {
        auto iter = res->_info.try_emplace(stream);
        auto &info = iter.first->second;
        if (stream->executed_layer() > info.last_frame) {
            info.last_frame = stream->executed_layer();
            info.usage = usage;
            info.ranges.clear();
        } else {
            info.usage = static_cast<Usage>(luisa::to_underlying(info.usage) | luisa::to_underlying(usage));
        }
        [&] {
            for (auto &&i : info.ranges) {
                if (i == range) return;
            }
            info.ranges.emplace_back(range);
        }();
    }
}
RWResource::~RWResource() {
    for (auto &&i : _info) {
        if (i.second.last_frame > i.first->synced_layer()) {
            LUISA_ERROR("Resource \"{}\" destroyed when {} is still using it.", get_name(), i.first->get_name());
        }
    }
    _info.clear();
}
}// namespace lc::validation