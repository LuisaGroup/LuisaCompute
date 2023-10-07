#include "rw_resource.h"
#include "stream.h"
#include <luisa/core/basic_traits.h>
#include <luisa/core/logging.h>
namespace lc::validation {
static std::recursive_mutex mtx;
struct ResMap {
    vstd::unordered_map<uint64_t, RWResource *> map;
    ~ResMap() {
        std::lock_guard lck{mtx};
        for (auto &&i : map) {
            delete i.second;
        }
    }
};
static ResMap res_map;
RWResource::RWResource(uint64_t handle, Tag tag, bool non_simultaneous)
    : Resource{tag}, _non_simultaneous{non_simultaneous}, _handle{handle} {
    std::lock_guard lck{mtx};
    res_map.map.force_emplace(handle, this);
}
void RWResource::set_usage(Stream *stream, RWResource *res, Usage usage, Range range) {
    std::lock_guard lck{mtx};
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
        auto iter = res->_info.try_emplace(stream->handle());
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
        auto ptr = RWResource::try_get<Stream>(i.first);
        if (ptr && i.second.last_frame > ptr->synced_layer()) {
            LUISA_ERROR("Resource {} destroyed when {} is still using it.", get_name(), ptr->get_name());
        }
    }
    _info.clear();
}
void RWResource::dispose(uint64_t handle) {
    std::lock_guard lck{mtx};
    auto iter = res_map.map.find(handle);
    if (iter != res_map.map.end()) {
        delete iter->second;
        res_map.map.erase(iter);
    }
}
RWResource *RWResource::_get(uint64_t handle) {
    std::lock_guard lck{mtx};
    auto iter = res_map.map.find(handle);
    if (iter != res_map.map.end()) {
        return iter->second;
    }
    return nullptr;
}
}// namespace lc::validation
