#include "stream.h"
#include "event.h"
#include <core/logging.h>
namespace lc::validation {
Stream::Stream(uint64_t handle) : Resource{handle} {}
void Stream::signal(Event *evt) {
    evt->signaled.force_emplace(this, _executed_layer);
}
uint64_t Stream::stream_synced_frame(Stream const *stream) const {
    auto iter = waited_stream.find(stream);
    if (iter == waited_stream.end()) {
        return stream->synced_layer();
    } else {
        return iter->second;
    }
}
void Stream::wait(Event *evt) {
    for (auto &&i : evt->signaled) {
        waited_stream.force_emplace(i.first, i.second);
    }
}
void Stream::check_compete() {
    // TODO
    // for (auto &&iter : res_usages) {
    //     auto res = iter.first;
    // }
}
}// namespace lc::validation