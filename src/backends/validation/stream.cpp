#include "stream.h"
#include "event.h"
#include "rw_resource.h"
#include <core/logging.h>
namespace lc::validation {
Stream::Stream(uint64_t handle, StreamTag stream_tag) : Resource{handle, Tag::STREAM}, _stream_tag{stream_tag} {}
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
namespace detail {
static vstd::string usage_name(Usage usage) {
    switch (usage) {
        case Usage::NONE:
            return "none";
        case Usage::READ:
            return "read";
        case Usage::READ_WRITE:
            return "read and write";
        default:
            return "write";
    }
}
}// namespace detail
void Stream::check_compete() {
    for (auto &&iter : res_usages) {
        auto res = iter.first;
        for (auto &&stream_iter : res->info()) {
            auto other_stream = stream_iter.first;
            if (other_stream == this) continue;
            auto synced_frame = stream_synced_frame(other_stream);
            if (stream_iter.second.last_frame > synced_frame) {
                // Texture type
                if (res->non_simultaneous()) {
                    LUISA_ERROR(
                        "Non-simultaneous resource \"{}\" is not allowed to be {} by \"{}\" and {} by \"{}\" simultaneously.",
                        res->get_name(),
                        detail::usage_name(stream_iter.second.usage),
                        other_stream->get_name(),
                        detail::usage_name(iter.second),
                        get_name());
                }
                // others, buffer, etc
                else if ((luisa::to_underlying(stream_iter.second.usage) & luisa::to_underlying(iter.second) & luisa::to_underlying(Usage::WRITE)) != 0) {
                    LUISA_ERROR(
                        "Resource \"{}\" is not allowed to be {} by \"{}\" and {} by \"{}\" simultaneously.",
                        res->get_name(),
                        detail::usage_name(stream_iter.second.usage),
                        other_stream->get_name(),
                        detail::usage_name(iter.second),
                        get_name());
                }
            }
        }
    }
}
void Stream::dispatch(){
    _executed_layer++;
}
void Stream::dispatch(CommandList &cmd_list) {
    _executed_layer++;
    res_usages.clear();
    using CmdTag = luisa::compute::Command::Tag;
    for (auto &&cmd : cmd_list.commands()) {
        switch (cmd->tag()) {
            // case CmdTag::EBufferUploadCommand:
        }
        // TODO: resources record
    }
}
void Stream::sync_layer(uint64_t layer){
    _synced_layer = std::max(_synced_layer, layer);
}
void Stream::sync() {
    _synced_layer = _executed_layer;
}
void Event::sync(){
    for(auto&& i : signaled){
        i.first->sync_layer(i.second);
    }
    signaled.clear();
}
}// namespace lc::validation