#pragma once
#include "rw_resource.h"
#include <ast/usage.h>
#include <vstl/common.h>
#include <runtime/rhi/command.h>
#include <runtime/command_list.h>
#include "range.h"
namespace lc::validation {
using namespace luisa::compute;
class Event;
class RWResource;
class Stream;
struct CompeteResource {
    Usage usage{Usage::NONE};
    vstd::vector<Range> ranges;
};
class Stream : public RWResource {
    StreamTag _stream_tag;
    uint64_t _executed_layer{0};
    uint64_t _synced_layer{0};
    vstd::unordered_map<Stream const *, uint64_t> waited_stream;
    uint64_t stream_synced_frame(Stream const *stream) const;
    void mark_handle(uint64_t v, Usage usage, Range range);
    void custom(DeviceInterface *dev, Command *cmd);
    void mark_shader_dispatch(DeviceInterface *dev, ShaderDispatchCommandBase *cmd, bool contain_bindings);

public:
    vstd::unordered_map<RWResource const *, CompeteResource> res_usages;
    auto executed_layer() const { return _executed_layer; }
    auto synced_layer() const { return _synced_layer; }
    vstd::string stream_tag() const;
    Stream(uint64_t handle, StreamTag stream_tag);
    void dispatch();
    void dispatch(DeviceInterface *dev, CommandList &cmd_list);
    void sync();
    void sync_layer(uint64_t layer);
    void signal(Event *evt);
    void wait(Event *evt);
    void check_compete();
};
}// namespace lc::validation