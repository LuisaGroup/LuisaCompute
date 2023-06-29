#pragma once
#include "rw_resource.h"
#include <luisa/ast/usage.h>
#include <luisa/vstl/common.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/command_list.h>
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

class CustomDispatchArgumentVisitor;

class Stream : public RWResource {

    friend class CustomDispatchArgumentVisitor;

private:
    StreamTag _stream_tag;
    uint64_t _executed_layer{0};
    uint64_t _synced_layer{0};
    // other streams waiting for this stream
    vstd::unordered_map<Stream *, uint64_t> waited_stream;
    vstd::unordered_map<uint64_t, vstd::vector<Range>> dstorage_range_check;
    uint64_t stream_synced_frame(Stream *stream) const;
    void mark_handle(uint64_t v, Usage usage, Range range);
    void custom(DeviceInterface *dev, Command *cmd);
    void mark_shader_dispatch(DeviceInterface *dev, ShaderDispatchCommandBase *cmd, bool contain_bindings);

public:
    vstd::unordered_map<RWResource *, CompeteResource> res_usages;
    auto executed_layer() const { return _executed_layer; }
    auto synced_layer() const { return _synced_layer; }
    vstd::string stream_tag() const;
    Stream(uint64_t handle, StreamTag stream_tag);
    void dispatch();
    void dispatch(DeviceInterface *dev, CommandList &cmd_list);
    void sync();
    void sync_layer(uint64_t layer);
    void signal(Event *evt, uint64_t fence);
    void wait(Event *evt, uint64_t fence);
    void check_compete();
};
}// namespace lc::validation
