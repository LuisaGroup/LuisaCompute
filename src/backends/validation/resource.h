#pragma once
#include <vstl/common.h>
#include <runtime/rhi/resource.h>
namespace lc::validation {
class Resource : public vstd::IOperatorNewBase{
public:
    using Tag = luisa::compute::Resource::Tag;

private:
    uint64_t _handle;
    Tag _tag;

public:
    vstd::string name;
    vstd::string get_tag_name(Tag tag) const;
    vstd::string get_name() const;
    Resource(uint64_t handle, Tag tag) : _handle{handle}, _tag{tag} {}
    auto handle() const { return _handle; }
    virtual ~Resource() = default;
};
}// namespace lc::validation