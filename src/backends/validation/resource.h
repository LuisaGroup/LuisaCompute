#pragma once
#include <luisa/vstl/common.h>
#include <luisa/runtime/rhi/resource.h>
namespace lc::validation {
class Resource : public vstd::IOperatorNewBase {
public:
    using Tag = luisa::compute::Resource::Tag;

private:
    Tag _tag;

public:
    vstd::string name;
    vstd::string get_tag_name(Tag tag) const;
    vstd::string get_name() const;
    Resource(Tag tag) : _tag{tag} {}
    Tag tag() const { return _tag; }
    virtual ~Resource() = default;
};
}// namespace lc::validation
