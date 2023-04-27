#pragma once
#include <vstl/common.h>
#include <runtime/rhi/resource.h>
namespace lc::validation {
class Resource : public vstd::IOperatorNewBase{
public:
    using Tag = luisa::compute::Resource::Tag;

private:
    Tag _tag;

public:
    vstd::string name;
    vstd::string get_tag_name(Tag tag) const;
    vstd::string get_name() const;
    Resource(Tag tag):  _tag{tag} {}
    virtual ~Resource() = default;
};
}// namespace lc::validation