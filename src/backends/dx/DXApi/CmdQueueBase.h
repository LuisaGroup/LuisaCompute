#pragma once
#include <Resource/Resource.h>
namespace lc::dx {
enum class CmdQueueTag {
    MainCmd,
    DStorage,
};
class CmdQueueBase : public Resource {
protected:
    CmdQueueTag tag;
    CmdQueueBase(Device *device, CmdQueueTag tag) : Resource{device}, tag{tag} {}
    ~CmdQueueBase() = default;

public:
    CmdQueueTag Tag() const { return tag; }
    Resource::Tag GetTag() const override {
        return Resource::Tag::CommandQueue;
    }
};
}// namespace lc::dx
