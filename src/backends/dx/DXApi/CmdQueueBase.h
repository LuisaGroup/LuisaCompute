#pragma once
namespace lc::dx {
enum class CmdQueueTag {
    MainCmd,
    DStorage,
};
class CmdQueueBase {
protected:
    CmdQueueTag tag;
    CmdQueueBase(CmdQueueTag tag) : tag{tag} {}
    ~CmdQueueBase() = default;

public:
    CmdQueueTag Tag() const { return tag; }
};
}// namespace lc::dx