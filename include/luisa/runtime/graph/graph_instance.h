#pragma once
#include <luisa/runtime/command_list.h>
namespace luisa::compute::graph {
class GraphInvoke {
public:
    GraphInvoke() noexcept {}
    GraphInvoke(GraphInvoke &&) noexcept = default;
    GraphInvoke &operator=(GraphInvoke &&) noexcept = default;
    virtual ~GraphInvoke() noexcept {}
};

class GraphInstance {
public:
    GraphInstance() noexcept {}
    GraphInstance(GraphInstance &&) noexcept = default;
    GraphInstance &operator=(GraphInstance &&) noexcept = default;
    virtual ~GraphInstance() noexcept {}
    virtual bool is_real_graph() noexcept { return false; }
    auto commit() noexcept {
        LUISA_ASSERT(!is_real_graph(), "real graph can't commit(), use launch() instead");
        return _command_list.commit();
    }
    auto launch() noexcept { return GraphInvoke{}; }
private:
    CommandList _command_list;
};
}// namespace luisa::compute