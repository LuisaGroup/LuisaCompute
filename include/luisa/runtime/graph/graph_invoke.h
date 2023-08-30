#pragma once
#include <luisa/vstl/functional.h>
namespace luisa::compute::graph {
class GraphInvokeCommit {
    friend class Stream;
    friend class GraphInvoke;
    luisa::function<void(Stream *)> func;
    GraphInvokeCommit(luisa::function<void(Stream *)> &&func) noexcept : func{std::move(func)} {}
};

class GraphInvoke {
    friend class Stream;
    friend class GraphBase;
    GraphInvoke(luisa::function<void(Stream *)> &&func) noexcept : _func{std::move(func)} {}
    luisa::function<void(Stream *)> _func;
public:

    GraphInvoke(GraphInvoke &&) noexcept = delete;
    GraphInvoke &operator=(GraphInvoke &&) noexcept = delete;
    virtual ~GraphInvoke() noexcept {}
    auto dispatch() && noexcept { return GraphInvokeCommit{std::move(_func)}; }
};
}// namespace luisa::compute::graph