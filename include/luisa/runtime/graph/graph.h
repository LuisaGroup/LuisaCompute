#pragma once
#include <luisa/runtime/shader.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/graph/graph_node.h>
#include <luisa/runtime/graph/kernel_node.h>
#include <luisa/runtime/graph/memory_node.h>
#include <luisa/runtime/graph/graph_instance.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/graph/graph_var.h>
#include <luisa/runtime/graph/utils.h>

namespace luisa::compute::graph {
class GraphExt;
class LC_RUNTIME_API GraphInterface : public Resource {
    template<typename... Args>
    friend class Graph;
protected:
    virtual void create_graph_instance(GraphBuilder *builder) noexcept {}
    virtual void destroy_graph_instance(GraphBuilder *builder) noexcept {}
    virtual void update_graph_instance_node_parms(GraphBuilder *builder) noexcept {}
    virtual void launch_graph_instance(Stream *stream) noexcept {}
};

class LC_RUNTIME_API GraphBase {
protected:
    template<typename T>
    using U = luisa::unique_ptr<T>;

    friend class GraphExt;
    GraphBase(GraphExt *graph_ext, const GraphBuilder *builder) noexcept;
    virtual ~GraphBase() noexcept;

    // delete copy
    GraphBase(const GraphBase &) = delete;
    GraphBase &operator=(const GraphBase &) = delete;

    GraphExt *_graph_ext = nullptr;
    GraphInterface *_impl = nullptr;
    U<GraphBuilder> _builder = nullptr;
    operator bool() const noexcept { return _impl != nullptr && _graph_ext != nullptr; }

public:
    GraphBase(GraphBase &&other) noexcept
        : _graph_ext{other._graph_ext},
          _impl{other._impl},
          _builder{std::move(other._builder)} {
        other._graph_ext = nullptr;
        other._impl = nullptr;
    }

    GraphBase &operator=(GraphBase &&other) noexcept {
        _graph_ext = other._graph_ext;
        _impl = other._impl;
        other._graph_ext = nullptr;
        other._impl = nullptr;
        _builder = std::move(other._builder);
        return *this;
    }
};

template<typename... Args>
class Graph : public GraphBase {
public:
    using GraphBase::GraphBase;
    virtual ~Graph() noexcept = default;
    GraphInvoke operator()(detail::graph_var_to_view_t<Args> &&...args) noexcept;
    GraphInvoke launch_without_check(detail::graph_var_to_view_t<Args> &&...args) noexcept;
    void check_parms_update(detail::graph_var_to_view_t<Args> &&...args) noexcept;
};

template<typename... Args>
GraphInvoke Graph<Args...>::operator()(detail::graph_var_to_view_t<Args> &&...args) noexcept {
    check_parms_update(std::forward<detail::graph_var_to_view_t<Args>>(args)...);
    return launch_without_check(std::forward<detail::graph_var_to_view_t<Args>>(args)...);
}

template<typename... Args>
void Graph<Args...>::check_parms_update(detail::graph_var_to_view_t<Args> &&...args) noexcept {
    detail::for_each_arg_with_index(
        [&]<typename Arg>(uint64_t I, Arg &&arg) noexcept {
            using raw_type = std::remove_cvref_t<Arg>;
            auto derived = _builder->_vars[I]->cast<GraphVar<raw_type>>();
            derived->update_check(arg);
        },
        std::forward<detail::graph_var_to_view_t<Args>>(args)...);
}

template<typename... Args>
GraphInvoke Graph<Args...>::launch_without_check(detail::graph_var_to_view_t<Args> &&...args) noexcept {
}
}// namespace luisa::compute::graph
