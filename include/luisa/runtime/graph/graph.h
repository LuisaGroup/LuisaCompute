#pragma once
#include <luisa/runtime/shader.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/graph/graph_node.h>
#include <luisa/runtime/graph/kernel_node.h>
#include <luisa/runtime/graph/memory_node.h>
#include <luisa/runtime/graph/graph_invoke.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/graph/graph_var.h>
#include <luisa/runtime/graph/utils.h>

namespace luisa::compute::graph {
class GraphExt;

class LC_RUNTIME_API GraphInterface : public Resource {
    template<typename... Args>
    friend class Graph;
    friend class GraphBase;
    //TODO: to make protected, now just for test
public:
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

    U<GraphBuilder> _builder = nullptr;
    operator bool() const noexcept { return _impl != nullptr && _graph_ext != nullptr; }
    GraphInterface *_impl = nullptr;
    bool _built = false;
    bool build_if_needed() noexcept {
        auto need_build = !_built;
        if (need_build) _impl->create_graph_instance(_builder.get());
        _built = true;
        return need_build;
    }

public:
    GraphBase(GraphBase &&other) noexcept
        : _graph_ext{other._graph_ext},
          _impl{other._impl},
          _builder{std::move(other._builder)}, _built{other._built} {
        other._graph_ext = nullptr;
        other._impl = nullptr;
        other._built = false;
    }

    GraphBase &operator=(GraphBase &&other) noexcept {
        _graph_ext = other._graph_ext;
        _impl = other._impl;
        other._graph_ext = nullptr;
        other._impl = nullptr;
        _builder = std::move(other._builder);
        _built = other._built;
        other._built = false;
        return *this;
    }
protected:
    GraphInvoke launch() noexcept {
        return GraphInvoke(
            [this](Stream *stream) noexcept { _impl->launch_graph_instance(stream); });
    }
};

template<typename... Args>
class Graph final : public GraphBase {
public:
    using GraphBase::GraphBase;
    virtual ~Graph() noexcept = default;
    GraphInvoke operator()(detail::graph_var_to_view_t<Args> &&...args) noexcept;
    void check_parms_update(detail::graph_var_to_view_t<Args> &&...args) noexcept;
};

template<typename... Args>
GraphInvoke Graph<Args...>::operator()(detail::graph_var_to_view_t<Args> &&...args) noexcept {
    check_parms_update(std::forward<detail::graph_var_to_view_t<Args>>(args)...);
    return launch();
}

template<typename... Args>
void Graph<Args...>::check_parms_update(detail::graph_var_to_view_t<Args> &&...args) noexcept {
    // first we need update the graph vars and nodes at front end
    detail::for_each_arg_with_index(
        [&]<typename Arg>(uint64_t I, Arg &&arg) noexcept {
            using raw_type = std::remove_cvref_t<Arg>;
            auto input_var = _builder->input_var(GraphInputVarId{I});
            auto derived = input_var->cast<GraphVar<raw_type>>();
            derived->input_var_update_check(arg);
        },
        std::forward<detail::graph_var_to_view_t<Args>>(args)...);
    auto need_update = _builder->propagate_need_update_flag_from_vars_to_nodes();

    // then we can build the graph if needed
    auto first_build = build_if_needed();
    if (!first_build) {
        // if it is not the first time build, we are going to update the graph instance node parms
        _impl->update_graph_instance_node_parms(_builder.get());
    }

#ifdef _DEBUG
    if (need_update) {
        _builder->check_var_overlap();
    }
#endif
}
}// namespace luisa::compute::graph
