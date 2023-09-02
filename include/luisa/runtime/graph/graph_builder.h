#pragma once
#include <luisa/vstl/unique_ptr.h>
#include <luisa/vstl/functional.h>
#include <luisa/vstl/vector.h>
#include <luisa/runtime/graph/graph_var.h>
#include <luisa/runtime/graph/graph_deps.h>
#include <luisa/ast/usage.h>

namespace luisa::compute::graph {
class KernelNode;
class CaptureNodeBase;
class GraphNode;
class KernelNodeCmdEncoder;
class LC_RUNTIME_API GraphBuilder {
    template<typename... Args>
    friend class GraphDef;
    template<typename... Args>
    friend class GraphDefBase;
    template<typename... Args>
    friend class Graph;

    template<size_t dimension, typename... Args>
    friend class Shader;

    template<typename T>
    using U = luisa::unique_ptr<T>;

    template<typename T>
    using S = luisa::shared_ptr<T>;

    bool _is_building = false;
public:
    // generic
    using node_id_t = uint64_t;
    using var_id_t = uint64_t;

    luisa::vector<U<GraphVarBase>> _vars;
    luisa::vector<luisa::vector<node_id_t>> _var_accessors;// var_id -> {node_id, ...}
    luisa::vector<GraphNode *> _nodes;
    luisa::vector<int> _node_need_update_flags;
    luisa::vector<GraphDependency> _deps;

    // concrete
    luisa::vector<S<KernelNode>> _kernel_nodes;
    luisa::vector<U<KernelNodeCmdEncoder>> _kernel_node_cmd_encoders;
    luisa::vector<S<CaptureNodeBase>> _capture_nodes;

    GraphBuilder() noexcept;
    ~GraphBuilder() noexcept;

    GraphBuilder(const GraphBuilder &other) noexcept;
    GraphBuilder &operator=(const GraphBuilder &) = delete;

    static GraphBuilder *current() noexcept;
    static bool is_building() noexcept;

    const GraphVarBase *graph_var(var_id_t id) const noexcept;
    const GraphNode *graph_node(node_id_t id) const noexcept;

    bool node_need_update(const GraphNode *node) const noexcept;
    bool var_need_update(const GraphVarBase *var) const noexcept;
    void clear_need_update_flags() noexcept;

    size_t graph_var_count() const noexcept { return _vars.size(); }
    size_t kernel_node_count() const noexcept { return _kernel_nodes.size(); }
    size_t graph_node_count() const noexcept { return _nodes.size(); }

    const auto &graph_vars() const noexcept { return _vars; }
    const auto &graph_nodes() const noexcept { return _nodes; }

    const auto &kernel_nodes() const noexcept { return _kernel_nodes; }
    const auto &capture_nodes() const noexcept { return _capture_nodes; }
    auto *kernel_node_cmd_encoder(node_id_t kernel_node_id) const noexcept {
        return _kernel_node_cmd_encoders[kernel_node_id].get();
    }

    const luisa::vector<GraphBuilder::node_id_t> &accessor_node_ids(const GraphVarBase *graph_var) const noexcept;
    const luisa::vector<GraphDependency> &graph_deps() const noexcept { return _deps; }
    class GraphvizOptions {
    public:
        bool show_vars = true;
    };
    void graphviz(std::ostream &o, GraphvizOptions options = {}) noexcept;
private:
    // only used by GraphDef >>>
    static void set_var_count(size_t size) noexcept;
    template<typename T, size_t I>
    [[nodiscard]] static T &define_graph_var() noexcept {
        auto var = make_unique<T>(GraphArgId{I});
        auto ptr = var.get();
        _current()->_vars[I] = std::move(var);
        return *ptr;
    }
    template<typename F>
    [[nodiscard]] static U<GraphBuilder> build(F &&fn) noexcept {
        _current() = make_unique<GraphBuilder>();
        _current()->_is_building = true;
        fn();
        _current()->_set_up_node_need_update_flags();
        _current()->_build_deps();
        _current()->_build_var_accessors();
        _current()->_is_building = false;
        return std::move(_current());
    }
    // only used by GraphDef <<<

    // only used by Shader:
    static KernelNode *add_kernel_node(span<uint64_t> arg_ids,
                                       const Resource *shader_resource, U<KernelNodeCmdEncoder> &&encoder,
                                       size_t dimension, const uint3 &block_size) noexcept;

    template<typename FCapture, typename... Args>
        requires std::is_invocable_v<FCapture, uint64_t, Args...>
    friend CaptureNodeBase& capture(std::array<Usage, sizeof...(Args)> usages, FCapture &&capture, GraphVar<Args>... args) noexcept;

    template<typename FCapture, typename... Args>
    static CaptureNodeBase *add_capture_node(span<Usage, sizeof...(Args)> usages, FCapture &&capture, GraphVar<Args>... args) noexcept;

    // only used by Graph >>>
    void propagate_need_update_flag_from_vars_to_nodes() noexcept;
    // only used by Graph <<<

    static U<GraphBuilder> &_current() noexcept;

    // private methods >>>
    void _build_deps() noexcept;
    void _build_var_accessors() noexcept;
    void _set_up_node_need_update_flags() noexcept;
    void _update_kernel_node_cmd_encoders(const KernelNode *node) noexcept;
    // private methods <<<
};
}// namespace luisa::compute::graph

namespace luisa::compute::graph {
template<typename FCapture, typename... Args>
    requires std::is_invocable_v<FCapture, uint64_t, Args...>
CaptureNodeBase& capture(std::array<Usage, sizeof...(Args)> usages, FCapture &&capture, GraphVar<Args>... args) noexcept {
    return *GraphBuilder::add_capture_node(
        usages,
        std::forward<FCapture>(capture),
        args...);
}
}// namespace luisa::compute::graph

#include <luisa/runtime/graph/capture_node.h>
namespace luisa::compute::graph {
template<typename FCapture, typename... Args>
CaptureNodeBase *GraphBuilder::add_capture_node(span<Usage, sizeof...(Args)> usages, FCapture &&capture, GraphVar<Args>... args) noexcept {
    using FuncCapture = std::remove_cvref_t<FCapture>;
    auto node = make_shared<CaptureNode<FuncCapture, Args...>>(current(), usages, capture, args...);
    auto ptr = node.get();
    current()->_capture_nodes.emplace_back(std::move(node));
    current()->_nodes.emplace_back(ptr);
    return ptr;
}
}// namespace luisa::compute::graph
