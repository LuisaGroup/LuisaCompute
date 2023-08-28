#pragma once
#include <luisa/vstl/unique_ptr.h>
#include <luisa/vstl/vector.h>
#include <luisa/runtime/graph/graph_var.h>
#include <luisa/runtime/graph/graph_deps.h>
namespace luisa::compute::graph {
class KernelNode;
class GraphNode;
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
    luisa::vector<GraphDependency> _deps;

    // concrete
    luisa::vector<S<KernelNode>> _kernel_nodes;

    GraphBuilder() noexcept;
    ~GraphBuilder() noexcept;

    GraphBuilder(const GraphBuilder &other) noexcept;
    GraphBuilder &operator=(const GraphBuilder &) = delete;

    static GraphBuilder *current() noexcept;
    static bool is_building() noexcept;

    const GraphVarBase *graph_var(var_id_t id) const noexcept;
    const GraphNode *graph_node(node_id_t id) const noexcept;
    void clear_need_update_flags() noexcept;
    size_t graph_var_count() const noexcept { return _vars.size(); }
    const auto &graph_vars() const noexcept { return _vars; }
    const auto &kernel_nodes() const noexcept { return _kernel_nodes; }
    const auto &graph_nodes() const noexcept { return _nodes; }
    const luisa::vector<GraphBuilder::node_id_t> &accessor_node_ids(const GraphVarBase *graph_var) const noexcept;
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
        _current()->_build_deps();
        _current()->_build_var_accessors();
        _current()->_is_building = false;
        return std::move(_current());
    }
    // only used by GraphDef <<<

    // only used by Shader >>>
    static KernelNode *add_kernel_node(span<uint64_t> arg_ids, const Resource *shader_resource) noexcept;
    // only used by Shader <<<

    static U<GraphBuilder> &_current() noexcept;

    // private methods >>>
    void _build_deps() noexcept;
    void _build_var_accessors() noexcept;
    // private methods <<<
};
}// namespace luisa::compute::graph