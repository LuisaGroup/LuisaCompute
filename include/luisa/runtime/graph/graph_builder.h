#pragma once
#include <luisa/vstl/unique_ptr.h>
#include <luisa/vstl/functional.h>
#include <luisa/vstl/vector.h>
#include <luisa/runtime/graph/graph_var_id.h>
#include <luisa/runtime/graph/graph_node_id.h>
#include <luisa/runtime/graph/graph_deps.h>
#include <luisa/runtime/graph/memory_node_direction.h>
#include <luisa/ast/usage.h>
#include <luisa/runtime/graph/sparse_2d_array.h>
#include <luisa/runtime/graph/utils.h>
#include <luisa/runtime/graph/input_sub_var_collection.h>

namespace luisa::compute::graph {
class GraphNode;
class KernelNode;
class MemoryNode;
class CaptureNodeBase;
class KernelNodeCmdEncoder;
class GraphBufferVarBase;
class GraphBasicVarBase;

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
    size_t _input_var_count = 0;
public:
    // # graph topology
    // ## input var : graph input arguments, used to calculate graph nodes' dependencies
    // input var with overlapping memory space is not allowed
    auto input_vars() noexcept { return span{_sub_vars}.subspan(0, _input_var_count); };
    auto input_vars() const noexcept { return span{_sub_vars}.subspan(0, _input_var_count); };
    const GraphVarBase *input_var(GraphInputVarId id) const noexcept { return input_vars()[id.value()]; }
    GraphVarBase *input_var(GraphInputVarId id) noexcept { return input_vars()[id.value()]; }

    // ## sub var : sub var of input var, used to split input var
    // sub var with overlapping memory space(in side a input var) is permitted
    // they are not considered when calculating graph nodes' dependencies.
    // in this collection, input var is also included (considered as a sub var of itself)
    luisa::vector<GraphVarBase *> _sub_vars;
    auto sub_vars() noexcept { return span{_sub_vars}; }
    auto sub_vars() const noexcept { return span{_sub_vars}; }
    const GraphVarBase *sub_var(GraphSubVarId id) const noexcept { return _sub_vars[id.value()]; }
    GraphVarBase *sub_var(GraphSubVarId id) noexcept { return _sub_vars[id.value()]; }
    auto pure_sub_vars() noexcept { return span{_sub_vars}.subspan(_input_var_count); }
    auto pure_sub_vars() const noexcept { return span{_sub_vars}.subspan(_input_var_count); }

    // ## graph node : graph nodes
    luisa::vector<GraphNode *> _graph_nodes;
    const GraphNode *node(GraphNodeId id) const noexcept { return _graph_nodes[id.value()]; }
    GraphNode *node(GraphNodeId id) noexcept { return _graph_nodes[id.value()]; }

    // ## graph nodes' dependencies
    luisa::vector<GraphDependency> _deps;
    auto deps() const noexcept { return span{_deps}; }
    auto deps() noexcept { return span{_deps}; }

    // # graph update
    // ## input var accessors : input_var_id -> {sub_var_id, ...}
    luisa::compute::Sparse2DArray<GraphSubVarId> _input_var_to_sub_vars;
    auto dep_sub_vars(GraphInputVarId id) const noexcept { return _input_var_to_sub_vars(id.value()); }
    auto dep_sub_vars(GraphInputVarId id) noexcept { return _input_var_to_sub_vars(id.value()); }

    // ## sub var accessors : sub_var_id -> {node_id, ...}
    luisa::compute::Sparse2DArray<GraphNodeId> _sub_var_to_nodes;
    auto dep_nodes(GraphSubVarId id) const noexcept { return _sub_var_to_nodes(id.value()); }
    // ## node need update flags : for backends to check whether a node need update
    luisa::vector<int> _node_need_update_flags;
    int &node_need_update_flag(GraphNodeId id) noexcept { return _node_need_update_flags[id.value()]; }
    int node_need_update_flag(GraphNodeId id) const noexcept { return _node_need_update_flags[id.value()]; }

    GraphBuilder() noexcept;
    ~GraphBuilder() noexcept;

    GraphBuilder(const GraphBuilder &other) noexcept;
    GraphBuilder &operator=(const GraphBuilder &) = delete;

    static GraphBuilder *current() noexcept;
    static bool is_building() noexcept;

    const GraphNode *graph_node(GraphNodeId id) const noexcept;

    bool node_need_update(GraphNodeId id) const noexcept;

    void clear_need_update_flags() noexcept;

    auto graph_nodes() const noexcept { return span{_graph_nodes}; }
    auto graph_nodes() noexcept { return span{_graph_nodes}; }
    auto kernel_nodes() const noexcept { return span{_kernel_nodes}; }
    auto kernel_nodes() noexcept { return span{_kernel_nodes}; }
    auto memory_nodes() const noexcept { return span{_memory_nodes}; }
    auto memory_nodes() noexcept { return span{_memory_nodes}; }
    auto capture_nodes() const noexcept { return span{_capture_nodes}; }
    auto capture_nodes() noexcept { return span{_capture_nodes}; }

    auto *kernel_node_cmd_encoder(GraphNodeId kernel_node_id) const noexcept {
        return _kernel_node_cmd_encoders[kernel_node_id.value()].get();
    }

    //const luisa::vector<GraphNodeId> &accessor_node_ids(const GraphVarBase *graph_var) const noexcept;
    const luisa::vector<GraphDependency> &graph_deps() const noexcept { return _deps; }
    class GraphvizOptions {
    public:
        bool show_vars = true;
        bool show_nodes = true;
    };
    void graphviz(std::ostream &o, GraphvizOptions options = {}) noexcept;
private:
    // only used by GraphDef >>>
    template<typename F>
    [[nodiscard]] static U<GraphBuilder> build(F &&fn) noexcept {
        _current() = make_unique<GraphBuilder>();
        _current()->_is_building = true;
        fn();
        _current()->_setup_node_need_update_flags();
        _current()->_build_deps();
        _current()->_build_var_accessors();
        _current()->_is_building = false;
        return std::move(_current());
    }
    static void set_var_count(size_t size) noexcept;
    template<typename T, size_t I>
        requires std::is_base_of_v<GraphVarBase, T>
    [[nodiscard]] static void define_input_var() noexcept {
        auto var = make_unique<T>(GraphInputVarId{I});
        auto ptr = var.get();
        _current()->_def_input_var(std::move(var));
        _current()->_sub_vars[I] = ptr;
    }
    // only used by GraphDef <<<

    // only used by GraphVarBase >>>
    friend class GraphVarBase;
    template<typename T, typename... Args>
        requires std::is_base_of_v<GraphVarBase, T>
    [[nodiscard]] static auto &emplace_sub_var(Args &&...args) noexcept {
        // get a new sub var id
        auto id = _current()->_sub_vars.size();
        // set the sub var id to the input var
        auto sub_var = make_unique<T>(GraphSubVarId{id}, std::forward<Args>(args)...);
        auto ptr = sub_var.get();
        _current()->_def_sub_var(std::move(sub_var));
        _current()->_sub_vars.emplace_back(ptr);
        return *ptr;
    }
    // only used by GraphVarBase <<<

    // only used by Shader:
    static KernelNode *add_kernel_node(span<GraphSubVarId> arg_ids,
                                       const Resource *shader_resource, U<KernelNodeCmdEncoder> &&encoder,
                                       size_t dimension, const uint3 &block_size) noexcept;

    template<typename T>
    friend class GraphVar;
    template<typename T>
    friend class GraphSubVar;
    static MemoryNode *add_memory_node(GraphSubVarId src_var_id, GraphSubVarId dst_var_id, MemoryNodeDirection direction) noexcept;

    template<typename FCapture, typename... Args>
        requires std::is_invocable_v<FCapture, uint64_t, Args...>
    friend CaptureNodeBase &capture(std::array<Usage, sizeof...(Args)> usages, FCapture &&capture, GraphVar<Args>... args) noexcept;

    template<typename FCapture, typename... Args>
    static CaptureNodeBase *add_capture_node(span<Usage, sizeof...(Args)> usages, FCapture &&capture, GraphVar<Args>... args) noexcept;

    // only used by Graph >>>
    bool propagate_need_update_flag_from_vars_to_nodes() noexcept;
    void check_var_overlap() noexcept;
    // only used by Graph <<<

    static U<GraphBuilder> &_current() noexcept;

    // # storage
    // ## concrete var
    // ### graph basic var : input var and its sub var
    friend class GraphBasicVarBase;
    InputSubVarCollection<GraphBasicVarBase> _basic_vars;
    void _def_input_var(U<GraphBasicVarBase> &&var) noexcept;
    void _def_sub_var(U<GraphBasicVarBase> &&var) noexcept;

    // ### graph buffer var : input var and its sub var
    friend class GraphBufferVarBase;
    InputSubVarCollection<GraphBufferVarBase> _buffer_vars;
    void _def_input_var(U<GraphBufferVarBase> &&buffer) noexcept;
    void _def_sub_var(U<GraphBufferVarBase> &&buffer) noexcept;

    // ### graph host memory var : input var and its sub var
    friend class GraphHostMemoryVar;
    InputSubVarCollection<GraphVar<void *>> _host_memory_vars;
    void _def_input_var(U<GraphVar<void *>> &&var) noexcept;
    void _def_sub_var(U<GraphVar<void *>> &&var) noexcept;

    template<typename T>
    void fill_sub_vars(const InputSubVarCollection<T> &collection) {
        for (auto &&sub_var : collection._sub_vars) {
            _sub_vars[sub_var->sub_var_id().value()] = sub_var.get();
        }
    }

    // ## concrete node
    // ### kernel node and its cmd encoder
    friend class KernelNode;
    luisa::vector<S<KernelNode>> _kernel_nodes;
    luisa::vector<U<KernelNodeCmdEncoder>> _kernel_node_cmd_encoders;

    // ### capture node
    friend class CaptureNodeBase;
    luisa::vector<S<CaptureNodeBase>> _capture_nodes;
    // ### memory node
    friend class MemoryNode;
    luisa::vector<S<MemoryNode>> _memory_nodes;

    // private methods >>>
    void _build_deps() noexcept;

    // build a dep tree:
    // input var -> {sub var, ...}
    // sub var -> {node, ...}
    // to calculate the related update flags
    // an update of a input var will cause all its sub var and nodes update
    void _build_var_accessors() noexcept;
    void _setup_node_need_update_flags() noexcept;
    void _update_kernel_node_cmd_encoders(const KernelNode *node) noexcept;
    void _check_buffer_var_overlap() noexcept;
    // private methods <<<
};
}// namespace luisa::compute::graph

namespace luisa::compute::graph {
template<typename FCapture, typename... Args>
    requires std::is_invocable_v<FCapture, uint64_t, Args...>
CaptureNodeBase &capture(std::array<Usage, sizeof...(Args)> usages, FCapture &&capture, GraphVar<Args>... args) noexcept {
    LUISA_ASSERT(GraphBuilder::is_building(), "This function is invocable in GraphDef only");
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
    current()->_graph_nodes.emplace_back(ptr);
    return ptr;
}
}// namespace luisa::compute::graph
