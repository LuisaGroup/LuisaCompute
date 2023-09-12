#pragma once
#include <luisa/runtime/graph/graph_node_id.h>
#include <luisa/runtime/graph/graph_var_id.h>
#include <luisa/runtime/graph/graph_deps.h>

#include <luisa/vstl/unique_ptr.h>
#include <luisa/core/logging.h>
#include <luisa/vstl/hash_map.h>
#include <luisa/runtime/device.h>
#include <luisa/vstl/vector.h>

namespace luisa::compute::graph {
enum class GraphNodeType {
    None = 0,
    Kernel = 1,
    MemoryCopy = 2,
    EventRecord = 3,
    EventWait = 4,
    Capture = 5,
};

class GraphBuilder;

class LC_RUNTIME_API GraphNode {
    friend class GraphBuilder;
public:
    static constexpr string_view graphviz_prefix = "node_";
    GraphNode(GraphBuilder *builder, GraphNodeType type) noexcept;
    GraphNode(GraphNode &&) noexcept = default;
    GraphNode &operator=(GraphNode &&) noexcept = default;
    virtual ~GraphNode() noexcept {}
    const auto &input_var_usage() const noexcept { return _input_var_usage; }
    const auto &sub_var_ids() const noexcept { return _sub_var_ids; }
    auto input_var_id(size_t index) const noexcept { return _input_var_usage[index].first; }
    auto sub_var_id(size_t index) const noexcept { return _sub_var_ids[index]; }
    auto node_id() const noexcept { return _node_id; }
    GraphNodeType type() const noexcept { return _type; }
    span<GraphDependency> deps(GraphBuilder *builder) const noexcept;
    string_view node_name() const noexcept { return _node_name; }
    auto &set_node_name(string_view name) noexcept {
        _node_name = name;
        return *this;
    }
    virtual void graphviz_def(std::ostream &o) const noexcept;
    virtual void graphviz_id(std::ostream &o) const noexcept;
    virtual void graphviz_arg_usages(std::ostream &o) const noexcept;
protected:
    template<typename T>
    using U = unique_ptr<T>;
    // virtual U<GraphNode> clone() const noexcept = 0;
    // GraphBuilder *builder() const noexcept { return _builder; }
    void add_arg_usage(GraphSubVarId sub_var_id, Usage usage) noexcept;
private:
    void set_dep_range(size_t begin, size_t count) noexcept {
        _dep_begin = begin;
        _dep_count = count;
    }
    vector<std::pair<GraphInputVarId, Usage>> _input_var_usage;
    vector<GraphSubVarId> _sub_var_ids;
    friend class GraphBuilder;
    GraphNodeId _node_id{GraphNodeId::invalid_id};
    GraphNodeType _type{GraphNodeType::None};
    size_t _dep_begin{0};
    size_t _dep_count{0};
    void *_native_handle{nullptr};
    string _node_name;
};
}// namespace luisa::compute::graph