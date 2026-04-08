#pragma once

#include "work_graph_kernel.h"
#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute {

class WorkGraphBuilder;

namespace detail {

struct WorkGraphNode;

struct WorkGraphNodeArray {
    luisa::string array_name;
    uint start;
    uint count;
};

struct WorkGraphEdge {
    uint source;
    uint dest;
    uint max_records;
    uint source_output_index;
    // note: work graph edge can go an entire array, or a specific node in that array
    // (or a node not in an array, of course)
    uint dest_array = ~0u;
};

struct WorkGraphNode {
    uint index;
    luisa::shared_ptr<const detail::FunctionBuilder> fn_builder;

    luisa::string name;
    WorkGraphLaunchType node_type;
    const Type* input_record_type;
    luisa::vector<WorkGraphEdge> out_edges;
    uint3 threadgroup_dim = uint3(1, 1, 1);
    uint3 dispatch_dim = uint3(1, 1, 1);
    bool input_record_has_dispatch_grid = false;
    bool defined = false;
    uint array = ~0u;
};

LUISA_DSL_API WorkGraphNode& index_to_node(WorkGraphBuilder* builder, uint node_index) noexcept;
LUISA_DSL_API WorkGraphEdge& indices_to_edge(WorkGraphBuilder* builder, uint node_index, uint edge_index) noexcept;
LUISA_DSL_API WorkGraphNodeArray& index_to_node_array(WorkGraphBuilder* builder, uint node_array_index) noexcept;

} // namespace luisa::compute::detail

template<typename T>
class WorkGraphNodeOutput {
public:
    explicit WorkGraphNodeOutput(WorkGraphBuilder* builder, uint node_index, uint edge_index) noexcept :
        _builder(builder), _node_index(node_index), _edge_index(edge_index) {}

    void write(Expr<T> data, Expr<bool> should_write) const noexcept {
        auto f = detail::FunctionBuilder::current();
        f->call(CallOp::WORK_GRAPH_OUTPUT, {
            f->literal(Type::of<uint>(), _edge_index),
            data.expression(),
            should_write.expression()
        });
    }

    // invalidated by modifying graph
    [[nodiscard]] detail::WorkGraphEdge* edge() const noexcept { return &detail::indices_to_edge(_builder, _node_index, _edge_index); }
    [[nodiscard]] WorkGraphBuilder* builder() const noexcept { return _builder; }

private:
    WorkGraphBuilder* _builder;
    uint _node_index;
    uint _edge_index;
};

template<typename T>
class WorkGraphNodeArrayOutput {
public:
    explicit WorkGraphNodeArrayOutput(WorkGraphBuilder* builder, uint node_index, uint edge_index) noexcept :
        _builder(builder), _node_index(node_index), _edge_index(edge_index) {}

    void write(Expr<T> data, Expr<uint> index, Expr<bool> should_write) const noexcept {
        auto f = detail::FunctionBuilder::current();
        f->call(CallOp::WORK_GRAPH_OUTPUT_ARRAY, {
            f->literal(Type::of<uint>(), _edge_index),
            index.expression(),
            data.expression(),
            should_write.expression()
        });
    }

    // invalidated by modifying graph
    [[nodiscard]] detail::WorkGraphEdge* edge() const noexcept { return &detail::indices_to_edge(_builder, _node_index, _edge_index); }
    [[nodiscard]] WorkGraphBuilder* builder() const noexcept { return _builder; }

private:
    WorkGraphBuilder* _builder;
    uint _node_index;
    uint _edge_index;
};

template<WorkGraphLaunchType NodeType, typename T>
class WorkGraphNode {
public:
    explicit WorkGraphNode(WorkGraphBuilder* builder, uint node_index) noexcept : _builder(builder), _node_index(node_index) {}

    [[nodiscard]] luisa::string_view name() const noexcept { return inner()->name; }

    template<typename EdgeRecord>
    [[nodiscard]] WorkGraphNodeOutput<EdgeRecord> output(uint max_records) noexcept {
        if (inner()->defined) {
            LUISA_WARNING("adding output to already defined work graph node, this is probably undesirable");
        }

        auto e = detail::WorkGraphEdge {
            inner()->index,
            ~0u,
            max_records,
            static_cast<uint>(inner()->out_edges.size())
        };

        inner()->out_edges.push_back(e);
        return WorkGraphNodeOutput<EdgeRecord>(_builder, _node_index, e.source_output_index);
    }

    // max_records is shared across *all* nodes in the array
    template<typename EdgeRecord>
    [[nodiscard]] WorkGraphNodeArrayOutput<EdgeRecord> array_output(uint max_records) noexcept {
        if (inner()->defined) {
            LUISA_WARNING("adding array output to already defined work graph node, this is probably undesirable");
        }

        auto array_edge = detail::WorkGraphEdge {
            inner()->index,
            ~0u,
            max_records,
            static_cast<uint>(inner()->out_edges.size())
        };

        inner()->out_edges.push_back(array_edge);
        return WorkGraphNodeArrayOutput<EdgeRecord>(_builder, _node_index, array_edge.source_output_index);
    }

    template<typename Def>
    void define(const WorkGraphNodeKernel<T, Def>& kernel) noexcept {
        LUISA_ASSERT(!inner()->defined, "redefining node kernel is not allowed");

        // yoink the function builder, make sure type of input record matches what we were declared with
        inner()->fn_builder = kernel.function_builder();
        inner()->fn_builder->set_name(inner()->name);
        inner()->defined = true;
    }

    template<typename EdgeRecord>
    void operator<<(const WorkGraphNodeOutput<EdgeRecord>& output) {
        static_assert(std::is_same_v<T, EdgeRecord>, "type mismatch between work graph node and incoming edge");

        auto* edge = output.edge();

        // node outputs must have fanout of 1
        LUISA_ASSERT(edge->dest == ~0u, "cannot add edge, it is already connected to different node");

        // ensure they come from same work graph
        LUISA_ASSERT(output.builder() == _builder, "all edges must be between nodes from same work graph builder");

        edge->dest = inner()->index;
    }

    // specifying NumThreads (threadgroup size) is not allowed for per-thread node launch
    void set_threadgroup_size(uint3 size) const requires (NodeType != WorkGraphLaunchType::THREAD) {
        inner()->threadgroup_dim = size;
    }

    // for broadcasting nodes, the `dispatch_dim` field is either the static size of a dispatch, or the max size
    // (dynamically sized using SV_DispatchGrid annotation)
    void set_dispatch_size(uint3 size) const
    requires (NodeType == WorkGraphLaunchType::BROADCASTING && !std::is_base_of_v<DispatchGridRecord, T>) {
        inner()->dispatch_dim = size;
    }

    void set_max_dispatch_size(uint3 size) const
    requires (NodeType == WorkGraphLaunchType::BROADCASTING && std::is_base_of_v<DispatchGridRecord, T>) {
        inner()->dispatch_dim = size;
    }

private:
    // invalidated by modifying graph
    [[nodiscard]] detail::WorkGraphNode* inner() const noexcept { return &detail::index_to_node(_builder, _node_index); }

    WorkGraphBuilder* _builder;
    uint _node_index;
};

template<WorkGraphLaunchType NodeType, typename T>
class WorkGraphNodeArray {
public:
    explicit WorkGraphNodeArray(WorkGraphBuilder* builder, uint node_array_index) noexcept :
        _builder(builder), _node_array_index(node_array_index) {}

    WorkGraphNode<NodeType, T> operator[](uint i) const noexcept {
        return WorkGraphNode<NodeType, T>(_builder, inner()->start + i);
    }

    template<typename EdgeRecord>
    void operator<<(const WorkGraphNodeArrayOutput<EdgeRecord>& output) {
        static_assert(std::is_same_v<T, EdgeRecord>, "type mismatch between work graph node array and incoming edge");
        auto* edge = output.edge();

        // node outputs must have fanout of 1
        LUISA_ASSERT(edge->dest == ~0u, "cannot add edge, it is already connected to different node");

        // ensure they come from same work graph
        LUISA_ASSERT(output.builder() == _builder, "all edges must be between nodes from same work graph builder");

        edge->dest = inner()->start;
        edge->dest_array = _node_array_index;
    }

private:
    // invalidated by modifying graph
    [[nodiscard]] detail::WorkGraphNodeArray* inner() const noexcept { return &detail::index_to_node_array(_builder, _node_array_index); }

    WorkGraphBuilder* _builder;
    uint _node_array_index;
};

class WorkGraph {
public:
    WorkGraph() = delete;

    [[nodiscard]] luisa::string_view name() const noexcept { return _name; }
    [[nodiscard]] auto& nodes() const noexcept { return _nodes; }
    [[nodiscard]] auto& node_arrays() const noexcept { return _node_arrays; }
    [[nodiscard]] auto& entry_points() const noexcept { return _entry_points; }

    [[nodiscard]] auto node_count() const noexcept { return _nodes.size(); }

private:
    friend class WorkGraphBuilder;
    explicit WorkGraph(luisa::string name, luisa::vector<detail::WorkGraphNode> nodes, luisa::vector<detail::WorkGraphNodeArray> node_arrays, luisa::vector<uint32_t> entry_points) noexcept :
        _name(std::move(name)), _nodes(std::move(nodes)), _node_arrays(std::move(node_arrays)), _entry_points(std::move(entry_points)) {}

    luisa::string _name;
    luisa::vector<detail::WorkGraphNode> _nodes;
    luisa::vector<detail::WorkGraphNodeArray> _node_arrays;
    luisa::vector<uint32_t> _entry_points;
};

class WorkGraphBuilder {
public:
    LUISA_DSL_API explicit WorkGraphBuilder(luisa::string name);

    template<WorkGraphLaunchType NodeType, typename InputRecord>
    WorkGraphNode<NodeType, InputRecord> add_node(luisa::string name) noexcept {
        const Type* input_record_type;
        if constexpr (std::is_same_v<InputRecord, WorkGraphEmptyRecord>) {
            input_record_type = nullptr;
        }
        else {
            input_record_type = Type::of<InputRecord>();
        }

        uint node_index = _nodes.size();
        _nodes.emplace_back(
            _nodes.size(),
            nullptr,

            std::move(name),
            NodeType,
            input_record_type
        );
        _nodes.back().input_record_has_dispatch_grid =
            NodeType == WorkGraphLaunchType::BROADCASTING && std::is_base_of_v<DispatchGridRecord, InputRecord>;

        return WorkGraphNode<NodeType, InputRecord>(this, node_index);
    }

    template<WorkGraphLaunchType NodeType, typename InputRecord>
    WorkGraphNodeArray<NodeType, InputRecord> add_node_array(luisa::string array_name, uint count) {
        const Type* input_record_type;
        if constexpr (std::is_same_v<InputRecord, WorkGraphEmptyRecord>) {
            input_record_type = nullptr;
        }
        else {
            input_record_type = Type::of<InputRecord>();
        }

        uint node_array_start_index = _nodes.size();
        uint node_array_index = _node_arrays.size();
        _node_arrays.emplace_back(array_name, node_array_start_index, count);

        _nodes.reserve(node_array_start_index + count);
        for (size_t i = 0; i < count; ++i) {
            detail::WorkGraphNode node {
                .index = uint(node_array_start_index + i),
                .fn_builder = nullptr,
                .name = luisa::format("{}_{}", array_name, i),
                .node_type = NodeType,
                .input_record_type = input_record_type,
                .array = node_array_index
            };

            node.input_record_has_dispatch_grid =
                NodeType == WorkGraphLaunchType::BROADCASTING && std::is_base_of_v<DispatchGridRecord, InputRecord>;

            _nodes.push_back(node);
        }

        return WorkGraphNodeArray<NodeType, InputRecord>(this, node_array_index);
    }

    LUISA_DSL_API WorkGraph build() noexcept;

private:
    friend detail::WorkGraphNode& detail::index_to_node(WorkGraphBuilder*, uint) noexcept;
    friend detail::WorkGraphEdge& detail::indices_to_edge(WorkGraphBuilder*, uint, uint) noexcept;
    friend detail::WorkGraphNodeArray& detail::index_to_node_array(WorkGraphBuilder*, uint) noexcept;

    luisa::string _name;
    luisa::vector<detail::WorkGraphNode> _nodes;
    luisa::vector<detail::WorkGraphNodeArray> _node_arrays;
};

} // namespace luisa::compute