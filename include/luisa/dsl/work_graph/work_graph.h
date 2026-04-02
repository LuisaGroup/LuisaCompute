#pragma once

#include "work_graph_kernel.h"
#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute {

class WorkGraphBuilder;

namespace detail {

struct WorkGraphNode;

struct WorkGraphEdge {
    uint source;
    uint dest;
    uint max_records;
    uint source_output_index;
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
    luisa::optional<uint> dispatch_grid_member = luisa::nullopt;
    bool defined = false;
};

LUISA_DSL_API WorkGraphNode& index_to_node(WorkGraphBuilder* builder, uint node_index) noexcept;
LUISA_DSL_API WorkGraphEdge& indices_to_edge(WorkGraphBuilder* builder, uint node_index, uint edge_index) noexcept;

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
            f->literal(Type::of<uint>(), 0u),
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

    // for broadcasting nodes, this is either the static size of a dispatch, or the max size
    // (if its dynamically sized using SV_DispatchGrid)
    void set_dispatch_size(uint3 size) const requires (NodeType == WorkGraphLaunchType::BROADCASTING) {
        inner()->dispatch_dim = size;
    }

    // sets field in InputRecord type to be marked with SV_DispatchGrid
    void set_dispatch_grid_field(uint index) const requires (NodeType == WorkGraphLaunchType::BROADCASTING) {
        auto t = Type::of<T>();
        auto members = t->members();

        LUISA_ASSERT(index < members.size(), "dispatch grid field index out of bounds");

        auto member_type = members[index];
        LUISA_ASSERT(
            member_type == Type::of<uint>() ||
            member_type == Type::of<uint2>() ||
            member_type == Type::of<uint3>(),
            "dispatch grid field must be uint, uint2, or uint3"
        );

        inner()->dispatch_grid_member = index;
    }

private:
    // invalidated by modifying graph
    [[nodiscard]] detail::WorkGraphNode* inner() const noexcept { return &detail::index_to_node(_builder, _node_index); }

    WorkGraphBuilder* _builder;
    uint _node_index;
};

class WorkGraph {
public:
    WorkGraph() = delete;

    [[nodiscard]] auto& nodes() const noexcept { return _nodes; }
    [[nodiscard]] auto node_count() const noexcept { return _nodes.size(); }
    [[nodiscard]] auto& entry_points() const noexcept { return _entry_points; }

private:
    friend class WorkGraphBuilder;
    explicit WorkGraph(luisa::vector<detail::WorkGraphNode> nodes, luisa::vector<uint32_t> entry_points) noexcept :
        _nodes(std::move(nodes)), _entry_points(std::move(entry_points)) {}

    luisa::vector<detail::WorkGraphNode> _nodes;
    luisa::vector<uint32_t> _entry_points;
};

class WorkGraphBuilder {
public:

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

        return WorkGraphNode<NodeType, InputRecord>(this, node_index);
    }

    LUISA_DSL_API WorkGraph build() noexcept;

private:
    friend detail::WorkGraphNode& detail::index_to_node(WorkGraphBuilder*, uint) noexcept;
    friend detail::WorkGraphEdge& detail::indices_to_edge(WorkGraphBuilder*, uint, uint) noexcept;
    luisa::vector<detail::WorkGraphNode> _nodes;
};

} // namespace luisa::compute