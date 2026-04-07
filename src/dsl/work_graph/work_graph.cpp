#include <luisa/dsl/work_graph/work_graph.h>

namespace luisa::compute {

namespace detail {

LUISA_DSL_API WorkGraphNode &index_to_node(WorkGraphBuilder *builder, uint node_index) noexcept {
    return builder->_nodes[node_index];
}

LUISA_DSL_API WorkGraphEdge &indices_to_edge(WorkGraphBuilder *builder, uint node_index, uint edge_index) noexcept {
    return builder->_nodes[node_index].out_edges[edge_index];
}

LUISA_DSL_API WorkGraphNodeArray &index_to_node_array(WorkGraphBuilder *builder, uint node_array_index) noexcept {
    return builder->_node_arrays[node_array_index];
}

} // namespace luisa::compute::detail

static bool visit(
    size_t i,
    luisa::span<const detail::WorkGraphNode> nodes,
    luisa::span<const detail::WorkGraphNodeArray> node_arrays,
    luisa::span<uint8_t> marks,
    luisa::vector<uint32_t>& entry_points
) {
    // marks: (has_in_edge << 2) | (heavy_mark << 1) | (light_mark)
    if (marks[i] & 2) {
        return true;
    }

    if (marks[i] & 1) {
        return false;
    }

    marks[i] |= 1;

    for (auto const& edge : nodes[i].out_edges) {
        if (edge.dest_array != ~0u) {
            auto const& array = node_arrays[edge.dest_array];
            for (uint s = array.start; s < array.start + array.count; s += 1) {
                marks[s] |= 4;
                bool ok = luisa::compute::visit(s, nodes, node_arrays, marks, entry_points);
                if (!ok) { return false; }
            }
        }
        else {
            marks[edge.dest] |= 4;
            bool ok = luisa::compute::visit(edge.dest, nodes, node_arrays, marks, entry_points);
            if (!ok) { return false; }
        }
    }

    marks[i] &= ~1;
    marks[i] |= 2;
    return true;
}

LUISA_DSL_API WorkGraphBuilder::WorkGraphBuilder(luisa::string name) : _name(std::move(name)), _nodes() {}

// validates topology of work graph, validates names of nodes are unique,
// and populates entry points
LUISA_DSL_API WorkGraph WorkGraphBuilder::build() noexcept {
    // DFS to verify it is a DAG
    luisa::vector<uint8_t> marks;
    luisa::vector<uint32_t> entry_points;
    marks.resize(_nodes.size(), 0);

    for (size_t i = 0; i < _nodes.size(); i += 1) {
        LUISA_ASSERT(_nodes[i].defined, "all nodes must be defined");

        bool ok = luisa::compute::visit(i, _nodes, _node_arrays, marks, entry_points);

        // TODO: allow for single node cycle
        LUISA_ASSERT(ok, "work graph must be a DAG");
    }

    luisa::unordered_set<luisa::string_view> names;
    for (auto const& node : _nodes) {
        auto [_, name_unique] = names.insert(node.name);
        LUISA_ASSERT(!node.name.empty(), "names of work graph nodes cannot be empty");
        LUISA_ASSERT(name_unique, "names of work graph nodes must be unique");
    }

    names.clear();
    for (auto const& node_array : _node_arrays) {
        auto [_, name_unique] = names.insert(node_array.array_name);
        LUISA_ASSERT(!node_array.array_name.empty(), "names of work graph node arrays cannot be empty");
        LUISA_ASSERT(name_unique, "names of work graph node arrays must be unique");
    }

    for (size_t i = 0; i < _nodes.size(); i += 1) {
        if ((marks[i] & 4) == 0) {
            entry_points.push_back(i);
        }
    }

    return WorkGraph(std::move(_name), std::move(_nodes), std::move(_node_arrays), std::move(entry_points));
}

} // namespace luisa::compute