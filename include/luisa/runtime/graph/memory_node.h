#pragma once
#include <luisa/runtime/graph/graph_node.h>
#include <luisa/runtime/graph/memory_node_direction.h>

namespace luisa::compute::graph {
class GraphBuilder;

class LC_RUNTIME_API MemoryNode : public GraphNode {
public:
    MemoryNode(GraphBuilder *builder, GraphSubVarId src_var_id, GraphSubVarId dst_var_id, MemoryNodeDirection direction) noexcept;
    auto src_var_id() const noexcept { return _src_var_id; }
    auto dst_var_id() const noexcept { return _dst_var_id; }
    auto direction() const noexcept { return _direction; }
    auto memory_node_id() const noexcept { return _memory_node_id; }
private:
    GraphNodeId _memory_node_id;
    MemoryNodeDirection _direction;
    GraphSubVarId _src_var_id;
    GraphSubVarId _dst_var_id;
};
}// namespace luisa::compute::graph