#include <luisa/runtime/graph/memory_node.h>
#include <luisa/runtime/graph/graph_builder.h>
using namespace luisa::compute::graph;

MemoryNode::MemoryNode(GraphBuilder *builder, GraphSubVarId src_var_id, GraphSubVarId dst_var_id, MemoryNodeDirection direction) noexcept
    : GraphNode{builder, GraphNodeType::MemoryCopy},
      _memory_node_id{builder->_memory_nodes.size()},
      _src_var_id{src_var_id},
      _dst_var_id{dst_var_id},
      _direction{direction} {
    this->add_arg_usage(src_var_id, Usage::READ);
    this->add_arg_usage(dst_var_id, Usage::WRITE);
}