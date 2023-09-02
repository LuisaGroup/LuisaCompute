#include <luisa/runtime/graph/graph_node.h>
#include <luisa/runtime/graph/graph_builder.h>

using namespace luisa::compute::graph;

GraphNode::GraphNode(GraphBuilder *builder, GraphNodeType type) noexcept
    : _node_id{builder->graph_nodes().size()},
      _type{type} {
    _arg_usage.clear();
}

void GraphNode::graphviz_def(std::ostream &o) const noexcept {
    graphviz_id(o);
    o << "[";
    if (!node_name().empty())
        o << "label=\"" << node_name() << "\", ";
    o << "]";
}

void GraphNode::graphviz_id(std::ostream &o) const noexcept {
    o << graphviz_prefix << _node_id;
}

void GraphNode::graphviz_arg_usages(std::ostream &o) const noexcept {
    for (auto &&[arg_id, usage] : _arg_usage) {
        o << GraphVarBase::graphviz_prefix << arg_id;
        o << "->";
        graphviz_id(o);
        if (usage == Usage::WRITE || usage == Usage::READ_WRITE)
            o << "[color=red, arrowhead = diamond, ]";
        else
            o << "[color=green, arrowhead = dot, ]";
        o << "\n";
    }
}