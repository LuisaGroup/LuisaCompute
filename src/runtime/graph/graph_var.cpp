#include <luisa/runtime/graph/graph_var.h>
#include <luisa/runtime/graph/graph_builder.h>

using namespace luisa::compute::graph;

void GraphVarBase::graphviz_def(std::ostream &o) const noexcept {
    graphviz_id(o);
    o << "[shape=rectangle,";
    if (!var_name().empty()) o << "label=\"" << var_name() << "\",";
    o << "]";
}

void GraphVarBase::graphviz_arg_usage(std::ostream &o) const noexcept {
    if (is_sub_var()) {
        o << GraphVarBase::graphviz_prefix << input_var_id();
        o << " -> ";
        graphviz_id(o);
        o << "[color=blue, arrowhead=none]\n";
        for (auto &id : _other_dependent_var_ids) {
            o << GraphVarBase::graphviz_prefix << id;
            o << " -> ";
            graphviz_id(o);
            o << "[color=orange, arrowhead=none]";
            o << "\n";
        }
    }
}

GraphVarBase &GraphVarBase::set_var_name(string_view name) noexcept {
    _name = name;
    LUISA_ASSERT(GraphBuilder::is_building(), "set_var_name can only be invoked in Graph Def");
    GraphBuilder::current()->sub_var(sub_var_id())->_name = name;
    return *this;
}
