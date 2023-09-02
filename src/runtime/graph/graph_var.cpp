#include <luisa/runtime/graph/graph_var.h>
#include <luisa/runtime/graph/graph_builder.h>

using namespace luisa::compute::graph;

void GraphVarBase::graphviz_def(std::ostream &o) const noexcept {
    graphviz_id(o);
    o << "[shape=rectangle,";
    if (!var_name().empty()) o << "label=\"" << var_name() << "\",";
    o << "]";
}

GraphVarBase &GraphVarBase::set_var_name(string_view name) noexcept {
    _name = name;
    LUISA_ASSERT(GraphBuilder::is_building(), "set_var_name can only be invoked in Graph Def");
    GraphBuilder::current()->_vars[arg_id()]->_name = name;
    return *this;
}
