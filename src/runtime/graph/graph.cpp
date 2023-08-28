#pragma once
#include <luisa/runtime/graph/graph.h>
#include <luisa/backends/ext/graph_ext.h>

using namespace luisa::compute::graph;

GraphBase::GraphBase(GraphExt *graph_ext, const GraphBuilder *builder) noexcept
    : _graph_ext{graph_ext}, _builder{make_unique<GraphBuilder>(*builder)} {
    LUISA_ASSERT(_graph_ext != nullptr, "invalid GraphExt handle");
    _impl = _graph_ext->create_graph_interface();
}

GraphBase::~GraphBase() noexcept {
    LUISA_ASSERT(_graph_ext != nullptr, "invalid GraphExt handle");
    if (_impl != nullptr) _graph_ext->destroy_graph_interface(_impl);
}