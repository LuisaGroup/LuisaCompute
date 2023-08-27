#pragma once
#include <luisa/runtime/graph/graph.h>

namespace luisa::compute {
thread_local Graph *Graph::current = nullptr;

void Graph::set_current(Graph *graph) noexcept { current = graph; }

Graph *Graph::get_current() noexcept { return current; }
}// namespace luisa::compute