#include <luisa/runtime/graph/capture_node.h>
#include <luisa/runtime/graph/graph_builder.h>
using namespace luisa::compute::graph;

CaptureNodeBase::CaptureNodeBase(GraphBuilder *builder) noexcept
    : GraphNode{builder, GraphNodeType::Capture}, _capture_id{builder->_capture_nodes.size()} {}