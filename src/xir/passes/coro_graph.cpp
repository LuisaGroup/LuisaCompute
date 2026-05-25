#include <luisa/core/logging.h>
#include <luisa/coro/coro_graph.h>
#include <luisa/xir/translators/xir2ast.h>

namespace luisa::compute::coroutine {

CoroGraph::CoroGraph(const Type *frame_type,
                     luisa::vector<uint> frame_slot_field_indices,
                     luisa::vector<Node> nodes,
                     luisa::vector<luisa::string> diagnostics) noexcept
    : _frame_type{frame_type},
      _frame_slot_field_indices{std::move(frame_slot_field_indices)},
      _nodes{std::move(nodes)},
      _diagnostics{std::move(diagnostics)} {}

CoroGraph::~CoroGraph() noexcept = default;

const CoroGraph::Node *CoroGraph::node(CoroToken token) const noexcept {
    for (auto &&n : _nodes) {
        if (n.token == token) { return &n; }
    }
    return nullptr;
}

luisa::shared_ptr<const CoroGraph>
CoroGraph::from_xir_split(const xir::CoroutineSplitInfo &split) noexcept {
    if (!split.is_supported || !split.changed) {
        for (auto &&d : split.diagnostics) {
            LUISA_WARNING("CoroGraph::from_xir_split: {}", d);
        }
        return nullptr;
    }
    luisa::vector<uint> field_indices;
    field_indices.reserve(split.frame_slots.size());
    for (auto &&slot : split.frame_slots) {
        field_indices.emplace_back(static_cast<uint>(slot.field_index));
    }
    luisa::vector<Node> nodes;
    nodes.reserve(split.continuations.size());
    for (auto &&cont : split.continuations) {
        if (cont.callable == nullptr) { continue; }
        auto ast = xir::xir_to_ast_translate(*cont.callable->definition(), {});
        if (ast == nullptr) {
            LUISA_WARNING("CoroGraph::from_xir_split: xir_to_ast_translate failed for continuation {}", cont.id);
            continue;
        }
        Node node{
            .token = static_cast<CoroToken>(cont.id == 0u ? coro_token_entry : cont.id + 1u),
            .builder = ast,
            .outgoing = {},
        };
        for (auto sus_id : cont.outgoing_suspends) {
            node.outgoing.emplace_back(static_cast<CoroToken>(sus_id + 1u));
        }
        nodes.emplace_back(std::move(node));
    }
    return luisa::make_shared<CoroGraph>(
        split.frame_type,
        std::move(field_indices),
        std::move(nodes),
        split.diagnostics);
}

}// namespace luisa::compute::coroutine
