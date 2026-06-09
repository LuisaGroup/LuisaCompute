#include <luisa/coro/coro_graph.h>
#include <luisa/core/stl/memory.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/passes/coro_materialize.h>

namespace luisa::compute::coro {

[[nodiscard]] size_t CoroGraph::node_count() const noexcept {
    return _nodes.size();
}

[[nodiscard]] const CoroGraph::Node &CoroGraph::node(size_t index) const noexcept {
    return _nodes[index];
}

[[nodiscard]] const CoroGraph::Node *CoroGraph::node_by_token(size_t token) const noexcept {
    auto it = _token_to_index.find(token);
    if (it == _token_to_index.end()) { return nullptr; }
    return &_nodes[it->second];
}

[[nodiscard]] const CoroGraph::Node *CoroGraph::node_by_name(luisa::string_view name) const noexcept {
    auto it = _name_to_index.find(luisa::string{name});
    if (it == _name_to_index.end()) { return nullptr; }
    return &_nodes[it->second];
}

[[nodiscard]] size_t CoroGraph::edge_count() const noexcept {
    return _edges.size();
}

[[nodiscard]] const CoroGraph::Edge *CoroGraph::edge(size_t from, size_t to) const noexcept {
    for (auto &e : _edges) {
        if (e.from_index == from && e.to_index == to) { return &e; }
    }
    return nullptr;
}

[[nodiscard]] CoroGraph CoroGraph::from_module(
    xir::Module &m, const xir::CoroMaterializeInfo &info,
    const xir::CoroCfgDistillResult &cfg) noexcept {

    CoroGraph graph;

    // --- Collect callables with frame args from the module ---
    // After coro-split, callables are created in scope order and appended to the function list.
    luisa::vector<const xir::CallableFunction *> callables;
    for (auto *f : m.function_list()) {
        if (!f->isa<xir::CallableFunction>() || f->definition() == nullptr) { continue; }
        auto *cf = static_cast<const xir::CallableFunction *>(f);
        // Check for frame arg (reference argument)
        bool has_frame = false;
        for (auto *arg : cf->arguments()) {
            if (arg->is_reference()) {
                has_frame = true;
                break;
            }
        }
        if (has_frame) { callables.push_back(cf); }
    }

    // --- Build nodes from cfg-distill scopes ---
    // Scope semantics:
    //   scope[0] is the entry (token=0). It may contain a suspend point with
    //   token T and name N.
    //   scope[i] (i>0) resumes at token = scope[i-1].suspend_token, and its
    //   node is named after scope[i-1].suspend_name.
    for (size_t i = 0u; i < cfg.scopes.size(); ++i) {
        auto &scope = cfg.scopes[i];
        Node node;
        node.index = i;
        node.is_terminal = scope.is_terminal;
        node.callable = (i < callables.size()) ? callables[i] : nullptr;

        if (i == 0u) {
            // Entry node: token=0, no name
            node.token = 0u;
            node.name = luisa::string{};
        } else {
            // Continuation node: token and name come from the PREVIOUS scope's suspend
            auto &prev = cfg.scopes[i - 1u];
            node.token = prev.suspend_token.has_value() ? *prev.suspend_token : 0u;
            node.name = prev.suspend_name.has_value() ? *prev.suspend_name : luisa::string{};
        }

        graph._nodes.push_back(std::move(node));

        // Build lookup maps (use the stored node, not the moved-from local)
        auto &stored = graph._nodes.back();
        graph._token_to_index.emplace(stored.token, i);
        if (!stored.name.empty()) {
            graph._name_to_index.emplace(stored.name, i);
        }
    }

    // --- Build edges from cfg-distill adjacency ---
    // cfg.edges[i] lists the scope indices that scope i transitions to.
    for (size_t from = 0u; from < cfg.edges.size(); ++from) {
        for (size_t to : cfg.edges[from]) {
            Edge edge;
            edge.from_index = from;
            edge.to_index = to;

            // Match with materialize TransitionEdge for field-wise info
            for (auto &te : info.edges) {
                if (te.from_scope == from && te.to_scope == to) {
                    edge.load_fields = te.load_fields;
                    edge.store_fields = te.store_fields;
                    break;
                }
            }
            graph._edges.push_back(std::move(edge));
        }
    }

    return graph;
}

}// namespace luisa::compute::coro
