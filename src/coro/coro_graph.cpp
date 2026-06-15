#include <luisa/coro/coro_graph.h>
#include <luisa/core/stl/memory.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/passes/coro_materialize.h>
#include <luisa/xir/passes/coro_split.h>

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

    static_cast<void>(m);
    xir::CoroSplitInfo split;
    return from_module(m, info, cfg, split);
}

[[nodiscard]] CoroGraph CoroGraph::from_module(
    xir::Module &m, const xir::CoroMaterializeInfo &info,
    const xir::CoroCfgDistillResult &cfg,
    const xir::CoroSplitInfo &split) noexcept {

    static_cast<void>(m);
    CoroGraph graph;

    luisa::vector<const xir::CallableFunction *> callables(cfg.scopes.size(), nullptr);
    for (auto &subroutine : split.subroutines) {
        if (subroutine.scope_index < callables.size()) {
            callables[subroutine.scope_index] = subroutine.callable;
        }
    }

    // --- Build nodes from cfg-distill scopes ---
    for (size_t i = 0u; i < cfg.scopes.size(); ++i) {
        auto &scope = cfg.scopes[i];
        Node node;
        node.index = i;
        node.is_terminal = scope.is_terminal;
        node.callable = (i < callables.size()) ? callables[i] : nullptr;
        node.token = scope.trigger_token;
        node.name = scope.trigger_name.has_value() ? *scope.trigger_name : luisa::string{};

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
