#include <luisa/coro/coro_graph.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/memory.h>
#include <luisa/dsl/coro_frame.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/passes/coro_materialize.h>
#include <luisa/xir/passes/coro_split.h>

namespace luisa::compute::coro {

namespace {

static void append_unique(luisa::vector<size_t> &fields, size_t field) noexcept {
    if (std::find(fields.begin(), fields.end(), field) == fields.end()) {
        fields.emplace_back(field);
    }
}

static void append_reserved_fields(luisa::vector<size_t> &fields) noexcept {
    for (auto i = 0u; i < CoroFrameDesc::reserved_field_count; i++) {
        append_unique(fields, i);
    }
}

}// namespace

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

[[nodiscard]] luisa::string CoroGraph::dump() const noexcept {
    luisa::string s;
    for (auto &node : _nodes) {
        auto name = node.name.empty() ? luisa::string{"<entry>"} : node.name;
        s.append(luisa::format("Node {} '{}' token={} terminal={}\n",
                               node.index, name, node.token, node.is_terminal));
        s.append(luisa::format("  Input Fields: {}\n", node.input_fields));
        s.append(luisa::format("  Output Fields: {}\n", node.output_fields));
        s.append(luisa::format("  Transition Targets: {}\n", node.targets));
    }
    for (auto &edge : _edges) {
        s.append(luisa::format("Edge {} -> {} load={} store={}\n",
                               edge.from_index, edge.to_index,
                               edge.load_fields, edge.store_fields));
    }
    return s;
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
        LUISA_ASSERT(
            subroutine.scope_index < callables.size() &&
                callables[subroutine.scope_index] == nullptr &&
                subroutine.callable != nullptr &&
                subroutine.trigger_token ==
                    cfg.scopes[subroutine.scope_index].trigger_token,
            "CoroGraph received inconsistent split metadata for scope {}.",
            subroutine.scope_index);
        callables[subroutine.scope_index] = subroutine.callable;
    }
    if (!split.subroutines.empty()) {
        LUISA_ASSERT(
            split.subroutines.size() == cfg.scopes.size(),
            "CoroGraph received {} callable(s) for {} scope(s).",
            split.subroutines.size(), cfg.scopes.size());
        for (size_t scope_index = 0u;
             scope_index < callables.size(); ++scope_index) {
            LUISA_ASSERT(
                callables[scope_index] != nullptr,
                "CoroGraph is missing the callable for scope {}.",
                scope_index);
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
        auto [_, token_inserted] =
            graph._token_to_index.emplace(stored.token, i);
        LUISA_ASSERT(token_inserted,
                     "CoroGraph received duplicate trigger token {}.",
                     stored.token);
        if (!stored.name.empty()) {
            graph._name_to_index.emplace(stored.name, i);
        }
    }

    for (auto &te : info.edges) {
        Edge *edge_ptr = nullptr;
        for (auto &edge : graph._edges) {
            if (edge.from_index == te.from_scope && edge.to_index == te.to_scope) {
                edge_ptr = &edge;
                break;
            }
        }
        if (edge_ptr == nullptr) {
            auto &edge = graph._edges.emplace_back();
            edge.from_index = te.from_scope;
            edge.to_index = te.to_scope;
            edge_ptr = &edge;
        }
        for (auto field : te.load_fields) {
            append_unique(edge_ptr->load_fields, field);
        }
        for (auto field : te.store_fields) {
            append_unique(edge_ptr->store_fields, field);
        }
    }

    for (auto &node : graph._nodes) {
        append_reserved_fields(node.input_fields);
        append_reserved_fields(node.output_fields);
    }
    for (auto &edge : graph._edges) {
        if (edge.from_index < graph._nodes.size()) {
            auto &from_node = graph._nodes[edge.from_index];
            append_unique(from_node.targets, edge.to_index);
            for (auto field : edge.store_fields) {
                append_unique(from_node.output_fields, field);
            }
        }
        if (edge.to_index < graph._nodes.size()) {
            auto &to_node = graph._nodes[edge.to_index];
            for (auto field : edge.load_fields) {
                append_unique(to_node.input_fields, field);
            }
        }
    }
    for (auto &node : graph._nodes) {
        luisa::sort(node.input_fields.begin(), node.input_fields.end());
        luisa::sort(node.output_fields.begin(), node.output_fields.end());
        luisa::sort(node.targets.begin(), node.targets.end());
    }

    return graph;
}

}// namespace luisa::compute::coro
