#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/ast/coro_suspend.h>
#include <luisa/coro/coro_slot_access.h>

namespace luisa::compute::xir {
class Module;
struct CoroMaterializeInfo;
struct CoroCfgDistillResult;
struct CoroSplitInfo;
class CallableFunction;
}// namespace luisa::compute::xir

namespace luisa::compute::coro {

/// Token-based transition graph between continuation scopes.
/// Built from coro CFG analysis, split feedback, and materialization feedback.
class LUISA_CORO_API CoroGraph {

public:
    /// A node in the coroutine graph — one per continuation scope.
    struct Node {
        size_t index{0u};               // scope index (0 = entry)
        luisa::string name;             // suspend name (empty for entry)
        size_t token{0u};               // suspend token value (0 for entry)
        bool is_terminal{false};        // terminal scope (no outgoing transitions)
        const xir::CallableFunction *callable{nullptr};// pointer to the continuation callable
        luisa::vector<size_t> input_fields;
        luisa::vector<size_t> output_fields;
        // Physical frame fields in live_begin(scope). Unlike input_fields,
        // this includes dormant state that passes through this continuation
        // and is first consumed by a later one. A compacting scheduler must
        // preserve this complete token-indexed payload while relocating a
        // queued frame.
        luisa::vector<size_t> relocation_fields;
        luisa::vector<size_t> targets;

        [[nodiscard]] auto input_field_span() const noexcept { return luisa::span{input_fields}; }
        [[nodiscard]] auto output_field_span() const noexcept { return luisa::span{output_fields}; }
        [[nodiscard]] auto relocation_field_span() const noexcept { return luisa::span{relocation_fields}; }
        [[nodiscard]] auto target_span() const noexcept { return luisa::span{targets}; }
    };

    /// A directed edge between two continuation scopes.
    struct Edge {
        size_t from_index{0u};          // source node index
        size_t to_index{0u};            // target node index
        luisa::vector<size_t> load_fields;  // frame fields loaded at resume
        luisa::vector<size_t> store_fields; // frame fields stored at suspend
    };

    /// One static suspend instruction. Unlike Edge, boundaries are never
    /// coalesced merely because they have the same source and destination;
    /// complete extension objects and their typed slot projections remain
    /// one-to-one with the source suspension site.
    struct Boundary {
        size_t index{0u};
        size_t from_index{0u};
        size_t to_index{0u};
        size_t token{0u};
        luisa::vector<CoroSuspendExtensionPtr> extensions;
        // Owner binding index -> typed projection into existing frame slots.
        luisa::vector<CoroSlotAccess> bindings;

        [[nodiscard]] auto extension_span() const noexcept {
            return luisa::span{extensions};
        }
        [[nodiscard]] auto binding_span() const noexcept {
            return luisa::span{bindings};
        }
    };

private:
    luisa::vector<Node> _nodes;
    luisa::vector<Edge> _edges;
    luisa::vector<Boundary> _boundaries;
    luisa::unordered_map<size_t, size_t> _token_to_index;
    luisa::unordered_map<luisa::string, size_t> _name_to_index;

public:
    CoroGraph() noexcept = default;
    ~CoroGraph() noexcept = default;

    // Disallow copy; allow move
    CoroGraph(const CoroGraph &) = delete;
    CoroGraph &operator=(const CoroGraph &) = delete;
    CoroGraph(CoroGraph &&) noexcept = default;
    CoroGraph &operator=(CoroGraph &&) noexcept = default;

    // --- Accessors ---

    [[nodiscard]] size_t node_count() const noexcept;
    [[nodiscard]] const Node &node(size_t index) const noexcept;
    [[nodiscard]] size_t entry_index() const noexcept { return 0u; }

    /// Lookup by suspend token. Returns nullptr if not found.
    [[nodiscard]] const Node *node_by_token(size_t token) const noexcept;

    /// Lookup by suspend name. Returns nullptr if not found.
    [[nodiscard]] const Node *node_by_name(luisa::string_view name) const noexcept;

    [[nodiscard]] size_t edge_count() const noexcept;
    [[nodiscard]] const Edge *edge(size_t from, size_t to) const noexcept;
    [[nodiscard]] size_t boundary_count() const noexcept {
        return _boundaries.size();
    }
    [[nodiscard]] const Boundary &boundary(size_t index) const noexcept {
        return _boundaries[index];
    }
    [[nodiscard]] luisa::string dump() const noexcept;

    // --- Iterators ---

    [[nodiscard]] auto nodes() const noexcept { return luisa::span{_nodes}; }
    [[nodiscard]] auto edges() const noexcept { return luisa::span{_edges}; }
    [[nodiscard]] auto boundaries() const noexcept {
        return luisa::span{_boundaries};
    }

    // --- Construction ---

    /// Build a CoroGraph from a post-materialize module and analysis results.
    ///
    /// @param m         Module after coro-split and coro-materialize.
    /// @param info      CoroMaterializeInfo with TransitionEdge data and name_to_field map.
    /// @param cfg       CoroCfgDistillResult with scope/token/name/terminal info.
    [[nodiscard]] static CoroGraph from_module(
        xir::Module &m, const xir::CoroMaterializeInfo &info,
        const xir::CoroCfgDistillResult &cfg) noexcept;
    [[nodiscard]] static CoroGraph from_module(
        xir::Module &m, const xir::CoroMaterializeInfo &info,
        const xir::CoroCfgDistillResult &cfg,
        const xir::CoroSplitInfo &split) noexcept;
};

}// namespace luisa::compute::coro
