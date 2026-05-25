#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/ast/function_builder.h>
#include <luisa/xir/passes/coroutine_split.h>

// Thin AST-side wrapper over CoroutineSplitInfo. Holds the frame type and one
// shared FunctionBuilder per continuation so user code can compose schedulers
// in DSL. Mirrors the public shape of the old impl's coroutine::CoroGraph
// (see https://github.com/LuisaGroup/LuisaCompute/blob/coroutine/include/luisa/coro/coro_graph.h)
// but consumes the XIR coroutine_split pipeline instead of the archived Rust IR.
//
// Typical use:
//
//     // 1) Author the coroutine in DSL (using $suspend) and lower to XIR.
//     auto xir_module = ast_to_xir_translate(coro_fn_ast, {});
//     auto *coro_xir_fn = ...; // pick the kernel from xir_module
//
//     // 2) Run the split pass to produce continuation callables.
//     auto split = xir::coroutine_split_run_on_function(coro_xir_fn);
//
//     // 3) Wrap as a CoroGraph for DSL-side use.
//     auto graph = coroutine::CoroGraph::from_xir_split(split);
//
//     // 4) Build a scheduler that calls graph->subroutine(token)(args...).
//
// Custom schedulers extend by deriving from CoroScheduler (forthcoming) or
// by composing CoroGraph::subroutine() inside their own DSL kernels.

namespace luisa::compute::coroutine {

using CoroToken = uint32_t;
constexpr CoroToken coro_token_terminated = 0u;
constexpr CoroToken coro_token_entry = 1u;

class LUISA_XIR_API CoroGraph {

public:
    using ASTBuilder = luisa::shared_ptr<const detail::FunctionBuilder>;

    struct Node {
        CoroToken token;
        ASTBuilder builder;
        luisa::vector<CoroToken> outgoing;
    };

private:
    const Type *_frame_type{};
    luisa::vector<uint> _frame_slot_field_indices;
    luisa::vector<Node> _nodes;
    luisa::vector<luisa::string> _diagnostics;

public:
    CoroGraph(const Type *frame_type,
              luisa::vector<uint> frame_slot_field_indices,
              luisa::vector<Node> nodes,
              luisa::vector<luisa::string> diagnostics) noexcept;
    ~CoroGraph() noexcept;

    [[nodiscard]] auto frame_type() const noexcept { return _frame_type; }
    [[nodiscard]] auto &frame_slot_field_indices() const noexcept { return _frame_slot_field_indices; }
    [[nodiscard]] auto &nodes() const noexcept { return _nodes; }
    [[nodiscard]] auto &diagnostics() const noexcept { return _diagnostics; }

    [[nodiscard]] const Node *node(CoroToken token) const noexcept;
    [[nodiscard]] const Node *entry() const noexcept { return node(coro_token_entry); }
    [[nodiscard]] auto subroutine_count() const noexcept { return static_cast<uint>(_nodes.size()); }

    // Build a CoroGraph from an XIR coroutine_split result. Returns nullptr
    // and surfaces diagnostics via stderr if the split is unsupported.
    [[nodiscard]] static luisa::shared_ptr<const CoroGraph>
    from_xir_split(const xir::CoroutineSplitInfo &split) noexcept;
};

}// namespace luisa::compute::coroutine
