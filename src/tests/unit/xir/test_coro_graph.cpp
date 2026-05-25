#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/coro/coro_graph.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/coroutine.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coroutine_split.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace luisa::compute::coroutine;
using namespace boost::ut;
using namespace boost::ut::literals;

void reg_coro_graph() {

    "coro_graph_returns_null_for_unsupported_split"_test = [] {
        CoroutineSplitInfo bad;
        bad.is_supported = false;
        auto graph = CoroGraph::from_xir_split(bad);
        expect(graph == nullptr);
    };

    "coro_graph_wraps_split_continuations_as_ast_builders"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *a = b.alloca_local(Type::of<float>());
        b.coro_register(a, "a");
        b.coro_suspend(1u);
        b.coro_suspend(2u);
        b.load(Type::of<float>(), a);
        b.return_void();

        auto split = coroutine_split_run_on_function(k);
        expect(split.is_supported);

        auto graph = CoroGraph::from_xir_split(split);
        expect(graph != nullptr);
        expect(graph->frame_type() != nullptr);
        // Entry + 2 resumed continuations.
        expect(graph->subroutine_count() == 3_u);
        // Entry node is reachable by coro_token_entry alias.
        expect(graph->entry() != nullptr);
        expect(graph->entry()->builder != nullptr);
        // Each non-null node carries a usable FunctionBuilder for DSL-side use.
        for (auto &&node : graph->nodes()) {
            expect(node.builder != nullptr);
        }
    };
}

int main() {
    reg_coro_graph();
    return 0;
}
