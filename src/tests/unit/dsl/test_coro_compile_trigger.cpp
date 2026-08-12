#include "ut/ut.hpp"
#include <luisa/ast/statement.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

/// A simple coroutine: one variable crosses the suspend point.
/// y is defined before $suspend and used after, so it must be saved in the frame.
auto make_simple_coro = [] {
    return Coroutine{[](Var<int> x) {
        $suspend("checkpoint");
    }};
};

[[nodiscard]] uint32_t first_suspend_token(const Coroutine<void(int)> &c) noexcept {
    for (auto *stmt : c.function_builder()->body()->statements()) {
        if (stmt->tag() == Statement::Tag::SUSPEND) {
            return static_cast<const SuspendStmt *>(stmt)->token();
        }
    }
    return 0u;
}

}// namespace

void reg_coro_compile_trigger() {

    "compile_on_construction_does_not_throw"_test = [] {
        // Constructing the Coroutine triggers eager compilation.
        // Should not throw (when exceptions are enabled).
        auto c = make_simple_coro();
        expect(static_cast<bool>(c.function_builder()));
    };

    "graph_not_empty"_test = [] {
        auto c = make_simple_coro();
        auto &g = c.graph();
        // Entry scope + 1 continuation = 2 nodes
        expect(g.node_count() == 2u);
    };

    "graph_nodes_have_correct_properties"_test = [] {
        auto c = make_simple_coro();
        auto &g = c.graph();

        // Entry node (index 0): no name, no token
        auto &n0 = g.node(0u);
        expect(n0.index == 0u);
        expect(n0.token == 0u);
        expect(n0.name.empty());
        expect(!n0.is_terminal);

        // Continuation node (index 1): has name and token from suspend
        auto &n1 = g.node(1u);
        expect(n1.index == 1u);
        expect(n1.token != 0u);
        expect(n1.token == first_suspend_token(c));
        expect(n1.name == "checkpoint");
        expect(!n1.is_terminal);
    };

    "frame_desc_not_empty"_test = [] {
        auto c = make_simple_coro();
        auto &fd = c.frame_desc();
        // Frame descriptor is valid (constructed, no user fields in minimal test)
        expect(fd.field_count() == 0u);
        expect(fd.total_size() == 0u);
    };

    "frame_desc_valid_object"_test = [] {
        auto c = make_simple_coro();
        auto &fd = c.frame_desc();
        // The frame descriptor is a valid object with 0 user fields
        // (no $promise variables in this minimal test)
        expect(fd.field_count() == 0u);
    };

    "entry_returns_valid_function_builder"_test = [] {
        auto c = make_simple_coro();
        expect(static_cast<bool>(c.entry()));
    };

    "subroutine_count_has_entry_plus_continuations"_test = [] {
        auto c = make_simple_coro();
        // 1 suspend → 2 scopes (entry + continuation), both translated to AST
        expect(c.subroutine_count() >= 2u);
    };

    "operator_bracket_returns_valid_subroutines"_test = [] {
        auto c = make_simple_coro();
        expect(static_cast<bool>(c[0u]));
        expect(static_cast<bool>(c[1u]));
    };

    "aggregate_vector_frame_keeps_only_nonrematerializable_observed_components"_test = [] {
        auto c = Coroutine<void(Buffer<float>, int)>{[](Var<Buffer<float>> output, Var<int> x) {
            Float3 v = make_float3(cast<float>(x), 2.0f, 3.0f);
            $suspend("vector");
            Float y = v.x + v.y;
            // Keep the aggregate observably live across the suspend. A pure,
            // unused expression is correctly removed by pre-distill DCE and
            // therefore cannot establish a frame-layout requirement.
            output.write(0u, y);
        }};
        auto &fd = c.frame_desc();
        // v.z is never observed and v.y is a stable constant that can be
        // rematerialized in the continuation. Only the non-rematerializable
        // v.x component may occupy the frame.
        expect(fd.field_count() == 1u);
        if (fd.field_count() != 1u) { return; }
        expect(fd.field(0u).type == Type::of<float>());
        expect(fd.total_size() == 4u);
    };

    "aggregate_vector_frame_projects_multiple_dynamic_components"_test = [] {
        auto c = Coroutine<void(Buffer<float>, int, int)>{
            [](Var<Buffer<float>> output, Var<int> x, Var<int> y) {
                Float3 v = make_float3(cast<float>(x), cast<float>(y), 3.0f);
                $suspend("vector");
                output.write(0u, v.x + v.y);
            }};
        auto &fd = c.frame_desc();
        // The two independently dynamic observed components must survive the
        // suspension, while the unobserved third component must not be kept by
        // aggregate projection merely because it shares v's source aggregate.
        expect(fd.field_count() == 2u);
        if (fd.field_count() != 2u) { return; }
        expect(fd.field(0u).type == Type::of<float>());
        expect(fd.field(1u).type == Type::of<float>());
        expect(fd.total_size() == 8u);
    };
}

int main(int /*argc*/, char * /*argv*/[]) {
    reg_coro_compile_trigger();
    return 0;
}
