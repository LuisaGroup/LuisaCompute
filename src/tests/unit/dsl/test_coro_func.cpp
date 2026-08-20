// Test Coroutine type: CTAD, entry(), operator[], subroutine_count,
// graph(), frame_desc(), Generator<T> wrapping
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>

#include "ut/ut.hpp"

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

auto make_simple_coro = [] {
    return Coroutine{[](Var<int> x) {
        $suspend("checkpoint");
    }};
};

void reg_coro_func() {

    "construct_with_explicit_type"_test = [] {
        auto c = Coroutine<void(int)>{[](Var<int> x) {
            $suspend("s");
        }};
        expect(c.subroutine_count() >= 2u);
    };

    "ctad_deduction_from_lambda"_test = [] {
        Coroutine c = [](Var<int> x) {
            $suspend("s");
        };
        expect(c.subroutine_count() >= 2u);
    };

    "entry_returns_valid_builder"_test = [] {
        auto c = make_simple_coro();
        expect(static_cast<bool>(c.entry()));
    };

    "operator_bracket_returns_continuation"_test = [] {
        auto c = make_simple_coro();
        expect(static_cast<bool>(c[1u]));
    };

    "operator_bracket_out_of_range_returns_empty"_test = [] {
        auto c = make_simple_coro();
        expect(!static_cast<bool>(c[999u]));
    };

    "subroutine_count_correct"_test = [] {
        auto c = make_simple_coro();
        expect(c.subroutine_count() >= 2u);
    };

    "graph_not_empty"_test = [] {
        auto c = make_simple_coro();
        auto &g = c.graph();
        expect(g.node_count() >= 2u);
    };

    "graph_entry_node_has_index_0"_test = [] {
        auto c = make_simple_coro();
        auto &g = c.graph();
        expect(g.node(0u).index == 0u);
        expect(g.node(0u).token == 0u);
        expect(!g.node(0u).is_terminal);
    };

    "graph_continuation_node_has_name_and_token"_test = [] {
        auto c = make_simple_coro();
        auto &g = c.graph();
        auto &n1 = g.node(1u);
        expect(n1.name == "checkpoint");
        expect(!n1.is_terminal);
    };

    "frame_desc_accessible"_test = [] {
        auto c = make_simple_coro();
        auto &fd = c.frame_desc();
        expect(fd.field_count() == 0u);
        expect(fd.total_size() == 0u);
    };

    "function_builder_accessible"_test = [] {
        auto c = make_simple_coro();
        auto fb = c.function_builder();
        expect(fb != nullptr);
    };

    "function_object_accessible"_test = [] {
        auto c = make_simple_coro();
        auto f = c.function();
        auto tag = f.tag();
        expect(tag == Function::Tag::COROUTINE);
    };

    "generator_construction_basic"_test = [] {
        Generator<int, int> gen([](Var<int> unused) noexcept {
            $suspend("gen_checkpoint");
            $suspend();
        });
        expect(gen.function_builder() != nullptr);
    };

    "generator_function_access"_test = [] {
        Generator<int, int> gen([](Var<int> unused) noexcept {
            $suspend("gen_checkpoint");
            $suspend();
        });
        auto f = gen.function();
        expect(f.tag() == Function::Tag::COROUTINE);
    };

    "coroutine_with_multiple_suspends"_test = [] {
        Coroutine c = [](Var<int> x) {
            $suspend("A");
            $suspend("B");
            $suspend("C");
        };
        expect(c.subroutine_count() >= 4u);
        expect(c.graph().node_count() >= 4u);
    };

    "continuations_project_source_arguments_by_use"_test = [] {
        auto c = Coroutine<void(Buffer<uint>, Buffer<uint>)>{
            [](BufferUInt before, BufferUInt after) noexcept {
                before.write(0u, 11u);
                $suspend("between-independent-resources");
                after.write(0u, 22u);
            }};
        auto lowered =
            luisa::compute::detail::compile_coroutine_pipeline(
                c.function_builder());
        auto *continuation =
            lowered.graph.node_by_name(
                "between-independent-resources");
        expect(continuation != nullptr);
        expect(lowered.subroutines.size() == 2u);
        expect(lowered.subroutine_source_argument_indices.size() ==
               lowered.subroutines.size());
        expect(lowered.subroutine_source_argument_indices[0u] ==
               luisa::vector<size_t>{0u})
            << "entry must retain only its first source resource";
        expect(lowered.subroutine_source_argument_indices[continuation->index] ==
               luisa::vector<size_t>{1u})
            << "resume must retain only its second source resource";
        expect(lowered.subroutines[0u]->arguments().size() == 2u);
        expect(lowered.subroutines[continuation->index]
                   ->arguments()
                   .size() == 2u);
    };
}

}// namespace

int main(int /*argc*/, char * /*argv*/[]) {
    reg_coro_func();
    return 0;
}
