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
        auto entry = c.entry();
        expect(entry != nullptr);
    };

    "operator_bracket_returns_continuation"_test = [] {
        auto c = make_simple_coro();
        auto sub1 = c[1u];
        expect(sub1 != nullptr);
        auto tag = sub1->function().tag();
        expect(tag == Function::Tag::CALLABLE);
    };

    "operator_bracket_out_of_range_returns_null"_test = [] {
        auto c = make_simple_coro();
        auto sub = c[999u];
        expect(sub == nullptr);
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
}

}// namespace

int main(int /*argc*/, char * /*argv*/[]) {
    reg_coro_func();
    return 0;
}
