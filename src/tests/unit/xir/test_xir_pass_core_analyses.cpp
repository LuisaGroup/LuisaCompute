// Test for XIR call-graph and uniformity analyses.
// This test covers:
// - call roots and caller edges
// - conservative kernel/callable uniformity propagation

#include "ut/ut.hpp"

#include <luisa/ast/type_registry.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/call_graph.h>
#include <luisa/xir/passes/uniformity_analysis.h>
#include <luisa/xir/verifier.h>

#include <algorithm>
#include <limits>

#include "../../../../src/xir/passes/natural_loop.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

void reg_call_graph() {
    "call_graph_reports_roots_and_caller_edges"_test = [] {
        Module module;
        auto *callee = module.create_callable(Type::of<int>());
        auto *callee_argument = callee->create_value_argument(Type::of<int>());
        auto *callee_body = callee->create_body_block();
        auto *kernel = module.create_kernel();
        auto *kernel_body = kernel->create_body_block();
        auto *one = module.create_constant_one(Type::of<int>());

        XIRBuilder builder;
        builder.set_insertion_point(callee_body);
        builder.return_(callee_argument);
        builder.set_insertion_point(kernel_body);
        auto *first = builder.call(Type::of<int>(), callee, {one});
        auto *second = builder.call(Type::of<int>(), callee, {first});
        builder.return_void();

        auto graph = compute_call_graph(&module);
        expect(graph.root_functions().size() == 1u);
        expect(graph.root_functions().front() == kernel);
        auto kernel_edges = graph.call_edges(kernel);
        expect(kernel_edges.size() == 2u);
        expect(static_cast<bool>(
            (kernel_edges[0u] == first && kernel_edges[1u] == second) ||
            (kernel_edges[0u] == second && kernel_edges[1u] == first)));
        expect(graph.call_edges(callee).empty());
    };

    "call_graph_keeps_unused_definition_as_root"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *kernel_body = kernel->create_body_block();
        auto *unused = module.create_callable(nullptr);
        auto *unused_body = unused->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(kernel_body);
        builder.return_void();
        builder.set_insertion_point(unused_body);
        builder.return_void();

        auto graph = compute_call_graph(&module);
        expect(graph.root_functions().size() == 2u);
        expect(graph.call_edges(kernel).empty());
        expect(graph.call_edges(unused).empty());
    };

    "call_graph_does_not_invent_edge_from_function_argument_use"_test = [] {
        Module module;
        auto *function_value = module.create_callable(Type::of<int>());
        auto *function_body = function_value->create_body_block();
        auto *consumer = module.create_callable(nullptr);
        consumer->create_value_argument(Type::of<int>());
        auto *consumer_body = consumer->create_body_block();
        auto *kernel = module.create_kernel();
        auto *kernel_body = kernel->create_body_block();
        auto *one = module.create_constant_one(Type::of<int>());
        XIRBuilder builder;
        builder.set_insertion_point(function_body);
        builder.return_(one);
        builder.set_insertion_point(consumer_body);
        builder.return_void();
        builder.set_insertion_point(kernel_body);
        auto *call = builder.call(nullptr, consumer, {function_value});
        builder.return_void();

        auto graph = compute_call_graph(&module);
        expect(graph.call_edges(kernel).size() == 1u);
        expect(graph.call_edges(kernel).front() == call);
        expect(graph.call_edges(function_value).empty());
        expect(std::find(graph.root_functions().begin(),
                         graph.root_functions().end(),
                         function_value) != graph.root_functions().end());
    };

    "call_graph_counts_only_the_callee_operand_when_function_is_reused"_test = [] {
        Module module;
        auto *callee = module.create_callable(Type::of<int>());
        callee->create_value_argument(Type::of<int>());
        auto *callee_body = callee->create_body_block();
        auto *one = module.create_constant_one(Type::of<int>());
        auto *kernel = module.create_kernel();
        auto *kernel_body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(callee_body);
        builder.return_(one);
        builder.set_insertion_point(kernel_body);
        // Deliberately verifier-invalid: a Function is not a legal ordinary
        // call argument. It nevertheless remains a valid use-list stress case:
        // this CallInst has two uses of `callee`, but only operand zero is an
        // edge in the call graph.
        auto *call = builder.call(Type::of<int>(), callee, {callee});
        builder.return_void();

        auto graph = compute_call_graph(&module);
        auto edges = graph.call_edges(kernel);
        expect(edges.size() == 1u);
        expect(edges.front() == call);
        expect(std::find(graph.root_functions().begin(),
                         graph.root_functions().end(),
                         callee) == graph.root_functions().end());
    };

    "call_graph_null_module_is_empty"_test = [] {
        auto graph = compute_call_graph(nullptr);
        expect(graph.root_functions().empty());
        expect(graph.call_edges(nullptr).empty());
    };
}

void reg_uniformity_analysis() {
    "uniformity_propagates_only_proven_kernel_values"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *argument = kernel->create_value_argument(Type::of<uint32_t>());
        auto *body = kernel->create_body_block();
        auto *one = module.create_constant_one(Type::of<uint32_t>());
        auto *block_id = module.create_special_register(DerivedSpecialRegisterTag::BLOCK_ID);
        auto *dispatch_id = module.create_special_register(DerivedSpecialRegisterTag::DISPATCH_ID);
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *sum = builder.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD, {argument, one});
        builder.return_void();

        UniformityAnalysis analysis;
        analysis.analyze(kernel);
        expect(analysis.is_uniform(argument));
        expect(analysis.is_uniform(one));
        expect(analysis.is_uniform(sum));
        expect(analysis.is_uniform(block_id));
        expect(!analysis.is_uniform(dispatch_id));
        expect(!analysis.is_uniform(body));
    };

    "uniformity_reanalysis_clears_previous_function_facts"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *kernel_argument = kernel->create_value_argument(Type::of<int>());
        auto *kernel_body = kernel->create_body_block();
        auto *callable = module.create_callable(Type::of<int>());
        auto *callable_argument = callable->create_value_argument(Type::of<int>());
        auto *callable_body = callable->create_body_block();
        auto *one = module.create_constant_one(Type::of<int>());
        XIRBuilder builder;
        builder.set_insertion_point(kernel_body);
        auto *kernel_sum = builder.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {kernel_argument, one});
        builder.return_void();
        builder.set_insertion_point(callable_body);
        auto *callable_sum = builder.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {callable_argument, one});
        builder.return_(callable_sum);

        UniformityAnalysis analysis;
        analysis.analyze(kernel);
        expect(analysis.is_uniform(kernel_sum));
        analysis.analyze(callable);
        expect(!analysis.is_uniform(kernel_argument));
        expect(!analysis.is_uniform(kernel_sum));
        expect(!analysis.is_uniform(callable_argument));
        expect(!analysis.is_uniform(callable_sum));
        expect(analysis.is_uniform(one));
    };
}

void reg_natural_loop_discovery() {
    "natural_loop_discovers_counted_loop_with_bounds"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *def = kernel->definition();
        auto *entry = kernel->create_body_block();
        auto *header = def->create_basic_block();
        auto *latch = def->create_basic_block();
        auto *exit = def->create_basic_block();
        auto *zero = module.create_constant_zero(Type::of<uint>());
        auto *one = module.create_constant_one(Type::of<uint>());
        uint32_t bound_value = 10u;
        auto *bound = module.create_constant(Type::of<uint>(), &bound_value);

        XIRBuilder b;
        b.set_insertion_point(entry);
        b.br(header);
        b.set_insertion_point(header);
        auto *iv = b.phi(Type::of<uint>());
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, bound});
        b.cond_br(cond, latch, exit);
        b.set_insertion_point(latch);
        auto *next = b.call(Type::of<uint>(), ArithmeticOp::BINARY_ADD, {iv, one});
        b.br(header);
        b.set_insertion_point(exit);
        b.return_void();
        iv->add_incoming(zero, entry);
        iv->add_incoming(next, latch);

        auto dom_tree = compute_dom_tree(kernel);
        auto loops = discover_natural_loops(def, dom_tree);
        expect(loops.size() == 1u);
        auto &loop = loops.front();
        expect(loop.header == header);
        expect(loop.preheader == entry);
        expect(loop.latches.size() == 1u && loop.latches.front() == latch);
        expect(loop.body_blocks.size() == 1u && loop.body_blocks.front() == latch);
        expect(loop.exit_blocks.size() == 1u && loop.exit_blocks.front() == exit);
        expect(loop.exit_edges.size() == 1u);
        expect(loop.exit_edges.front().first == header);
        expect(loop.exit_edges.front().second == exit);
        expect(loop.back_edges.size() == 1u);

        auto bounds = analyze_loop_bounds(loop);
        expect(bounds.is_valid());
        expect(bounds.induction_phi == iv);
        expect(bounds.start_value == zero);
        expect(bounds.bound_value == bound);
        expect(bounds.body_entry == latch);
        expect(bounds.exit_block == exit);
        expect(bounds.continue_on_true);
        expect(bounds.induction_is_lhs);
        expect(bounds.normalized_strict_less);
        expect(bounds.stride_is_constant && bounds.stride == 1);
        expect(bounds.trip_count_is_constant);
        expect(bounds.constant_trip_count == 10u);
    };

    "natural_loop_normalizes_inverted_branch_and_operand_order"_test = [] {
        auto run_case = [](ArithmeticOp op, bool iv_is_lhs,
                           bool body_on_true) noexcept {
            Module module;
            auto *kernel = module.create_kernel();
            auto *def = kernel->definition();
            auto *entry = kernel->create_body_block();
            auto *header = def->create_basic_block();
            auto *latch = def->create_basic_block();
            auto *exit = def->create_basic_block();
            auto *zero = module.create_constant_zero(Type::of<uint>());
            auto *one = module.create_constant_one(Type::of<uint>());
            uint32_t bound_value = 10u;
            auto *bound =
                module.create_constant(Type::of<uint>(), &bound_value);
            XIRBuilder b;
            b.set_insertion_point(entry);
            b.br(header);
            b.set_insertion_point(header);
            auto *iv = b.phi(Type::of<uint>());
            auto *condition = b.call(
                Type::of<bool>(), op,
                iv_is_lhs ?
                    std::initializer_list<Value *>{iv, bound} :
                    std::initializer_list<Value *>{bound, iv});
            if (body_on_true) {
                b.cond_br(condition, latch, exit);
            } else {
                b.cond_br(condition, exit, latch);
            }
            b.set_insertion_point(latch);
            auto *next = b.call(
                Type::of<uint>(), ArithmeticOp::BINARY_ADD, {iv, one});
            b.br(header);
            b.set_insertion_point(exit);
            b.return_void();
            iv->add_incoming(zero, entry);
            iv->add_incoming(next, latch);

            auto loops = discover_natural_loops(
                def, compute_dom_tree(kernel));
            expect(loops.size() == 1u);
            auto bounds = analyze_loop_bounds(loops.front());
            expect(bounds.is_valid());
            expect(bounds.normalized_strict_less);
            expect(bounds.trip_count_is_constant);
            expect(bounds.constant_trip_count == 10u);
        };
        // !(iv >= bound)
        run_case(ArithmeticOp::BINARY_GREATER_EQUAL, true, false);
        // bound > iv
        run_case(ArithmeticOp::BINARY_GREATER, false, true);
        // !(bound <= iv)
        run_case(ArithmeticOp::BINARY_LESS_EQUAL, false, false);
    };

    "natural_loop_does_not_invent_trip_count_for_reversed_less"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *def = kernel->definition();
        auto *entry = kernel->create_body_block();
        auto *header = def->create_basic_block();
        auto *latch = def->create_basic_block();
        auto *exit = def->create_basic_block();
        auto *zero = module.create_constant_zero(Type::of<uint>());
        auto *one = module.create_constant_one(Type::of<uint>());
        uint32_t bound_value = 10u;
        auto *bound = module.create_constant(Type::of<uint>(), &bound_value);
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.br(header);
        b.set_insertion_point(header);
        auto *iv = b.phi(Type::of<uint>());
        auto *condition = b.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS, {bound, iv});
        b.cond_br(condition, latch, exit);
        b.set_insertion_point(latch);
        auto *next = b.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD, {iv, one});
        b.br(header);
        b.set_insertion_point(exit);
        b.return_void();
        iv->add_incoming(zero, entry);
        iv->add_incoming(next, latch);

        auto loops = discover_natural_loops(def, compute_dom_tree(kernel));
        expect(loops.size() == 1u);
        auto bounds = analyze_loop_bounds(loops.front());
        expect(bounds.is_valid());
        expect(!bounds.normalized_strict_less);
        expect(!bounds.trip_count_is_constant);
    };

    "natural_loop_trip_count_requires_no_induction_wraparound"_test = [] {
        auto run_case = []<typename T>(
                            T start_value, T bound_value, T stride_value,
                            bool expected_constant,
                            uint64_t expected_trip_count = 0u) noexcept {
            Module module;
            auto *kernel = module.create_kernel();
            auto *def = kernel->definition();
            auto *entry = kernel->create_body_block();
            auto *header = def->create_basic_block();
            auto *latch = def->create_basic_block();
            auto *exit = def->create_basic_block();
            auto *start = module.create_constant(
                Type::of<T>(), &start_value);
            auto *bound = module.create_constant(
                Type::of<T>(), &bound_value);
            auto *stride = module.create_constant(
                Type::of<T>(), &stride_value);
            XIRBuilder b;
            b.set_insertion_point(entry);
            b.br(header);
            b.set_insertion_point(header);
            auto *iv = b.phi(Type::of<T>());
            auto *condition = b.call(
                Type::of<bool>(), ArithmeticOp::BINARY_LESS,
                {iv, bound});
            b.cond_br(condition, latch, exit);
            b.set_insertion_point(latch);
            auto *next = b.call(
                Type::of<T>(), ArithmeticOp::BINARY_ADD,
                {iv, stride});
            b.br(header);
            b.set_insertion_point(exit);
            b.return_void();
            iv->add_incoming(start, entry);
            iv->add_incoming(next, latch);

            auto loops = discover_natural_loops(
                def, compute_dom_tree(kernel));
            expect(loops.size() == 1u);
            auto bounds = analyze_loop_bounds(loops.front());
            expect(bounds.is_valid());
            expect(bounds.trip_count_is_constant == expected_constant);
            if (expected_constant) {
                expect(bounds.constant_trip_count ==
                       expected_trip_count);
            }
        };

        // The mathematical ceil-division is two, but the second recurrence
        // wraps 8-bit IV 200 + 200 to 144, so the loop does not terminate.
        run_case(uint8_t{0u}, uint8_t{255u}, uint8_t{200u}, false);
        // Exact arrival at the bound is safe and must remain analyzable.
        run_case(uint8_t{0u}, uint8_t{254u}, uint8_t{127u}, true, 2u);
        // A less obvious full-width sibling: even numbers can never reach
        // UINT32_MAX before wrapping.
        run_case(uint32_t{0u}, std::numeric_limits<uint32_t>::max(),
                 uint32_t{2u}, false);
        // Signed wrap is equally invalid for the closed-form count.
        run_case(std::numeric_limits<int8_t>::min(),
                 std::numeric_limits<int8_t>::max(),
                 int8_t{127}, false);
        run_case(std::numeric_limits<int8_t>::min(),
                 int8_t{126}, int8_t{127}, true, 2u);
    };

    "natural_loop_rejects_multiple_exit_edges_to_one_block"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *def = kernel->definition();
        auto *entry = kernel->create_body_block();
        auto *header = def->create_basic_block();
        auto *latch = def->create_basic_block();
        auto *exit = def->create_basic_block();
        auto *zero = module.create_constant_zero(Type::of<uint>());
        auto *one = module.create_constant_one(Type::of<uint>());
        auto *predicate = module.create_constant_one(Type::of<bool>());
        uint32_t bound_value = 4u;
        auto *bound = module.create_constant(Type::of<uint>(), &bound_value);
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.br(header);
        b.set_insertion_point(header);
        auto *iv = b.phi(Type::of<uint>());
        auto *condition = b.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, bound});
        b.cond_br(condition, latch, exit);
        b.set_insertion_point(latch);
        auto *next = b.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD, {iv, one});
        b.cond_br(predicate, header, exit);
        b.set_insertion_point(exit);
        b.return_void();
        iv->add_incoming(zero, entry);
        iv->add_incoming(next, latch);

        auto loops = discover_natural_loops(def, compute_dom_tree(kernel));
        expect(loops.size() == 1u);
        expect(loops.front().exit_blocks.size() == 1u);
        expect(loops.front().exit_edges.size() == 2u);
        expect(!analyze_loop_bounds(loops.front()).is_valid());
    };

    "natural_loop_discovers_multiple_latches_but_rejects_canonical_bounds"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *def = kernel->definition();
        auto *entry = kernel->create_body_block();
        auto *header = def->create_basic_block();
        auto *body = def->create_basic_block();
        auto *left_latch = def->create_basic_block();
        auto *right_latch = def->create_basic_block();
        auto *exit = def->create_basic_block();
        auto *zero = module.create_constant_zero(Type::of<uint>());
        auto *one = module.create_constant_one(Type::of<uint>());
        auto *predicate = kernel->create_value_argument(Type::of<bool>());
        uint32_t bound_value = 4u;
        auto *bound = module.create_constant(Type::of<uint>(), &bound_value);

        XIRBuilder b;
        b.set_insertion_point(entry);
        b.br(header);
        b.set_insertion_point(header);
        auto *iv = b.phi(Type::of<uint>());
        auto *condition = b.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, bound});
        b.cond_br(condition, body, exit);
        b.set_insertion_point(body);
        b.cond_br(predicate, left_latch, right_latch);
        b.set_insertion_point(left_latch);
        auto *left_next = b.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD, {iv, one});
        b.br(header);
        b.set_insertion_point(right_latch);
        auto *right_next = b.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD, {iv, one});
        b.br(header);
        b.set_insertion_point(exit);
        b.return_void();
        iv->add_incoming(zero, entry);
        iv->add_incoming(left_next, left_latch);
        iv->add_incoming(right_next, right_latch);

        expect(xir_verify_module(&module).succeeded());
        auto loops = discover_natural_loops(
            def, compute_dom_tree(kernel));
        expect(loops.size() == 1u);
        expect(loops.front().latches.size() == 2u);
        expect(!analyze_loop_bounds(loops.front()).is_valid());
        expect(xir_verify_module(&module).succeeded());
    };

    "natural_loop_orders_inner_loops_first"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *def = kernel->definition();
        auto *entry = kernel->create_body_block();
        auto *outer_header = def->create_basic_block();
        auto *inner_header = def->create_basic_block();
        auto *inner_latch = def->create_basic_block();
        auto *outer_latch = def->create_basic_block();
        auto *exit = def->create_basic_block();
        auto *cond_value = module.create_constant_one(Type::of<bool>());

        XIRBuilder b;
        b.set_insertion_point(entry);
        b.br(outer_header);
        b.set_insertion_point(outer_header);
        b.cond_br(cond_value, inner_header, exit);
        b.set_insertion_point(inner_header);
        b.cond_br(cond_value, inner_latch, outer_latch);
        b.set_insertion_point(inner_latch);
        b.br(inner_header);
        b.set_insertion_point(outer_latch);
        b.br(outer_header);
        b.set_insertion_point(exit);
        b.return_void();

        auto dom_tree = compute_dom_tree(kernel);
        auto loops = discover_natural_loops(def, dom_tree);
        expect(loops.size() == 2u);
        // inner loop first: its body is a subset of the outer loop body
        expect(loops[0].header == inner_header);
        expect(loops[1].header == outer_header);
        expect(loops[1].contains(inner_header));
        expect(loops[1].contains(inner_latch));
        expect(!loops[0].contains(outer_latch));
        // outer_header also branches to the exit, so the inner loop has no
        // single-successor preheader until the edge is split
        expect(loops[0].preheader == nullptr);
        expect(loops[1].preheader == entry);
    };

    "natural_loop_reports_no_loop_in_straight_line_cfg"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto dom_tree = compute_dom_tree(kernel);
        auto loops = discover_natural_loops(kernel->definition(), dom_tree);
        expect(loops.empty());
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    reg_call_graph();
    reg_uniformity_analysis();
    reg_natural_loop_discovery();
    return 0;
}
