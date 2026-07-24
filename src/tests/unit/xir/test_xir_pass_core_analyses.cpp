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
        expect(loop.back_edges.size() == 1u);

        auto bounds = analyze_loop_bounds(loop);
        expect(bounds.is_valid());
        expect(bounds.induction_phi == iv);
        expect(bounds.start_value == zero);
        expect(bounds.bound_value == bound);
        expect(bounds.stride_is_constant && bounds.stride == 1);
        expect(bounds.trip_count_is_constant);
        expect(bounds.constant_trip_count == 10u);
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
