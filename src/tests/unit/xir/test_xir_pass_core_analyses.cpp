// Test for XIR call-graph and uniformity analyses.
// This test covers:
// - call roots and caller edges
// - conservative kernel/callable uniformity propagation

#include "ut/ut.hpp"

#include <luisa/ast/type_registry.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/call_graph.h>
#include <luisa/xir/passes/uniformity_analysis.h>

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

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    reg_call_graph();
    reg_uniformity_analysis();
    return 0;
}
