#include "ut/ut.hpp"

#include <luisa/ast/type_registry.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/module.h>
#include <luisa/xir/op.h>
#include <luisa/xir/special_register.h>

#include "warp_uniformity.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace luisa::compute::simd::schedule;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void register_basic_uniformity_tests() {
    "simd_warp_uniformity_classifies_kernel_values"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *argument = kernel->create_value_argument(Type::of<uint>());
        auto *resource = kernel->create_resource_argument(
            Type::buffer(Type::of<uint>()));
        auto *body = kernel->create_body_block();
        auto *one = module.create_constant_one(Type::of<uint>());
        auto *lane = module.create_special_register(
            DerivedSpecialRegisterTag::WARP_LANE_ID);
        auto *warp_size = module.create_special_register(
            DerivedSpecialRegisterTag::WARP_SIZE);
        auto *block_id = module.create_special_register(
            DerivedSpecialRegisterTag::BLOCK_ID);
        auto *dispatch_id = module.create_special_register(
            DerivedSpecialRegisterTag::DISPATCH_ID);

        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *uniform_sum = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {argument, one});
        auto *varying_sum = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {lane, one});
        builder.return_void();

        WarpUniformityAnalysis analysis;
        analysis.analyze(kernel);
        expect(analysis.is_uniform(argument));
        expect(analysis.is_warp_uniform(argument));
        expect(analysis.is_uniform(resource));
        expect(analysis.is_warp_uniform(resource));
        expect(analysis.is_uniform(one));
        expect(analysis.is_warp_uniform(one));
        expect(analysis.is_uniform(uniform_sum));
        expect(analysis.is_warp_uniform(uniform_sum));
        expect(!analysis.is_uniform(lane));
        expect(!analysis.is_uniform(varying_sum));
        expect(analysis.is_uniform(warp_size));
        expect(analysis.is_uniform(block_id));
        expect(!analysis.is_uniform(dispatch_id));
    };

    "simd_warp_uniformity_preserves_uniform_control_phi"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *entry = kernel->create_body_block();
        auto *true_block = kernel->create_basic_block();
        auto *false_block = kernel->create_basic_block();
        auto *merge = kernel->create_basic_block();
        auto *one = module.create_constant_one(Type::of<uint>());
        uint32_t two_value = 2u;
        auto *two = module.create_constant(Type::of<uint>(), &two_value);

        XIRBuilder builder;
        builder.set_insertion_point(entry);
        builder.cond_br(condition, true_block, false_block);
        builder.set_insertion_point(true_block);
        builder.br(merge);
        builder.set_insertion_point(false_block);
        builder.br(merge);
        builder.set_insertion_point(merge);
        auto *same = builder.phi(
            Type::of<uint>(), {{one, true_block}, {one, false_block}});
        auto *different = builder.phi(
            Type::of<uint>(), {{one, true_block}, {two, false_block}});
        builder.return_void();

        WarpUniformityAnalysis analysis;
        analysis.analyze(kernel);
        expect(analysis.is_uniform(same));
        expect(analysis.is_warp_uniform(same));
        expect(analysis.is_uniform(different));
        expect(analysis.is_warp_uniform(different));
    };

    "simd_warp_uniformity_rejects_varying_control_phi"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *entry = kernel->create_body_block();
        auto *true_block = kernel->create_basic_block();
        auto *false_block = kernel->create_basic_block();
        auto *merge = kernel->create_basic_block();
        auto *lane = module.create_special_register(
            DerivedSpecialRegisterTag::WARP_LANE_ID);
        auto *zero = module.create_constant_zero(Type::of<uint>());
        auto *one = module.create_constant_one(Type::of<uint>());
        uint32_t two_value = 2u;
        auto *two = module.create_constant(Type::of<uint>(), &two_value);

        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *condition = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
            {lane, zero});
        builder.cond_br(condition, true_block, false_block);
        builder.set_insertion_point(true_block);
        builder.br(merge);
        builder.set_insertion_point(false_block);
        builder.br(merge);
        builder.set_insertion_point(merge);
        auto *same = builder.phi(
            Type::of<uint>(), {{one, true_block}, {one, false_block}});
        auto *different = builder.phi(
            Type::of<uint>(), {{one, true_block}, {two, false_block}});
        builder.return_void();

        WarpUniformityAnalysis analysis;
        analysis.analyze(kernel);
        expect(analysis.is_warp_uniform(same));
        expect(!analysis.is_uniform(different));
    };

    "simd_warp_uniformity_preserves_cohort_control_phi"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *entry = kernel->create_body_block();
        auto *true_block = kernel->create_basic_block();
        auto *false_block = kernel->create_basic_block();
        auto *merge = kernel->create_basic_block();
        auto *lane = module.create_special_register(
            DerivedSpecialRegisterTag::WARP_LANE_ID);
        auto *zero = module.create_constant_zero(Type::of<uint>());
        auto *one = module.create_constant_one(Type::of<uint>());
        uint32_t two_value = 2u;
        auto *two = module.create_constant(Type::of<uint>(), &two_value);

        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *active_sum = builder.call(
            Type::of<uint>(), ThreadGroupOp::WARP_ACTIVE_SUM, {lane});
        auto *condition = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_NOT_EQUAL,
            {active_sum, zero});
        builder.cond_br(condition, true_block, false_block);
        builder.set_insertion_point(true_block);
        builder.br(merge);
        builder.set_insertion_point(false_block);
        builder.br(merge);
        builder.set_insertion_point(merge);
        auto *same = builder.phi(
            Type::of<uint>(), {{one, true_block}, {one, false_block}});
        auto *different = builder.phi(
            Type::of<uint>(), {{one, true_block}, {two, false_block}});
        builder.return_void();

        WarpUniformityAnalysis analysis;
        analysis.analyze(kernel);
        expect(analysis.is_cohort_uniform(active_sum));
        expect(analysis.is_cohort_uniform(condition));
        expect(analysis.is_warp_uniform(same));
        expect(analysis.is_cohort_uniform(different));
        expect(!analysis.is_warp_uniform(different));
    };

    "simd_warp_uniformity_keeps_recurrent_uniform_phi_cohort_local"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *keep_iterating =
            kernel->create_value_argument(Type::of<bool>());
        auto *entry = kernel->create_body_block();
        auto *header = kernel->create_basic_block();
        auto *body = kernel->create_basic_block();
        auto *exit = kernel->create_basic_block();
        auto *zero = module.create_constant_zero(Type::of<uint>());
        auto *one = module.create_constant_one(Type::of<uint>());

        XIRBuilder builder;
        builder.set_insertion_point(entry);
        builder.br(header);
        builder.set_insertion_point(header);
        auto *iteration = builder.phi(Type::of<uint>());
        builder.cond_br(keep_iterating, body, exit);
        builder.set_insertion_point(body);
        auto *next = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {iteration, one});
        builder.br(header);
        iteration->add_incoming(zero, entry);
        iteration->add_incoming(next, body);
        builder.set_insertion_point(exit);
        builder.return_void();

        WarpUniformityAnalysis analysis;
        analysis.analyze(kernel);
        expect(analysis.is_cohort_uniform(iteration));
        expect(analysis.is_cohort_uniform(next));
        expect(!analysis.is_warp_uniform(iteration));
    };
}

void register_collective_uniformity_tests() {
    "simd_warp_uniformity_tracks_loop_epoch_escape_and_derivatives"_test = [] {
        for (auto variant : {0u, 1u, 2u, 3u}) {
            Module module;
            auto kernel = module.create_kernel();
            auto entry = kernel->create_body_block();
            auto header = kernel->create_basic_block();
            auto latch = kernel->create_basic_block();
            auto exit = kernel->create_basic_block();
            auto type = Type::of<uint32_t>();
            auto zero = module.create_constant_zero(type);
            auto one = module.create_constant_one(type);
            auto lane = module.create_warp_lane_id();
            XIRBuilder builder;
            builder.set_insertion_point(entry);
            auto storage = builder.alloca_local(type);
            builder.br(header);
            builder.set_insertion_point(header);
            auto iteration = builder.phi(type, {{zero, entry}});
            auto population = builder.call(type, ThreadGroupOp::WARP_ACTIVE_SUM, {one});
            auto derived = variant == 1u || variant == 3u ? builder.call(type, ArithmeticOp::BINARY_ADD, {population, one}) : nullptr;
            if (variant == 3u) { builder.store(storage, derived); }
            builder.cond_br(builder.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iteration, lane}), latch, exit);
            builder.set_insertion_point(latch);
            iteration->add_incoming(builder.call(type, ArithmeticOp::BINARY_ADD, {iteration, one}), latch);
            builder.br(header);
            builder.set_insertion_point(exit);
            auto single_incoming = variant == 2u ? builder.phi(type, {{population, header}}) : nullptr;
            if (variant == 0u) { builder.store(storage, population); }
            if (variant == 1u) { builder.store(storage, derived); }
            if (variant == 2u) { builder.store(storage, single_incoming); }
            builder.return_void();

            WarpUniformityAnalysis analysis;
            analysis.analyze(kernel);
            // One syntactic incoming edge does not imply one dynamic epoch:
            // each lane exits at a different header visit with a different sum.
            if (variant == 0u) { expect(!analysis.is_uniform(population)); }
            if (variant == 1u) {
                expect(!analysis.is_uniform(derived));
                expect(analysis.is_cohort_uniform(population));
            }
            if (variant == 2u) { expect(!analysis.is_uniform(single_incoming)); }
            if (variant == 3u) {
                // No escaped use: keep the useful one-scalar representation
                // for the collective and arithmetic within the same epoch.
                expect(analysis.is_cohort_uniform(population));
                expect(analysis.is_cohort_uniform(derived));
            }
        }
    };
    "simd_warp_uniformity_rejects_inner_epoch_values_in_the_outer_loop"_test = [] {
        Module module;
        auto kernel = module.create_kernel();
        auto entry = kernel->create_body_block();
        auto outer_header = kernel->create_basic_block();
        auto inner_entry = kernel->create_basic_block();
        auto inner_header = kernel->create_basic_block();
        auto inner_latch = kernel->create_basic_block();
        auto outer_latch = kernel->create_basic_block();
        auto exit = kernel->create_basic_block();
        auto type = Type::of<uint32_t>();
        auto zero = module.create_constant_zero(type);
        auto one = module.create_constant_one(type);
        auto lane = module.create_warp_lane_id();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto storage = builder.alloca_local(type);
        builder.br(outer_header);
        builder.set_insertion_point(outer_header);
        auto outer_iteration = builder.phi(type, {{zero, entry}});
        builder.cond_br(builder.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS_EQUAL, {outer_iteration, one}), inner_entry, exit);
        builder.set_insertion_point(inner_entry);
        builder.br(inner_header);
        builder.set_insertion_point(inner_header);
        auto inner_iteration = builder.phi(type, {{zero, inner_entry}});
        auto population = builder.call(type, ThreadGroupOp::WARP_ACTIVE_SUM, {one});
        builder.cond_br(builder.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {inner_iteration, lane}), inner_latch, outer_latch);
        builder.set_insertion_point(inner_latch);
        inner_iteration->add_incoming(builder.call(type, ArithmeticOp::BINARY_ADD, {inner_iteration, one}), inner_latch);
        builder.br(inner_header);
        builder.set_insertion_point(outer_latch);
        auto escaped_derived = builder.call(type, ArithmeticOp::BINARY_ADD, {population, one});
        builder.store(storage, escaped_derived);
        outer_iteration->add_incoming(builder.call(type, ArithmeticOp::BINARY_ADD, {outer_iteration, one}), outer_latch);
        builder.br(outer_header);
        builder.set_insertion_point(exit);
        builder.return_void();

        WarpUniformityAnalysis analysis;
        analysis.analyze(kernel);
        // Membership in a common outer loop does not make inner-loop exit
        // snapshots equal, and the derived expression must remain lane-wise.
        expect(!analysis.is_uniform(population));
        expect(!analysis.is_uniform(escaped_derived));
    };
    "simd_warp_uniformity_understands_collective_results"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *lane = module.create_special_register(
            DerivedSpecialRegisterTag::WARP_LANE_ID);
        auto *zero = module.create_constant_zero(Type::of<uint>());

        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *active_sum = builder.call(
            Type::of<uint>(), ThreadGroupOp::WARP_ACTIVE_SUM, {lane});
        auto *prefix_sum = builder.call(
            Type::of<uint>(), ThreadGroupOp::WARP_PREFIX_SUM, {lane});
        auto *read_uniform_lane = builder.call(
            Type::of<uint>(), ThreadGroupOp::WARP_READ_LANE,
            {lane, zero});
        auto *read_varying_lane = builder.call(
            Type::of<uint>(), ThreadGroupOp::WARP_READ_LANE,
            {lane, lane});
        auto *is_first = builder.call(
            Type::of<bool>(),
            ThreadGroupOp::WARP_IS_FIRST_ACTIVE_LANE, {});
        auto *first_lane = builder.call(
            Type::of<uint>(), ThreadGroupOp::WARP_FIRST_ACTIVE_LANE, {});
        builder.return_void();

        WarpUniformityAnalysis analysis;
        analysis.analyze(kernel);
        expect(analysis.is_uniform(active_sum));
        expect(analysis.is_cohort_uniform(active_sum));
        expect(!analysis.is_uniform(prefix_sum));
        expect(analysis.is_uniform(read_uniform_lane));
        expect(analysis.is_cohort_uniform(read_uniform_lane));
        expect(!analysis.is_uniform(read_varying_lane));
        expect(!analysis.is_uniform(is_first));
        expect(analysis.is_uniform(first_lane));
        expect(analysis.is_cohort_uniform(first_lane));
    };
}

void register_callable_tests() {
    "simd_warp_uniformity_keeps_uninlined_callable_arguments_varying"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *kernel_argument =
            kernel->create_value_argument(Type::of<uint>());
        auto *kernel_body = kernel->create_body_block();
        auto *callable = module.create_callable(Type::of<uint>());
        auto *callable_argument =
            callable->create_value_argument(Type::of<uint>());
        auto *callable_body = callable->create_body_block();
        auto *one = module.create_constant_one(Type::of<uint>());

        XIRBuilder builder;
        builder.set_insertion_point(kernel_body);
        auto *kernel_sum = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {kernel_argument, one});
        builder.return_void();
        builder.set_insertion_point(callable_body);
        auto *callable_sum = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {callable_argument, one});
        builder.return_(callable_sum);

        WarpUniformityAnalysis analysis;
        analysis.analyze(kernel);
        expect(analysis.is_uniform(kernel_sum));
        analysis.analyze(callable);
        expect(!analysis.is_uniform(kernel_sum));
        expect(!analysis.is_uniform(callable_argument));
        expect(!analysis.is_uniform(callable_sum));
        expect(analysis.is_uniform(one));
    };
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    register_basic_uniformity_tests();
    register_collective_uniformity_tests();
    register_callable_tests();
    return 0;
}
