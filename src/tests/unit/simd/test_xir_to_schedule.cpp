#include "ut/ut.hpp"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <sstream>
#include <string>
#include <vector>

#include <luisa/ast/type_registry.h>
#include <luisa/dsl/rtx/ray_query.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/dce.h>

#include "block_barrier.h"
#include "xir_to_schedule.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace luisa::compute::simd::schedule;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] std::string diagnostics_text(
    const XIRToScheduleResult &result) {
    std::ostringstream out;
    for (auto &&diagnostic : result.diagnostics) {
        out << to_string(diagnostic.code) << ": "
            << diagnostic.message << '\n';
    }
    return out.str();
}

[[nodiscard]] const simd::schedule::BasicBlock *find_block(
    const simd::schedule::Function &function,
    std::string_view name) noexcept {
    for (auto &&block : function.blocks()) {
        if (block.name == name) { return &block; }
    }
    return nullptr;
}

[[nodiscard]] const simd::schedule::Value *find_value(
    const simd::schedule::Function &function,
    std::string_view name) noexcept {
    for (auto &&value : function.values()) {
        if (value.name == name) { return &value; }
    }
    return nullptr;
}

void register_diamond_tests() {
    "simd_xir_lowering_projects_divergent_phi_and_collective"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        kernel->set_name("diamond_collective");
        auto *entry = kernel->create_body_block();
        auto *left = kernel->create_basic_block();
        auto *right = kernel->create_basic_block();
        auto *merge = kernel->create_basic_block();
        entry->set_name("entry");
        left->set_name("left");
        right->set_name("right");
        merge->set_name("merge");

        auto *lane = module.create_warp_lane_id();
        auto *one = module.create_constant_one(Type::of<uint>());
        uint32_t two_data = 2u;
        auto *two = module.create_constant(Type::of<uint>(), &two_data);
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *condition = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS, {lane, two});
        condition->set_name("condition");
        builder.cond_br(condition, left, right);
        builder.set_insertion_point(left);
        auto *left_value = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD, {lane, one});
        left_value->set_name("left_value");
        builder.br(merge);
        builder.set_insertion_point(right);
        auto *right_value = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_SUB, {lane, one});
        right_value->set_name("right_value");
        builder.br(merge);
        builder.set_insertion_point(merge);
        auto *selected = builder.phi(
            Type::of<uint>(),
            {{left_value, left}, {right_value, right}});
        selected->set_name("selected");
        auto *sum = builder.call(
            Type::of<uint>(), ThreadGroupOp::WARP_ACTIVE_SUM, {selected});
        sum->set_name("sum");
        builder.return_void();

        auto result = lower_xir_to_schedule(
            kernel, {.logical_warp_width = 8u});
        expect(result.succeeded()) << diagnostics_text(result);
        if (!result.succeeded()) { return; }
        auto &&function = *result.function;
        expect(function.logical_warp_width() == 8u);
        expect(function.convergence_points().size() == 1u);
        expect(verify(function).succeeded());

        auto *entry_block = find_block(function, "entry");
        auto *left_block = find_block(function, "left");
        auto *right_block = find_block(function, "right");
        auto *merge_block = find_block(function, "merge");
        expect(entry_block != nullptr && left_block != nullptr &&
               right_block != nullptr && merge_block != nullptr);
        if (entry_block == nullptr || left_block == nullptr ||
            right_block == nullptr || merge_block == nullptr) {
            return;
        }
        auto *split = std::get_if<SplitTerminator>(
            &entry_block->terminator);
        expect(split != nullptr);
        expect(split != nullptr && split->convergence.has_value());
        expect(entry_block->strategy == RegionStrategy::cohort);

        for (auto *path : {left_block, right_block}) {
            auto *branch = std::get_if<BranchTerminator>(
                &path->terminator);
            expect(branch != nullptr);
            if (branch == nullptr) { continue; }
            expect(branch->edge.target == merge_block->id);
            expect(branch->edge.joins.size() == 1u);
            expect(branch->edge.assignments.size() == 1u);
        }

        auto *state = find_value(function, "selected");
        auto *active_mask = find_value(function, "active_mask");
        auto *sum_value = find_value(function, "sum");
        expect(state != nullptr &&
               state->origin == ValueOrigin::state_slot);
        expect(active_mask != nullptr &&
               active_mask->value_class == ValueClass::mask);
        expect(sum_value != nullptr &&
               sum_value->value_class == ValueClass::cohort_uniform);
        auto *lane_value = find_value(function, "warp_lane_id");
        expect(lane_value != nullptr);
        if (lane_value != nullptr) {
            auto *metadata = std::get_if<SpecialRegisterValueMetadata>(
                &lane_value->metadata);
            expect(metadata != nullptr);
            expect(metadata != nullptr &&
                   metadata->tag == static_cast<uint32_t>(
                                        DerivedSpecialRegisterTag::WARP_LANE_ID));
        }
        expect(active_mask != nullptr &&
               std::holds_alternative<SchedulerBuiltinValueMetadata>(
                   active_mask->metadata));
        expect(merge_block->instructions.size() == 1u);
        if (!merge_block->instructions.empty()) {
            auto &&collective = merge_block->instructions.front();
            expect(collective.opcode == Opcode::warp_collective);
            expect(collective.collective_id == 0u);
            expect(collective.participant_mask.has_value());
            expect(active_mask != nullptr &&
                   collective.participant_mask == active_mask->id);
        }
    };

    "simd_xir_lowering_materializes_undef_phi_incoming_as_zero"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *entry = kernel->create_body_block();
        auto *defined = kernel->create_basic_block();
        auto *undefined = kernel->create_basic_block();
        auto *merge = kernel->create_basic_block();
        entry->set_name("entry");
        defined->set_name("defined");
        undefined->set_name("undefined");
        merge->set_name("merge");

        auto *one = module.create_constant_one(Type::of<uint>());
        auto *undef = module.create_undefined(Type::of<uint>());
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        builder.cond_br(condition, defined, undefined);
        builder.set_insertion_point(defined);
        builder.br(merge);
        builder.set_insertion_point(undefined);
        builder.br(merge);
        builder.set_insertion_point(merge);
        auto *selected = builder.phi(
            Type::of<uint>(),
            {{one, defined}, {undef, undefined}});
        selected->set_name("selected_with_undef");
        static_cast<void>(builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {selected, one}));
        builder.return_void();

        auto result = lower_xir_to_schedule(
            kernel, {.logical_warp_width = 8u});
        expect(result.succeeded()) << diagnostics_text(result);
        if (!result.succeeded()) { return; }
        expect(verify(*result.function).succeeded());

        auto *undefined_block = find_block(
            *result.function, "undefined");
        expect(undefined_block != nullptr);
        if (undefined_block == nullptr) { return; }
        auto *branch = std::get_if<BranchTerminator>(
            &undefined_block->terminator);
        expect(branch != nullptr);
        if (branch == nullptr) { return; }
        expect(branch->edge.assignments.size() == 1u);
        if (branch->edge.assignments.size() != 1u) { return; }
        auto *source = result.function->value(
            branch->edge.assignments.front().source);
        expect(source != nullptr);
        if (source == nullptr) { return; }
        expect(source->origin == ValueOrigin::constant);
        auto *constant = std::get_if<ConstantValueMetadata>(
            &source->metadata);
        expect(constant != nullptr);
        if (constant != nullptr) {
            expect(constant->bytes.size() == sizeof(uint32_t));
            expect(std::all_of(
                constant->bytes.begin(), constant->bytes.end(),
                [](auto byte) noexcept {
                    return byte == std::byte{0};
                }));
        }
    };

    "simd_xir_lowering_keeps_uniform_branch_scalar"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *entry = kernel->create_body_block();
        auto *left = kernel->create_basic_block();
        auto *right = kernel->create_basic_block();
        entry->set_name("entry");
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        builder.cond_br(condition, left, right);
        builder.set_insertion_point(left);
        builder.return_void();
        builder.set_insertion_point(right);
        builder.return_void();

        auto result = lower_xir_to_schedule(
            kernel, {.logical_warp_width = 4u});
        expect(result.succeeded()) << diagnostics_text(result);
        if (!result.succeeded()) { return; }
        auto *schedule_entry = find_block(*result.function, "entry");
        expect(schedule_entry != nullptr);
        expect(schedule_entry != nullptr &&
               schedule_entry->strategy ==
                   RegionStrategy::uniform_control);
        expect(result.function->convergence_points().empty());
        auto *parameter = find_value(*result.function, "arg0");
        expect(parameter != nullptr);
        if (parameter != nullptr) {
            auto *metadata = std::get_if<ParameterValueMetadata>(
                &parameter->metadata);
            expect(metadata != nullptr);
            expect(metadata != nullptr && metadata->index == 0u);
            expect(parameter->value_class == ValueClass::warp_uniform);
        }
    };

    "simd_xir_lowering_orders_nested_shared_convergence"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *entry = kernel->create_body_block();
        auto *nested = kernel->create_basic_block();
        auto *outer_right = kernel->create_basic_block();
        auto *inner_left = kernel->create_basic_block();
        auto *inner_right = kernel->create_basic_block();
        auto *merge = kernel->create_basic_block();
        entry->set_name("entry");
        nested->set_name("nested");
        outer_right->set_name("outer_right");
        inner_left->set_name("inner_left");
        inner_right->set_name("inner_right");
        merge->set_name("merge");

        auto *lane = module.create_warp_lane_id();
        uint32_t two_data = 2u;
        uint32_t four_data = 4u;
        auto *two = module.create_constant(Type::of<uint>(), &two_data);
        auto *four = module.create_constant(Type::of<uint>(), &four_data);
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *outer_condition = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS, {lane, four});
        builder.cond_br(outer_condition, nested, outer_right);
        builder.set_insertion_point(nested);
        auto *inner_condition = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS, {lane, two});
        builder.cond_br(inner_condition, inner_left, inner_right);
        builder.set_insertion_point(outer_right);
        builder.br(merge);
        builder.set_insertion_point(inner_left);
        builder.br(merge);
        builder.set_insertion_point(inner_right);
        builder.br(merge);
        builder.set_insertion_point(merge);
        builder.return_void();

        auto result = lower_xir_to_schedule(
            kernel, {.logical_warp_width = 8u});
        expect(result.succeeded()) << diagnostics_text(result);
        if (!result.succeeded()) { return; }
        auto &&function = *result.function;
        expect(function.convergence_points().size() == 2u);
        auto *schedule_entry = find_block(function, "entry");
        auto *schedule_nested = find_block(function, "nested");
        auto *schedule_outer_right = find_block(function, "outer_right");
        auto *schedule_inner_left = find_block(function, "inner_left");
        auto *schedule_inner_right = find_block(function, "inner_right");
        expect(schedule_entry != nullptr && schedule_nested != nullptr &&
               schedule_outer_right != nullptr &&
               schedule_inner_left != nullptr &&
               schedule_inner_right != nullptr);
        if (schedule_entry == nullptr || schedule_nested == nullptr ||
            schedule_outer_right == nullptr ||
            schedule_inner_left == nullptr ||
            schedule_inner_right == nullptr) {
            return;
        }

        auto *outer_split = std::get_if<SplitTerminator>(
            &schedule_entry->terminator);
        auto *inner_split = std::get_if<SplitTerminator>(
            &schedule_nested->terminator);
        expect(outer_split != nullptr && inner_split != nullptr);
        if (outer_split == nullptr || inner_split == nullptr ||
            !outer_split->convergence || !inner_split->convergence) {
            return;
        }
        auto outer = *outer_split->convergence;
        auto inner = *inner_split->convergence;
        expect(function.convergence(inner)->parent == outer);

        auto expect_joins = [&](const simd::schedule::BasicBlock *block,
                                std::initializer_list<ConvergenceId> expected) {
            auto *branch = std::get_if<BranchTerminator>(
                &block->terminator);
            expect(branch != nullptr);
            if (branch != nullptr) {
                expect(std::equal(
                    branch->edge.joins.begin(), branch->edge.joins.end(),
                    expected.begin(), expected.end()));
            }
        };
        expect_joins(schedule_outer_right, {outer});
        expect_joins(schedule_inner_left, {inner, outer});
        expect_joins(schedule_inner_right, {inner, outer});
        expect(verify(function).succeeded());
    };
}

void register_loop_tests() {
    "simd_xir_lowering_marks_loop_epoch_and_phi_edges"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *entry = kernel->create_body_block();
        auto *header = kernel->create_basic_block();
        auto *body = kernel->create_basic_block();
        auto *exit = kernel->create_basic_block();
        entry->set_name("entry");
        header->set_name("header");
        body->set_name("body");
        exit->set_name("exit");
        auto *lane = module.create_warp_lane_id();
        auto *zero = module.create_constant_zero(Type::of<uint>());
        auto *one = module.create_constant_one(Type::of<uint>());

        XIRBuilder builder;
        builder.set_insertion_point(entry);
        builder.br(header);
        builder.set_insertion_point(header);
        auto *index = builder.phi(Type::of<uint>());
        index->set_name("index");
        auto *bound = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD, {lane, one});
        auto *condition = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS,
            {index, bound});
        builder.cond_br(condition, body, exit);
        builder.set_insertion_point(body);
        auto *next = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD, {index, one});
        next->set_name("next");
        builder.br(header);
        builder.set_insertion_point(exit);
        builder.return_void();
        index->add_incoming(zero, entry);
        index->add_incoming(next, body);

        auto result = lower_xir_to_schedule(
            kernel, {.logical_warp_width = 4u});
        expect(result.succeeded()) << diagnostics_text(result);
        if (!result.succeeded()) { return; }
        auto &&function = *result.function;
        expect(function.loops().size() == 1u);
        auto *schedule_entry = find_block(function, "entry");
        auto *schedule_body = find_block(function, "body");
        expect(schedule_entry != nullptr && schedule_body != nullptr);
        if (schedule_entry == nullptr || schedule_body == nullptr) {
            return;
        }
        auto *entry_branch = std::get_if<BranchTerminator>(
            &schedule_entry->terminator);
        auto *back_branch = std::get_if<BranchTerminator>(
            &schedule_body->terminator);
        expect(entry_branch != nullptr && back_branch != nullptr);
        if (entry_branch == nullptr || back_branch == nullptr) { return; }
        expect(entry_branch->edge.assignments.size() == 1u);
        expect(!entry_branch->edge.loop_back.has_value());
        expect(back_branch->edge.assignments.size() == 1u);
        expect(back_branch->edge.loop_back == LoopId{0u});
        expect(verify(function).succeeded());
    };
}

void register_block_barrier_tests() {
    "simd_xir_canonicalizes_terminal_block_barrier"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *entry = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        builder.call(
            nullptr, ThreadGroupOp::SYNCHRONIZE_BLOCK, {});
        builder.return_void();

        auto canonical = canonicalize_block_barriers(kernel);
        expect(canonical.succeeded());
        expect(canonical.barrier_count == 1u);
        expect(canonical.split_block_count == 1u);
        auto result = lower_xir_to_schedule(
            kernel, {.logical_warp_width = 1u});
        expect(result.succeeded()) << diagnostics_text(result);
        if (!result.succeeded()) { return; }

        auto barrier_count = size_t{0u};
        for (auto &&block : result.function->blocks()) {
            auto *barrier = std::get_if<BlockBarrierTerminator>(
                &block.terminator);
            if (barrier == nullptr) { continue; }
            barrier_count++;
            expect(
                barrier->resume_edge.target.value <
                result.function->blocks().size());
            if (barrier->resume_edge.target.value <
                result.function->blocks().size()) {
                const auto &resume = result.function->blocks()[
                    barrier->resume_edge.target.value];
                expect(std::holds_alternative<ReturnTerminator>(
                    resume.terminator));
            }
        }
        expect(barrier_count == 1u);
        expect(verify(*result.function).succeeded());
    };

    "simd_xir_canonicalizes_and_lowers_block_barrier_phase"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *entry = kernel->create_body_block();
        auto *one = module.create_constant_one(Type::of<uint>());
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        builder.call(
            nullptr, ThreadGroupOp::SYNCHRONIZE_BLOCK, {});
        auto *value = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {one, one});
        value->set_name("after_barrier");
        builder.return_void();

        auto canonical = canonicalize_block_barriers(kernel);
        expect(canonical.succeeded());
        expect(canonical.barrier_count == 1u);
        expect(canonical.split_block_count == 1u);
        // Cross-block instruction moves must rebuild operand use-lists; DCE
        // is the minimal permanent oracle for the previously corrupted list.
        static_cast<void>(dce_pass_run_on_module(&module));
        auto result = lower_xir_to_schedule(
            kernel, {.logical_warp_width = 8u});
        expect(result.succeeded()) << diagnostics_text(result);
        if (!result.succeeded()) { return; }
        auto barrier_count = size_t{0u};
        for (auto &&block : result.function->blocks()) {
            if (auto *barrier = std::get_if<
                    BlockBarrierTerminator>(&block.terminator)) {
                barrier_count++;
                expect(barrier->barrier_id == 0u);
                expect(
                    barrier->resume_edge.target.value <
                    result.function->blocks().size());
            }
        }
        expect(barrier_count == 1u);
        expect(verify(*result.function).succeeded());
    };

    "simd_xir_lowers_repeated_block_barrier_instances"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *entry = kernel->create_body_block();
        auto *header = kernel->create_basic_block();
        auto *body = kernel->create_basic_block();
        auto *exit = kernel->create_basic_block();
        entry->set_name("entry");
        header->set_name("header");
        body->set_name("body");
        exit->set_name("exit");
        auto *zero = module.create_constant_zero(Type::of<uint>());
        auto *one = module.create_constant_one(Type::of<uint>());
        auto trip_count_value = uint32_t{3u};
        auto *trip_count = module.create_constant(
            Type::of<uint>(), &trip_count_value);

        XIRBuilder builder;
        builder.set_insertion_point(entry);
        builder.br(header);
        builder.set_insertion_point(header);
        auto *iteration = builder.phi(Type::of<uint>());
        auto *condition = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS,
            {iteration, trip_count});
        builder.cond_br(condition, body, exit);
        builder.set_insertion_point(body);
        auto *next = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {iteration, one});
        builder.call(
            nullptr, ThreadGroupOp::SYNCHRONIZE_BLOCK, {});
        builder.br(header);
        builder.set_insertion_point(exit);
        builder.return_void();
        iteration->add_incoming(zero, entry);
        iteration->add_incoming(next, body);

        auto canonical = canonicalize_block_barriers(kernel);
        expect(canonical.succeeded());
        expect(canonical.barrier_count == 1u);
        auto result = lower_xir_to_schedule(
            kernel, {.logical_warp_width = 8u});
        expect(result.succeeded()) << diagnostics_text(result);
        if (!result.succeeded()) { return; }
        expect(result.function->loops().size() == 1u);
        auto *schedule_body = find_block(*result.function, "body");
        expect(schedule_body != nullptr);
        if (schedule_body == nullptr) { return; }
        auto *barrier = std::get_if<BlockBarrierTerminator>(
            &schedule_body->terminator);
        expect(barrier != nullptr);
        expect(barrier != nullptr &&
               barrier->resume_edge.loop_back == LoopId{0u});
        expect(verify(*result.function).succeeded());
    };

    "simd_xir_rejects_divergent_static_block_barriers"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *entry = kernel->create_body_block();
        auto *left = kernel->create_basic_block();
        auto *right = kernel->create_basic_block();
        auto *merge = kernel->create_basic_block();
        auto *lane = module.create_warp_lane_id();
        auto *one = module.create_constant_one(Type::of<uint>());
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *condition = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS,
            {lane, one});
        builder.cond_br(condition, left, right);
        builder.set_insertion_point(left);
        builder.call(
            nullptr, ThreadGroupOp::SYNCHRONIZE_BLOCK, {});
        builder.br(merge);
        builder.set_insertion_point(right);
        builder.call(
            nullptr, ThreadGroupOp::SYNCHRONIZE_BLOCK, {});
        builder.br(merge);
        builder.set_insertion_point(merge);
        builder.return_void();

        auto canonical = canonicalize_block_barriers(kernel);
        expect(canonical.succeeded());
        expect(canonical.barrier_count == 2u);
        auto result = lower_xir_to_schedule(
            kernel, {.logical_warp_width = 8u});
        expect(!result.succeeded());
        auto found = false;
        for (auto &&diagnostic : result.diagnostics) {
            found |= diagnostic.code ==
                     XIRToScheduleDiagnosticCode::
                         non_uniform_block_barrier;
        }
        expect(found);
    };
}

void register_reducible_cfg_tests() {
    "simd_xir_lowering_accepts_exhaustive_five_block_forward_cfgs"_test = [] {
        constexpr auto block_count = 5u;
        constexpr auto encoded_graph_count = 15u * 7u * 3u;
        auto accepted_graph_count = 0u;
        for (auto encoded_graph = 0u;
             encoded_graph < encoded_graph_count; encoded_graph++) {
            auto encoding = encoded_graph;
            std::array<uint32_t, block_count - 1u> successor_masks{};
            for (auto source = 0u; source + 1u < block_count; source++) {
                auto option_count =
                    (1u << (block_count - source - 1u)) - 1u;
                successor_masks[source] = encoding % option_count + 1u;
                encoding /= option_count;
            }

            std::array<bool, block_count> reachable{};
            reachable[0u] = true;
            for (auto source = 0u; source + 1u < block_count; source++) {
                if (!reachable[source]) { continue; }
                auto mask = successor_masks[source];
                for (auto bit = 0u;
                     bit < block_count - source - 1u; bit++) {
                    if ((mask & (1u << bit)) != 0u) {
                        reachable[source + bit + 1u] = true;
                    }
                }
            }
            if (!std::all_of(
                    reachable.begin(), reachable.end(),
                    [](bool value) noexcept { return value; })) {
                continue;
            }

            Module module;
            auto *kernel = module.create_kernel();
            std::array<xir::BasicBlock *, block_count> blocks{};
            blocks[0u] = kernel->create_body_block();
            for (auto i = 1u; i < block_count; i++) {
                blocks[i] = kernel->create_basic_block();
            }
            auto *lane = module.create_warp_lane_id();
            XIRBuilder builder;
            for (auto source = 0u; source + 1u < block_count; source++) {
                builder.set_insertion_point(blocks[source]);
                std::array<xir::BasicBlock *, block_count - 1u> targets{};
                auto target_count = 0u;
                auto mask = successor_masks[source];
                for (auto bit = 0u;
                     bit < block_count - source - 1u; bit++) {
                    if ((mask & (1u << bit)) != 0u) {
                        targets[target_count++] =
                            blocks[source + bit + 1u];
                    }
                }
                if (target_count == 1u) {
                    builder.br(targets[0u]);
                } else {
                    auto *branch = builder.indexed_branch(lane);
                    branch->set_default_block(targets[0u]);
                    for (auto i = 1u; i < target_count; i++) {
                        branch->add_case(i - 1u, targets[i]);
                    }
                }
            }
            builder.set_insertion_point(blocks.back());
            builder.return_void();

            auto result = lower_xir_to_schedule(
                kernel, {.logical_warp_width = 8u});
            expect(result.succeeded())
                << "five-block forward graph " << encoded_graph << '\n'
                << diagnostics_text(result);
            if (!result.succeeded()) { return; }
            expect(verify(*result.function).succeeded());
            ++accepted_graph_count;
        }
        expect(accepted_graph_count == 122u);
    };

    "simd_xir_lowering_accepts_forward_reducible_cfg_family"_test = [] {
        constexpr auto graph_count = 96u;
        constexpr auto block_count = 12u;
        auto random_state = uint32_t{0x6d2b79f5u};
        auto next_random = [&]() noexcept {
            random_state = random_state * 1664525u + 1013904223u;
            return random_state;
        };
        for (auto graph = 0u; graph < graph_count; graph++) {
            Module module;
            auto *kernel = module.create_kernel();
            std::vector<xir::BasicBlock *> blocks;
            blocks.reserve(block_count);
            blocks.emplace_back(kernel->create_body_block());
            for (auto i = 1u; i < block_count; i++) {
                blocks.emplace_back(kernel->create_basic_block());
            }
            auto *lane = module.create_warp_lane_id();
            XIRBuilder builder;
            for (auto i = 0u; i + 1u < block_count; i++) {
                builder.set_insertion_point(blocks[i]);
                auto choose_forward_target = [&]() noexcept {
                    auto remaining = block_count - i - 1u;
                    return blocks[i + 1u + next_random() % remaining];
                };
                switch (next_random() % 3u) {
                    case 0u:
                        builder.br(blocks[i + 1u]);
                        break;
                    case 1u: {
                        auto threshold_value = next_random() % 8u;
                        auto *threshold = module.create_constant(
                            Type::of<uint>(), &threshold_value);
                        auto *condition = builder.call(
                            Type::of<bool>(), ArithmeticOp::BINARY_LESS,
                            {lane, threshold});
                        builder.cond_br(
                            condition, blocks[i + 1u],
                            choose_forward_target());
                        break;
                    }
                    default: {
                        auto *branch = builder.indexed_branch(lane);
                        branch->set_default_block(blocks[i + 1u]);
                        branch->add_case(0u, choose_forward_target());
                        branch->add_case(2u, choose_forward_target());
                        break;
                    }
                }
            }
            builder.set_insertion_point(blocks.back());
            builder.return_void();

            auto result = lower_xir_to_schedule(
                kernel, {.logical_warp_width = 8u});
            expect(result.succeeded())
                << "forward reducible graph " << graph << '\n'
                << diagnostics_text(result);
            if (!result.succeeded()) { return; }
            expect(verify(*result.function).succeeded());
        }
    };
}

void register_diagnostic_tests() {
    "simd_xir_lowering_rejects_structured_control"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *entry = kernel->create_body_block();
        auto *condition = module.create_constant_one(Type::of<bool>());
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *if_inst = builder.if_(condition);
        auto *left = if_inst->create_true_block();
        auto *right = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        builder.set_insertion_point(left);
        builder.br(merge);
        builder.set_insertion_point(right);
        builder.br(merge);
        builder.set_insertion_point(merge);
        builder.return_void();

        auto result = lower_xir_to_schedule(kernel);
        expect(!result.succeeded());
        expect(!result.diagnostics.empty());
        if (!result.diagnostics.empty()) {
            expect(result.diagnostics.front().code ==
                   XIRToScheduleDiagnosticCode::structured_control_flow);
            expect(result.diagnostics.front().message.find(
                       "destructure_cfg") != std::string::npos);
        }
    };

    "simd_xir_lowering_rejects_irreducible_cycle"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *entry = kernel->create_body_block();
        auto *left = kernel->create_basic_block();
        auto *right = kernel->create_basic_block();
        auto *exit = kernel->create_basic_block();
        auto *lane = module.create_warp_lane_id();
        auto *zero = module.create_constant_zero(Type::of<uint>());
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *condition = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_NOT_EQUAL,
            {lane, zero});
        builder.cond_br(condition, left, right);
        builder.set_insertion_point(left);
        builder.br(right);
        builder.set_insertion_point(right);
        builder.cond_br(condition, left, exit);
        builder.set_insertion_point(exit);
        builder.return_void();

        auto result = lower_xir_to_schedule(kernel);
        expect(!result.succeeded());
        auto found = false;
        for (auto &&diagnostic : result.diagnostics) {
            found |= diagnostic.code ==
                     XIRToScheduleDiagnosticCode::irreducible_control_flow;
        }
        expect(found);
    };
}

void register_memory_layout_tests() {
    "simd_xir_lowering_proves_packet_lane_buffer_layout"_test = [] {
        auto lower = [](luisa::uint3 block_size) {
            Module module;
            auto *kernel = module.create_kernel();
            kernel->set_block_size(block_size);
            auto *buffer_type = Type::buffer(Type::of<float>());
            auto *lhs = kernel->create_resource_argument(buffer_type);
            auto *rhs = kernel->create_resource_argument(buffer_type);
            auto *output = kernel->create_resource_argument(buffer_type);
            auto *entry = kernel->create_body_block();
            auto *dispatch_id = module.create_dispatch_id();
            auto *zero = module.create_constant_zero(Type::of<uint32_t>());
            auto *one = module.create_constant_one(Type::of<uint32_t>());
            uint32_t two_value = 2u;
            uint32_t five_value = 5u;
            uint32_t nine_value = 9u;
            auto *two = module.create_constant(
                Type::of<uint32_t>(), &two_value);
            auto *five = module.create_constant(
                Type::of<uint32_t>(), &five_value);
            auto *nine = module.create_constant(
                Type::of<uint32_t>(), &nine_value);
            XIRBuilder builder;
            builder.set_insertion_point(entry);
            auto *column = builder.call(
                Type::of<uint32_t>(), ArithmeticOp::EXTRACT,
                {dispatch_id, zero});
            auto *row = builder.call(
                Type::of<uint32_t>(), ArithmeticOp::EXTRACT,
                {dispatch_id, one});
            auto *lhs_row = builder.call(
                Type::of<uint32_t>(), ArithmeticOp::BINARY_MUL,
                {row, five});
            auto *lhs_index = builder.call(
                Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD,
                {lhs_row, two});
            auto *rhs_row = builder.call(
                Type::of<uint32_t>(), ArithmeticOp::BINARY_MUL,
                {two, nine});
            auto *rhs_index = builder.call(
                Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD,
                {rhs_row, column});
            auto *output_row = builder.call(
                Type::of<uint32_t>(), ArithmeticOp::BINARY_MUL,
                {row, nine});
            auto *output_index = builder.call(
                Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD,
                {output_row, column});
            auto *a = builder.call(
                Type::of<float>(), ResourceReadOp::BUFFER_READ,
                {lhs, lhs_index});
            auto *b = builder.call(
                Type::of<float>(), ResourceReadOp::BUFFER_READ,
                {rhs, rhs_index});
            auto *product = builder.call(
                Type::of<float>(), ArithmeticOp::BINARY_MUL, {a, b});
            builder.call(
                ResourceWriteOp::BUFFER_WRITE,
                {output, output_index, product});
            builder.return_void();
            return lower_xir_to_schedule(
                kernel, {.logical_warp_width = 8u});
        };

        auto row_aligned = lower(luisa::make_uint3(64u, 1u, 1u));
        expect(row_aligned.succeeded()) << diagnostics_text(row_aligned);
        if (!row_aligned.succeeded()) { return; }
        auto equal_reads = size_t{0u};
        auto consecutive_reads = size_t{0u};
        auto consecutive_writes = size_t{0u};
        for (auto &&block : row_aligned.function->blocks()) {
            for (auto &&instruction : block.instructions) {
                if (instruction.opcode == Opcode::resource_read) {
                    equal_reads +=
                        instruction.cohort_uniform_operand_index == 1u;
                    consecutive_reads +=
                        instruction.lane_consecutive_operand_index == 1u;
                } else if (instruction.opcode == Opcode::resource_write) {
                    consecutive_writes +=
                        instruction.lane_consecutive_operand_index == 1u;
                }
            }
        }
        expect(equal_reads == 1u);
        expect(consecutive_reads == 1u);
        expect(consecutive_writes == 1u);
        expect(verify(*row_aligned.function).succeeded());

        auto row_crossing = lower(luisa::make_uint3(4u, 8u, 1u));
        expect(row_crossing.succeeded()) << diagnostics_text(row_crossing);
        if (!row_crossing.succeeded()) { return; }
        auto annotated = size_t{0u};
        for (auto &&block : row_crossing.function->blocks()) {
            for (auto &&instruction : block.instructions) {
                annotated +=
                    instruction.cohort_uniform_operand_index.has_value();
                annotated +=
                    instruction.lane_consecutive_operand_index.has_value();
            }
        }
        expect(annotated == 0u);
        expect(verify(*row_crossing.function).succeeded());
    };
}

void register_debug_instruction_tests() {
    "simd_xir_lowering_preserves_debug_side_effects"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *entry = kernel->create_body_block();
        auto *condition = module.create_constant_one(Type::of<bool>());
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *clock = builder.clock();
        builder.print("clock={}", {clock});
        builder.assert_(condition, "clock assertion");
        builder.return_void();

        auto result = lower_xir_to_schedule(
            kernel, {.logical_warp_width = 8u});
        expect(result.succeeded()) << diagnostics_text(result);
        if (!result.succeeded()) { return; }
        auto print_count = size_t{0u};
        auto assert_count = size_t{0u};
        auto clock_count = size_t{0u};
        for (auto &&block : result.function->blocks()) {
            for (auto &&instruction : block.instructions) {
                if (instruction.opcode == Opcode::print) {
                    print_count++;
                    expect(instruction.message == "clock={}");
                } else if (instruction.opcode == Opcode::assert_) {
                    assert_count++;
                    expect(instruction.message == "clock assertion");
                } else if (instruction.opcode == Opcode::clock) {
                    clock_count++;
                }
            }
        }
        expect(print_count == 1u);
        expect(assert_count == 1u);
        expect(clock_count == 1u);
        expect(verify(*result.function).succeeded());
    };
}

void register_ray_query_pipeline_tests() {
    "simd_xir_lowering_projects_capture_free_ray_query_pipeline"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *surface = module.create_callable(nullptr);
        auto *procedural = module.create_callable(nullptr);
        for (auto *callback : {surface, procedural}) {
            static_cast<void>(callback->create_reference_argument(
                Type::of<RayQueryAll>()));
            auto *body = callback->create_body_block();
            XIRBuilder callback_builder;
            callback_builder.set_insertion_point(body);
            callback_builder.return_void();
        }
        auto *entry = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *query = builder.alloca_local(Type::of<RayQueryAll>());
        auto *pipeline = builder.ray_query_pipeline(
            query, surface, procedural, {});
        builder.return_void();

        auto result = lower_xir_to_schedule(
            kernel, {.logical_warp_width = 8u});
        expect(result.succeeded()) << diagnostics_text(result);
        if (!result.succeeded()) { return; }
        expect(result.ray_query_pipelines.size() == 1u);
        expect(result.ray_query_pipelines.front() == pipeline);
        auto pipeline_count = size_t{0u};
        for (auto &&block : result.function->blocks()) {
            for (auto &&instruction : block.instructions) {
                if (instruction.opcode != Opcode::ray_query_pipeline) {
                    continue;
                }
                ++pipeline_count;
                expect(instruction.source_op == 0u);
                expect(instruction.operands.size() == 1u);
            }
        }
        expect(pipeline_count == 1u);
        expect(verify(*result.function).succeeded());
    };

    "simd_xir_lowering_preserves_ray_query_pipeline_captures"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *captured = kernel->create_value_argument(Type::of<int>());
        auto *surface = module.create_callable(nullptr);
        auto *procedural = module.create_callable(nullptr);
        for (auto *callback : {surface, procedural}) {
            static_cast<void>(callback->create_reference_argument(
                Type::of<RayQueryAll>()));
            static_cast<void>(callback->create_value_argument(
                Type::of<int>()));
            auto *body = callback->create_body_block();
            XIRBuilder callback_builder;
            callback_builder.set_insertion_point(body);
            callback_builder.return_void();
        }
        auto *entry = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *query = builder.alloca_local(Type::of<RayQueryAll>());
        std::array<xir::Value *, 1u> captures{captured};
        auto *pipeline = builder.ray_query_pipeline(
            query, surface, procedural, captures);
        builder.return_void();

        auto result = lower_xir_to_schedule(
            kernel, {.logical_warp_width = 8u});
        expect(result.succeeded()) << diagnostics_text(result);
        if (!result.succeeded()) { return; }
        expect(result.ray_query_pipelines.size() == 1u);
        expect(result.ray_query_pipelines.front() == pipeline);
        auto pipeline_count = size_t{0u};
        for (auto &&block : result.function->blocks()) {
            for (auto &&instruction : block.instructions) {
                if (instruction.opcode != Opcode::ray_query_pipeline) {
                    continue;
                }
                pipeline_count++;
                expect(instruction.source_op == 0u);
                expect(instruction.operands.size() == 2u);
                if (instruction.operands.size() == 2u) {
                    auto *capture = result.function->value(
                        instruction.operands[1u]);
                    expect(capture != nullptr);
                    if (capture != nullptr) {
                        expect(capture->value_class ==
                               ValueClass::warp_uniform);
                    }
                }
            }
        }
        expect(pipeline_count == 1u);
        expect(verify(*result.function).succeeded());
    };

    "simd_xir_lowering_validates_parameter_class_overrides"_test = [] {
        Module module;
        auto *callable = module.create_callable(nullptr);
        static_cast<void>(
            callable->create_value_argument(Type::of<int>()));
        auto *body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.return_void();

        std::array classes{ValueClass::mask};
        auto invalid_class = lower_xir_to_schedule(
            callable,
            {.logical_warp_width = 8u,
             .parameter_value_classes = classes});
        expect(!invalid_class.succeeded());
        expect(!invalid_class.diagnostics.empty());

        classes[0u] = ValueClass::warp_uniform;
        auto preserved_class = lower_xir_to_schedule(
            callable,
            {.logical_warp_width = 8u,
             .parameter_value_classes = classes});
        expect(preserved_class.succeeded()) << diagnostics_text(preserved_class);
        if (preserved_class.succeeded()) {
            auto parameter_count = size_t{0u};
            for (auto &&value : preserved_class.function->values()) {
                if (value.origin != ValueOrigin::parameter) { continue; }
                parameter_count++;
                expect(value.value_class ==
                       ValueClass::warp_uniform);
            }
            expect(parameter_count == 1u);
        }

        auto invalid_count = lower_xir_to_schedule(
            callable,
            {.logical_warp_width = 8u,
             .parameter_value_classes =
                 std::span<const ValueClass>{}});
        expect(invalid_count.succeeded()) << diagnostics_text(invalid_count);
        std::array<ValueClass, 2u> too_many{
            ValueClass::varying, ValueClass::varying};
        invalid_count = lower_xir_to_schedule(
            callable,
            {.logical_warp_width = 8u,
             .parameter_value_classes = too_many});
        expect(!invalid_count.succeeded());
        expect(!invalid_count.diagnostics.empty());
    };
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    register_diamond_tests();
    register_loop_tests();
    register_block_barrier_tests();
    register_reducible_cfg_tests();
    register_memory_layout_tests();
    register_debug_instruction_tests();
    register_ray_query_pipeline_tests();
    register_diagnostic_tests();
    return 0;
}
