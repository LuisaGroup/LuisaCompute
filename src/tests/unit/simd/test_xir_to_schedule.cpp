#include "ut/ut.hpp"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <sstream>
#include <string>
#include <vector>

#include <luisa/ast/type_registry.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/module.h>

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

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    register_diamond_tests();
    register_loop_tests();
    register_reducible_cfg_tests();
    register_memory_layout_tests();
    register_diagnostic_tests();
    return 0;
}
