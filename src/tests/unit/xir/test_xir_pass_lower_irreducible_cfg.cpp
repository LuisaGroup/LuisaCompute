#include "ut/ut.hpp"

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/lower_irreducible_cfg.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/translators/xir2ast.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] size_t count_noncanonical_raw_branches(
    FunctionDefinition *definition) noexcept {
    luisa::unordered_map<BasicBlock *, LoopInst *> loop_prepares;
    definition->traverse_instructions(
        [&](Instruction *instruction) noexcept {
            if (!instruction->isa<LoopInst>()) { return; }
            auto *loop = static_cast<LoopInst *>(instruction);
            loop_prepares.emplace(loop->prepare_block(), loop);
        });
    auto count = size_t{0u};
    definition->traverse_instructions(
        [&](Instruction *instruction) noexcept {
            if (instruction->isa<IndexedBranchInst>()) {
                ++count;
                return;
            }
            if (!instruction->isa<ConditionalBranchInst>()) {
                return;
            }
            auto *branch = static_cast<ConditionalBranchInst *>(
                instruction);
            auto iter = loop_prepares.find(
                branch->parent_block());
            auto canonical = iter != loop_prepares.end() &&
                             branch->true_block() ==
                                 iter->second->body_block() &&
                             branch->false_block() ==
                                 iter->second->merge_block();
            count += canonical ? 0u : 1u;
        });
    return count;
}

}// namespace

void reg_xir_pass_lower_irreducible_cfg() {
    "lower_irreducible_cfg_builds_one_entry_dispatch_without_cloning_body"_test = [] {
        Module module;
        auto *callable = module.create_callable(nullptr);
        auto *choose_entry = callable->create_value_argument(
            Type::of<bool>());
        auto *leave_left = callable->create_value_argument(
            Type::of<bool>());
        auto *leave_right = callable->create_value_argument(
            Type::of<bool>());
        auto *entry = callable->create_body_block();
        auto *left_predecessor = callable->create_basic_block();
        auto *right_predecessor = callable->create_basic_block();
        auto *left = callable->create_basic_block();
        auto *right = callable->create_basic_block();
        auto *exit = callable->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        builder.cond_br(
            choose_entry,
            left_predecessor,
            right_predecessor);
        builder.set_insertion_point(left_predecessor);
        builder.br(left);
        builder.set_insertion_point(right_predecessor);
        builder.br(right);
        builder.set_insertion_point(left);
        builder.cond_br(leave_left, exit, right);
        builder.set_insertion_point(right);
        builder.cond_br(leave_right, exit, left);
        builder.set_insertion_point(exit);
        builder.return_void();

        expect(xir_verify_function(callable).succeeded());
        auto rejected = restructure_cfg_pass_run_on_function(
            callable);
        expect(!rejected.succeeded());
        expect(rejected.irreducible_region_count == 1u);
        expect(count_noncanonical_raw_branches(
                   callable->definition()) == 3u);

        auto lowered =
            lower_irreducible_cfg_pass_run_on_function(
                callable);
        expect(lowered.succeeded());
        expect(lowered.lowered_region_count == 1u);
        expect(lowered.created_dispatch_block_count == 1u);
        expect(lowered.created_edge_block_count == 4u)
            << "each distinct predecessor/entry pair must carry exactly "
               "one selector store";
        expect(lowered.remaining_irreducible_region_count == 0u);
        expect(xir_verify_function(
                   callable,
                   {.require_no_phi = true})
                   .succeeded());

        auto restructured =
            restructure_cfg_pass_run_on_function(callable);
        expect(restructured.succeeded());
        expect(restructured.irreducible_region_count == 0u);
        expect(count_noncanonical_raw_branches(
                   callable->definition()) == 0u)
            << "only structured branches or canonical loop prepares may "
               "remain after restructuring";
        auto ast = xir_to_ast_translate(
            *callable->definition(), {});
        expect(ast != nullptr);
    };

    "lower_irreducible_cfg_finds_nested_region_inside_single_entry_outer_scc"_test = [] {
        Module module;
        auto *callable = module.create_callable(nullptr);
        auto *leave_outer = callable->create_value_argument(
            Type::of<bool>());
        auto *choose_inner_entry = callable->create_value_argument(
            Type::of<bool>());
        auto *leave_left = callable->create_value_argument(
            Type::of<bool>());
        auto *leave_right = callable->create_value_argument(
            Type::of<bool>());
        auto *entry = callable->create_body_block();
        auto *outer_header = callable->create_basic_block();
        auto *inner_entry_selector = callable->create_basic_block();
        auto *left = callable->create_basic_block();
        auto *right = callable->create_basic_block();
        auto *exit = callable->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        builder.br(outer_header);
        builder.set_insertion_point(outer_header);
        builder.cond_br(
            leave_outer, exit, inner_entry_selector);
        builder.set_insertion_point(inner_entry_selector);
        builder.cond_br(choose_inner_entry, left, right);
        builder.set_insertion_point(left);
        builder.cond_br(leave_left, outer_header, right);
        builder.set_insertion_point(right);
        builder.cond_br(leave_right, outer_header, left);
        builder.set_insertion_point(exit);
        builder.return_void();

        expect(xir_verify_function(callable).succeeded());
        auto rejected = restructure_cfg_pass_run_on_function(
            callable);
        expect(!rejected.succeeded());
        expect(rejected.irreducible_region_count == 1u)
            << "the unique outer header must not hide the two-entry "
               "left/right cycle";

        auto lowered =
            lower_irreducible_cfg_pass_run_on_function(
                callable);
        expect(lowered.succeeded());
        expect(lowered.lowered_region_count == 1u);
        expect(lowered.created_dispatch_block_count == 1u);
        expect(lowered.created_edge_block_count == 4u);
        expect(lowered.remaining_irreducible_region_count == 0u);
        expect(xir_verify_function(
                   callable,
                   {.require_no_phi = true})
                   .succeeded());

        auto restructured =
            restructure_cfg_pass_run_on_function(callable);
        expect(restructured.succeeded());
        expect(restructured.irreducible_region_count == 0u);
        auto ast = xir_to_ast_translate(
            *callable->definition(), {});
        expect(ast != nullptr);
    };
}

int main() {
    reg_xir_pass_lower_irreducible_cfg();
    return 0;
}
