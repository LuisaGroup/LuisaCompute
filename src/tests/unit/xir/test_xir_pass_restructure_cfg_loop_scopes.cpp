// A selection merge must be inferred in a graph whose loop scopes are known.
// Reduced from the native SPIR-V coroutine continuation of a surface shader.
#include "ut/ut.hpp"
#include "xir_cfg_test_utils.h"

#include <luisa/ast/type_registry.h>
#include <luisa/core/logging.h>

#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/translators/xir_interchange.h>
#include <luisa/xir/verifier.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "indexed_selection_merge_cannot_cross_a_nested_loop"_test = [] {
        for (auto mutation : {RestructureCFGMutationMode::TRANSACTIONAL,
                              RestructureCFGMutationMode::IN_PLACE_DISCARDABLE}) {
            for (auto case_count : {1u, 8u}) {
                for (auto rotation : {0u, 3u, 7u}) {
                    Module module;
                    auto *kernel = module.create_kernel();
                    std::array<BasicBlock *, 11u> blocks;
                    blocks[0] = kernel->create_body_block();
                    for (auto i = 1u; i < blocks.size(); ++i) {
                        // Equivalent CFGs with different owned-block/allocation
                        // orders must all converge, including shadow replay.
                        blocks[1u + (i - 1u + rotation) % (blocks.size() - 1u)] =
                            kernel->create_basic_block();
                    }
                    auto *output = kernel->create_reference_argument(Type::of<uint32_t>());
                    auto *stop = kernel->create_value_argument(Type::of<bool>());
                    auto *take_switch = kernel->create_value_argument(Type::of<bool>());
                    auto *inner_index = kernel->create_value_argument(Type::of<uint32_t>());
                    auto *outer_index = kernel->create_value_argument(Type::of<uint32_t>());
                    auto *enter = kernel->create_value_argument(Type::of<bool>());
                    auto *one = module.create_constant_one(Type::of<uint32_t>());
                    XIRBuilder b;
                    b.set_insertion_point(blocks[0]);
                    b.br(blocks[7]);
                    b.set_insertion_point(blocks[1]);
                    auto *after_inner = b.store(output, one);
                    b.br(blocks[6]);
                    b.set_insertion_point(blocks[2]);
                    b.cond_br(stop, blocks[1], blocks[3]);
                    b.set_insertion_point(blocks[3]);
                    b.cond_br(take_switch, blocks[4], blocks[2]);
                    b.set_insertion_point(blocks[4]);
                    auto *inner = b.indexed_branch(inner_index);
                    inner->set_default_block(blocks[5]);
                    for (auto i = 0u; i < case_count; ++i) {
                        inner->add_case(i, blocks[5]);
                    }
                    b.set_insertion_point(blocks[5]);
                    auto *inner_payload = b.store(output, one);
                    b.br(blocks[2]);
                    b.set_insertion_point(blocks[6]);
                    auto *returned = b.return_void();
                    b.set_insertion_point(blocks[7]);
                    auto *outer = b.indexed_branch(outer_index);
                    outer->set_default_block(blocks[9]);
                    outer->add_case(0u, blocks[10]);
                    outer->add_case(1u, blocks[8]);
                    b.set_insertion_point(blocks[8]);
                    auto *outer_payload = b.store(output, one);
                    b.br(blocks[7]);
                    b.set_insertion_point(blocks[9]);
                    b.unreachable_();
                    b.set_insertion_point(blocks[10]);
                    b.cond_br(enter, blocks[2], blocks[6]);

                    expect(xir_verify_module(&module).succeeded());
                    auto info = restructure_cfg_pass_run_on_function(
                        kernel, {.mutation_mode = mutation,
                                 .verify_remaining_divergent_index = true});
                    expect(info.succeeded());
                    expect(info.iteration_limit_count == 0u);
                    expect(info.unstructured_branch_count == 0u);
                    expect(info.invalid_construct_count == 0u);
                    expect(after_inner->is_linked());
                    expect(inner_payload->is_linked());
                    expect(outer_payload->is_linked());
                    expect(returned->is_linked());
                    auto verification = xir_verify_module(
                        &module, {.require_no_phi = true,
                                  .require_unique_merge_blocks = true,
                                  .require_canonical_break_continue_targets = true});
                    expect(verification.succeeded())
                        << (verification.errors.empty() ? "" :
                                                          verification.errors.front().message.c_str());
                    // Raw conditionals are legal loop prepares or direct
                    // exits through an enclosing construct's boundary. Check
                    // that ownership explicitly instead of requiring an extra
                    // If/selector around an already structured transfer.
                    auto dominance = compute_dom_tree(kernel);
                    for (auto *block : kernel->basic_blocks()) {
                        auto *term = block->terminator();
                        expect(!term->isa<IndexedBranchInst>());
                        if (!term->isa<ConditionalBranchInst>()) { continue; }
                        expect(luisa::test::raw_conditional_has_structured_owner(
                            kernel, static_cast<ConditionalBranchInst *>(term), dominance));
                    }
                    if (info.succeeded()) {
                        auto again = restructure_cfg_pass_run_on_function(
                            kernel, {.mutation_mode = mutation});
                        expect(again.succeeded());
                        expect(!again.changed());
                    }
                }
            }
        }
    };

    "crossing_declared_loop_epochs_restructure_without_unbounded_exit_growth"_test = [] {
        for (auto mutation : {RestructureCFGMutationMode::TRANSACTIONAL,
                              RestructureCFGMutationMode::IN_PLACE_DISCARDABLE}) {
            Module module;
            auto *kernel = module.create_kernel();
            std::array<BasicBlock *, 21u> blocks;
            blocks[0] = kernel->create_body_block();
            for (auto i = 1u; i < blocks.size(); ++i) {
                blocks[i] = kernel->create_basic_block();
            }
            auto *condition = kernel->create_value_argument(Type::of<bool>());
            auto *index = kernel->create_value_argument(Type::of<uint32_t>());
            auto *output = kernel->create_reference_argument(Type::of<uint32_t>());
            auto *one = module.create_constant_one(Type::of<uint32_t>());
            XIRBuilder b;
            b.set_insertion_point(blocks[0]);
            auto *outer_loop = b.simple_loop();
            outer_loop->set_body_block(blocks[7]);
            outer_loop->set_merge_block(blocks[14]);
            b.set_insertion_point(blocks[7]);
            auto *outer = b.switch_(index);
            outer->set_default_block(blocks[9]);
            outer->add_case(0u, blocks[10]);
            outer->add_case(1u, blocks[8]);
            // Deliberately crosses the inner loop's epoch. The ordinary XIR
            // verifier checks ownership/typing, not this SPIR-V hierarchy.
            outer->set_merge_block(blocks[17]);
            b.set_insertion_point(blocks[10]);
            auto *enter = b.if_(condition);
            enter->set_true_target(blocks[12]);
            enter->set_false_target(blocks[18]);
            enter->set_merge_block(blocks[18]);
            b.set_insertion_point(blocks[12]);
            auto *inner_loop = b.loop();
            inner_loop->set_prepare_block(blocks[2]);
            inner_loop->set_body_block(blocks[3]);
            inner_loop->set_update_block(blocks[11]);
            inner_loop->set_merge_block(blocks[13]);
            b.set_insertion_point(blocks[2]);
            b.cond_br(condition, blocks[3], blocks[13]);
            b.set_insertion_point(blocks[3]);
            auto *take_switch = b.if_(condition);
            take_switch->set_true_target(blocks[19]);
            take_switch->set_false_target(blocks[11]);
            take_switch->set_merge_block(blocks[19]);
            b.set_insertion_point(blocks[4]);
            auto *inner = b.indexed_branch(index);
            inner->set_default_block(blocks[20]);
            inner->add_case(0u, blocks[15]);
            for (auto [source, target] : {
                     std::pair{1u, 18u}, {5u, 11u}, {8u, 7u}}) {
                b.set_insertion_point(blocks[source]);
                b.store(output, one);
                b.br(blocks[target]);
            }
            for (auto [source, target] : {
                     std::pair{11u, 2u}, {13u, 1u}, {15u, 20u}, {17u, 4u}, {18u, 6u}, {19u, 17u}, {20u, 5u}}) {
                b.set_insertion_point(blocks[source]);
                b.br(blocks[target]);
            }
            b.set_insertion_point(blocks[6]);
            b.return_void();
            for (auto i : {9u, 14u, 16u}) {
                b.set_insertion_point(blocks[i]);
                b.unreachable_();
            }
            expect(xir_verify_module(&module).succeeded());
            auto info = restructure_cfg_pass_run_on_function(
                kernel, {.mutation_mode = mutation});
            // This input is valid XIR with inconsistent region annotations.
            // Its former rejection was the repeated-obligation guard firing,
            // not a required pass outcome. The finite counterpart in
            // crossing_loop_epochs_preserve_bounded_execution checks complete
            // store sequences for this same crossing declaration. Keep this
            // original infinite-loop shape as a convergence/structure check.
            expect(info.succeeded());
            expect(info.invalid_construct_count == 0u);
            expect(info.unstructured_branch_count == 0u);
            expect(info.iteration_limit_count == 0u);
            auto verification = xir_verify_module(
                &module, {.require_no_phi = true,
                          .require_unique_merge_blocks = true,
                          .require_canonical_break_continue_targets = true});
            expect(verification.succeeded())
                << (verification.errors.empty() ? "" :
                                                  verification.errors.front().message.c_str());
            if (info.succeeded()) {
                auto record_graph = [&](uint32_t invocation, const RestructureCFGInfo &stats) {
                    auto graph = xir_to_interchange_text(&module);
                    expect(graph.succeeded());
                    if (auto *directory = std::getenv("LUISA_XIR_CFG_TEST_DUMP_DIR")) {
                        std::filesystem::create_directories(directory);
                        auto mode_name = mutation == RestructureCFGMutationMode::TRANSACTIONAL ? "transactional" : "in-place";
                        auto filename = std::string{"crossing-epochs-"} + mode_name + "-pass-" + std::to_string(invocation) + ".xir";
                        std::ofstream file{std::filesystem::path{directory} / filename};
                        file << graph.text;
                        expect(file.good());
                        LUISA_INFO(
                            "crossing epochs {} invocation {}: loops={}, ifs={}, switches={}, canonicalized={}.",
                            mode_name, invocation, stats.restructured_loop_count,
                            stats.restructured_if_count, stats.restructured_switch_count,
                            stats.canonicalized_cfg_count);
                    }
                    return graph;
                };
                auto previous = record_graph(1u, info);
                for (auto invocation = 2u; invocation <= 4u; ++invocation) {
                    auto again = restructure_cfg_pass_run_on_function(
                        kernel, {.mutation_mode = mutation});
                    expect(again.succeeded());
                    expect(!again.changed()) << "invocation=" << invocation;
                    expect(again.iteration_limit_count == 0u);
                    auto current = record_graph(invocation, again);
                    auto identical = current.text == previous.text;
                    expect(identical) << "public-pass fixed point invocation=" << invocation;
                    previous = std::move(current);
                }
            }
        }
    };
    "conditional_latch_preserves_both_exit_payloads"_test = [] {
        for (auto mutation : {RestructureCFGMutationMode::TRANSACTIONAL,
                              RestructureCFGMutationMode::IN_PLACE_DISCARDABLE}) {
            Module module;
            auto *kernel = module.create_kernel();
            auto *output = kernel->create_reference_argument(Type::of<uint>());
            auto *first = kernel->create_value_argument(Type::of<bool>());
            auto *second = kernel->create_value_argument(Type::of<bool>());
            auto *entry = kernel->create_body_block();
            auto *header = kernel->create_basic_block();
            auto *latch = kernel->create_basic_block();
            auto *left = kernel->create_basic_block();
            auto *right = kernel->create_basic_block();
            auto *exit = kernel->create_basic_block();
            XIRBuilder b;
            b.set_insertion_point(entry);
            b.br(header);
            b.set_insertion_point(header);
            b.cond_br(first, left, latch);
            b.set_insertion_point(latch);
            b.cond_br(second, right, header);
            b.set_insertion_point(left);
            auto *left_store = b.store(output, module.create_constant_zero(Type::of<uint>()));
            b.br(exit);
            b.set_insertion_point(right);
            auto *right_store = b.store(output, module.create_constant_one(Type::of<uint>()));
            b.br(exit);
            b.set_insertion_point(exit);
            b.return_void();
            auto info = restructure_cfg_pass_run_on_function(kernel, {.mutation_mode = mutation});
            expect(info.succeeded());
            bool left_reachable = false, right_reachable = false;
            kernel->traverse_basic_blocks([&](BasicBlock *block) {
                for (auto *inst : block->instructions()) {
                    left_reachable |= inst == left_store;
                    right_reachable |= inst == right_store;
                }
            });
            expect(left_reachable && right_reachable) << "loop recovery must preserve both exit stores";
            expect(xir_verify_module(&module, {.require_unique_merge_blocks = true,
                                               .require_canonical_break_continue_targets = true})
                       .succeeded());
        }
    };
}
