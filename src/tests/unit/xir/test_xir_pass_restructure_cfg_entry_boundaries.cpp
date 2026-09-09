// Regressions for construct entries that are boundaries of enclosing loops.

#include "ut/ut.hpp"
#include "common/xir_cfg_test_utils.h"

#include <luisa/ast/type_registry.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/runtime/rtx/ray.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/verifier.h>
#include <luisa/xir/translators/xir_interchange.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "cloned_query_initialization_preserves_its_frontier_consumers"_test = [] {
        for (auto mode : {RestructureCFGMutationMode::TRANSACTIONAL,
                          RestructureCFGMutationMode::IN_PLACE_DISCARDABLE}) {
            for (auto shape : {0u, 1u, 2u, 3u}) {
                Module module;
                auto *kernel = module.create_kernel();
                auto *entry = kernel->create_body_block();
                auto *condition = kernel->create_value_argument(Type::of<bool>());
                auto *accel = kernel->create_resource_argument(Type::of<Accel>());
                auto *ray = kernel->create_value_argument(Type::of<Ray>());
                auto *output = kernel->create_reference_argument(Type::of<bool>());
                auto *shared = kernel->create_basic_block();
                auto *other = kernel->create_basic_block();
                auto *merge = kernel->create_basic_block();
                auto *query_type = Type::custom("LC_RayQueryAll");
                XIRBuilder b;
                b.set_insertion_point(entry);
                auto *query = b.alloca_local(query_type);
                auto *root = b.if_(condition);
                root->set_true_target(shared);
                root->set_false_target(other);
                root->set_merge_block(merge);
                b.set_insertion_point(other);
                b.store(output, module.create_constant_zero(Type::of<bool>()));
                b.br(shared);
                b.set_insertion_point(shared);
                auto *initializer = b.call(query_type, ResourceQueryOp::RAY_TRACING_QUERY_ALL,
                                           {accel, ray, module.create_constant_one(Type::of<uint32_t>())});
                b.store(query, initializer);
                Value *handle = query;
                if (shape == 3u) {
                    // The post-merge frontier E is inside the initializer's
                    // lifetime. E and D also belong to one executable SCC:
                    // the first arrival at E uses D's existing object, while
                    // an outer iteration initializes it again at D.
                    auto *consumer = kernel->create_basic_block();
                    auto *exit = kernel->create_basic_block();
                    b.br(consumer);
                    b.set_insertion_point(consumer);
                    b.store(output, b.call(Type::of<bool>(), RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED, {handle}));
                    b.cond_br(condition, shared, merge);
                    b.set_insertion_point(merge);
                    b.cond_br(condition, consumer, exit);
                    b.set_insertion_point(exit);
                } else {
                    b.br(merge);
                    b.set_insertion_point(merge);
                    if (shape == 2u) {
                        // Discovering the first query's consumer also discovers a
                        // second query whose lifetime extends beyond this block.
                        b.store(output, b.call(Type::of<bool>(), RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED, {handle}));
                        auto *next = b.alloca_local(query_type);
                        b.store(next, b.call(query_type, ResourceQueryOp::RAY_TRACING_QUERY_ALL,
                                             {accel, ray, module.create_constant_one(Type::of<uint32_t>())}));
                        handle = next;
                    }
                    if (shape != 0u) {
                        auto *yes = kernel->create_basic_block();
                        auto *no = kernel->create_basic_block();
                        auto *join = kernel->create_basic_block();
                        b.cond_br(condition, yes, no);
                        b.set_insertion_point(yes);
                        b.store(output, b.call(Type::of<bool>(), RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED, {handle}));
                        b.br(join);
                        b.set_insertion_point(no);
                        b.store(output, b.call(Type::of<bool>(), RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED, {handle}));
                        b.br(join);
                        b.set_insertion_point(join);
                    } else {
                        b.store(output, b.call(Type::of<bool>(), RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED, {handle}));
                    }
                }
                b.return_void();

                // This is the native query lifetime contract: one direct binding
                // for each affine object must dominate every consumer. A merge
                // block is allowed to consume a query initialized in its arm when
                // that initializer dominates the merge in the executable CFG.
                auto verify_lifetimes = [&] {
                    auto dom = compute_dom_tree(kernel);
                    for (auto *block : kernel->basic_blocks()) {
                        for (auto *inst : block->instructions()) {
                            if (!inst->isa<AllocaInst>() || inst->type() != query_type) { continue; }
                            StoreInst *binding = nullptr;
                            luisa::vector<Instruction *> consumers;
                            for (auto *use : inst->use_list()) {
                                auto *user = static_cast<Instruction *>(use->user());
                                if (!dom.contains(user->parent_block())) { continue; }
                                if (user->isa<StoreInst>() && static_cast<StoreInst *>(user)->variable() == inst) {
                                    if (binding != nullptr) { return false; }
                                    binding = static_cast<StoreInst *>(user);
                                } else {
                                    consumers.emplace_back(user);
                                }
                            }
                            if (binding == nullptr && consumers.empty()) { continue; }
                            if (binding == nullptr || !binding->value()->isa<ResourceQueryInst>()) { return false; }
                            for (size_t i = 0u; i < consumers.size(); ++i) {
                                auto *consumer = consumers[i];
                                if (!dom.dominates(binding->parent_block(), consumer->parent_block())) { return false; }
                                if (binding->parent_block() == consumer->parent_block()) {
                                    auto *cursor = binding->next();
                                    while (cursor != consumer && cursor != binding->parent_block()->instructions().tail_sentinel()) {
                                        cursor = cursor->next();
                                    }
                                    if (cursor != consumer) { return false; }
                                }
                                if (consumer->type() == query_type) {
                                    for (auto *use : consumer->use_list()) {
                                        consumers.emplace_back(static_cast<Instruction *>(use->user()));
                                    }
                                }
                            }
                        }
                    }
                    return true;
                };
                expect(xir_verify_module(&module).succeeded());
                expect(verify_lifetimes());
                auto dump = [&](bool before) {
                    if (auto *directory = std::getenv("LUISA_XIR_CFG_TEST_DUMP_DIR")) {
                        std::filesystem::create_directories(directory);
                        auto name = std::string{"affine-frontier-"} + std::to_string(shape) +
                                    (mode == RestructureCFGMutationMode::TRANSACTIONAL ? "-transactional" : "-in-place") +
                                    (before ? "-input.xir" : "-output.xir");
                        auto interchange = xir_to_interchange_text(&module);
                        expect(interchange.succeeded());
                        std::ofstream file{std::filesystem::path{directory} / name};
                        file << interchange.text;
                        expect(file.good());
                    }
                };
                dump(true);
                auto info = restructure_cfg_pass_run_on_function(kernel, {.mutation_mode = mode});
                dump(false);
                expect(info.succeeded()) << "shape=" << shape;
                expect(xir_verify_module(&module, {.require_no_phi = true,
                                                   .require_unique_merge_blocks = true,
                                                   .require_canonical_break_continue_targets = true})
                           .succeeded());
                auto dom = compute_dom_tree(kernel);
                kernel->traverse_instructions([&](Instruction *inst) {
                    expect(!inst->isa<IndexedBranchInst>());
                    if (inst->isa<ConditionalBranchInst>()) {
                        expect(luisa::test::raw_conditional_has_structured_owner(
                            kernel, static_cast<ConditionalBranchInst *>(inst), dom));
                    }
                });
                expect(verify_lifetimes()) << "node splitting must copy the consumers of an affine query together with its initializer, shape=" << shape;
            }
        }
    };

    "nested_selection_does_not_clone_enclosing_loop_update"_test = [] {
        for (auto mode : {RestructureCFGMutationMode::TRANSACTIONAL,
                          RestructureCFGMutationMode::IN_PLACE_DISCARDABLE}) {
            Module module;
            auto *kernel = module.create_kernel();
            auto *definition = kernel->definition();
            auto *entry = kernel->create_body_block();
            auto *loop_condition =
                kernel->create_value_argument(Type::of<bool>());
            auto *outer_condition =
                kernel->create_value_argument(Type::of<bool>());
            auto *hit_condition =
                kernel->create_value_argument(Type::of<bool>());
            auto *loop_header = definition->create_basic_block();
            auto *outer_header = definition->create_basic_block();
            auto *inner_header = definition->create_basic_block();
            auto *hit = definition->create_basic_block();
            auto *update = definition->create_basic_block();
            auto *exit = definition->create_basic_block();
            XIRBuilder builder;

            builder.set_insertion_point(entry);
            auto *update_marker =
                builder.alloca_local(Type::of<uint32_t>());
            builder.store(
                update_marker,
                module.create_constant_zero(Type::of<uint32_t>()));
            builder.br(loop_header);

            builder.set_insertion_point(loop_header);
            builder.cond_br(
                loop_condition, outer_header, exit);
            builder.set_insertion_point(outer_header);
            builder.cond_br(
                outer_condition, inner_header, update);
            builder.set_insertion_point(inner_header);
            builder.cond_br(hit_condition, hit, update);
            builder.set_insertion_point(hit);
            builder.br(exit);
            builder.set_insertion_point(update);
            builder.store(
                update_marker,
                module.create_constant_one(Type::of<uint32_t>()));
            builder.br(loop_header);
            builder.set_insertion_point(exit);
            builder.return_void();

            expect(xir_verify_module(&module).succeeded());
            auto info = restructure_cfg_pass_run_on_function(
                kernel,
                {.mutation_mode = mode});
            expect(info.succeeded());
            expect(info.unstructured_branch_count == 0u);
            auto verification = xir_verify_module(&module);
            expect(verification.succeeded())
                << (verification.errors.empty() ?
                        "unknown verification error" :
                        verification.errors.front().message.c_str());

            auto marker_store_count = size_t{0u};
            definition->traverse_basic_blocks(
                [&](BasicBlock *block) noexcept {
                    for (auto *instruction : block->instructions()) {
                        if (!instruction->isa<StoreInst>()) {
                            continue;
                        }
                        marker_store_count +=
                            static_cast<StoreInst *>(instruction)
                                ->variable() == update_marker;
                    }
                });
            expect(marker_store_count == 2u)
                << "entry canonicalization may subdivide the boundary edge, "
                   "but must not clone its update payload";
            auto loop_count = size_t{0u};
            definition->traverse_basic_blocks([&](BasicBlock *block) noexcept {
                if (block->is_terminated() && block->terminator()->isa<LoopInst>()) {
                    ++loop_count;
                    expect(static_cast<LoopInst *>(block->terminator())->update_block() == update)
                        << "a unique payloadful latch without a prepare bypass keeps its update role";
                }
            });
            expect(loop_count == 1u);
        }
    };

    return 0;
}
