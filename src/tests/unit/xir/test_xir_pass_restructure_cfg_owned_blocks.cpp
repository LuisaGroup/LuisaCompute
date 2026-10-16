// Structure is an invariant of every owned block, including disconnected
// shells retained after coroutine continuation and exit canonicalization.

#include "ut/ut.hpp"

#include <luisa/ast/type_registry.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/unreachable.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "restructure_conditional_in_disconnected_owned_shell"_test = [] {
        for (auto mutation : {
                 RestructureCFGMutationMode::TRANSACTIONAL,
                 RestructureCFGMutationMode::IN_PLACE_DISCARDABLE}) {
            Module module;
            auto *kernel = module.create_kernel();
            auto *entry = kernel->create_body_block();
            auto *condition =
                kernel->create_value_argument(Type::of<bool>());
            auto *header = kernel->create_basic_block();
            auto *true_arm = kernel->create_basic_block();
            auto *false_arm = kernel->create_basic_block();
            XIRBuilder builder;
            builder.set_insertion_point(entry);
            auto *entry_return = builder.return_void();
            builder.set_insertion_point(header);
            builder.cond_br(condition, true_arm, false_arm);
            builder.set_insertion_point(true_arm);
            auto *true_return = builder.return_void();
            builder.set_insertion_point(false_arm);
            auto *false_return = builder.return_void();

            expect(xir_verify_module(&module).succeeded());
            auto info = restructure_cfg_pass_run_on_function(
                kernel, {.mutation_mode = mutation,
                         .verify_remaining_divergent_index = true});
            expect(info.succeeded());
            expect(info.unstructured_branch_count == 0u);
            expect(info.restructured_if_count == 1u);
            expect(header->terminator()->isa<IfInst>());
            // Rebuilding an owned shell must not drop it or modify any of
            // its executable edges, nor make the shell entry-reachable.
            expect(entry->terminator() == entry_return);
            expect(true_arm->terminator() == true_return);
            expect(false_arm->terminator() == false_return);
            if (header->terminator()->isa<IfInst>()) {
                auto *selection =
                    static_cast<IfInst *>(header->terminator());
                expect(selection->condition() == condition);
                expect(selection->true_block() == true_arm);
                expect(selection->false_block() == false_arm);
            }
            auto verification = xir_verify_module(
                &module,
                {.require_no_unstructured_control_flow = true,
                 .require_unique_merge_blocks = true});
            expect(verification.succeeded())
                << (verification.errors.empty() ? "" :
                        verification.errors.front().message.c_str());

            if (info.succeeded()) {
                auto second = restructure_cfg_pass_run_on_function(
                    kernel, {.mutation_mode = mutation,
                             .verify_remaining_divergent_index = true});
                expect(second.succeeded());
                expect(!second.changed());
            }
        }
    };

    "disconnected_selection_batch_does_not_claim_a_live_merge"_test = [] {
        for (auto mutation : {
                 RestructureCFGMutationMode::TRANSACTIONAL,
                 RestructureCFGMutationMode::IN_PLACE_DISCARDABLE}) {
            Module module;
            auto *kernel = module.create_kernel();
            auto *entry = kernel->create_body_block();
            auto *condition =
                kernel->create_value_argument(Type::of<bool>());
            auto *output =
                kernel->create_reference_argument(Type::of<uint32_t>());
            XIRBuilder builder;
            builder.set_insertion_point(entry);
            auto *entry_return = builder.return_void();
            luisa::vector<BasicBlock *> headers;
            luisa::vector<StoreInst *> stores;
            constexpr auto site_count = size_t{17u};
            for (auto i = size_t{0u}; i < site_count; ++i) {
                auto *header = kernel->create_basic_block();
                auto *true_arm = kernel->create_basic_block();
                auto *false_arm = kernel->create_basic_block();
                headers.emplace_back(header);
                builder.set_insertion_point(header);
                builder.cond_br(condition, true_arm, false_arm);
                for (auto *arm : {true_arm, false_arm}) {
                    builder.set_insertion_point(arm);
                    stores.emplace_back(builder.store(
                        output, module.create_constant_one(Type::of<uint32_t>())));
                    builder.br(entry);
                }
            }
            // Entry globally post-dominates both arms of every dead site,
            // but is not their lexical continuation: none has an entry-
            // rooted dominator-tree node. Their payload and edges are kept.
            expect(xir_verify_module(&module).succeeded());
            auto info = restructure_cfg_pass_run_on_function(
                kernel, {.mutation_mode = mutation,
                         .verify_remaining_divergent_index = true});
            expect(info.succeeded());
            expect(info.restructured_if_count == site_count);
            expect(entry->terminator() == entry_return);
            for (auto *header : headers) {
                expect(header->terminator()->isa<IfInst>());
                if (!header->terminator()->isa<IfInst>()) { continue; }
                auto *selection = static_cast<IfInst *>(header->terminator());
                auto *merge = selection->merge_block();
                // A transparent structural merge may precede the synthetic
                // unreachable block, but it must never point into live code.
                while (merge->terminator()->isa<BranchInst>()) {
                    merge = static_cast<BranchInst *>(merge->terminator())
                                ->target_block();
                }
                expect(merge->terminator()->isa<UnreachableInst>());
                for (auto *arm : {selection->true_block(), selection->false_block()}) {
                    expect(arm->terminator()->isa<BranchInst>());
                    if (arm->terminator()->isa<BranchInst>()) {
                        expect(static_cast<BranchInst *>(arm->terminator())
                                   ->target_block() == entry);
                    }
                }
            }
            for (auto *store : stores) { expect(store->is_linked()); }
            expect(xir_verify_module(
                       &module,
                       {.require_no_unstructured_control_flow = true,
                        .require_unique_merge_blocks = true})
                       .succeeded());
        }
    };
    return 0;
}
