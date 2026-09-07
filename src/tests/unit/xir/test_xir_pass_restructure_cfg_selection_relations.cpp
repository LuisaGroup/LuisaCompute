// Scale and oracle regressions for incremental selection-exit relations.

#include "ut/ut.hpp"

#include <luisa/ast/type_registry.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/verifier.h>

#include <cstdlib>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void enable_relation_oracle() noexcept {
#ifdef _WIN32
    _putenv_s(
        "LUISA_XIR_VERIFY_SELECTION_EXIT_RELATION_UPDATES", "1");
#else
    setenv(
        "LUISA_XIR_VERIFY_SELECTION_EXIT_RELATION_UPDATES", "1", 1);
#endif
}

[[nodiscard]] bool branch_chain_reaches(
    BasicBlock *from, BasicBlock *to) noexcept {
    luisa::unordered_set<BasicBlock *> visited;
    auto *block = from;
    while (block != nullptr && visited.emplace(block).second) {
        if (block == to) { return true; }
        if (!block->is_terminated() ||
            !block->terminator()->isa<BranchInst>()) {
            return false;
        }
        block = static_cast<BranchInst *>(
                    block->terminator())
                    ->target_block();
    }
    return false;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "local_switch_funnels_preserve_incremental_relation_version"_test = [] {
        enable_relation_oracle();
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *selector =
            kernel->create_value_argument(Type::of<uint32_t>());
        auto *def = kernel->definition();
        XIRBuilder builder;

        // The enclosing loop gives every Switch a non-empty lexical context.
        // The root If owns the shared continuation as a structured merge, so
        // nested Switches cannot merely adopt it and must form one-target exit
        // funnels. Each rewrite changes the Switch's lexical break target.
        builder.set_insertion_point(body);
        auto *loop = builder.simple_loop();
        auto *loop_body = loop->create_body_block();
        auto *loop_merge = loop->create_merge_block();
        builder.set_insertion_point(loop_body);
        auto *root = builder.if_(condition);
        auto *cursor = root->create_true_block();
        auto *root_false = root->create_false_block();
        auto *shared_continuation = root->create_merge_block();
        builder.set_insertion_point(root_false);
        builder.br(shared_continuation);

        struct Site {
            SwitchInst *selection;
            BasicBlock *exit_arm;
            BasicBlock *original_merge;
        };
        constexpr auto site_count = size_t{65u};
        luisa::vector<Site> sites;
        sites.reserve(site_count);
        for (auto i = size_t{0u}; i < site_count; ++i) {
            builder.set_insertion_point(cursor);
            auto *selection = builder.switch_(selector);
            auto *next = selection->create_default_block();
            auto *exit_arm = selection->create_case_block(
                static_cast<uint32_t>(i + 1u));
            auto *original_merge = selection->create_merge_block();
            builder.set_insertion_point(exit_arm);
            builder.br(shared_continuation);
            builder.set_insertion_point(original_merge);
            builder.unreachable_();
            sites.emplace_back(
                Site{selection, exit_arm, original_merge});
            cursor = next;
        }
        builder.set_insertion_point(cursor);
        builder.br(shared_continuation);
        builder.set_insertion_point(shared_continuation);
        builder.break_(loop_merge);
        builder.set_insertion_point(loop_merge);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = restructure_cfg_pass_run_on_function(kernel);
        expect(info.succeeded());
        expect(info.iteration_limit_count == 0u);
        expect(info.unstructured_branch_count == 0u);
        expect(info.selection_exit_cfg_invalidation_count >= site_count);
        expect(info.selection_exit_local_invalidation_count ==
               info.selection_exit_cfg_invalidation_count);
        expect(info.selection_exit_global_invalidation_count == 0u);
        expect(info.selection_exit_relation_incremental_update_count >=
               site_count);
        expect(info.selection_exit_boundary_analysis_count <
               info.selection_exit_relation_incremental_update_count);
        expect(info.selection_exit_site_query_count <= 6u * site_count);

        for (auto site : sites) {
            expect(site.selection->merge_block() != site.original_merge);
            expect(branch_chain_reaches(
                site.exit_arm, site.selection->merge_block()));
            expect(branch_chain_reaches(
                site.selection->merge_block(), shared_continuation));
        }
        expect(xir_verify_module(
                   &module,
                   {.require_unique_merge_blocks = true,
                    .require_canonical_break_continue_targets = true})
                   .succeeded());
    };

    return 0;
}

