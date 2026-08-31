// Scale regressions for structured-construct exit discovery.

#include "ut/ut.hpp"

#include <luisa/ast/type_registry.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/return.h>
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

    "construct_exit_enumeration_walks_sparse_region_support"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *entry = kernel->create_body_block();
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *outer = builder.if_(condition);
        auto *outer_true = outer->create_true_block();
        auto *outer_false = outer->create_false_block();
        auto *outer_merge = outer->create_merge_block();

        constexpr auto nested_selection_count = size_t{256u};
        builder.set_insertion_point(outer_true);
        for (auto i = size_t{0u};
             i < nested_selection_count; ++i) {
            auto *inner = builder.if_(condition);
            auto *inner_true = inner->create_true_block();
            auto *inner_false = inner->create_false_block();
            auto *inner_merge = inner->create_merge_block();
            builder.set_insertion_point(inner_true);
            builder.br(inner_merge);
            builder.set_insertion_point(inner_false);
            builder.br(inner_merge);
            builder.set_insertion_point(inner_merge);
        }
        builder.br(outer_merge);
        builder.set_insertion_point(outer_false);
        builder.br(outer_merge);
        builder.set_insertion_point(outer_merge);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = restructure_cfg_pass_run_on_function(
            kernel,
            {.mutation_mode =
                 RestructureCFGMutationMode::IN_PLACE_DISCARDABLE});

        expect(info.succeeded());
        expect(!info.changed());
        expect(info.construct_exit_boundary_analysis_count == 1u);
        // Each inner diamond contributes its header and two arm blocks. Its
        // merge is the region boundary, and the outer construct has no parent
        // to inspect. Unrelated function blocks are never membership-probed.
        expect(info.construct_exit_region_block_visit_count ==
               3u * nested_selection_count);
        expect(info.construct_exit_region_edge_visit_count ==
               4u * nested_selection_count);
        expect(info.construct_exit_region_membership_query_count ==
               info.construct_exit_region_edge_visit_count)
            << "exit enumeration may query membership only for successors "
               "of blocks in the sparse construct region";
    };

    return 0;
}
