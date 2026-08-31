// Regressions for construct entries that are boundaries of enclosing loops.

#include "ut/ut.hpp"

#include <luisa/ast/type_registry.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/store.h>
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

    "nested_selection_does_not_clone_enclosing_loop_update"_test = [] {
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
            {.mutation_mode =
                 RestructureCFGMutationMode::IN_PLACE_DISCARDABLE});
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
    };

    return 0;
}
