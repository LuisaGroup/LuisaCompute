// A prepare backedge must not execute payload on a distinct update path.
// No CPU shader evaluation: this checks compiler SSA and CFG contracts only.
#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/verifier.h>
#include <array>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;

int main(int argc, char **argv) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "prepare_bypass_must_not_execute_update_payload"_test = [] {
        for (auto mutation : {RestructureCFGMutationMode::TRANSACTIONAL,
                              RestructureCFGMutationMode::IN_PLACE_DISCARDABLE}) {
            for (auto rotation : {0u, 1u, 4u}) {
                for (const auto *type : {Type::of<uint32_t>(), Type::of<float3>()}) {
                    Module module;
                    auto *kernel = module.create_kernel();
                    std::array<BasicBlock *, 6u> blocks{};
                    blocks[0] = kernel->create_body_block();
                    for (auto i = 1u; i < blocks.size(); ++i) {
                        blocks[1u + (i - 1u + rotation) % 5u] = kernel->create_basic_block();
                    }
                    auto [entry, prepare, body, define, update, merge] = blocks;
                    auto *repeat = kernel->create_value_argument(Type::of<bool>());
                    auto *skip = kernel->create_value_argument(Type::of<bool>());
                    auto *input = kernel->create_reference_argument(type);
                    auto *output = kernel->create_reference_argument(type);
                    XIRBuilder b;
                    b.set_insertion_point(entry);
                    auto *loop = b.loop();
                    loop->set_prepare_block(prepare);
                    loop->set_body_block(body);
                    loop->set_update_block(update);
                    loop->set_merge_block(merge);
                    b.set_insertion_point(prepare);
                    b.cond_br(repeat, body, merge);
                    b.set_insertion_point(body);
                    b.cond_br(skip, prepare, define);
                    b.set_insertion_point(define);
                    auto *value = b.load(type, input);
                    b.br(update);
                    b.set_insertion_point(update);
                    b.store(output, value);
                    b.br(prepare);
                    b.set_insertion_point(merge);
                    b.return_void();
                    expect(xir_verify_module(&module).succeeded());
                    const auto info = restructure_cfg_pass_run_on_function(
                        kernel, {.mutation_mode = mutation});
                    expect(info.succeeded());
                    expect(info.iteration_limit_count == 0u);
                    auto verification = xir_verify_module(
                        &module, {.require_no_phi = true,
                                  .require_unique_merge_blocks = true,
                                  .require_canonical_break_continue_targets = true});
                    expect(verification.succeeded()) <<
                        (verification.errors.empty() ? "" : verification.errors.front().message.c_str());
                    if (info.succeeded()) {
                        const auto again = restructure_cfg_pass_run_on_function(
                            kernel, {.mutation_mode = mutation});
                        expect(again.succeeded());
                        expect(!again.changed());
                    }
                }
            }
        }
    };
}
