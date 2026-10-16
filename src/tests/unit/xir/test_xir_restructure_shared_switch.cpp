#include "ut/ut.hpp"

#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/indexed_branch.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/verifier.h>

#include <array>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

int main() {
    "parallel_switch_labels_are_one_case_construct_not_cross_entries"_test = [] {
        for (auto *type : {Type::of<int32_t>(), Type::of<uint32_t>(),
                           Type::of<int64_t>(), Type::of<uint64_t>()}) {
            for (auto structured : {false, true}) {
                for (auto alias_default : {false, true}) {
                    for (auto labels : {1u, 2u, 3u}) {
                        Module module;
                        auto *kernel = module.create_kernel();
                        auto *selector = kernel->create_value_argument(type);
                        auto *output = kernel->create_resource_argument(Type::buffer(Type::of<uint>()));
                        auto *header = kernel->create_body_block();
                        auto *shared = kernel->create_basic_block();
                        auto *single = kernel->create_basic_block();
                        auto *fallback = alias_default ? shared : kernel->create_basic_block();
                        auto *merge = kernel->create_basic_block();
                        XIRBuilder builder;
                        builder.set_insertion_point(header);
                        auto *branch = [&]() -> IndexedBranchTerminatorInstruction * {
                            if (structured) {
                                auto *sw = builder.switch_(selector);
                                sw->set_merge_block(merge);
                                return sw;
                            }
                            return builder.indexed_branch(selector);
                        }();
                        branch->set_default_block(fallback);
                        for (auto i = 0u; i < labels; i++) { branch->add_case(i + 10u, shared); }
                        branch->add_case(3u, single);
                        const std::array blocks{shared, single, fallback};
                        for (auto i = 0u; i < (alias_default ? 2u : 3u); i++) {
                            builder.set_insertion_point(blocks[i]);
                            auto value = 17u + i;
                            builder.call(ResourceWriteOp::BUFFER_WRITE,
                                         {output, module.create_constant_zero(Type::of<uint>()),
                                          module.create_constant(Type::of<uint>(), &value)});
                            builder.br(merge);
                        }
                        builder.set_insertion_point(merge);
                        builder.return_void();
                        expect(xir_verify_module(&module).succeeded());
                        auto info = restructure_cfg_pass_run_on_function(kernel);
                        expect(info.succeeded()) << "labels=" << labels << " default_alias=" << alias_default
                                                 << " structured=" << structured << " type=" << type->description();
                        if (!info.succeeded()) { continue; }
                        expect(xir_verify_module(&module, {.require_no_unstructured_control_flow = true,
                                                           .require_unique_merge_blocks = true})
                                   .succeeded());
                        expect(header->terminator()->isa<SwitchInst>());
                        if (!header->terminator()->isa<SwitchInst>()) { continue; }
                        auto *sw = static_cast<SwitchInst *>(header->terminator());
                        expect(sw->case_count() == labels + 1u);
                        for (auto i = 0u; i < labels; i++) {
                            expect(sw->case_value(i) == i + 10u);
                            expect(sw->case_block(i) == sw->case_block(0u));
                        }
                        if (alias_default) { expect(sw->default_block() == sw->case_block(0u)); }
                        auto writes = 0u;
                        auto loops = 0u;
                        kernel->traverse_instructions([&](const Instruction *inst) {
                            writes += inst->isa<ResourceWriteInst>();
                            loops += inst->isa<LoopInst>() || inst->isa<SimpleLoopInst>();
                        });
                        expect(writes == (alias_default ? 2u : 3u));
                        expect(loops == 0u) << "Parallel label edges must not generate a cycle";
                    }
                }
            }
        }
    };
}
