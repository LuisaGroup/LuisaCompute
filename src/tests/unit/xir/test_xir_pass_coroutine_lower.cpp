#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/coroutine.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coroutine.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

void reg_coroutine_lower() {

    "coroutine_lower_ignores_plain_kernel"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();

        auto info = coroutine_lower_run_on_function(k);
        expect(!info.changed);
        expect(info.created_state_alloca_count == 0u);
        expect(info.created_frame_alloca_count == 0u);
        expect(info.created_switch_count == 0u);
    };

    "coroutine_lower_builds_dispatcher_and_frame"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *local = b.alloca_local(Type::of<float>());
        auto *one = m.create_constant_one(Type::of<float>());
        b.store(local, one);
        b.coro_register(local, "local");
        b.coro_suspend(1u);
        auto *loaded = b.load(Type::of<float>(), local);
        static_cast<void>(loaded);
        b.return_void();

        auto lower = coroutine_lower_run_on_function(k);
        expect(lower.changed);
        expect(lower.removed_register_count == 1u);
        expect(lower.removed_suspend_count == 1u);
        expect(lower.created_state_alloca_count == 1u);
        expect(lower.created_frame_alloca_count == 1u);
        expect(lower.created_switch_count == 1u);

        size_t coro_count = 0u;
        size_t switch_count = 0u;
        size_t gep_store_count = 0u;
        size_t gep_load_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            switch (inst->derived_instruction_tag()) {
                case DerivedInstructionTag::CORO_REGISTER:
                case DerivedInstructionTag::CORO_SUSPEND: coro_count++; break;
                case DerivedInstructionTag::SWITCH: switch_count++; break;
                case DerivedInstructionTag::STORE: {
                    auto *store = static_cast<StoreInst *>(inst);
                    if (store->variable()->isa<GEPInst>()) { gep_store_count++; }
                    break;
                }
                case DerivedInstructionTag::LOAD: {
                    auto *load = static_cast<LoadInst *>(inst);
                    if (load->variable()->isa<GEPInst>()) { gep_load_count++; }
                    break;
                }
                default: break;
            }
        });
        expect(coro_count == 0u);
        expect(switch_count == 1u);
        expect(gep_store_count >= 1u);
        expect(gep_load_count >= 1u);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_coroutine_lower();
    return 0;
}
