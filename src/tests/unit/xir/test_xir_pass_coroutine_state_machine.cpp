#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/coroutine.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coroutine_split.h>
#include <luisa/xir/passes/coroutine_state_machine_scheduler.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] KernelFunction *make_flat_coroutine(Module &m) noexcept {
    auto *k = m.create_kernel();
    auto *body = k->create_body_block();
    XIRBuilder b;
    b.set_insertion_point(body);
    auto *a = b.alloca_local(Type::of<float>());
    b.coro_register(a, "a");
    b.coro_suspend(1u);
    b.coro_suspend(2u);
    b.load(Type::of<float>(), a);
    b.return_void();
    return k;
}

[[nodiscard]] size_t count_calls(FunctionDefinition *def) noexcept {
    size_t n = 0;
    def->traverse_instructions([&](const Instruction *inst) noexcept {
        if (inst->derived_instruction_tag() == DerivedInstructionTag::CALL) { ++n; }
    });
    return n;
}

[[nodiscard]] bool has_simple_loop_with_switch(FunctionDefinition *def) noexcept {
    auto found = false;
    def->traverse_instructions([&](const Instruction *inst) noexcept {
        if (inst->derived_instruction_tag() == DerivedInstructionTag::SIMPLE_LOOP) {
            auto loop = static_cast<const SimpleLoopInst *>(inst);
            if (auto body = loop->body_block(); body != nullptr) {
                for (auto bi : body->instructions()) {
                    if (bi->derived_instruction_tag() == DerivedInstructionTag::SWITCH) {
                        found = true;
                        return;
                    }
                }
            }
        }
    });
    return found;
}

}// namespace

void reg_coroutine_state_machine() {

    "state_machine_emit_rejects_unsupported_split"_test = [] {
        Module m;
        CoroutineSplitInfo bad;
        bad.is_supported = false;
        auto info = coroutine_state_machine_scheduler_emit(&m, bad);
        expect(!info.ok);
        expect(info.kernel == nullptr);
        expect(!info.diagnostics.empty());
    };

    "state_machine_emit_builds_dispatch_kernel"_test = [] {
        Module m;
        auto *coro = make_flat_coroutine(m);
        auto split = coroutine_split_run_on_function(coro);
        expect(split.is_supported);
        expect(split.continuations.size() == 3_u);

        auto info = coroutine_state_machine_scheduler_emit(&m, split);
        expect(info.ok);
        expect(info.kernel != nullptr);

        auto *def = info.kernel->definition();
        // Kernel calls the entry once + N-1 dispatched continuations from the
        // switch (one CallInst per case body) = N total CALL instructions.
        expect(count_calls(def) == split.continuations.size());
        // The dispatch is a simple_loop whose body contains a switch.
        expect(has_simple_loop_with_switch(def));
    };
}

int main() {
    reg_coroutine_state_machine();
    return 0;
}
