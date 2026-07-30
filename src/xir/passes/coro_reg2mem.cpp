#include <luisa/ast/type.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_reg2mem.h>
#include <luisa/xir/passes/coro_split.h>
#include <luisa/xir/passes/reg2mem.h>

namespace luisa::compute::xir {

namespace detail {

[[nodiscard]] static bool has_coroutine_intrinsic(CallableFunction *func) noexcept {
    if (func == nullptr || func->definition() == nullptr) { return false; }
    bool found = false;
    func->traverse_instructions([&](Instruction *inst) noexcept {
        switch (inst->derived_instruction_tag()) {
            case DerivedInstructionTag::CORO_SUSPEND:
            case DerivedInstructionTag::CORO_RESUME:
            case DerivedInstructionTag::CORO_TERMINATE:
                found = true;
                break;
            default:
                break;
        }
    });
    return found;
}

static void run_on_callable(CallableFunction *func, CoroReg2MemInfo &info) noexcept {
    if (func == nullptr || func->definition() == nullptr) { return; }
    auto reg2mem_info = reg2mem_pass_run_on_function(func);
    info.lowered_phi_count += reg2mem_info.lowered_phi_count;
    info.lowered_cross_block_value_count += reg2mem_info.lowered_cross_block_value_count;
    info.callable_count++;
}

}// namespace detail

CoroReg2MemInfo coro_reg2mem_pass_run_on_module(Module *m) noexcept {
    CoroReg2MemInfo info;
    if (m == nullptr) { return info; }
    for (auto *f : m->function_list()) {
        if (!f->isa<CallableFunction>() || f->definition() == nullptr) { continue; }
        auto *cf = static_cast<CallableFunction *>(f);
        if (!detail::has_coroutine_intrinsic(cf)) { continue; }

        detail::run_on_callable(cf, info);
    }

    return info;
}

CoroReg2MemInfo coro_reg2mem_pass_run_on_split(const CoroSplitInfo &split) noexcept {
    CoroReg2MemInfo info;
    for (auto &subroutine : split.subroutines) {
        detail::run_on_callable(subroutine.callable, info);
    }

    return info;
}

}// namespace luisa::compute::xir
