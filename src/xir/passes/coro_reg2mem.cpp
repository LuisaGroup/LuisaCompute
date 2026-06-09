#include <luisa/ast/type.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_reg2mem.h>
#include <luisa/xir/passes/reg2mem.h>

namespace luisa::compute::xir {

namespace detail {

[[nodiscard]] static Value *get_frame_arg(CallableFunction *func) noexcept {
    for (auto *arg : func->arguments()) {
        if (arg->is_reference()) { return arg; }
    }
    return nullptr;
}

}// namespace detail

CoroReg2MemInfo coro_reg2mem_pass_run_on_module(Module *m) noexcept {
    CoroReg2MemInfo info;

    for (auto *f : m->function_list()) {
        if (!f->isa<CallableFunction>() || f->definition() == nullptr) { continue; }
        auto *cf = static_cast<CallableFunction *>(f);
        if (detail::get_frame_arg(cf) == nullptr) { continue; }

        auto reg2mem_info = reg2mem_pass_run_on_function(cf);
        info.lowered_phi_count += reg2mem_info.lowered_phi_count;
        info.lowered_cross_block_value_count += reg2mem_info.lowered_cross_block_value_count;
        info.callable_count++;
    }

    return info;
}

}// namespace luisa::compute::xir
