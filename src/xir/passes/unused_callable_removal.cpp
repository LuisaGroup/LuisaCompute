#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/unused_callable_removal.h>
#include <luisa/xir/passes/pass_pipeline.h>

namespace luisa::compute::xir {

namespace detail {

static void collect_reachable_callables(Function *f, luisa::unordered_set<Function *> &reachable) noexcept {
    if (reachable.emplace(f).second) {
        if (auto def = f->definition()) {
            def->traverse_instructions([&](Instruction *inst) noexcept {
                for (auto &&op_use : inst->operand_uses()) {
                    if (auto op = op_use->value(); op != nullptr && op->isa<Function>()) {
                        collect_reachable_callables(static_cast<Function *>(op), reachable);
                    }
                }
            });
        }
    }
}

}// namespace detail

UnusedCallableRemovalInfo unused_callable_removal_pass_run_on_module(Module *module, PassReport *report) noexcept {
    luisa::unordered_set<Function *> reachable;
    for (auto f : module->function_list()) {
        if (f->isa<KernelFunction>()) {
            detail::collect_reachable_callables(f, reachable);
        }
    }
    luisa::vector<Function *> removable;
    for (auto f : module->function_list()) {
        if (f->isa<CallableFunction>() && !reachable.contains(f)) {
            removable.emplace_back(f);
        }
    }
    for (auto f : removable) { f->remove_self(); }
    if (report != nullptr) {
        report->set("removed_callable", removable.size());
    }
    return {.removed_callable_count = removable.size()};
}

}// namespace luisa::compute::xir
