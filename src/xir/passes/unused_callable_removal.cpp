#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/metadata/signature_constraint.h>
#include <luisa/xir/passes/unused_callable_removal.h>
#include <luisa/xir/passes/pass_pipeline.h>

namespace luisa::compute::xir {

namespace detail {

static void collect_reachable_callables(Function *f, luisa::unordered_set<Function *> &reachable) noexcept {
    if (reachable.emplace(f).second) {
        if (auto def = f->definition()) {
            // Inspect every owned block, not only CFG-reachable blocks. This
            // pass does not delete disconnected blocks, so a function operand
            // in one of them must remain valid after the pass returns.
            for (auto *block : def->basic_blocks()) {
                for (auto *inst : block->instructions()) {
                    for (auto &&op_use : inst->operand_uses()) {
                        if (auto op = op_use->value(); op != nullptr && op->isa<Function>()) {
                            collect_reachable_callables(static_cast<Function *>(op), reachable);
                        }
                    }
                }
            }
        }
    }
}

}// namespace detail

UnusedCallableRemovalInfo unused_callable_removal_pass_run_on_module(Module *module, PassReport *report) noexcept {
    luisa::unordered_set<Function *> reachable;
    if (module != nullptr) {
        for (auto f : module->function_list()) {
            if (f->isa<KernelFunction>()) {
                detail::collect_reachable_callables(f, reachable);
            }
        }
    }
    luisa::unordered_set<Function *> removable;
    if (module != nullptr) {
        for (auto f : module->function_list()) {
            if (!f->isa<CallableFunction>() || reachable.contains(f)) {
                continue;
            }
            auto *callable = static_cast<CallableFunction *>(f);
            // A bodyless callable is declaration-like: without an explicit
            // internal-linkage contract it may be implemented or referenced
            // outside this module. Signature-constrained callables likewise
            // have an externally fixed ABI (for example callback entry
            // points). Neither is locally provable dead.
            if (callable->body_block() == nullptr ||
                callable->find_metadata<SignatureConstraintMD>() != nullptr) {
                continue;
            }
            removable.emplace(callable);
        }
    }
    // Destroy callers before callees so removing a caller first drops its
    // callee uses. An unreachable recursive SCC has no use-free root; keep it
    // conservatively instead of destroying a Function with live operands.
    size_t removed_count = 0u;
    for (;;) {
        Function *next = nullptr;
        for (auto *f : removable) {
            if (f->use_list().empty()) {
                next = f;
                break;
            }
        }
        if (next == nullptr) { break; }
        removable.erase(next);
        next->remove_self();
        ++removed_count;
    }
    if (report != nullptr) {
        report->set("removed_callable", removed_count);
    }
    return {.removed_callable_count = removed_count};
}

}// namespace luisa::compute::xir
