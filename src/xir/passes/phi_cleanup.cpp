#include <luisa/xir/passes/phi_cleanup.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/phi.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

static void run_phi_cleanup_on_function(Function *function, PhiCleanupInfo &info) noexcept {
    if (function == nullptr || !function->is_definition()) return;
    auto def = function->definition();
    if (def == nullptr || def->body_block() == nullptr) return;
    auto dom_tree = compute_dom_tree(function);
    bool changed;
    do {
        changed = false;
        luisa::vector<PhiInst *> phis;
        def->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<PhiInst>()) phis.emplace_back(static_cast<PhiInst *>(inst));
        });
        for (auto phi : phis) {
            if (simplify_phi_instruction(phi, &dom_tree)) {
                info.removed_phi_count++;
                changed = true;
            }
        }
    } while (changed);
}

}// namespace detail

PhiCleanupInfo phi_cleanup_pass_run_on_function(Function *function) noexcept {
    PhiCleanupInfo info;
    detail::run_phi_cleanup_on_function(function, info);
    return info;
}

PhiCleanupInfo phi_cleanup_pass_run_on_module(Module *module, PassReport *report) noexcept {
    PhiCleanupInfo info;
    for (auto f : module->function_list()) {
        detail::run_phi_cleanup_on_function(f, info);
    }
    if (report != nullptr) {
        report->set("removed_phi", info.removed_phi_count);
    }
    return info;
}

}// namespace luisa::compute::xir
