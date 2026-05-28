#include <luisa/xir/passes/dead_arg_elim.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/call.h>

namespace luisa::compute::xir {

namespace detail {

static void dead_arg_elim_pass_on_function_def(FunctionDefinition *def, DeadArgElimInfo &info) noexcept {
    // Skip kernel (entry-point) functions: removing their arguments would break
    // the SPIR-V codegen's resource property index mapping which relies on
    // argument positions.
    if (def->derived_function_tag() == DerivedFunctionTag::KERNEL) { return; }

    // Collect indices of unused parameters (those with no uses within the function body).
    luisa::vector<size_t> unused_indices;
    {
        size_t idx = 0;
        for (auto arg : def->arguments()) {
            if (arg->use_list().empty()) {
                unused_indices.push_back(idx);
            }
            idx++;
        }
    }

    if (unused_indices.empty()) { return; }

    // Process in reverse order so that earlier indices remain valid after removal.
    for (auto it = unused_indices.rbegin(); it != unused_indices.rend(); ++it) {
        size_t idx = *it;

        // Remove the corresponding argument from every CallInst that targets this function.
        // The function's use_list tracks uses from CallInst callee operands.
        for (auto use : def->use_list()) {
            auto user = use->user();
            if (user->isa<CallInst>()) {
                static_cast<CallInst *>(user)->remove_argument(idx);
            }
        }

        // Remove the argument from the function definition's argument list.
        size_t cur = 0;
        for (auto arg : def->arguments()) {
            if (cur == idx) {
                arg->remove_self();
                break;
            }
            cur++;
        }

        info.removed_arg_count++;
    }
}

}// namespace detail

DeadArgElimInfo dead_arg_elim_pass_run_on_function(FunctionDefinition *def) noexcept {
    DeadArgElimInfo info;
    detail::dead_arg_elim_pass_on_function_def(def, info);
    return info;
}

DeadArgElimInfo dead_arg_elim_pass_run_on_module(Module *module, PassReport *report) noexcept {
    DeadArgElimInfo info;
    for (auto f : module->function_list()) {
        if (auto def = f->definition()) {
            detail::dead_arg_elim_pass_on_function_def(def, info);
        }
    }
    if (report != nullptr) {
        report->set("removed_arg", info.removed_arg_count);
    }
    return info;
}

}// namespace luisa::compute::xir
