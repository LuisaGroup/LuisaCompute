#include <luisa/xir/passes/dead_arg_elim.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/metadata/signature_constraint.h>

namespace luisa::compute::xir {

namespace detail {

static void dead_arg_elim_pass_on_function_def(FunctionDefinition *def, DeadArgElimInfo &info) noexcept {
    if (def == nullptr || def->body_block() == nullptr) { return; }
    // Skip entry-point functions: their argument positions are part of the
    // backend ABI even when an argument is unused by the function body.
    if (def->derived_function_tag() == DerivedFunctionTag::KERNEL ||
        def->derived_function_tag() == DerivedFunctionTag::RASTER_STAGE) {
        return;
    }
    // Signature-constrained functions and functions referenced by anything
    // other than an ordinary call have an externally fixed ABI (ray-query
    // callbacks are the important example).
    if (def->find_metadata<SignatureConstraintMD>() != nullptr) { return; }
    luisa::vector<CallInst *> call_sites;
    for (auto *use : def->use_list()) {
        auto *user = use->user();
        if (user == nullptr || !user->isa<CallInst>()) { return; }
        auto *call = static_cast<CallInst *>(user);
        // A Function is itself a Value. A malformed or partially constructed
        // module can therefore mention `def` as an ordinary argument of an
        // unrelated call. Such a use is not a call site of `def`; treating it
        // as one would remove an arbitrary argument from the unrelated call.
        // Reject the whole function before changing either signature.
        if (call->callee() != def ||
            use != call->operand_use(CallInst::operand_index_callee)) {
            return;
        }
        if (call->argument_count() != def->arguments().count_size()) { return; }
        call_sites.emplace_back(call);
    }

    // Collect indices of unused parameters (those with no uses within the function body).
    luisa::vector<size_t> unused_indices;
    {
        size_t idx = 0;
        for (auto arg : def->arguments()) {
            // Removing an argument changes both the function signature and
            // every call-site operand list. There is no replacement value
            // that can uniquely own argument-local metadata, so annotated
            // arguments are part of the preserved ABI even when unused.
            if (arg->use_list().empty() &&
                arg->metadata_list().empty()) {
                unused_indices.push_back(idx);
            }
            idx++;
        }
    }

    if (unused_indices.empty()) { return; }

    // Process in reverse order so that earlier indices remain valid after removal.
    for (auto it = unused_indices.rbegin(); it != unused_indices.rend(); ++it) {
        size_t idx = *it;

        // All call sites were validated before the first mutation.
        for (auto *call : call_sites) {
            call->remove_argument(idx);
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
    if (module != nullptr) {
        for (auto f : module->function_list()) {
            if (auto def = f->definition()) {
                detail::dead_arg_elim_pass_on_function_def(def, info);
            }
        }
    }
    if (report != nullptr) {
        report->set("removed_arg", info.removed_arg_count);
    }
    return info;
}

}// namespace luisa::compute::xir
