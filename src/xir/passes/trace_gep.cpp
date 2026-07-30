#include <luisa/xir/module.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/passes/trace_gep.h>

namespace luisa::compute::xir {

namespace detail {

[[nodiscard]] static Value *collect_gep_indices_recursive(GEPInst *inst, luisa::vector<Value *> &indices) noexcept {
    auto origin = inst->base();
    if (origin->isa<GEPInst>()) {
        auto *base_gep = static_cast<GEPInst *>(origin);
        // Chain compression mutates the outer GEP in place but can leave an
        // internal GEP dead. An annotated internal address has no replacement
        // metadata owner, so it is a hard tracing boundary.
        if (base_gep->metadata_list().empty()) {
            origin = collect_gep_indices_recursive(base_gep, indices);
        }
    }
    for (auto i : inst->index_uses()) {
        indices.emplace_back(i->value());
    }
    return origin;
}

[[nodiscard]] static bool try_trace_gep_inst(GEPInst *inst) noexcept {
    if (!inst->base()->isa<GEPInst>()) { return false; }
    if (!static_cast<GEPInst *>(inst->base())
             ->metadata_list()
             .empty()) {
        return false;
    }
    luisa::vector<Value *> indices;
    auto origin = collect_gep_indices_recursive(inst, indices);
    inst->set_operand_count(1 + indices.size());
    inst->set_operand(0, origin);
    for (size_t i = 0; i < indices.size(); i++) {
        inst->set_operand(i + 1, indices[i]);
    }
    return true;
}

static void trace_gep_instructions_in_function(Function *function, TraceGEPInfo &info) noexcept {
    if (function == nullptr) { return; }
    if (auto definition = function->definition()) {
        if (definition->body_block() == nullptr) { return; }
        luisa::vector<GEPInst *> geps;
        definition->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<GEPInst>()) {
                geps.emplace_back(static_cast<GEPInst *>(inst));
            }
        });
        for (auto gep : geps) {
            if (try_trace_gep_inst(gep)) { info.traced_gep_count++; }
            if (gep->index_count() == 0 &&
                gep->metadata_list().empty()) {
                gep->replace_all_uses_with(gep->base());
                gep->remove_self();
                info.removed_noop_gep_count++;
            }
        }
    }
}

}// namespace detail

TraceGEPInfo trace_gep_pass_run_on_function(Function *function) noexcept {
    TraceGEPInfo info;
    detail::trace_gep_instructions_in_function(function, info);
    return info;
}

TraceGEPInfo trace_gep_pass_run_on_module(Module *module) noexcept {
    TraceGEPInfo info;
    if (module == nullptr) { return info; }
    for (auto f : module->function_list()) {
        detail::trace_gep_instructions_in_function(f, info);
    }
    return info;
}

}// namespace luisa::compute::xir
