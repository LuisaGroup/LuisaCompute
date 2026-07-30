#include <luisa/core/logging.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/passes/trace_gep.h>
#include <luisa/xir/passes/transpose_gep.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

static void trace_gep_chain(Instruction *inst, luisa::fixed_vector<Value *, 16> &chain) noexcept {
    switch (inst->derived_instruction_tag()) {
        case DerivedInstructionTag::ALLOCA: {
            LUISA_DEBUG_ASSERT(!chain.empty(), "Invalid GEP chain.");
            chain.emplace_back(inst);
            break;
        }
        case DerivedInstructionTag::GEP: {
            auto gep_inst = static_cast<GEPInst *>(inst);
            auto index_uses = gep_inst->index_uses();
            for (auto it = index_uses.rbegin(); it != index_uses.rend(); ++it) {
                LUISA_DEBUG_ASSERT((*it)->value() != nullptr, "Invalid GEP index.");
                chain.emplace_back((*it)->value());
            }
            auto base = gep_inst->base();
            LUISA_DEBUG_ASSERT(base->isa<Instruction>(), "Invalid GEP base.");
            trace_gep_chain(static_cast<Instruction *>(base), chain);
            break;
        }
        default: LUISA_ERROR_WITH_LOCATION("Invalid GEP.");
    }
}

[[nodiscard]] static auto trace_gep_chain(Instruction *inst) noexcept {
    luisa::fixed_vector<Value *, 16> gep_chain;
    trace_gep_chain(inst, gep_chain);
    std::reverse(gep_chain.begin(), gep_chain.end());
    return gep_chain;
}

// Check if transposition should be skipped for large arrays with dynamic indexing.
// Transposition converts O(1) GEP+load/store into O(N) load-insert/extract-store,
// which is a regression for large arrays with dynamically-computed indices.
[[nodiscard]] static bool should_skip_transpose(const luisa::fixed_vector<Value *, 16> &gep_chain) noexcept {
    auto alloca_inst = static_cast<AllocaInst *>(gep_chain.front());
    auto alloca_type = alloca_inst->type();
    // Only skip for array types larger than the threshold (16 elements).
    // Arrays ≤16 have a fast path in the SPIR-V codegen using OpSelect chains.
    if (!alloca_type->is_array() || alloca_type->dimension() <= 16u) { return false; }
    // Check if any GEP index is non-constant (dynamic).
    for (size_t i = 1u; i < gep_chain.size(); ++i) {
        if (!gep_chain[i]->isa<Constant>()) { return true; }
    }
    return false;
}

// Load(GEP(agg, indices...)) => Extract(Load(agg), indices...)
static void transpose_load_gep(LoadInst *load, TransposeGEPInfo &info) noexcept {
    LUISA_DEBUG_ASSERT(load->variable()->isa<Instruction>(), "Invalid pointer.");
    auto gep_chain = trace_gep_chain(static_cast<Instruction *>(load->variable()));
    // Skip transposition for large arrays with dynamic indexing.
    if (should_skip_transpose(gep_chain)) { return; }
    XIRBuilder b;
    b.set_insertion_point(load);
    auto alloca_inst = gep_chain.front();
    auto alloca_load = b.load(alloca_inst->type(), alloca_inst);
    gep_chain[0] = alloca_load;
    auto extract = b.call(load->type(), ArithmeticOp::EXTRACT, gep_chain);
    for (auto *metadata : load->metadata_list()) {
        extract->metadata_list().push_front(metadata->clone());
    }
    load->replace_all_uses_with(extract);
    load->remove_self();
    info.transposed_load_count++;
}

// Store(GEP(agg, indices...), elem) => Store(agg, Insert(Load(agg), elem, indices...))
static void transpose_store_gep(StoreInst *store, TransposeGEPInfo &info) noexcept {
    LUISA_DEBUG_ASSERT(store->variable()->isa<Instruction>(), "Invalid pointer.");
    auto gep_chain = trace_gep_chain(static_cast<Instruction *>(store->variable()));
    // Skip transposition for large arrays with dynamic indexing.
    if (should_skip_transpose(gep_chain)) { return; }
    XIRBuilder b;
    b.set_insertion_point(store);
    auto alloca_inst = gep_chain.front();
    auto alloca_load = b.load(alloca_inst->type(), alloca_inst);
    gep_chain[0] = alloca_load;
    gep_chain.insert(gep_chain.begin() + 1, store->value());
    auto insert = b.call(alloca_inst->type(), ArithmeticOp::INSERT, gep_chain);
    auto *replacement = b.store(alloca_inst, insert);
    for (auto *metadata : store->metadata_list()) {
        replacement->metadata_list().push_front(metadata->clone());
    }
    store->remove_self();
    info.transposed_store_count++;
}

static void run_transpose_gep_pass_on_function(Function *function, TransposeGEPInfo &info) noexcept {
    if (function == nullptr) { return; }
    if (auto def = function->definition()) {
        if (def->body_block() == nullptr) { return; }
        // run the trace gep pass first to ensure that no nested GEP chains exist
        auto trace_gep_info = trace_gep_pass_run_on_function(def);
        info.traced_gep_count += trace_gep_info.traced_gep_count;
        info.removed_noop_gep_count += trace_gep_info.removed_noop_gep_count;
        if (trace_gep_info.changed()) {
            LUISA_VERBOSE(
                "Traced {} GEP chain(s) and removed {} no-op GEP(s) in "
                "transpose_gep pass.",
                trace_gep_info.traced_gep_count,
                trace_gep_info.removed_noop_gep_count);
        }
        // run the pass
        luisa::vector<GEPInst *> geps;
        {
            luisa::unordered_set<AllocaInst *> non_applicable_allocas;
            def->traverse_instructions([&](Instruction *inst) noexcept {
                switch (inst->derived_instruction_tag()) {
                    case DerivedInstructionTag::ALLOCA: [[fallthrough]];
                    case DerivedInstructionTag::LOAD: [[fallthrough]];
                    case DerivedInstructionTag::STORE: break;
                    case DerivedInstructionTag::GEP: {
                        if (auto gep = static_cast<GEPInst *>(inst); gep->index_count() != 0) {
                            geps.emplace_back(gep);
                        }
                        break;
                    }
                    default: {
                        for (auto &&op_use : inst->operand_uses()) {
                            if (auto op = op_use->value()) {
                                if (auto base = trace_pointer_base_local_alloca_inst(op)) {
                                    non_applicable_allocas.emplace(base);
                                }
                            }
                        }
                        break;
                    }
                }
            });
            geps.erase(std::remove_if(geps.begin(), geps.end(), [&](GEPInst *gep) noexcept {
                           auto base = trace_pointer_base_local_alloca_inst(gep->base());
                           if (base == nullptr || non_applicable_allocas.contains(base)) { return true; }
                           // Removing an annotated address computation has no
                           // single replacement when it feeds multiple memory
                           // operations.
                           if (!gep->metadata_list().empty()) { return true; }
                           // Skip GEPs targeting large arrays with dynamic indices:
                           // transposition would create O(N) full-array copy per element.
                           auto base_type = base->type();
                           if (base_type->is_array() && base_type->dimension() > 16u) {
                               for (size_t idx_i = 0u; idx_i < gep->index_count(); ++idx_i) {
                                   if (!gep->index(idx_i)->isa<Constant>()) { return true; }
                               }
                           }
                           return false;
                       }),
                       geps.end());
        }
        luisa::fixed_vector<LoadInst *, 64> gep_loads;
        luisa::fixed_vector<StoreInst *, 64> gep_stores;
        for (auto gep : geps) {
            gep_loads.clear();
            gep_stores.clear();
            for (auto &&use : gep->use_list()) {
                if (auto user = use->user()) {
                    LUISA_DEBUG_ASSERT(user->isa<Instruction>(), "Invalid user.");
                    switch (static_cast<Instruction *>(user)->derived_instruction_tag()) {
                        case DerivedInstructionTag::LOAD: {
                            gep_loads.emplace_back(static_cast<LoadInst *>(user));
                            break;
                        }
                        case DerivedInstructionTag::STORE: {
                            gep_stores.emplace_back(static_cast<StoreInst *>(user));
                            break;
                        }
                        default: break;
                    }
                }
            }
            for (auto load : gep_loads) { transpose_load_gep(load, info); }
            for (auto store : gep_stores) { transpose_store_gep(store, info); }
            LUISA_DEBUG_ASSERT(gep->use_list().empty(), "Unexpected users of transposed GEP.");
            gep->remove_self();
            info.removed_gep_count++;
        }
    }
}

}// namespace detail

TransposeGEPInfo transpose_gep_pass_run_on_function(Function *function) noexcept {
    TransposeGEPInfo info;
    detail::run_transpose_gep_pass_on_function(function, info);
    return info;
}

TransposeGEPInfo transpose_gep_pass_run_on_module(Module *module) noexcept {
    TransposeGEPInfo info;
    if (module == nullptr) { return info; }
    for (auto f : module->function_list()) {
        detail::run_transpose_gep_pass_on_function(f, info);
    }
    return info;
}

}// namespace luisa::compute::xir
