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

// Load(GEP(agg, indices...)) => Extract(Load(agg), indices...)
static void transpose_load_gep(LoadInst *load, TransposeGEPInfo &info) noexcept {
    LUISA_DEBUG_ASSERT(load->variable()->isa<Instruction>(), "Invalid pointer.");
    auto gep_chain = trace_gep_chain(static_cast<Instruction *>(load->variable()));
    XIRBuilder b;
    b.set_insertion_point(load);
    auto alloca_inst = gep_chain.front();
    auto alloca_load = b.load(alloca_inst->type(), alloca_inst);
    gep_chain[0] = alloca_load;
    auto extract = b.call(load->type(), ArithmeticOp::EXTRACT, gep_chain);
    load->replace_all_uses_with(extract);
    load->remove_self();
    info.transposed_load_count++;
}

// Store(GEP(agg, indices...), elem) => Store(agg, Insert(Load(agg), elem, indices...))
static void transpose_store_gep(StoreInst *store, TransposeGEPInfo &info) noexcept {
    LUISA_DEBUG_ASSERT(store->variable()->isa<Instruction>(), "Invalid pointer.");
    auto gep_chain = trace_gep_chain(static_cast<Instruction *>(store->variable()));
    XIRBuilder b;
    b.set_insertion_point(store);
    auto alloca_inst = gep_chain.front();
    auto alloca_load = b.load(alloca_inst->type(), alloca_inst);
    gep_chain[0] = alloca_load;
    gep_chain.insert(gep_chain.begin() + 1, store->value());
    auto insert = b.call(alloca_inst->type(), ArithmeticOp::INSERT, gep_chain);
    (void)b.store(alloca_inst, insert);
    store->remove_self();
    info.transposed_store_count++;
}

static void run_transpose_gep_pass_on_function(Function *function, TransposeGEPInfo &info) noexcept {
    if (auto def = function->definition()) {
        // run the trace gep pass first to ensure that no nested GEP chains exist
        if (auto trace_gep_info = trace_gep_pass_run_on_function(def); trace_gep_info.traced_gep_count != 0) {
            LUISA_VERBOSE("Traced {} GEP chain(s) in transpose_gep pass.", trace_gep_info.traced_gep_count);
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
                           return base == nullptr || non_applicable_allocas.contains(base);
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
    for (auto f : module->function_list()) {
        detail::run_transpose_gep_pass_on_function(f, info);
    }
    return info;
}

}// namespace luisa::compute::xir
