#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/optional.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/instructions/atomic.h>
#include <luisa/xir/passes/alias_analysis.h>
#include <luisa/xir/passes/pass_pipeline.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

static thread_local luisa::unordered_map<Instruction *, AllocaInst *> s_inst_to_base_alloca;

static luisa::optional<int64_t> try_get_constant_int_value(Value *v) noexcept {
    if (v->isa<Constant>()) {
        auto c = static_cast<Constant *>(v);
        auto type = c->type();
        auto size = type->size();
        if (size == 4) {
            return static_cast<int64_t>(*static_cast<const int32_t *>(c->data()));
        } else if (size == 8) {
            return *static_cast<const int64_t *>(c->data());
        }
    }
    return luisa::nullopt;
}

static AllocaInst *get_base_alloca(Instruction *inst) noexcept {
    auto it = s_inst_to_base_alloca.find(inst);
    if (it != s_inst_to_base_alloca.end()) {
        return it->second;
    }
    Value *ptr = nullptr;
    switch (inst->derived_instruction_tag()) {
        case DerivedInstructionTag::LOAD:
            ptr = static_cast<LoadInst *>(inst)->variable();
            break;
        case DerivedInstructionTag::STORE:
            ptr = static_cast<StoreInst *>(inst)->variable();
            break;
        default:
            return nullptr;
    }
    auto base = trace_pointer_base_local_alloca_inst(ptr);
    if (base != nullptr) {
        s_inst_to_base_alloca[inst] = base;
    }
    return base;
}

static Value *get_local_pointer(Instruction *inst) noexcept {
    switch (inst->derived_instruction_tag()) {
        case DerivedInstructionTag::LOAD:
            return static_cast<LoadInst *>(inst)->variable();
        case DerivedInstructionTag::STORE:
            return static_cast<StoreInst *>(inst)->variable();
        default:
            return nullptr;
    }
}

static Value *get_resource_handle(Instruction *inst) noexcept {
    switch (inst->derived_instruction_tag()) {
        case DerivedInstructionTag::RESOURCE_READ:
        case DerivedInstructionTag::RESOURCE_WRITE:
            return inst->operand(0);
        case DerivedInstructionTag::ATOMIC:
            return static_cast<AtomicInst *>(inst)->base();
        default:
            return nullptr;
    }
}

static AliasResult alias_gep_offsets(GEPInst *gep_a, GEPInst *gep_b) noexcept {
    if (gep_a->index_count() == 0 || gep_b->index_count() == 0) {
        return AliasResult::MayAlias;
    }
    auto off_a = try_get_constant_int_value(gep_a->index(0));
    auto off_b = try_get_constant_int_value(gep_b->index(0));
    if (off_a.has_value() && off_b.has_value()) {
        if (*off_a != *off_b) {
            return AliasResult::NoAlias;
        }
    }
    return AliasResult::MayAlias;
}

static size_t get_global_index_count(Instruction *inst) noexcept {
    switch (inst->derived_instruction_tag()) {
        case DerivedInstructionTag::ATOMIC:
            return static_cast<AtomicInst *>(inst)->index_count();
        case DerivedInstructionTag::RESOURCE_READ:
            return inst->operand_count() > 0 ? inst->operand_count() - 1 : 0;
        case DerivedInstructionTag::RESOURCE_WRITE:
            return inst->operand_count() > 1 ? inst->operand_count() - 2 : 0;
        default:
            return 0;
    }
}

static Value *get_global_index(Instruction *inst, size_t i) noexcept {
    switch (inst->derived_instruction_tag()) {
        case DerivedInstructionTag::ATOMIC:
            return static_cast<AtomicInst *>(inst)->index_uses()[i]->value();
        case DerivedInstructionTag::RESOURCE_READ:
        case DerivedInstructionTag::RESOURCE_WRITE:
            return inst->operand(1 + i);
        default:
            return nullptr;
    }
}

static AliasResult alias_global_indices(Instruction *a, Instruction *b) noexcept {
    auto a_count = get_global_index_count(a);
    auto b_count = get_global_index_count(b);
    if (a_count != b_count) return AliasResult::MayAlias;
    for (size_t i = 0; i < a_count; i++) {
        auto idx_a = get_global_index(a, i);
        auto idx_b = get_global_index(b, i);
        if (idx_a == nullptr || idx_b == nullptr) return AliasResult::MayAlias;
        auto const_a = try_get_constant_int_value(idx_a);
        auto const_b = try_get_constant_int_value(idx_b);
        if (const_a.has_value() && const_b.has_value()) {
            if (*const_a != *const_b) return AliasResult::NoAlias;
        } else {
            return AliasResult::MayAlias;
        }
    }
    return AliasResult::MayAlias;
}

} // namespace detail

AliasAnalysisInfo alias_analysis_pass_run_on_function(FunctionDefinition *def) noexcept {
    AliasAnalysisInfo info;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        switch (inst->derived_instruction_tag()) {
            case DerivedInstructionTag::LOAD: {
                auto load = static_cast<LoadInst *>(inst);
                if (auto base = trace_pointer_base_local_alloca_inst(load->variable())) {
                    detail::s_inst_to_base_alloca[inst] = base;
                }
                break;
            }
            case DerivedInstructionTag::STORE: {
                auto store = static_cast<StoreInst *>(inst);
                if (auto base = trace_pointer_base_local_alloca_inst(store->variable())) {
                    detail::s_inst_to_base_alloca[inst] = base;
                }
                break;
            }
            default:
                break;
        }
    });
    return info;
}

AliasAnalysisInfo alias_analysis_pass_run_on_module(Module *module, PassReport *report) noexcept {
    detail::s_inst_to_base_alloca.clear();
    AliasAnalysisInfo info;
    for (auto f : module->function_list()) {
        if (auto def = f->definition()) {
            auto func_info = alias_analysis_pass_run_on_function(def);
            info.queried_count += func_info.queried_count;
        }
    }
    if (report != nullptr) {
        report->set("alias_analysis_queried", info.queried_count);
    }
    return info;
}

AliasResult alias_analysis_query(Instruction *a, Instruction *b) noexcept {
    if (a == b) return AliasResult::MustAlias;

    auto mem_a = get_memory_info(a);
    auto mem_b = get_memory_info(b);

    if (mem_a.is_pure() || mem_b.is_pure()) {
        return AliasResult::NoAlias;
    }

    if (mem_a.scope != mem_b.scope) {
        return AliasResult::NoAlias;
    }

    if (mem_a.scope == MemoryScope::SHARED) {
        return AliasResult::MustAlias;
    }

    if (mem_a.scope == MemoryScope::LOCAL) {
        auto base_a = detail::get_base_alloca(a);
        auto base_b = detail::get_base_alloca(b);
        if (base_a == nullptr || base_b == nullptr) {
            return AliasResult::MayAlias;
        }
        if (base_a != base_b) {
            return AliasResult::NoAlias;
        }
        auto ptr_a = detail::get_local_pointer(a);
        auto ptr_b = detail::get_local_pointer(b);
        if (ptr_a != nullptr && ptr_b != nullptr &&
            ptr_a->isa<GEPInst>() && ptr_b->isa<GEPInst>()) {
            return detail::alias_gep_offsets(
                static_cast<GEPInst *>(ptr_a),
                static_cast<GEPInst *>(ptr_b));
        }
        return AliasResult::MayAlias;
    }

    if (mem_a.scope == MemoryScope::GLOBAL) {
        auto handle_a = detail::get_resource_handle(a);
        auto handle_b = detail::get_resource_handle(b);
        if (handle_a == nullptr || handle_b == nullptr) {
            return AliasResult::MayAlias;
        }
        if (handle_a != handle_b) {
            return AliasResult::NoAlias;
        }
        return detail::alias_global_indices(a, b);
    }

    return AliasResult::MayAlias;
}

} // namespace luisa::compute::xir
