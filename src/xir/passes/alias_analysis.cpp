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
    Value *ptr = nullptr;
    switch (inst->derived_instruction_tag()) {
        case DerivedInstructionTag::LOAD:
            ptr = static_cast<LoadInst *>(inst)->variable();
            break;
        case DerivedInstructionTag::STORE:
            ptr = static_cast<StoreInst *>(inst)->variable();
            break;
        case DerivedInstructionTag::ATOMIC:
            ptr = static_cast<AtomicInst *>(inst)->base();
            break;
        default:
            return nullptr;
    }
    // Derive this from the current operand graph on every query. XIR passes can
    // retarget GEPs after alias analysis has run, and a process-global cache
    // keyed only by Instruction * otherwise returns stale (or recycled) data.
    auto base = trace_pointer_base_value(ptr);
    return base != nullptr && base->isa<AllocaInst>() ?
               static_cast<AllocaInst *>(base) :
               nullptr;
}

static Value *get_local_pointer(Instruction *inst) noexcept {
    switch (inst->derived_instruction_tag()) {
        case DerivedInstructionTag::LOAD:
            return static_cast<LoadInst *>(inst)->variable();
        case DerivedInstructionTag::STORE:
            return static_cast<StoreInst *>(inst)->variable();
        case DerivedInstructionTag::ATOMIC:
            return static_cast<AtomicInst *>(inst)->base();
        default:
            return nullptr;
    }
}

static AtomicInst *get_indexed_atomic(Instruction *inst) noexcept {
    if (inst->isa<AtomicInst>()) {
        auto atomic = static_cast<AtomicInst *>(inst);
        if (atomic->index_count() != 0u) { return atomic; }
    }
    return nullptr;
}

static AliasResult alias_atomic_indices(AtomicInst *a, AtomicInst *b) noexcept {
    if (a->base() != b->base() || a->index_count() != b->index_count()) {
        return AliasResult::MayAlias;
    }
    auto all_equal = true;
    for (auto i = 0u; i < a->index_count(); i++) {
        auto index_a = a->index_uses()[i]->value();
        auto index_b = b->index_uses()[i]->value();
        if (index_a == nullptr || index_b == nullptr) {
            all_equal = false;
            continue;
        }
        if (index_a == index_b) { continue; }
        auto constant_a = try_get_constant_int_value(index_a);
        auto constant_b = try_get_constant_int_value(index_b);
        if (constant_a.has_value() && constant_b.has_value()) {
            if (*constant_a != *constant_b) { return AliasResult::NoAlias; }
        } else {
            all_equal = false;
        }
    }
    return all_equal ? AliasResult::MustAlias : AliasResult::MayAlias;
}

static bool is_byte_addressed_access(Instruction *inst) noexcept {
    if (inst->isa<ResourceReadInst>()) {
        switch (static_cast<ResourceReadInst *>(inst)->op()) {
            case ResourceReadOp::BYTE_BUFFER_READ:
            case ResourceReadOp::BYTE_BUFFER_VOLATILE_READ:
            case ResourceReadOp::BINDLESS_BYTE_BUFFER_READ:
            case ResourceReadOp::DEVICE_ADDRESS_READ:
                return true;
            default: break;
        }
    } else if (inst->isa<ResourceWriteInst>()) {
        switch (static_cast<ResourceWriteInst *>(inst)->op()) {
            case ResourceWriteOp::BYTE_BUFFER_WRITE:
            case ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE:
            case ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE:
            case ResourceWriteOp::DEVICE_ADDRESS_WRITE:
                return true;
            default: break;
        }
    }
    return false;
}

static bool is_bindless_access(Instruction *inst) noexcept {
    if (inst->isa<ResourceReadInst>()) {
        switch (static_cast<ResourceReadInst *>(inst)->op()) {
            case ResourceReadOp::BINDLESS_BUFFER_READ:
            case ResourceReadOp::BINDLESS_BYTE_BUFFER_READ:
            case ResourceReadOp::BINDLESS_TEXTURE2D_READ:
            case ResourceReadOp::BINDLESS_TEXTURE3D_READ:
            case ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL:
            case ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL:
                return true;
            default: break;
        }
    } else if (inst->isa<ResourceWriteInst>()) {
        switch (static_cast<ResourceWriteInst *>(inst)->op()) {
            case ResourceWriteOp::BINDLESS_BUFFER_WRITE:
            case ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE:
                return true;
            default: break;
        }
    }
    return false;
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
    // Offsets are comparable only when they are relative to the exact same
    // immediate base. For nested GEPs, index(0) belongs to a different aggregate
    // level and a numeric mismatch does not prove disjointness.
    if (gep_a->base() != gep_b->base()) { return AliasResult::MayAlias; }
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

}// namespace detail

AliasAnalysisInfo alias_analysis_pass_run_on_function(FunctionDefinition *def) noexcept {
    AliasAnalysisInfo info;
    static_cast<void>(def);
    return info;
}

AliasAnalysisInfo alias_analysis_pass_run_on_module(Module *module, PassReport *report) noexcept {
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
    if (a == nullptr || b == nullptr) { return AliasResult::MayAlias; }
    if (a == b) return AliasResult::MustAlias;

    auto mem_a = get_memory_info(a);
    auto mem_b = get_memory_info(b);

    if (mem_a.is_pure() || mem_b.is_pure()) {
        return AliasResult::NoAlias;
    }

    if (mem_a.scope != mem_b.scope) {
        return AliasResult::NoAlias;
    }

    if (mem_a.scope == MemoryScope::LOCAL || mem_a.scope == MemoryScope::SHARED) {
        auto base_a = detail::get_base_alloca(a);
        auto base_b = detail::get_base_alloca(b);
        if (base_a == nullptr || base_b == nullptr) {
            return AliasResult::MayAlias;
        }
        if (base_a != base_b) {
            return AliasResult::NoAlias;
        }
        auto atomic_a = detail::get_indexed_atomic(a);
        auto atomic_b = detail::get_indexed_atomic(b);
        if (atomic_a != nullptr || atomic_b != nullptr) {
            return atomic_a != nullptr && atomic_b != nullptr ?
                       detail::alias_atomic_indices(atomic_a, atomic_b) :
                       AliasResult::MayAlias;
        }
        auto ptr_a = detail::get_local_pointer(a);
        auto ptr_b = detail::get_local_pointer(b);
        if (ptr_a != nullptr && ptr_a == ptr_b) { return AliasResult::MustAlias; }
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
        // Distinct SSA handles are not a no-alias guarantee: two kernel
        // resource arguments may be bound to the same buffer or overlapping
        // views. Only compare indices once the handle itself is identical.
        if (handle_a != handle_b) { return AliasResult::MayAlias; }
        if (detail::is_bindless_access(a) || detail::is_bindless_access(b) ||
            detail::is_byte_addressed_access(a) || detail::is_byte_addressed_access(b)) {
            return AliasResult::MayAlias;
        }
        return detail::alias_global_indices(a, b);
    }

    return AliasResult::MayAlias;
}

}// namespace luisa::compute::xir
