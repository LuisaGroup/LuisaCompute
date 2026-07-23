#include <luisa/core/logging.h>
#include <luisa/xir/undefined.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/metadata/reg2mem_spill.h>

#include <atomic>

#include "helpers.h"

namespace luisa::compute::xir {

Value *trace_pointer_base_value(Value *pointer) noexcept {
    return pointer != nullptr && pointer->isa<GEPInst>() ?
               trace_pointer_base_value(static_cast<GEPInst *>(pointer)->base()) :
               pointer;
}

AllocaInst *trace_pointer_base_local_alloca_inst(Value *pointer) noexcept {
    if (auto base = trace_pointer_base_value(pointer);
        base != nullptr && base->isa<AllocaInst>() &&
        static_cast<AllocaInst *>(base)->op() == AllocaOp::LOCAL) {
        return static_cast<AllocaInst *>(base);
    }
    return nullptr;
}

InstructionMemoryInfo get_memory_info(Instruction *inst) noexcept {
    auto pointer_scope = [](Value *pointer) noexcept {
        auto base = trace_pointer_base_value(pointer);
        if (base != nullptr && base->isa<AllocaInst>() &&
            static_cast<AllocaInst *>(base)->is_shared()) {
            return MemoryScope::SHARED;
        }
        return MemoryScope::LOCAL;
    };
    switch (inst->derived_instruction_tag()) {
        case DerivedInstructionTag::ARITHMETIC:
        case DerivedInstructionTag::CAST:
        case DerivedInstructionTag::GEP:
        case DerivedInstructionTag::PHI:
            return {MemoryScope::NONE, MemoryEffects::NONE, false};
        case DerivedInstructionTag::CLOCK:
            return {MemoryScope::GLOBAL, MemoryEffects::READ, false};
        case DerivedInstructionTag::RESOURCE_QUERY:
            return {MemoryScope::GLOBAL, MemoryEffects::NONE, false};
        case DerivedInstructionTag::RAY_QUERY_OBJECT_READ:
            return {MemoryScope::LOCAL, MemoryEffects::READ, false};
        case DerivedInstructionTag::ALLOCA: {
            auto alloca = static_cast<AllocaInst *>(inst);
            return {alloca->is_shared() ? MemoryScope::SHARED : MemoryScope::LOCAL,
                    MemoryEffects::NONE, false};
        }
        case DerivedInstructionTag::LOAD: {
            auto load = static_cast<LoadInst *>(inst);
            return {pointer_scope(load->variable()), MemoryEffects::READ, false};
        }
        case DerivedInstructionTag::STORE: {
            auto store = static_cast<StoreInst *>(inst);
            return {pointer_scope(store->variable()), MemoryEffects::WRITE, false};
        }
        case DerivedInstructionTag::RESOURCE_READ: {
            auto read = static_cast<ResourceReadInst *>(inst);
            auto is_volatile = read->op() == ResourceReadOp::BUFFER_VOLATILE_READ ||
                               read->op() == ResourceReadOp::BYTE_BUFFER_VOLATILE_READ;
            return {MemoryScope::GLOBAL, MemoryEffects::READ, is_volatile};
        }
        case DerivedInstructionTag::RESOURCE_WRITE: {
            auto write = static_cast<ResourceWriteInst *>(inst);
            auto is_volatile = write->op() == ResourceWriteOp::BUFFER_VOLATILE_WRITE ||
                               write->op() == ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE;
            return {MemoryScope::GLOBAL, MemoryEffects::WRITE, is_volatile};
        }
        case DerivedInstructionTag::ATOMIC: {
            auto atomic = static_cast<AtomicInst *>(inst);
            auto base = trace_pointer_base_value(atomic->base());
            auto scope = base != nullptr && base->isa<AllocaInst>() ?
                             pointer_scope(base) :
                             MemoryScope::GLOBAL;
            return {scope, MemoryEffects::READ_WRITE, false};
        }
        case DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE:
            return {MemoryScope::LOCAL, MemoryEffects::WRITE, false};
        case DerivedInstructionTag::THREAD_GROUP:
            return {MemoryScope::SHARED, MemoryEffects::READ_WRITE, true};
        case DerivedInstructionTag::CALL:
            return {MemoryScope::GLOBAL, MemoryEffects::READ_WRITE, false};
        case DerivedInstructionTag::PRINT:
        case DerivedInstructionTag::DEBUG_BREAK:
        case DerivedInstructionTag::ASSERT:
        case DerivedInstructionTag::ASSUME:
            return {MemoryScope::NONE, MemoryEffects::NONE, true};
        case DerivedInstructionTag::AUTODIFF_SCOPE:
        case DerivedInstructionTag::AUTODIFF_INTRINSIC:
            return {MemoryScope::GLOBAL, MemoryEffects::READ_WRITE, true};
        default:
            return {MemoryScope::NONE, MemoryEffects::NONE, true};
    }
}

bool contains_structured_control_flow(FunctionDefinition *function) noexcept {
    if (function == nullptr) { return false; }
    // Inspect every block owned by the function, not just blocks reachable
    // from the body. A temporarily unreachable structured region is still a
    // hard boundary for CFG-only transforms and may become reachable again.
    for (auto *block : function->basic_blocks()) {
        for (auto *inst : block->instructions()) {
            switch (inst->derived_instruction_tag()) {
                case DerivedInstructionTag::IF:
                case DerivedInstructionTag::SWITCH:
                case DerivedInstructionTag::LOOP:
                case DerivedInstructionTag::SIMPLE_LOOP:
                case DerivedInstructionTag::BREAK:
                case DerivedInstructionTag::CONTINUE:
                case DerivedInstructionTag::RAY_QUERY_LOOP:
                case DerivedInstructionTag::RAY_QUERY_DISPATCH:
                case DerivedInstructionTag::AUTODIFF_SCOPE:
                case DerivedInstructionTag::OUTLINE:
                    return true;
                default: break;
            }
        }
    }
    return false;
}

bool remove_redundant_phi_instruction(PhiInst *phi) noexcept {
    if (phi->use_list().empty()) {
        phi->remove_self();
        return true;
    }
    static constexpr auto is_invariant = [](Value *v) noexcept {
        if (v == nullptr) { return true; }
        switch (v->derived_value_tag()) {
            case DerivedValueTag::UNDEFINED: [[fallthrough]];
            case DerivedValueTag::CONSTANT: [[fallthrough]];
            case DerivedValueTag::ARGUMENT: [[fallthrough]];
            case DerivedValueTag::SPECIAL_REGISTER: return true;
            default: break;
        }
        return false;
    };
    auto all_same_except_undef = true;
    auto undef_incoming = static_cast<Value *>(nullptr);
    auto same_incoming = static_cast<Value *>(nullptr);
    // check if all incoming values are the same
    for (auto value_use : phi->incoming_value_uses()) {
        LUISA_DEBUG_ASSERT(value_use->value() != nullptr, "Invalid incoming value.");
        if (auto value = value_use->value(); value->isa<Undefined>()) {
            undef_incoming = value;
        } else {
            // if we haven't seen any incoming value yet, set it as the same incoming
            if (same_incoming == nullptr) { same_incoming = value; }
            // otherwise, check if the current incoming value is the same as the previous one
            if (same_incoming != value) {
                all_same_except_undef = false;
                break;
            }
        }
    }
    if (all_same_except_undef && is_invariant(same_incoming)) {
        if (same_incoming != nullptr) {
            phi->replace_all_uses_with(same_incoming);
        } else if (undef_incoming != nullptr) {
            phi->replace_all_uses_with(undef_incoming);
        } else {
            LUISA_DEBUG_ASSERT(phi->use_list().empty(), "Invalid phi node.");
        }
        phi->remove_self();
        return true;
    }
    return false;
}

bool simplify_phi_instruction(PhiInst *phi, const DomTree *dom_tree) noexcept {
    if (phi->use_list().empty()) {
        phi->remove_self();
        return true;
    }
    // Find the unique non-self, non-undef incoming value (if any).
    Value *unique = nullptr;
    for (auto value_use : phi->incoming_value_uses()) {
        auto v = value_use->value();
        if (v == nullptr || v == phi) continue;
        if (v->isa<Undefined>()) continue;
        if (unique == nullptr) {
            unique = v;
        } else if (unique != v) {
            return false;
        }
    }
    if (unique == nullptr) {
        if (auto m = phi->parent_function() ? phi->parent_function()->parent_module() : nullptr) {
            auto undef = m->create_undefined(phi->type());
            phi->replace_all_uses_with(undef);
        }
        phi->remove_self();
        return true;
    }
    // The unique value must dominate the phi's block (and thus all its uses).
    // Loop-carried phis are a classic counter-example: a phi in the loop header
    // may have an undef initial value and a single back-edge value V, but V is
    // defined later in the loop body and does not dominate the header. Replacing
    // the phi with V would create a self-referential cycle across iterations.
    auto dominates_phi = [&]() noexcept -> bool {
        // Constants, function arguments, and special registers dominate everything.
        if (unique->isa<Constant>() || unique->isa<Argument>() || unique->isa<SpecialRegister>()) { return true; }
        auto unique_inst = unique->isa<Instruction>() ? static_cast<Instruction *>(unique) : nullptr;
        if (unique_inst == nullptr) { return false; }
        auto unique_block = unique_inst->parent_block();
        auto phi_block = phi->parent_block();
        if (unique_block == nullptr || phi_block == nullptr) { return false; }
        if (unique_block == phi_block) { return false; }// phis are at the block start
        if (dom_tree != nullptr) { return dom_tree->dominates(unique_block, phi_block); }
        // Without a dominator tree, be conservative and only allow values that
        // are guaranteed to dominate everything (constants/arguments handled above).
        return false;
    };
    if (!dominates_phi()) { return false; }
    phi->replace_all_uses_with(unique);
    phi->remove_self();
    return true;
}

void lower_phi_node_to_local_variable(PhiInst *phi) noexcept {
    if (!simplify_phi_instruction(phi)) {
        auto f = phi->parent_function();
        LUISA_DEBUG_ASSERT(f != nullptr && f->definition() != nullptr, "Invalid function.");
        XIRBuilder b;
        // create alloca at the beginning of the function
        b.set_insertion_point(f->definition()->body_block()->instructions().head_sentinel());
        auto phi_alloca = b.alloca_local(phi->type());
        static_cast<void>(phi_alloca->create_metadata<Reg2MemSpillMD>());
        phi_alloca->add_comment("alloca to lower phi node");
        static std::atomic_uint64_t phi_counter{0u};
        auto phi_id = phi_counter.fetch_add(1u, std::memory_order_relaxed) + 1u;
        phi_alloca->set_name(luisa::format("_phi_{}", phi_id));
        if (auto m = f->parent_module()) {
            auto undef = m->create_undefined(phi->type());
            b.store(phi_alloca, undef);
        }
        // store incoming values at the end of their respective blocks
        for (size_t i = 0; i < phi->incoming_count(); i++) {
            if (auto incoming = phi->incoming(i); incoming.value != nullptr && !incoming.value->isa<Undefined>()) {
                LUISA_DEBUG_ASSERT(incoming.block != nullptr, "Invalid incoming block.");
                b.set_insertion_point(incoming.block->terminator()->prev());
                b.store(phi_alloca, incoming.value);
            }
        }
        // replace phi uses with local load instructions
        b.set_insertion_point(phi);
        auto phi_load = b.load(phi->type(), phi_alloca);
        phi_load->add_comment("load from phi alloca");
        phi->replace_all_uses_with(phi_load);
        phi->remove_self();
    }
}

void hoist_alloca_instructions_to_entry_block(FunctionDefinition *f) noexcept {
    luisa::vector<AllocaInst *> collected;
    f->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<AllocaInst>()) {
            collected.emplace_back(static_cast<AllocaInst *>(inst));
        }
    });
    if (!collected.empty()) {
        XIRBuilder b;
        b.set_insertion_point(f->body_block()->instructions().head_sentinel());
        for (auto inst : collected) {
            b.append(inst->remove_self());
        }
    }
}

}// namespace luisa::compute::xir
