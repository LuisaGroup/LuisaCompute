#include <luisa/core/logging.h>
#include <luisa/xir/undefined.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>

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
        phi_alloca->add_comment("alloca to lower phi node");
        static int phi_counter = 0;
        phi_alloca->set_name(luisa::format("_phi_{}", ++phi_counter));
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
