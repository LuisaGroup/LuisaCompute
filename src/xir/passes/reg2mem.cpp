#include <luisa/core/stl/format.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/metadata/reg2mem_spill.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/use.h>
#include <luisa/xir/user.h>
#include <luisa/xir/passes/reg2mem.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/passes/dom_tree.h>

#include <atomic>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

static void lower_phi_nodes_in_function(FunctionDefinition *def, Reg2MemInfo &info) noexcept {
    luisa::vector<PhiInst *> phi_nodes;
    // `FunctionDefinition::traverse_instructions` follows executable edges
    // from the body block. That is the right domain for dominance, but not for
    // this lowering contract: verifier options such as `require_no_phi` inspect
    // every block owned by the function, including disconnected CFG
    // components and unreachable structured merge roles. Collect from the
    // ownership list so a successful reg2mem run really eliminates every PHI.
    for (auto *block : def->basic_blocks()) {
        for (auto *inst : block->instructions()) {
            if (inst->isa<PhiInst>()) {
                phi_nodes.emplace_back(static_cast<PhiInst *>(inst));
            }
        }
    }
    for (auto phi : phi_nodes) {
        lower_phi_node_to_local_variable(phi);
    }
    info.lowered_phi_count += phi_nodes.size();
}

[[nodiscard]] static bool is_non_dominating_cross_block_use(
    Instruction *definition, Instruction *user,
    const DomTree &dom_tree) noexcept {
    // Phi operands are edge uses and obey incoming-edge dominance rather than
    // ordinary instruction-use dominance.
    if (user->isa<PhiInst>()) { return false; }
    auto def_block = definition->parent_block();
    auto use_block = user->parent_block();
    if (def_block == nullptr || use_block == nullptr) { return false; }
    if (!dom_tree.contains(def_block) || !dom_tree.contains(use_block)) {
        return false;
    }
    return use_block != def_block &&
           !dom_tree.dominates(def_block, use_block);
}

static void lower_cross_block_uses_in_function(FunctionDefinition *def, Reg2MemInfo &info) noexcept {
    // Compute dominator tree to avoid unnecessary alloca creation.
    // We only need to lower values to allocas when the defining block
    // does NOT dominate a cross-block use — otherwise SSA dominance
    // guarantees the value is already available at the use site.
    // Cross-block repair queries ancestry only. Dominance frontiers are a
    // separate derived relation and have no observer in this operation.
    auto dom_tree = compute_dom_tree(
        static_cast<Function *>(def),
        {.compute_dominance_frontiers = false});

    luisa::vector<Instruction *> candidates;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->is_terminator()) { return; }
        if (inst->type() == nullptr) { return; }
        if (inst->is_lvalue()) { return; }
        if (inst->isa<AllocaInst>()) { return; }
        if (inst->isa<PhiInst>()) { return; }
        auto def_block = inst->parent_block();
        if (def_block == nullptr) { return; }
        // Skip instructions in unreachable blocks (not in dom tree)
        if (!dom_tree.contains(def_block)) { return; }
        // Only consider this instruction as a candidate if it has at least
        // one cross-block use that is NOT dominated by the defining block.
        for (auto use : inst->use_list()) {
            auto u = use->user();
            if (u == nullptr) { continue; }
            auto u_val = static_cast<Value *>(u);
            if (!u_val->isa<Instruction>()) { continue; }
            auto user_inst = static_cast<Instruction *>(u_val);
            if (is_non_dominating_cross_block_use(
                    inst, user_inst, dom_tree)) {
                candidates.emplace_back(inst);
                break;
            }
        }
    });
    if (candidates.empty()) { return; }
    XIRBuilder b;
    auto entry_head = def->body_block()->instructions().head_sentinel();
    for (auto inst : candidates) {
        b.set_insertion_point(entry_head);
        auto slot = b.alloca_local(inst->type());
        slot->create_metadata<Reg2MemSpillMD>()->set_kind(
            Reg2MemSpillKind::CROSS_BLOCK);
        slot->add_comment("alloca to lower cross-block value");
        static std::atomic_uint64_t xblock_counter{0u};
        auto xblock_id = xblock_counter.fetch_add(1u, std::memory_order_relaxed) + 1u;
        slot->set_name(luisa::format("_xblock_{}", xblock_id));
        b.set_insertion_point(inst);
        b.store(slot, inst);
        luisa::vector<Use *> cross_block_uses;
        for (auto use : inst->use_list()) {
            auto u = use->user();
            if (u == nullptr) { continue; }
            auto u_val = static_cast<Value *>(u);
            if (!u_val->isa<Instruction>()) { continue; }
            auto user_inst = static_cast<Instruction *>(u_val);
            if (is_non_dominating_cross_block_use(
                    inst, user_inst, dom_tree)) {
                cross_block_uses.emplace_back(use);
            }
        }
        for (auto use : cross_block_uses) {
            auto user_inst = static_cast<Instruction *>(static_cast<Value *>(use->user()));
            b.set_insertion_point(user_inst->prev());
            auto reload = b.load(inst->type(), slot);
            User::set_operand_use_value(use, reload);
        }
        info.lowered_cross_block_value_count += 1;
    }
}

static void hoist_allocas_to_top_of_function(
    FunctionDefinition *def, Reg2MemInfo &info) noexcept {
    luisa::vector<AllocaInst *> allocas;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<AllocaInst>()) {
            allocas.emplace_back(static_cast<AllocaInst *>(inst));
        }
    });
    // Preserve traversal order and make the operation idempotent. In
    // particular, do not detach/reinsert an already-canonical entry prefix:
    // doing so is an IR mutation that used to go unreported by this pass.
    XIRBuilder b;
    auto *insertion_point =
        def->body_block()->instructions().head_sentinel();
    b.set_insertion_point(insertion_point);
    for (auto *alloca : allocas) {
        if (alloca->parent_block() == def->body_block() &&
            alloca->prev() == insertion_point) {
            insertion_point = alloca;
            b.set_insertion_point(insertion_point);
            continue;
        }
        insertion_point = b.append(alloca->remove_self());
        info.hoisted_alloca_count++;
    }
}

static void run_reg2mem_pass_on_function(Function *function, Reg2MemInfo &info) noexcept {
    if (function == nullptr) { return; }
    if (auto definition = function->definition();
        definition != nullptr && definition->body_block() != nullptr) {
        lower_phi_nodes_in_function(definition, info);
        lower_cross_block_uses_in_function(definition, info);
        hoist_allocas_to_top_of_function(definition, info);
    }
}

}// namespace detail

Reg2MemInfo reg2mem_pass_run_on_function(Function *function) noexcept {
    Reg2MemInfo info;
    if (function != nullptr) {
        detail::run_reg2mem_pass_on_function(function, info);
    }
    return info;
}

Reg2MemInfo reg2mem_pass_run_on_module(Module *module, PassReport *report) noexcept {
    Reg2MemInfo info;
    if (module != nullptr) {
        for (auto f : module->function_list()) {
            detail::run_reg2mem_pass_on_function(f, info);
        }
    }
    if (report != nullptr) {
        report->set("lowered_phi", info.lowered_phi_count);
        report->set("lowered_cross_block_value", info.lowered_cross_block_value_count);
        report->set("hoisted_alloca", info.hoisted_alloca_count);
    }
    return info;
}

Reg2MemInfo reg2mem_pass_repair_cross_block_rvalue_uses_on_function(
    Function *function) noexcept {
    Reg2MemInfo info;
    if (auto definition =
            function == nullptr ? nullptr : function->definition()) {
        detail::lower_cross_block_uses_in_function(definition, info);
    }
    return info;
}

namespace {

void audit_reg2mem_spill_metadata(
    const MetadataListMixin &owner, bool valid_alloca_owner,
    Reg2MemSpillAuditInfo &info) noexcept {
    for (auto metadata : owner.metadata_list()) {
        if (!metadata->isa<Reg2MemSpillMD>()) { continue; }
        auto spill = static_cast<const Reg2MemSpillMD *>(metadata);
        if (!valid_alloca_owner) {
            info.remaining_invalid_spill_count++;
            continue;
        }
        switch (spill->kind()) {
            case Reg2MemSpillKind::PHI:
                info.remaining_phi_spill_count++;
                break;
            case Reg2MemSpillKind::CROSS_BLOCK:
                info.remaining_cross_block_spill_count++;
                break;
            default:
                info.remaining_invalid_spill_count++;
                break;
        }
    }
}

}// namespace

Reg2MemSpillAuditInfo audit_reg2mem_spills_on_function(
    const Function *function) noexcept {
    Reg2MemSpillAuditInfo info;
    if (function == nullptr) { return info; }
    audit_reg2mem_spill_metadata(*function, false, info);
    for (auto argument : function->arguments()) {
        audit_reg2mem_spill_metadata(*argument, false, info);
    }
    for (auto block : function->basic_blocks()) {
        audit_reg2mem_spill_metadata(*block, false, info);
        for (auto instruction : block->instructions()) {
            audit_reg2mem_spill_metadata(
                *instruction, instruction->isa<AllocaInst>(), info);
        }
    }
    return info;
}

Reg2MemSpillAuditInfo audit_reg2mem_spills_on_module(
    const Module *module, PassReport *report) noexcept {
    Reg2MemSpillAuditInfo info;
    if (module == nullptr) {
        if (report != nullptr) {
            report->set("remaining_phi_spill", 0u);
            report->set("remaining_cross_block_spill", 0u);
            report->set("remaining_invalid_spill", 0u);
            report->set("remaining_spill", 0u);
        }
        return info;
    }
    audit_reg2mem_spill_metadata(*module, false, info);
    for (auto constant : module->constant_list()) {
        audit_reg2mem_spill_metadata(*constant, false, info);
    }
    for (auto undefined : module->undefined_list()) {
        audit_reg2mem_spill_metadata(*undefined, false, info);
    }
    for (auto special_register : module->special_register_list()) {
        audit_reg2mem_spill_metadata(*special_register, false, info);
    }
    for (auto function : module->function_list()) {
        auto function_info = audit_reg2mem_spills_on_function(function);
        info.remaining_phi_spill_count +=
            function_info.remaining_phi_spill_count;
        info.remaining_cross_block_spill_count +=
            function_info.remaining_cross_block_spill_count;
        info.remaining_invalid_spill_count +=
            function_info.remaining_invalid_spill_count;
    }
    if (report != nullptr) {
        report->set("remaining_phi_spill", info.remaining_phi_spill_count);
        report->set("remaining_cross_block_spill",
                    info.remaining_cross_block_spill_count);
        report->set("remaining_invalid_spill",
                    info.remaining_invalid_spill_count);
        report->set("remaining_spill", info.remaining_spill_count());
    }
    return info;
}

}// namespace luisa::compute::xir
