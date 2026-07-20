#include <luisa/core/stl/format.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>
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
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<PhiInst>()) { phi_nodes.emplace_back(static_cast<PhiInst *>(inst)); }
    });
    for (auto phi : phi_nodes) {
        lower_phi_node_to_local_variable(phi);
    }
    info.lowered_phi_count += phi_nodes.size();
}

static void lower_cross_block_uses_in_function(FunctionDefinition *def, Reg2MemInfo &info) noexcept {
    // Compute dominator tree to avoid unnecessary alloca creation.
    // We only need to lower values to allocas when the defining block
    // does NOT dominate a cross-block use — otherwise SSA dominance
    // guarantees the value is already available at the use site.
    auto dom_tree = compute_dom_tree(static_cast<Function *>(def));

    luisa::vector<Instruction *> candidates;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->is_terminator()) { return; }
        if (inst->type() == nullptr) { return; }
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
            auto use_block = user_inst->parent_block();
            if (use_block == nullptr) { continue; }
            if (!dom_tree.contains(use_block)) { continue; }
            if (use_block != def_block && !dom_tree.dominates(def_block, use_block)) {
                candidates.emplace_back(inst);
                break;
            }
        }
    });
    if (candidates.empty()) { return; }
    XIRBuilder b;
    auto entry_head = def->body_block()->instructions().head_sentinel();
    for (auto inst : candidates) {
        auto def_block = inst->parent_block();
        b.set_insertion_point(entry_head);
        auto slot = b.alloca_local(inst->type());
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
            auto use_block = user_inst->parent_block();
            if (use_block == nullptr) { continue; }
            if (!dom_tree.contains(use_block)) { continue; }
            if (use_block != def_block && !dom_tree.dominates(def_block, use_block)) {
                cross_block_uses.emplace_back(use);
            }
        }
        for (auto use : cross_block_uses) {
            auto user_inst = static_cast<Instruction *>(static_cast<Value *>(use->user()));
            b.set_insertion_point(user_inst->prev());
            auto reload = b.load(inst->type(), slot);
            reload->add_comment("load from cross-block alloca");
            User::set_operand_use_value(use, reload);
        }
        info.lowered_cross_block_value_count += 1;
    }
}

static void hoist_allocas_to_top_of_funtion(FunctionDefinition *def) noexcept {
    luisa::vector<AllocaInst *> allocas;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<AllocaInst>()) {
            allocas.emplace_back(static_cast<AllocaInst *>(inst));
        }
    });
    XIRBuilder b;
    b.set_insertion_point(def->body_block()->instructions().head_sentinel());
    for (auto a : allocas) { b.append(a->remove_self()); }
}

static void run_reg2mem_pass_on_function(Function *function, Reg2MemInfo &info) noexcept {
    if (auto definition = function->definition()) {
        lower_phi_nodes_in_function(definition, info);
        lower_cross_block_uses_in_function(definition, info);
        hoist_allocas_to_top_of_funtion(definition);
    }
}

}// namespace detail

Reg2MemInfo reg2mem_pass_run_on_function(Function *function) noexcept {
    Reg2MemInfo info;
    detail::run_reg2mem_pass_on_function(function, info);
    return info;
}

Reg2MemInfo reg2mem_pass_run_on_module(Module *module, PassReport *report) noexcept {
    Reg2MemInfo info;
    for (auto f : module->function_list()) {
        detail::run_reg2mem_pass_on_function(f, info);
    }
    if (report != nullptr) {
        report->set("lowered_phi", info.lowered_phi_count);
        report->set("lowered_cross_block_value", info.lowered_cross_block_value_count);
    }
    return info;
}

}// namespace luisa::compute::xir
