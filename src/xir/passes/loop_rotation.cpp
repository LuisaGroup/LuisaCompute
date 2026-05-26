#include <luisa/xir/passes/loop_rotation.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>

namespace luisa::compute::xir {

namespace detail {

static void rotate_loop(LoopInst *loop, LoopRotationInfo &info) noexcept {
    auto prepare = loop->prepare_block();
    auto body = loop->body_block();
    auto update = loop->update_block();
    auto merge = loop->merge_block();
    if (!prepare || !body || !update || !merge) return;

    // Prepare block must terminate with cond_br(cond, body, merge)
    auto prepare_term = prepare->terminator();
    if (!prepare_term || !prepare_term->isa<ConditionalBranchInst>()) return;
    auto cond_br_inst = static_cast<ConditionalBranchInst *>(prepare_term);
    if (cond_br_inst->true_block() != body || cond_br_inst->false_block() != merge) return;

    // Update block must terminate with br prepare
    auto update_term = update->terminator();
    if (!update_term || !update_term->isa<BranchInst>()) return;
    auto br_inst = static_cast<BranchInst *>(update_term);
    if (br_inst->target_block() != prepare) return;

    auto cond = cond_br_inst->condition();
    XIRBuilder builder;

    // Step 1: Collect loop-carried phi nodes in prepare.
    // These are phi nodes that have an incoming from the update block,
    // meaning they carry values across loop iterations.
    luisa::vector<PhiInst *> loop_carried_phis;
    for (auto inst : prepare->instructions()) {
        if (inst->is_terminator()) break;
        if (!inst->isa<PhiInst>()) continue;
        auto phi = static_cast<PhiInst *>(inst);
        for (size_t i = 0; i < phi->incoming_count(); ++i) {
            if (phi->incoming(i).block == update) {
                loop_carried_phis.push_back(phi);
                break;
            }
        }
    }

    // Step 2: Create new phi nodes in update for loop-carried values,
    // and set up the value mapping. Insert before the update terminator
    // so they land before all non-phi instructions (which are inserted
    // later, also before the terminator).
    luisa::unordered_map<PhiInst *, PhiInst *> old_to_new_phi;
    for (auto old_phi : loop_carried_phis) {
        auto new_phi = luisa::make_managed<PhiInst>(update, old_phi->type());
        new_phi->add_incoming(old_phi, prepare);
        auto *raw = new_phi.get();
        old_to_new_phi[old_phi] = raw;
        update_term->insert_before_self(std::move(new_phi));
    }

    // Step 3: Replace uses of old phis with new phis, and complete the
    // new phis' update-incoming edges.
    for (auto old_phi : loop_carried_phis) {
        auto new_phi = old_to_new_phi[old_phi];

        // Replace all uses of old_phi with new_phi. This also replaces
        // the prepare-incoming of new_phi (which we restore below).
        old_phi->replace_all_uses_with(new_phi);

        // Restore the prepare-incoming of new_phi to point back to old_phi.
        for (size_t i = 0; i < new_phi->incoming_count(); ++i) {
            if (new_phi->incoming(i).block == prepare) {
                new_phi->set_incoming(i, old_phi, prepare);
                break;
            }
        }

        // Add the update-incoming to new_phi using the now-updated value
        // from the old phi's update-incoming.
        for (size_t i = 0; i < old_phi->incoming_count(); ++i) {
            if (old_phi->incoming(i).block == update) {
                new_phi->add_incoming(old_phi->incoming(i).value, update);
                break;
            }
        }
    }

    // If the condition itself is a loop-carried phi, use the new phi instead.
    if (cond->isa<PhiInst>()) {
        auto cond_phi = static_cast<PhiInst *>(cond);
        auto it = old_to_new_phi.find(cond_phi);
        if (it != old_to_new_phi.end()) {
            cond = it->second;
        }
    }

    // Step 4: Move all non-phi, non-terminator instructions from prepare
    // to update. These compute the loop condition and must be re-evaluated
    // on each iteration. They are inserted before the update terminator,
    // after the new phi nodes and after any existing update instructions.
    luisa::vector<Instruction *> to_move;
    for (auto inst : prepare->instructions()) {
        if (!inst->is_terminator() && !inst->isa<PhiInst>()) {
            to_move.push_back(inst);
        }
    }
    for (auto inst : to_move) {
        auto m = inst->remove_self();
        update_term->insert_before_self(std::move(m));
    }

    // Step 5: Replace terminators.
    // Prepare becomes a pre-header: unconditional branch to body.
    cond_br_inst->remove_self();
    builder.set_insertion_point(prepare);
    builder.br(body);

    // Update becomes the new loop latch: conditional branch back to body
    // (or exit to merge) using the (possibly updated) condition.
    br_inst->remove_self();
    builder.set_insertion_point(update);
    builder.cond_br(cond, body, merge);

    info.rotated_loop_count++;
}

static void run(FunctionDefinition *def, LoopRotationInfo &info) noexcept {
    luisa::vector<LoopInst *> loops;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<LoopInst>())
            loops.push_back(static_cast<LoopInst *>(inst));
    });

    for (auto loop : loops) {
        rotate_loop(loop, info);
    }
}

}// namespace detail

LoopRotationInfo loop_rotation_pass_run_on_function(FunctionDefinition *def) noexcept {
    LoopRotationInfo info;
    detail::run(def, info);
    return info;
}

LoopRotationInfo loop_rotation_pass_run_on_module(Module *module, PassReport *report) noexcept {
    LoopRotationInfo info;
    for (auto f : module->function_list()) {
        auto def = f->definition();
        if (def) detail::run(def, info);
    }
    return info;
}

}// namespace luisa::compute::xir
