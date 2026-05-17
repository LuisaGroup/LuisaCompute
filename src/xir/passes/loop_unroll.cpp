#include <luisa/xir/passes/loop_unroll.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/constant.h>
#include <luisa/core/logging.h>

namespace luisa::compute::xir {

namespace detail {

static constexpr size_t MAX_UNROLL_COUNT = 16;

// Analyze a counted loop to determine trip count. Returns 0 if not analyzable.
[[nodiscard]] static size_t analyze_trip_count(LoopInst *loop) noexcept {
    auto body = loop->body_block();
    auto update = loop->update_block();
    if (!body || !update) return 0;

    // Find the conditional branch in the body
    ConditionalBranchInst *cond_br = nullptr;
    for (auto inst : body->instructions()) {
        if (inst->isa<ConditionalBranchInst>()) {
            cond_br = static_cast<ConditionalBranchInst *>(inst);
            break;
        }
    }
    if (!cond_br) return 0;

    auto cond = cond_br->condition();
    if (!cond || !cond->isa<ArithmeticInst>()) return 0;
    auto cmp = static_cast<ArithmeticInst *>(cond);
    auto op = cmp->op();
    if (op != ArithmeticOp::BINARY_LESS && op != ArithmeticOp::BINARY_LESS_EQUAL) return 0;

    // Find constant bound and induction variable
    Value *induction = nullptr;
    Value *bound = nullptr;
    if (cmp->operand(1)->isa<Constant>()) {
        bound = cmp->operand(1);
        induction = cmp->operand(0);
    } else if (cmp->operand(0)->isa<Constant>()) {
        bound = cmp->operand(0);
        induction = cmp->operand(1);
    }
    if (!bound || !induction) return 0;

    auto bc = static_cast<Constant *>(bound);
    int64_t bound_val = 0;
    if (bc->type()->is_int32()) bound_val = bc->as<int32_t>();
    else if (bc->type()->is_uint32()) bound_val = static_cast<int64_t>(bc->as<uint32_t>());
    else return 0;

    // Find initial value and step from Phi
    int64_t start = 0;
    int64_t step = 1;
    if (induction->isa<PhiInst>()) {
        auto phi = static_cast<PhiInst *>(induction);
        for (size_t i = 0; i < phi->incoming_count(); ++i) {
            auto inc = phi->incoming(i);
            if (inc.block == update) {
                if (inc.value->isa<ArithmeticInst>()) {
                    auto add = static_cast<ArithmeticInst *>(inc.value);
                    if (add->op() == ArithmeticOp::BINARY_ADD &&
                        add->operand(0) == phi && add->operand(1)->isa<Constant>()) {
                        auto sc = static_cast<Constant *>(add->operand(1));
                        if (sc->type()->is_int32()) step = sc->as<int32_t>();
                        else if (sc->type()->is_uint32()) step = static_cast<int64_t>(sc->as<uint32_t>());
                    }
                }
            } else if (inc.block != body) {
                if (inc.value->isa<Constant>()) {
                    auto sc = static_cast<Constant *>(inc.value);
                    if (sc->type()->is_int32()) start = sc->as<int32_t>();
                    else if (sc->type()->is_uint32()) start = static_cast<int64_t>(sc->as<uint32_t>());
                }
            }
        }
    }

    if (step <= 0) return 0;
    int64_t trips = (bound_val - start + (op == ArithmeticOp::BINARY_LESS_EQUAL ? 1 : 0) + step - 1) / step;
    if (trips <= 0 || static_cast<size_t>(trips) > MAX_UNROLL_COUNT) return 0;
    return static_cast<size_t>(trips);
}

static void unroll(LoopInst *loop, size_t trips, LoopUnrollInfo &info) noexcept {
    auto func = loop->parent_function();
    if (!func) return;

    auto body = loop->body_block();
    auto update = loop->update_block();
    auto merge = loop->merge_block();
    auto prepare = loop->prepare_block();
    if (!body || !merge) return;

    // Simple unrolling: clone body blocks for each iteration
    // This is a simplified approach that creates a linear chain
    XIRBuilder builder;

    // Create a chain of cloned body blocks
    BasicBlock *prev = prepare;
    for (size_t i = 0; i < trips; ++i) {
        auto clone_bb = func->create_basic_block();

        // Branch from previous block
        if (prev && !prev->is_terminated()) {
            builder.set_insertion_point(prev);
            builder.br(clone_bb);
        }

        // Clone body instructions (non-terminators only)
        builder.set_insertion_point(clone_bb);
        luisa::unordered_map<Value *, Value *> vmap;
        for (auto inst : body->instructions()) {
            if (inst->is_terminator()) continue;

            if (inst->isa<AllocaInst>()) {
                auto a = static_cast<AllocaInst *>(inst);
                auto na = builder.alloca_(a->type(), a->op());
                vmap[inst] = na;
            } else if (inst->isa<LoadInst>()) {
                auto l = static_cast<LoadInst *>(inst);
                auto var = vmap.count(l->variable()) ? vmap[l->variable()] : l->variable();
                auto nl = builder.load(l->type(), var);
                vmap[inst] = nl;
            } else if (inst->isa<StoreInst>()) {
                auto s = static_cast<StoreInst *>(inst);
                auto var = vmap.count(s->variable()) ? vmap[s->variable()] : s->variable();
                auto val = vmap.count(s->value()) ? vmap[s->value()] : s->value();
                builder.store(var, val);
            } else if (inst->isa<ArithmeticInst>()) {
                auto a = static_cast<ArithmeticInst *>(inst);
                luisa::vector<Value *> ops;
                for (size_t j = 0; j < a->operand_count(); ++j) {
                    auto op = a->operand(j);
                    ops.push_back(vmap.count(op) ? vmap[op] : op);
                }
                auto na = builder.call(a->type(), a->op(), ops);
                vmap[inst] = na;
            } else if (inst->isa<PhiInst>()) {
                // Phi handled by value map from previous iteration
                continue;
            }
        }

        prev = clone_bb;
    }

    // Branch last cloned block to merge
    if (prev && !prev->is_terminated()) {
        builder.set_insertion_point(prev);
        builder.br(merge);
    }

    // Clean up: remove the loop and its update block
    if (update) {
        luisa::vector<Instruction *> to_remove;
        for (auto inst : update->instructions())
            if (!inst->is_terminator()) to_remove.push_back(inst);
        for (auto inst : to_remove) inst->remove_self();
    }

    loop->remove_self();
    info.unrolled_loop_count++;
}

static void run(Function *function, LoopUnrollInfo &info) noexcept {
    auto def = function->definition();
    if (!def) return;

    luisa::vector<LoopInst *> loops;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<LoopInst>())
            loops.push_back(static_cast<LoopInst *>(inst));
    });

    for (auto loop : loops) {
        auto trips = analyze_trip_count(loop);
        if (trips > 0) unroll(loop, trips, info);
    }
}

}// namespace detail

LoopUnrollInfo loop_unroll_pass_run_on_function(Function *function) noexcept {
    LoopUnrollInfo info;
    detail::run(function, info);
    return info;
}

LoopUnrollInfo loop_unroll_pass_run_on_module(Module *module) noexcept {
    LoopUnrollInfo info;
    for (auto f : module->function_list())
        detail::run(f, info);
    return info;
}

}// namespace luisa::compute::xir
