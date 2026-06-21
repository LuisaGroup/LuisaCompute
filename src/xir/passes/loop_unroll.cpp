#include <luisa/xir/passes/loop_unroll.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/constant.h>
#include <luisa/core/logging.h>

namespace luisa::compute::xir {

// Loop unroller: handles `for(i=start; i<bound; i+=step)` with constant bound.
// Enabled in Phase A pipeline (before SROA) and normalization pipeline.

namespace detail {

class LoopUnrollResolver final : public InstructionCloneValueResolver {
    luisa::unordered_map<const Value *, Value *> _map;

public:
    void insert_or_assign(const Value *from, Value *to) noexcept {
        _map.insert_or_assign(from, to);
    }
    [[nodiscard]] Value *resolve(const Value *value) noexcept override {
        if (value == nullptr) return nullptr;
        auto it = _map.find(value);
        if (it != _map.end()) return it->second;
        return const_cast<Value *>(value);
    }
};

[[nodiscard]] static size_t analyze_trip_count(LoopInst *loop, size_t max_trip_count) noexcept {
    auto prepare = loop->prepare_block();
    auto update = loop->update_block();
    if (!prepare || !update) return 0;

    ConditionalBranchInst *cond_br = nullptr;
    for (auto inst : prepare->instructions()) {
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

    Value *induction = nullptr;
    Value *bound = nullptr;
    if (cmp->operand(1)->isa<Constant>()) {
        // Pattern: induction < bound (constant on right) — expected form
        bound = cmp->operand(1);
        induction = cmp->operand(0);
    } else {
        // Constant on left means `bound < induction` which is a different
        // loop pattern (decrementing or inverted). Bail out rather than
        // computing a wrong trip count.
        return 0;
    }
    if (!bound || !induction) return 0;

    auto bc = static_cast<Constant *>(bound);
    int64_t bound_val = 0;
    if (bc->type()->is_int32())
        bound_val = bc->as<int32_t>();
    else if (bc->type()->is_uint32())
        bound_val = static_cast<int64_t>(bc->as<uint32_t>());
    else
        return 0;

    if (!induction->isa<PhiInst>()) return 0;
    auto phi = static_cast<PhiInst *>(induction);
    if (phi->parent_block() != prepare) return 0;

    int64_t start = 0;
    int64_t step = 1;
    for (size_t i = 0; i < phi->incoming_count(); ++i) {
        auto inc = phi->incoming(i);
        if (inc.block == update) {
            if (inc.value->isa<ArithmeticInst>()) {
                auto add = static_cast<ArithmeticInst *>(inc.value);
                if (add->op() == ArithmeticOp::BINARY_ADD &&
                    add->operand(0) == phi && add->operand(1)->isa<Constant>()) {
                    auto sc = static_cast<Constant *>(add->operand(1));
                    if (sc->type()->is_int32())
                        step = sc->as<int32_t>();
                    else if (sc->type()->is_uint32())
                        step = static_cast<int64_t>(sc->as<uint32_t>());
                }
            }
        } else {
            if (inc.value->isa<Constant>()) {
                auto sc = static_cast<Constant *>(inc.value);
                if (sc->type()->is_int32())
                    start = sc->as<int32_t>();
                else if (sc->type()->is_uint32())
                    start = static_cast<int64_t>(sc->as<uint32_t>());
            }
        }
    }

    if (step <= 0) return 0;
    int64_t trips = (bound_val - start + (op == ArithmeticOp::BINARY_LESS_EQUAL ? 1 : 0) + step - 1) / step;
    if (trips <= 0 || static_cast<size_t>(trips) > max_trip_count) return 0;
    return static_cast<size_t>(trips);
}

static void unroll(LoopInst *loop, size_t trips, LoopUnrollInfo &info) noexcept {
    auto func = loop->parent_function();
    if (!func) return;

    auto prepare = loop->prepare_block();
    auto body = loop->body_block();
    auto update = loop->update_block();
    auto merge = loop->merge_block();
    if (!prepare || !body || !merge) return;

    auto loop_parent_block = loop->parent_block();

    XIRBuilder builder;
    LoopUnrollResolver resolver;

    luisa::vector<PhiInst *> phis;
    for (auto inst : prepare->instructions()) {
        if (inst->isa<PhiInst>()) {
            phis.push_back(static_cast<PhiInst *>(inst));
        }
    }

    for (auto phi : phis) {
        for (size_t i = 0; i < phi->incoming_count(); ++i) {
            auto inc = phi->incoming(i);
            if (inc.block != update) {
                resolver.insert_or_assign(phi, inc.value);
                break;
            }
        }
    }

    luisa::vector<BasicBlock *> iter_blocks;
    iter_blocks.reserve(trips);
    for (size_t i = 0; i < trips; ++i) {
        iter_blocks.push_back(func->create_basic_block());
    }

    for (size_t iter = 0; iter < trips; ++iter) {
        auto iter_block = iter_blocks[iter];
        builder.set_insertion_point(iter_block);

        for (auto inst : body->instructions()) {
            if (inst->is_terminator()) continue;
            if (inst->isa<PhiInst>()) continue;
            auto cloned = inst->clone_with_metadata(builder, resolver);
            resolver.insert_or_assign(inst, cloned);
        }

        if (update) {
            for (auto inst : update->instructions()) {
                if (inst->is_terminator()) continue;
                auto cloned = inst->clone_with_metadata(builder, resolver);
                resolver.insert_or_assign(inst, cloned);
            }
        }

        for (auto phi : phis) {
            for (size_t i = 0; i < phi->incoming_count(); ++i) {
                auto inc = phi->incoming(i);
                if (inc.block == update) {
                    resolver.insert_or_assign(phi, resolver.resolve(inc.value));
                    break;
                }
            }
        }

        if (iter + 1 < trips) {
            builder.br(iter_blocks[iter + 1]);
        } else {
            builder.br(merge);
        }
    }

    loop->remove_self();
    builder.set_insertion_point(loop_parent_block);
    if (trips > 0) {
        builder.br(iter_blocks[0]);
    } else {
        builder.br(merge);
    }

    info.unrolled_loop_count++;
}

// Analyze a SimpleLoopInst: find the induction variable and trip count from
// the body block's conditional branch.  SimpleLoopInst is emitted by the DSL
// for `$for(i, begin, end)` style loops.
[[nodiscard]] static size_t analyze_simple_loop_trip_count(SimpleLoopInst *loop, size_t max_trip_count) noexcept {
    auto body = loop->body_block();
    auto merge = loop->merge_block();
    if (!body || !merge) return 0;

    // Find the conditional branch in the body block
    ConditionalBranchInst *cond_br = nullptr;
    for (auto inst : body->instructions()) {
        if (inst->isa<ConditionalBranchInst>()) {
            cond_br = static_cast<ConditionalBranchInst *>(inst);
            break;
        }
    }
    if (!cond_br) return 0;

    // One target must be the merge block (exit), the other must be the body (back edge)
    auto true_target = cond_br->true_block();
    auto false_target = cond_br->false_block();
    bool true_is_exit = (true_target == merge);
    bool false_is_exit = (false_target == merge);
    if (!true_is_exit && !false_is_exit) return 0;
    bool back_edge_is_true = !true_is_exit;
    auto back_edge_target = back_edge_is_true ? true_target : false_target;
    if (back_edge_target != body) return 0;

    auto cond = cond_br->condition();
    if (!cond || !cond->isa<ArithmeticInst>()) return 0;
    auto cmp = static_cast<ArithmeticInst *>(cond);
    auto op = cmp->op();
    // The condition is typically `induction < bound` for the back-edge (continue)
    // or `induction >= bound` for the exit edge.  Normalize to the "continue" form.
    ArithmeticOp normalized_op;
    Value *induction = nullptr;
    Value *bound = nullptr;
    if (back_edge_is_true) {
        // The true branch goes back to body → condition is "continue" condition
        if (op == ArithmeticOp::BINARY_LESS || op == ArithmeticOp::BINARY_LESS_EQUAL) {
            normalized_op = op;
            induction = cmp->operand(0);
            bound = cmp->operand(1);
        } else if (op == ArithmeticOp::BINARY_GREATER_EQUAL || op == ArithmeticOp::BINARY_GREATER) {
            // Inverted: exit on `i >= bound`, back-edge on `i < bound`
            return 0;  // Only handle the simple case for now
        } else {
            return 0;
        }
    } else {
        // The false branch goes back to body → condition is "exit" condition,
        // so the continue condition is the negation.
        if (op == ArithmeticOp::BINARY_GREATER_EQUAL) {
            normalized_op = ArithmeticOp::BINARY_LESS;
            induction = cmp->operand(0);
            bound = cmp->operand(1);
        } else if (op == ArithmeticOp::BINARY_GREATER) {
            normalized_op = ArithmeticOp::BINARY_LESS_EQUAL;
            induction = cmp->operand(0);
            bound = cmp->operand(1);
        } else {
            return 0;
        }
    }
    if (!bound || !induction) return 0;
    if (!bound->isa<Constant>()) return 0;

    auto bc = static_cast<Constant *>(bound);
    int64_t bound_val = 0;
    if (bc->type()->is_int32())
        bound_val = bc->as<int32_t>();
    else if (bc->type()->is_uint32())
        bound_val = static_cast<int64_t>(bc->as<uint32_t>());
    else
        return 0;

    // Find the initial value of the induction variable (must be set before the loop)
    // The induction variable is typically an alloca that is stored once before the loop.
    // For SimpleLoopInst, the initial value is loaded inside the body from an alloca
    // that is set in the parent block.  We look for a PhiInst or a Load from an alloca
    // that is stored with a constant in the parent block.
    int64_t start = 0;
    int64_t step = 1;

    if (induction->isa<PhiInst>()) {
        auto phi = static_cast<PhiInst *>(induction);
        for (size_t i = 0; i < phi->incoming_count(); ++i) {
            auto inc = phi->incoming(i);
            if (inc.block == body) {
                // Back-edge: this is the updated value
                if (inc.value->isa<ArithmeticInst>()) {
                    auto add = static_cast<ArithmeticInst *>(inc.value);
                    if (add->op() == ArithmeticOp::BINARY_ADD &&
                        add->operand(0) == phi && add->operand(1)->isa<Constant>()) {
                        auto sc = static_cast<Constant *>(add->operand(1));
                        if (sc->type()->is_int32())
                            step = sc->as<int32_t>();
                        else if (sc->type()->is_uint32())
                            step = static_cast<int64_t>(sc->as<uint32_t>());
                    }
                }
            } else {
                // Entry edge: initial value from before the loop
                if (inc.value->isa<Constant>()) {
                    auto sc = static_cast<Constant *>(inc.value);
                    if (sc->type()->is_int32())
                        start = sc->as<int32_t>();
                    else if (sc->type()->is_uint32())
                        start = static_cast<int64_t>(sc->as<uint32_t>());
                }
            }
        }
    } else if (induction->isa<LoadInst>()) {
        // Load from an alloca: find the initial store in the parent block
        auto load = static_cast<LoadInst *>(induction);
        auto var = load->variable();
        if (var && var->isa<AllocaInst>()) {
            auto parent_block = loop->parent_block();
            if (parent_block) {
                parent_block->traverse_instructions([&](Instruction *inst) noexcept {
                    if (inst->isa<StoreInst>()) {
                        auto store = static_cast<StoreInst *>(inst);
                        if (store->variable() == var && store->value()->isa<Constant>()) {
                            auto sc = static_cast<Constant *>(store->value());
                            if (sc->type()->is_int32())
                                start = sc->as<int32_t>();
                            else if (sc->type()->is_uint32())
                                start = static_cast<int64_t>(sc->as<uint32_t>());
                        }
                    }
                });
            }
            // Try to find the step by looking for a store to the same alloca in the body
            // that adds a constant.
            body->traverse_instructions([&](Instruction *inst) noexcept {
                if (inst->isa<StoreInst>()) {
                    auto store = static_cast<StoreInst *>(inst);
                    if (store->variable() == var && store->value()->isa<ArithmeticInst>()) {
                        auto add = static_cast<ArithmeticInst *>(store->value());
                        if (add->op() == ArithmeticOp::BINARY_ADD) {
                            if (add->operand(1)->isa<Constant>()) {
                                auto sc = static_cast<Constant *>(add->operand(1));
                                if (sc->type()->is_int32())
                                    step = sc->as<int32_t>();
                                else if (sc->type()->is_uint32())
                                    step = static_cast<int64_t>(sc->as<uint32_t>());
                            }
                        }
                    }
                }
            });
        }
    } else {
        return 0;
    }

    if (step <= 0) return 0;
    int64_t trips = (bound_val - start + (normalized_op == ArithmeticOp::BINARY_LESS_EQUAL ? 1 : 0) + step - 1) / step;
    if (trips <= 0 || static_cast<size_t>(trips) > max_trip_count) return 0;
    return static_cast<size_t>(trips);
}

static void unroll_simple_loop(SimpleLoopInst *loop, size_t trips, LoopUnrollInfo &info) noexcept {
    auto func = loop->parent_function();
    if (!func) return;

    auto body = loop->body_block();
    auto merge = loop->merge_block();
    if (!body || !merge) return;

    auto loop_parent_block = loop->parent_block();

    XIRBuilder builder;
    LoopUnrollResolver resolver;

    // Find phi nodes in the body block
    luisa::vector<PhiInst *> phis;
    luisa::vector<Instruction *> non_phi_non_term;
    for (auto inst : body->instructions()) {
        if (inst->is_terminator()) continue;
        if (inst->isa<PhiInst>()) {
            phis.push_back(static_cast<PhiInst *>(inst));
        }
    }

    // For each phi, get the initial value (from outside the loop)
    for (auto phi : phis) {
        for (size_t i = 0; i < phi->incoming_count(); ++i) {
            auto inc = phi->incoming(i);
            if (inc.block != body) {
                resolver.insert_or_assign(phi, inc.value);
                break;
            }
        }
    }

    // Create unrolled iteration blocks
    luisa::vector<BasicBlock *> iter_blocks;
    iter_blocks.reserve(trips);
    for (size_t i = 0; i < trips; ++i) {
        iter_blocks.push_back(func->create_basic_block());
    }

    for (size_t iter = 0; iter < trips; ++iter) {
        auto iter_block = iter_blocks[iter];
        builder.set_insertion_point(iter_block);

        // Clone all non-phi, non-terminator instructions from the body
        for (auto inst : body->instructions()) {
            if (inst->is_terminator()) continue;
            if (inst->isa<PhiInst>()) continue;
            auto cloned = inst->clone_with_metadata(builder, resolver);
            resolver.insert_or_assign(inst, cloned);
        }

        // Update phi values for the next iteration
        for (auto phi : phis) {
            for (size_t i = 0; i < phi->incoming_count(); ++i) {
                auto inc = phi->incoming(i);
                if (inc.block == body) {
                    resolver.insert_or_assign(phi, resolver.resolve(inc.value));
                    break;
                }
            }
        }

        if (iter + 1 < trips) {
            builder.br(iter_blocks[iter + 1]);
        } else {
            builder.br(merge);
        }
    }

    loop->remove_self();
    builder.set_insertion_point(loop_parent_block);
    if (trips > 0) {
        builder.br(iter_blocks[0]);
    } else {
        builder.br(merge);
    }

    info.unrolled_loop_count++;
}

static void run(Function *function, LoopUnrollInfo &info, const LoopUnrollOptions &options) noexcept {
    auto def = function->definition();
    if (!def) return;

    // Collect LoopInst (structured, from $while or complex $for)
    luisa::vector<LoopInst *> loops;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<LoopInst>())
            loops.push_back(static_cast<LoopInst *>(inst));
    });

    for (auto loop : loops) {
        auto trips = analyze_trip_count(loop, options.max_trip_count);
        if (trips > 0) {
            if (options.unroll_pure_only) {
                bool has_buffer_write = false;
                if (auto body = loop->body_block()) {
                    body->traverse_instructions([&](Instruction *inst) noexcept {
                        if (inst->derived_instruction_tag() == DerivedInstructionTag::RESOURCE_WRITE) {
                            has_buffer_write = true;
                        }
                    });
                }
                if (!has_buffer_write) unroll(loop, trips, info);
            } else {
                unroll(loop, trips, info);
            }
        }
    }

    // Collect SimpleLoopInst (from $for with ranges)
    luisa::vector<SimpleLoopInst *> simple_loops;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<SimpleLoopInst>())
            simple_loops.push_back(static_cast<SimpleLoopInst *>(inst));
    });

    for (auto loop : simple_loops) {
        auto trips = analyze_simple_loop_trip_count(loop, options.max_trip_count);
        if (trips > 0) {
            if (options.unroll_pure_only) {
                bool has_buffer_write = false;
                if (auto body = loop->body_block()) {
                    body->traverse_instructions([&](Instruction *inst) noexcept {
                        if (inst->derived_instruction_tag() == DerivedInstructionTag::RESOURCE_WRITE) {
                            has_buffer_write = true;
                        }
                    });
                }
                if (!has_buffer_write) unroll_simple_loop(loop, trips, info);
            } else {
                unroll_simple_loop(loop, trips, info);
            }
        }
    }
}

}// namespace detail

LoopUnrollInfo loop_unroll_pass_run_on_function(Function *function, LoopUnrollOptions options) noexcept {
    LoopUnrollInfo info;
    detail::run(function, info, options);
    return info;
}

LoopUnrollInfo loop_unroll_pass_run_on_module(Module *module, LoopUnrollOptions options) noexcept {
    LoopUnrollInfo info;
    for (auto f : module->function_list())
        detail::run(f, info, options);
    return info;
}

}// namespace luisa::compute::xir
