#include <luisa/core/stl/algorithm.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/passes/scalar_evolution.h>

namespace luisa::compute::xir {

// SCEV base class implementations
SCEVUnknown::SCEVUnknown(Instruction *inst) noexcept : _inst{inst} {}
const Type *SCEVUnknown::type() const noexcept { return _inst->type(); }

SCEVConstant::SCEVConstant(Constant *c) noexcept : _constant{c} {}
const Type *SCEVConstant::type() const noexcept { return _constant->type(); }

SCEVAddRec::SCEVAddRec(const SCEV *start, const SCEV *stride, LoopInst *loop) noexcept
    : _start{start}, _stride{stride}, _loop{loop} {}
const Type *SCEVAddRec::type() const noexcept { return _start->type(); }

SCEVAddExpr::SCEVAddExpr(luisa::vector<const SCEV *> ops) noexcept : _operands{std::move(ops)} {}
const Type *SCEVAddExpr::type() const noexcept { return _operands.empty() ? nullptr : _operands.front()->type(); }

SCEVMulExpr::SCEVMulExpr(luisa::vector<const SCEV *> ops) noexcept : _operands{std::move(ops)} {}
const Type *SCEVMulExpr::type() const noexcept { return _operands.empty() ? nullptr : _operands.front()->type(); }

namespace {

// Global storage for SCEV query interface
struct SCEVStorage {
    luisa::unordered_map<Instruction *, const SCEV *> value_to_scev;
};

SCEVStorage &get_scev_storage() noexcept {
    static SCEVStorage storage;
    return storage;
}

struct SCEVAnalyzer {
    FunctionDefinition *def{nullptr};
    DomTree dom_tree;
    luisa::vector<luisa::unique_ptr<SCEV>> allocated;
    luisa::unordered_map<Value *, const SCEV *> cache;
    luisa::unordered_set<BasicBlock *> loop_blocks;
    LoopInst *current_loop{nullptr};

    [[nodiscard]] const SCEV *get_scev(Value *v) noexcept {
        if (v == nullptr) { return nullptr; }
        if (auto it = cache.find(v); it != cache.end()) { return it->second; }

        if (v->isa<Constant>()) {
            auto *c = static_cast<Constant *>(v);
            auto scev = luisa::make_unique<SCEVConstant>(c);
            auto *ptr = scev.get();
            allocated.emplace_back(std::move(scev));
            cache[v] = ptr;
            return ptr;
        }

        if (!v->isa<Instruction>()) {
            auto scev = luisa::make_unique<SCEVUnknown>(static_cast<Instruction *>(v));
            auto *ptr = scev.get();
            allocated.emplace_back(std::move(scev));
            cache[v] = ptr;
            return ptr;
        }

        auto *inst = static_cast<Instruction *>(v);
        auto *ptr = build_scev(inst);
        cache[v] = ptr;
        return ptr;
    }

    [[nodiscard]] const SCEV *build_scev(Instruction *inst) noexcept {
        if (inst->isa<PhiInst>()) {
            return build_phi_scev(static_cast<PhiInst *>(inst));
        }
        if (inst->isa<ArithmeticInst>()) {
            return build_arithmetic_scev(static_cast<ArithmeticInst *>(inst));
        }
        auto scev = luisa::make_unique<SCEVUnknown>(inst);
        auto *ptr = scev.get();
        allocated.emplace_back(std::move(scev));
        return ptr;
    }

    [[nodiscard]] bool is_loop_invariant(Value *v) noexcept {
        luisa::unordered_set<Value *> visited;
        return is_loop_invariant_impl(v, visited);
    }

    [[nodiscard]] bool is_loop_invariant_impl(Value *v, luisa::unordered_set<Value *> &visited) noexcept {
        if (v == nullptr) { return false; }
        if (v->isa<Constant>()) { return true; }
        if (!v->isa<Instruction>()) { return false; }
        if (!visited.emplace(v).second) { return false; }
        auto *inst = static_cast<Instruction *>(v);
        auto *bb = inst->parent_block();
        if (loop_blocks.contains(bb)) {
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                if (!is_loop_invariant_impl(inst->operand(i), visited)) { return false; }
            }
            return true;
        }
        return true;
    }

    [[nodiscard]] const SCEV *build_phi_scev(PhiInst *phi) noexcept {
        auto count = phi->incoming_count();
        if (count != 2) {
            auto scev = luisa::make_unique<SCEVUnknown>(phi);
            auto *ptr = scev.get();
            allocated.emplace_back(std::move(scev));
            return ptr;
        }

        auto *preheader_block = current_loop->parent_block();
        auto *update_block = current_loop->update_block();

        // Find start (preheader incoming) and recurrence (latch incoming)
        Value *start_val = nullptr;
        Value *recur_val = nullptr;

        for (size_t i = 0; i < count; ++i) {
            auto inc = phi->incoming(i);
            if (inc.block == preheader_block) {
                start_val = inc.value;
            } else if (inc.block == update_block) {
                recur_val = inc.value;
            }
        }

        if (start_val == nullptr || recur_val == nullptr) {
            auto scev = luisa::make_unique<SCEVUnknown>(phi);
            auto *ptr = scev.get();
            allocated.emplace_back(std::move(scev));
            return ptr;
        }

        // Analyze recurrence: look for BINARY_ADD(phi, stride)
        if (recur_val->isa<Instruction>()) {
            auto *recur_inst = static_cast<Instruction *>(recur_val);
            if (recur_inst->isa<ArithmeticInst>()) {
                auto *arith = static_cast<ArithmeticInst *>(recur_inst);
                if (arith->op() == ArithmeticOp::BINARY_ADD) {
                    // Check if one operand is the phi itself
                    Value *stride_val = nullptr;
                    for (size_t i = 0; i < arith->operand_count(); ++i) {
                        if (arith->operand(i) == phi) { continue; }
                        stride_val = arith->operand(i);
                    }
                    if (stride_val != nullptr && is_loop_invariant(stride_val)) {
                        auto *start = get_scev(start_val);
                        auto *stride = get_scev(stride_val);
                        auto scev = luisa::make_unique<SCEVAddRec>(start, stride, current_loop);
                        auto *ptr = scev.get();
                        allocated.emplace_back(std::move(scev));
                        return ptr;
                    }
                } else if (arith->op() == ArithmeticOp::BINARY_MUL) {
                    // Check if one operand is the phi itself
                    Value *scale_val = nullptr;
                    for (size_t i = 0; i < arith->operand_count(); ++i) {
                        if (arith->operand(i) == phi) { continue; }
                        scale_val = arith->operand(i);
                    }
                    if (scale_val != nullptr && is_loop_invariant(scale_val)) {
                        // SCEVAddRec(0, start * (scale - 1), loop)
                        // Simplified: use Unknown for now, this is a complex pattern
                        auto scev = luisa::make_unique<SCEVUnknown>(phi);
                        auto *ptr = scev.get();
                        allocated.emplace_back(std::move(scev));
                        return ptr;
                    }
                }
            }
        }

        // Fallback: create add recurrence with unknown stride
        // Check if recur_val is not the phi itself (simple add recurrence)
        if (recur_val != phi) {
            auto *start = get_scev(start_val);
            auto *recur = get_scev(recur_val);
            // If recurrence contains the phi as an operand, we have a real recurrence
            // For now, just create Unknown
            auto scev = luisa::make_unique<SCEVUnknown>(phi);
            auto *ptr = scev.get();
            allocated.emplace_back(std::move(scev));
            return ptr;
        }

        auto scev = luisa::make_unique<SCEVUnknown>(phi);
        auto *ptr = scev.get();
        allocated.emplace_back(std::move(scev));
        return ptr;
    }

    [[nodiscard]] const SCEV *build_arithmetic_scev(ArithmeticInst *inst) noexcept {
        if (inst->op() == ArithmeticOp::BINARY_ADD) {
            luisa::vector<const SCEV *> ops;
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                if (auto *scev = get_scev(inst->operand(i))) {
                    // Flatten nested add exprs
                    if (scev->kind() == SCEV::Kind::ADD) {
                        auto *add = static_cast<const SCEVAddExpr *>(scev);
                        for (auto *op : add->operands()) { ops.emplace_back(op); }
                    } else {
                        ops.emplace_back(scev);
                    }
                }
            }
            auto scev = luisa::make_unique<SCEVAddExpr>(std::move(ops));
            auto *ptr = scev.get();
            allocated.emplace_back(std::move(scev));
            return simplify(ptr);
        }
        if (inst->op() == ArithmeticOp::BINARY_MUL) {
            luisa::vector<const SCEV *> ops;
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                if (auto *scev = get_scev(inst->operand(i))) {
                    // Flatten nested mul exprs
                    if (scev->kind() == SCEV::Kind::MUL) {
                        auto *mul = static_cast<const SCEVMulExpr *>(scev);
                        for (auto *op : mul->operands()) { ops.emplace_back(op); }
                    } else {
                        ops.emplace_back(scev);
                    }
                }
            }
            auto scev = luisa::make_unique<SCEVMulExpr>(std::move(ops));
            auto *ptr = scev.get();
            allocated.emplace_back(std::move(scev));
            return simplify(ptr);
        }
        auto scev = luisa::make_unique<SCEVUnknown>(inst);
        auto *ptr = scev.get();
        allocated.emplace_back(std::move(scev));
        return ptr;
    }

    [[nodiscard]] const SCEV *simplify(const SCEV *scev) noexcept {
        if (scev->kind() == SCEV::Kind::ADD) {
            auto *add = static_cast<const SCEVAddExpr *>(scev);
            auto ops = add->operands();
            if (ops.size() == 1) { return ops[0]; }
            // Fold constants
            luisa::vector<const SCEV *> non_const;
            const SCEVConstant *folded_const = nullptr;
            for (auto *op : ops) {
                if (op->kind() == SCEV::Kind::CONSTANT) {
                    folded_const = static_cast<const SCEVConstant *>(op);
                } else {
                    non_const.emplace_back(op);
                }
            }
            if (folded_const != nullptr && non_const.empty()) {
                return folded_const;
            }
            if (folded_const != nullptr && non_const.size() + 1 == ops.size()) {
                non_const.emplace_back(folded_const);
                auto scev_new = luisa::make_unique<SCEVAddExpr>(std::move(non_const));
                auto *ptr = scev_new.get();
                allocated.emplace_back(std::move(scev_new));
                return ptr;
            }
            return scev;
        }
        if (scev->kind() == SCEV::Kind::MUL) {
            auto *mul = static_cast<const SCEVMulExpr *>(scev);
            auto ops = mul->operands();
            if (ops.size() == 1) { return ops[0]; }
            // If any operand is constant 0, result is 0
            // If operand is constant 1, drop it
            luisa::vector<const SCEV *> filtered;
            for (auto *op : ops) {
                if (op->kind() == SCEV::Kind::CONSTANT) {
                    auto *c = static_cast<const SCEVConstant *>(op);
                    auto *type = c->constant()->type();
                    if (type->is_int32()) {
                        if (c->constant()->as<int32_t>() == 0) {
                            return op;
                        }
                        if (c->constant()->as<int32_t>() == 1) { continue; }
                    } else if (type->is_uint32()) {
                        if (c->constant()->as<uint32_t>() == 0u) {
                            return op;
                        }
                        if (c->constant()->as<uint32_t>() == 1u) { continue; }
                    } else if (type->is_float32()) {
                        if (c->constant()->as<float>() == 0.0f) {
                            return op;
                        }
                        if (c->constant()->as<float>() == 1.0f) { continue; }
                    }
                }
                filtered.emplace_back(op);
            }
            if (filtered.size() != ops.size()) {
                if (filtered.empty()) { return ops[0]; }
                auto scev_new = luisa::make_unique<SCEVMulExpr>(std::move(filtered));
                auto *ptr = scev_new.get();
                allocated.emplace_back(std::move(scev_new));
                return ptr;
            }
            return scev;
        }
        if (scev->kind() == SCEV::Kind::ADD_REC) {
            auto *ar = static_cast<const SCEVAddRec *>(scev);
            if (ar->stride()->kind() == SCEV::Kind::CONSTANT) {
                auto *c = static_cast<const SCEVConstant *>(ar->stride());
                auto *type = c->constant()->type();
                bool is_zero = false;
                if (type->is_int32()) { is_zero = (c->constant()->as<int32_t>() == 0); }
                else if (type->is_uint32()) { is_zero = (c->constant()->as<uint32_t>() == 0u); }
                if (is_zero) { return ar->start(); }
            }
            return scev;
        }
        return scev;
    }

    void collect_loop_blocks(LoopInst *loop) noexcept {
        loop_blocks.clear();
        loop_blocks.emplace(loop->prepare_block());
        loop_blocks.emplace(loop->body_block());
        if (loop->update_block()) { loop_blocks.emplace(loop->update_block()); }

        // Collect all blocks reachable from body_block up to update_block/merge_block
        luisa::vector<BasicBlock *> worklist;
        worklist.emplace_back(loop->body_block());
        auto *merge_block = loop->merge_block();

        while (!worklist.empty()) {
            auto *bb = worklist.back();
            worklist.pop_back();
            bb->traverse_successors(false, [&](BasicBlock *succ) noexcept {
                if (succ == merge_block) { return; }
                if (loop_blocks.emplace(succ).second) {
                    worklist.emplace_back(succ);
                }
            });
        }
    }

    void analyze_loop(LoopInst *loop) noexcept {
        current_loop = loop;
        collect_loop_blocks(loop);

        // Process all instructions in loop blocks to build SCEVs
        luisa::vector<BasicBlock *> blocks{loop_blocks.begin(), loop_blocks.end()};
        luisa::sort(blocks.begin(), blocks.end(), [&](BasicBlock *a, BasicBlock *b) noexcept {
            return dom_tree.dominates(a, b) && !dom_tree.dominates(b, a);
        });

        for (auto *bb : blocks) {
            for (auto *inst : bb->instructions()) {
                static_cast<void>(get_scev(inst));
            }
        }
    }

    SCEVInfo run() noexcept {
        dom_tree = compute_dom_tree(def);
        SCEVInfo info;

        luisa::vector<LoopInst *> loops;
        def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
            if (!bb->is_terminated()) { return; }
            auto *term = bb->terminator();
            if (term->isa<LoopInst>()) {
                loops.emplace_back(static_cast<LoopInst *>(term));
            }
        });

        for (auto *loop : loops) {
            analyze_loop(loop);
            info.analyzed_loop_count++;
        }

        // Store results for global query. Release ownership from the analyzer's
        // unique_ptrs so the cached SCEVs stay alive for later passes to query.
        auto &storage = get_scev_storage();
        for (auto &uptr : allocated) {
            auto *raw = uptr.release();
            for (auto &[val, scev_ptr] : cache) {
                if (scev_ptr == raw && val->isa<Instruction>()) {
                    storage.value_to_scev[static_cast<Instruction *>(val)] = raw;
                    break;
                }
            }
        }
        allocated.clear();

        return info;
    }
};

}// namespace

SCEVInfo scev_pass_run_on_function(FunctionDefinition *def) noexcept {
    if (def == nullptr) { return {}; }
    SCEVAnalyzer analyzer;
    analyzer.def = def;
    return analyzer.run();
}

SCEVInfo scev_pass_run_on_module(Module *module, PassReport *report) noexcept {
    SCEVInfo info;
    for (auto f : module->function_list()) {
        if (auto *def = f->definition()) {
            auto result = scev_pass_run_on_function(def);
            info.analyzed_loop_count += result.analyzed_loop_count;
        }
    }
    if (report != nullptr) {
        report->set("analyzed_loop", info.analyzed_loop_count);
    }
    return info;
}

const SCEV *scev_get_for_value(Instruction *inst) noexcept {
    if (inst == nullptr) { return nullptr; }
    auto &storage = get_scev_storage();
    auto it = storage.value_to_scev.find(inst);
    if (it != storage.value_to_scev.end()) {
        return it->second;
    }
    return nullptr;
}

}// namespace luisa::compute::xir
