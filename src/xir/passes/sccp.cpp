#include <luisa/xir/passes/sccp.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/undefined.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/indexed_branch.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/core/stl/hash.h>
#include <luisa/core/stl/optional.h>

#include <utility>

#include "coro_semantic_graph.h"
#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

enum class LatticeState : uint8_t {
    TOP,
    CONSTANT,
    BOTTOM,
};

struct LatticeValue {
    LatticeState state{LatticeState::TOP};
    Constant *constant{nullptr};

    [[nodiscard]] bool is_top() const noexcept { return state == LatticeState::TOP; }
    [[nodiscard]] bool is_constant() const noexcept { return state == LatticeState::CONSTANT; }
    [[nodiscard]] bool is_bottom() const noexcept { return state == LatticeState::BOTTOM; }

    static LatticeValue make_top() noexcept { return {LatticeState::TOP, nullptr}; }
    static LatticeValue make_bottom() noexcept { return {LatticeState::BOTTOM, nullptr}; }
    static LatticeValue make_constant(Constant *c) noexcept { return {LatticeState::CONSTANT, c}; }

    [[nodiscard]] LatticeValue meet(const LatticeValue &other) const noexcept {
        if (is_top()) return other;
        if (other.is_top()) return *this;
        if (is_bottom() || other.is_bottom()) return make_bottom();
        if (constant == other.constant) return *this;
        return make_bottom();
    }
};

// Forward declaration of the evaluator used by const_fold
extern Constant *try_fold_arithmetic_for_sccp(Module *module, ArithmeticOp op,
                                              const Type *type,
                                              luisa::span<Constant *const> operands) noexcept;

struct EdgeHash {
    [[nodiscard]] uint64_t operator()(const std::pair<BasicBlock *, BasicBlock *> &e) const noexcept {
        return luisa::hash_combine({
            luisa::hash_value(reinterpret_cast<uint64_t>(e.first)),
            luisa::hash_value(reinterpret_cast<uint64_t>(e.second)),
        });
    }
};

struct EdgeEqual {
    [[nodiscard]] bool operator()(const std::pair<BasicBlock *, BasicBlock *> &a,
                                  const std::pair<BasicBlock *, BasicBlock *> &b) const noexcept {
        return a.first == b.first && a.second == b.second;
    }
};

[[nodiscard]] luisa::optional<
    IndexedBranchTerminatorInstruction::case_value_type>
decode_indexed_branch_constant(const Constant *constant) noexcept {
    if (constant == nullptr || constant->type() == nullptr) {
        return luisa::nullopt;
    }
    switch (constant->type()->tag()) {
        case Type::Tag::BOOL: return constant->as<bool>();
        case Type::Tag::INT8:
            return luisa::bit_cast<uint8_t>(constant->as<int8_t>());
        case Type::Tag::UINT8: return constant->as<uint8_t>();
        case Type::Tag::INT16:
            return luisa::bit_cast<uint16_t>(constant->as<int16_t>());
        case Type::Tag::UINT16: return constant->as<uint16_t>();
        case Type::Tag::INT32:
            return luisa::bit_cast<uint32_t>(constant->as<int32_t>());
        case Type::Tag::UINT32: return constant->as<uint32_t>();
        case Type::Tag::INT64:
            return luisa::bit_cast<uint64_t>(constant->as<int64_t>());
        case Type::Tag::UINT64: return constant->as<uint64_t>();
        default: return luisa::nullopt;
    }
}

struct SCCPSolver {

    Module *module;
    luisa::unordered_map<Value *, LatticeValue> value_lattice;
    luisa::unordered_set<BasicBlock *> executable_blocks;
    luisa::unordered_set<std::pair<BasicBlock *, BasicBlock *>, EdgeHash, EdgeEqual> executable_edges;
    luisa::vector<Instruction *> ssa_worklist;
    luisa::vector<BasicBlock *> cfg_worklist;
    const CoroSemanticGraph *coro_graph{nullptr};

    template<typename Visit>
    void traverse_domain_blocks(FunctionDefinition *def,
                                Visit &&visit) noexcept {
        if (coro_graph != nullptr) {
            for (size_t i = 0u; i < coro_graph->block_count(); ++i) {
                visit(coro_graph->block(i));
            }
        } else {
            def->traverse_basic_blocks(
                std::forward<Visit>(visit));
        }
    }

    template<typename Visit>
    void traverse_domain_instructions(FunctionDefinition *def,
                                      Visit &&visit) noexcept {
        traverse_domain_blocks(
            def, [&](BasicBlock *block) noexcept {
                block->traverse_instructions(visit);
            });
    }

    void mark_all_successors_executable(BasicBlock *block) noexcept {
        if (coro_graph != nullptr) {
            auto id = coro_graph->block_id(block);
            for (auto successor : coro_graph->successors(id)) {
                mark_edge_executable(block,
                                     coro_graph->block(successor));
            }
        } else {
            block->traverse_successors(
                true, [&](BasicBlock *successor) noexcept {
                    mark_edge_executable(block, successor);
                });
        }
    }

    [[nodiscard]] LatticeValue get_lattice(Value *v) noexcept {
        if (v == nullptr) return LatticeValue::make_bottom();
        switch (v->derived_value_tag()) {
            case DerivedValueTag::CONSTANT:
                return LatticeValue::make_constant(static_cast<Constant *>(v));
            case DerivedValueTag::UNDEFINED:
                return LatticeValue::make_bottom();
            // Function arguments, special registers (dispatch_id, thread_id, ...),
            // and other non-instruction runtime values are not statically known.
            // Treating them as TOP would let phi meets and arithmetic folds collapse
            // to a wrong constant; they must start at BOTTOM so the lattice stays sound.
            case DerivedValueTag::ARGUMENT: [[fallthrough]];
            case DerivedValueTag::SPECIAL_REGISTER: [[fallthrough]];
            case DerivedValueTag::FUNCTION: [[fallthrough]];
            case DerivedValueTag::BASIC_BLOCK:
                return LatticeValue::make_bottom();
            default:
                break;
        }
        auto it = value_lattice.find(v);
        if (it != value_lattice.end()) return it->second;
        return LatticeValue::make_top();
    }

    bool update_lattice(Value *v, LatticeValue new_val) noexcept {
        auto &current = value_lattice[v];
        auto merged = current.meet(new_val);
        if (merged.state != current.state || merged.constant != current.constant) {
            current = merged;
            return true;
        }
        return false;
    }

    bool is_edge_executable(BasicBlock *from, BasicBlock *to) noexcept {
        return executable_edges.contains({from, to});
    }

    void mark_edge_executable(BasicBlock *from, BasicBlock *to) noexcept {
        if (executable_edges.emplace(from, to).second) {
            if (executable_blocks.emplace(to).second) {
                cfg_worklist.push_back(to);
            } else {
                for (auto inst : to->instructions()) {
                    if (inst->isa<PhiInst>()) {
                        ssa_worklist.push_back(inst);
                    }
                }
            }
        }
    }

    void mark_block_executable(BasicBlock *block) noexcept {
        if (executable_blocks.emplace(block).second) {
            cfg_worklist.push_back(block);
        }
    }

    void visit_phi(PhiInst *phi) noexcept {
        auto result = LatticeValue::make_top();
        for (size_t i = 0; i < phi->incoming_count(); ++i) {
            auto inc = phi->incoming(i);
            if (!is_edge_executable(inc.block, phi->parent_block())) continue;
            auto val = get_lattice(inc.value);
            result = result.meet(val);
            if (result.is_bottom()) break;
        }
        if (update_lattice(phi, result)) {
            for (auto &&use : phi->use_list()) {
                if (auto user = use->user(); user != nullptr && user->isa<Instruction>()) {
                    auto user_inst = static_cast<Instruction *>(user);
                    if (executable_blocks.contains(user_inst->parent_block())) {
                        ssa_worklist.push_back(user_inst);
                    }
                }
            }
        }
    }

    void visit_arithmetic(ArithmeticInst *inst) noexcept {
        luisa::vector<Constant *> const_operands;
        bool all_const = true;
        bool any_bottom = false;
        for (size_t i = 0; i < inst->operand_count(); ++i) {
            auto lat = get_lattice(inst->operand(i));
            if (lat.is_bottom()) {
                any_bottom = true;
            } else if (lat.is_top()) {
                all_const = false;
            } else {
                const_operands.push_back(lat.constant);
            }
        }

        LatticeValue result;
        if (any_bottom) {
            result = LatticeValue::make_bottom();
        } else if (!all_const) {
            result = LatticeValue::make_top();
        } else {
            auto folded = try_fold_arithmetic_for_sccp(
                module, inst->op(), inst->type(),
                luisa::span<Constant *const>{const_operands.data(), const_operands.size()});
            result = folded ? LatticeValue::make_constant(folded) : LatticeValue::make_bottom();
        }

        if (update_lattice(inst, result)) {
            for (auto &&use : inst->use_list()) {
                if (auto user = use->user(); user != nullptr && user->isa<Instruction>()) {
                    auto user_inst = static_cast<Instruction *>(user);
                    if (executable_blocks.contains(user_inst->parent_block())) {
                        ssa_worklist.push_back(user_inst);
                    }
                }
            }
        }
    }

    void visit_terminator(Instruction *term) noexcept {
        auto block = term->parent_block();
        switch (term->derived_instruction_tag()) {
            case DerivedInstructionTag::BRANCH: {
                auto br = static_cast<BranchInst *>(term);
                if (auto target = br->target_block()) {
                    mark_edge_executable(block, target);
                }
                break;
            }
            case DerivedInstructionTag::CONDITIONAL_BRANCH: {
                auto cond_br = static_cast<ConditionalBranchTerminatorInstruction *>(term);
                auto cond_lat = get_lattice(cond_br->condition());
                if (cond_lat.is_constant() && cond_lat.constant->type()->is_bool()) {
                    bool val = cond_lat.constant->as<bool>();
                    if (val) {
                        if (auto tb = cond_br->true_block()) mark_edge_executable(block, tb);
                    } else {
                        if (auto fb = cond_br->false_block()) mark_edge_executable(block, fb);
                    }
                } else {
                    if (auto tb = cond_br->true_block()) mark_edge_executable(block, tb);
                    if (auto fb = cond_br->false_block()) mark_edge_executable(block, fb);
                }
                break;
            }
            case DerivedInstructionTag::IF: {
                // Structured IF: rewrite() does not flatten this terminator (the
                // structural frame must be preserved for destructure_cfg). Both
                // successor edges remain reachable at runtime regardless of the
                // condition's lattice value, so conservatively mark both
                // executable to keep downstream phi lattices sound.
                auto cond_br = static_cast<ConditionalBranchTerminatorInstruction *>(term);
                if (auto tb = cond_br->true_block()) mark_edge_executable(block, tb);
                if (auto fb = cond_br->false_block()) mark_edge_executable(block, fb);
                break;
            }
            case DerivedInstructionTag::INDEXED_BRANCH: {
                auto *indexed_branch =
                    static_cast<IndexedBranchInst *>(term);
                auto selector = get_lattice(indexed_branch->value());
                if (selector.is_constant()) {
                    if (auto case_value =
                            decode_indexed_branch_constant(
                                selector.constant)) {
                        auto *target = indexed_branch->default_block();
                        for (auto i = 0u;
                             i < indexed_branch->case_count(); i++) {
                            if (indexed_branch->case_value(i) ==
                                *case_value) {
                                target =
                                    indexed_branch->case_block(i);
                                break;
                            }
                        }
                        if (target != nullptr) {
                            mark_edge_executable(block, target);
                        }
                        break;
                    }
                }
                if (auto *target =
                        indexed_branch->default_block()) {
                    mark_edge_executable(block, target);
                }
                for (auto i = 0u;
                     i < indexed_branch->case_count(); i++) {
                    if (auto *target =
                            indexed_branch->case_block(i)) {
                        mark_edge_executable(block, target);
                    }
                }
                break;
            }
            default: {
                mark_all_successors_executable(block);
                break;
            }
        }
    }

    void visit_instruction(Instruction *inst) noexcept {
        if (inst->isa<PhiInst>()) {
            visit_phi(static_cast<PhiInst *>(inst));
        } else if (inst->isa<ArithmeticInst>()) {
            visit_arithmetic(static_cast<ArithmeticInst *>(inst));
        } else if (inst->is_terminator()) {
            visit_terminator(inst);
        } else {
            if (inst->type() != nullptr) {
                update_lattice(inst, LatticeValue::make_bottom());
            }
        }
    }

    void visit_block(BasicBlock *block) noexcept {
        for (auto inst : block->instructions()) {
            visit_instruction(inst);
        }
    }

    void solve(FunctionDefinition *def,
               const CoroSemanticGraph *semantic_graph) noexcept {
        coro_graph = semantic_graph;
        mark_block_executable(def->body_block());

        while (!cfg_worklist.empty() || !ssa_worklist.empty()) {
            while (!cfg_worklist.empty()) {
                auto block = cfg_worklist.back();
                cfg_worklist.pop_back();
                visit_block(block);
            }
            while (!ssa_worklist.empty()) {
                auto inst = ssa_worklist.back();
                ssa_worklist.pop_back();
                if (executable_blocks.contains(inst->parent_block())) {
                    visit_instruction(inst);
                }
            }
        }
    }

    SCCPInfo rewrite(FunctionDefinition *def) noexcept {
        SCCPInfo info;
        luisa::vector<Instruction *> to_replace;
        luisa::unordered_map<BasicBlock *, LoopInst *> loop_prepares;
        traverse_domain_blocks(def, [&](BasicBlock *block) noexcept {
            if (block != nullptr && block->is_terminated() &&
                block->terminator()->isa<LoopInst>()) {
                auto *loop = static_cast<LoopInst *>(
                    block->terminator());
                if (loop->prepare_block() != nullptr) {
                    loop_prepares.emplace(loop->prepare_block(), loop);
                }
            }
        });

        traverse_domain_instructions(def, [&](Instruction *inst) noexcept {
            if (inst->is_terminator()) return;
            if (inst->isa<PhiInst>() || inst->isa<ArithmeticInst>()) {
                auto lat = get_lattice(inst);
                if (lat.is_constant()) {
                    to_replace.push_back(inst);
                }
            }
        });

        for (auto inst : to_replace) {
            // SCCP constants are module-uniqued and cannot be the unique owner
            // of metadata from one source instruction.
            if (!inst->metadata_list().empty()) { continue; }
            auto lat = get_lattice(inst);
            inst->replace_all_uses_with(lat.constant);
            inst->remove_self();
            info.folded_inst_count++;
        }

        traverse_domain_blocks(def, [&](BasicBlock *block) noexcept {
            if (!executable_blocks.contains(block)) return;
            auto term = block->terminator();
            // Only flatten plain conditional branches. Structured terminators (IF,
            // LOOP, SWITCH, ...) carry a merge_block and structural semantics that
            // destructure_cfg relies on; collapsing them here would corrupt the
            // structured CFG frame.
            if (term->derived_instruction_tag() != DerivedInstructionTag::CONDITIONAL_BRANCH) return;
            auto cond_br = static_cast<ConditionalBranchTerminatorInstruction *>(term);
            auto cond_lat = get_lattice(cond_br->condition());
            if (!cond_lat.is_constant() || !cond_lat.constant->type()->is_bool()) return;
            bool val = cond_lat.constant->as<bool>();
            auto kept = val ? cond_br->true_block() : cond_br->false_block();
            auto dropped = val ? cond_br->false_block() : cond_br->true_block();
            if (kept == nullptr) return;
            if (!val) {
                auto prepare = loop_prepares.find(block);
                if (prepare != loop_prepares.end() &&
                    cond_br->true_block() ==
                        prepare->second->body_block() &&
                    cond_br->false_block() ==
                        prepare->second->merge_block()) {
                    // Keep the canonical zero-trip structured-loop prepare.
                    // Replacing it with Branch(merge) while retaining the
                    // owning LoopInst breaks the role contract.
                    return;
                }
            }
            // The block is no longer a predecessor of `dropped`. Strip stale phi
            // incomings now so downstream passes don't observe dangling references.
            // Skipped if both edges go to the same block (dropping would also strip
            // the surviving incoming).
            if (dropped != nullptr && dropped != kept) {
                for (auto inst : dropped->instructions()) {
                    if (!inst->isa<PhiInst>()) continue;
                    auto phi = static_cast<PhiInst *>(inst);
                    for (size_t i = phi->incoming_count(); i-- > 0;) {
                        if (phi->incoming(i).block == block) {
                            phi->remove_incoming(i);
                        }
                    }
                }
            }
            auto removed = term->remove_self();
            XIRBuilder builder;
            builder.set_insertion_point(block);
            auto *branch = builder.br(kept);
            for (auto *metadata : removed->metadata_list()) {
                branch->metadata_list().push_front(metadata->clone());
            }
            info.removed_branch_count++;
        });

        traverse_domain_blocks(def, [&](BasicBlock *block) noexcept {
            if (!executable_blocks.contains(block)) { return; }
            auto *term = block->terminator();
            if (!term->isa<IndexedBranchInst>()) { return; }
            auto *indexed_branch =
                static_cast<IndexedBranchInst *>(term);
            auto selector = get_lattice(indexed_branch->value());
            if (!selector.is_constant()) { return; }
            auto case_value =
                decode_indexed_branch_constant(selector.constant);
            if (!case_value) { return; }
            auto *kept = indexed_branch->default_block();
            for (auto i = 0u;
                 i < indexed_branch->case_count(); i++) {
                if (indexed_branch->case_value(i) == *case_value) {
                    kept = indexed_branch->case_block(i);
                    break;
                }
            }
            if (kept == nullptr) { return; }
            luisa::unordered_set<BasicBlock *> dropped;
            if (auto *target = indexed_branch->default_block();
                target != nullptr && target != kept) {
                dropped.emplace(target);
            }
            for (auto i = 0u;
                 i < indexed_branch->case_count(); i++) {
                if (auto *target =
                        indexed_branch->case_block(i);
                    target != nullptr && target != kept) {
                    dropped.emplace(target);
                }
            }
            for (auto *target : dropped) {
                for (auto *inst : target->instructions()) {
                    if (!inst->isa<PhiInst>()) { continue; }
                    auto *phi = static_cast<PhiInst *>(inst);
                    for (auto i = phi->incoming_count();
                         i-- > 0u;) {
                        if (phi->incoming(i).block == block) {
                            phi->remove_incoming(i);
                        }
                    }
                }
            }
            auto removed = term->remove_self();
            XIRBuilder builder;
            builder.set_insertion_point(block);
            auto *branch = builder.br(kept);
            for (auto *metadata : removed->metadata_list()) {
                branch->metadata_list().push_front(metadata->clone());
            }
            info.removed_branch_count++;
        });

        return info;
    }
};

// Minimal constant folder for SCCP — evaluates pure arithmetic on constant operands.
Constant *try_fold_arithmetic_for_sccp(Module *module, ArithmeticOp op,
                                       const Type *type,
                                       luisa::span<Constant *const> operands) noexcept {
    if (type == nullptr || operands.empty()) return nullptr;
    switch (op) {
        case ArithmeticOp::AGGREGATE:
        case ArithmeticOp::SHUFFLE:
        case ArithmeticOp::INSERT:
        case ArithmeticOp::EXTRACT:
        case ArithmeticOp::ALL:
        case ArithmeticOp::ANY:
        case ArithmeticOp::REDUCE_SUM:
        case ArithmeticOp::REDUCE_PRODUCT:
        case ArithmeticOp::REDUCE_MIN:
        case ArithmeticOp::REDUCE_MAX:
        case ArithmeticOp::OUTER_PRODUCT:
        case ArithmeticOp::MATRIX_COMP_NEG:
        case ArithmeticOp::MATRIX_COMP_ADD:
        case ArithmeticOp::MATRIX_COMP_SUB:
        case ArithmeticOp::MATRIX_COMP_MUL:
        case ArithmeticOp::MATRIX_COMP_DIV:
        case ArithmeticOp::MATRIX_LINALG_MUL:
        case ArithmeticOp::MATRIX_DETERMINANT:
        case ArithmeticOp::MATRIX_TRANSPOSE:
        case ArithmeticOp::MATRIX_INVERSE:
        case ArithmeticOp::CROSS:
        case ArithmeticOp::DOT:
        case ArithmeticOp::LENGTH:
        case ArithmeticOp::LENGTH_SQUARED:
        case ArithmeticOp::NORMALIZE:
        case ArithmeticOp::FACEFORWARD:
        case ArithmeticOp::REFLECT:
        case ArithmeticOp::SELECT:
            return nullptr;
        default:
            break;
    }
    if (!type->is_scalar()) return nullptr;

    auto size = type->size();
    luisa::vector<std::byte> result_data(size);
    std::memset(result_data.data(), 0, size);

    auto get_data = [&](size_t i) -> const void * {
        return i < operands.size() ? operands[i]->data() : nullptr;
    };

    auto ok = op == ArithmeticOp::POW_INT && operands.size() > 1u ?
                  eval_pow_int_op(
                      type, operands[1]->type(), result_data.data(),
                      get_data(0), get_data(1)) :
                  eval_scalar_op(type, op, result_data.data(),
                                 get_data(0),
                                 operands.size() > 1u ?
                                     operands[1u]->type() :
                                     nullptr,
                                 get_data(1), get_data(2));
    if (!ok) return nullptr;
    return module->create_constant(type, result_data.data());
}

static void run_sccp_on_function(Function *function, SCCPInfo &info) noexcept {
    if (function == nullptr) { return; }
    auto def = function->definition();
    if (def == nullptr || def->body_block() == nullptr) return;
    SCCPSolver solver;
    solver.module = function->parent_module();
    CoroSemanticGraph coro_graph{def};
    solver.solve(def, coro_graph.valid() ? &coro_graph : nullptr);
    auto result = solver.rewrite(def);
    info.folded_inst_count += result.folded_inst_count;
    info.removed_branch_count += result.removed_branch_count;
}

}// namespace detail

SCCPInfo sccp_pass_run_on_function(Function *function) noexcept {
    SCCPInfo info;
    detail::run_sccp_on_function(function, info);
    return info;
}

SCCPInfo sccp_pass_run_on_module(Module *module, PassReport *report) noexcept {
    SCCPInfo info;
    if (module == nullptr) {
        if (report != nullptr) {
            report->set("folded_inst", 0u);
            report->set("removed_branch", 0u);
        }
        return info;
    }
    for (auto f : module->function_list()) {
        detail::run_sccp_on_function(f, info);
    }
    if (report != nullptr) {
        report->set("folded_inst", info.folded_inst_count);
        report->set("removed_branch", info.removed_branch_count);
    }
    return info;
}

}// namespace luisa::compute::xir
