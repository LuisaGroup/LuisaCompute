#include <luisa/xir/passes/reassociate.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/constant.h>

namespace luisa::compute::xir {

namespace detail {

[[nodiscard]] static bool has_single_use(const Value *value) noexcept {
    size_t count = 0;
    for (auto &&use : value->use_list()) {
        count++;
        if (count > 1) return false;
    }
    return count == 1;
}

[[nodiscard]] static int get_operand_rank(const Value *value,
                                          const luisa::unordered_map<const Value *, size_t> &inst_ranks) noexcept {
    if (value->isa<Constant>()) return 0;
    if (!value->isa<Instruction>()) return 1;
    auto it = inst_ranks.find(value);
    return it != inst_ranks.end() ? static_cast<int>(2 + it->second) : 1;
}

static void linearize_add_mul_operands(Value *value, ArithmeticOp op,
                                       luisa::vector<Value *> &operands) noexcept {
    if (value->isa<ArithmeticInst>()) {
        auto inst = static_cast<ArithmeticInst *>(value);
        if (inst->op() == op && inst->operand_count() == 2 && has_single_use(inst)) {
            linearize_add_mul_operands(inst->operand(0), op, operands);
            linearize_add_mul_operands(inst->operand(1), op, operands);
            return;
        }
    }
    operands.push_back(value);
}

[[nodiscard]] static bool operands_changed(const luisa::vector<Value *> &operands,
                                           Value *orig_lhs, Value *orig_rhs) noexcept {
    if (operands.size() != 2) return true;
    return operands[0] != orig_lhs || operands[1] != orig_rhs;
}

static Value *rebuild_left_associative(const Type *type, ArithmeticOp op,
                                       luisa::span<Value *const> operands,
                                       XIRBuilder &builder) noexcept {
    if (operands.empty()) return nullptr;
    if (operands.size() == 1) return operands[0];
    auto lhs = operands[0];
    for (size_t i = 1; i < operands.size(); ++i) {
        lhs = builder.call(type, op, {lhs, operands[i]});
    }
    return lhs;
}

static void process_add_or_mul(ArithmeticInst *inst, ArithmeticOp op,
                               const luisa::unordered_map<const Value *, size_t> &inst_ranks,
                               Module *module, XIRBuilder &builder,
                               ReassociateInfo &info) noexcept {
    // Phase 1: Linearize operands
    luisa::vector<Value *> operands;
    linearize_add_mul_operands(inst->operand(0), op, operands);
    linearize_add_mul_operands(inst->operand(1), op, operands);

    if (!operands_changed(operands, inst->operand(0), inst->operand(1))) return;

    // Phase 2: Sort by rank, then by pointer for grouping
    std::sort(operands.begin(), operands.end(),
              [&inst_ranks](Value *a, Value *b) noexcept {
                  auto ra = get_operand_rank(a, inst_ranks);
                  auto rb = get_operand_rank(b, inst_ranks);
                  if (ra != rb) return ra < rb;
                  return a < b;
              });

    // Phase 3: Rebuild
    builder.set_insertion_point(inst);
    auto replacement = rebuild_left_associative(inst->type(), op, operands, builder);
    if (replacement == nullptr) return;

    inst->replace_all_uses_with(replacement);
    inst->remove_self();
    info.reassociated_inst_count++;
}

static void process_sub(ArithmeticInst *inst,
                        const luisa::unordered_map<const Value *, size_t> &inst_ranks,
                        Module *module, XIRBuilder &builder,
                        ReassociateInfo &info) noexcept {
    builder.set_insertion_point(inst);
    auto neg_rhs = builder.call(inst->type(), ArithmeticOp::UNARY_MINUS, {inst->operand(1)});

    // Linearize: lhs (flatten if single-use BINARY_ADD) + neg_rhs
    luisa::vector<Value *> operands;
    linearize_add_mul_operands(inst->operand(0), ArithmeticOp::BINARY_ADD, operands);
    operands.push_back(neg_rhs);

    // Sort
    std::sort(operands.begin(), operands.end(),
              [&inst_ranks](Value *a, Value *b) noexcept {
                  auto ra = get_operand_rank(a, inst_ranks);
                  auto rb = get_operand_rank(b, inst_ranks);
                  if (ra != rb) return ra < rb;
                  return a < b;
              });

    // Rebuild as add chain
    auto replacement = rebuild_left_associative(inst->type(), ArithmeticOp::BINARY_ADD, operands, builder);
    if (replacement == nullptr) return;

    inst->replace_all_uses_with(replacement);
    inst->remove_self();
    info.reassociated_inst_count++;
}

static void reassociate_on_function(FunctionDefinition *def, ReassociateInfo &info) noexcept {
    auto module = def->parent_module();
    XIRBuilder builder;

    // Assign ranks to all instructions in traversal order
    luisa::unordered_map<const Value *, size_t> inst_ranks;
    size_t next_rank = 0;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        inst_ranks[inst] = next_rank++;
    });

    // Collect worklist
    luisa::vector<ArithmeticInst *> worklist;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<ArithmeticInst>()) {
            auto arith = static_cast<ArithmeticInst *>(inst);
            auto op = arith->op();
            if (op == ArithmeticOp::BINARY_ADD || op == ArithmeticOp::BINARY_MUL || op == ArithmeticOp::BINARY_SUB) {
                worklist.push_back(arith);
            }
        }
    });

    // Process each instruction
    for (auto inst : worklist) {
        auto op = inst->op();
        if (op == ArithmeticOp::BINARY_ADD || op == ArithmeticOp::BINARY_MUL) {
            process_add_or_mul(inst, op, inst_ranks, module, builder, info);
        } else if (op == ArithmeticOp::BINARY_SUB) {
            process_sub(inst, inst_ranks, module, builder, info);
        }
    }
}

}// namespace detail

ReassociateInfo reassociate_pass_run_on_function(FunctionDefinition *def) noexcept {
    ReassociateInfo info;
    detail::reassociate_on_function(def, info);
    return info;
}

ReassociateInfo reassociate_pass_run_on_module(Module *module, PassReport *report) noexcept {
    ReassociateInfo info;
    for (auto f : module->function_list()) {
        if (auto def = f->definition()) {
            detail::reassociate_on_function(def, info);
        }
    }
    if (report != nullptr) {
        report->set("reassociated_inst", info.reassociated_inst_count);
    }
    return info;
}

}// namespace luisa::compute::xir
