#include <luisa/xir/passes/cse.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/cast.h>

namespace luisa::compute::xir {

namespace detail {

struct ExpressionKey {
    DerivedInstructionTag tag;
    const Type *type;
    uint32_t op_code;
    luisa::vector<Value *> operands;

    bool operator==(const ExpressionKey &rhs) const noexcept {
        if (tag != rhs.tag || type != rhs.type || op_code != rhs.op_code) return false;
        if (operands.size() != rhs.operands.size()) return false;
        for (size_t i = 0; i < operands.size(); ++i) {
            if (operands[i] != rhs.operands[i]) return false;
        }
        return true;
    }
};

struct ExpressionKeyHash {
    size_t operator()(const ExpressionKey &key) const noexcept {
        size_t h = std::hash<int>{}(static_cast<int>(key.tag));
        h ^= std::hash<const void *>{}(key.type) + 0x9e3779b9u + (h << 6u) + (h >> 2u);
        h ^= std::hash<uint32_t>{}(key.op_code) + 0x9e3779b9u + (h << 6u) + (h >> 2u);
        for (auto op : key.operands) {
            h ^= std::hash<const void *>{}(op) + 0x9e3779b9u + (h << 6u) + (h >> 2u);
        }
        return h;
    }
};

[[nodiscard]] static bool is_cse_candidate(Instruction *inst) noexcept {
    if (inst->type() == nullptr) return false;
    switch (inst->derived_instruction_tag()) {
        case DerivedInstructionTag::ARITHMETIC:
        case DerivedInstructionTag::CAST:
            return true;
        default:
            return false;
    }
}

[[nodiscard]] static bool is_commutative(ArithmeticOp op) noexcept {
    switch (op) {
        case ArithmeticOp::BINARY_ADD:
        case ArithmeticOp::BINARY_MUL:
        case ArithmeticOp::BINARY_BIT_AND:
        case ArithmeticOp::BINARY_BIT_OR:
        case ArithmeticOp::BINARY_BIT_XOR:
        case ArithmeticOp::BINARY_EQUAL:
        case ArithmeticOp::BINARY_NOT_EQUAL:
        case ArithmeticOp::MIN:
        case ArithmeticOp::MAX:
            return true;
        default:
            return false;
    }
}

[[nodiscard]] static ExpressionKey make_key(Instruction *inst) noexcept {
    ExpressionKey key{};
    key.tag = inst->derived_instruction_tag();
    key.type = inst->type();
    switch (key.tag) {
        case DerivedInstructionTag::ARITHMETIC:
            key.op_code = static_cast<uint32_t>(static_cast<ArithmeticInst *>(inst)->op());
            break;
        case DerivedInstructionTag::CAST:
            key.op_code = static_cast<uint32_t>(static_cast<CastInst *>(inst)->op());
            break;
        default:
            key.op_code = 0;
            break;
    }
    key.operands.reserve(inst->operand_count());
    for (size_t i = 0; i < inst->operand_count(); ++i) {
        key.operands.push_back(inst->operand(i));
    }
    if (key.tag == DerivedInstructionTag::ARITHMETIC &&
        key.operands.size() == 2u &&
        is_commutative(static_cast<ArithmeticInst *>(inst)->op())) {
        if (std::less<const void *>{}(key.operands[1], key.operands[0])) {
            std::swap(key.operands[0], key.operands[1]);
        }
    }
    return key;
}

using ExprTable = luisa::unordered_map<ExpressionKey, Instruction *, ExpressionKeyHash>;

static void cse_domtree_walk(const DomTreeNode *node, ExprTable &table, CSEInfo &info) noexcept {
    if (!node) return;
    auto block = node->block();

    luisa::vector<ExpressionKey> added_this_scope;
    luisa::vector<Instruction *> to_remove;

    for (auto inst : block->instructions()) {
        if (!is_cse_candidate(inst)) continue;
        auto key = make_key(inst);
        auto it = table.find(key);
        if (it != table.end()) {
            inst->replace_all_uses_with(it->second);
            to_remove.push_back(inst);
            info.eliminated_inst_count++;
        } else {
            table.emplace(key, inst);
            added_this_scope.push_back(key);
        }
    }

    for (auto inst : to_remove) {
        inst->remove_self();
    }

    for (auto child : node->children()) {
        cse_domtree_walk(child, table, info);
    }

    for (auto &key : added_this_scope) {
        table.erase(key);
    }
}

static void run_cse_on_function(Function *function, CSEInfo &info) noexcept {
    auto def = function->definition();
    if (!def) return;
    auto dom = compute_dom_tree(function);
    ExprTable table;
    cse_domtree_walk(dom.root(), table, info);
}

}// namespace detail

CSEInfo cse_pass_run_on_function(Function *function) noexcept {
    CSEInfo info;
    detail::run_cse_on_function(function, info);
    return info;
}

CSEInfo cse_pass_run_on_module(Module *module) noexcept {
    CSEInfo info;
    for (auto f : module->function_list()) {
        detail::run_cse_on_function(f, info);
    }
    return info;
}

}// namespace luisa::compute::xir
