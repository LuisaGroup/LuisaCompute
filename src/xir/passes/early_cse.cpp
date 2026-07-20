#include <luisa/core/stl/hash.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/early_cse.h>
#include <luisa/xir/passes/pass_pipeline.h>

namespace luisa::compute::xir {

namespace {

struct InstKey {
    DerivedInstructionTag tag;
    luisa::vector<const Value *> operands;
    const Type *type;
    uint64_t sub_op{0};

    bool operator==(const InstKey &other) const noexcept {
        if (tag != other.tag || type != other.type || sub_op != other.sub_op ||
            operands.size() != other.operands.size()) {
            return false;
        }
        for (size_t i = 0; i < operands.size(); ++i) {
            if (operands[i] != other.operands[i]) { return false; }
        }
        return true;
    }
};

struct InstKeyHash {
    size_t operator()(const InstKey &key) const noexcept {
        uint64_t h = key.sub_op;
        h = luisa::hash_combine({h, static_cast<uint64_t>(key.tag)});
        h = luisa::hash_combine({h, reinterpret_cast<uint64_t>(key.type)});
        for (auto *op : key.operands) {
            h = luisa::hash_combine({h, reinterpret_cast<uint64_t>(op)});
        }
        return static_cast<size_t>(h);
    }
};

[[nodiscard]] bool is_side_effect_free(const Instruction *inst) noexcept {
    switch (inst->derived_instruction_tag()) {
        case DerivedInstructionTag::ARITHMETIC:
        case DerivedInstructionTag::CAST:
        case DerivedInstructionTag::GEP:
        case DerivedInstructionTag::RESOURCE_QUERY: return true;
        default: return false;
    }
}

[[nodiscard]] InstKey make_key(const Instruction *inst) noexcept {
    InstKey key;
    key.tag = inst->derived_instruction_tag();
    key.type = inst->type();
    key.operands.reserve(inst->operand_count());
    for (size_t i = 0; i < inst->operand_count(); ++i) {
        key.operands.push_back(inst->operand(i));
    }
    // Capture instruction sub-op (e.g., ADD vs MUL for arithmetic).
    if (key.tag == DerivedInstructionTag::ARITHMETIC) {
        key.sub_op = static_cast<uint64_t>(static_cast<const ArithmeticInst *>(inst)->op());
    } else if (key.tag == DerivedInstructionTag::CAST) {
        key.sub_op = static_cast<uint64_t>(static_cast<const CastInst *>(inst)->op());
    } else if (key.tag == DerivedInstructionTag::RESOURCE_QUERY) {
        key.sub_op = static_cast<uint64_t>(static_cast<const ResourceQueryInst *>(inst)->op());
    }
    return key;
}

[[nodiscard]] EarlyCSEInfo early_cse_on_definition(FunctionDefinition *def) noexcept {
    EarlyCSEInfo info{};
    luisa::unordered_map<InstKey, Instruction *, InstKeyHash> seen;
    luisa::vector<Instruction *> to_remove;

    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        seen.clear();
        for (auto *inst : bb->instructions()) {
            if (!is_side_effect_free(inst)) { continue; }
            auto key = make_key(inst);
            auto it = seen.find(key);
            if (it != seen.end()) {
                inst->replace_all_uses_with(it->second);
                to_remove.push_back(inst);
            } else {
                seen.emplace(std::move(key), inst);
            }
        }
    });

    for (auto *inst : to_remove) {
        inst->remove_self();
        ++info.eliminated_inst_count;
    }

    return info;
}

}// namespace

EarlyCSEInfo early_cse_pass_run_on_function(Function *function) noexcept {
    if (function == nullptr || !function->is_definition()) { return {}; }
    return early_cse_on_definition(static_cast<FunctionDefinition *>(function));
}

EarlyCSEInfo early_cse_pass_run_on_module(Module *module, PassReport *report) noexcept {
    EarlyCSEInfo total{};
    if (module == nullptr) { return total; }
    for (auto *f : module->function_list()) {
        auto info = early_cse_pass_run_on_function(f);
        total.eliminated_inst_count += info.eliminated_inst_count;
    }
    if (report != nullptr) {
        report->set("early_cse_eliminated", total.eliminated_inst_count);
    }
    return total;
}

}// namespace luisa::compute::xir
