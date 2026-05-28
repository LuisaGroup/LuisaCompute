#include <luisa/ast/type_registry.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/div_rem_pairs.h>
#include <luisa/xir/passes/pass_pipeline.h>

namespace luisa::compute::xir {

namespace detail {

struct DivModKey {
    Value *x;
    Value *y;
    [[nodiscard]] bool operator==(const DivModKey &o) const noexcept { return x == o.x && y == o.y; }
    [[nodiscard]] uint64_t hash() const noexcept {
        return luisa::hash_combine({luisa::hash_value(x), luisa::hash_value(y)});
    }
};

static void div_rem_pairs_on_function(FunctionDefinition *def, DivRemPairsInfo &info) noexcept {
    luisa::unordered_map<DivModKey, ArithmeticInst *> div_map;
    luisa::unordered_map<DivModKey, ArithmeticInst *> mod_map;

    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (!inst->isa<ArithmeticInst>()) return;
        auto ari = static_cast<ArithmeticInst *>(inst);
        auto op = ari->op();
        if (op == ArithmeticOp::BINARY_DIV) {
            DivModKey key{ari->operand(0), ari->operand(1)};
            if (div_map.find(key) == div_map.end()) { div_map.emplace(key, ari); }
        } else if (op == ArithmeticOp::BINARY_MOD) {
            DivModKey key{ari->operand(0), ari->operand(1)};
            if (mod_map.find(key) == mod_map.end()) { mod_map.emplace(key, ari); }
        }
    });

    for (auto &[key, mod_inst] : mod_map) {
        auto it = div_map.find(key);
        if (it == div_map.end()) continue;
        auto div_inst = it->second;

        // Insert the replacement at the end of the block so that it
        // comes after both div_inst and mod_inst, avoiding use-before-def
        // when div_inst appears after mod_inst in the same block.
        XIRBuilder b;
        b.set_insertion_point(mod_inst->parent_block());
        auto mul = b.call(div_inst->type(), ArithmeticOp::BINARY_MUL, {div_inst, key.y});
        auto sub = b.call(mod_inst->type(), ArithmeticOp::BINARY_SUB, {key.x, mul});

        mod_inst->replace_all_uses_with(sub);
        mod_inst->remove_self();
        info.merged_pair_count++;
    }
}

}// namespace detail

DivRemPairsInfo div_rem_pairs_pass_run_on_function(FunctionDefinition *def) noexcept {
    DivRemPairsInfo info;
    detail::div_rem_pairs_on_function(def, info);
    return info;
}

DivRemPairsInfo div_rem_pairs_pass_run_on_module(Module *module, PassReport *report) noexcept {
    DivRemPairsInfo info;
    for (auto f : module->function_list()) {
        if (auto def = f->definition()) {
            detail::div_rem_pairs_on_function(def, info);
        }
    }
    if (report != nullptr) {
        report->set("merged_div_rem_pair", info.merged_pair_count);
    }
    return info;
}

}// namespace luisa::compute::xir
