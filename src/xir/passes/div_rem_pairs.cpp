#include <luisa/ast/type_registry.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/dom_tree.h>
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

[[nodiscard]] static DivModKey div_mod_key(ArithmeticInst *inst) noexcept {
    return DivModKey{inst->operand(0), inst->operand(1)};
}

static void div_rem_pairs_on_function(FunctionDefinition *def, DivRemPairsInfo &info) noexcept {
    if (def == nullptr || def->body_block() == nullptr) { return; }
    luisa::vector<ArithmeticInst *> div_insts;
    luisa::vector<ArithmeticInst *> mod_insts;
    luisa::unordered_map<Instruction *, size_t> instruction_indices;
    auto index = 0u;

    auto is_integer_type = [](const Type *type) noexcept {
        auto t = type;
        while (t->is_vector() || t->is_matrix() || t->is_array()) { t = t->element(); }
        return t->is_int() || t->is_uint();
    };

    def->traverse_instructions([&](Instruction *inst) noexcept {
        instruction_indices.emplace(inst, index++);
        if (!inst->isa<ArithmeticInst>()) return;
        auto ari = static_cast<ArithmeticInst *>(inst);
        if (!is_integer_type(ari->type())) return;
        auto op = ari->op();
        if (op == ArithmeticOp::BINARY_DIV) {
            div_insts.emplace_back(ari);
        } else if (op == ArithmeticOp::BINARY_MOD) {
            mod_insts.emplace_back(ari);
        }
    });

    auto dom_tree = compute_dom_tree(def);
    auto dominates_inst = [&](Instruction *div_inst, Instruction *mod_inst) noexcept {
        auto div_block = div_inst->parent_block();
        auto mod_block = mod_inst->parent_block();
        if (div_block == mod_block) {
            return instruction_indices.at(div_inst) < instruction_indices.at(mod_inst);
        }
        return dom_tree.dominates(div_block, mod_block);
    };

    for (auto *mod_inst : mod_insts) {
        if (!mod_inst->is_linked()) { continue; }
        auto key = div_mod_key(mod_inst);
        ArithmeticInst *div_inst = nullptr;
        for (auto *candidate : div_insts) {
            if (candidate->is_linked() &&
                div_mod_key(candidate) == key &&
                dominates_inst(candidate, mod_inst)) {
                div_inst = candidate;
                break;
            }
        }
        if (div_inst == nullptr) { continue; }

        XIRBuilder b;
        b.set_insertion_point(mod_inst->prev());
        auto x = mod_inst->operand(0);
        auto y = mod_inst->operand(1);
        auto mul = b.call(div_inst->type(), ArithmeticOp::BINARY_MUL, {div_inst, y});
        auto sub = b.call(mod_inst->type(), ArithmeticOp::BINARY_SUB, {x, mul});
        // The subtraction is the semantic replacement for the remainder.
        // Keep source/debug metadata on that value rather than on the
        // implementation-detail multiplication.
        for (auto *metadata : mod_inst->metadata_list()) {
            sub->metadata_list().push_front(metadata->clone());
        }

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
    if (module != nullptr) {
        for (auto f : module->function_list()) {
            if (auto def = f->definition()) {
                detail::div_rem_pairs_on_function(def, info);
            }
        }
    }
    if (report != nullptr) {
        report->set("merged_div_rem_pair", info.merged_pair_count);
    }
    return info;
}

}// namespace luisa::compute::xir
