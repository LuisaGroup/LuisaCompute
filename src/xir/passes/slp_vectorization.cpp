#include <luisa/xir/passes/slp_vectorization.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/alloca.h>
#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

[[nodiscard]] static unsigned get_slp_vector_factor(const Type *elem_type) noexcept {
    // Type::vector only supports 2..4.
    auto size = elem_type->size();
    if (size >= 8) return 2; // 64-bit
    return 4;                // 8/16/32-bit
}

[[nodiscard]] static int64_t get_constant_int_value(Constant *c) noexcept {
    if (c == nullptr) return 0;
    if (c->type()->is_int32()) return static_cast<int64_t>(c->as<int32_t>());
    if (c->type()->is_uint32()) return static_cast<int64_t>(c->as<uint32_t>());
    if (c->type()->is_int64()) return c->as<int64_t>();
    if (c->type()->is_uint64()) return static_cast<int64_t>(c->as<uint64_t>());
    return 0;
}

[[nodiscard]] static Constant *create_constant_int(Module *m, const Type *type, int64_t value) noexcept {
    if (type == nullptr || m == nullptr) return nullptr;
    if (type->is_int32()) { int32_t v = static_cast<int32_t>(value); return m->create_constant(type, &v); }
    if (type->is_uint32()) { uint32_t v = static_cast<uint32_t>(value); return m->create_constant(type, &v); }
    if (type->is_int64()) { int64_t v = value; return m->create_constant(type, &v); }
    if (type->is_uint64()) { uint64_t v = static_cast<uint64_t>(value); return m->create_constant(type, &v); }
    return nullptr;
}

[[nodiscard]] static bool get_gep_constant_offset(GEPInst *gep, int64_t &out) noexcept {
    if (gep == nullptr || gep->index_count() != 1) return false;
    auto *index = gep->index(0);
    if (index == nullptr || !index->isa<Constant>()) return false;
    out = get_constant_int_value(static_cast<Constant *>(index));
    return true;
}

enum class TreeEntryState { Vectorize, NeedToGather };

struct TreeEntry;
struct SLPTreeBuilder;

[[nodiscard]] static TreeEntryState get_vectorization_state(const luisa::vector<Instruction *> &bundle) noexcept;

[[nodiscard]] static bool are_pointers_consecutive(const luisa::vector<Instruction *> &bundle, bool is_load) noexcept {
    const AllocaInst *base_alloca = nullptr;
    int64_t first_offset = 0;
    for (size_t i = 0; i < bundle.size(); ++i) {
        Value *ptr = nullptr;
        if (is_load) {
            ptr = static_cast<LoadInst *>(bundle[i])->variable();
        } else {
            ptr = static_cast<StoreInst *>(bundle[i])->variable();
        }
        GEPInst *gep = nullptr;
        if (ptr != nullptr && ptr->isa<GEPInst>()) { gep = static_cast<GEPInst *>(ptr); }
        if (gep == nullptr) return false;
        auto *alloca = trace_pointer_base_local_alloca_inst(gep);
        if (alloca == nullptr) return false;
        int64_t offset = 0;
        if (!get_gep_constant_offset(gep, offset)) return false;
        if (i == 0) {
            base_alloca = alloca;
            first_offset = offset;
        } else {
            if (alloca != base_alloca) return false;
            if (offset != first_offset + static_cast<int64_t>(i)) return false;
        }
    }
    return true;
}

[[nodiscard]] static luisa::vector<luisa::vector<StoreInst *>> collect_store_seeds(FunctionDefinition *def) noexcept {
    luisa::vector<luisa::vector<StoreInst *>> seeds;
    for (auto *bb : def->basic_blocks()) {
        luisa::vector<StoreInst *> run;
        const AllocaInst *current_alloca = nullptr;
        const Type *current_elem_type = nullptr;
        int64_t current_base_offset = 0;
        unsigned max_factor = 0;
        auto flush_run = [&]() noexcept {
            if (run.size() >= 2) {
                seeds.emplace_back(run.begin(), run.end());
            }
            run.clear();
            current_alloca = nullptr;
            current_elem_type = nullptr;
            current_base_offset = 0;
            max_factor = 0;
        };
        for (auto *inst : bb->instructions()) {
            StoreInst *store = nullptr;
            if (inst->isa<StoreInst>()) { store = static_cast<StoreInst *>(inst); }
            if (store != nullptr) {
                auto *value = store->value();
                auto *value_type = value != nullptr ? value->type() : nullptr;
                if (value_type != nullptr && value_type->is_scalar() && value_type->is_arithmetic()) {
                    auto *ptr = store->variable();
                    GEPInst *gep = nullptr;
                    if (ptr != nullptr && ptr->isa<GEPInst>()) { gep = static_cast<GEPInst *>(ptr); }
                    int64_t offset = 0;
                    if (gep != nullptr && get_gep_constant_offset(gep, offset)) {
                        auto *base_alloca = trace_pointer_base_local_alloca_inst(gep);
                        if (base_alloca != nullptr) {
                            bool can_extend = !run.empty() &&
                                              base_alloca == current_alloca &&
                                              value_type == current_elem_type &&
                                              offset == current_base_offset + static_cast<int64_t>(run.size()) &&
                                              run.size() < max_factor;
                            if (!run.empty() && !can_extend) {
                                flush_run();
                            }
                            if (run.empty()) {
                                current_alloca = base_alloca;
                                current_elem_type = value_type;
                                current_base_offset = offset;
                                max_factor = get_slp_vector_factor(value_type);
                            }
                            run.emplace_back(store);
                            if (run.size() == max_factor) {
                                flush_run();
                            }
                            continue;
                        }
                    }
                }
            }
            // Not a qualifying store, or failed checks.
            if (!run.empty()) { flush_run(); }
        }
        if (!run.empty()) { flush_run(); }
    }
    return seeds;
}

struct TreeEntry {
    TreeEntryState state{TreeEntryState::Vectorize};
    luisa::vector<Instruction *> scalars;
    luisa::vector<TreeEntry *> operand_entries; // aligned with operand index, null = gather
    Instruction *vectorized{nullptr};
    bool processed{false};
};

struct SLPTreeBuilder {
    FunctionDefinition *def{nullptr};
    Module *module{nullptr};
    luisa::vector<luisa::unique_ptr<TreeEntry>> entries;
    luisa::unordered_map<Instruction *, TreeEntry *> scalar_to_entry;
    XIRBuilder builder;

    TreeEntry *build_tree(const luisa::vector<Instruction *> &seeds) noexcept;
    TreeEntry *build_tree_rec(luisa::vector<Instruction *> bundle, unsigned depth) noexcept;
    static TreeEntryState get_vectorization_state(const luisa::vector<Instruction *> &bundle) noexcept;
    void vectorize_tree(TreeEntry *root) noexcept;
};

TreeEntry *SLPTreeBuilder::build_tree(const luisa::vector<Instruction *> &seeds) noexcept {
    if (seeds.size() < 2) return nullptr;
    return build_tree_rec(seeds, 0);
}

TreeEntry *SLPTreeBuilder::build_tree_rec(luisa::vector<Instruction *> bundle, unsigned depth) noexcept {
    if (depth > 12 || bundle.empty()) return nullptr;
    if (auto it = scalar_to_entry.find(bundle[0]); it != scalar_to_entry.end()) {
        return it->second;
    }
    {
        luisa::unordered_set<Instruction *> seen;
        for (auto *s : bundle) {
            if (!seen.insert(s).second) return nullptr;
        }
    }
    auto entry = luisa::make_unique<TreeEntry>();
    entry->scalars = std::move(bundle);
    entry->state = get_vectorization_state(entry->scalars);
    if (entry->state == TreeEntryState::Vectorize) {
        entry->operand_entries.resize(entry->scalars[0]->operand_count(), nullptr);
        for (size_t op_idx = 0; op_idx < entry->scalars[0]->operand_count(); ++op_idx) {
            auto tag = entry->scalars[0]->derived_instruction_tag();
            if (tag == DerivedInstructionTag::LOAD && op_idx == 0) continue;
            if (tag == DerivedInstructionTag::STORE && op_idx == 0) continue;
            auto *op_type = entry->scalars[0]->operand(op_idx)->type();
            if (op_type == nullptr || !op_type->is_scalar() || !op_type->is_arithmetic()) continue;
            luisa::vector<Instruction *> op_bundle;
            op_bundle.reserve(entry->scalars.size());
            bool ok = true;
            for (auto *inst : entry->scalars) {
                auto *op = inst->operand(op_idx);
                if (op == nullptr || !op->isa<Instruction>() || op->type() != op_type) {
                    ok = false;
                    break;
                }
                op_bundle.emplace_back(static_cast<Instruction *>(op));
            }
            if (!ok) {
                // Cannot form an instruction bundle for this operand; gather it instead.
                continue;
            }
            auto *child = build_tree_rec(std::move(op_bundle), depth + 1);
            if (child != nullptr && child->state == TreeEntryState::Vectorize) {
                entry->operand_entries[op_idx] = child;
            }
            // Otherwise we gather this operand from the scalar instructions.
        }
    }
    auto *entry_ptr = entry.get();
    entries.emplace_back(std::move(entry));
    for (auto *s : entry_ptr->scalars) {
        scalar_to_entry[s] = entry_ptr;
    }
    return entry_ptr;
}

TreeEntryState SLPTreeBuilder::get_vectorization_state(const luisa::vector<Instruction *> &bundle) noexcept {
    return ::luisa::compute::xir::detail::get_vectorization_state(bundle);
}

TreeEntryState get_vectorization_state(const luisa::vector<Instruction *> &bundle) noexcept {
    if (bundle.size() < 2) return TreeEntryState::NeedToGather;
    auto tag = bundle[0]->derived_instruction_tag();
    for (auto *s : bundle) {
        if (s->derived_instruction_tag() != tag) return TreeEntryState::NeedToGather;
    }
    switch (tag) {
        case DerivedInstructionTag::ARITHMETIC: {
            auto *first = static_cast<ArithmeticInst *>(bundle[0]);
            auto op = first->op();
            auto *result_type = first->type();
            if (result_type == nullptr || !result_type->is_scalar() || !result_type->is_arithmetic()) {
                return TreeEntryState::NeedToGather;
            }
            for (auto *s : bundle) {
                auto *a = static_cast<ArithmeticInst *>(s);
                if (a->op() != op) return TreeEntryState::NeedToGather;
                if (s->type() != result_type) return TreeEntryState::NeedToGather;
            }
            return TreeEntryState::Vectorize;
        }
        case DerivedInstructionTag::CAST: {
            auto *first = static_cast<CastInst *>(bundle[0]);
            auto op = first->op();
            auto *result_type = first->type();
            if (result_type == nullptr || !result_type->is_scalar() || !result_type->is_arithmetic()) {
                return TreeEntryState::NeedToGather;
            }
            for (auto *s : bundle) {
                auto *c = static_cast<CastInst *>(s);
                if (c->op() != op) return TreeEntryState::NeedToGather;
                if (s->type() != result_type) return TreeEntryState::NeedToGather;
            }
            return TreeEntryState::Vectorize;
        }
        case DerivedInstructionTag::LOAD: {
            auto *result_type = bundle[0]->type();
            if (result_type == nullptr || !result_type->is_scalar() || !result_type->is_arithmetic()) {
                return TreeEntryState::NeedToGather;
            }
            for (auto *s : bundle) {
                if (s->type() != result_type) return TreeEntryState::NeedToGather;
            }
            if (!are_pointers_consecutive(bundle, true)) return TreeEntryState::NeedToGather;
            return TreeEntryState::Vectorize;
        }
        case DerivedInstructionTag::STORE: {
            auto *first = static_cast<StoreInst *>(bundle[0]);
            auto *value_type = first->value()->type();
            if (value_type == nullptr || !value_type->is_scalar() || !value_type->is_arithmetic()) {
                return TreeEntryState::NeedToGather;
            }
            for (auto *s : bundle) {
                auto *st = static_cast<StoreInst *>(s);
                if (st->value()->type() != value_type) return TreeEntryState::NeedToGather;
            }
            if (!are_pointers_consecutive(bundle, false)) return TreeEntryState::NeedToGather;
            return TreeEntryState::Vectorize;
        }
        default:
            return TreeEntryState::NeedToGather;
    }
}

void SLPTreeBuilder::vectorize_tree(TreeEntry *root) noexcept {
    if (root == nullptr || root->processed) return;
    for (auto *child : root->operand_entries) {
        if (child != nullptr) { vectorize_tree(child); }
    }
    auto *first = root->scalars[0];
    const Type *elem_type = nullptr;
    if (first->derived_instruction_tag() == DerivedInstructionTag::STORE) {
        elem_type = static_cast<StoreInst *>(first)->value()->type();
    } else {
        elem_type = first->type();
    }
    if (elem_type == nullptr) return;
    auto *vec_type = Type::vector(elem_type, root->scalars.size());
    if (vec_type == nullptr) return;
    builder.set_insertion_point(first);
    luisa::vector<Value *> vec_operands;
    vec_operands.reserve(first->operand_count());
    for (size_t i = 0; i < first->operand_count(); ++i) {
        auto tag = first->derived_instruction_tag();
        if ((tag == DerivedInstructionTag::LOAD || tag == DerivedInstructionTag::STORE) && i == 0) {
            continue;// skip pointer operand
        }
        auto *child = (i < root->operand_entries.size()) ? root->operand_entries[i] : nullptr;
        if (child != nullptr && child->vectorized != nullptr) {
            vec_operands.emplace_back(child->vectorized);
        } else {
            luisa::vector<Value *> scalar_ops;
            scalar_ops.reserve(root->scalars.size());
            for (auto *s : root->scalars) {
                scalar_ops.emplace_back(s->operand(i));
            }
            vec_operands.emplace_back(builder.call(vec_type, ArithmeticOp::AGGREGATE, scalar_ops));
        }
    }
    switch (first->derived_instruction_tag()) {
        case DerivedInstructionTag::ARITHMETIC: {
            auto *a = static_cast<ArithmeticInst *>(first);
            root->vectorized = builder.call(vec_type, a->op(), vec_operands);
            break;
        }
        case DerivedInstructionTag::CAST: {
            auto *c = static_cast<CastInst *>(first);
            if (!vec_operands.empty()) {
                root->vectorized = builder.cast_(vec_type, c->op(), vec_operands[0]);
            }
            break;
        }
        case DerivedInstructionTag::LOAD: {
            auto *gep = static_cast<GEPInst *>(static_cast<LoadInst *>(first)->variable());
            int64_t offset = 0;
            if (!get_gep_constant_offset(gep, offset)) { return; }
            auto *vec_gep = builder.gep(vec_type, gep->base(), {create_constant_int(module, gep->index(0)->type(), offset)});
            root->vectorized = builder.load(vec_type, vec_gep);
            break;
        }
        case DerivedInstructionTag::STORE: {
            if (vec_operands.empty()) { return; }
            auto *gep = static_cast<GEPInst *>(static_cast<StoreInst *>(first)->variable());
            int64_t offset = 0;
            if (!get_gep_constant_offset(gep, offset)) { return; }
            auto *vec_gep = builder.gep(vec_type, gep->base(), {create_constant_int(module, gep->index(0)->type(), offset)});
            root->vectorized = builder.store(vec_gep, vec_operands[0]);
            break;
        }
        default:
            break;
    }
    root->processed = true;
}

}// namespace detail

SLPVectorizationInfo slp_vectorization_pass_run_on_function(Function *function) noexcept {
    auto def = function->definition();
    if (!def) return {};
    SLPVectorizationInfo info;
    detail::SLPTreeBuilder builder;
    builder.def = def;
    builder.module = def->parent_module();
    auto seeds = detail::collect_store_seeds(def);
    for (auto &chain : seeds) {
        if (chain.size() < 2) continue;
        luisa::vector<Instruction *> seed_bundle(chain.begin(), chain.end());
        auto *root = builder.build_tree(seed_bundle);
        if (!root || root->state != detail::TreeEntryState::Vectorize) continue;
        builder.vectorize_tree(root);
        // Remove original scalar stores now that the vector store is in place.
        if (root->scalars[0]->derived_instruction_tag() == DerivedInstructionTag::STORE) {
            for (auto *s : root->scalars) { static_cast<StoreInst *>(s)->remove_self(); }
        }
        info.vectorized_tree_count++;
        info.vectorized_inst_count += chain.size();
    }
    return info;
}

SLPVectorizationInfo slp_vectorization_pass_run_on_module(Module *module, PassReport *report) noexcept {
    SLPVectorizationInfo info;
    for (auto *f : module->function_list()) {
        auto func_info = slp_vectorization_pass_run_on_function(f);
        info.vectorized_tree_count += func_info.vectorized_tree_count;
        info.vectorized_inst_count += func_info.vectorized_inst_count;
    }
    if (report) {
        report->set("vectorized_tree_count", info.vectorized_tree_count);
        report->set("vectorized_inst_count", info.vectorized_inst_count);
    }
    return info;
}

}// namespace luisa::compute::xir
