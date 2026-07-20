#include <luisa/xir/passes/sroa.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/metadata/reg2mem_spill.h>
#include <luisa/xir/constant.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

// Decompose only one level: struct→members, array→elements.
// Does NOT recurse into nested aggregate members.
static void collect_elem_types(const Type *type, luisa::vector<const Type *> &elems,
                               bool decompose_vectors, bool decompose_matrices) noexcept {
    if (type->is_structure()) {
        auto members = type->members();
        elems.assign(members.begin(), members.end());
    } else if (type->is_array()) {
        auto elem = type->element();
        for (size_t i = 0; i < type->dimension(); ++i) {
            elems.push_back(elem);
        }
    } else if (type->is_vector() && decompose_vectors) {
        auto elem = type->element();
        for (size_t i = 0; i < type->dimension(); ++i) {
            elems.push_back(elem);
        }
    } else if (type->is_matrix() && decompose_matrices) {
        auto col_type = Type::vector(type->element(), type->dimension());
        for (size_t i = 0; i < type->dimension(); ++i) {
            elems.push_back(col_type);
        }
    }
}

[[nodiscard]] static bool is_sroa_candidate(AllocaInst *alloca, const SROAOptions &options) noexcept {
    if (alloca->op() != AllocaOp::LOCAL) return false;
    auto type = alloca->type();
    if (type->is_scalar()) return false;
    if (!(type->is_structure() || type->is_array() ||
          (type->is_vector() && options.decompose_vectors) ||
          (type->is_matrix() && options.decompose_matrices))) {
        return false;
    }

    luisa::vector<const Type *> elem_types;
    collect_elem_types(type, elem_types, options.decompose_vectors, options.decompose_matrices);
    if (elem_types.size() <= 1u) { return false; }

    luisa::vector<const Instruction *> work_list;
    luisa::unordered_set<const Instruction *> visited;
    for (auto &&use : alloca->use_list()) {
        if (auto user = use->user(); user != nullptr && user->isa<Instruction>()) {
            work_list.push_back(static_cast<const Instruction *>(user));
        }
    }
    while (!work_list.empty()) {
        auto u = work_list.back();
        work_list.pop_back();
        if (!visited.emplace(u).second) continue;
        if (u->isa<LoadInst>() || u->isa<StoreInst>()) continue;
        if (u->isa<GEPInst>()) {
            auto gep = static_cast<const GEPInst *>(u);
            // Only the first index of a GEP directly rooted at the alloca
            // chooses which replacement alloca to use. It must be constant.
            // Indices below that level may remain dynamic in the rebuilt GEP.
            if (gep->base() == alloca) {
                uint64_t element_index = 0u;
                if (gep->index_uses().empty() ||
                    !try_decode_constant_nonnegative_integer(
                        gep->index_uses().front()->value(), element_index) ||
                    element_index >= elem_types.size()) {
                    return false;
                }
            }
            for (auto &&gep_use : u->use_list()) {
                if (auto gep_user = gep_use->user();
                    gep_user != nullptr && gep_user->isa<Instruction>()) {
                    work_list.push_back(static_cast<const Instruction *>(gep_user));
                }
            }
        } else {
            return false;
        }
    }
    return true;
}

static void decompose_alloca(AllocaInst *alloca, SROAInfo &info, XIRBuilder &builder,
                             luisa::unordered_map<const Instruction *, Instruction *> &replacement_map,
                             const SROAOptions &options) noexcept {
    auto type = alloca->type();
    luisa::vector<const Type *> elem_types;
    collect_elem_types(type, elem_types, options.decompose_vectors, options.decompose_matrices);

    if (elem_types.size() <= 1) return;

    // Create one replacement alloca for each top-level element.
    builder.set_insertion_point(alloca);
    luisa::vector<AllocaInst *> element_allocas;
    auto original_name = alloca->name();
    auto spill_metadata = alloca->find_metadata<Reg2MemSpillMD>();
    for (auto et : elem_types) {
        auto sa = builder.alloca_local(et);
        if (original_name.has_value()) {
            sa->set_name(luisa::format("{}_{}", original_name.value(), element_allocas.size()));
        }
        if (spill_metadata != nullptr) {
            sa->metadata_list().push_front(spill_metadata->clone());
        }
        element_allocas.push_back(sa);
        info.inserted_alloca_count++;
    }

    // Collect GEP instructions that use this alloca
    luisa::vector<GEPInst *> geps;
    for (auto &&use : alloca->use_list()) {
        if (auto user = use->user(); user != nullptr && user->isa<GEPInst>()) {
            geps.push_back(static_cast<GEPInst *>(user));
        }
    }

    for (auto gep : geps) {
        LUISA_ASSERT(!gep->index_uses().empty(), "SROA: GEP has no indices.");
        auto first_idx_val = gep->index_uses()[0]->value();
        uint64_t elem_idx = 0u;
        LUISA_ASSERT(try_decode_constant_nonnegative_integer(first_idx_val, elem_idx),
                     "SROA: expected a nonnegative constant integer GEP index.");
        LUISA_ASSERT(elem_idx < element_allocas.size(), "SROA: GEP index out of bounds.");

        auto target_alloca = element_allocas[elem_idx];

        if (gep->index_count() > 1) {
            builder.set_insertion_point(gep);
            luisa::vector<Value *> remaining_indices;
            for (size_t i = 1; i < gep->index_count(); ++i) {
                remaining_indices.push_back(gep->index_uses()[i]->value());
            }
            auto new_gep = builder.gep(gep->type(), target_alloca, remaining_indices);
            replacement_map[gep] = new_gep;
        } else {
            replacement_map[gep] = target_alloca;
        }

        gep->replace_all_uses_with(replacement_map[gep]);
        gep->remove_self();
    }

    // Handle direct loads/stores on the aggregate alloca
    luisa::vector<Instruction *> direct_users;
    for (auto &&use : alloca->use_list()) {
        if (auto user = use->user(); user != nullptr && user->isa<Instruction>()) {
            direct_users.push_back(static_cast<Instruction *>(user));
        }
    }

    for (auto user : direct_users) {
        if (user->isa<LoadInst>()) {
            auto load = static_cast<LoadInst *>(user);
            builder.set_insertion_point(load);
            luisa::vector<Value *> elem_values;
            for (auto sa : element_allocas) {
                elem_values.push_back(builder.load(sa->type(), sa));
            }
            auto replacement = builder.call(type, ArithmeticOp::AGGREGATE, elem_values);
            load->replace_all_uses_with(replacement);
            load->remove_self();
        } else if (user->isa<StoreInst>()) {
            auto store = static_cast<StoreInst *>(user);
            builder.set_insertion_point(store);
            auto val = store->value();
            for (size_t i = 0; i < elem_types.size(); ++i) {
                auto idx_val = static_cast<uint32_t>(i);
                auto idx_const = alloca->parent_module()->create_constant(Type::of<uint32_t>(), &idx_val);
                auto extract = builder.call(elem_types[i], ArithmeticOp::EXTRACT, {val, idx_const});
                builder.store(element_allocas[i], extract);
            }
            store->remove_self();
        }
    }

    alloca->remove_self();
    info.decomposed_alloca_count++;
}

static void sroa_pass_on_function(Function *function, SROAInfo &info, const SROAOptions &options) noexcept {
    auto def = function->definition();
    if (!def) return;

    luisa::vector<AllocaInst *> candidates;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<AllocaInst>()) {
            auto alloca = static_cast<AllocaInst *>(inst);
            if (is_sroa_candidate(alloca, options)) {
                candidates.push_back(alloca);
            }
        }
    });

    if (candidates.empty()) return;

    XIRBuilder builder;
    luisa::unordered_map<const Instruction *, Instruction *> replacement_map;
    for (auto alloca : candidates) {
        decompose_alloca(alloca, info, builder, replacement_map, options);
    }
}

}// namespace detail

SROAInfo sroa_pass_run_on_function(Function *function, SROAOptions options) noexcept {
    SROAInfo info;
    detail::sroa_pass_on_function(function, info, options);
    return info;
}

SROAInfo sroa_pass_run_on_module(Module *module, SROAOptions options, PassReport *report) noexcept {
    SROAInfo info;
    for (auto f : module->function_list()) {
        detail::sroa_pass_on_function(f, info, options);
    }
    if (report != nullptr) {
        report->set("decomposed_alloca", info.decomposed_alloca_count);
        report->set("inserted_alloca", info.inserted_alloca_count);
    }
    return info;
}

}// namespace luisa::compute::xir
