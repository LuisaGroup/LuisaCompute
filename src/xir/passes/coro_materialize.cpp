#include "helpers.h"

#include <luisa/ast/type.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/passes/coro_materialize.h>
#include <luisa/xir/passes/coro_split.h>

namespace luisa::compute::xir {

namespace detail {

static constexpr size_t FRAME_RESERVED_FIELD_COUNT = 3u;
static constexpr uint32_t FRAME_FIELD_TOKEN_CMAT = 1u;

struct RegisterInfo {
    luisa::string name;
    const Type *type;
};

[[nodiscard]] static Value *find_frame_operand(CallableFunction *func) noexcept {
    if (func == nullptr || func->definition() == nullptr) { return nullptr; }
    Value *frame = nullptr;
    func->traverse_instructions([&](Instruction *inst) noexcept {
        if (frame != nullptr) { return; }
        switch (inst->derived_instruction_tag()) {
            case DerivedInstructionTag::CORO_SUSPEND:
                frame = static_cast<CoroSuspendInst *>(inst)->frame();
                break;
            case DerivedInstructionTag::CORO_RESUME:
                frame = static_cast<CoroResumeInst *>(inst)->frame();
                break;
            default:
                break;
        }
    });
    return frame;
}

[[nodiscard]] static bool is_token_store(Value *frame_arg, StoreInst *store) noexcept {
    auto *var = store->variable();
    if (!var->isa<Instruction>()) { return false; }
    auto *inst = static_cast<Instruction *>(var);
    if (inst->derived_instruction_tag() != DerivedInstructionTag::GEP) { return false; }
    auto *gep = static_cast<GEPInst *>(inst);
    if (gep->base() != frame_arg) { return false; }
    if (gep->index_count() != 1u) { return false; }
    auto *idx = gep->index(0u);
    if (!idx->isa<Constant>()) { return false; }
    return static_cast<Constant *>(idx)->as<uint32_t>() == FRAME_FIELD_TOKEN_CMAT;
}

[[nodiscard]] static luisa::vector<RegisterInfo> collect_registers(
    Module *mod, const luisa::unordered_set<luisa::string> *filter = nullptr) noexcept {
    luisa::unordered_set<luisa::string> seen;
    luisa::vector<RegisterInfo> regs;

    for (auto *f : mod->function_list()) {
        if (!f->isa<CallableFunction>() || f->definition() == nullptr) { continue; }
        auto *def = f->definition();
        def->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->derived_instruction_tag() != DerivedInstructionTag::ALLOCA) { return; }
            auto *alloca = static_cast<AllocaInst *>(inst);
            if (!alloca->is_local()) { return; }
            auto name_opt = alloca->name();
            if (!name_opt.has_value()) { return; }
            luisa::string name(name_opt.value());
            if (filter != nullptr && !filter->contains(name)) { return; }
            if (seen.insert(name).second) {
                regs.push_back({std::move(name), alloca->type()});
            }
        });
    }
    luisa::sort(regs.begin(), regs.end(), [](auto &a, auto &b) noexcept {
        return a.name < b.name;
    });
    return regs;
}

[[nodiscard]] static const Type *build_frame_type(const luisa::vector<RegisterInfo> &regs) noexcept {
    luisa::vector<const Type *> fields;
    fields.push_back(Type::of<uint3>());   // [0] coro_id
    fields.push_back(Type::of<uint32_t>());// [1] token
    fields.push_back(Type::of<uint32_t>());// [2] skip
    for (auto &reg : regs) { fields.push_back(reg.type); }
    return Type::structure(fields);
}

[[nodiscard]] static luisa::unordered_map<luisa::string, size_t> build_field_map(
    const luisa::vector<RegisterInfo> &regs) noexcept {
    luisa::unordered_map<luisa::string, size_t> map;
    for (size_t i = 0u; i < regs.size(); ++i) {
        map.emplace(regs[i].name, FRAME_RESERVED_FIELD_COUNT + i);
    }
    return map;
}

static void store_user_vars_to_frame(XIRBuilder &b, Module *mod, Value *frame_arg,
                                     const luisa::vector<RegisterInfo> &regs,
                                     const luisa::unordered_map<luisa::string, size_t> &field_map,
                                     const luisa::unordered_map<luisa::string, Value *> &local_map,
                                     const luisa::unordered_set<luisa::string> *live_filter,
                                     size_t &count) noexcept {
    for (auto &reg : regs) {
        if (live_filter != nullptr && !live_filter->contains(reg.name)) { continue; }
        auto it = local_map.find(reg.name);
        if (it == local_map.end()) { continue; }
        auto *local_val = it->second;
        auto fi = field_map.at(reg.name);
        auto *idx_c = mod->create_constant(Type::of<uint32_t>(), &fi);
        auto *gep = b.gep(reg.type, frame_arg, {idx_c});
        if (local_val->isa<AllocaInst>()) {
            auto *loaded = b.load(reg.type, local_val);
            b.store(gep, loaded);
        } else {
            b.store(gep, local_val);
        }
        count++;
    }
}

static void load_user_vars_from_frame(XIRBuilder &b, Module *mod, Value *frame_arg,
                                      const luisa::vector<RegisterInfo> &regs,
                                      const luisa::unordered_map<luisa::string, size_t> &field_map,
                                      const luisa::unordered_map<luisa::string, Value *> &local_map,
                                      const luisa::unordered_set<luisa::string> *live_filter,
                                      size_t &count) noexcept {
    for (auto &reg : regs) {
        if (live_filter != nullptr && !live_filter->contains(reg.name)) { continue; }
        auto it = local_map.find(reg.name);
        if (it == local_map.end()) { continue; }
        auto *local_val = it->second;
        auto fi = field_map.at(reg.name);
        auto *idx_c = mod->create_constant(Type::of<uint32_t>(), &fi);
        auto *gep = b.gep(reg.type, frame_arg, {idx_c});
        auto *loaded = b.load(reg.type, gep);
        if (local_val->isa<AllocaInst>()) {
            b.store(local_val, loaded);
        }
        count++;
    }
}

static void process_callable(Module *mod, CallableFunction *func, Value *frame_arg,
                             const luisa::vector<RegisterInfo> &regs,
                             const luisa::unordered_map<luisa::string, size_t> &field_map,
                             const luisa::unordered_set<luisa::string> *live_in,
                             const luisa::unordered_set<luisa::string> *live_out,
                             bool materialize_user_vars,
                             CoroMaterializeInfo &info) noexcept {

    if (func == nullptr || func->definition() == nullptr || frame_arg == nullptr) { return; }

    luisa::unordered_map<luisa::string, Value *> local_map;
    func->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->derived_instruction_tag() == DerivedInstructionTag::ALLOCA) {
            auto *alloca = static_cast<AllocaInst *>(inst);
            if (!alloca->is_local()) { return; }
            auto name_opt = alloca->name();
            if (name_opt.has_value()) {
                local_map.emplace(name_opt.value(), alloca);
            }
        }
    });
    XIRBuilder b;

    luisa::vector<CoroSuspendInst *> suspends;
    func->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->derived_instruction_tag() == DerivedInstructionTag::CORO_SUSPEND) {
            suspends.push_back(static_cast<CoroSuspendInst *>(inst));
        }
    });
    for (auto *s : suspends) {
        auto token = s->token();
        auto *prev = s->prev();
        if (prev != nullptr && !prev->is_sentinel()) {
            b.set_insertion_point(prev);
        } else {
            b.set_insertion_point(s->parent_block());
        }
        if (materialize_user_vars) {
            store_user_vars_to_frame(b, mod, frame_arg, regs, field_map, local_map,
                                     live_out, info.store_inserted_count);
        }

        auto *field_token = mod->create_constant(Type::of<uint32_t>(), &FRAME_FIELD_TOKEN_CMAT);
        auto *gep0 = b.gep(Type::of<uint32_t>(), frame_arg, {field_token});
        auto *tok_c = mod->create_constant(Type::of<uint32_t>(), &token);
        b.store(gep0, tok_c);
        b.return_void();
        s->remove_self();
        info.suspend_lowered_count++;
    }

    luisa::vector<CoroTerminateInst *> terms;
    func->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->derived_instruction_tag() == DerivedInstructionTag::CORO_TERMINATE) {
            terms.push_back(static_cast<CoroTerminateInst *>(inst));
        }
    });
    for (auto *t : terms) {
        auto *prev = t->prev();
        if (prev != nullptr && !prev->is_sentinel()) {
            b.set_insertion_point(prev);
        } else {
            b.set_insertion_point(t->parent_block());
        }
        auto *field_token = mod->create_constant(Type::of<uint32_t>(), &FRAME_FIELD_TOKEN_CMAT);
        auto *gep0 = b.gep(Type::of<uint32_t>(), frame_arg, {field_token});
        auto term_tok = TERMINAL_TOKEN;
        auto *term_c = mod->create_constant(Type::of<uint32_t>(), &term_tok);
        b.store(gep0, term_c);
        b.return_void();
        t->remove_self();
        info.terminal_lowered_count++;
    }

    size_t token_store_count = 0;
    if (suspends.empty()) {
        luisa::vector<StoreInst *> token_stores;
        func->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->derived_instruction_tag() == DerivedInstructionTag::STORE) {
                auto *store = static_cast<StoreInst *>(inst);
                if (is_token_store(frame_arg, store)) {
                    token_stores.push_back(store);
                }
            }
        });
        for (auto *ts : token_stores) {
            auto *prev = ts->prev();
            if (prev != nullptr && !prev->is_sentinel()) {
                b.set_insertion_point(prev);
            } else {
                b.set_insertion_point(ts->parent_block());
            }
            if (materialize_user_vars) {
                store_user_vars_to_frame(b, mod, frame_arg, regs, field_map, local_map,
                                         live_out, info.store_inserted_count);
            }
        }
    }

    luisa::vector<CoroResumeInst *> resumes;
    func->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->derived_instruction_tag() == DerivedInstructionTag::CORO_RESUME) {
            resumes.push_back(static_cast<CoroResumeInst *>(inst));
        }
    });
    for (auto *r : resumes) {
        b.set_insertion_point(r);
        if (materialize_user_vars) {
            load_user_vars_from_frame(b, mod, frame_arg, regs, field_map, local_map,
                                      live_in, info.load_inserted_count);
        }
        r->remove_self();
        info.resume_lowered_count++;
    }

}

[[nodiscard]] static luisa::unordered_set<luisa::string> collect_live_register_names(
    const CoroCfgDistillResult *cfg) noexcept {
    luisa::unordered_set<luisa::string> names;
    if (cfg == nullptr) { return names; }
    for (auto &scope : cfg->scopes) {
        for (auto &name : scope.live_in_variables) { names.emplace(name); }
        for (auto &name : scope.live_out_variables) { names.emplace(name); }
    }
    return names;
}

static void append_field_indices(luisa::vector<size_t> &dst,
                                 const luisa::vector<luisa::string> &names,
                                 const luisa::unordered_map<luisa::string, size_t> &field_map) noexcept {
    luisa::unordered_set<size_t> seen;
    for (auto &name : names) {
        if (auto it = field_map.find(name); it != field_map.end()) {
            if (seen.emplace(it->second).second) {
                dst.emplace_back(it->second);
            }
        }
    }
    luisa::sort(dst.begin(), dst.end());
}

static void populate_transition_edges(CoroMaterializeInfo &info,
                                      const CoroCfgDistillResult *cfg,
                                      const luisa::unordered_map<luisa::string, size_t> &field_map) noexcept {
    if (cfg == nullptr) { return; }
    info.edges.clear();
    for (size_t from = 0u; from < cfg->edges.size(); ++from) {
        for (auto to : cfg->edges[from]) {
            if (to >= cfg->scopes.size()) { continue; }
            CoroMaterializeInfo::TransitionEdge edge;
            edge.from_scope = from;
            edge.to_scope = to;
            append_field_indices(edge.store_fields, cfg->scopes[from].live_out_variables, field_map);
            append_field_indices(edge.load_fields, cfg->scopes[to].live_in_variables, field_map);
            info.edges.emplace_back(std::move(edge));
        }
    }
}

static void append_value_field_indices(luisa::vector<size_t> &dst,
                                       const luisa::vector<Value *> &values,
                                       const luisa::unordered_map<Value *, size_t> &field_map) noexcept {
    luisa::unordered_set<size_t> seen;
    for (auto *value : values) {
        if (auto it = field_map.find(value); it != field_map.end()) {
            auto field_index = it->second;
            if (seen.emplace(field_index).second) {
                dst.emplace_back(field_index);
            }
        }
    }
    luisa::sort(dst.begin(), dst.end());
}

static void populate_value_transition_edges(CoroMaterializeInfo &info,
                                            const CoroCfgDistillResult &cfg,
                                            const luisa::unordered_map<Value *, size_t> &field_map) noexcept {
    info.edges.clear();
    for (auto &transition : cfg.transition_edges) {
        if (transition.to_scope >= cfg.scopes.size()) { continue; }
        CoroMaterializeInfo::TransitionEdge edge;
        edge.from_scope = transition.from_scope;
        edge.to_scope = transition.to_scope;
        append_value_field_indices(edge.store_fields, transition.store_values, field_map);
        append_value_field_indices(edge.load_fields, cfg.scopes[transition.to_scope].live_in_values, field_map);
        info.edges.emplace_back(std::move(edge));
    }
}

[[nodiscard]] static luisa::vector<luisa::unordered_set<luisa::string>> make_scope_live_sets(
    const CoroCfgDistillResult *cfg, bool live_in) noexcept {
    luisa::vector<luisa::unordered_set<luisa::string>> sets;
    if (cfg == nullptr) { return sets; }
    sets.reserve(cfg->scopes.size());
    for (auto &scope : cfg->scopes) {
        auto &set = sets.emplace_back();
        auto &names = live_in ? scope.live_in_variables : scope.live_out_variables;
        for (auto &name : names) { set.emplace(name); }
    }
    return sets;
}

}// namespace detail

CoroMaterializeInfo coro_materialize_pass_run_on_module(Module *m) noexcept {
    CoroMaterializeInfo info;

    auto regs = detail::collect_registers(m);
    info.register_count = regs.size();
    for (const auto &reg : regs) {
        info.name_to_type.emplace(reg.name, reg.type);
    }

    luisa::vector<CallableFunction *> callables;
    for (auto *f : m->function_list()) {
        if (f->isa<CallableFunction>() && f->definition() != nullptr) {
            if (detail::find_frame_operand(static_cast<CallableFunction *>(f)) != nullptr) {
                callables.push_back(static_cast<CallableFunction *>(f));
            }
        }
    }

    auto field_map = detail::build_field_map(regs);
    info.name_to_field = field_map;
    info.frame_field_count = detail::FRAME_RESERVED_FIELD_COUNT + regs.size();

    for (auto *func : callables) {
        detail::process_callable(m, func, detail::find_frame_operand(func), regs, field_map, nullptr, nullptr, true, info);
        info.callable_count++;
    }

    return info;
}

CoroMaterializeInfo coro_materialize_pass_run_on_module_with_cfg(
    Module *m, const CoroCfgDistillResult &cfg) noexcept {
    return coro_materialize_pass_run_on_module(m);
}

CoroMaterializeInfo coro_materialize_pass_run_on_module_with_cfg(
    Module *m, const CoroCfgDistillResult &cfg, const CoroSplitInfo &split) noexcept {
    CoroMaterializeInfo info;

    luisa::vector<const CoroSplitInfo::Subroutine *> subroutines(cfg.scopes.size(), nullptr);
    for (auto &subroutine : split.subroutines) {
        if (subroutine.scope_index < subroutines.size()) {
            subroutines[subroutine.scope_index] = &subroutine;
        }
    }

    luisa::unordered_map<Value *, size_t> value_field_map;
    info.register_count = cfg.frame_values.size();
    info.frame_field_count = detail::FRAME_RESERVED_FIELD_COUNT + cfg.frame_values.size();
    for (size_t i = 0u; i < cfg.frame_values.size(); ++i) {
        auto &value = cfg.frame_values[i];
        auto field_index = i + detail::FRAME_RESERVED_FIELD_COUNT;
        info.name_to_field.emplace(value.name, field_index);
        info.name_to_type.emplace(value.name, value.type);
        value_field_map.emplace(value.value, field_index);
    }
    detail::populate_value_transition_edges(info, cfg, value_field_map);

    for (auto *subroutine : subroutines) {
        if (subroutine == nullptr) { continue; }
        detail::process_callable(m, subroutine->callable, subroutine->frame_argument,
                                 {}, {}, nullptr, nullptr, false, info);
        info.callable_count++;
    }

    return info;
}

}// namespace luisa::compute::xir
