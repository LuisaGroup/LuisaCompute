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
#include <luisa/xir/passes/coro_materialize.h>

namespace luisa::compute::xir {

namespace detail {

static constexpr uint32_t TERMINAL_TOKEN = 0xFFFFFFFFu;

struct RegisterInfo {
    luisa::string name;
    const Type *type;
};

[[nodiscard]] static Value *get_frame_arg(CallableFunction *func) noexcept {
    for (auto *arg : func->arguments()) {
        if (arg->is_reference()) { return arg; }
    }
    return nullptr;
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
    return true;
}

[[nodiscard]] static luisa::vector<RegisterInfo> collect_registers(Module *mod) noexcept {
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
            if (seen.insert(name).second) {
                regs.push_back({std::move(name), alloca->type()});
            }
        });
    }
    return regs;
}

[[nodiscard]] static const Type *build_frame_type(const luisa::vector<RegisterInfo> &regs) noexcept {
    luisa::vector<const Type *> fields;
    fields.push_back(Type::of<uint32_t>());// [0] token
    fields.push_back(Type::of<uint32_t>());// [1] skip
    for (auto &reg : regs) { fields.push_back(reg.type); }
    return Type::structure(fields);
}

[[nodiscard]] static luisa::unordered_map<luisa::string, size_t> build_field_map(
    const luisa::vector<RegisterInfo> &regs) noexcept {
    luisa::unordered_map<luisa::string, size_t> map;
    constexpr size_t user_offset = 2u;
    for (size_t i = 0u; i < regs.size(); ++i) {
        map.emplace(regs[i].name, user_offset + i);
    }
    return map;
}

static void store_user_vars_to_frame(XIRBuilder &b, Module *mod, Value *frame_arg,
                                     const luisa::vector<RegisterInfo> &regs,
                                     const luisa::unordered_map<luisa::string, size_t> &field_map,
                                     const luisa::unordered_map<luisa::string, Value *> &local_map,
                                     size_t &count) noexcept {
    for (auto &reg : regs) {
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
                                      size_t &count) noexcept {
    for (auto &reg : regs) {
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

static void process_callable(Module *mod, CallableFunction *func,
                             const luisa::vector<RegisterInfo> &regs,
                             const luisa::unordered_map<luisa::string, size_t> &field_map,
                             CoroMaterializeInfo &info) noexcept {

    auto *frame_arg = get_frame_arg(func);
    if (frame_arg == nullptr) { return; }

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
        store_user_vars_to_frame(b, mod, frame_arg, regs, field_map, local_map, info.store_inserted_count);

        auto *field_zero = mod->create_constant_zero(Type::of<uint32_t>());
        auto *gep0 = b.gep(Type::of<uint32_t>(), frame_arg, {field_zero});
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
        auto *field_zero = mod->create_constant_zero(Type::of<uint32_t>());
        auto *gep0 = b.gep(Type::of<uint32_t>(), frame_arg, {field_zero});
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
            store_user_vars_to_frame(b, mod, frame_arg, regs, field_map, local_map, info.store_inserted_count);
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
        load_user_vars_from_frame(b, mod, frame_arg, regs, field_map, local_map, info.load_inserted_count);
        r->remove_self();
        info.resume_lowered_count++;
    }

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
            if (detail::get_frame_arg(static_cast<CallableFunction *>(f)) != nullptr) {
                callables.push_back(static_cast<CallableFunction *>(f));
            }
        }
    }

    auto field_map = detail::build_field_map(regs);
    info.name_to_field = field_map;
    info.frame_field_count = 2u + regs.size();

    for (auto *func : callables) {
        detail::process_callable(m, func, regs, field_map, info);
        info.callable_count++;
    }

    return info;
}

}// namespace luisa::compute::xir
