#include <luisa/xir/passes/loop_vectorization.h>
#include <luisa/xir/passes/scalar_evolution.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/cast.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

[[nodiscard]] static unsigned get_vectorization_factor(const Type *elem_type) noexcept {
    auto size = elem_type->size();
    if (size >= 8) return 2;
    if (size >= 4) return 4;
    if (size >= 2) return 8;
    return 16;
}

[[nodiscard]] static const Type *get_vector_type(const Type *scalar, unsigned vf) noexcept {
    return Type::vector(scalar, vf);
}

[[nodiscard]] static int64_t get_constant_int_value(Constant *c) noexcept {
    if (c == nullptr) return 0;
    if (c->type()->is_int32()) return static_cast<int64_t>(c->as<int32_t>());
    if (c->type()->is_uint32()) return static_cast<int64_t>(c->as<uint32_t>());
    if (c->type()->is_int64()) return c->as<int64_t>();
    if (c->type()->is_uint64()) return static_cast<int64_t>(c->as<uint64_t>());
    return 0;
}

[[nodiscard]] static Constant *create_int_constant(Module *module, const Type *type, int64_t value) noexcept {
    if (type->is_int32()) {
        auto v = static_cast<int32_t>(value);
        return module->create_constant(type, &v);
    }
    if (type->is_uint32()) {
        auto v = static_cast<uint32_t>(value);
        return module->create_constant(type, &v);
    }
    if (type->is_int64()) {
        return module->create_constant(type, &value);
    }
    if (type->is_uint64()) {
        auto v = static_cast<uint64_t>(value);
        return module->create_constant(type, &v);
    }
    return nullptr;
}

class LoopCloneResolver final : public InstructionCloneValueResolver {
    luisa::unordered_map<const Value *, Value *> _map;

public:
    void map(const Value *from, Value *to) noexcept { _map.insert_or_assign(from, to); }
    [[nodiscard]] Value *resolve(const Value *value) noexcept override {
        if (value == nullptr) return nullptr;
        auto it = _map.find(value);
        if (it != _map.end()) return it->second;
        return const_cast<Value *>(value);
    }
};

struct LoopVectorizationContext {
    FunctionDefinition *def{nullptr};
    Module *module{nullptr};
    LoopInst *loop{nullptr};
    BasicBlock *parent_block{nullptr};
    BasicBlock *prepare{nullptr};
    BasicBlock *body{nullptr};
    BasicBlock *update{nullptr};
    BasicBlock *merge{nullptr};
    PhiInst *induction_phi{nullptr};
    Instruction *induction_add{nullptr};
    ArithmeticInst *compare_inst{nullptr};
    Constant *start_const{nullptr};
    Constant *step_const{nullptr};
    Constant *bound_const{nullptr};
    const Type *elem_type{nullptr};
    unsigned vf{1};
    int64_t const_trip_count{-1};
};

[[nodiscard]] static bool is_simple_vectorizable_arithmetic_op(ArithmeticOp op) noexcept {
    switch (op) {
        case ArithmeticOp::BINARY_ADD:
        case ArithmeticOp::BINARY_SUB:
        case ArithmeticOp::BINARY_MUL:
        case ArithmeticOp::BINARY_DIV:
        case ArithmeticOp::BINARY_MOD:
        case ArithmeticOp::BINARY_BIT_AND:
        case ArithmeticOp::BINARY_BIT_OR:
        case ArithmeticOp::BINARY_BIT_XOR:
            return true;
        default:
            return false;
    }
}

[[nodiscard]] static bool is_loop_structure_valid(LoopVectorizationContext &ctx) noexcept {
    ctx.prepare = ctx.loop->prepare_block();
    ctx.body = ctx.loop->body_block();
    ctx.update = ctx.loop->update_block();
    ctx.merge = ctx.loop->merge_block();
    if (ctx.prepare == nullptr || ctx.body == nullptr || ctx.update == nullptr || ctx.merge == nullptr) {
        return false;
    }
    ctx.parent_block = ctx.loop->parent_block();
    if (ctx.parent_block == nullptr) { return false; }

    auto *prep_term = ctx.prepare->terminator();
    if (prep_term == nullptr || !prep_term->isa<ConditionalBranchInst>()) { return false; }
    auto *cond_br = static_cast<ConditionalBranchInst *>(prep_term);
    if (cond_br->true_block() != ctx.body || cond_br->false_block() != ctx.merge) { return false; }

    auto *body_term = ctx.body->terminator();
    if (body_term == nullptr || !body_term->isa<BranchInst>() ||
        static_cast<BranchInst *>(body_term)->target_block() != ctx.update) {
        return false;
    }

    auto *update_term = ctx.update->terminator();
    if (update_term == nullptr || !update_term->isa<BranchInst>() ||
        static_cast<BranchInst *>(update_term)->target_block() != ctx.prepare) {
        return false;
    }

    return true;
}

[[nodiscard]] static bool analyze_trip_count(LoopVectorizationContext &ctx) noexcept {
    auto *prep_term = ctx.prepare->terminator();
    auto *cond_br = static_cast<ConditionalBranchInst *>(prep_term);
    auto *cond = cond_br->condition();
    if (cond == nullptr || !cond->isa<ArithmeticInst>()) { return false; }
    auto *cmp = static_cast<ArithmeticInst *>(cond);
    auto op = cmp->op();
    if (op != ArithmeticOp::BINARY_LESS && op != ArithmeticOp::BINARY_LESS_EQUAL) { return false; }
    if (!cmp->operand(1)->isa<Constant>()) { return false; }
    ctx.bound_const = static_cast<Constant *>(cmp->operand(1));
    auto *induction = cmp->operand(0);
    if (induction == nullptr || !induction->isa<PhiInst>()) { return false; }
    ctx.induction_phi = static_cast<PhiInst *>(induction);
    if (ctx.induction_phi->parent_block() != ctx.prepare) { return false; }
    ctx.compare_inst = cmp;

    auto *scev = scev_get_for_value(ctx.induction_phi);
    if (scev == nullptr || scev->kind() != SCEV::Kind::ADD_REC) { return false; }
    auto *add_rec = static_cast<const SCEVAddRec *>(scev);
    if (add_rec->start()->kind() != SCEV::Kind::CONSTANT ||
        add_rec->stride()->kind() != SCEV::Kind::CONSTANT) {
        return false;
    }
    ctx.start_const = static_cast<const SCEVConstant *>(add_rec->start())->constant();
    ctx.step_const = static_cast<const SCEVConstant *>(add_rec->stride())->constant();

    Value *recur_val = nullptr;
    for (size_t i = 0; i < ctx.induction_phi->incoming_count(); ++i) {
        auto inc = ctx.induction_phi->incoming(i);
        if (inc.block == ctx.update) {
            recur_val = inc.value;
        }
    }
    if (recur_val == nullptr || !recur_val->isa<Instruction>()) { return false; }
    ctx.induction_add = static_cast<Instruction *>(recur_val);

    int64_t start = get_constant_int_value(ctx.start_const);
    int64_t step = get_constant_int_value(ctx.step_const);
    int64_t bound = get_constant_int_value(ctx.bound_const);
    if (step <= 0) { return false; }

    int64_t trips = (bound - start + (op == ArithmeticOp::BINARY_LESS_EQUAL ? 1 : 0) + step - 1) / step;
    if (trips < 0) { return false; }
    ctx.const_trip_count = trips;
    return true;
}

[[nodiscard]] static bool check_legality(LoopVectorizationContext &ctx) noexcept {
    if (!is_loop_structure_valid(ctx)) { return false; }
    scev_pass_run_on_function(ctx.def);
    if (!analyze_trip_count(ctx)) { return false; }

    // Require unit stride for the simple vectorization scheme.
    if (get_constant_int_value(ctx.step_const) != 1) { return false; }

    // Determine vector element type from memory accesses in the body.
    const Type *elem_type = nullptr;
    for (auto *inst : ctx.body->instructions()) {
        if (inst->isa<LoadInst>()) {
            auto ty = inst->type();
            if (ty != nullptr && ty->is_scalar() && ty->is_arithmetic()) { elem_type = ty; }
        } else if (inst->isa<StoreInst>()) {
            auto *store = static_cast<StoreInst *>(inst);
            auto ty = store->value()->type();
            if (ty != nullptr && ty->is_scalar() && ty->is_arithmetic()) { elem_type = ty; }
        }
        if (elem_type != nullptr) { break; }
    }
    if (elem_type == nullptr) { return false; }
    ctx.elem_type = elem_type;
    ctx.vf = get_vectorization_factor(elem_type);
    if (ctx.vf <= 1) { return false; }

    // Check all instructions in body and update are vectorizable.
    auto check_block = [&](BasicBlock *bb) noexcept -> bool {
        for (auto *inst : bb->instructions()) {
            if (inst->is_terminator()) { continue; }
            switch (inst->derived_instruction_tag()) {
                case DerivedInstructionTag::ARITHMETIC: {
                    auto *arith = static_cast<ArithmeticInst *>(inst);
                    if (!is_simple_vectorizable_arithmetic_op(arith->op())) { return false; }
                    break;
                }
                case DerivedInstructionTag::CAST:
                case DerivedInstructionTag::LOAD:
                case DerivedInstructionTag::STORE:
                case DerivedInstructionTag::GEP:
                case DerivedInstructionTag::PHI:
                    break;
                default:
                    return false;
            }
        }
        return true;
    };
    if (!check_block(ctx.body)) { return false; }
    if (!check_block(ctx.update)) { return false; }

    // Check memory accesses use GEPs with the induction variable as the only varying index.
    for (auto *inst : ctx.body->instructions()) {
        if (inst->isa<LoadInst>()) {
            auto *load = static_cast<LoadInst *>(inst);
            if (!load->variable()->isa<GEPInst>()) { return false; }
            auto *gep = static_cast<GEPInst *>(load->variable());
            bool has_iv = false;
            for (size_t i = 0; i < gep->operand_count(); ++i) {
                if (gep->operand(i) == ctx.induction_phi) { has_iv = true; }
            }
            if (!has_iv) { return false; }
        } else if (inst->isa<StoreInst>()) {
            auto *store = static_cast<StoreInst *>(inst);
            if (!store->variable()->isa<GEPInst>()) { return false; }
            auto *gep = static_cast<GEPInst *>(store->variable());
            bool has_iv = false;
            for (size_t i = 0; i < gep->operand_count(); ++i) {
                if (gep->operand(i) == ctx.induction_phi) { has_iv = true; }
            }
            if (!has_iv) { return false; }
        }
    }

    // The induction variable may only be used by the compare, the induction add, and GEP indices.
    for (auto &&use : ctx.induction_phi->use_list()) {
        auto *user = use->user();
        if (user == nullptr || !user->isa<Instruction>()) { return false; }
        auto *user_inst = static_cast<Instruction *>(user);
        if (user_inst == ctx.compare_inst) { continue; }
        if (user_inst == ctx.induction_add) { continue; }
        if (user_inst->isa<GEPInst>()) { continue; }
        return false;
    }

    // Skip loops that would not execute at least one full vector iteration.
    if (ctx.const_trip_count > 0 && static_cast<uint64_t>(ctx.const_trip_count) < ctx.vf) { return false; }

    return true;
}

[[nodiscard]] static Constant *create_splat_constant(Module *module, const Type *vec_type, Constant *scalar) noexcept {
    auto elem_type = vec_type->element();
    auto dim = vec_type->dimension();
    luisa::vector<std::byte> data(vec_type->size());
    auto elem_data = static_cast<const std::byte *>(scalar->data());
    for (auto i = 0u; i < dim; ++i) {
        auto offset = i * elem_type->size();
        std::memcpy(data.data() + offset, elem_data, elem_type->size());
    }
    return module->create_constant(vec_type, data.data());
}

[[nodiscard]] static Value *widen_operand(Value *v, const Type *elem_type, const Type *vec_type,
                                          unsigned vf, PhiInst *induction_phi,
                                          luisa::unordered_map<const Value *, Value *> &widen_map,
                                          XIRBuilder &builder) noexcept {
    if (v == nullptr) { return nullptr; }
    if (v == induction_phi) { return v; }
    if (auto it = widen_map.find(v); it != widen_map.end()) { return it->second; }
    if (v->type() != elem_type) { return v; }
    if (v->isa<Constant>()) {
        auto *c = static_cast<Constant *>(v);
        auto *splat = create_splat_constant(builder.insertion_point()->parent_module(), vec_type, c);
        widen_map[v] = splat;
        return splat;
    }
    // Splat a scalar value into a vector via AGGREGATE.
    luisa::vector<Value *> elems;
    elems.reserve(vf);
    for (auto i = 0u; i < vf; ++i) { elems.emplace_back(v); }
    auto *splat = builder.call(vec_type, ArithmeticOp::AGGREGATE, elems);
    widen_map[v] = splat;
    return splat;
}

static void vectorize_loop(LoopVectorizationContext &ctx, LoopVectorizationInfo &info) noexcept {
    auto *module = ctx.module;
    auto *func = ctx.def;
    auto *elem_type = ctx.elem_type;
    auto *vec_type = get_vector_type(elem_type, ctx.vf);
    auto vf = ctx.vf;

    int64_t start = get_constant_int_value(ctx.start_const);
    int64_t step = get_constant_int_value(ctx.step_const);
    int64_t full_vector_iters = ctx.const_trip_count / static_cast<int64_t>(vf);
    int64_t epilogue_start = start + full_vector_iters * step;

    auto *epilogue_start_const = create_int_constant(module, ctx.induction_phi->type(), epilogue_start);
    if (epilogue_start_const == nullptr) { return; }

    // Create vector loop blocks.
    auto *prepare_v = func->create_basic_block();
    auto *body_v = func->create_basic_block();
    auto *update_v = func->create_basic_block();

    LoopCloneResolver resolver;
    resolver.map(ctx.prepare, prepare_v);
    resolver.map(ctx.body, body_v);
    resolver.map(ctx.update, update_v);

    XIRBuilder builder;

    // Clone non-terminator instructions into the vector loop blocks.
    auto clone_block = [&](BasicBlock *from, BasicBlock *to) noexcept {
        builder.set_insertion_point(to);
        for (auto *inst : from->instructions()) {
            if (inst->is_terminator()) { continue; }
            auto *cloned = inst->clone_with_metadata(builder, resolver);
            resolver.map(inst, cloned);
        }
    };

    clone_block(ctx.prepare, prepare_v);
    clone_block(ctx.body, body_v);
    clone_block(ctx.update, update_v);

    auto *induction_phi_v = static_cast<PhiInst *>(resolver.resolve(ctx.induction_phi));
    auto *induction_add_v = static_cast<Instruction *>(resolver.resolve(ctx.induction_add));
    auto *compare_v = static_cast<ArithmeticInst *>(resolver.resolve(ctx.compare_inst));

    luisa::unordered_map<const Value *, Value *> widen_map;

    auto widen_instruction = [&](Instruction *inst) noexcept -> bool {
        if (inst == induction_phi_v || inst == induction_add_v || inst == compare_v) { return false; }
        if (inst->isa<PhiInst>()) {
            auto *phi = static_cast<PhiInst *>(inst);
            if (phi->type() != elem_type) { return false; }
            builder.set_insertion_point(phi);
            auto *phi_v = builder.phi(vec_type);
            for (size_t i = 0; i < phi->incoming_count(); ++i) {
                auto inc = phi->incoming(i);
                auto *val = widen_operand(inc.value, elem_type, vec_type, vf, induction_phi_v, widen_map, builder);
                auto *block = static_cast<BasicBlock *>(resolver.resolve(inc.block));
                phi_v->add_incoming(val, block);
            }
            phi->replace_all_uses_with(phi_v);
            widen_map[phi] = phi_v;
            return true;
        }
        if (inst->isa<LoadInst>()) {
            auto *load = static_cast<LoadInst *>(inst);
            if (load->type() != elem_type) { return false; }
            builder.set_insertion_point(load);
            Value *load_ptr = load->variable();
            if (load->variable()->isa<GEPInst>()) {
                auto *gep = static_cast<GEPInst *>(load->variable());
                luisa::vector<Value *> indices;
                indices.reserve(gep->index_count());
                for (size_t i = 0; i < gep->index_count(); ++i) { indices.emplace_back(gep->index(i)); }
                load_ptr = builder.gep(vec_type, gep->base(), indices);
            }
            auto *load_v = builder.load(vec_type, load_ptr);
            load->replace_all_uses_with(load_v);
            widen_map[load] = load_v;
            return true;
        }
        if (inst->isa<StoreInst>()) {
            auto *store = static_cast<StoreInst *>(inst);
            auto *val = widen_operand(store->value(), elem_type, vec_type, vf, induction_phi_v, widen_map, builder);
            if (val == store->value()) { return false; }
            builder.set_insertion_point(store);
            Value *store_ptr = store->variable();
            if (store->variable()->isa<GEPInst>()) {
                auto *gep = static_cast<GEPInst *>(store->variable());
                luisa::vector<Value *> indices;
                indices.reserve(gep->index_count());
                for (size_t i = 0; i < gep->index_count(); ++i) { indices.emplace_back(gep->index(i)); }
                store_ptr = builder.gep(vec_type, gep->base(), indices);
            }
            auto *store_v = builder.store(store_ptr, val);
            widen_map[store] = store_v;
            return true;
        }
        if (inst->isa<ArithmeticInst>()) {
            auto *arith = static_cast<ArithmeticInst *>(inst);
            if (arith->type() != elem_type) { return false; }
            builder.set_insertion_point(arith);
            luisa::vector<Value *> ops;
            ops.reserve(arith->operand_count());
            for (size_t i = 0; i < arith->operand_count(); ++i) {
                ops.emplace_back(widen_operand(arith->operand(i), elem_type, vec_type, vf, induction_phi_v, widen_map, builder));
            }
            auto *arith_v = builder.call(vec_type, arith->op(), ops);
            arith->replace_all_uses_with(arith_v);
            widen_map[arith] = arith_v;
            return true;
        }
        if (inst->isa<CastInst>()) {
            auto *cast = static_cast<CastInst *>(inst);
            if (cast->type() != elem_type) { return false; }
            builder.set_insertion_point(cast);
            auto *src = widen_operand(cast->operand(0), elem_type, vec_type, vf, induction_phi_v, widen_map, builder);
            auto *cast_v = builder.cast_(vec_type, cast->op(), src);
            cast->replace_all_uses_with(cast_v);
            widen_map[cast] = cast_v;
            return true;
        }
        return false;
    };

    auto widen_block = [&](BasicBlock *bb) noexcept {
        luisa::vector<Instruction *> to_remove;
        for (auto *inst : bb->instructions()) {
            if (inst->is_terminator()) { continue; }
            if (widen_instruction(inst)) { to_remove.emplace_back(inst); }
        }
        for (auto *inst : to_remove) { inst->remove_self(); }
    };

    // Process update first so loop-carried values are widened before prepare PHIs use them.
    widen_block(update_v);
    widen_block(body_v);
    widen_block(prepare_v);

    // Change induction step from +step to +VF.
    if (induction_add_v != nullptr && induction_add_v->isa<ArithmeticInst>()) {
        auto *arith = static_cast<ArithmeticInst *>(induction_add_v);
        if (arith->op() == ArithmeticOp::BINARY_ADD) {
            auto *vf_const = create_int_constant(module, ctx.induction_phi->type(), static_cast<int64_t>(vf));
            if (vf_const != nullptr) {
                for (size_t i = 0; i < arith->operand_count(); ++i) {
                    if (arith->operand(i) != induction_phi_v) {
                        arith->set_operand(i, vf_const);
                        break;
                    }
                }
            }
        }
    }

    // Build a new vector loop compare: phi_v < epilogue_start.
    ArithmeticInst *cmp_v = nullptr;
    if (compare_v != nullptr) {
        builder.set_insertion_point(compare_v);
        cmp_v = builder.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {induction_phi_v, epilogue_start_const});
        compare_v->replace_all_uses_with(cmp_v);
        compare_v->remove_self();
    }

    // Rebuild terminators for the vector loop blocks.
    if (prepare_v->is_terminated()) {
        if (auto *t = prepare_v->terminator(); t != nullptr) { t->remove_self(); }
    }
    builder.set_insertion_point(prepare_v);
    builder.cond_br(cmp_v, body_v, ctx.prepare);

    if (body_v->is_terminated()) {
        if (auto *t = body_v->terminator(); t != nullptr) { t->remove_self(); }
    }
    builder.set_insertion_point(body_v);
    builder.br(update_v);

    if (update_v->is_terminated()) {
        if (auto *t = update_v->terminator(); t != nullptr) { t->remove_self(); }
    }
    builder.set_insertion_point(update_v);
    builder.br(prepare_v);

    // Replace original loop with the vector loop in the parent block.
    ctx.loop->remove_self();
    builder.set_insertion_point(ctx.parent_block);
    auto *loop_v = builder.loop();
    loop_v->set_prepare_block(prepare_v);
    loop_v->set_body_block(body_v);
    loop_v->set_update_block(update_v);
    loop_v->set_merge_block(ctx.prepare);

    // Re-wire the epilogue (original prepare) so its initial PHI value starts at epilogue_start
    // and its initial predecessor is the vector loop's prepare block.
    for (auto *inst : ctx.prepare->instructions()) {
        if (inst->is_terminator()) { break; }
        if (!inst->isa<PhiInst>()) { continue; }
        auto *phi = static_cast<PhiInst *>(inst);
        for (size_t i = 0; i < phi->incoming_count(); ++i) {
            auto inc = phi->incoming(i);
            if (inc.block == ctx.parent_block) {
                Value *new_value = inc.value;
                if (phi == ctx.induction_phi) { new_value = epilogue_start_const; }
                phi->set_incoming(i, new_value, prepare_v);
                break;
            }
        }
    }

    // Count created vector instructions for statistics.
    size_t vec_inst_count = 0;
    for (auto *bb : {prepare_v, body_v, update_v}) {
        for (auto *inst : bb->instructions()) {
            if (inst->is_terminator()) { continue; }
            if (inst->type() == vec_type) { vec_inst_count++; }
        }
    }

    info.vectorized_loop_count++;
    info.created_vector_inst_count += vec_inst_count;
}

static void run_on_function(Function *function, LoopVectorizationInfo &info) noexcept {
    auto *def = function->definition();
    if (def == nullptr) { return; }

    luisa::vector<LoopInst *> loops;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<LoopInst>()) { loops.emplace_back(static_cast<LoopInst *>(inst)); }
    });

    for (auto *loop : loops) {
        LoopVectorizationContext ctx;
        ctx.loop = loop;
        ctx.def = def;
        ctx.module = def->parent_module();
        if (!check_legality(ctx)) { continue; }
        vectorize_loop(ctx, info);
    }
}

}// namespace detail

LoopVectorizationInfo loop_vectorization_pass_run_on_function(Function *function) noexcept {
    LoopVectorizationInfo info;
    detail::run_on_function(function, info);
    return info;
}

LoopVectorizationInfo loop_vectorization_pass_run_on_module(Module *module, PassReport *report) noexcept {
    LoopVectorizationInfo info;
    for (auto *f : module->function_list()) {
        detail::run_on_function(f, info);
    }
    if (report != nullptr) {
        report->set("vectorized_loop_count", info.vectorized_loop_count);
        report->set("created_vector_inst_count", info.created_vector_inst_count);
    }
    return info;
}

}// namespace luisa::compute::xir
