#include "coro_call_alias.h"
#include <algorithm>
#include <cstring>

#include <luisa/xir/builder.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/constant.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/logging.h>

namespace luisa::compute::xir::detail {

class CoroCallAliasLowering::Impl {
    struct Index {
        Value *value{nullptr};
        AllocaInst *storage{nullptr};
    };
    struct Step {
        const Type *type{nullptr};
        luisa::vector<Index> indices;
    };
    struct Path {
        Value *base{nullptr};
        luisa::vector<Step> steps;
    };
    struct Family {
        AllocaInst *selector{nullptr};
        luisa::vector<Path> paths;
    };
    struct Choices {
        AllocaInst *selector{nullptr};
        luisa::vector<Path> paths;
    };
    struct Resolver final : InstructionCloneValueResolver {
        luisa::unordered_map<const Value *, Value *> values;
        Value *resolve(const Value *v) noexcept override {
            if (auto i = values.find(v); i != values.end()) { return i->second; }
            return const_cast<Value *>(v);
        }
    };

    Function *_root;
    luisa::unordered_map<Value *, Family> _families;
    luisa::unordered_set<Value *> _projection_roots;

    bool _has_alias(Value *v) const {
        if (v == nullptr) { return false; }
        if (_families.contains(v)) { return true; }
        return v->isa<GEPInst>() && _has_alias(static_cast<GEPInst *>(v)->base());
    }

    Choices _choices(Value *v) const {
        if (auto i = _families.find(v); i != _families.end()) {
            LUISA_ASSERT(!i->second.paths.empty(), "Coroutine alias has no incoming binding.");
            return {i->second.selector, i->second.paths};
        }
        if (v->isa<GEPInst>()) {
            auto *gep = static_cast<GEPInst *>(v);
            auto choices = _choices(gep->base());
            Step step;
            step.type = gep->type();
            for (size_t i = 0u; i < gep->index_count(); ++i) { step.indices.push_back({gep->index(i), nullptr}); }
            for (auto &path : choices.paths) { path.steps.emplace_back(step); }
            return choices;
        }
        return {nullptr, {{v, {}}}};
    }

    Value *_index(XIRBuilder &b, const Index &index) {
        return index.storage == nullptr ? index.value : b.load(index.storage->type(), index.storage);
    }

    static bool _same_path_shape(const Path &a, const Path &b) noexcept {
        if (a.base != b.base || a.steps.size() != b.steps.size()) { return false; }
        for (size_t i = 0u; i < a.steps.size(); ++i) {
            auto &as = a.steps[i];
            auto &bs = b.steps[i];
            if (as.type != bs.type || as.indices.size() != bs.indices.size()) { return false; }
            for (size_t j = 0u; j < as.indices.size(); ++j) {
                auto &ai = as.indices[j];
                auto &bi = bs.indices[j];
                auto *av = ai.storage == nullptr ? ai.value : ai.storage;
                auto *bv = bi.storage == nullptr ? bi.value : bi.storage;
                if (av->type() != bv->type() || av->isa<Constant>() != bv->isa<Constant>()) { return false; }
                if (av->isa<Constant>() &&
                    std::memcmp(static_cast<Constant *>(av)->data(), static_cast<Constant *>(bv)->data(), av->type()->size()) != 0) {
                    return false;
                }
            }
        }
        return true;
    }

    Value *_materialize(XIRBuilder &b, const Path &path) {
        auto *value = path.base;
        for (auto &step : path.steps) {
            luisa::vector<Value *> indices;
            for (auto &index : step.indices) { indices.emplace_back(_index(b, index)); }
            value = b.gep(step.type, value, indices);
        }
        return value;
    }

    template<typename Emit>
    void _dispatch(XIRBuilder &b, const Choices &choices, Emit &&emit) {
        if (choices.selector == nullptr) {
            emit(choices.paths.front(), 0u);
            return;
        }
        auto *branch = b.indexed_branch(b.load(Type::of<uint32_t>(), choices.selector));
        auto *merge = _root->create_basic_block();
        auto *invalid = _root->create_basic_block();
        branch->set_default_block(invalid);
        b.set_insertion_point(invalid);
        b.unreachable_("Invalid coroutine alias selector.");
        for (size_t i = 0u; i < choices.paths.size(); ++i) {
            auto *block = _root->create_basic_block();
            branch->add_case(i, block);
            b.set_insertion_point(block);
            emit(choices.paths[i], i);
            b.br(merge);
        }
        b.set_insertion_point(merge);
    }

    void _lower_suspend(CoroSuspendInst *suspend, XIRBuilder &prologue) {
        XIRBuilder b;
        b.set_insertion_point(suspend->prev());
        auto *module = _root->parent_module();
        luisa::vector<Value *> values;
        for (size_t i = 0; i < suspend->extension_binding_value_count(); ++i) {
            values.emplace_back(suspend->extension_binding_value(i));
        }
        luisa::vector<CoroSuspendExtensionPtr> extensions;
        for (auto &&extension : suspend->extensions()) {
            luisa::vector<CoroSuspendBinding> bindings;
            luisa::vector<CoroSuspendBindingProjection> projections;
            auto append = [&](Value *value, CoroSuspendBindingAccess access,
                              CoroSuspendBindingLifetime lifetime, uint32_t index) {
                values[index] = value;
                // Names are private to this normalized owner, never exposed to
                // a scheduler or inserted into the plugin's logical schema.
                auto name = luisa::format("__coro_projection_{}", index);
                while (std::any_of(extension->bindings().begin(), extension->bindings().end(), [&](auto &b) { return b.name == name; })) { name += "_"; }
                bindings.push_back({std::move(name), access, lifetime, index});
            };
            for (auto &&binding : extension->bindings()) {
                auto *value = values[binding.index];
                if (!_has_alias(value)) {
                    bindings.emplace_back(binding);
                    continue;
                }
                auto &projection = projections.emplace_back();
                projection.binding = binding;
                auto choices = _choices(value);
                auto *selector = choices.selector == nullptr ? nullptr : b.load(Type::of<uint32_t>(), choices.selector);
                for (uint32_t tag = 0; tag < choices.paths.size(); ++tag) {
                    auto &path = choices.paths[tag];
                    // The normalized stage preserves every inactive carrier.
                    // Give never-entered activation storage a defined value so
                    // transporting it cannot read uninitialized packed bits.
                    if (path.base->isa<AllocaInst>() && static_cast<AllocaInst *>(path.base)->is_local() &&
                        _projection_roots.emplace(path.base).second) {
                        prologue.store(path.base, module->create_constant_zero(path.base->type()));
                    }
                    Value *condition = module->create_constant_one(Type::of<bool>());
                    if (selector != nullptr) {
                        condition = b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
                                           {selector, module->create_constant(Type::of<uint32_t>(), &tag)});
                    }
                    luisa::vector<Index> indices;
                    for (auto &step : path.steps) { indices.insert(indices.end(), step.indices.begin(), step.indices.end()); }
                    // Enumerate the finite local-object domain into guarded
                    // static projections. Only guards cross the suspend; no
                    // host/device pointer or copied reference value does.
                    auto emit = [&](auto &&self, size_t depth, Value *pointer, Value *guard) -> void {
                        if (depth == indices.size()) {
                            auto first = projection.alternatives.empty();
                            auto vi = first ? binding.index : static_cast<uint32_t>(values.size());
                            if (!first) { values.emplace_back(nullptr); }
                            append(pointer, CoroSuspendBindingAccess::read_write, binding.lifetime, vi);
                            auto ci = static_cast<uint32_t>(values.size());
                            values.emplace_back(nullptr);
                            append(guard, CoroSuspendBindingAccess::read, binding.lifetime, ci);
                            projection.alternatives.push_back({vi, ci});
                            return;
                        }
                        auto *index = _index(b, indices[depth]);
                        uint64_t constant = 0;
                        bool fixed = try_decode_constant_nonnegative_integer(index, constant);
                        auto *type = pointer->type();
                        LUISA_ASSERT(fixed || !type->is_structure(), "Dynamic structure member in coroutine alias.");
                        auto count = fixed ? constant + 1u : type->dimension();
                        for (uint64_t i = fixed ? constant : 0u; i < count; ++i) {
                            auto *element = type->is_structure() ? type->members()[i] :
                                            type->is_matrix()    ? Type::vector(type->element(), type->dimension()) :
                                                                   type->element();
                            uint32_t offset = static_cast<uint32_t>(i);
                            auto *literal = module->create_constant(Type::of<uint32_t>(), &offset);
                            auto *next_guard = guard;
                            if (!fixed) {
                                auto *equal = b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
                                                     {index, b.static_cast_if_necessary(index->type(), literal)});
                                next_guard = b.call(Type::of<bool>(), ArithmeticOp::BINARY_BIT_AND, {guard, equal});
                            }
                            self(self, depth + 1u, b.gep(element, pointer, {literal}), next_guard);
                        }
                    };
                    emit(emit, 0u, path.base, condition);
                }
            }
            extensions.emplace_back(projections.empty() ? extension->clone() :
                                                          make_coro_suspend_projected_extension(extension->clone(), std::move(bindings), std::move(projections)));
        }
        luisa::vector<Value *> exports;
        for (size_t i = 0; i < suspend->frame_export_count(); ++i) { exports.emplace_back(suspend->frame_export_value(i)); }
        b.coro_suspend(suspend->token(), suspend->name(), suspend->frame(), suspend->frame_export_names(), exports,
                       std::move(extensions), values);
        suspend->remove_self();
    }

public:
    explicit Impl(Function *root) noexcept : _root{root} {}

    void add_parameter(Argument *argument, XIRBuilder &prologue) {
        auto *selector = prologue.alloca_local(Type::of<uint32_t>());
        prologue.store(selector, _root->parent_module()->create_constant_zero(Type::of<uint32_t>()));
        _families.emplace(argument, Family{selector, {}});
    }

    void bind_call(CallInst *call, XIRBuilder &b, XIRBuilder &prologue) {
        size_t arg_index = 0u;
        for (auto *arg : call->callee()->arguments()) {
            auto *actual = call->argument(arg_index++);
            auto iter = _families.find(arg);
            if (iter == _families.end()) { continue; }
            auto &family = iter->second;
            auto choices = _choices(actual);
            _dispatch(b, choices, [&](const Path &source, size_t) {
                auto candidate = std::find_if(family.paths.begin(), family.paths.end(),
                                              [&](const Path &path) { return _same_path_shape(path, source); });
                auto tag = static_cast<uint32_t>(candidate - family.paths.begin());
                if (candidate == family.paths.end()) {
                    Path captured{source.base, {}};
                    for (auto &step : source.steps) {
                        Step copy{step.type, {}};
                        for (auto &index : step.indices) {
                            if (index.storage == nullptr && index.value->isa<Constant>()) {
                                copy.indices.emplace_back(index);
                            } else {
                                auto *type = index.storage == nullptr ? index.value->type() : index.storage->type();
                                auto *slot = prologue.alloca_local(type);
                                prologue.store(slot, _root->parent_module()->create_constant_zero(type));
                                copy.indices.push_back({nullptr, slot});
                            }
                        }
                        captured.steps.emplace_back(std::move(copy));
                    }
                    family.paths.emplace_back(std::move(captured));
                }
                // A nonrecursive function has one active invocation. Reuse its
                // capture slots for equivalent paths, refreshing their indices
                // on every call. Copying paths per call would otherwise grow
                // exponentially through a chain with several sites per level.
                auto &captured = family.paths[tag];
                for (size_t i = 0u; i < source.steps.size(); ++i) {
                    for (size_t j = 0u; j < source.steps[i].indices.size(); ++j) {
                        if (auto *slot = captured.steps[i].indices[j].storage) {
                            b.store(slot, _index(b, source.steps[i].indices[j]));
                        }
                    }
                }
                b.store(family.selector, _root->parent_module()->create_constant(Type::of<uint32_t>(), &tag));
            });
        }
    }

    void lower_uses(XIRBuilder &prologue) {
        luisa::vector<Instruction *> instructions;
        luisa::vector<GEPInst *> symbolic_geps;
        for (auto *bb : _root->definition()->basic_blocks()) {
            for (auto *inst : bb->instructions()) {
                if (inst->isa<GEPInst>() && _has_alias(inst)) {
                    symbolic_geps.emplace_back(static_cast<GEPInst *>(inst));
                    continue;
                }
                for (size_t i = 0u; i < inst->operand_count(); ++i) {
                    if (_has_alias(inst->operand(i))) {
                        instructions.emplace_back(inst);
                        break;
                    }
                }
            }
        }
        for (auto *inst : instructions) {
            if (inst->isa<CoroSuspendInst>()) {
                _lower_suspend(static_cast<CoroSuspendInst *>(inst), prologue);
                continue;
            }
            LUISA_ASSERT(!inst->is_terminator(), "Unsupported coroutine alias terminator.");
            auto *block = inst->parent_block();
            auto *merge = _root->create_basic_block();
            luisa::vector<Instruction *> tail;
            for (auto *next = inst->next(); !next->is_sentinel(); next = next->next()) { tail.emplace_back(next); }
            AllocaInst *result = inst->type() == nullptr ? nullptr : prologue.alloca_local(inst->type());
            XIRBuilder b;
            b.set_insertion_point(inst);
            Resolver resolver;
            auto emit = [&](auto &&self, size_t operand_index) -> void {
                while (operand_index < inst->operand_count() && !_has_alias(inst->operand(operand_index))) { ++operand_index; }
                if (operand_index == inst->operand_count()) {
                    auto *clone = inst->clone_with_metadata(b, resolver);
                    if (result != nullptr) { b.store(result, clone); }
                    return;
                }
                auto *operand = inst->operand(operand_index);
                auto choices = _choices(operand);
                _dispatch(b, choices, [&](const Path &path, size_t) {
                    resolver.values[operand] = _materialize(b, path);
                    self(self, operand_index + 1u);
                });
            };
            emit(emit, 0u);
            b.br(merge);
            b.set_insertion_point(merge);
            if (result != nullptr) { inst->replace_all_uses_with(b.load(inst->type(), result)); }
            for (auto *next : tail) { b.append(next->remove_self()); }
            inst->remove_self();
        }
        // Dependencies between access chains are broken before releasing them.
        for (auto *gep : symbolic_geps) { gep->set_operands({}); }
        for (auto *gep : symbolic_geps) { gep->remove_self(); }
    }
};

CoroCallAliasLowering::CoroCallAliasLowering(Function *root) : _impl{luisa::make_unique<Impl>(root)} {}
CoroCallAliasLowering::~CoroCallAliasLowering() noexcept = default;
void CoroCallAliasLowering::add_parameter(Argument *argument, XIRBuilder &prologue) { _impl->add_parameter(argument, prologue); }
void CoroCallAliasLowering::bind_call(CallInst *call, XIRBuilder &builder, XIRBuilder &prologue) { _impl->bind_call(call, builder, prologue); }
void CoroCallAliasLowering::lower_uses(XIRBuilder &prologue) { _impl->lower_uses(prologue); }

}// namespace luisa::compute::xir::detail
