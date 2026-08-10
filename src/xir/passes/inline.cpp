#include <luisa/xir/passes/inline.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/builder.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/undefined.h>
#include <luisa/xir/function.h>
#include <luisa/xir/value.h>
#include <luisa/xir/metadata/signature_constraint.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

[[nodiscard]] static luisa::vector<CallInst *> collect_call_sites(Function *callee) noexcept {
    luisa::vector<CallInst *> calls;
    for (auto &&use : callee->use_list()) {
        if (auto user = use->user(); user != nullptr && user->isa<CallInst>()) {
            auto call = static_cast<CallInst *>(user);
            if (call->callee() == callee &&
                use == call->operand_use(CallInst::operand_index_callee)) {
                calls.push_back(call);
            }
        }
    }
    return calls;
}

class InlineValueNumbering {
public:
    static constexpr auto invalid_id =
        std::numeric_limits<uint32_t>::max();

private:
    // The exact value vector owns the keys once. Hash slots contain id + 1,
    // leaving zero as the empty sentinel. At a maximum load factor of 1/2,
    // linear probing always terminates and needs no per-value node allocation.
    luisa::vector<const Value *> _values;
    luisa::vector<uint32_t> _slots;

    [[nodiscard]] size_t _initial_slot(const Value *value) const noexcept {
        return luisa::hash<const Value *>{}(value) &
               (_slots.size() - 1u);
    }

public:
    explicit InlineValueNumbering(size_t value_count) noexcept {
        LUISA_ASSERT(
            value_count < static_cast<size_t>(invalid_id),
            "Inline clone layout exceeds the 32-bit dense value domain.");
        if (value_count == 0u) { return; }
        constexpr auto max_slot_count =
            size_t{1u} << (std::numeric_limits<size_t>::digits - 1u);
        LUISA_ASSERT(
            value_count <= max_slot_count / 2u,
            "Inline clone layout size overflow.");
        auto slot_count = std::bit_ceil(value_count * 2u);
        _values.reserve(value_count);
        _slots.resize(slot_count, 0u);
    }

    void append(const Value *value) noexcept {
        LUISA_ASSERT(value != nullptr && !_slots.empty(),
                     "Invalid inline clone layout value.");
        LUISA_ASSERT(_values.size() < _slots.size() &&
                         _values.size() <
                             static_cast<size_t>(invalid_id),
                     "Inline clone layout exceeded its capacity.");
        auto id = static_cast<uint32_t>(_values.size());
        auto slot = _initial_slot(value);
        for (;;) {
            auto encoded = _slots[slot];
            if (encoded == 0u) {
                _values.emplace_back(value);
                _slots[slot] = id + 1u;
                return;
            }
            LUISA_ASSERT(_values[encoded - 1u] != value,
                         "Duplicate value in inline clone layout.");
            slot = (slot + 1u) & (_slots.size() - 1u);
        }
    }

    [[nodiscard]] uint32_t find(const Value *value) const noexcept {
        if (value == nullptr || _slots.empty()) { return invalid_id; }
        auto slot = _initial_slot(value);
        for (;;) {
            auto encoded = _slots[slot];
            if (encoded == 0u) { return invalid_id; }
            auto id = encoded - 1u;
            if (_values[id] == value) { return id; }
            slot = (slot + 1u) & (_slots.size() - 1u);
        }
    }

    [[nodiscard]] size_t size() const noexcept {
        return _values.size();
    }
};

class InlineValueResolver final : public InstructionCloneValueResolver {
    const InlineValueNumbering *_numbering;
    luisa::vector<Value *> _dense_values;
    luisa::vector<uint8_t> _dense_mapped;
    luisa::unordered_map<const Value *, Value *> _fallback_map;
    Module *_module;
    size_t *_dense_fallback_count;

    void _note_dense_fallback() noexcept {
        if (_numbering != nullptr &&
            _dense_fallback_count != nullptr) {
            ++*_dense_fallback_count;
        }
    }

public:
    explicit InlineValueResolver(
        Function *caller_func,
        const InlineValueNumbering *numbering = nullptr,
        size_t *dense_fallback_count = nullptr) noexcept
        : _numbering{numbering},
          _module{caller_func->parent_module()},
          _dense_fallback_count{dense_fallback_count} {
        if (_numbering != nullptr) {
            _dense_values.resize(_numbering->size(), nullptr);
            _dense_mapped.resize(_numbering->size(), false);
        }
    }
    void emplace(const Value *from, Value *to) noexcept {
        if (_numbering != nullptr) {
            auto id = _numbering->find(from);
            if (id != InlineValueNumbering::invalid_id) {
                if (!_dense_mapped[id]) {
                    _dense_values[id] = to;
                    _dense_mapped[id] = true;
                }
                return;
            }
            _note_dense_fallback();
        }
        _fallback_map.emplace(from, to);
    }
    [[nodiscard]] Value *resolve(const Value *value) noexcept override {
        if (value == nullptr) return nullptr;
        switch (value->derived_value_tag()) {
            case DerivedValueTag::UNDEFINED:
            case DerivedValueTag::FUNCTION:
            case DerivedValueTag::CONSTANT:
            case DerivedValueTag::SPECIAL_REGISTER:
                return const_cast<Value *>(value);
            default: break;
        }
        if (_numbering != nullptr) {
            auto id = _numbering->find(value);
            if (id != InlineValueNumbering::invalid_id) {
                if (_dense_mapped[id]) {
                    return _dense_values[id];
                }
                if (value->derived_value_tag() ==
                    DerivedValueTag::BASIC_BLOCK) {
                    return nullptr;
                }
                if (value->type() != nullptr) {
                    auto *undef =
                        _module->create_undefined(value->type());
                    _dense_values[id] = undef;
                    _dense_mapped[id] = true;
                    return undef;
                }
                LUISA_ERROR(
                    "Inline: unresolved value (tag={}).",
                    to_string(value->derived_value_tag()));
            }
            _note_dense_fallback();
        }
        auto it = _fallback_map.find(value);
        if (it == _fallback_map.end()) {
            if (value->derived_value_tag() == DerivedValueTag::BASIC_BLOCK) {
                return nullptr;
            }
            if (value->type() != nullptr) {
                auto undef = _module->create_undefined(value->type());
                _fallback_map.emplace(value, undef);
                return undef;
            }
            LUISA_ERROR("Inline: unresolved value (tag={}).", to_string(value->derived_value_tag()));
        }
        return it->second;
    }
};

[[nodiscard]] static bool ordinary_inline_cost_is_bounded(
    size_t call_site_count, size_t instruction_count) noexcept {
    if (call_site_count == 1u) {
        return instruction_count <=
               default_inline_single_use_instruction_budget;
    }
    return call_site_count <= default_inline_multi_use_call_site_budget &&
           instruction_count <=
               default_inline_multi_use_instruction_budget;
}

[[nodiscard]] static bool has_single_block(FunctionDefinition *def) noexcept {
    size_t count = 0u;
    for (auto *block : def->basic_blocks()) {
        static_cast<void>(block);
        ++count;
    }
    return count == 1u && def->body_block() != nullptr;
}

[[nodiscard]] static bool can_inline_single_block(
    FunctionDefinition *def) noexcept {
    if (!has_single_block(def)) { return false; }
    auto *block = def->body_block();
    if (!block->is_terminated() ||
        !block->terminator()->isa<ReturnInst>()) {
        return false;
    }
    for (auto *inst : block->instructions()) {
        if ((inst->is_terminator() && !inst->isa<ReturnInst>()) ||
            inst->isa<PhiInst>()) {
            return false;
        }
    }
    return true;
}

struct InlineBarrierFlags {
    bool disallow_autodiff_scope{false};
    bool allow_autodiff_scope{false};
};

static void accumulate_inline_barrier(
    const Instruction *instruction,
    InlineBarrierFlags &flags) noexcept {
    switch (instruction->derived_instruction_tag()) {
        case DerivedInstructionTag::IF:
        case DerivedInstructionTag::SWITCH:
        case DerivedInstructionTag::LOOP:
        case DerivedInstructionTag::SIMPLE_LOOP:
        case DerivedInstructionTag::BREAK:
        case DerivedInstructionTag::CONTINUE:
        case DerivedInstructionTag::RAY_QUERY_LOOP:
        case DerivedInstructionTag::RAY_QUERY_DISPATCH:
        case DerivedInstructionTag::OUTLINE:
        case DerivedInstructionTag::CORO_SUSPEND:
        case DerivedInstructionTag::CORO_RESUME:
        case DerivedInstructionTag::CORO_TERMINATE:
            flags.disallow_autodiff_scope = true;
            flags.allow_autodiff_scope = true;
            break;
        case DerivedInstructionTag::AUTODIFF_SCOPE:
            flags.disallow_autodiff_scope = true;
            break;
        default: break;
    }
}

[[nodiscard]] static bool contains_inline_barrier(
    FunctionDefinition *def,
    bool allow_autodiff_scope) noexcept {
    InlineBarrierFlags flags;
    for (auto *block : def->basic_blocks()) {
        for (auto *inst : block->instructions()) {
            accumulate_inline_barrier(inst, flags);
            if (allow_autodiff_scope ?
                    flags.allow_autodiff_scope :
                    flags.disallow_autodiff_scope) {
                return true;
            }
        }
    }
    return false;
}

[[nodiscard]] static bool typed_value_operand_valid(const Value *value) noexcept {
    return value != nullptr && value->type() != nullptr &&
           !value->isa<BasicBlock>() && !value->isa<Function>() &&
           !value->type()->is_resource();
}

[[nodiscard]] static bool rvalue_operand_valid(const Value *value) noexcept {
    return typed_value_operand_valid(value) && !value->is_lvalue();
}

[[nodiscard]] static bool argument_matches(const Argument *formal,
                                           const Value *actual) noexcept {
    if (formal == nullptr || actual == nullptr ||
        actual->type() != formal->type()) {
        return false;
    }
    if (formal->is_resource()) {
        return actual->isa<ResourceArgument>() && !actual->is_lvalue();
    }
    if (formal->is_reference()) {
        return typed_value_operand_valid(actual) && actual->is_lvalue();
    }
    return rvalue_operand_valid(actual);
}

struct InlineFunctionSummary {
    bool has_valid_definition{false};
    bool has_single_block{false};
    bool can_inline_single_block{false};
    bool return_shape_is_valid{false};
    bool has_return_metadata{false};
    bool has_single_body_metadata{false};
    bool contains_barrier_disallow_autodiff{false};
    bool contains_barrier_allow_autodiff{false};
    size_t block_count{0u};
    size_t instruction_count{0u};
    size_t local_value_count{0u};
};

[[nodiscard]] static InlineFunctionSummary summarize_inline_function(
    Function *function, size_t &summary_function_count,
    size_t &summary_instruction_scan_count) noexcept {
    InlineFunctionSummary summary;
    auto *definition =
        function == nullptr ? nullptr : function->definition();
    if (definition == nullptr || definition->body_block() == nullptr) {
        return summary;
    }
    summary.has_valid_definition = true;
    ++summary_function_count;
    auto block_count = size_t{0u};
    auto return_count = size_t{0u};
    auto single_block_forbidden = false;
    auto return_shape_is_valid = true;
    InlineBarrierFlags barrier_flags;
    for (auto *block : definition->basic_blocks()) {
        ++block_count;
        for (auto *inst : block->instructions()) {
            ++summary.instruction_count;
            ++summary_instruction_scan_count;
            single_block_forbidden |=
                (inst->is_terminator() &&
                 !inst->isa<ReturnInst>()) ||
                inst->isa<PhiInst>();
            accumulate_inline_barrier(inst, barrier_flags);
            if (!inst->isa<ReturnInst>()) { continue; }
            auto *return_inst = static_cast<ReturnInst *>(inst);
            auto *return_value = return_inst->return_value();
            return_shape_is_valid &=
                (function->type() == nullptr) ==
                    (return_value == nullptr) &&
                (return_value == nullptr ||
                 return_value->type() == function->type());
            summary.has_return_metadata |=
                !inst->metadata_list().empty();
            ++return_count;
        }
    }
    summary.block_count = block_count;
    summary.local_value_count =
        function->arguments().count_size() + block_count +
        summary.instruction_count;
    summary.contains_barrier_disallow_autodiff =
        barrier_flags.disallow_autodiff_scope;
    summary.contains_barrier_allow_autodiff =
        barrier_flags.allow_autodiff_scope;
    summary.has_single_block = block_count == 1u;
    if (summary.has_single_block) {
        auto *body = definition->body_block();
        summary.has_single_body_metadata =
            !body->metadata_list().empty();
        summary.can_inline_single_block =
            body->is_terminated() &&
            body->terminator()->isa<ReturnInst>() &&
            !single_block_forbidden;
    }
    summary.return_shape_is_valid =
        return_shape_is_valid &&
        (function->type() == nullptr || return_count != 0u);
    return summary;
}

class InlineCloneLayout {
    InlineValueNumbering _numbering;
    luisa::vector<BasicBlock *> _reachable_blocks;
    luisa::vector<uint8_t> _reachable_values;

public:
    InlineCloneLayout(Function *function,
                      const InlineFunctionSummary &summary) noexcept
        : _numbering{summary.local_value_count} {
        LUISA_ASSERT(summary.has_valid_definition,
                     "Cannot number an invalid inline callee.");
        auto *definition = function->definition();
        for (auto *argument : function->arguments()) {
            _numbering.append(argument);
        }
        for (auto *block : definition->basic_blocks()) {
            _numbering.append(block);
            for (auto *instruction : block->instructions()) {
                _numbering.append(instruction);
            }
        }
        LUISA_ASSERT(
            _numbering.size() == summary.local_value_count,
            "Inline callee changed while building its clone layout.");
        _reachable_values.resize(_numbering.size(), false);
        _reachable_blocks.reserve(summary.block_count);
        definition->traverse_basic_blocks(
            BasicBlockTraversalOrder::REVERSE_POST_ORDER,
            [&](BasicBlock *block) noexcept {
                auto id = _numbering.find(block);
                LUISA_ASSERT(
                    id != InlineValueNumbering::invalid_id,
                    "Reachable inline block is not in its definition.");
                LUISA_ASSERT(!_reachable_values[id],
                             "Inline RPO contains a duplicate block.");
                _reachable_values[id] = true;
                _reachable_blocks.emplace_back(block);
            });
    }

    [[nodiscard]] const InlineValueNumbering &numbering() const noexcept {
        return _numbering;
    }

    [[nodiscard]] const luisa::vector<BasicBlock *> &
    reachable_blocks() const noexcept {
        return _reachable_blocks;
    }

    [[nodiscard]] bool is_reachable(
        const BasicBlock *block) const noexcept {
        auto id = _numbering.find(block);
        return id != InlineValueNumbering::invalid_id &&
               _reachable_values[id];
    }

    [[nodiscard]] size_t value_count() const noexcept {
        return _numbering.size();
    }
};

class InlineCalleeVersion {
    Function *_function;
    InlineInfo *_info;
    InlineFunctionSummary _summary;
    luisa::unique_ptr<InlineCloneLayout> _clone_layout;

public:
    InlineCalleeVersion(Function *function, InlineInfo &info) noexcept
        : _function{function},
          _info{&info},
          _summary{summarize_inline_function(
              function, info.inline_pass_summary_function_count,
              info.inline_pass_summary_instruction_scan_count)} {}

    [[nodiscard]] const InlineFunctionSummary &summary() const noexcept {
        return _summary;
    }

    [[nodiscard]] const InlineCloneLayout &acquire_clone_layout() noexcept {
        if (_clone_layout == nullptr) {
            _clone_layout = luisa::make_unique<InlineCloneLayout>(
                _function, _summary);
            ++_info->inline_pass_clone_layout_function_count;
            _info->inline_pass_clone_layout_value_count +=
                _clone_layout->value_count();
        }
        ++_info->inline_pass_dense_resolver_apply_count;
        return *_clone_layout;
    }

    [[nodiscard]] size_t *dense_fallback_count() const noexcept {
        return &_info->inline_pass_dense_resolver_fallback_count;
    }
};

// A successful inline operation only removes a CallInst, moves existing
// caller instructions, and clones a callee already proven free of every
// inline barrier. It therefore preserves both caller barrier predicates even
// though it changes the caller definition. The cache is valid for the whole
// pass invocation, not merely for one structural version of the caller.
class InlineCallerBarrierCache {
    luisa::unordered_map<Function *, InlineBarrierFlags> _flags;
    InlineInfo *_info;

public:
    explicit InlineCallerBarrierCache(
        InlineInfo &info, size_t expected_function_count = 0u) noexcept
        : _info{&info} {
        _flags.reserve(expected_function_count);
    }

    [[nodiscard]] bool contains(
        Function *function, bool allow_autodiff_scope) noexcept {
        if (auto iter = _flags.find(function);
            iter != _flags.end()) {
            ++_info->inline_pass_caller_barrier_cache_hit_count;
            return allow_autodiff_scope ?
                       iter->second.allow_autodiff_scope :
                       iter->second.disallow_autodiff_scope;
        }
        auto *definition = function->definition();
        LUISA_ASSERT(definition != nullptr,
                     "Cannot inspect an undefined inline caller.");
        InlineBarrierFlags flags;
        ++_info->inline_pass_caller_barrier_function_count;
        for (auto *block : definition->basic_blocks()) {
            for (auto *instruction : block->instructions()) {
                ++_info
                      ->inline_pass_caller_barrier_instruction_scan_count;
                accumulate_inline_barrier(instruction, flags);
                if (flags.disallow_autodiff_scope &&
                    flags.allow_autodiff_scope) {
                    break;
                }
            }
            if (flags.disallow_autodiff_scope &&
                flags.allow_autodiff_scope) {
                break;
            }
        }
        auto [iter, inserted] = _flags.emplace(function, flags);
        LUISA_ASSERT(inserted,
                     "Failed to cache inline caller barrier state.");
        return allow_autodiff_scope ?
                   iter->second.allow_autodiff_scope :
                   iter->second.disallow_autodiff_scope;
    }
};

[[nodiscard]] static bool validate_call_shape(
    CallInst *call, Function *callee,
    const InlineFunctionSummary &summary) noexcept {
    if (call->type() != callee->type() ||
        call->argument_count() !=
            callee->arguments().count_size() ||
        !summary.has_valid_definition ||
        !summary.return_shape_is_valid) {
        return false;
    }
    auto argument_index = 0u;
    for (auto *formal : callee->arguments()) {
        if (!argument_matches(
                formal, call->argument(argument_index++))) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] static bool has_unmappable_inline_metadata(
    CallInst *call,
    const InlineFunctionSummary &summary) noexcept {
    if (call == nullptr || !summary.has_valid_definition ||
        !call->metadata_list().empty() ||
        summary.has_return_metadata) {
        return true;
    }
    if (summary.has_single_block) {
        return summary.has_single_body_metadata;
    }
    auto *call_block = call->parent_block();
    return call_block == nullptr ||
           !call_block->metadata_list().empty();
}

[[nodiscard]] static bool validate_call_shape(CallInst *call,
                                              Function *callee) noexcept {
    if (call->type() != callee->type()) { return false; }
    if (call->argument_count() != callee->arguments().count_size()) { return false; }
    auto argument_index = 0u;
    for (auto *formal : callee->arguments()) {
        auto *actual = call->argument(argument_index++);
        if (!argument_matches(formal, actual)) { return false; }
    }
    auto *definition = callee->definition();
    if (definition == nullptr || definition->body_block() == nullptr) {
        return false;
    }
    auto return_count = 0u;
    for (auto *block : definition->basic_blocks()) {
        for (auto *inst : block->instructions()) {
            if (!inst->isa<ReturnInst>()) { continue; }
            auto *return_inst = static_cast<ReturnInst *>(inst);
            auto *return_value = return_inst->return_value();
            if ((call->type() == nullptr) != (return_value == nullptr)) { return false; }
            if (return_value != nullptr && return_value->type() != call->type()) { return false; }
            return_count++;
        }
    }
    return call->type() == nullptr || return_count != 0u;
}

[[nodiscard]] static bool has_unmappable_inline_metadata(
    CallInst *call, FunctionDefinition *callee_def) noexcept {
    if (call == nullptr || callee_def == nullptr) { return true; }
    if (!call->metadata_list().empty()) { return true; }
    if (has_single_block(callee_def)) {
        // Single-block inlining splices instructions into the caller's
        // existing block. The callee block itself has no one-to-one
        // replacement, and merging its metadata into the caller block can
        // create duplicate metadata kinds or change the annotation's scope.
        if (auto *body = callee_def->body_block();
            body != nullptr && !body->metadata_list().empty()) {
            return true;
        }
    } else {
        auto *call_block = call->parent_block();
        if (call_block == nullptr ||
            !call_block->metadata_list().empty()) {
            return true;
        }
    }
    for (auto *block : callee_def->basic_blocks()) {
        for (auto *inst : block->instructions()) {
            if (inst->isa<ReturnInst>() &&
                !inst->metadata_list().empty()) {
                return true;
            }
        }
    }
    return false;
}

[[nodiscard]] static bool inline_single_block_call(CallInst *call,
                                                   Function *callee,
                                                   bool prevalidated = false,
                                                   const InlineCloneLayout *clone_layout = nullptr,
                                                   size_t *dense_fallback_count = nullptr) noexcept {
    auto *callee_def = callee->definition();
    auto *caller = call->parent_function();
    if (callee_def == nullptr || caller == nullptr ||
        (!prevalidated && !can_inline_single_block(callee_def))) {
        return false;
    }
    auto *block = callee_def->body_block();
    XIRBuilder builder;
    builder.set_insertion_point(call);
    InlineValueResolver resolver{
        caller,
        clone_layout == nullptr ? nullptr :
                                  &clone_layout->numbering(),
        dense_fallback_count};
    auto *module = caller->parent_module();
    size_t i = 0u;
    for (auto *arg : callee->arguments()) {
        auto *call_arg = i < call->argument_count() ?
                             call->argument(i) :
                             static_cast<Value *>(module->create_undefined(arg->type()));
        if (arg->is_lvalue() && !call_arg->is_lvalue()) {
            auto *tmp = builder.alloca_local(arg->type());
            builder.store(tmp, call_arg);
            resolver.emplace(arg, tmp);
        } else {
            resolver.emplace(arg, call_arg);
        }
        ++i;
    }
    for (auto *inst : block->instructions()) {
        if (inst->isa<ReturnInst>()) {
            auto *ret = static_cast<ReturnInst *>(inst);
            if (call->type() != nullptr) {
                if (ret->return_value() == nullptr) { return false; }
                call->replace_all_uses_with(resolver.resolve(ret->return_value()));
            }
            call->remove_self();
            return true;
        }
        auto *clone = inst->clone_with_metadata(builder, resolver);
        LUISA_ASSERT(clone != nullptr, "Inline: clone failed.");
        resolver.emplace(inst, clone);
    }
    return false;
}

[[nodiscard]] static bool inline_multi_block_call(
    CallInst *call, Function *callee,
    const InlineCloneLayout *clone_layout = nullptr,
    size_t *dense_fallback_count = nullptr) noexcept {
    auto callee_def = callee->definition();
    if (!callee_def) return false;
    auto caller_func = call->parent_function();
    if (!caller_func) return false;
    auto caller_def = caller_func->definition();
    if (!caller_def) return false;

    auto call_block = call->parent_block();
    luisa::vector<Instruction *> to_move;
    auto past_call = false;
    for (auto *inst : call_block->instructions()) {
        if (inst == call) {
            past_call = true;
        } else if (past_call) {
            to_move.emplace_back(inst);
        }
    }

    auto module = caller_func->parent_module();
    XIRBuilder builder;
    InlineValueResolver resolver{
        caller_func,
        clone_layout == nullptr ? nullptr :
                                  &clone_layout->numbering(),
        dense_fallback_count};

    // Map callee args -> call args
    {
        size_t i = 0;
        for (auto arg : callee->arguments()) {
            auto call_arg = i < call->argument_count() ? call->argument(i) : static_cast<Value *>(module->create_undefined(arg->type()));
            if (arg->is_lvalue() && !call_arg->is_lvalue()) {
                builder.set_insertion_point(call);
                auto tmp = builder.alloca_local(arg->type());
                builder.store(tmp, call_arg);
                resolver.emplace(arg, tmp);
            } else {
                resolver.emplace(arg, call_arg);
            }
            ++i;
        }
    }

    // Collect reachable callee blocks in RPO for instruction cloning.
    luisa::vector<BasicBlock *> discovered_callee_blocks;
    auto *callee_blocks = clone_layout == nullptr ?
                              &discovered_callee_blocks :
                              &clone_layout->reachable_blocks();
    if (clone_layout == nullptr) {
        callee_def->traverse_basic_blocks(
            BasicBlockTraversalOrder::REVERSE_POST_ORDER,
            [&](BasicBlock *bb) noexcept {
                discovered_callee_blocks.push_back(bb);
            });
    }
    luisa::unordered_set<const BasicBlock *> callee_reachable;
    if (clone_layout == nullptr) {
        callee_reachable.reserve(callee_blocks->size());
        callee_reachable.insert(callee_blocks->begin(),
                                callee_blocks->end());
    }
    auto is_callee_reachable =
        [&](const BasicBlock *block) noexcept {
            return clone_layout == nullptr ?
                       callee_reachable.contains(block) :
                       clone_layout->is_reachable(block);
        };

    luisa::vector<BasicBlock *> new_blocks;
    new_blocks.reserve(callee_blocks->size());
    for (auto bb : *callee_blocks) {
        auto nb = caller_func->create_basic_block();
        for (auto *metadata : bb->metadata_list()) {
            nb->metadata_list().push_front(metadata->clone());
        }
        new_blocks.push_back(nb);
        resolver.emplace(bb, nb);
    }

    // Create single-exit merge block and return value alloca
    auto merge_bb = caller_func->create_basic_block();

    // Map unreachable blocks to dedicated empty blocks so structured
    // terminators (IfInst, LoopInst) referencing them get valid targets.
    {
        for (auto bb : callee_def->basic_blocks()) {
            if (!is_callee_reachable(bb)) {
                auto nb = caller_func->create_basic_block();
                for (auto *metadata : bb->metadata_list()) {
                    nb->metadata_list().push_front(metadata->clone());
                }
                resolver.emplace(bb, nb);
                builder.set_insertion_point(nb);
                builder.unreachable_();
            }
        }
    }
    Instruction *ret_alloca = nullptr;
    if (call->type()) {
        builder.set_insertion_point(call);
        ret_alloca = builder.alloca_local(call->type());
    }

    // Clone instructions from callee into new blocks.
    // We make two passes:
    //   Pass 1: clone all alloca instructions first. They have no operand
    //   dependencies and may be referenced by instructions that appear
    //   earlier in RPO (e.g., alloca inside a branch referenced from a
    //   predecessor block after previous inlining).
    //   Pass 2: clone everything else.
    luisa::vector<std::pair<const PhiInst *, PhiInst *>> phi_nodes;
    for (size_t i = 0; i < callee_blocks->size(); ++i) {
        builder.set_insertion_point(new_blocks[i]);
        for (auto inst : (*callee_blocks)[i]->instructions()) {
            if (inst->isa<AllocaInst>()) {
                auto c = inst->clone_with_metadata(builder, resolver);
                LUISA_ASSERT(c, "Inline: clone failed.");
                resolver.emplace(inst, c);
            }
        }
    }
    for (size_t i = 0; i < callee_blocks->size(); ++i) {
        builder.set_insertion_point(new_blocks[i]);
        for (auto inst : (*callee_blocks)[i]->instructions()) {
            if (inst->isa<ReturnInst>()) {
                auto r = static_cast<ReturnInst *>(inst);
                if (ret_alloca && r->operand_count() > 0) {
                    auto val = resolver.resolve(r->operand(0));
                    builder.store(ret_alloca, val);
                }
                builder.br(merge_bb);
            } else if (inst->isa<PhiInst>()) {
                auto phi = static_cast<PhiInst *>(inst);
                auto dup_phi = builder.phi(phi->type());
                for (auto *metadata : phi->metadata_list()) {
                    dup_phi->metadata_list().push_front(metadata->clone());
                }
                phi_nodes.emplace_back(phi, dup_phi);
                resolver.emplace(inst, dup_phi);
            } else if (!inst->isa<AllocaInst>()) {
                auto c = inst->clone_with_metadata(builder, resolver);
                LUISA_ASSERT(c, "Inline: clone failed.");
                resolver.emplace(inst, c);
            }
        }
    }
    // Patch phi node operands now that all blocks and values are mapped.
    for (auto [original_phi, dup_phi] : phi_nodes) {
        // Only executable callee blocks were cloned. Disconnected owned
        // blocks are represented by terminal empty shells, so their original
        // outgoing edges no longer exist and must not survive as Phi labels.
        // Keeping those labels creates an incoming-without-predecessor pair.
        for (size_t i = 0; i < original_phi->incoming_count(); i++) {
            auto incoming = original_phi->incoming(i);
            if (!is_callee_reachable(incoming.block)) { continue; }
            auto resolved_value = resolver.resolve(incoming.value);
            auto resolved_block = resolver.resolve(incoming.block);
            dup_phi->add_incoming(
                resolved_value, static_cast<BasicBlock *>(resolved_block));
        }
        if (original_phi->parent_block() == callee_def->body_block()) {
            // A function entry is reached by an implicit invocation edge,
            // which is not represented in a standalone function's Phi list.
            // Inlining materializes that edge as call_block -> cloned entry.
            // The entry value on that formerly implicit edge is undefined.
            dup_phi->add_incoming(
                module->create_undefined(original_phi->type()), call_block);
        }
    }

    // Wire caller: split the call block
    auto *entry_block = static_cast<BasicBlock *>(
        resolver.resolve(callee_def->body_block()));
    LUISA_ASSERT(entry_block != nullptr,
                 "Inline callee entry block was not mapped.");
    luisa::vector<BasicBlock *> original_successors;
    if (call_block->is_terminated()) {
        call_block->traverse_successors(
            false, [&](BasicBlock *successor) noexcept {
                original_successors.emplace_back(successor);
            });
    }

    // Load return value in merge block
    if (ret_alloca) {
        builder.set_insertion_point(merge_bb);
        auto loaded = builder.load(call->type(), ret_alloca);
        call->replace_all_uses_with(loaded);
    }

    // Remove the call
    call->remove_self();

    // Move post-call instructions to merge_bb
    builder.set_insertion_point(merge_bb);
    for (auto inst : to_move) {
        if (!inst->is_terminator()) {
            auto m = inst->remove_self();
            builder.append(std::move(m));
        }
    }

    // Move terminator from call_block to merge_bb
    if (call_block->is_terminated()) {
        auto m = call_block->terminator()->remove_self();
        builder.set_insertion_point(merge_bb);
        if (merge_bb->is_terminated()) merge_bb->terminator()->remove_self();
        builder.append(std::move(m));
        // Moving the terminator transfers every original outgoing edge from
        // call_block to merge_bb. Phi incoming labels describe predecessor
        // edges, so their labels must move with those edges. This also covers
        // duplicate branch targets and a former self-edge to call_block.
        for (auto *successor : original_successors) {
            for (auto *inst : successor->instructions()) {
                if (!inst->isa<PhiInst>()) { break; }
                auto *phi = static_cast<PhiInst *>(inst);
                for (auto i = 0u; i < phi->incoming_count(); ++i) {
                    auto incoming = phi->incoming(i);
                    if (incoming.block == call_block) {
                        phi->set_incoming(i, incoming.value, merge_bb);
                    }
                }
            }
        }
    }

    // Branch from call_block to inlined entry
    builder.set_insertion_point(call_block);
    builder.br(entry_block);

    // Defensive: if merge_bb has no terminator (can happen when call_block
    // was already unterminated in malformed IR), add unreachable.
    if (!merge_bb->is_terminated()) {
        builder.set_insertion_point(merge_bb);
        builder.unreachable_();
    }

    return true;
}

[[nodiscard]] static bool inline_call(CallInst *call, Function *callee,
                                      InlineInfo &info,
                                      InlineOptions options = {},
                                      luisa::unordered_set<CallInst *> *reported_malformed_calls = nullptr,
                                      InlineCalleeVersion *callee_version = nullptr,
                                      InlineCallerBarrierCache *caller_barriers = nullptr) noexcept {
    auto *callee_def = callee->definition();
    auto *caller = call->parent_function();
    auto *caller_def = caller == nullptr ? nullptr : caller->definition();
    if (callee_def == nullptr || caller_def == nullptr) { return false; }
    if (callee_def->body_block() == nullptr) {
        ++info.skipped_declaration_call_count;
        return false;
    }
    auto *callee_summary = callee_version == nullptr ?
                               nullptr :
                               &callee_version->summary();
    auto call_shape_is_valid = callee_summary == nullptr ?
                                   validate_call_shape(call, callee) :
                                   validate_call_shape(
                                       call, callee, *callee_summary);
    if (!call_shape_is_valid) {
        if (reported_malformed_calls == nullptr ||
            reported_malformed_calls->emplace(call).second) {
            ++info.rejected_malformed_call_count;
        }
        return false;
    }
    if (callee->find_metadata<SignatureConstraintMD>() != nullptr) {
        ++info.skipped_constrained_call_count;
        return false;
    }
    auto has_unmappable_metadata = callee_summary == nullptr ?
                                       has_unmappable_inline_metadata(
                                           call, callee_def) :
                                       has_unmappable_inline_metadata(
                                           call, *callee_summary);
    if (has_unmappable_metadata) {
        ++info.skipped_metadata_call_count;
        return false;
    }
    auto callee_contains_barrier = callee_summary == nullptr ?
                                       contains_inline_barrier(
                                           callee_def, false) :
                                       callee_summary
                                           ->contains_barrier_disallow_autodiff;
    if (callee_contains_barrier) {
        ++info.skipped_structured_call_count;
        return false;
    }
    auto single_block = callee_summary == nullptr ?
                            has_single_block(callee_def) :
                            callee_summary->has_single_block;
    if (single_block) {
        if (callee_summary != nullptr &&
            !callee_summary->can_inline_single_block) {
            return false;
        }
        auto *clone_layout = callee_version == nullptr ?
                                 nullptr :
                                 &callee_version->acquire_clone_layout();
        return inline_single_block_call(
            call, callee, callee_summary != nullptr, clone_layout,
            callee_version == nullptr ?
                nullptr :
                callee_version->dense_fallback_count());
    }
    auto caller_contains_barrier = caller_barriers == nullptr ?
                                       contains_inline_barrier(
                                           caller_def,
                                           options.allow_autodiff_scope_in_caller) :
                                       caller_barriers->contains(
                                           caller,
                                           options.allow_autodiff_scope_in_caller);
    if (caller_contains_barrier) {
        ++info.skipped_structured_call_count;
        return false;
    }
    auto *clone_layout = callee_version == nullptr ?
                             nullptr :
                             &callee_version->acquire_clone_layout();
    return inline_multi_block_call(
        call, callee, clone_layout,
        callee_version == nullptr ?
            nullptr :
            callee_version->dense_fallback_count());
}

[[nodiscard]] static luisa::unordered_set<Function *>
find_recursive_callables(luisa::span<Function *const> callables,
                         InlineInfo &info) noexcept {
    const auto function_count = callables.size();
    info.recursion_analysis_function_count += function_count;
    luisa::unordered_map<Function *, size_t> function_ids;
    function_ids.reserve(function_count);
    for (auto i = 0u; i < function_count; ++i) {
        function_ids.emplace(callables[i], i);
    }

    struct CallEdge {
        size_t caller;
        size_t callee;
    };
    luisa::vector<CallEdge> edges;
    luisa::vector<uint8_t> has_self_edge(function_count, false);
    // A linked CallInst contributes exactly one use at its callee operand.
    // Enumerating those uses is therefore set-equivalent to scanning every
    // owned instruction, while visiting only actual function uses. Filtering
    // the operand identity excludes a function passed as an ordinary value;
    // parent_function() preserves calls in disconnected owned blocks.
    for (auto callee_id = 0u; callee_id < function_count;
         ++callee_id) {
        auto *callee = callables[callee_id];
        for (auto &&use : callee->use_list()) {
            ++info.recursion_analysis_call_use_visit_count;
            auto *user = use->user();
            if (user == nullptr || !user->isa<CallInst>()) { continue; }
            auto *call = static_cast<CallInst *>(user);
            if (call->callee() != callee ||
                use != call->operand_use(
                           CallInst::operand_index_callee)) {
                continue;
            }
            auto *caller = call->parent_function();
            if (auto iter = function_ids.find(caller);
                iter != function_ids.end()) {
                auto caller_id = iter->second;
                edges.emplace_back(caller_id, callee_id);
                has_self_edge[caller_id] |=
                    caller_id == callee_id;
            }
        }
    }
    info.recursion_analysis_edge_count += edges.size();

    // Materialize both directions as CSR. Kosaraju's two traversals identify
    // the exact cyclic SCCs in O(F + E) time and storage. A singleton SCC is
    // recursive only when its function has an explicit self edge.
    luisa::vector<size_t> forward_offsets(function_count + 1u, 0u);
    luisa::vector<size_t> reverse_offsets(function_count + 1u, 0u);
    for (auto edge : edges) {
        ++forward_offsets[edge.caller + 1u];
        ++reverse_offsets[edge.callee + 1u];
    }
    for (auto i = 0u; i < function_count; ++i) {
        forward_offsets[i + 1u] += forward_offsets[i];
        reverse_offsets[i + 1u] += reverse_offsets[i];
    }
    auto forward_cursors = forward_offsets;
    auto reverse_cursors = reverse_offsets;
    luisa::vector<size_t> forward_targets(edges.size());
    luisa::vector<size_t> reverse_targets(edges.size());
    for (auto edge : edges) {
        forward_targets[forward_cursors[edge.caller]++] =
            edge.callee;
        reverse_targets[reverse_cursors[edge.callee]++] =
            edge.caller;
    }

    struct DFSFrame {
        size_t function;
        size_t next_edge;
    };
    luisa::vector<uint8_t> visited(function_count, false);
    luisa::vector<size_t> finish_order;
    finish_order.reserve(function_count);
    luisa::vector<DFSFrame> dfs;
    dfs.reserve(function_count);
    for (auto root = 0u; root < function_count; ++root) {
        if (visited[root]) { continue; }
        visited[root] = true;
        ++info.recursion_analysis_vertex_visit_count;
        dfs.emplace_back(root, forward_offsets[root]);
        while (!dfs.empty()) {
            auto &frame = dfs.back();
            auto edge_end = forward_offsets[frame.function + 1u];
            if (frame.next_edge != edge_end) {
                auto next = forward_targets[frame.next_edge++];
                ++info.recursion_analysis_edge_visit_count;
                if (!visited[next]) {
                    visited[next] = true;
                    ++info.recursion_analysis_vertex_visit_count;
                    dfs.emplace_back(next, forward_offsets[next]);
                }
            } else {
                finish_order.emplace_back(frame.function);
                dfs.pop_back();
            }
        }
    }

    std::fill(visited.begin(), visited.end(), false);
    luisa::vector<size_t> worklist;
    worklist.reserve(function_count);
    luisa::vector<size_t> component;
    component.reserve(function_count);
    luisa::unordered_set<Function *> recursive;
    recursive.reserve(function_count);
    for (auto order_index = finish_order.size(); order_index != 0u;
         --order_index) {
        auto root = finish_order[order_index - 1u];
        if (visited[root]) { continue; }
        component.clear();
        visited[root] = true;
        ++info.recursion_analysis_vertex_visit_count;
        worklist.emplace_back(root);
        while (!worklist.empty()) {
            auto current = worklist.back();
            worklist.pop_back();
            component.emplace_back(current);
            for (auto edge_index = reverse_offsets[current];
                 edge_index != reverse_offsets[current + 1u];
                 ++edge_index) {
                ++info.recursion_analysis_edge_visit_count;
                auto next = reverse_targets[edge_index];
                if (!visited[next]) {
                    visited[next] = true;
                    ++info.recursion_analysis_vertex_visit_count;
                    worklist.emplace_back(next);
                }
            }
        }
        if (component.size() > 1u ||
            has_self_edge[component.front()]) {
            for (auto function_id : component) {
                recursive.emplace(callables[function_id]);
            }
        }
    }
    return recursive;
}

static void inline_run(Module *module, InlineInfo &info) noexcept {
    if (module == nullptr) { return; }
    // Early exit if no callables
    bool has_callables = false;
    for (auto f : module->function_list()) {
        if (f->derived_function_tag() == DerivedFunctionTag::CALLABLE) {
            has_callables = true;
            break;
        }
    }
    if (!has_callables) return;

    // Collect callables (safe iteration before modification)
    luisa::vector<Function *> callables;
    for (auto f : module->function_list())
        if (f->derived_function_tag() == DerivedFunctionTag::CALLABLE)
            callables.push_back(f);

    auto recursive = find_recursive_callables(callables, info);
    InlineCallerBarrierCache caller_barriers{info, callables.size()};

    // Defer removal to after iteration to avoid corrupting the list
    luisa::vector<Function *> to_remove;
    for (auto callee : callables) {
        auto def = callee->definition();
        if (!def) continue;
        if (recursive.contains(callee)) {
            ++info.skipped_recursive_callable_count;
            continue;
        }
        auto edges = collect_call_sites(callee);
        if (edges.empty()) continue;

        // The summary counts the definition's owned instructions rather than
        // traversing its executable CFG. Besides matching the amount of IR an
        // inline clone must inspect, this keeps the heuristic total for
        // bodyless and disconnected definitions. Reuse the same summary for
        // legality and cloning so the callee is scanned only once.
        InlineCalleeVersion callee_version{callee, info};
        auto instruction_count =
            callee_version.summary().instruction_count;
        if (!ordinary_inline_cost_is_bounded(
                edges.size(), instruction_count)) {
            ++info.skipped_costly_callable_count;
            continue;
        }

        for (auto call : edges)
            if (inline_call(call, callee, info, {}, nullptr,
                            &callee_version, &caller_barriers))
                info.inlined_call_count++;

        if (callee->use_list().empty()) { to_remove.push_back(callee); }
    }
    for (auto callee : to_remove) {
        callee->remove_self();
        info.removed_callable_count++;
    }
}

}// namespace detail

namespace {

void set_inline_report(const InlineInfo &info, PassReport *report) noexcept {
    if (report == nullptr) { return; }
    report->set("inlined_call", info.inlined_call_count);
    report->set("removed_callable", info.removed_callable_count);
    report->set("skipped_recursive_callable",
                info.skipped_recursive_callable_count);
    report->set("skipped_structured_call",
                info.skipped_structured_call_count);
    report->set("skipped_constrained_call",
                info.skipped_constrained_call_count);
    report->set("skipped_metadata_call",
                info.skipped_metadata_call_count);
    report->set("skipped_declaration_call",
                info.skipped_declaration_call_count);
    report->set("rejected_malformed_call",
                info.rejected_malformed_call_count);
    report->set("skipped_costly_callable",
                info.skipped_costly_callable_count);
    report->set("call_site_summary_function",
                info.call_site_summary_function_count);
    report->set("call_site_summary_instruction_scan",
                info.call_site_summary_instruction_scan_count);
    report->set("call_site_cached_apply",
                info.call_site_cached_apply_count);
    report->set("call_site_revalidated_apply",
                info.call_site_revalidated_apply_count);
    report->set("call_site_clone_layout_function",
                info.call_site_clone_layout_function_count);
    report->set("call_site_clone_layout_value",
                info.call_site_clone_layout_value_count);
    report->set("call_site_dense_resolver_apply",
                info.call_site_dense_resolver_apply_count);
    report->set("call_site_dense_resolver_fallback",
                info.call_site_dense_resolver_fallback_count);
    report->set("inline_pass_summary_function",
                info.inline_pass_summary_function_count);
    report->set("inline_pass_summary_instruction_scan",
                info.inline_pass_summary_instruction_scan_count);
    report->set("inline_pass_clone_layout_function",
                info.inline_pass_clone_layout_function_count);
    report->set("inline_pass_clone_layout_value",
                info.inline_pass_clone_layout_value_count);
    report->set("inline_pass_dense_resolver_apply",
                info.inline_pass_dense_resolver_apply_count);
    report->set("inline_pass_dense_resolver_fallback",
                info.inline_pass_dense_resolver_fallback_count);
    report->set("inline_pass_caller_barrier_function",
                info.inline_pass_caller_barrier_function_count);
    report->set("inline_pass_caller_barrier_instruction_scan",
                info.inline_pass_caller_barrier_instruction_scan_count);
    report->set("inline_pass_caller_barrier_cache_hit",
                info.inline_pass_caller_barrier_cache_hit_count);
    report->set("recursion_analysis_function",
                info.recursion_analysis_function_count);
    report->set("recursion_analysis_call_use_visit",
                info.recursion_analysis_call_use_visit_count);
    report->set("recursion_analysis_edge",
                info.recursion_analysis_edge_count);
    report->set("recursion_analysis_vertex_visit",
                info.recursion_analysis_vertex_visit_count);
    report->set("recursion_analysis_edge_visit",
                info.recursion_analysis_edge_visit_count);
}

}// namespace

InlineInfo inline_pass_run_on_module(Module *module, PassReport *report) noexcept {
    InlineInfo info;
    if (module != nullptr) {
        detail::inline_run(module, info);
    }
    set_inline_report(info, report);
    return info;
}

InlineInfo inline_all_pass_run_on_module(Module *module, PassReport *report) noexcept {
    return inline_all_pass_run_on_module(module, {}, report);
}

InlineInfo inline_all_pass_run_on_module(Module *module, InlineOptions options, PassReport *report) noexcept {
    InlineInfo info;
    if (!module) {
        set_inline_report(info, report);
        return info;
    }
    luisa::unordered_set<CallInst *> reported_malformed_calls;
    detail::InlineCallerBarrierCache caller_barriers{info};
    for (;;) {
        luisa::vector<Function *> callables;
        for (auto f : module->function_list())
            if (f->derived_function_tag() == DerivedFunctionTag::CALLABLE)
                callables.push_back(f);
        if (callables.empty()) break;
        auto recursive = detail::find_recursive_callables(callables, info);
        luisa::unordered_set<Function *> callable_set{callables.begin(), callables.end()};
        luisa::vector<Function *> leaves;
        for (auto callee : callables) {
            if (recursive.contains(callee)) {
                ++info.skipped_recursive_callable_count;
                continue;
            }
            auto def = callee->definition();
            if (!def) continue;
            bool is_leaf = true;
            for (auto *block : def->basic_blocks()) {
                for (auto *inst : block->instructions()) {
                    if (!is_leaf) { break; }
                    if (inst->derived_instruction_tag() ==
                        DerivedInstructionTag::CALL) {
                        auto *call = static_cast<const CallInst *>(inst);
                        if (callable_set.contains(const_cast<Function *>(
                                static_cast<const Function *>(
                                    call->callee())))) {
                            is_leaf = false;
                        }
                    }
                }
                if (!is_leaf) { break; }
            }
            if (is_leaf) leaves.push_back(callee);
        }
        if (leaves.empty()) break;
        bool progress = false;
        for (auto callee : leaves) {
            auto def = callee->definition();
            if (!def) continue;
            auto edges = detail::collect_call_sites(callee);
            if (edges.empty()) { continue; }
            detail::InlineCalleeVersion callee_version{callee, info};
            for (auto call : edges) {
                if (detail::inline_call(
                        call, callee, info, options,
                        &reported_malformed_calls,
                        &callee_version, &caller_barriers)) {
                    info.inlined_call_count++;
                    progress = true;
                }
            }
        }
        if (!progress) break;
        for (auto *callee : leaves) {
            if (callee->use_list().empty()) {
                callee->remove_self();
                ++info.removed_callable_count;
            }
        }
    }
    set_inline_report(info, report);
    return info;
}

InlineInfo inline_call_sites_pass_run_on_module(
    Module *module, luisa::span<CallInst *const> call_sites,
    InlineOptions options, PassReport *report) noexcept {
    InlineInfo info;
    if (module == nullptr || call_sites.empty()) {
        set_inline_report(info, report);
        return info;
    }
    luisa::unordered_set<CallInst *> reported_malformed_calls;
    luisa::vector<Function *> all_callables;
    for (auto *function : module->function_list()) {
        if (function->derived_function_tag() ==
            DerivedFunctionTag::CALLABLE) {
            all_callables.emplace_back(function);
        }
    }
    auto recursive =
        detail::find_recursive_callables(all_callables, info);
    luisa::unordered_map<Function *, detail::InlineFunctionSummary>
        function_summaries;
    auto summary_of = [&](Function *function) noexcept {
        if (auto iter = function_summaries.find(function);
            iter != function_summaries.end()) {
            return iter->second;
        }
        auto summary =
            detail::summarize_inline_function(
                function,
                info.call_site_summary_function_count,
                info.call_site_summary_instruction_scan_count);
        function_summaries.emplace(function, summary);
        return summary;
    };
    struct PreparedInlineCall {
        CallInst *call;
        Function *callee;
        bool single_block;
    };
    luisa::unordered_set<Function *> reported_recursive;
    luisa::unordered_set<CallInst *> seen_calls;
    luisa::vector<PreparedInlineCall> plan;
    plan.reserve(call_sites.size());
    for (auto *call : call_sites) {
        if (call == nullptr) {
            ++info.rejected_malformed_call_count;
            continue;
        }
        if (!seen_calls.emplace(call).second) { continue; }
        auto *callee = call->callee();
        auto *caller = call->parent_function();
        auto malformed = callee == nullptr || caller == nullptr ||
                         caller->parent_module() != module ||
                         callee->parent_module() != module ||
                         callee->derived_function_tag() !=
                             DerivedFunctionTag::CALLABLE ||
                         callee->definition() == nullptr;
        if (!malformed &&
            callee->definition()->body_block() == nullptr) {
            ++info.skipped_declaration_call_count;
            continue;
        }
        auto callee_summary = malformed ?
                                  detail::InlineFunctionSummary{} :
                                  summary_of(callee);
        malformed |= !malformed &&
                     !detail::validate_call_shape(
                         call, callee, callee_summary);
        if (!malformed && callee_summary.has_single_block &&
            !callee_summary.can_inline_single_block) {
            malformed = true;
        }
        if (malformed) {
            ++info.rejected_malformed_call_count;
            continue;
        }
        if (recursive.contains(callee)) {
            if (reported_recursive.emplace(callee).second) {
                ++info.skipped_recursive_callable_count;
            }
            continue;
        }
        if (callee->find_metadata<SignatureConstraintMD>() != nullptr) {
            ++info.skipped_constrained_call_count;
            continue;
        }
        if (detail::has_unmappable_inline_metadata(
                call, callee_summary)) {
            ++info.skipped_metadata_call_count;
            continue;
        }
        auto caller_contains_barrier = false;
        if (!callee_summary.has_single_block) {
            auto caller_summary = summary_of(caller);
            caller_contains_barrier =
                options.allow_autodiff_scope_in_caller ?
                    caller_summary.contains_barrier_allow_autodiff :
                    caller_summary.contains_barrier_disallow_autodiff;
        }
        if (callee_summary.contains_barrier_disallow_autodiff ||
            caller_contains_barrier) {
            ++info.skipped_structured_call_count;
            continue;
        }
        plan.emplace_back(call, callee,
                          callee_summary.has_single_block);
    }
    if (info.rejected_malformed_call_count != 0u ||
        info.skipped_recursive_callable_count != 0u ||
        info.skipped_structured_call_count != 0u ||
        info.skipped_constrained_call_count != 0u ||
        info.skipped_metadata_call_count != 0u ||
        info.skipped_declaration_call_count != 0u ||
        plan.size() != seen_calls.size()) {
        set_inline_report(info, report);
        return info;
    }
    // Every summary above describes an immutable function definition. An
    // inline operation mutates only its caller, so a prepared callee remains
    // valid unless that function was itself an earlier caller in this plan.
    // Track exactly that invalidation frontier: independent call sites reuse
    // their preflight decision, while nested call chains retain the complete
    // generic validation path after their callee changes.
    luisa::unordered_set<Function *> mutated_functions;
    luisa::unordered_map<Function *, detail::InlineCloneLayout>
        clone_layouts;
    clone_layouts.reserve(plan.size());
    auto clone_layout_of = [&](Function *function) noexcept
        -> detail::InlineCloneLayout & {
        if (auto iter = clone_layouts.find(function);
            iter != clone_layouts.end()) {
            return iter->second;
        }
        auto summary_iter = function_summaries.find(function);
        LUISA_ASSERT(summary_iter != function_summaries.end(),
                     "Missing prevalidated inline function summary.");
        auto [iter, inserted] = clone_layouts.try_emplace(
            function, function, summary_iter->second);
        LUISA_ASSERT(inserted,
                     "Failed to cache inline clone layout.");
        ++info.call_site_clone_layout_function_count;
        info.call_site_clone_layout_value_count +=
            iter->second.value_count();
        return iter->second;
    };
    for (auto &&prepared : plan) {
        auto *call = prepared.call;
        auto *callee = prepared.callee;
        auto *caller = call->parent_function();
        auto revalidate = mutated_functions.contains(callee);
        auto succeeded = false;
        if (revalidate) {
            ++info.call_site_revalidated_apply_count;
            succeeded = detail::inline_call(
                call, callee, info, options,
                &reported_malformed_calls);
        } else {
            ++info.call_site_cached_apply_count;
            ++info.call_site_dense_resolver_apply_count;
            auto &clone_layout = clone_layout_of(callee);
            succeeded = prepared.single_block ?
                            detail::inline_single_block_call(
                                call, callee, true, &clone_layout,
                                &info.call_site_dense_resolver_fallback_count) :
                            detail::inline_multi_block_call(
                                call, callee, &clone_layout,
                                &info.call_site_dense_resolver_fallback_count);
        }
        if (!succeeded) {
            LUISA_ERROR_WITH_LOCATION(
                "Inline call-site plan changed after successful preflight.");
        }
        mutated_functions.emplace(caller);
        ++info.inlined_call_count;
    }
    luisa::unordered_set<Function *> planned_callees;
    for (auto &&prepared : plan) {
        planned_callees.emplace(prepared.callee);
    }
    luisa::vector<Function *> unused_callables;
    for (auto *function : module->function_list()) {
        if (planned_callees.contains(function) &&
            function->use_list().empty()) {
            unused_callables.emplace_back(function);
        }
    }
    for (auto *callee : unused_callables) {
        callee->remove_self();
        ++info.removed_callable_count;
    }
    set_inline_report(info, report);
    return info;
}

}// namespace luisa::compute::xir
