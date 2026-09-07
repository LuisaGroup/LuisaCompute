#include <chrono>
#include <cstdlib>

#include <luisa/ast/type.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/logging.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/atomic.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/passes/pointer_usage.h>

namespace luisa::compute::xir {

struct PointerUsageAnalysis::Impl {
    struct InstructionSnapshot {
        Instruction *inst;
        BasicBlock *parent;
        const Type *type;
        DerivedInstructionTag tag;
        luisa::vector<Value *> operands;
        AtomicOp atomic_op{};
    };

    struct CalleeSnapshot {
        Function *function;
        luisa::weak_ptr<uint8_t> lifetime_token;
        luisa::vector<Argument *> arguments;
    };

    struct PointerPath {
        luisa::vector<Value *> pointers;
        luisa::vector<luisa::vector<Value *>> edges;
        bool connected{false};
    };

    struct AccessEvent {
        Value *pointer;
        AggregateFieldBitmask mask;
        bool read;
        bool write;
        bool definite_write;

        AccessEvent(Value *p, AggregateFieldBitmask m, bool r, bool w, bool d) noexcept
            : pointer{p}, mask{std::move(m)}, read{r}, write{w}, definite_write{d} {}
    };

    struct MaskResult {
        AggregateFieldBitmask mask;
        bool precise{true};
        bool valid{true};

        explicit MaskResult(const Type *type) noexcept : mask{type} {}
    };

    struct EmitResult {
        bool emitted{false};
        bool conservative{false};
        bool invalid{false};
    };

    struct IndexResult {
        bool valid{false};
        bool constant{false};
        size_t value{0u};
    };

    FunctionDefinition *def{nullptr};
    luisa::weak_ptr<uint8_t> lifetime_token;
    bool snapshot_valid{false};
    BasicBlock *snapshot_body{nullptr};
    luisa::vector<Argument *> snapshot_arguments;
    luisa::vector<BasicBlock *> snapshot_blocks;
    luisa::vector<InstructionSnapshot> snapshot_instructions;
    luisa::vector<CalleeSnapshot> snapshot_callees;
    PointerUsageAnalysisInfo info;
    luisa::vector<BasicBlock *> ordered_blocks;
    luisa::unordered_map<BasicBlock *, size_t> block_indices;
    luisa::unordered_set<BasicBlock *> owned_blocks;
    luisa::unordered_set<BasicBlock *> reachable_blocks;
    luisa::unordered_map<BasicBlock *, luisa::vector<BasicBlock *>> successors;
    luisa::unordered_map<BasicBlock *, luisa::vector<BasicBlock *>> predecessors;
    luisa::vector<Value *> tracked_pointers;
    luisa::unordered_set<Value *> tracked_pointer_set;
    // The transfer functions are pointwise in pointer-view identity after
    // access projection: no K/T/L coordinate reads another pointer's state.
    // Keep the complete pointer graph above for alias validation, while this
    // subset selects the exact product-lattice coordinates to materialize.
    luisa::vector<Value *> result_pointers;
    luisa::unordered_set<Value *> result_pointer_set;
    luisa::unordered_map<Value *, luisa::vector<Value *>>
        result_pointers_by_root;
    luisa::unordered_map<Value *, PointerPath> paths;
    luisa::unordered_set<Value *> resolving_paths;
    luisa::unordered_map<BasicBlock *, luisa::vector<AccessEvent>> events;
    luisa::unordered_map<BasicBlock *, BasicBlockPointerUsage> block_results;
    size_t forward_block_evaluation_count{0u};
    size_t backward_block_evaluation_count{0u};

    void clear() noexcept {
        def = nullptr;
        lifetime_token.reset();
        snapshot_valid = false;
        snapshot_body = nullptr;
        snapshot_arguments.clear();
        snapshot_blocks.clear();
        snapshot_instructions.clear();
        snapshot_callees.clear();
        info = {};
        ordered_blocks.clear();
        block_indices.clear();
        owned_blocks.clear();
        reachable_blocks.clear();
        successors.clear();
        predecessors.clear();
        tracked_pointers.clear();
        tracked_pointer_set.clear();
        result_pointers.clear();
        result_pointer_set.clear();
        result_pointers_by_root.clear();
        paths.clear();
        resolving_paths.clear();
        events.clear();
        block_results.clear();
        forward_block_evaluation_count = 0u;
        backward_block_evaluation_count = 0u;
    }

    void capture_snapshot() noexcept {
        lifetime_token = def->lifetime_token();
        snapshot_body = def->body_block();
        for (auto *argument : def->arguments()) { snapshot_arguments.emplace_back(argument); }
        luisa::unordered_set<Function *> captured_callees;
        for (auto *block : def->basic_blocks()) {
            snapshot_blocks.emplace_back(block);
            for (auto *inst : block->instructions()) {
                InstructionSnapshot snapshot{
                    .inst = inst,
                    .parent = inst->parent_block(),
                    .type = inst->type(),
                    .tag = inst->derived_instruction_tag()};
                for (size_t i = 0u; i < inst->operand_count(); ++i) {
                    snapshot.operands.emplace_back(inst->operand(i));
                }
                if (inst->isa<AtomicInst>()) {
                    snapshot.atomic_op = static_cast<AtomicInst *>(inst)->op();
                }
                if (inst->isa<CallInst>() && inst->operand_count() != 0u) {
                    auto *callee_value = inst->operand(0u);
                    if (callee_value != nullptr && callee_value->isa<Function>()) {
                        auto *callee = static_cast<Function *>(callee_value);
                        if (captured_callees.emplace(callee).second) {
                            CalleeSnapshot callee_snapshot{
                                .function = callee,
                                .lifetime_token = callee->lifetime_token()};
                            for (auto *argument : callee->arguments()) {
                                callee_snapshot.arguments.emplace_back(argument);
                            }
                            snapshot_callees.emplace_back(std::move(callee_snapshot));
                        }
                    }
                }
                snapshot_instructions.emplace_back(std::move(snapshot));
            }
        }
        snapshot_valid = true;
    }

    [[nodiscard]] bool is_current() const noexcept {
        if (!snapshot_valid || def == nullptr || lifetime_token.expired() ||
            def->body_block() != snapshot_body) {
            return false;
        }
        size_t argument_index = 0u;
        for (auto *argument : def->arguments()) {
            if (argument_index >= snapshot_arguments.size() ||
                snapshot_arguments[argument_index] != argument) {
                return false;
            }
            ++argument_index;
        }
        if (argument_index != snapshot_arguments.size()) { return false; }
        size_t block_index = 0u;
        size_t instruction_index = 0u;
        for (auto *block : def->basic_blocks()) {
            if (block_index >= snapshot_blocks.size() || snapshot_blocks[block_index] != block) { return false; }
            ++block_index;
            for (auto *inst : block->instructions()) {
                if (instruction_index >= snapshot_instructions.size()) { return false; }
                auto &snapshot = snapshot_instructions[instruction_index++];
                if (snapshot.inst != inst || snapshot.parent != inst->parent_block() ||
                    snapshot.type != inst->type() || snapshot.tag != inst->derived_instruction_tag() ||
                    snapshot.operands.size() != inst->operand_count()) {
                    return false;
                }
                for (size_t i = 0u; i < inst->operand_count(); ++i) {
                    if (snapshot.operands[i] != inst->operand(i)) { return false; }
                }
                if (inst->isa<AtomicInst>() &&
                    snapshot.atomic_op != static_cast<AtomicInst *>(inst)->op()) {
                    return false;
                }
            }
        }
        if (block_index != snapshot_blocks.size() || instruction_index != snapshot_instructions.size()) {
            return false;
        }
        for (auto &&callee_snapshot : snapshot_callees) {
            if (callee_snapshot.lifetime_token.expired()) { return false; }
            size_t index = 0u;
            for (auto *argument : callee_snapshot.function->arguments()) {
                if (index >= callee_snapshot.arguments.size() ||
                    callee_snapshot.arguments[index] != argument) {
                    return false;
                }
                ++index;
            }
            if (index != callee_snapshot.arguments.size()) { return false; }
        }
        return true;
    }

    [[nodiscard]] static bool is_integer_index_type(const Type *type) noexcept {
        if (type == nullptr) { return false; }
        switch (type->tag()) {
            case Type::Tag::INT8:
            case Type::Tag::UINT8:
            case Type::Tag::INT16:
            case Type::Tag::UINT16:
            case Type::Tag::INT32:
            case Type::Tag::UINT32:
            case Type::Tag::INT64:
            case Type::Tag::UINT64: return true;
            default: return false;
        }
    }

    [[nodiscard]] static IndexResult decode_index(Value *value) noexcept {
        IndexResult result;
        if (value == nullptr || !is_integer_index_type(value->type())) { return result; }
        result.valid = true;
        if (!value->isa<Constant>()) { return result; }
        result.constant = true;
        uint64_t decoded = 0u;
        if (!try_decode_constant_nonnegative_integer(value, decoded) ||
            decoded > static_cast<uint64_t>(SIZE_MAX)) {
            result.valid = false;
        } else {
            result.value = static_cast<size_t>(decoded);
        }
        return result;
    }

    [[nodiscard]] static const Type *indexed_type(const Type *type, const IndexResult &index) noexcept {
        if (type == nullptr || !index.valid) { return nullptr; }
        switch (type->tag()) {
            case Type::Tag::VECTOR:
            case Type::Tag::ARRAY:
                if (index.constant && index.value >= type->dimension()) { return nullptr; }
                return type->element();
            case Type::Tag::MATRIX:
                if (index.constant && index.value >= type->dimension()) { return nullptr; }
                return Type::vector(type->element(), type->dimension());
            case Type::Tag::STRUCTURE: {
                if (!index.constant) { return nullptr; }
                auto members = type->members();
                return index.value < members.size() ? members[index.value] : nullptr;
            }
            default: return nullptr;
        }
    }

    [[nodiscard]] bool initialize_cfg(FunctionDefinition *function) noexcept {
        def = function;
        if (def == nullptr || def->body_block() == nullptr) { return false; }
        for (auto *block : def->basic_blocks()) {
            if (block == nullptr || block->parent_function() != def) { return false; }
            block_indices.emplace(block, ordered_blocks.size());
            ordered_blocks.emplace_back(block);
            owned_blocks.emplace(block);
        }
        if (!owned_blocks.contains(def->body_block())) { return false; }
        luisa::vector<BasicBlock *> worklist{def->body_block()};
        reachable_blocks.emplace(def->body_block());
        while (!worklist.empty()) {
            auto *block = worklist.back();
            worklist.pop_back();
            if (!block->is_terminated()) { return false; }
            auto *terminator = block->terminator();
            for (auto *inst : block->instructions()) {
                if (inst == nullptr || inst->parent_block() != block ||
                    (inst->is_terminator() && inst != terminator)) {
                    return false;
                }
            }
            auto &block_successors = successors[block];
            for (auto *use : terminator->operand_uses()) {
                auto *value = use == nullptr ? nullptr : use->value();
                if (value == nullptr || !value->isa<BasicBlock>()) { continue; }
                auto *successor = static_cast<BasicBlock *>(value);
                if (!owned_blocks.contains(successor)) { return false; }
                if (std::find(block_successors.begin(), block_successors.end(), successor) == block_successors.end()) {
                    block_successors.emplace_back(successor);
                    predecessors[successor].emplace_back(block);
                }
                if (reachable_blocks.emplace(successor).second) {
                    worklist.emplace_back(successor);
                }
            }
        }
        return true;
    }

    void track_pointer(Value *pointer) noexcept {
        if (pointer != nullptr && pointer->type() != nullptr && tracked_pointer_set.emplace(pointer).second) {
            tracked_pointers.emplace_back(pointer);
        }
    }

    void collect_pointers() noexcept {
        for (auto *argument : def->arguments()) {
            if (argument->is_reference() && argument->parent_function() == def) {
                track_pointer(argument);
            }
        }
        for (auto *block : ordered_blocks) {
            if (!reachable_blocks.contains(block)) { continue; }
            for (auto *inst : block->instructions()) {
                if (inst->isa<AllocaInst>() || inst->isa<GEPInst>()) {
                    track_pointer(inst);
                }
            }
        }
        info.tracked_pointer_count = tracked_pointers.size();
    }

    void select_result_pointers(
        luisa::optional<luisa::span<Value *const>> requested) noexcept {
        auto append = [&](Value *pointer) noexcept {
            if (pointer != nullptr &&
                tracked_pointer_set.contains(pointer) &&
                result_pointer_set.emplace(pointer).second) {
                result_pointers.emplace_back(pointer);
            }
        };
        if (requested) {
            for (auto *pointer : *requested) {
                if (pointer == nullptr ||
                    !tracked_pointer_set.contains(pointer)) {
                    ++info.invalid_access_count;
                    continue;
                }
                append(pointer);
            }
        } else {
            for (auto *pointer : tracked_pointers) { append(pointer); }
        }
        info.materialized_pointer_count = result_pointers.size();
    }

    void index_result_pointers_by_root() noexcept {
        // Access projection is root-disjoint: a pointer view can overlap a
        // requested coordinate only when both resolve to the same reference
        // argument or alloca root. Partitioning the product coordinates by
        // that root is therefore an exact sparse index, not an alias
        // approximation. Pointer discovery and validation remain global.
        for (auto *pointer : result_pointers) {
            auto *path = resolve_path(pointer);
            if (path == nullptr || !path->connected ||
                path->pointers.empty()) {
                ++info.invalid_access_count;
                continue;
            }
            result_pointers_by_root[path->pointers.front()]
                .emplace_back(pointer);
        }
    }

    [[nodiscard]] bool validate_projection(const Type *base_type, luisa::span<Value *const> indices,
                                           const Type *result_type) noexcept {
        auto *type = base_type;
        for (auto *index_value : indices) {
            auto index = decode_index(index_value);
            type = indexed_type(type, index);
            if (type == nullptr) { return false; }
        }
        return type == result_type;
    }

    [[nodiscard]] PointerPath *resolve_path(Value *pointer) noexcept {
        if (auto iter = paths.find(pointer); iter != paths.end()) { return &iter->second; }
        if (!tracked_pointer_set.contains(pointer) || !resolving_paths.emplace(pointer).second) { return nullptr; }
        PointerPath path;
        if (pointer->isa<ReferenceArgument>() || pointer->isa<AllocaInst>()) {
            path.pointers.emplace_back(pointer);
            path.connected = true;
        } else if (pointer->isa<GEPInst>()) {
            auto *gep = static_cast<GEPInst *>(pointer);
            auto *base = gep->operand_count() == 0u ? nullptr : gep->base();
            if (auto *base_path = resolve_path(base); base_path != nullptr && base_path->connected) {
                path = *base_path;
                path.pointers.emplace_back(pointer);
                luisa::vector<Value *> edge;
                edge.reserve(gep->index_count());
                for (size_t i = 0u; i < gep->index_count(); ++i) { edge.emplace_back(gep->index(i)); }
                path.edges.emplace_back(std::move(edge));
                path.connected = true;
            }
        }
        resolving_paths.erase(pointer);
        auto [iter, inserted] = paths.emplace(pointer, std::move(path));
        static_cast<void>(inserted);
        return &iter->second;
    }

    [[nodiscard]] MaskResult build_mask(Value *pointer, luisa::span<Value *const> indices) noexcept {
        MaskResult result{pointer->type()};
        auto *type = pointer->type();
        luisa::vector<size_t> constant_indices;
        constant_indices.reserve(indices.size());
        for (auto *index_value : indices) {
            auto index = decode_index(index_value);
            if (!index.valid) {
                result.valid = false;
                result.precise = false;
                break;
            }
            auto *next = indexed_type(type, index);
            if (next == nullptr) {
                result.valid = false;
                result.precise = false;
                break;
            }
            if (!index.constant) { result.precise = false; }
            constant_indices.emplace_back(index.value);
            type = next;
        }
        if (!result.valid || !result.precise) {
            result.mask.set(true);
        } else {
            result.mask.access(luisa::span{constant_indices}).set(true);
        }
        return result;
    }

    [[nodiscard]] static luisa::vector<Value *> flatten_indices(const PointerPath &path) noexcept {
        luisa::vector<Value *> indices;
        for (auto &&edge : path.edges) {
            for (auto *index : edge) { indices.emplace_back(index); }
        }
        return indices;
    }

    [[nodiscard]] static bool equivalent_index(Value *lhs, Value *rhs) noexcept {
        if (lhs == rhs) { return true; }
        auto lhs_index = decode_index(lhs);
        auto rhs_index = decode_index(rhs);
        return lhs_index.valid && rhs_index.valid && lhs_index.constant && rhs_index.constant &&
               lhs_index.value == rhs_index.value;
    }

    [[nodiscard]] static bool equivalent_indices(luisa::span<Value *const> lhs,
                                                 luisa::span<Value *const> rhs) noexcept {
        if (lhs.size() != rhs.size()) { return false; }
        for (size_t i = 0u; i < lhs.size(); ++i) {
            if (!equivalent_index(lhs[i], rhs[i])) { return false; }
        }
        return true;
    }

    [[nodiscard]] static bool bit_is_set(const AggregateFieldBitmask &mask, size_t index) noexcept {
        return (mask.raw_bits()[index / 64u] & (uint64_t{1u} << (index % 64u))) != 0u;
    }

    static void set_bit(AggregateFieldBitmask &mask, size_t index) noexcept {
        mask.raw_bits()[index / 64u] |= uint64_t{1u} << (index % 64u);
    }

    [[nodiscard]] static luisa::optional<size_t> first_set_bit(const AggregateFieldBitmask &mask) noexcept {
        for (size_t i = 0u; i < mask.size(); ++i) {
            if (bit_is_set(mask, i)) { return i; }
        }
        return luisa::nullopt;
    }

    [[nodiscard]] static size_t set_bit_count(const AggregateFieldBitmask &mask) noexcept {
        size_t count = 0u;
        for (size_t i = 0u; i < mask.size(); ++i) {
            count += bit_is_set(mask, i) ? 1u : 0u;
        }
        return count;
    }

    [[nodiscard]] EmitResult emit_access(BasicBlock *block, Value *pointer,
                                         luisa::span<Value *const> extra_indices,
                                         bool read, bool write, bool force_may) noexcept {
        EmitResult result;
        auto *path = resolve_path(pointer);
        if (path == nullptr || !path->connected || path->pointers.empty()) {
            result.invalid = tracked_pointer_set.contains(pointer);
            return result;
        }
        auto *root = path->pointers.front();
        auto access_indices = flatten_indices(*path);
        for (auto *index : extra_indices) { access_indices.emplace_back(index); }
        auto root_access = build_mask(root, access_indices);
        result.invalid |= !root_access.valid;
        result.conservative |= force_may || !root_access.valid || !root_access.precise;
        auto pointer_indices = flatten_indices(*path);
        auto candidates = result_pointers_by_root.find(root);
        if (candidates == result_pointers_by_root.end()) {
            // The access cannot affect any requested lattice coordinate. Its
            // path and indices were still validated above, preserving the
            // full-function malformed-access contract of projected analysis.
            return result;
        }
        for (auto *target : candidates->second) {
            auto *target_path = resolve_path(target);
            if (target_path == nullptr || !target_path->connected ||
                target_path->pointers.empty()) {
                result.invalid = true;
                continue;
            }
            LUISA_DEBUG_ASSERT(
                target_path->pointers.front() == root,
                "Pointer-result root index is inconsistent.");
            auto target_indices = flatten_indices(*target_path);
            AggregateFieldBitmask target_mask{target->type()};
            auto definite = false;
            auto exact_view = pointer->type() == target->type() &&
                              equivalent_indices(pointer_indices, target_indices);
            if (exact_view) {
                auto relative = build_mask(target, extra_indices);
                target_mask = std::move(relative.mask);
                result.invalid |= !relative.valid;
                result.conservative |= !relative.valid || !relative.precise;
                definite = root_access.valid && relative.valid && relative.precise;
            } else {
                auto target_projection = build_mask(root, target_indices);
                auto projection_size_matches = !target_projection.precise ||
                                               set_bit_count(target_projection.mask) == target_mask.size();
                result.invalid |= !target_projection.valid || !projection_size_matches;
                if (root_access.valid && root_access.precise &&
                    target_projection.valid && target_projection.precise &&
                    projection_size_matches) {
                    if (auto target_offset = first_set_bit(target_projection.mask)) {
                        for (size_t i = 0u; i < target_mask.size(); ++i) {
                            auto root_index = *target_offset + i;
                            if (root_index < root_access.mask.size() && bit_is_set(root_access.mask, root_index)) {
                                set_bit(target_mask, i);
                            }
                        }
                        definite = true;
                    }
                } else if (root_access.valid && root_access.precise && root_access.mask.access().all() &&
                           target_projection.valid && projection_size_matches) {
                    target_mask.set(true);
                    definite = true;
                    result.conservative |= !target_projection.valid || !target_projection.precise;
                } else {
                    target_mask.set(true);
                    result.conservative = true;
                }
            }
            if (target_mask.access().none()) { continue; }
            events[block].emplace_back(target, std::move(target_mask), read, write,
                                       write && !force_may && definite);
            result.emitted = true;
        }
        return result;
    }

    void record_emit(EmitResult result) noexcept {
        if (result.conservative) { ++info.conservative_access_count; }
        if (result.invalid) { ++info.invalid_access_count; }
    }

    [[nodiscard]] bool is_tracked_pointer(Value *value) const noexcept {
        return value != nullptr && tracked_pointer_set.contains(value);
    }

    void collect_instruction_events(BasicBlock *block, Instruction *inst) noexcept {
        luisa::vector<bool> handled(inst->operand_count(), false);
        switch (inst->derived_instruction_tag()) {
            case DerivedInstructionTag::ALLOCA: break;
            case DerivedInstructionTag::GEP: {
                if (inst->operand_count() != 0u) {
                    handled[0] = true;
                    for (size_t i = 1u; i < handled.size(); ++i) { handled[i] = true; }
                }
                break;
            }
            case DerivedInstructionTag::LOAD: {
                if (inst->operand_count() != 1u) {
                    ++info.invalid_access_count;
                    if (inst->operand_count() == 0u) { break; }
                }
                handled[0] = true;
                auto *pointer = inst->operand(0u);
                if (is_tracked_pointer(pointer)) {
                    record_emit(emit_access(block, pointer, {}, true, false, false));
                } else {
                    ++info.invalid_access_count;
                }
                break;
            }
            case DerivedInstructionTag::STORE: {
                if (inst->operand_count() != 2u) {
                    ++info.invalid_access_count;
                    if (inst->operand_count() < 2u) { break; }
                }
                handled[0] = true;
                auto *pointer = inst->operand(0u);
                if (is_tracked_pointer(pointer)) {
                    record_emit(emit_access(block, pointer, {}, false, true, false));
                } else {
                    ++info.invalid_access_count;
                }
                break;
            }
            case DerivedInstructionTag::ATOMIC: {
                auto *atomic = static_cast<AtomicInst *>(inst);
                auto value_count = atomic->value_count();
                if (inst->operand_count() < 1u + value_count) {
                    ++info.invalid_access_count;
                    break;
                }
                auto index_count = inst->operand_count() - 1u - value_count;
                handled[0] = true;
                luisa::vector<Value *> indices;
                for (size_t i = 0u; i < index_count; ++i) {
                    handled[1u + i] = true;
                    indices.emplace_back(inst->operand(1u + i));
                }
                auto *pointer = inst->operand(0u);
                if (is_tracked_pointer(pointer)) {
                    record_emit(emit_access(block, pointer, luisa::span{indices}, true, true, false));
                }
                break;
            }
            case DerivedInstructionTag::CALL: {
                if (inst->operand_count() == 0u) {
                    ++info.invalid_access_count;
                    break;
                }
                handled[0] = true;
                Function *callee = nullptr;
                if (auto *value = inst->operand(0u); value != nullptr && value->isa<Function>()) {
                    callee = static_cast<Function *>(value);
                } else {
                    ++info.invalid_access_count;
                }
                luisa::vector<Argument *> arguments;
                if (callee != nullptr) {
                    for (auto *argument : callee->arguments()) { arguments.emplace_back(argument); }
                    if (arguments.size() + 1u != inst->operand_count()) { ++info.invalid_access_count; }
                }
                for (size_t i = 1u; i < inst->operand_count(); ++i) {
                    auto *argument_value = inst->operand(i);
                    if (!is_tracked_pointer(argument_value)) { continue; }
                    handled[i] = true;
                    auto unknown = callee == nullptr || i - 1u >= arguments.size();
                    auto by_reference = !unknown && arguments[i - 1u]->is_reference();
                    // XIR pointers are not first-class values: a tracked
                    // pointer may only be passed to a reference formal. If a
                    // malformed call passes it to a value/resource formal,
                    // reject the analysis result and still model the escape as
                    // an opaque read/write. Returning a successful read-only
                    // result here would let a consumer make an unsound
                    // liveness or dead-store decision on invalid IR.
                    if (!unknown && !by_reference) {
                        ++info.invalid_access_count;
                    }
                    record_emit(emit_access(
                        block, argument_value, {}, true, true, true));
                }
                break;
            }
            default: break;
        }
        for (size_t i = 0u; i < inst->operand_count(); ++i) {
            if (handled[i]) { continue; }
            auto *operand = inst->operand(i);
            if (is_tracked_pointer(operand)) {
                record_emit(emit_access(block, operand, {}, true, true, true));
            }
        }
    }

    void collect_events() noexcept {
        for (auto *pointer : tracked_pointers) {
            auto *path = resolve_path(pointer);
            if (path == nullptr || !path->connected) {
                ++info.invalid_access_count;
            } else if (pointer->isa<GEPInst>()) {
                auto *gep = static_cast<GEPInst *>(pointer);
                auto *base = gep->operand_count() == 0u ? nullptr : gep->base();
                if (base == nullptr || base->type() == nullptr) {
                    ++info.invalid_access_count;
                    continue;
                }
                luisa::vector<Value *> indices;
                for (size_t i = 0u; i < gep->index_count(); ++i) { indices.emplace_back(gep->index(i)); }
                if (!validate_projection(base->type(), luisa::span{indices}, gep->type())) {
                    ++info.invalid_access_count;
                }
            }
        }
        for (auto *block : ordered_blocks) {
            if (!reachable_blocks.contains(block)) { continue; }
            for (auto *inst : block->instructions()) {
                collect_instruction_events(block, inst);
            }
        }
    }

    [[nodiscard]] PointerUsageMap make_state(bool kill_top = false) const noexcept {
        PointerUsageMap state;
        for (auto *pointer : result_pointers) {
            auto usage = luisa::make_unique<PointerUsage>(pointer->type());
            if (kill_top) { usage->kill.set(true); }
            state.emplace(pointer, std::move(usage));
        }
        return state;
    }

    void clear_forward_state(PointerUsageMap &state) const noexcept {
        for (auto *pointer : result_pointers) {
            auto &usage = *state.at(pointer);
            usage.kill.set(false);
            usage.touch.set(false);
        }
    }

    void copy_forward_state(
        PointerUsageMap &target,
        const PointerUsageMap &source) const noexcept {
        for (auto *pointer : result_pointers) {
            auto &dst = *target.at(pointer);
            auto &src = *source.at(pointer);
            dst.kill = src.kill;
            dst.touch = src.touch;
        }
    }

    void clear_live_state(PointerUsageMap &state) const noexcept {
        for (auto *pointer : result_pointers) {
            state.at(pointer)->live.set(false);
        }
    }

    void copy_live_state(
        PointerUsageMap &target,
        const PointerUsageMap &source) const noexcept {
        for (auto *pointer : result_pointers) {
            target.at(pointer)->live = source.at(pointer)->live;
        }
    }

    [[nodiscard]] static bool same_forward_state(const PointerUsageMap &a, const PointerUsageMap &b) noexcept {
        if (a.size() != b.size()) { return false; }
        for (auto &&[pointer, usage] : a) {
            auto iter = b.find(pointer);
            if (iter == b.end() || usage->kill != iter->second->kill || usage->touch != iter->second->touch) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] static bool same_live_state(const PointerUsageMap &a, const PointerUsageMap &b) noexcept {
        if (a.size() != b.size()) { return false; }
        for (auto &&[pointer, usage] : a) {
            auto iter = b.find(pointer);
            if (iter == b.end() || usage->live != iter->second->live) { return false; }
        }
        return true;
    }

    void initialize_results() noexcept {
        for (auto *block : ordered_blocks) {
            if (!reachable_blocks.contains(block)) { continue; }
            auto &result = block_results[block];
            auto top = block != def->body_block();
            result.in = make_state(top);
            result.out = make_state(top);
        }
        info.analyzed_block_count = block_results.size();
    }

    void run_forward() noexcept {
        // Reuse two type-shaped scratch states for every transfer. The
        // product-lattice equations are unchanged; only their storage is
        // separated from the persistent per-block solution. Constructing a
        // hash map and one heap-owned PointerUsage for every coordinate at
        // every block evaluation made projected analyses allocation-bound.
        auto new_in = make_state();
        auto new_out = make_state();
        luisa::vector<BasicBlock *> worklist;
        luisa::vector<uint8_t> queued(ordered_blocks.size(), 0u);
        for (auto *block : ordered_blocks) {
            if (!reachable_blocks.contains(block)) { continue; }
            worklist.emplace_back(block);
            queued[block_indices.at(block)] = 1u;
        }
        for (size_t cursor = 0u; cursor < worklist.size(); ++cursor) {
            auto *block = worklist[cursor];
            queued[block_indices.at(block)] = 0u;
            ++forward_block_evaluation_count;
            clear_forward_state(new_in);
            if (block != def->body_block()) {
                auto pred_iter = predecessors.find(block);
                if (pred_iter != predecessors.end() &&
                    !pred_iter->second.empty()) {
                    copy_forward_state(
                        new_in,
                        block_results.at(
                                         pred_iter->second.front())
                            .out);
                    for (size_t i = 1u;
                         i < pred_iter->second.size(); ++i) {
                        auto &pred_out =
                            block_results.at(pred_iter->second[i]).out;
                        for (auto *pointer : result_pointers) {
                            new_in.at(pointer)->kill &=
                                pred_out.at(pointer)->kill;
                            new_in.at(pointer)->touch |=
                                pred_out.at(pointer)->touch;
                        }
                    }
                }
            }
            copy_forward_state(new_out, new_in);
            if (auto event_iter = events.find(block);
                event_iter != events.end()) {
                for (auto &event : event_iter->second) {
                    auto &usage = *new_out.at(event.pointer);
                    if (event.write) {
                        usage.touch |= event.mask;
                        if (event.definite_write) {
                            usage.kill |= event.mask;
                        }
                    }
                }
            }
            auto &result = block_results.at(block);
            if (!same_forward_state(result.in, new_in) ||
                !same_forward_state(result.out, new_out)) {
                copy_forward_state(result.in, new_in);
                copy_forward_state(result.out, new_out);
                if (auto iter = successors.find(block);
                    iter != successors.end()) {
                    for (auto *successor : iter->second) {
                        auto index = block_indices.at(successor);
                        if (queued[index] == 0u) {
                            queued[index] = 1u;
                            worklist.emplace_back(successor);
                        }
                    }
                }
            }
        }
    }

    void run_backward() noexcept {
        auto new_out = make_state();
        auto new_in = make_state();
        luisa::vector<BasicBlock *> worklist;
        luisa::vector<uint8_t> queued(ordered_blocks.size(), 0u);
        for (auto iter = ordered_blocks.rbegin();
             iter != ordered_blocks.rend(); ++iter) {
            auto *block = *iter;
            if (!reachable_blocks.contains(block)) { continue; }
            worklist.emplace_back(block);
            queued[block_indices.at(block)] = 1u;
        }
        for (size_t cursor = 0u; cursor < worklist.size(); ++cursor) {
            auto *block = worklist[cursor];
            queued[block_indices.at(block)] = 0u;
            ++backward_block_evaluation_count;
            clear_live_state(new_out);
            if (auto succ_iter = successors.find(block);
                succ_iter != successors.end()) {
                for (auto *successor : succ_iter->second) {
                    auto &successor_in = block_results.at(successor).in;
                    for (auto *pointer : result_pointers) {
                        new_out.at(pointer)->live |=
                            successor_in.at(pointer)->live;
                    }
                }
            }
            copy_live_state(new_in, new_out);
            if (auto event_iter = events.find(block);
                event_iter != events.end()) {
                for (auto iter = event_iter->second.rbegin();
                     iter != event_iter->second.rend(); ++iter) {
                    auto &usage = *new_in.at(iter->pointer);
                    if (iter->write && iter->definite_write) {
                        usage.live &= ~iter->mask;
                    }
                    if (iter->read) { usage.live |= iter->mask; }
                }
            }
            auto &result = block_results.at(block);
            if (!same_live_state(result.in, new_in) ||
                !same_live_state(result.out, new_out)) {
                copy_live_state(result.in, new_in);
                copy_live_state(result.out, new_out);
                if (auto iter = predecessors.find(block);
                    iter != predecessors.end()) {
                    for (auto *predecessor : iter->second) {
                        auto index = block_indices.at(predecessor);
                        if (queued[index] == 0u) {
                            queued[index] = 1u;
                            worklist.emplace_back(predecessor);
                        }
                    }
                }
            }
        }
    }

    [[nodiscard]] PointerUsageAnalysisInfo run(
        FunctionDefinition *function,
        luisa::optional<luisa::span<Value *const>> requested =
            luisa::nullopt) noexcept {
        clear();
        if (function != nullptr && function->body_block() == nullptr &&
            function->derived_function_tag() ==
                DerivedFunctionTag::CALLABLE) {
            // Callable declarations own no CFG. Retain an empty, current
            // snapshot so queries are well-defined and distinguish them from
            // malformed bodyless kernels.
            def = function;
            capture_snapshot();
            return info;
        }
        if (!initialize_cfg(function)) {
            info.invalid_function_count = 1u;
            return info;
        }
        using ProfileClock = std::chrono::steady_clock;
        const auto profile_enabled = []() noexcept {
            if (auto profile = std::getenv(
                    "LUISA_XIR_PROFILE_POINTER_USAGE")) {
                return luisa::string_view{profile} == "1";
            }
            return false;
        }();
        auto phase_begin = profile_enabled ?
                               ProfileClock::now() :
                               ProfileClock::time_point{};
        auto collect_pointers_ms = 0.0;
        auto select_results_ms = 0.0;
        auto collect_events_ms = 0.0;
        auto initialize_results_ms = 0.0;
        auto forward_ms = 0.0;
        auto backward_ms = 0.0;
        auto snapshot_ms = 0.0;
        const auto finish_phase =
            [&phase_begin, profile_enabled]() noexcept {
                if (!profile_enabled) { return 0.0; }
                auto now = ProfileClock::now();
                auto elapsed =
                    std::chrono::duration<double, std::milli>{
                        now - phase_begin}
                        .count();
                phase_begin = now;
                return elapsed;
            };
        collect_pointers();
        collect_pointers_ms = finish_phase();
        select_result_pointers(requested);
        index_result_pointers_by_root();
        select_results_ms = finish_phase();
        collect_events();
        collect_events_ms = finish_phase();
        initialize_results();
        initialize_results_ms = finish_phase();
        run_forward();
        forward_ms = finish_phase();
        run_backward();
        backward_ms = finish_phase();
        capture_snapshot();
        snapshot_ms = finish_phase();
        if (profile_enabled) {
            LUISA_INFO(
                "Pointer-usage timing: function='{}' pointers={}/{} "
                "blocks={} collect_pointers={:.3f} ms "
                "select_results={:.3f} ms collect_events={:.3f} ms "
                "initialize_results={:.3f} ms forward={:.3f} ms "
                "({} block evaluations) backward={:.3f} ms "
                "({} block evaluations) snapshot={:.3f} ms.",
                def->name().value_or("<unnamed>"),
                info.tracked_pointer_count,
                info.materialized_pointer_count,
                info.analyzed_block_count,
                collect_pointers_ms, select_results_ms,
                collect_events_ms, initialize_results_ms,
                forward_ms, forward_block_evaluation_count,
                backward_ms,
                backward_block_evaluation_count, snapshot_ms);
        }
        return info;
    }
};

PointerUsageAnalysis::PointerUsageAnalysis() noexcept : _impl{luisa::make_unique<Impl>()} {}
PointerUsageAnalysis::~PointerUsageAnalysis() noexcept = default;
PointerUsageAnalysis::PointerUsageAnalysis(PointerUsageAnalysis &&) noexcept = default;
PointerUsageAnalysis &PointerUsageAnalysis::operator=(PointerUsageAnalysis &&) noexcept = default;

void PointerUsageAnalysis::clear() noexcept {
    if (_impl != nullptr) { _impl->clear(); }
}

PointerUsageAnalysisInfo PointerUsageAnalysis::analyze(FunctionDefinition *function) noexcept {
    if (_impl == nullptr) { _impl = luisa::make_unique<Impl>(); }
    return _impl->run(function);
}

PointerUsageAnalysisInfo PointerUsageAnalysis::analyze(
    FunctionDefinition *function,
    luisa::span<Value *const> result_pointers) noexcept {
    if (_impl == nullptr) { _impl = luisa::make_unique<Impl>(); }
    return _impl->run(function, result_pointers);
}

FunctionDefinition *PointerUsageAnalysis::function() const noexcept {
    return _impl == nullptr || _impl->lifetime_token.expired() ? nullptr : _impl->def;
}

bool PointerUsageAnalysis::is_current() const noexcept {
    return _impl != nullptr && _impl->is_current();
}

const BasicBlockPointerUsage *PointerUsageAnalysis::block_usage(BasicBlock *block) const noexcept {
    if (_impl == nullptr || block == nullptr || !_impl->is_current()) { return nullptr; }
    return current_block_usage(block);
}

const BasicBlockPointerUsage *PointerUsageAnalysis::current_block_usage(
    BasicBlock *block) const noexcept {
    if (_impl == nullptr || block == nullptr) { return nullptr; }
    auto iter = _impl->block_results.find(block);
    return iter == _impl->block_results.end() ? nullptr : &iter->second;
}

const PointerUsage *PointerUsageAnalysis::in_usage(BasicBlock *block, Value *pointer) const noexcept {
    auto *usage = block_usage(block);
    if (usage == nullptr) { return nullptr; }
    auto iter = usage->in.find(pointer);
    return iter == usage->in.end() ? nullptr : iter->second.get();
}

const PointerUsage *PointerUsageAnalysis::out_usage(BasicBlock *block, Value *pointer) const noexcept {
    auto *usage = block_usage(block);
    if (usage == nullptr) { return nullptr; }
    auto iter = usage->out.find(pointer);
    return iter == usage->out.end() ? nullptr : iter->second.get();
}

PointerUsageAnalysisInfo pointer_usage_pass_run_on_function(
    FunctionDefinition *function, PointerUsageAnalysis *analysis) noexcept {
    PointerUsageAnalysis local;
    return (analysis == nullptr ? local : *analysis).analyze(function);
}

PointerUsageAnalysisInfo pointer_usage_pass_run_on_module(Module *module, PassReport *report) noexcept {
    PointerUsageAnalysisInfo info;
    if (module == nullptr) {
        info.invalid_function_count = 1u;
    } else {
        for (auto *function : module->function_list()) {
            if (auto *def = function->definition()) {
                auto function_info = pointer_usage_pass_run_on_function(def);
                info.tracked_pointer_count += function_info.tracked_pointer_count;
                info.materialized_pointer_count += function_info.materialized_pointer_count;
                info.analyzed_block_count += function_info.analyzed_block_count;
                info.conservative_access_count += function_info.conservative_access_count;
                info.invalid_access_count += function_info.invalid_access_count;
                info.invalid_function_count += function_info.invalid_function_count;
            }
        }
    }
    if (report != nullptr) {
        report->set("tracked_pointer", info.tracked_pointer_count);
        report->set("materialized_pointer", info.materialized_pointer_count);
        report->set("analyzed_block", info.analyzed_block_count);
        report->set("conservative_access", info.conservative_access_count);
        report->set("invalid_access", info.invalid_access_count);
        report->set("invalid_function", info.invalid_function_count);
    }
    return info;
}

}// namespace luisa::compute::xir
