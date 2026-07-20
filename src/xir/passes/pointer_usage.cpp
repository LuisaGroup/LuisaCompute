#include <luisa/ast/type.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/optional.h>
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
    luisa::unordered_set<BasicBlock *> owned_blocks;
    luisa::unordered_set<BasicBlock *> reachable_blocks;
    luisa::unordered_map<BasicBlock *, luisa::vector<BasicBlock *>> successors;
    luisa::unordered_map<BasicBlock *, luisa::vector<BasicBlock *>> predecessors;
    luisa::vector<Value *> tracked_pointers;
    luisa::unordered_set<Value *> tracked_pointer_set;
    luisa::unordered_map<Value *, PointerPath> paths;
    luisa::unordered_set<Value *> resolving_paths;
    luisa::unordered_map<BasicBlock *, luisa::vector<AccessEvent>> events;
    luisa::unordered_map<BasicBlock *, BasicBlockPointerUsage> block_results;

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
        owned_blocks.clear();
        reachable_blocks.clear();
        successors.clear();
        predecessors.clear();
        tracked_pointers.clear();
        tracked_pointer_set.clear();
        paths.clear();
        resolving_paths.clear();
        events.clear();
        block_results.clear();
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
        auto *constant = static_cast<Constant *>(value);
        auto *type = constant->type();
        if (type->is_int8()) {
            auto v = constant->as<int8_t>();
            if (v < 0) { result.valid = false; }
            result.value = static_cast<size_t>(v);
        } else if (type->is_uint8()) {
            result.value = constant->as<uint8_t>();
        } else if (type->is_int16()) {
            auto v = constant->as<int16_t>();
            if (v < 0) { result.valid = false; }
            result.value = static_cast<size_t>(v);
        } else if (type->is_uint16()) {
            result.value = constant->as<uint16_t>();
        } else if (type->is_int32()) {
            auto v = constant->as<int32_t>();
            if (v < 0) { result.valid = false; }
            result.value = static_cast<size_t>(v);
        } else if (type->is_uint32()) {
            result.value = constant->as<uint32_t>();
        } else if (type->is_int64()) {
            auto v = constant->as<int64_t>();
            if (v < 0) { result.valid = false; }
            result.value = static_cast<size_t>(v);
        } else if (type->is_uint64()) {
            auto v = constant->as<uint64_t>();
            if (v > static_cast<uint64_t>(SIZE_MAX)) { result.valid = false; }
            result.value = static_cast<size_t>(v);
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
        for (auto *target : tracked_pointers) {
            auto *target_path = resolve_path(target);
            if (target_path == nullptr || !target_path->connected || target_path->pointers.empty() ||
                target_path->pointers.front() != root) {
                continue;
            }
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
                    record_emit(emit_access(block, argument_value, {}, true, unknown || by_reference, unknown || by_reference));
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
        for (auto *pointer : tracked_pointers) {
            auto usage = luisa::make_unique<PointerUsage>(pointer->type());
            if (kill_top) { usage->kill.set(true); }
            state.emplace(pointer, std::move(usage));
        }
        return state;
    }

    [[nodiscard]] PointerUsageMap copy_state(const PointerUsageMap &source) const noexcept {
        PointerUsageMap copy;
        for (auto *pointer : tracked_pointers) {
            auto usage = luisa::make_unique<PointerUsage>(pointer->type());
            if (auto iter = source.find(pointer); iter != source.end()) { *usage = *iter->second; }
            copy.emplace(pointer, std::move(usage));
        }
        return copy;
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
        bool changed;
        do {
            changed = false;
            for (auto *block : ordered_blocks) {
                if (!reachable_blocks.contains(block)) { continue; }
                auto new_in = make_state();
                if (block != def->body_block()) {
                    auto pred_iter = predecessors.find(block);
                    if (pred_iter != predecessors.end() && !pred_iter->second.empty()) {
                        new_in = copy_state(block_results.at(pred_iter->second.front()).out);
                        for (size_t i = 1u; i < pred_iter->second.size(); ++i) {
                            auto &pred_out = block_results.at(pred_iter->second[i]).out;
                            for (auto *pointer : tracked_pointers) {
                                new_in.at(pointer)->kill &= pred_out.at(pointer)->kill;
                                new_in.at(pointer)->touch |= pred_out.at(pointer)->touch;
                            }
                        }
                    }
                }
                auto new_out = copy_state(new_in);
                if (auto event_iter = events.find(block); event_iter != events.end()) {
                    for (auto &event : event_iter->second) {
                        auto &usage = *new_out.at(event.pointer);
                        if (event.write) {
                            usage.touch |= event.mask;
                            if (event.definite_write) { usage.kill |= event.mask; }
                        }
                    }
                }
                auto &result = block_results.at(block);
                if (!same_forward_state(result.in, new_in) || !same_forward_state(result.out, new_out)) {
                    result.in = std::move(new_in);
                    result.out = std::move(new_out);
                    changed = true;
                }
            }
        } while (changed);
    }

    void run_backward() noexcept {
        bool changed;
        do {
            changed = false;
            for (auto block_iter = ordered_blocks.rbegin(); block_iter != ordered_blocks.rend(); ++block_iter) {
                auto *block = *block_iter;
                if (!reachable_blocks.contains(block)) { continue; }
                auto new_out = make_state();
                if (auto succ_iter = successors.find(block); succ_iter != successors.end()) {
                    for (auto *successor : succ_iter->second) {
                        auto &successor_in = block_results.at(successor).in;
                        for (auto *pointer : tracked_pointers) {
                            new_out.at(pointer)->live |= successor_in.at(pointer)->live;
                        }
                    }
                }
                auto new_in = copy_state(new_out);
                if (auto event_iter = events.find(block); event_iter != events.end()) {
                    for (auto iter = event_iter->second.rbegin(); iter != event_iter->second.rend(); ++iter) {
                        auto &usage = *new_in.at(iter->pointer);
                        if (iter->write && iter->definite_write) { usage.live &= ~iter->mask; }
                        if (iter->read) { usage.live |= iter->mask; }
                    }
                }
                auto &result = block_results.at(block);
                if (!same_live_state(result.in, new_in) || !same_live_state(result.out, new_out)) {
                    for (auto *pointer : tracked_pointers) {
                        result.in.at(pointer)->live = new_in.at(pointer)->live;
                        result.out.at(pointer)->live = new_out.at(pointer)->live;
                    }
                    changed = true;
                }
            }
        } while (changed);
    }

    [[nodiscard]] PointerUsageAnalysisInfo run(FunctionDefinition *function) noexcept {
        clear();
        if (!initialize_cfg(function)) {
            info.invalid_function_count = 1u;
            return info;
        }
        collect_pointers();
        collect_events();
        initialize_results();
        run_forward();
        run_backward();
        capture_snapshot();
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

FunctionDefinition *PointerUsageAnalysis::function() const noexcept {
    return _impl == nullptr || _impl->lifetime_token.expired() ? nullptr : _impl->def;
}

bool PointerUsageAnalysis::is_current() const noexcept {
    return _impl != nullptr && _impl->is_current();
}

const BasicBlockPointerUsage *PointerUsageAnalysis::block_usage(BasicBlock *block) const noexcept {
    if (_impl == nullptr || block == nullptr || !_impl->is_current()) { return nullptr; }
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
                info.analyzed_block_count += function_info.analyzed_block_count;
                info.conservative_access_count += function_info.conservative_access_count;
                info.invalid_access_count += function_info.invalid_access_count;
                info.invalid_function_count += function_info.invalid_function_count;
            }
        }
    }
    if (report != nullptr) {
        report->set("tracked_pointer", info.tracked_pointer_count);
        report->set("analyzed_block", info.analyzed_block_count);
        report->set("conservative_access", info.conservative_access_count);
        report->set("invalid_access", info.invalid_access_count);
        report->set("invalid_function", info.invalid_function_count);
    }
    return info;
}

}// namespace luisa::compute::xir
