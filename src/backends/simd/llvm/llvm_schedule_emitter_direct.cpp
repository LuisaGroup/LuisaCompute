#include "llvm_schedule_emitter.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <unordered_set>

namespace luisa::compute::simd::detail {

bool ScheduleEmitter::_can_emit_direct_control_flow() const noexcept {
    std::vector<bool> covered_convergences(
        _source.convergence_points().size(), false);
    for (auto &&block : _source.blocks()) {
        auto supported = std::visit(
            [&](const auto &control) noexcept {
                using T = std::decay_t<decltype(control)>;
                if constexpr (std::is_same_v<
                                  T, schedule::SplitTerminator>) {
                    auto *condition = _source.value(control.condition);
                    if (condition != nullptr &&
                        schedule::is_uniform(condition->value_class)) {
                        return true;
                    }
                    auto diamond =
                        _find_predicated_memory_diamond(block);
                    if (diamond && control.convergence) {
                        covered_convergences[control.convergence->value] =
                            true;
                    }
                    return diamond.has_value();
                } else if constexpr (std::is_same_v<
                                         T,
                                         schedule::SwitchTerminator>) {
                    auto *selector = _source.value(control.selector);
                    return selector != nullptr &&
                           schedule::is_uniform(
                               selector->value_class);
                } else {
                    return std::is_same_v<
                               T, schedule::BranchTerminator> ||
                           std::is_same_v<
                               T, schedule::ReturnTerminator> ||
                           std::is_same_v<
                               T, schedule::UnreachableTerminator>;
                }
            },
            block.terminator);
        if (!supported) { return false; }
    }
    return std::all_of(
        covered_convergences.begin(), covered_convergences.end(),
        [](bool covered) noexcept { return covered; });
}

[[nodiscard]] std::optional<std::vector<schedule::BlockId>>
ScheduleEmitter::_find_predicated_acyclic_order() const noexcept {
    static constexpr auto max_block_count = size_t{16u};
    static constexpr auto max_instruction_count = size_t{32u};
    if (!_is_surface_filter_handler_entry() ||
        (_width != 4u && _width != 8u && _width != 16u) ||
        _source.blocks().empty() ||
        _source.blocks().size() > max_block_count ||
        !_source.loops().empty()) {
        return std::nullopt;
    }
    auto instruction_count = size_t{0u};
    for (auto &&block : _source.blocks()) {
        instruction_count += block.instructions.size();
    }
    if (instruction_count > max_instruction_count) {
        return std::nullopt;
    }

    auto block_count = _source.blocks().size();
    if (_source.entry().value >= block_count) {
        return std::nullopt;
    }
    std::vector<std::vector<schedule::BlockId>> successors(block_count);
    std::vector<size_t> indegrees(block_count, 0u);
    std::vector<uint8_t> seen_blocks(block_count, uint8_t{0u});
    auto add_edge = [&](schedule::BlockId source,
                        const schedule::ControlEdge &edge) noexcept {
        if (source.value >= block_count ||
            edge.target.value >= block_count || edge.loop_back) {
            return false;
        }
        successors[source.value].emplace_back(edge.target);
        indegrees[edge.target.value]++;
        return true;
    };
    for (auto &&block : _source.blocks()) {
        if (block.id.value >= block_count ||
            seen_blocks[block.id.value] != 0u) {
            return std::nullopt;
        }
        seen_blocks[block.id.value] = 1u;
        auto supported = std::visit(
            [&](const auto &terminator) noexcept {
                using T = std::decay_t<decltype(terminator)>;
                if constexpr (std::is_same_v<
                                  T, schedule::BranchTerminator>) {
                    return add_edge(block.id, terminator.edge);
                } else if constexpr (std::is_same_v<
                                         T, schedule::SplitTerminator>) {
                    return add_edge(block.id, terminator.true_edge) &&
                           add_edge(block.id, terminator.false_edge);
                } else if constexpr (std::is_same_v<
                                         T, schedule::SwitchTerminator>) {
                    for (auto &&item : terminator.cases) {
                        if (!add_edge(block.id, item.edge)) {
                            return false;
                        }
                    }
                    return add_edge(block.id, terminator.default_edge);
                } else if constexpr (std::is_same_v<
                                         T, schedule::JoinTerminator>) {
                    auto *point = _source.convergence(
                        terminator.convergence);
                    if (point == nullptr) { return false; }
                    schedule::ControlEdge edge{point->target};
                    return add_edge(block.id, edge);
                } else if constexpr (std::is_same_v<
                                         T, schedule::ReturnTerminator>) {
                    return !terminator.value.has_value();
                } else {
                    return false;
                }
            },
            block.terminator);
        if (!supported) { return std::nullopt; }
    }

    // Only the entry may start without a predecessor. Kahn's algorithm below
    // then simultaneously proves that every block is reachable and that the
    // handler has no cycle hidden outside Schedule's natural-loop table.
    if (indegrees[_source.entry().value] != 0u) {
        return std::nullopt;
    }
    for (auto index = size_t{0u}; index < block_count; index++) {
        if (indegrees[index] == 0u &&
            index != _source.entry().value) {
            return std::nullopt;
        }
    }
    std::vector<schedule::BlockId> ready;
    ready.reserve(block_count);
    ready.emplace_back(_source.entry());
    std::vector<schedule::BlockId> order;
    order.reserve(block_count);
    for (auto cursor = size_t{0u}; cursor < ready.size(); cursor++) {
        auto block = ready[cursor];
        order.emplace_back(block);
        for (auto target : successors[block.value]) {
            if (--indegrees[target.value] == 0u) {
                ready.emplace_back(target);
            }
        }
    }
    if (order.size() != block_count) { return std::nullopt; }
    return order;
}

[[nodiscard]] std::optional<
    ScheduleEmitter::InterleavedScalarBufferReadGroup>
ScheduleEmitter::_find_interleaved_scalar_buffer_read_group(
    const schedule::BasicBlock &block,
    size_t begin_instruction) const noexcept {
    static constexpr auto max_region_instruction_count = size_t{24u};
    if (!_enable_interleaved_scalar_buffer_reads ||
        _entry_abi != ScheduleEntryABI::packet ||
        (_width != 8u && _width != 16u) ||
        _static_block_size[0u] < _width ||
        _static_block_size[0u] % _width != 0u ||
        begin_instruction >= block.instructions.size()) {
        return std::nullopt;
    }

    auto defining_instruction = [&](schedule::ValueId id) noexcept
        -> const schedule::Instruction * {
        auto *value = _source.value(id);
        if (value == nullptr || !value->defining_block) {
            return nullptr;
        }
        auto *defining_block = _source.block(*value->defining_block);
        if (defining_block == nullptr) { return nullptr; }
        for (auto &&instruction : defining_block->instructions) {
            if (instruction.result && *instruction.result == id) {
                return &instruction;
            }
        }
        return nullptr;
    };
    auto constant_nonnegative_integer =
        [&](schedule::ValueId id) noexcept -> std::optional<uint64_t> {
        auto *value = _source.value(id);
        if (value == nullptr ||
            value->origin != schedule::ValueOrigin::constant ||
            value->type == nullptr || !value->type->is_scalar() ||
            (!value->type->is_int() && !value->type->is_uint()) ||
            value->type->size() == 0u ||
            value->type->size() > sizeof(uint64_t)) {
            return std::nullopt;
        }
        auto *metadata = std::get_if<schedule::ConstantValueMetadata>(
            &value->metadata);
        if (metadata == nullptr ||
            metadata->bytes.size() < value->type->size()) {
            return std::nullopt;
        }
        auto decoded = uint64_t{0u};
        std::memcpy(&decoded, metadata->bytes.data(),
                    value->type->size());
        if (value->type->is_int()) {
            auto sign_bit = uint64_t{1u}
                            << (value->type->size() * 8u - 1u);
            if ((decoded & sign_bit) != 0u) {
                return std::nullopt;
            }
        }
        return decoded;
    };
    std::unordered_set<uint32_t> visiting;
    std::function<std::optional<int64_t>(schedule::ValueId)>
        lane_stride = [&](schedule::ValueId id)
        -> std::optional<int64_t> {
        auto *value = _source.value(id);
        if (value == nullptr || value->type == nullptr ||
            !value->type->is_scalar() ||
            (!value->type->is_int() && !value->type->is_uint())) {
            return std::nullopt;
        }
        if (schedule::is_uniform(value->value_class)) {
            return int64_t{0};
        }
        if (value->origin == schedule::ValueOrigin::special_register) {
            auto *metadata =
                std::get_if<schedule::SpecialRegisterValueMetadata>(
                    &value->metadata);
            if (metadata != nullptr &&
                static_cast<xir::DerivedSpecialRegisterTag>(
                    metadata->tag) ==
                    xir::DerivedSpecialRegisterTag::WARP_LANE_ID) {
                return int64_t{1};
            }
            return std::nullopt;
        }
        if (!visiting.emplace(id.value).second) {
            return std::nullopt;
        }
        auto leave = [&]() noexcept { visiting.erase(id.value); };
        auto *instruction = defining_instruction(id);
        if (instruction == nullptr ||
            instruction->opcode != schedule::Opcode::arithmetic ||
            !instruction->source_op) {
            leave();
            return std::nullopt;
        }
        auto op = static_cast<xir::ArithmeticOp>(
            *instruction->source_op);
        if (op == xir::ArithmeticOp::EXTRACT &&
            instruction->operands.size() == 2u &&
            constant_nonnegative_integer(instruction->operands[1u]) ==
                std::optional<uint64_t>{0u}) {
            auto *aggregate = _source.value(instruction->operands[0u]);
            auto *metadata = aggregate == nullptr ? nullptr :
                                                    std::get_if<
                                                        schedule::SpecialRegisterValueMetadata>(
                                                        &aggregate->metadata);
            if (metadata != nullptr) {
                auto tag = static_cast<xir::DerivedSpecialRegisterTag>(
                    metadata->tag);
                if (tag == xir::DerivedSpecialRegisterTag::DISPATCH_ID ||
                    tag == xir::DerivedSpecialRegisterTag::THREAD_ID) {
                    leave();
                    return int64_t{1};
                }
            }
            leave();
            return std::nullopt;
        }
        if (instruction->operands.size() != 2u) {
            leave();
            return std::nullopt;
        }
        auto lhs = lane_stride(instruction->operands[0u]);
        auto rhs = lane_stride(instruction->operands[1u]);
        std::optional<int64_t> result;
        auto bounded = [](std::optional<int64_t> value) noexcept {
            return value && *value >= 0 && *value <= 4;
        };
        if (op == xir::ArithmeticOp::BINARY_ADD &&
            bounded(lhs) && bounded(rhs) && *lhs + *rhs <= 4) {
            result = *lhs + *rhs;
        } else if (op == xir::ArithmeticOp::BINARY_SUB &&
                   bounded(lhs) && bounded(rhs) && *lhs >= *rhs) {
            result = *lhs - *rhs;
        } else if (op == xir::ArithmeticOp::BINARY_MUL) {
            if (auto constant = constant_nonnegative_integer(
                    instruction->operands[0u]);
                constant && *constant <= 4u && bounded(rhs) &&
                *constant * static_cast<uint64_t>(*rhs) <= 4u) {
                result = static_cast<int64_t>(*constant) * *rhs;
            } else if (auto constant = constant_nonnegative_integer(
                           instruction->operands[1u]);
                       constant && *constant <= 4u && bounded(lhs) &&
                       *constant * static_cast<uint64_t>(*lhs) <= 4u) {
                result = *lhs * static_cast<int64_t>(*constant);
            }
        }
        leave();
        return result;
    };
    auto affine_root = [&](schedule::ValueId id) noexcept {
        auto root = id;
        auto offset = uint64_t{0u};
        for (auto depth = uint32_t{0u}; depth < 8u; depth++) {
            auto *instruction = defining_instruction(root);
            if (instruction == nullptr ||
                instruction->opcode != schedule::Opcode::arithmetic ||
                instruction->source_op != static_cast<uint32_t>(
                                              xir::ArithmeticOp::BINARY_ADD) ||
                instruction->operands.size() != 2u) {
                break;
            }
            auto lhs_constant = constant_nonnegative_integer(
                instruction->operands[0u]);
            auto rhs_constant = constant_nonnegative_integer(
                instruction->operands[1u]);
            schedule::ValueId next{};
            std::optional<uint64_t> increment;
            if (lhs_constant && !rhs_constant) {
                next = instruction->operands[1u];
                increment = lhs_constant;
            } else if (!lhs_constant && rhs_constant) {
                next = instruction->operands[0u];
                increment = rhs_constant;
            } else {
                break;
            }
            if (*increment >
                std::numeric_limits<uint64_t>::max() - offset) {
                return std::optional<std::pair<schedule::ValueId, uint64_t>>{};
            }
            root = next;
            offset += *increment;
        }
        return std::optional{
            std::pair{root, offset}};
    };
    auto is_parameter_buffer = [&](schedule::ValueId id) noexcept {
        auto *value = _source.value(id);
        if (value == nullptr ||
            value->origin != schedule::ValueOrigin::parameter ||
            value->type == nullptr || !value->type->is_buffer()) {
            return false;
        }
        auto *metadata = std::get_if<schedule::ParameterValueMetadata>(
            &value->metadata);
        return metadata != nullptr &&
               static_cast<xir::DerivedArgumentTag>(
                   metadata->argument_tag) ==
                   xir::DerivedArgumentTag::RESOURCE;
    };
    auto parse_group_read =
        [&](const schedule::Instruction &instruction,
            schedule::ValueId expected_buffer,
            schedule::ValueId expected_root,
            uint64_t expected_offset,
            const Type *expected_type) noexcept {
            if (instruction.opcode != schedule::Opcode::resource_read ||
                instruction.source_op != static_cast<uint32_t>(
                                             xir::ResourceReadOp::BUFFER_READ) ||
                !instruction.result || instruction.operands.size() != 2u ||
                instruction.operands[0u] != expected_buffer) {
                return false;
            }
            auto *result = _source.value(*instruction.result);
            auto normalized = affine_root(instruction.operands[1u]);
            return result != nullptr && result->type == expected_type &&
                   normalized && normalized->first == expected_root &&
                   normalized->second == expected_offset;
        };

    auto &&first = block.instructions[begin_instruction];
    if (first.opcode != schedule::Opcode::resource_read ||
        first.source_op != static_cast<uint32_t>(
                               xir::ResourceReadOp::BUFFER_READ) ||
        !first.result || first.operands.size() != 2u ||
        !is_parameter_buffer(first.operands[0u])) {
        return std::nullopt;
    }
    auto *buffer = _source.value(first.operands[0u]);
    auto *result = _source.value(*first.result);
    auto *index = _source.value(first.operands[1u]);
    if (buffer == nullptr || result == nullptr || index == nullptr ||
        result->type == nullptr || !_is_scalar_data(result->type) ||
        result->type->is_bool() ||
        result->type->size() != sizeof(uint32_t) ||
        buffer->type->element() != result->type ||
        index->type == nullptr || !index->type->is_uint() ||
        index->type->size() != sizeof(uint32_t)) {
        return std::nullopt;
    }
    auto normalized = affine_root(first.operands[1u]);
    if (!normalized) { return std::nullopt; }
    visiting.clear();
    auto stride = lane_stride(normalized->first);
    if (!stride || *stride < 2 || *stride > 4) {
        return std::nullopt;
    }
    auto field_count = static_cast<uint32_t>(*stride);
    if (normalized->second >
        std::numeric_limits<uint32_t>::max() -
            (field_count - 1u)) {
        return std::nullopt;
    }
    InterleavedScalarBufferReadGroup group{
        .begin_instruction = begin_instruction,
        .end_instruction = begin_instruction,
        .field_count = field_count,
        .buffer = first.operands[0u],
        .first_index = first.operands[1u],
        .element_type = result->type,
        .results = {*first.result},
    };
    auto next_offset = normalized->second + 1u;
    auto scan_end = std::min(
        block.instructions.size(),
        begin_instruction + max_region_instruction_count);
    for (auto i = begin_instruction + 1u; i < scan_end; i++) {
        auto &&instruction = block.instructions[i];
        if (parse_group_read(
                instruction, group.buffer, normalized->first,
                next_offset, group.element_type)) {
            group.results.emplace_back(*instruction.result);
            group.end_instruction = i;
            next_offset++;
            if (group.results.size() == field_count) {
                return group;
            }
            continue;
        }
        switch (instruction.opcode) {
            case schedule::Opcode::arithmetic:
            case schedule::Opcode::cast: break;
            case schedule::Opcode::resource_read: {
                if (instruction.source_op != static_cast<uint32_t>(
                                                 xir::ResourceReadOp::BUFFER_READ)) {
                    return std::nullopt;
                }
                break;
            }
            case schedule::Opcode::resource_write: {
                if (instruction.source_op != static_cast<uint32_t>(
                                                 xir::ResourceWriteOp::BUFFER_WRITE) ||
                    instruction.operands.size() != 3u ||
                    !is_parameter_buffer(instruction.operands[0u])) {
                    return std::nullopt;
                }
                if (std::find(
                        group.crossed_write_buffers.begin(),
                        group.crossed_write_buffers.end(),
                        instruction.operands[0u]) ==
                    group.crossed_write_buffers.end()) {
                    group.crossed_write_buffers.emplace_back(
                        instruction.operands[0u]);
                }
                break;
            }
            default: return std::nullopt;
        }
    }
    return std::nullopt;
}

[[nodiscard]] ::llvm::Value *
ScheduleEmitter::_interleaved_scalar_buffer_read_guard(
    const InterleavedScalarBufferReadGroup &group) {
    auto *source = _load_value(group.buffer);
    auto *index_value = _source.value(group.first_index);
    auto *index = index_value == nullptr ? nullptr :
                                           _as_lane_vector(
                                               _load_value(group.first_index),
                                               *index_value);
    if (source == nullptr || index == nullptr ||
        !index->getType()->isVectorTy()) {
        _fail("interleaved scalar buffer read has invalid source operands");
        return nullptr;
    }
    auto *source_base = _builder.CreateExtractValue(source, {0u});
    auto *source_size = _builder.CreateExtractValue(source, {1u});
    auto *first_index = _builder.CreateZExtOrTrunc(
        _builder.CreateExtractElement(index, uint64_t{0u}),
        _builder.getInt64Ty());
    auto *byte_offset = _builder.CreateMul(
        first_index, _builder.getInt64(sizeof(uint32_t)),
        "buffer.interleaved.byte.offset");
    auto *offset_in_range = _builder.CreateICmpULE(
        byte_offset, source_size,
        "buffer.interleaved.offset.in.range");
    auto required_bytes = static_cast<uint64_t>(_width) *
                          group.field_count * sizeof(uint32_t);
    auto *bytes_available = _builder.CreateSub(
        source_size, byte_offset,
        "buffer.interleaved.bytes.available");
    auto *guard = _builder.CreateAnd(
        offset_in_range,
        _builder.CreateICmpULE(
            _builder.getInt64(required_bytes), bytes_available),
        "buffer.interleaved.in.bounds");

    auto *source_begin = _builder.CreatePtrToInt(
        source_base, _builder.getInt64Ty());
    auto *source_end = _builder.CreateAdd(source_begin, source_size);
    auto *source_range_valid = _builder.CreateICmpUGE(
        source_end, source_begin);
    for (auto write_buffer : group.crossed_write_buffers) {
        auto *target = _load_value(write_buffer);
        if (target == nullptr) {
            _fail("interleaved scalar buffer read has an invalid alias guard target");
            return nullptr;
        }
        auto *target_base = _builder.CreateExtractValue(target, {0u});
        auto *target_size = _builder.CreateExtractValue(target, {1u});
        auto *target_begin = _builder.CreatePtrToInt(
            target_base, _builder.getInt64Ty());
        auto *target_end = _builder.CreateAdd(target_begin, target_size);
        auto *target_range_valid = _builder.CreateICmpUGE(
            target_end, target_begin);
        auto *ordered_disjoint = _builder.CreateOr(
            _builder.CreateICmpULE(source_end, target_begin),
            _builder.CreateICmpULE(target_end, source_begin));
        auto *disjoint = _builder.CreateOr(
            _builder.CreateICmpEQ(target_size, _builder.getInt64(0u)),
            _builder.CreateAnd(
                _builder.CreateAnd(
                    source_range_valid, target_range_valid),
                ordered_disjoint),
            "buffer.interleaved.disjoint");
        guard = _builder.CreateAnd(
            guard, disjoint,
            "buffer.interleaved.alias.guard");
    }
    return guard;
}

[[nodiscard]] std::vector<::llvm::Value *>
ScheduleEmitter::_load_interleaved_scalar_buffer_read_group(
    const InterleavedScalarBufferReadGroup &group) {
    auto *source = _load_value(group.buffer);
    auto *index_value = _source.value(group.first_index);
    auto *index = index_value == nullptr ? nullptr :
                                           _as_lane_vector(
                                               _load_value(group.first_index),
                                               *index_value);
    if (source == nullptr || index == nullptr ||
        !index->getType()->isVectorTy()) {
        _fail("interleaved scalar buffer read has invalid load operands");
        return {};
    }
    auto *base = _builder.CreateExtractValue(source, {0u});
    auto *first_index = _builder.CreateZExtOrTrunc(
        _builder.CreateExtractElement(index, uint64_t{0u}),
        _builder.getInt64Ty());
    auto *byte_offset = _builder.CreateMul(
        first_index, _builder.getInt64(sizeof(uint32_t)));
    auto *address = _builder.CreateGEP(
        _builder.getInt8Ty(), base, byte_offset,
        "buffer.interleaved.address");
    auto *mask = _expand_lane_mask(
        _active_mask, group.field_count, group.field_count);
    auto *element = _data_type(group.element_type, false);
    if (mask == nullptr || element == nullptr) { return {}; }
    auto *lanes = ::llvm::FixedVectorType::get(
        element, _width * group.field_count);
    auto *loaded = _builder.CreateMaskedLoad(
        lanes, address, ::llvm::Align{1u}, mask,
        ::llvm::Constant::getNullValue(lanes),
        "buffer.interleaved.scalar.load");
    std::vector<::llvm::Value *> result;
    result.reserve(group.field_count);
    for (auto field = uint32_t{0u}; field < group.field_count; field++) {
        std::vector<int> shuffle;
        shuffle.reserve(_width);
        for (auto lane = uint32_t{0u}; lane < _width; lane++) {
            shuffle.emplace_back(static_cast<int>(
                lane * group.field_count + field));
        }
        result.emplace_back(_builder.CreateShuffleVector(
            loaded, ::llvm::PoisonValue::get(lanes), shuffle,
            "buffer.interleaved.scalar.deinterleave"));
    }
    return result;
}

void ScheduleEmitter::_emit_interleaved_scalar_buffer_read_group(
    const schedule::BasicBlock &block,
    const InterleavedScalarBufferReadGroup &group) {
    auto *guard = _interleaved_scalar_buffer_read_guard(group);
    if (guard == nullptr) { return; }
    auto &context = _module.getContext();
    auto *fast = ::llvm::BasicBlock::Create(
        context, "buffer.interleaved.fast", _entry);
    auto *fallback = ::llvm::BasicBlock::Create(
        context, "buffer.interleaved.fallback", _entry);
    auto *merge = ::llvm::BasicBlock::Create(
        context, "buffer.interleaved.merge", _entry);
    _builder.CreateCondBr(guard, fast, fallback);

    auto incoming_locals = _locals;
    auto before_uniform_reads = _result.uniform_buffer_broadcast_count;
    auto before_contiguous_reads = _result.contiguous_buffer_read_count;
    auto before_contiguous_writes = _result.contiguous_buffer_write_count;
    auto before_transposed_reads = _result.transposed_buffer_read_count;
    auto before_transposed_writes = _result.transposed_buffer_write_count;
    auto before_paired_gathers = _result.paired_leaf_gather_count;

    _builder.SetInsertPoint(fast);
    auto fields = _load_interleaved_scalar_buffer_read_group(group);
    if (fields.size() != group.results.size()) {
        if (!_failed()) {
            _fail("interleaved scalar buffer read field count mismatch");
        }
        return;
    }
    _interleaved_scalar_read_overrides.clear();
    for (auto i = size_t{0u}; i < fields.size(); i++) {
        _interleaved_scalar_read_overrides.emplace(
            group.results[i].value, fields[i]);
    }
    for (auto i = group.begin_instruction;
         i <= group.end_instruction; i++) {
        _emit_instruction(block.instructions[i]);
        if (_failed()) { return; }
    }
    _interleaved_scalar_read_overrides.clear();
    auto fast_locals = _locals;
    auto *fast_exit = _builder.GetInsertBlock();
    _builder.CreateBr(merge);
    auto fast_uniform_reads = _result.uniform_buffer_broadcast_count;
    auto fast_contiguous_reads = _result.contiguous_buffer_read_count;
    auto fast_contiguous_writes = _result.contiguous_buffer_write_count;
    auto fast_transposed_reads = _result.transposed_buffer_read_count;
    auto fast_transposed_writes = _result.transposed_buffer_write_count;
    auto fast_paired_gathers = _result.paired_leaf_gather_count;

    _locals = incoming_locals;
    _result.uniform_buffer_broadcast_count = before_uniform_reads;
    _result.contiguous_buffer_read_count = before_contiguous_reads;
    _result.contiguous_buffer_write_count = before_contiguous_writes;
    _result.transposed_buffer_read_count = before_transposed_reads;
    _result.transposed_buffer_write_count = before_transposed_writes;
    _result.paired_leaf_gather_count = before_paired_gathers;
    _builder.SetInsertPoint(fallback);
    for (auto i = group.begin_instruction;
         i <= group.end_instruction; i++) {
        _emit_instruction(block.instructions[i]);
        if (_failed()) { return; }
    }
    auto fallback_locals = _locals;
    auto *fallback_exit = _builder.GetInsertBlock();
    _builder.CreateBr(merge);

    auto retain_larger_delta = [](size_t before, size_t fast_value,
                                  size_t fallback_value) noexcept {
        return before + std::max(
                            fast_value - before,
                            fallback_value - before);
    };
    _result.uniform_buffer_broadcast_count = retain_larger_delta(
        before_uniform_reads, fast_uniform_reads,
        _result.uniform_buffer_broadcast_count);
    _result.contiguous_buffer_read_count = retain_larger_delta(
        before_contiguous_reads, fast_contiguous_reads,
        _result.contiguous_buffer_read_count);
    _result.contiguous_buffer_write_count = retain_larger_delta(
        before_contiguous_writes, fast_contiguous_writes,
        _result.contiguous_buffer_write_count);
    _result.transposed_buffer_read_count = retain_larger_delta(
        before_transposed_reads, fast_transposed_reads,
        _result.transposed_buffer_read_count);
    _result.transposed_buffer_write_count = retain_larger_delta(
        before_transposed_writes, fast_transposed_writes,
        _result.transposed_buffer_write_count);
    _result.paired_leaf_gather_count = retain_larger_delta(
        before_paired_gathers, fast_paired_gathers,
        _result.paired_leaf_gather_count);
    _result.interleaved_scalar_buffer_read_group_count++;
    _result.interleaved_scalar_buffer_read_count += group.field_count;
    _result.interleaved_scalar_buffer_read_alias_guard_count +=
        !group.crossed_write_buffers.empty();

    _builder.SetInsertPoint(merge);
    _locals = std::move(incoming_locals);
    for (auto i = group.begin_instruction;
         i <= group.end_instruction; i++) {
        auto &&instruction = block.instructions[i];
        if (!instruction.result) { continue; }
        auto fast_iter = fast_locals.find(instruction.result->value);
        auto fallback_iter = fallback_locals.find(
            instruction.result->value);
        if (fast_iter == fast_locals.end() ||
            fallback_iter == fallback_locals.end() ||
            fast_iter->second == nullptr ||
            fallback_iter->second == nullptr ||
            fast_iter->second->getType() !=
                fallback_iter->second->getType()) {
            _fail("interleaved scalar buffer read cannot merge a region value");
            return;
        }
        auto *phi = _builder.CreatePHI(
            fast_iter->second->getType(), 2u,
            "buffer.interleaved.value");
        phi->addIncoming(fast_iter->second, fast_exit);
        phi->addIncoming(fallback_iter->second, fallback_exit);
        _locals.insert_or_assign(instruction.result->value, phi);
    }
}

void ScheduleEmitter::_build_direct(::llvm::Value *initial_mask) {
    auto &context = _module.getContext();
    std::vector<bool> inlined_blocks(
        _source.blocks().size(), false);
    for (auto &&block : _source.blocks()) {
        if (auto diamond = _find_predicated_memory_diamond(block)) {
            inlined_blocks[diamond->true_block->id.value] = true;
            inlined_blocks[diamond->false_block->id.value] = true;
        }
    }
    std::vector<::llvm::BasicBlock *> blocks(
        _source.blocks().size(), nullptr);
    for (auto &&block : _source.blocks()) {
        if (!inlined_blocks[block.id.value]) {
            blocks[block.id.value] = ::llvm::BasicBlock::Create(
                context,
                "direct.schedule." + std::to_string(block.id.value),
                _entry);
        }
    }
    auto *activate = ::llvm::BasicBlock::Create(
        context, "direct.activate", _entry);
    auto *inactive = ::llvm::BasicBlock::Create(
        context, "direct.inactive", _entry);
    auto *active = _builder.CreateOrReduce(initial_mask);
    _builder.CreateCondBr(active, activate, inactive);

    _builder.SetInsertPoint(inactive);
    _builder.CreateRetVoid();

    _builder.SetInsertPoint(activate);
    _active_mask = _width == 1u ?
                       static_cast<::llvm::Value *>(
                           ::llvm::ConstantVector::getSplat(
                               ::llvm::ElementCount::getFixed(1u),
                               _builder.getTrue())) :
                       initial_mask;
    // With a statically row-aligned packet, any nonempty dispatch-edge mask
    // is a prefix and therefore contains lane zero. Direct control never
    // changes that mask, so keep the seed in a register constant instead of
    // repeating first-active extraction in hot memory loops.
    auto lane_zero_is_active =
        _entry_abi == ScheduleEntryABI::packet &&
        (_width == 1u ||
         (_static_block_size[0u] >= _width &&
          _static_block_size[0u] % _width == 0u));
    _seed_lane = lane_zero_is_active ?
                     static_cast<::llvm::Value *>(
                         _builder.getInt32(0u)) :
                     _safe_first_lane(_active_mask);
    _builder.CreateBr(blocks[_source.entry().value]);
    for (auto &&block : _source.blocks()) {
        if (inlined_blocks[block.id.value]) { continue; }
        _builder.SetInsertPoint(blocks[block.id.value]);
        _locals.clear();
        for (auto i = size_t{0u}; i < block.instructions.size(); i++) {
            if (auto group =
                    _find_interleaved_scalar_buffer_read_group(block, i)) {
                _emit_interleaved_scalar_buffer_read_group(block, *group);
                i = group->end_instruction;
            } else {
                _emit_instruction(block.instructions[i]);
            }
            if (_failed()) { return; }
        }
        _emit_direct_terminator(block, blocks);
        if (_failed()) { return; }
    }
}

void ScheduleEmitter::_build_predicated_acyclic(
    ::llvm::Value *initial_mask) {
    auto &context = _module.getContext();
    auto *mask_type = _layout.mask_type();
    auto *zero_mask = ::llvm::Constant::getNullValue(mask_type);
    auto block_count = _source.blocks().size();

    std::vector<::llvm::AllocaInst *> incoming_masks;
    incoming_masks.reserve(block_count);
    for (auto index = size_t{0u}; index < block_count; index++) {
        auto *mask = _builder.CreateAlloca(
            mask_type, nullptr,
            "predicated.acyclic.mask." + std::to_string(index));
        _builder.CreateStore(zero_mask, mask);
        incoming_masks.emplace_back(mask);
    }
    _builder.CreateStore(
        initial_mask, incoming_masks[_source.entry().value]);

    std::vector<::llvm::BasicBlock *> checks;
    std::vector<::llvm::BasicBlock *> bodies;
    checks.reserve(block_count);
    bodies.reserve(block_count);
    for (auto block : _predicated_acyclic_order) {
        checks.emplace_back(::llvm::BasicBlock::Create(
            context,
            "predicated.acyclic.check." +
                std::to_string(block.value),
            _entry));
        bodies.emplace_back(::llvm::BasicBlock::Create(
            context,
            "predicated.acyclic.body." +
                std::to_string(block.value),
            _entry));
    }
    auto *exit = ::llvm::BasicBlock::Create(
        context, "predicated.acyclic.exit", _entry);
    _builder.CreateBr(checks.front());

    auto merge_edge = [&](const schedule::ControlEdge &edge,
                          ::llvm::Value *mask) {
        _apply_assignments(edge.assignments, mask);
        if (_failed()) { return; }
        auto *slot = incoming_masks[edge.target.value];
        auto *previous = _builder.CreateLoad(mask_type, slot);
        _builder.CreateStore(_builder.CreateOr(previous, mask), slot);
    };

    for (auto order_index = size_t{0u};
         order_index < block_count; order_index++) {
        auto block_id = _predicated_acyclic_order[order_index];
        auto *block = _source.block(block_id);
        auto *next = order_index + 1u < block_count ?
                         checks[order_index + 1u] :
                         exit;

        _builder.SetInsertPoint(checks[order_index]);
        auto *block_mask = _builder.CreateLoad(
            mask_type, incoming_masks[block_id.value],
            "predicated.acyclic.active.mask");
        _builder.CreateCondBr(
            _builder.CreateOrReduce(block_mask),
            bodies[order_index], next);

        _builder.SetInsertPoint(bodies[order_index]);
        _active_mask = block_mask;
        _seed_lane = _safe_first_lane(_active_mask);
        _locals.clear();
        for (auto &&instruction : block->instructions) {
            _emit_instruction(instruction, nullptr, _active_mask);
            if (_failed()) { return; }
        }
        std::visit(
            [&](const auto &terminator) {
                using T = std::decay_t<decltype(terminator)>;
                if constexpr (std::is_same_v<
                                  T, schedule::BranchTerminator>) {
                    merge_edge(terminator.edge, _active_mask);
                } else if constexpr (std::is_same_v<
                                         T, schedule::SplitTerminator>) {
                    auto *condition_value = _source.value(
                        terminator.condition);
                    if (condition_value == nullptr) {
                        _fail("predicated acyclic split has an invalid condition");
                        return;
                    }
                    auto *condition = _as_lane_vector(
                        _load_value(terminator.condition),
                        *condition_value);
                    if (condition == nullptr) { return; }
                    auto *true_mask = _builder.CreateAnd(
                        _active_mask, condition);
                    auto *false_mask = _builder.CreateAnd(
                        _active_mask, _builder.CreateNot(condition));
                    merge_edge(terminator.true_edge, true_mask);
                    if (!_failed()) {
                        merge_edge(terminator.false_edge, false_mask);
                    }
                } else if constexpr (std::is_same_v<
                                         T, schedule::SwitchTerminator>) {
                    auto *selector_value = _source.value(
                        terminator.selector);
                    if (selector_value == nullptr) {
                        _fail("predicated acyclic switch has an invalid selector");
                        return;
                    }
                    auto *selector = _as_lane_vector(
                        _load_value(terminator.selector),
                        *selector_value);
                    if (selector == nullptr) { return; }
                    auto *remaining_mask = _active_mask;
                    auto *element_type = ::llvm::cast<
                        ::llvm::IntegerType>(
                        ::llvm::cast<::llvm::FixedVectorType>(
                            selector->getType())
                            ->getElementType());
                    for (auto &&item : terminator.cases) {
                        auto *label = _builder.CreateVectorSplat(
                            _width,
                            ::llvm::ConstantInt::get(
                                element_type, item.value));
                        auto *matches = _builder.CreateICmpEQ(
                            selector, label);
                        // Match Schedule's ordered switch semantics: a lane
                        // consumed by an earlier label cannot enter a later
                        // duplicate label.
                        auto *case_mask = _builder.CreateAnd(
                            remaining_mask, matches);
                        merge_edge(item.edge, case_mask);
                        if (_failed()) { return; }
                        remaining_mask = _builder.CreateAnd(
                            remaining_mask,
                            _builder.CreateNot(matches));
                    }
                    merge_edge(
                        terminator.default_edge, remaining_mask);
                } else if constexpr (std::is_same_v<
                                         T, schedule::JoinTerminator>) {
                    auto *point = _source.convergence(
                        terminator.convergence);
                    schedule::ControlEdge edge{point->target};
                    edge.assignments = terminator.assignments;
                    merge_edge(edge, _active_mask);
                } else if constexpr (!std::is_same_v<
                                         T, schedule::ReturnTerminator>) {
                    _fail("unsupported terminator reached predicated acyclic LLVM emission");
                }
            },
            block->terminator);
        if (_failed()) { return; }
        _builder.CreateBr(next);
    }

    _builder.SetInsertPoint(exit);
    _builder.CreateRetVoid();
}

}// namespace luisa::compute::simd::detail
