#include "llvm_schedule_emitter.h"

namespace luisa::compute::simd::detail {

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_lane_offsets(
    ::llvm::Value *index, uint64_t stride) {
    auto *i64_lanes = ::llvm::FixedVectorType::get(
        _builder.getInt64Ty(), _width);
    auto *extended = _builder.CreateZExtOrTrunc(index, i64_lanes);
    return stride == 1u ? extended : _builder.CreateMul(
        extended,
        _builder.CreateVectorSplat(
            _width, _builder.getInt64(stride)));
}

[[nodiscard]] std::optional<uint64_t> ScheduleEmitter::_constant_aggregate_index(
    schedule::ValueId id) const noexcept {
    auto *value = _source.value(id);
    if (value == nullptr ||
        value->origin != schedule::ValueOrigin::constant ||
        value->type == nullptr || !value->type->is_scalar() ||
        value->type->is_float() || value->type->size() == 0u ||
        value->type->size() > sizeof(uint64_t)) {
        return std::nullopt;
    }
    auto *metadata = std::get_if<schedule::ConstantValueMetadata>(
        &value->metadata);
    if (metadata == nullptr ||
        metadata->bytes.size() < value->type->size()) {
        return std::nullopt;
    }
    auto result = uint64_t{0u};
    std::memcpy(
        &result, metadata->bytes.data(), value->type->size());
    auto bits = static_cast<uint32_t>(value->type->size() * 8u);
    if (bits < 64u) {
        result &= (uint64_t{1u} << bits) - 1u;
    }
    if (value->type->is_int() &&
        (result & (uint64_t{1u} << (bits - 1u))) != 0u) {
        return std::nullopt;
    }
    return result;
}

[[nodiscard]] bool ScheduleEmitter::_advance_aggregate_offset(
    ::llvm::Value *&offsets, const Type *&current_type,
    schedule::ValueId index_id) {
    auto *index_value = _source.value(index_id);
    if (index_value == nullptr || index_value->type == nullptr ||
        !index_value->type->is_scalar() ||
        index_value->type->is_float() || current_type == nullptr ||
        _child_count(current_type) == 0u) {
        _fail("aggregate address has an invalid index or type path");
        return false;
    }
    if (current_type->is_structure()) {
        auto index = _constant_aggregate_index(index_id);
        if (!index || *index >= _child_count(current_type)) {
            _fail("structure address requires a valid constant member index");
            return false;
        }
        auto child_offset = _child_offset(
            current_type, static_cast<uint32_t>(*index));
        if (child_offset != 0u) {
            offsets = _builder.CreateAdd(
                offsets,
                _builder.CreateVectorSplat(
                    _width, _builder.getInt64(child_offset)));
        }
        current_type = _child_type(
            current_type, static_cast<uint32_t>(*index));
        return true;
    }

    auto *index = _as_lane_vector(
        _load_value(index_id), *index_value);
    if (index == nullptr) { return false; }
    auto *i64_lanes = ::llvm::FixedVectorType::get(
        _builder.getInt64Ty(), _width);
    auto *extended = index_value->type->is_int() ?
        _builder.CreateSExtOrTrunc(index, i64_lanes) :
        _builder.CreateZExtOrTrunc(index, i64_lanes);
    auto stride = current_type->is_vector() ?
        current_type->element()->size() :
        current_type->size() / current_type->dimension();
    if (stride != 1u) {
        extended = _builder.CreateMul(
            extended,
            _builder.CreateVectorSplat(
                _width, _builder.getInt64(stride)));
    }
    offsets = _builder.CreateAdd(offsets, extended);
    current_type = _child_type(current_type, 0u);
    return true;
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_local_alloca(
    const schedule::Instruction &instruction) {
    if (!instruction.result ||
        !_is_local_lvalue(*instruction.result) ||
        instruction.result->value >= _local_allocations.size()) {
        _fail("thread-local allocation is malformed");
        return nullptr;
    }
    auto *handle = _local_allocations[instruction.result->value];
    if (handle == nullptr) {
        _fail("thread-local allocation has no packet storage");
    }
    return handle;
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_local_gep(
    const schedule::Instruction &instruction) {
    if (!instruction.result || instruction.operands.empty()) {
        _fail("thread-local GEP is malformed");
        return nullptr;
    }
    auto *base_value = _source.value(instruction.operands.front());
    auto *result_value = _source.value(*instruction.result);
    auto *handle = _load_value(instruction.operands.front());
    if (base_value == nullptr || result_value == nullptr ||
        handle == nullptr || base_value->type == nullptr) {
        _fail("thread-local GEP has invalid values");
        return nullptr;
    }
    auto *base = _local_base(_builder, handle);
    auto *offsets = _local_offsets(_builder, handle);
    auto *current_type = base_value->type;
    for (auto i = size_t{1u}; i < instruction.operands.size(); i++) {
        if (!_advance_aggregate_offset(
                offsets, current_type,
                instruction.operands[i])) {
            return nullptr;
        }
    }
    if (current_type != result_value->type) {
        _fail("thread-local GEP result type does not match its index path");
        return nullptr;
    }
    return _local_handle(base, offsets);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_local_load(
    const schedule::Instruction &instruction) {
    if (!instruction.result || instruction.operands.size() != 1u) {
        _fail("thread-local load is malformed");
        return nullptr;
    }
    auto *variable = _source.value(instruction.operands.front());
    auto *result = _source.value(*instruction.result);
    auto *handle = _load_value(instruction.operands.front());
    if (variable == nullptr || result == nullptr || handle == nullptr ||
        variable->type != result->type) {
        _fail("thread-local load has mismatched value types");
        return nullptr;
    }
    return _gather_data(
        _local_base(_builder, handle),
        _local_offsets(_builder, handle), result->type);
}

void ScheduleEmitter::_local_store(const schedule::Instruction &instruction) {
    if (instruction.operands.size() != 2u) {
        _fail("thread-local store is malformed");
        return;
    }
    auto *variable = _source.value(instruction.operands[0u]);
    auto *written_value = _source.value(instruction.operands[1u]);
    auto *handle = _load_value(instruction.operands[0u]);
    auto *written = written_value == nullptr ? nullptr :
        _as_lane_vector(
            _load_value(instruction.operands[1u]), *written_value);
    if (variable == nullptr || written_value == nullptr ||
        handle == nullptr || written == nullptr ||
        variable->type != written_value->type) {
        _fail("thread-local store has mismatched value types");
        return;
    }
    _scatter_data(
        _local_base(_builder, handle),
        _local_offsets(_builder, handle),
        written_value->type, written);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_atomic(
    const schedule::Instruction &instruction) {
    if (!instruction.result || !instruction.source_op) {
        _fail("atomic instruction is missing a result or operation");
        return nullptr;
    }
    auto op = static_cast<xir::AtomicOp>(*instruction.source_op);
    auto value_count = xir::atomic_op_value_count(op);
    if (instruction.operands.size() < 2u + value_count) {
        _fail("atomic instruction has an invalid operand count");
        return nullptr;
    }
    auto index_count =
        instruction.operands.size() - 1u - value_count;
    auto *result = _source.value(*instruction.result);
    auto *buffer_value = _source.value(instruction.operands[0u]);
    auto *buffer = _load_value(instruction.operands[0u]);
    if (result == nullptr || buffer_value == nullptr ||
        buffer_value->type == nullptr || buffer == nullptr ||
        !buffer_value->type->is_buffer() ||
        buffer_value->type->element() == nullptr ||
        index_count == 0u || !_is_scalar_data(result->type)) {
        _fail("LLVM packet codegen requires a scalar typed-buffer atomic target");
        return nullptr;
    }

    auto *current_type = buffer_value->type->element();
    ::llvm::Value *offsets = nullptr;
    for (auto i = size_t{0u}; i < index_count; i++) {
        auto index_id = instruction.operands[1u + i];
        auto *index_value = _source.value(index_id);
        if (index_value == nullptr || index_value->type == nullptr ||
            !index_value->type->is_scalar() ||
            index_value->type->is_float()) {
            _fail("buffer atomic has an invalid aggregate index");
            return nullptr;
        }
        if (i == 0u) {
            auto *index = _as_lane_vector(
                _load_value(index_id), *index_value);
            if (index == nullptr) { return nullptr; }
            offsets = _lane_offsets(
                index,
                static_cast<uint64_t>(current_type->size()));
            continue;
        }
        if (!_advance_aggregate_offset(
                offsets, current_type, index_id)) {
            return nullptr;
        }
    }
    if (current_type != result->type) {
        _fail("buffer atomic result type does not match its aggregate index path");
        return nullptr;
    }
    std::vector<::llvm::Value *> values;
    values.reserve(value_count);
    for (auto i = size_t{0u}; i < value_count; i++) {
        auto operand_id = instruction.operands[
            instruction.operands.size() - value_count + i];
        auto *operand = _source.value(operand_id);
        auto *value = operand == nullptr ? nullptr :
            _as_lane_vector(_load_value(operand_id), *operand);
        if (value == nullptr) { return nullptr; }
        values.emplace_back(value);
    }

    auto *base = _builder.CreateExtractValue(buffer, {0u});
    auto *result_type = _layout.expression_type(*result);
    if (result_type == nullptr || !result_type->isVectorTy()) {
        _fail("buffer atomic result must be lane-varying");
        return nullptr;
    }
    ::llvm::Value *old_values =
        ::llvm::Constant::getNullValue(result_type);
    auto alignment = ::llvm::MaybeAlign{result->type->alignment()};
    for (auto lane = uint32_t{0u}; lane < _width; lane++) {
        auto *before = _builder.GetInsertBlock();
        auto *active = _builder.CreateExtractElement(
            _active_mask, lane);
        auto *atomic_block = ::llvm::BasicBlock::Create(
            _module.getContext(),
            "atomic.lane." + std::to_string(lane), _entry);
        auto *continue_block = ::llvm::BasicBlock::Create(
            _module.getContext(),
            "atomic.continue." + std::to_string(lane), _entry);
        _builder.CreateCondBr(active, atomic_block, continue_block);

        _builder.SetInsertPoint(atomic_block);
        auto *offset = _builder.CreateExtractElement(offsets, lane);
        auto *pointer = _builder.CreateGEP(
            _builder.getInt8Ty(), base, offset);
        std::vector<::llvm::Value *> lane_values;
        lane_values.reserve(values.size());
        for (auto *value : values) {
            lane_values.emplace_back(
                _builder.CreateExtractElement(value, lane));
        }
        ::llvm::Value *old = nullptr;
        if (op == xir::AtomicOp::COMPARE_EXCHANGE) {
            auto *expected = lane_values[0u];
            auto *desired = lane_values[1u];
            auto *value_type = expected->getType();
            auto floating = value_type->isFloatingPointTy();
            if (floating) {
                auto *integer = ::llvm::IntegerType::get(
                    _module.getContext(),
                    value_type->getPrimitiveSizeInBits());
                expected = _builder.CreateBitCast(expected, integer);
                desired = _builder.CreateBitCast(desired, integer);
            }
            auto *pair = _builder.CreateAtomicCmpXchg(
                pointer, expected, desired, alignment,
                ::llvm::AtomicOrdering::Monotonic,
                ::llvm::AtomicOrdering::Monotonic);
            old = _builder.CreateExtractValue(pair, {0u});
            if (floating) {
                old = _builder.CreateBitCast(old, value_type);
            }
        } else {
            auto atomic_op = ::llvm::AtomicRMWInst::BAD_BINOP;
            auto floating = result->type->is_float16() ||
                            result->type->is_float32() ||
                            result->type->is_float64();
            auto signed_integer = result->type->is_int();
            switch (op) {
                case xir::AtomicOp::EXCHANGE:
                    atomic_op = ::llvm::AtomicRMWInst::Xchg;
                    break;
                case xir::AtomicOp::FETCH_ADD:
                    atomic_op = floating ?
                        ::llvm::AtomicRMWInst::FAdd :
                        ::llvm::AtomicRMWInst::Add;
                    break;
                case xir::AtomicOp::FETCH_SUB:
                    atomic_op = floating ?
                        ::llvm::AtomicRMWInst::FSub :
                        ::llvm::AtomicRMWInst::Sub;
                    break;
                case xir::AtomicOp::FETCH_AND:
                    atomic_op = ::llvm::AtomicRMWInst::And;
                    break;
                case xir::AtomicOp::FETCH_OR:
                    atomic_op = ::llvm::AtomicRMWInst::Or;
                    break;
                case xir::AtomicOp::FETCH_XOR:
                    atomic_op = ::llvm::AtomicRMWInst::Xor;
                    break;
                case xir::AtomicOp::FETCH_MIN:
                    atomic_op = floating ?
                        ::llvm::AtomicRMWInst::FMin :
                        signed_integer ?
                            ::llvm::AtomicRMWInst::Min :
                            ::llvm::AtomicRMWInst::UMin;
                    break;
                case xir::AtomicOp::FETCH_MAX:
                    atomic_op = floating ?
                        ::llvm::AtomicRMWInst::FMax :
                        signed_integer ?
                            ::llvm::AtomicRMWInst::Max :
                            ::llvm::AtomicRMWInst::UMax;
                    break;
                case xir::AtomicOp::COMPARE_EXCHANGE: break;
            }
            if (atomic_op == ::llvm::AtomicRMWInst::BAD_BINOP) {
                _fail("unsupported direct-buffer atomic operation");
                return nullptr;
            }
            old = _builder.CreateAtomicRMW(
                atomic_op, pointer, lane_values[0u], alignment,
                ::llvm::AtomicOrdering::Monotonic);
        }
        auto *updated = _builder.CreateInsertElement(
            old_values, old, lane);
        _builder.CreateBr(continue_block);
        auto *atomic_end = _builder.GetInsertBlock();

        _builder.SetInsertPoint(continue_block);
        auto *phi = _builder.CreatePHI(result_type, 2u);
        phi->addIncoming(old_values, before);
        phi->addIncoming(updated, atomic_end);
        old_values = phi;
    }
    return old_values;
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_leaf_pointers(
    ::llvm::Value *base, ::llvm::Value *offsets,
    size_t leaf_offset) {
    if (leaf_offset != 0u) {
        offsets = _builder.CreateAdd(
            offsets,
            _builder.CreateVectorSplat(
                _width, _builder.getInt64(leaf_offset)));
    }
    return _builder.CreateGEP(
        _builder.getInt8Ty(), base, offsets);
}

[[nodiscard]] ::llvm::AllocaInst *ScheduleEmitter::_entry_scratch(
    ::llvm::Type *type, std::string_view name) {
    auto &entry_block = _entry->getEntryBlock();
    ::llvm::IRBuilder<> builder{
        &entry_block, entry_block.begin()};
    return builder.CreateAlloca(
        type, nullptr, ::llvm::StringRef{name.data(), name.size()});
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_texture_read(
    const schedule::Instruction &instruction) {
    if (!instruction.result || instruction.operands.size() != 2u) {
        _fail("texture read instruction is malformed");
        return nullptr;
    }
    auto *result = _source.value(*instruction.result);
    auto *texture_value = _source.value(instruction.operands[0u]);
    auto *coordinate_value = _source.value(instruction.operands[1u]);
    auto *texture = _load_value(instruction.operands[0u]);
    auto *coordinate = coordinate_value == nullptr ? nullptr :
        _as_lane_vector(
            _load_value(instruction.operands[1u]),
            *coordinate_value);
    if (result == nullptr || texture_value == nullptr ||
        texture == nullptr || coordinate == nullptr ||
        !texture_value->type->is_texture() ||
        result->value_class != schedule::ValueClass::varying ||
        !result->type->is_vector() ||
        result->type->dimension() != 4u ||
        !coordinate_value->type->is_vector() ||
        (coordinate_value->type->dimension() != 2u &&
         coordinate_value->type->dimension() != 3u)) {
        _fail("LLVM packet codegen requires varying float4/uint4 direct texture reads");
        return nullptr;
    }
    auto *element = result->type->element();
    auto floating = element->is_float32();
    auto integer = element->is_int32() || element->is_uint32();
    if (!floating && !integer) {
        _fail("direct texture reads currently support float32 and int32 elements");
        return nullptr;
    }
    auto op = static_cast<xir::ResourceReadOp>(
        *instruction.source_op);
    auto expected_dimension =
        op == xir::ResourceReadOp::TEXTURE2D_READ ? 2u :
        op == xir::ResourceReadOp::TEXTURE3D_READ ? 3u : 0u;
    if (expected_dimension == 0u ||
        coordinate_value->type->dimension() != expected_dimension) {
        _fail("direct texture read dimension mismatch");
        return nullptr;
    }
    if (_width > 64u) {
        _fail("packet texture callbacks support widths up to 64 lanes");
        return nullptr;
    }

    std::array<::llvm::Value *, 3u> coordinates{
        nullptr, nullptr,
        _builder.CreateVectorSplat(
            _width, _builder.getInt32(0u))};
    for (auto axis = uint32_t{0u}; axis < expected_dimension; axis++) {
        coordinates[axis] = _extract_child(
            coordinate, coordinate_value->type, axis, true);
    }
    auto *scalar_type = _data_type(element, false);
    auto *lane_type = ::llvm::FixedVectorType::get(scalar_type, _width);
    auto *scratch_type = ::llvm::ArrayType::get(lane_type, 4u);
    auto *scratch = _entry_scratch(
        scratch_type,
        "texture.read.packet." +
            std::to_string(instruction.result->value));
    auto *read = _builder.CreateExtractValue(
        texture, {floating ? 1u : 2u});
    auto *object = _builder.CreateExtractValue(texture, {0u});
    auto *level = _builder.CreateExtractValue(texture, {6u});
    auto *read_type = ::llvm::FunctionType::get(
        _builder.getVoidTy(),
        {::llvm::PointerType::getUnqual(_module.getContext()),
         _builder.getInt32Ty(), _builder.getInt32Ty(),
         _builder.getInt64Ty(),
         ::llvm::PointerType::getUnqual(_module.getContext()),
         ::llvm::PointerType::getUnqual(_module.getContext()),
         ::llvm::PointerType::getUnqual(_module.getContext()),
         ::llvm::PointerType::getUnqual(_module.getContext())},
        false);
    std::array<::llvm::AllocaInst *, 3u> coordinate_scratch{};
    auto *coordinate_type = ::llvm::FixedVectorType::get(
        _builder.getInt32Ty(), _width);
    for (auto axis = uint32_t{0u}; axis < 3u; axis++) {
        coordinate_scratch[axis] = _entry_scratch(
            coordinate_type,
            "texture.read.coordinate." + std::to_string(axis));
        _builder.CreateStore(
            coordinates[axis], coordinate_scratch[axis]);
    }
    auto *packed_mask_type = ::llvm::IntegerType::get(
        _module.getContext(), _width);
    auto *packed_mask = _builder.CreateBitCast(
        _active_mask, packed_mask_type);
    auto *active_mask_bits = _builder.CreateZExtOrTrunc(
        packed_mask, _builder.getInt64Ty());
    _builder.CreateStore(
        ::llvm::Constant::getNullValue(scratch_type), scratch);
    _builder.CreateCall(
        read_type, read,
        {object, level, _builder.getInt32(_width), active_mask_bits,
         coordinate_scratch[0u], coordinate_scratch[1u],
         coordinate_scratch[2u], scratch});
    auto *result_type = _data_type(result->type, true);
    auto *pixels = static_cast<::llvm::Value *>(
        ::llvm::PoisonValue::get(result_type));
    for (auto component = uint32_t{0u}; component < 4u;
         component++) {
        auto *component_pointer = _builder.CreateGEP(
            scratch_type, scratch,
            {_builder.getInt32(0u),
             _builder.getInt32(component)});
        auto *lanes = _builder.CreateLoad(
            lane_type, component_pointer);
        pixels = _insert_child(
            pixels, lanes, result->type, component, true);
    }
    return pixels;
}

void ScheduleEmitter::_texture_write(
    const schedule::Instruction &instruction) {
    if (instruction.operands.size() != 3u) {
        _fail("texture write instruction is malformed");
        return;
    }
    auto *texture_value = _source.value(instruction.operands[0u]);
    auto *coordinate_value = _source.value(instruction.operands[1u]);
    auto *written_value = _source.value(instruction.operands[2u]);
    auto *texture = _load_value(instruction.operands[0u]);
    auto *coordinate = coordinate_value == nullptr ? nullptr :
        _as_lane_vector(
            _load_value(instruction.operands[1u]),
            *coordinate_value);
    auto *written = written_value == nullptr ? nullptr :
        _as_lane_vector(
            _load_value(instruction.operands[2u]), *written_value);
    if (texture_value == nullptr || coordinate_value == nullptr ||
        written_value == nullptr || texture == nullptr ||
        coordinate == nullptr || written == nullptr ||
        !texture_value->type->is_texture() ||
        !written_value->type->is_vector() ||
        written_value->type->dimension() != 4u ||
        !coordinate_value->type->is_vector()) {
        _fail("direct texture write has invalid operands");
        return;
    }
    auto op = static_cast<xir::ResourceWriteOp>(
        *instruction.source_op);
    auto expected_dimension =
        op == xir::ResourceWriteOp::TEXTURE2D_WRITE ? 2u :
        op == xir::ResourceWriteOp::TEXTURE3D_WRITE ? 3u : 0u;
    if (expected_dimension == 0u ||
        coordinate_value->type->dimension() != expected_dimension) {
        _fail("direct texture write dimension mismatch");
        return;
    }
    auto *element = written_value->type->element();
    auto floating = element->is_float32();
    auto integer = element->is_int32() || element->is_uint32();
    if (!floating && !integer) {
        _fail("direct texture writes currently support float32 and int32 elements");
        return;
    }
    if (_width > 64u) {
        _fail("packet texture callbacks support widths up to 64 lanes");
        return;
    }
    auto *scalar_type = _data_type(element, false);
    auto *lane_type = ::llvm::FixedVectorType::get(scalar_type, _width);
    auto *scratch_type = ::llvm::ArrayType::get(lane_type, 4u);
    auto *scratch = _entry_scratch(
        scratch_type,
        "texture.write.packet." + std::to_string(
            instruction.operands[2u].value));
    auto *write = _builder.CreateExtractValue(
        texture, {floating ? 3u : 4u});
    auto *object = _builder.CreateExtractValue(texture, {0u});
    auto *level = _builder.CreateExtractValue(texture, {6u});
    auto *write_type = ::llvm::FunctionType::get(
        _builder.getVoidTy(),
        {::llvm::PointerType::getUnqual(_module.getContext()),
         _builder.getInt32Ty(), _builder.getInt32Ty(),
         _builder.getInt64Ty(),
         ::llvm::PointerType::getUnqual(_module.getContext()),
         ::llvm::PointerType::getUnqual(_module.getContext()),
         ::llvm::PointerType::getUnqual(_module.getContext()),
         ::llvm::PointerType::getUnqual(_module.getContext())},
        false);
    std::array<::llvm::Value *, 3u> coordinates{
        nullptr, nullptr,
        _builder.CreateVectorSplat(
            _width, _builder.getInt32(0u))};
    for (auto axis = uint32_t{0u}; axis < expected_dimension; axis++) {
        coordinates[axis] = _extract_child(
            coordinate, coordinate_value->type, axis, true);
    }
    std::array<::llvm::AllocaInst *, 3u> coordinate_scratch{};
    auto *coordinate_type = ::llvm::FixedVectorType::get(
        _builder.getInt32Ty(), _width);
    for (auto axis = uint32_t{0u}; axis < 3u; axis++) {
        coordinate_scratch[axis] = _entry_scratch(
            coordinate_type,
            "texture.write.coordinate." + std::to_string(axis));
        _builder.CreateStore(coordinates[axis], coordinate_scratch[axis]);
    }
    for (auto component = uint32_t{0u}; component < 4u; component++) {
        auto *component_pointer = _builder.CreateGEP(
            scratch_type, scratch,
            {_builder.getInt32(0u), _builder.getInt32(component)});
        auto *lanes = _extract_child(
            written, written_value->type, component, true);
        _builder.CreateStore(lanes, component_pointer);
    }
    auto *packed_mask_type = ::llvm::IntegerType::get(
        _module.getContext(), _width);
    auto *packed_mask = _builder.CreateBitCast(
        _active_mask, packed_mask_type);
    auto *active_mask_bits = _builder.CreateZExtOrTrunc(
        packed_mask, _builder.getInt64Ty());
    _builder.CreateCall(
        write_type, write,
        {object, level, _builder.getInt32(_width), active_mask_bits,
         coordinate_scratch[0u], coordinate_scratch[1u],
         coordinate_scratch[2u], scratch});
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_gather_data(
    ::llvm::Value *base, ::llvm::Value *offsets,
    const Type *type, size_t leaf_offset) {
    if (_is_scalar_data(type)) {
        auto *pointers = _leaf_pointers(base, offsets, leaf_offset);
        auto *element = type->is_bool() ?
            static_cast<::llvm::Type *>(_builder.getInt8Ty()) :
            _data_type(type, false);
        auto *lanes = ::llvm::FixedVectorType::get(element, _width);
        auto *gathered = _builder.CreateMaskedGather(
            lanes, pointers, ::llvm::Align{1u}, _active_mask,
            ::llvm::Constant::getNullValue(lanes));
        return type->is_bool() ?
            _builder.CreateICmpNE(
                gathered, ::llvm::Constant::getNullValue(lanes)) :
            gathered;
    }
    return _assemble(type, true, [&](uint32_t i) {
        return _gather_data(
            base, offsets, _child_type(type, i),
            leaf_offset + _child_offset(type, i));
    });
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_load_contiguous_data(
    ::llvm::Value *base, ::llvm::Value *index,
    const Type *type) {
    if (!_is_scalar_data(type) || index == nullptr ||
        !index->getType()->isVectorTy()) {
        _fail("contiguous buffer load requires a scalar type and lane index");
        return nullptr;
    }
    auto *seed = _builder.CreateExtractElement(index, _seed_lane);
    auto *seed_index = _builder.CreateZExtOrTrunc(
        seed, _builder.getInt64Ty());
    auto *seed_lane = _builder.CreateZExtOrTrunc(
        _seed_lane, _builder.getInt64Ty());
    auto *offset = _builder.CreateSub(seed_index, seed_lane);
    if (type->size() != 1u) {
        offset = _builder.CreateMul(
            offset, _builder.getInt64(type->size()));
    }
    auto *address = _builder.CreateGEP(
        _builder.getInt8Ty(), base, offset);
    auto *element = type->is_bool() ?
        static_cast<::llvm::Type *>(_builder.getInt8Ty()) :
        _data_type(type, false);
    auto *lanes = ::llvm::FixedVectorType::get(element, _width);
    auto *loaded = _builder.CreateMaskedLoad(
        lanes, address, ::llvm::Align{1u}, _active_mask,
        ::llvm::Constant::getNullValue(lanes),
        "buffer.contiguous.load");
    return type->is_bool() ?
        _builder.CreateICmpNE(
            loaded, ::llvm::Constant::getNullValue(lanes)) :
        loaded;
}

void ScheduleEmitter::_scatter_data(
    ::llvm::Value *base, ::llvm::Value *offsets,
    const Type *type, ::llvm::Value *value,
    size_t leaf_offset) {
    if (_is_scalar_data(type)) {
        auto *pointers = _leaf_pointers(base, offsets, leaf_offset);
        if (type->is_bool()) {
            value = _builder.CreateZExt(
                value,
                ::llvm::FixedVectorType::get(
                    _builder.getInt8Ty(), _width));
        }
        _builder.CreateMaskedScatter(
            value, pointers, ::llvm::Align{1u}, _active_mask);
        return;
    }
    for (auto i = uint32_t{0u}; i < _child_count(type); i++) {
        _scatter_data(
            base, offsets, _child_type(type, i),
            _extract_child(value, type, i, true),
            leaf_offset + _child_offset(type, i));
    }
}

void ScheduleEmitter::_store_contiguous_data(
    ::llvm::Value *base, ::llvm::Value *index,
    const Type *type, ::llvm::Value *value) {
    if (!_is_scalar_data(type) || index == nullptr || value == nullptr ||
        !index->getType()->isVectorTy()) {
        _fail("contiguous buffer store requires a scalar type and lane index");
        return;
    }
    auto *seed = _builder.CreateExtractElement(index, _seed_lane);
    auto *seed_index = _builder.CreateZExtOrTrunc(
        seed, _builder.getInt64Ty());
    auto *seed_lane = _builder.CreateZExtOrTrunc(
        _seed_lane, _builder.getInt64Ty());
    auto *offset = _builder.CreateSub(seed_index, seed_lane);
    if (type->size() != 1u) {
        offset = _builder.CreateMul(
            offset, _builder.getInt64(type->size()));
    }
    auto *address = _builder.CreateGEP(
        _builder.getInt8Ty(), base, offset);
    if (type->is_bool()) {
        value = _builder.CreateZExt(
            value,
            ::llvm::FixedVectorType::get(
                _builder.getInt8Ty(), _width));
    }
    _builder.CreateMaskedStore(
        value, address, ::llvm::Align{1u}, _active_mask);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_resource_read(
    const schedule::Instruction &instruction) {
    if (!instruction.result || !instruction.source_op) {
        _fail("buffer read instruction is malformed");
        return nullptr;
    }
    auto op = static_cast<xir::ResourceReadOp>(
        *instruction.source_op);
    if (op == xir::ResourceReadOp::BINDLESS_BUFFER_READ ||
        op == xir::ResourceReadOp::BINDLESS_BYTE_BUFFER_READ) {
        return _bindless_resource_read(instruction);
    }
    if (instruction.operands.size() != 2u) {
        _fail("direct resource read instruction is malformed");
        return nullptr;
    }
    if (op == xir::ResourceReadOp::TEXTURE2D_READ ||
        op == xir::ResourceReadOp::TEXTURE3D_READ) {
        return _texture_read(instruction);
    }
    auto byte_address =
        op == xir::ResourceReadOp::BYTE_BUFFER_READ ||
        op == xir::ResourceReadOp::BYTE_BUFFER_VOLATILE_READ;
    if (!byte_address &&
        op != xir::ResourceReadOp::BUFFER_READ &&
        op != xir::ResourceReadOp::BUFFER_VOLATILE_READ) {
        _fail("LLVM packet codegen only supports direct buffer reads");
        return nullptr;
    }
    auto *buffer_value = _source.value(instruction.operands[0u]);
    auto *index_value = _source.value(instruction.operands[1u]);
    auto *result_value = _source.value(*instruction.result);
    auto *buffer = _load_value(instruction.operands[0u]);
    auto *index = index_value == nullptr ?
                      nullptr :
                      _load_value(instruction.operands[1u]);
    if (buffer_value == nullptr || result_value == nullptr ||
        buffer == nullptr || index == nullptr ||
        !buffer_value->type->is_buffer()) {
        _fail("buffer read has invalid operands");
        return nullptr;
    }
    auto stride = byte_address ? 1u :
        static_cast<uint64_t>(buffer_value->type->element()->size());
    auto *base = _builder.CreateExtractValue(buffer, {0u});

    // A non-volatile typed-buffer read through a warp/cohort-uniform index
    // observes one address for every executing lane. A canonical induction
    // can also carry a use-site proof: its state remains varying because loop
    // exits may merge different epochs, but every lane in this instruction's
    // in-loop continuation has the same index. Load that address once and
    // broadcast its Luisa value instead of materializing W identical pointers
    // and a masked gather. The Schedule block never executes with an empty
    // active mask, so this does not make a previously inactive access
    // observable. Volatile and byte-address reads retain their original
    // per-lane memory operations.
    // W2 needs a dynamic first-active-lane extraction but avoids only one
    // additional address/load lane. Repeated n-body A/B measurements show
    // that trade is neutral-to-negative, whereas W4/W8/W16 consistently win.
    // W1 uses the statically known lane zero and retains the scalar form.
    auto cohort_uniform_at_use =
        instruction.cohort_uniform_operand_index == 1u &&
        _width != 2u;
    if (_enable_uniform_buffer_broadcast &&
        op == xir::ResourceReadOp::BUFFER_READ &&
        (schedule::is_uniform(index_value->value_class) ||
         cohort_uniform_at_use)) {
        if (index->getType()->isVectorTy()) {
            if (!cohort_uniform_at_use) {
                _fail("uniform buffer index unexpectedly has a vector representation");
                return nullptr;
            }
            index = _extract_lane(
                index, index_value->type,
                _width == 1u ? _builder.getInt32(0u) :
                               _seed_lane);
        }
        auto *offset = _builder.CreateZExtOrTrunc(
            index, _builder.getInt64Ty());
        if (stride != 1u) {
            offset = _builder.CreateMul(
                offset, _builder.getInt64(stride));
        }
        auto *address = _builder.CreateGEP(
            _builder.getInt8Ty(), base, offset);
        auto *uniform = _load_uniform_data(
            address, result_value->type);
        if (uniform == nullptr) { return nullptr; }
        _result.uniform_buffer_broadcast_count++;
        return schedule::is_uniform(result_value->value_class) ?
                   uniform :
                   _splat_data(uniform, result_value->type);
    }

    auto lane_consecutive_at_use =
        instruction.lane_consecutive_operand_index == 1u;
    // W2 masked contiguous operations cost more than LLVM's two-lane
    // gather/scatter legalization on the measured host. Keep the proof in
    // Schedule IR, but select this wide-memory lowering only from W4 upward.
    if (_enable_lane_affine_buffer && _width >= 4u &&
        op == xir::ResourceReadOp::BUFFER_READ &&
        lane_consecutive_at_use &&
        _is_scalar_data(result_value->type) &&
        buffer_value->type->element() == result_value->type) {
        index = _as_lane_vector(index, *index_value);
        if (index == nullptr) { return nullptr; }
        auto *loaded = _load_contiguous_data(
            base, index, result_value->type);
        if (loaded != nullptr) {
            _result.contiguous_buffer_read_count++;
        }
        return loaded;
    }

    index = _as_lane_vector(index, *index_value);
    if (index == nullptr) { return nullptr; }
    return _gather_data(
        base, _lane_offsets(index, stride), result_value->type);
}

void ScheduleEmitter::_resource_write(const schedule::Instruction &instruction) {
    if (!instruction.source_op) {
        _fail("buffer write instruction is malformed");
        return;
    }
    auto op = static_cast<xir::ResourceWriteOp>(
        *instruction.source_op);
    if (op == xir::ResourceWriteOp::BINDLESS_BUFFER_WRITE ||
        op == xir::ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE) {
        _bindless_resource_write(instruction);
        return;
    }
    if (instruction.operands.size() != 3u) {
        _fail("direct resource write instruction is malformed");
        return;
    }
    if (op == xir::ResourceWriteOp::TEXTURE2D_WRITE ||
        op == xir::ResourceWriteOp::TEXTURE3D_WRITE) {
        _texture_write(instruction);
        return;
    }
    auto byte_address =
        op == xir::ResourceWriteOp::BYTE_BUFFER_WRITE ||
        op == xir::ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE;
    if (!byte_address &&
        op != xir::ResourceWriteOp::BUFFER_WRITE &&
        op != xir::ResourceWriteOp::BUFFER_VOLATILE_WRITE) {
        _fail("LLVM packet codegen only supports direct buffer writes");
        return;
    }
    auto *buffer_value = _source.value(instruction.operands[0u]);
    auto *index_value = _source.value(instruction.operands[1u]);
    auto *written_value = _source.value(instruction.operands[2u]);
    auto *buffer = _load_value(instruction.operands[0u]);
    auto *index = _as_lane_vector(
        _load_value(instruction.operands[1u]), *index_value);
    auto *value = _as_lane_vector(
        _load_value(instruction.operands[2u]), *written_value);
    if (buffer_value == nullptr || buffer == nullptr ||
        index == nullptr || value == nullptr ||
        !buffer_value->type->is_buffer()) {
        _fail("buffer write has invalid operands");
        return;
    }
    auto stride = byte_address ? 1u :
        static_cast<uint64_t>(buffer_value->type->element()->size());
    auto *base = _builder.CreateExtractValue(buffer, {0u});
    auto lane_consecutive_at_use =
        instruction.lane_consecutive_operand_index == 1u;
    if (_enable_lane_affine_buffer && _width >= 4u &&
        op == xir::ResourceWriteOp::BUFFER_WRITE &&
        lane_consecutive_at_use &&
        _is_scalar_data(written_value->type) &&
        buffer_value->type->element() == written_value->type) {
        _store_contiguous_data(
            base, index, written_value->type, value);
        if (!_failed()) {
            _result.contiguous_buffer_write_count++;
        }
        return;
    }
    _scatter_data(
        base, _lane_offsets(index, stride), written_value->type, value);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_resource_query(
    const schedule::Instruction &instruction) {
    if (!instruction.result || !instruction.source_op) {
        _fail("buffer query instruction is malformed");
        return nullptr;
    }
    auto op = static_cast<xir::ResourceQueryOp>(
        *instruction.source_op);
    if (op == xir::ResourceQueryOp::BINDLESS_BUFFER_SIZE ||
        op == xir::ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE ||
        op == xir::ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS) {
        return _bindless_resource_query(instruction);
    }
    if (instruction.operands.size() != 1u) {
        _fail("direct resource query instruction is malformed");
        return nullptr;
    }
    auto *result = _source.value(*instruction.result);
    auto *buffer_value = _source.value(instruction.operands[0u]);
    auto *buffer = _load_value(instruction.operands[0u]);
    if (result == nullptr || buffer_value == nullptr ||
        buffer == nullptr) {
        _fail("resource query has invalid operands");
        return nullptr;
    }
    if (op == xir::ResourceQueryOp::TEXTURE2D_SIZE ||
        op == xir::ResourceQueryOp::TEXTURE3D_SIZE) {
        if (!buffer_value->type->is_texture() ||
            !result->type->is_vector()) {
            _fail("texture size query has invalid types");
            return nullptr;
        }
        auto dimension =
            op == xir::ResourceQueryOp::TEXTURE2D_SIZE ? 2u : 3u;
        if (result->type->dimension() != dimension) {
            _fail("texture size query result dimension mismatch");
            return nullptr;
        }
        auto *object = _builder.CreateExtractValue(buffer, {0u});
        auto *size = _builder.CreateExtractValue(buffer, {5u});
        auto *level = _builder.CreateExtractValue(buffer, {6u});
        auto *size_type = ::llvm::FunctionType::get(
            _builder.getInt32Ty(),
            {::llvm::PointerType::getUnqual(_module.getContext()),
             _builder.getInt32Ty(), _builder.getInt32Ty()},
            false);
        auto *uniform = _assemble(
            result->type, false, [&](uint32_t axis) {
                return _builder.CreateCall(
                    size_type, size,
                    {object, level, _builder.getInt32(axis)});
            });
        return result->value_class ==
                schedule::ValueClass::varying ?
            _splat_data(uniform, result->type) : uniform;
    }
    if (!buffer_value->type->is_buffer()) {
        _fail("buffer query has invalid resource type");
        return nullptr;
    }
    auto *value = _builder.CreateExtractValue(buffer, {1u});
    switch (op) {
        case xir::ResourceQueryOp::BUFFER_SIZE:
            value = _builder.CreateUDiv(
                value,
                _builder.getInt64(
                    buffer_value->type->element()->size()));
            break;
        case xir::ResourceQueryOp::BYTE_BUFFER_SIZE: break;
        case xir::ResourceQueryOp::BUFFER_DEVICE_ADDRESS: {
            auto *pointer = _builder.CreateExtractValue(buffer, {0u});
            value = _builder.CreatePtrToInt(
                pointer, _builder.getInt64Ty());
            break;
        }
        default:
            _fail("LLVM packet codegen only supports direct buffer queries");
            return nullptr;
    }
    auto *destination = _data_type(result->type, false);
    value = _builder.CreateZExtOrTrunc(value, destination);
    return result->value_class == schedule::ValueClass::varying ?
        _splat_data(value, result->type) : value;
}

}// namespace luisa::compute::simd::detail
