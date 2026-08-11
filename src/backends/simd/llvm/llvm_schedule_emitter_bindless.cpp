#include "llvm_schedule_emitter.h"

#include <limits>

namespace luisa::compute::simd::detail {

[[nodiscard]] ScheduleEmitter::BindlessBufferLanes
ScheduleEmitter::_bindless_buffer_lanes(
    schedule::ValueId bindless_id, schedule::ValueId slot_id) {
    auto *bindless_value = _source.value(bindless_id);
    auto *slot_value = _source.value(slot_id);
    auto *bindless = _load_value(bindless_id);
    auto *slot = slot_value == nullptr ? nullptr :
        _as_lane_vector(_load_value(slot_id), *slot_value);
    if (bindless_value == nullptr || slot_value == nullptr ||
        bindless == nullptr || slot == nullptr ||
        bindless_value->type == nullptr ||
        !bindless_value->type->is_bindless_array() ||
        slot_value->type == nullptr ||
        !slot_value->type->is_scalar() ||
        slot_value->type->is_float()) {
        _fail("bindless buffer access has invalid array or slot operands");
        return {};
    }

    auto *i64_lanes = ::llvm::FixedVectorType::get(
        _builder.getInt64Ty(), _width);
    auto *slot_indices = _builder.CreateZExtOrTrunc(slot, i64_lanes);
    auto *slots = _builder.CreateExtractValue(bindless, {0u});
    auto *slot_count = _builder.CreateExtractValue(bindless, {1u});
    auto *out_of_range = _builder.CreateICmpUGE(
        slot_indices, _builder.CreateVectorSplat(_width, slot_count));
    _trap_if(
        _builder.CreateOrReduce(
            _builder.CreateAnd(_active_mask, out_of_range)),
        "bindless.slot.out_of_range");
    auto *no_slots = _builder.CreateICmpEQ(
        slots,
        ::llvm::ConstantPointerNull::get(
            ::llvm::PointerType::getUnqual(_module.getContext())));
    _trap_if(
        _builder.CreateAnd(
            _builder.CreateOrReduce(_active_mask), no_slots),
        "bindless.slots.null");

    auto *slot_offsets = _builder.CreateMul(
        slot_indices,
        _builder.CreateVectorSplat(
            _width,
            _builder.getInt64(sizeof(SIMDHostBindlessSlot))));
    auto *slot_bases = _builder.CreateGEP(
        _builder.getInt8Ty(),
        _builder.CreateVectorSplat(_width, slots), slot_offsets);
    auto field_pointers = [&](size_t offset) noexcept {
        return offset == 0u ? slot_bases :
            _builder.CreateGEP(
                _builder.getInt8Ty(), slot_bases,
                _builder.CreateVectorSplat(
                    _width, _builder.getInt64(offset)));
    };

    auto *pointer_type = ::llvm::PointerType::getUnqual(
        _module.getContext());
    auto *pointer_lanes = ::llvm::FixedVectorType::get(
        pointer_type, _width);
    constexpr auto data_offset =
        offsetof(SIMDHostBindlessSlot, buffer) +
        offsetof(SIMDHostBufferView, data);
    auto *data = _builder.CreateMaskedGather(
        pointer_lanes, field_pointers(data_offset),
        ::llvm::Align{alignof(void *)}, _active_mask,
        ::llvm::Constant::getNullValue(pointer_lanes));

    constexpr auto size_offset =
        offsetof(SIMDHostBindlessSlot, buffer) +
        offsetof(SIMDHostBufferView, size_bytes);
    auto *size_bytes = _builder.CreateMaskedGather(
        i64_lanes, field_pointers(size_offset),
        ::llvm::Align{alignof(size_t)}, _active_mask,
        ::llvm::Constant::getNullValue(i64_lanes));
    return {.data = data, .size_bytes = size_bytes};
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_bindless_access_offsets(
    const BindlessBufferLanes &buffer, ::llvm::Value *index,
    uint64_t stride, size_t access_size) {
    auto *i64_lanes = ::llvm::FixedVectorType::get(
        _builder.getInt64Ty(), _width);
    auto *extended = _builder.CreateZExtOrTrunc(index, i64_lanes);
    if (stride != 1u) {
        auto *would_overflow = _builder.CreateICmpUGT(
            extended,
            _builder.CreateVectorSplat(
                _width,
                _builder.getInt64(
                    std::numeric_limits<uint64_t>::max() / stride)));
        _trap_if(
            _builder.CreateOrReduce(
                _builder.CreateAnd(_active_mask, would_overflow)),
            "bindless.buffer.offset_overflow");
        extended = _builder.CreateMul(
            extended,
            _builder.CreateVectorSplat(
                _width, _builder.getInt64(stride)));
    }
    auto *required = _builder.CreateVectorSplat(
        _width, _builder.getInt64(access_size));
    auto *has_space = _builder.CreateICmpUGE(
        buffer.size_bytes, required);
    auto *last_offset = _builder.CreateSelect(
        has_space,
        _builder.CreateSub(buffer.size_bytes, required),
        ::llvm::Constant::getNullValue(i64_lanes));
    auto *in_bounds = _builder.CreateAnd(
        has_space, _builder.CreateICmpULE(extended, last_offset));
    auto *pointer_lanes = ::llvm::cast<::llvm::VectorType>(
        buffer.data->getType());
    auto *is_bound = _builder.CreateICmpNE(
        buffer.data,
        ::llvm::Constant::getNullValue(pointer_lanes));
    auto *invalid = _builder.CreateAnd(
        _active_mask,
        _builder.CreateNot(_builder.CreateAnd(in_bounds, is_bound)));
    _trap_if(
        _builder.CreateOrReduce(invalid),
        "bindless.buffer.invalid_access");
    return extended;
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_bindless_resource_read(
    const schedule::Instruction &instruction) {
    if (!instruction.result || !instruction.source_op ||
        instruction.operands.size() != 3u) {
        _fail("bindless buffer read instruction is malformed");
        return nullptr;
    }
    auto op = static_cast<xir::ResourceReadOp>(
        *instruction.source_op);
    auto byte_address =
        op == xir::ResourceReadOp::BINDLESS_BYTE_BUFFER_READ;
    if (!byte_address &&
        op != xir::ResourceReadOp::BINDLESS_BUFFER_READ) {
        _fail("unsupported bindless resource read operation");
        return nullptr;
    }
    auto *index_value = _source.value(instruction.operands[2u]);
    auto *result = _source.value(*instruction.result);
    auto *index = index_value == nullptr ? nullptr :
        _as_lane_vector(
            _load_value(instruction.operands[2u]), *index_value);
    if (index_value == nullptr || result == nullptr || index == nullptr ||
        index_value->type == nullptr ||
        !index_value->type->is_scalar() ||
        index_value->type->is_float() || !_is_data(result->type) ||
        result->value_class != schedule::ValueClass::varying) {
        _fail("bindless buffer read has invalid index or result type");
        return nullptr;
    }
    auto buffer = _bindless_buffer_lanes(
        instruction.operands[0u], instruction.operands[1u]);
    if (buffer.data == nullptr) { return nullptr; }
    auto stride = byte_address ? 1u : result->type->size();
    auto *offsets = _bindless_access_offsets(
        buffer, index, stride, result->type->size());
    return _gather_data(buffer.data, offsets, result->type);
}

void ScheduleEmitter::_bindless_resource_write(
    const schedule::Instruction &instruction) {
    if (!instruction.source_op || instruction.operands.size() != 4u) {
        _fail("bindless buffer write instruction is malformed");
        return;
    }
    auto op = static_cast<xir::ResourceWriteOp>(
        *instruction.source_op);
    auto byte_address =
        op == xir::ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE;
    if (!byte_address &&
        op != xir::ResourceWriteOp::BINDLESS_BUFFER_WRITE) {
        _fail("unsupported bindless resource write operation");
        return;
    }
    auto *index_value = _source.value(instruction.operands[2u]);
    auto *written_value = _source.value(instruction.operands[3u]);
    auto *index = index_value == nullptr ? nullptr :
        _as_lane_vector(
            _load_value(instruction.operands[2u]), *index_value);
    auto *written = written_value == nullptr ? nullptr :
        _as_lane_vector(
            _load_value(instruction.operands[3u]), *written_value);
    if (index_value == nullptr || written_value == nullptr ||
        index == nullptr || written == nullptr ||
        index_value->type == nullptr ||
        !index_value->type->is_scalar() ||
        index_value->type->is_float() ||
        !_is_data(written_value->type)) {
        _fail("bindless buffer write has invalid index or value type");
        return;
    }
    auto buffer = _bindless_buffer_lanes(
        instruction.operands[0u], instruction.operands[1u]);
    if (buffer.data == nullptr) { return; }
    auto stride = byte_address ? 1u : written_value->type->size();
    auto *offsets = _bindless_access_offsets(
        buffer, index, stride, written_value->type->size());
    _scatter_data(
        buffer.data, offsets, written_value->type, written);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_bindless_resource_query(
    const schedule::Instruction &instruction) {
    if (!instruction.result || !instruction.source_op ||
        instruction.operands.size() < 2u) {
        _fail("bindless buffer query instruction is malformed");
        return nullptr;
    }
    auto op = static_cast<xir::ResourceQueryOp>(
        *instruction.source_op);
    auto size_query = op == xir::ResourceQueryOp::BINDLESS_BUFFER_SIZE;
    auto byte_size_query =
        op == xir::ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE;
    auto address_query =
        op == xir::ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS;
    auto expected_operands = size_query ? 3u : 2u;
    if ((!size_query && !byte_size_query && !address_query) ||
        instruction.operands.size() != expected_operands) {
        _fail("unsupported bindless resource query operation");
        return nullptr;
    }
    auto *result = _source.value(*instruction.result);
    if (result == nullptr || result->type == nullptr ||
        !result->type->is_scalar() || result->type->is_float()) {
        _fail("bindless buffer query requires an integer scalar result");
        return nullptr;
    }
    auto buffer = _bindless_buffer_lanes(
        instruction.operands[0u], instruction.operands[1u]);
    if (buffer.data == nullptr) { return nullptr; }
    auto *i64_lanes = ::llvm::FixedVectorType::get(
        _builder.getInt64Ty(), _width);
    ::llvm::Value *value = nullptr;
    if (address_query) {
        value = _builder.CreatePtrToInt(buffer.data, i64_lanes);
    } else if (byte_size_query) {
        value = buffer.size_bytes;
    } else {
        auto *stride_value = _source.value(instruction.operands[2u]);
        auto *stride = stride_value == nullptr ? nullptr :
            _as_lane_vector(
                _load_value(instruction.operands[2u]), *stride_value);
        if (stride_value == nullptr || stride == nullptr ||
            stride_value->type == nullptr ||
            !stride_value->type->is_scalar() ||
            stride_value->type->is_float()) {
            _fail("bindless buffer size has an invalid stride");
            return nullptr;
        }
        stride = _builder.CreateZExtOrTrunc(stride, i64_lanes);
        auto *zero = ::llvm::Constant::getNullValue(i64_lanes);
        auto *zero_stride = _builder.CreateICmpEQ(stride, zero);
        _trap_if(
            _builder.CreateOrReduce(
                _builder.CreateAnd(_active_mask, zero_stride)),
            "bindless.buffer.zero_stride");
        auto *safe_stride = _builder.CreateSelect(
            zero_stride,
            _builder.CreateVectorSplat(
                _width, _builder.getInt64(1u)),
            stride);
        value = _builder.CreateUDiv(buffer.size_bytes, safe_stride);
    }
    auto *destination = _data_type(result->type, true);
    value = _builder.CreateZExtOrTrunc(value, destination);
    if (result->value_class == schedule::ValueClass::varying) {
        return value;
    }
    return _builder.CreateExtractElement(
        value, _safe_first_lane(_active_mask));
}

}// namespace luisa::compute::simd::detail
