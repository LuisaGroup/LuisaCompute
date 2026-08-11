#include "llvm_schedule_emitter.h"

#include <limits>

namespace luisa::compute::simd::detail {

[[nodiscard]] ScheduleEmitter::BindlessArrayLanes
ScheduleEmitter::_bindless_array_lanes(
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
        _fail("bindless access has invalid array or slot operands");
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

    // An inactive tail lane may carry poison or an arbitrary slot index. It
    // must be made harmless before pointer arithmetic, not merely masked at
    // the eventual gather/callback.
    slot_indices = _builder.CreateSelect(
        _active_mask, slot_indices,
        ::llvm::Constant::getNullValue(i64_lanes),
        "bindless.safe.slot.indices");
    return {
        .view = bindless,
        .slots = slots,
        .slot_count = slot_count,
        .slot_indices = slot_indices,
    };
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_bindless_callback_mask(
    bool varying_result) {
    if (varying_result) {
        auto *packed_mask_type = ::llvm::IntegerType::get(
            _module.getContext(), _width);
        return _builder.CreateZExtOrTrunc(
            _builder.CreateBitCast(_active_mask, packed_mask_type),
            _builder.getInt64Ty());
    }
    auto *first_lane = _builder.CreateZExt(
        _safe_first_lane(_active_mask), _builder.getInt64Ty());
    return _builder.CreateShl(
        _builder.getInt64(1u), first_lane,
        "bindless.uniform.callback.mask");
}

[[nodiscard]] ScheduleEmitter::BindlessBufferLanes
ScheduleEmitter::_bindless_buffer_lanes(
    schedule::ValueId bindless_id, schedule::ValueId slot_id) {
    auto array = _bindless_array_lanes(bindless_id, slot_id);
    if (array.view == nullptr) { return {}; }
    auto *i64_lanes = ::llvm::FixedVectorType::get(
        _builder.getInt64Ty(), _width);

    auto *slot_offsets = _builder.CreateMul(
        array.slot_indices,
        _builder.CreateVectorSplat(
            _width,
            _builder.getInt64(sizeof(SIMDHostBindlessSlot))));
    auto *slot_bases = _builder.CreateGEP(
        _builder.getInt8Ty(),
        _builder.CreateVectorSplat(_width, array.slots), slot_offsets);
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

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_bindless_texture_read(
    const schedule::Instruction &instruction) {
    if (!instruction.result || !instruction.source_op) {
        _fail("bindless texture read instruction is malformed");
        return nullptr;
    }
    auto op = static_cast<xir::ResourceReadOp>(
        *instruction.source_op);
    auto dimension =
        op == xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ ||
                op == xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL ?
            2u :
        op == xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ ||
                op == xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL ?
            3u :
            0u;
    auto has_level =
        op == xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL ||
        op == xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL;
    if (dimension == 0u || instruction.operands.size() !=
                               (has_level ? 4u : 3u)) {
        _fail("unsupported bindless texture read operation");
        return nullptr;
    }
    if (_width > 64u) {
        _fail("bindless texture packet callbacks support widths up to 64 lanes");
        return nullptr;
    }
    auto *result = _source.value(*instruction.result);
    auto *coordinate_value = _source.value(instruction.operands[2u]);
    auto *coordinate = coordinate_value == nullptr ? nullptr :
        _as_lane_vector(
            _load_value(instruction.operands[2u]), *coordinate_value);
    if (result == nullptr || result->type == nullptr ||
        !result->type->is_vector() || result->type->dimension() != 4u ||
        !result->type->element()->is_float32() ||
        coordinate_value == nullptr || coordinate == nullptr ||
        coordinate_value->type == nullptr ||
        !coordinate_value->type->is_vector() ||
        coordinate_value->type->dimension() != dimension ||
        !coordinate_value->type->element()->is_uint32()) {
        _fail("bindless texture read requires uint coordinates and a float4 result");
        return nullptr;
    }
    auto array = _bindless_array_lanes(
        instruction.operands[0u], instruction.operands[1u]);
    if (array.view == nullptr) { return nullptr; }

    auto &context = _module.getContext();
    auto *pointer_type = ::llvm::PointerType::getUnqual(context);
    auto *null_pointer = ::llvm::ConstantPointerNull::get(pointer_type);
    auto *i32_lanes = ::llvm::FixedVectorType::get(
        _builder.getInt32Ty(), _width);
    auto *float_lanes = ::llvm::FixedVectorType::get(
        _builder.getFloatTy(), _width);
    auto *safe_slot_indices = _builder.CreateTrunc(
        array.slot_indices, i32_lanes);
    auto *slot_scratch = _entry_scratch(
        i32_lanes, "bindless.texture.read.slots");
    _builder.CreateStore(safe_slot_indices, slot_scratch);

    std::array<::llvm::AllocaInst *, 3u> coordinate_scratch{};
    auto *zero_i32 = ::llvm::Constant::getNullValue(i32_lanes);
    for (auto axis = uint32_t{0u}; axis < 3u; axis++) {
        auto *lanes = axis < dimension ?
            _extract_child(
                coordinate, coordinate_value->type, axis, true) :
            zero_i32;
        lanes = _builder.CreateSelect(
            _active_mask, lanes, zero_i32,
            "bindless.texture.read.safe.coordinate");
        coordinate_scratch[axis] = _entry_scratch(
            i32_lanes,
            "bindless.texture.read.coordinate." + std::to_string(axis));
        _builder.CreateStore(lanes, coordinate_scratch[axis]);
    }

    ::llvm::Value *level_pointer = null_pointer;
    if (has_level) {
        auto *level_value = _source.value(instruction.operands[3u]);
        auto *level = level_value == nullptr ? nullptr :
            _as_lane_vector(
                _load_value(instruction.operands[3u]), *level_value);
        if (level_value == nullptr || level == nullptr ||
            level_value->type == nullptr ||
            !level_value->type->is_scalar() ||
            level_value->type->is_float()) {
            _fail("bindless texture read has an invalid mip level");
            return nullptr;
        }
        level = _builder.CreateZExtOrTrunc(level, i32_lanes);
        level = _builder.CreateSelect(_active_mask, level, zero_i32);
        auto *level_scratch = _entry_scratch(
            i32_lanes, "bindless.texture.read.levels");
        _builder.CreateStore(level, level_scratch);
        level_pointer = level_scratch;
    }

    auto *scratch_type = ::llvm::ArrayType::get(float_lanes, 4u);
    auto *scratch = _entry_scratch(
        scratch_type,
        "bindless.texture.read.result." +
            std::to_string(instruction.result->value));
    _builder.CreateStore(
        ::llvm::Constant::getNullValue(scratch_type), scratch);
    auto *callback = _builder.CreateExtractValue(array.view, {3u});
    auto *missing_callback = _builder.CreateICmpEQ(
        callback, null_pointer);
    _trap_if(
        _builder.CreateAnd(
            _builder.CreateOrReduce(_active_mask), missing_callback),
        "bindless.texture.read.callback.null");
    auto *active_mask_bits = _bindless_callback_mask(
        result->value_class == schedule::ValueClass::varying);
    auto *callback_type = ::llvm::FunctionType::get(
        _builder.getVoidTy(),
        {pointer_type, _builder.getInt64Ty(),
         _builder.getInt32Ty(), _builder.getInt32Ty(),
         _builder.getInt64Ty(), pointer_type,
         pointer_type, pointer_type, pointer_type,
         pointer_type, pointer_type},
        false);
    _builder.CreateCall(
        callback_type, callback,
        {array.slots, array.slot_count,
         _builder.getInt32(dimension), _builder.getInt32(_width),
         active_mask_bits, slot_scratch,
         coordinate_scratch[0u], coordinate_scratch[1u],
         coordinate_scratch[2u], level_pointer, scratch});

    auto *pixels = static_cast<::llvm::Value *>(
        ::llvm::PoisonValue::get(_data_type(result->type, true)));
    for (auto component = uint32_t{0u}; component < 4u; component++) {
        auto *component_pointer = _builder.CreateGEP(
            scratch_type, scratch,
            {_builder.getInt32(0u), _builder.getInt32(component)});
        pixels = _insert_child(
            pixels, _builder.CreateLoad(float_lanes, component_pointer),
            result->type, component, true);
    }
    return result->value_class == schedule::ValueClass::varying ?
        pixels :
        _extract_lane(pixels, result->type,
                      _safe_first_lane(_active_mask));
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_bindless_texture_query(
    const schedule::Instruction &instruction) {
    if (!instruction.result || !instruction.source_op ||
        instruction.operands.size() < 2u) {
        _fail("bindless texture query instruction is malformed");
        return nullptr;
    }
    auto op = static_cast<xir::ResourceQueryOp>(
        *instruction.source_op);
    auto size_query =
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL;
    auto sample_query = !size_query &&
        op >= xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE &&
        op <= xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER;
    if (!size_query && !sample_query) {
        _fail("unsupported bindless texture query operation");
        return nullptr;
    }
    auto dimension =
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE ||
                op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL ||
                op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE ||
                op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL ||
                op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD ||
                op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL ||
                op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER ||
                op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER ||
                op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER ||
                op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER ?
            2u :
            3u;
    auto has_level =
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER;
    auto has_gradient =
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER;
    auto explicit_sampler =
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER;
    if (has_gradient) {
        _fail("SIMD bindless texture gradient sampling is not implemented yet");
        return nullptr;
    }
    if (_width > 64u) {
        _fail("bindless texture packet callbacks support widths up to 64 lanes");
        return nullptr;
    }

    auto array = _bindless_array_lanes(
        instruction.operands[0u], instruction.operands[1u]);
    if (array.view == nullptr) { return nullptr; }
    auto &context = _module.getContext();
    auto *pointer_type = ::llvm::PointerType::getUnqual(context);
    auto *null_pointer = ::llvm::ConstantPointerNull::get(pointer_type);
    auto *i32_lanes = ::llvm::FixedVectorType::get(
        _builder.getInt32Ty(), _width);
    auto *safe_slot_indices = _builder.CreateTrunc(
        array.slot_indices, i32_lanes);
    auto *slot_scratch = _entry_scratch(
        i32_lanes, "bindless.texture.query.slots");
    _builder.CreateStore(safe_slot_indices, slot_scratch);
    auto *zero_i32 = ::llvm::Constant::getNullValue(i32_lanes);

    if (size_query) {
        auto expected_operands = has_level ? 3u : 2u;
        auto *result = _source.value(*instruction.result);
        if (instruction.operands.size() != expected_operands ||
            result == nullptr || result->type == nullptr ||
            !result->type->is_vector() ||
            result->type->dimension() != dimension ||
            !result->type->element()->is_uint32()) {
            _fail("bindless texture size query has invalid operands or result type");
            return nullptr;
        }
        ::llvm::Value *level_pointer = null_pointer;
        if (has_level) {
            auto *level_value = _source.value(instruction.operands[2u]);
            auto *level = level_value == nullptr ? nullptr :
                _as_lane_vector(
                    _load_value(instruction.operands[2u]), *level_value);
            if (level_value == nullptr || level == nullptr ||
                level_value->type == nullptr ||
                !level_value->type->is_scalar() ||
                level_value->type->is_float()) {
                _fail("bindless texture size query has an invalid mip level");
                return nullptr;
            }
            level = _builder.CreateZExtOrTrunc(level, i32_lanes);
            level = _builder.CreateSelect(_active_mask, level, zero_i32);
            auto *level_scratch = _entry_scratch(
                i32_lanes, "bindless.texture.size.levels");
            _builder.CreateStore(level, level_scratch);
            level_pointer = level_scratch;
        }
        auto *scratch_type = ::llvm::ArrayType::get(i32_lanes, 3u);
        auto *scratch = _entry_scratch(
            scratch_type,
            "bindless.texture.size.result." +
                std::to_string(instruction.result->value));
        _builder.CreateStore(
            ::llvm::Constant::getNullValue(scratch_type), scratch);
        auto *callback = _builder.CreateExtractValue(array.view, {4u});
        auto *missing_callback = _builder.CreateICmpEQ(
            callback, null_pointer);
        _trap_if(
            _builder.CreateAnd(
                _builder.CreateOrReduce(_active_mask), missing_callback),
            "bindless.texture.size.callback.null");
        auto *callback_type = ::llvm::FunctionType::get(
            _builder.getVoidTy(),
            {pointer_type, _builder.getInt64Ty(),
             _builder.getInt32Ty(), _builder.getInt32Ty(),
             _builder.getInt64Ty(), pointer_type,
             pointer_type, pointer_type},
            false);
        auto *active_mask_bits = _bindless_callback_mask(
            result->value_class == schedule::ValueClass::varying);
        _builder.CreateCall(
            callback_type, callback,
            {array.slots, array.slot_count,
             _builder.getInt32(dimension), _builder.getInt32(_width),
             active_mask_bits, slot_scratch, level_pointer, scratch});
        auto *sizes = static_cast<::llvm::Value *>(
            ::llvm::PoisonValue::get(_data_type(result->type, true)));
        for (auto axis = uint32_t{0u}; axis < dimension; axis++) {
            auto *axis_pointer = _builder.CreateGEP(
                scratch_type, scratch,
                {_builder.getInt32(0u), _builder.getInt32(axis)});
            sizes = _insert_child(
                sizes, _builder.CreateLoad(i32_lanes, axis_pointer),
                result->type, axis, true);
        }
        return result->value_class == schedule::ValueClass::varying ?
            sizes :
            _extract_lane(sizes, result->type,
                          _safe_first_lane(_active_mask));
    }

    auto expected_operands =
        has_level ? (explicit_sampler ? 6u : 4u) :
                    (explicit_sampler ? 5u : 3u);
    auto *result = _source.value(*instruction.result);
    auto *coordinate_value = _source.value(instruction.operands[2u]);
    auto *coordinate = coordinate_value == nullptr ? nullptr :
        _as_lane_vector(
            _load_value(instruction.operands[2u]), *coordinate_value);
    if (instruction.operands.size() != expected_operands ||
        result == nullptr || result->type == nullptr ||
        !result->type->is_vector() || result->type->dimension() != 4u ||
        !result->type->element()->is_float32() ||
        coordinate_value == nullptr || coordinate == nullptr ||
        coordinate_value->type == nullptr ||
        !coordinate_value->type->is_vector() ||
        coordinate_value->type->dimension() != dimension ||
        !coordinate_value->type->element()->is_float32()) {
        _fail("bindless texture sample has invalid operands or result type");
        return nullptr;
    }

    auto *float_lanes = ::llvm::FixedVectorType::get(
        _builder.getFloatTy(), _width);
    auto *zero_float = ::llvm::Constant::getNullValue(float_lanes);
    std::array<::llvm::AllocaInst *, 3u> coordinate_scratch{};
    for (auto axis = uint32_t{0u}; axis < 3u; axis++) {
        auto *lanes = axis < dimension ?
            _extract_child(
                coordinate, coordinate_value->type, axis, true) :
            zero_float;
        lanes = _builder.CreateSelect(
            _active_mask, lanes, zero_float,
            "bindless.texture.sample.safe.coordinate");
        coordinate_scratch[axis] = _entry_scratch(
            float_lanes,
            "bindless.texture.sample.coordinate." +
                std::to_string(axis));
        _builder.CreateStore(lanes, coordinate_scratch[axis]);
    }

    ::llvm::Value *level_pointer = null_pointer;
    auto sampler_operand = size_t{3u};
    if (has_level) {
        auto *level_value = _source.value(instruction.operands[3u]);
        auto *level = level_value == nullptr ? nullptr :
            _as_lane_vector(
                _load_value(instruction.operands[3u]), *level_value);
        if (level_value == nullptr || level == nullptr ||
            level_value->type == nullptr ||
            !level_value->type->is_float32()) {
            _fail("bindless texture sample has an invalid mip level");
            return nullptr;
        }
        level = _builder.CreateSelect(_active_mask, level, zero_float);
        auto *level_scratch = _entry_scratch(
            float_lanes, "bindless.texture.sample.levels");
        _builder.CreateStore(level, level_scratch);
        level_pointer = level_scratch;
        sampler_operand = 4u;
    }

    ::llvm::Value *sampler_pointer = null_pointer;
    if (explicit_sampler) {
        auto *filter_value = _source.value(
            instruction.operands[sampler_operand]);
        auto *address_value = _source.value(
            instruction.operands[sampler_operand + 1u]);
        auto *filter = filter_value == nullptr ? nullptr :
            _as_lane_vector(
                _load_value(instruction.operands[sampler_operand]),
                *filter_value);
        auto *address = address_value == nullptr ? nullptr :
            _as_lane_vector(
                _load_value(instruction.operands[sampler_operand + 1u]),
                *address_value);
        if (filter_value == nullptr || address_value == nullptr ||
            filter == nullptr || address == nullptr ||
            filter_value->type == nullptr ||
            address_value->type == nullptr ||
            !filter_value->type->is_scalar() ||
            !address_value->type->is_scalar() ||
            filter_value->type->is_float() ||
            address_value->type->is_float()) {
            _fail("bindless texture sample has an invalid explicit sampler");
            return nullptr;
        }
        filter = _builder.CreateZExtOrTrunc(filter, i32_lanes);
        address = _builder.CreateZExtOrTrunc(address, i32_lanes);
        auto *sampler = _builder.CreateOr(
            _builder.CreateShl(
                filter,
                _builder.CreateVectorSplat(
                    _width, _builder.getInt32(2u))),
            address);
        sampler = _builder.CreateSelect(
            _active_mask, sampler, zero_i32);
        auto *sampler_scratch = _entry_scratch(
            i32_lanes, "bindless.texture.sample.samplers");
        _builder.CreateStore(sampler, sampler_scratch);
        sampler_pointer = sampler_scratch;
    }

    auto *scratch_type = ::llvm::ArrayType::get(float_lanes, 4u);
    auto *scratch = _entry_scratch(
        scratch_type,
        "bindless.texture.sample.result." +
            std::to_string(instruction.result->value));
    _builder.CreateStore(
        ::llvm::Constant::getNullValue(scratch_type), scratch);
    auto *callback = _builder.CreateExtractValue(array.view, {2u});
    auto *missing_callback = _builder.CreateICmpEQ(
        callback, null_pointer);
    _trap_if(
        _builder.CreateAnd(
            _builder.CreateOrReduce(_active_mask), missing_callback),
        "bindless.texture.sample.callback.null");
    auto *callback_type = ::llvm::FunctionType::get(
        _builder.getVoidTy(),
        {pointer_type, _builder.getInt64Ty(),
         _builder.getInt32Ty(), _builder.getInt32Ty(),
         _builder.getInt64Ty(), pointer_type, pointer_type,
         pointer_type, pointer_type, pointer_type,
         pointer_type, pointer_type},
        false);
    auto *active_mask_bits = _bindless_callback_mask(
        result->value_class == schedule::ValueClass::varying);
    _builder.CreateCall(
        callback_type, callback,
        {array.slots, array.slot_count,
         _builder.getInt32(dimension), _builder.getInt32(_width),
         active_mask_bits, slot_scratch, sampler_pointer,
         coordinate_scratch[0u], coordinate_scratch[1u],
         coordinate_scratch[2u], level_pointer, scratch});
    auto *pixels = static_cast<::llvm::Value *>(
        ::llvm::PoisonValue::get(_data_type(result->type, true)));
    for (auto component = uint32_t{0u}; component < 4u; component++) {
        auto *component_pointer = _builder.CreateGEP(
            scratch_type, scratch,
            {_builder.getInt32(0u), _builder.getInt32(component)});
        pixels = _insert_child(
            pixels, _builder.CreateLoad(float_lanes, component_pointer),
            result->type, component, true);
    }
    return result->value_class == schedule::ValueClass::varying ?
        pixels :
        _extract_lane(pixels, result->type,
                      _safe_first_lane(_active_mask));
}

}// namespace luisa::compute::simd::detail
