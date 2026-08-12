#include "llvm_schedule_emitter.h"

#include <array>

namespace luisa::compute::simd::detail {

namespace {

[[nodiscard]] bool is_float_array(
    const Type *type, uint32_t dimension) noexcept {
    return type != nullptr && type->is_array() &&
           type->dimension() == dimension &&
           type->element()->is_float32();
}

[[nodiscard]] bool is_ray_type(const Type *type) noexcept {
    return type != nullptr && type->is_structure() &&
           type->members().size() == 4u &&
           is_float_array(type->members()[0u], 3u) &&
           type->members()[1u]->is_float32() &&
           is_float_array(type->members()[2u], 3u) &&
           type->members()[3u]->is_float32();
}

[[nodiscard]] bool is_surface_hit_type(const Type *type) noexcept {
    return type != nullptr && type->is_structure() &&
           type->members().size() == 4u &&
           type->members()[0u]->is_uint32() &&
           type->members()[1u]->is_uint32() &&
           type->members()[2u]->is_vector() &&
           type->members()[2u]->dimension() == 2u &&
           type->members()[2u]->element()->is_float32() &&
           type->members()[3u]->is_float32();
}

[[nodiscard]] bool is_float4x4_type(const Type *type) noexcept {
    return type != nullptr && type->is_matrix() &&
           type->dimension() == 4u &&
           type->element()->is_float32();
}

[[nodiscard]] bool is_motion_srt_type(const Type *type) noexcept {
    if (type == nullptr || !type->is_structure() ||
        type->members().size() != 5u) {
        return false;
    }
    constexpr std::array dimensions{3u, 4u, 3u, 3u, 3u};
    for (auto i = size_t{0u}; i < dimensions.size(); i++) {
        if (!is_float_array(type->members()[i], dimensions[i])) {
            return false;
        }
    }
    return true;
}

}// namespace

[[nodiscard]] ScheduleEmitter::AccelInstanceAddress
ScheduleEmitter::_accel_instance_address(
    ::llvm::Value *accel, schedule::ValueId index_id,
    bool varying) {
    auto *index_value = _source.value(index_id);
    auto *raw_index = _load_value(index_id);
    if (accel == nullptr || index_value == nullptr ||
        raw_index == nullptr || index_value->type == nullptr ||
        (!index_value->type->is_int32() &&
         !index_value->type->is_uint32())) {
        _fail("acceleration-structure instance index is invalid");
        return {};
    }
    auto &context = _module.getContext();
    auto *pointer_type = ::llvm::PointerType::getUnqual(context);
    auto *null_pointer =
        ::llvm::ConstantPointerNull::get(pointer_type);
    auto *has_active = _builder.CreateOrReduce(_active_mask);
    auto *table = _builder.CreateExtractValue(accel, {3u});
    _trap_if(
        _builder.CreateAnd(
            has_active,
            _builder.CreateICmpEQ(table, null_pointer)),
        "accel.instance.table.null");
    auto *data = _builder.CreateLoad(pointer_type, table);
    data->setAlignment(::llvm::Align{alignof(void *)});
    auto *size_pointer = _byte_pointer(
        table, offsetof(SIMDHostAccelInstanceTable, size));
    auto *size = _builder.CreateLoad(
        _builder.getInt64Ty(), size_pointer);
    size->setAlignment(::llvm::Align{alignof(size_t)});
    _trap_if(
        _builder.CreateAnd(
            has_active,
            _builder.CreateICmpEQ(data, null_pointer)),
        "accel.instance.data.null");

    if (varying) {
        auto *index = _as_lane_vector(raw_index, *index_value);
        if (index == nullptr) { return {}; }
        auto *i32_lanes = ::llvm::FixedVectorType::get(
            _builder.getInt32Ty(), _width);
        auto *i64_lanes = ::llvm::FixedVectorType::get(
            _builder.getInt64Ty(), _width);
        index = _builder.CreateZExtOrTrunc(index, i32_lanes);
        index = _builder.CreateSelect(
            _active_mask, index,
            ::llvm::Constant::getNullValue(i32_lanes),
            "accel.safe.instance.index");
        auto *wide_index = _builder.CreateZExt(index, i64_lanes);
        auto *invalid = _builder.CreateAnd(
            _active_mask,
            _builder.CreateICmpUGE(
                wide_index,
                _builder.CreateVectorSplat(_width, size)));
        _trap_if(
            _builder.CreateOrReduce(invalid),
            "accel.instance.index.out.of.bounds");
        auto *offsets = _builder.CreateMul(
            wide_index,
            _builder.CreateVectorSplat(
                _width,
                _builder.getInt64(sizeof(SIMDHostAccelInstance))));
        return {
            .data = data,
            .offsets = offsets,
        };
    }

    auto *index = _builder.CreateZExtOrTrunc(
        raw_index, _builder.getInt64Ty());
    auto *invalid = _builder.CreateICmpUGE(index, size);
    _trap_if(
        _builder.CreateAnd(has_active, invalid),
        "accel.instance.index.out.of.bounds");
    auto *offset = _builder.CreateMul(
        index,
        _builder.getInt64(sizeof(SIMDHostAccelInstance)));
    return {
        .data = data,
        .scalar = _builder.CreateGEP(
            _builder.getInt8Ty(), data, offset),
    };
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_accel_instance_query(
    const schedule::Instruction &instruction) {
    if (!instruction.result || !instruction.source_op ||
        instruction.operands.size() != 2u) {
        _fail("acceleration-structure instance query is malformed");
        return nullptr;
    }
    auto op = static_cast<xir::ResourceQueryOp>(
        *instruction.source_op);
    auto transform =
        op == xir::ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM;
    auto user_id =
        op == xir::ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID;
    auto visibility =
        op == xir::ResourceQueryOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK;
    if (!transform && !user_id && !visibility) {
        _fail("unsupported acceleration-structure instance query");
        return nullptr;
    }

    auto *result = _source.value(*instruction.result);
    auto *accel_value = _source.value(instruction.operands[0u]);
    auto *index_value = _source.value(instruction.operands[1u]);
    auto *accel = _load_value(instruction.operands[0u]);
    if (result == nullptr || accel_value == nullptr ||
        index_value == nullptr || accel == nullptr ||
        accel_value->type == nullptr || !accel_value->type->is_accel() ||
        (transform ? !is_float4x4_type(result->type) :
                     !result->type->is_uint32())) {
        _fail("acceleration-structure instance query has invalid types");
        return nullptr;
    }

    auto varying =
        result->value_class == schedule::ValueClass::varying;
    if (!varying && !schedule::is_uniform(index_value->value_class)) {
        _fail("uniform instance query requires a uniform instance index");
        return nullptr;
    }
    auto address = _accel_instance_address(
        accel, instruction.operands[1u], varying);
    if (address.data == nullptr) { return nullptr; }
    auto *data = address.data;
    auto *instance_offsets = address.offsets;
    auto *scalar_instance = address.scalar;

    auto gather = [&](const Type *type, size_t offset) {
        if (varying) {
            return _gather_data(
                data, instance_offsets, type, offset);
        }
        auto *pointer = _byte_pointer(scalar_instance, offset);
        auto *load = _builder.CreateLoad(
            _data_type(type, false), pointer,
            "accel.instance.scalar.load");
        load->setAlignment(::llvm::Align{type->alignment()});
        return static_cast<::llvm::Value *>(load);
    };
    if (user_id) {
        return gather(
            result->type,
            offsetof(SIMDHostAccelInstance, user_id));
    }
    if (visibility) {
        constexpr auto mask_offset =
            offsetof(SIMDHostAccelInstance, mask);
        ::llvm::Value *stored = nullptr;
        if (varying) {
            auto *pointers = _leaf_pointers(
                data, instance_offsets, mask_offset);
            auto *i8_lanes = ::llvm::FixedVectorType::get(
                _builder.getInt8Ty(), _width);
            stored = _builder.CreateMaskedGather(
                i8_lanes, pointers, ::llvm::Align{1u},
                _active_mask,
                ::llvm::Constant::getNullValue(i8_lanes));
        } else {
            auto *pointer = _byte_pointer(
                scalar_instance, mask_offset);
            auto *load = _builder.CreateLoad(
                _builder.getInt8Ty(), pointer);
            load->setAlignment(::llvm::Align{1u});
            stored = load;
        }
        auto *destination = varying ?
                                static_cast<::llvm::Type *>(::llvm::FixedVectorType::get(
                                    _builder.getInt32Ty(), _width)) :
                                _builder.getInt32Ty();
        return _builder.CreateZExt(stored, destination);
    }

    auto gather_float = [&](size_t offset) {
        if (varying) {
            auto *pointers = _leaf_pointers(
                data, instance_offsets, offset);
            auto *float_lanes = ::llvm::FixedVectorType::get(
                _builder.getFloatTy(), _width);
            return static_cast<::llvm::Value *>(
                _builder.CreateMaskedGather(
                    float_lanes, pointers, ::llvm::Align{alignof(float)},
                    _active_mask,
                    ::llvm::Constant::getNullValue(float_lanes)));
        }
        auto *pointer = _byte_pointer(scalar_instance, offset);
        auto *load = _builder.CreateLoad(
            _builder.getFloatTy(), pointer,
            "accel.instance.scalar.load");
        load->setAlignment(::llvm::Align{alignof(float)});
        return static_cast<::llvm::Value *>(load);
    };
    auto *zero = ::llvm::ConstantFP::get(
        _builder.getFloatTy(), 0.0);
    auto *one = ::llvm::ConstantFP::get(
        _builder.getFloatTy(), 1.0);
    return _assemble(
        result->type, varying, [&](uint32_t column) {
            auto *column_type = _child_type(
                result->type, column);
            return _assemble(
                column_type, varying, [&](uint32_t row) {
                    if (row < 3u) {
                        auto component = row * 4u + column;
                        return gather_float(
                            offsetof(SIMDHostAccelInstance, affine) +
                            component * sizeof(float));
                    }
                    auto *constant = column == 3u ? one : zero;
                    return varying ?
                               static_cast<::llvm::Value *>(
                                   _builder.CreateVectorSplat(
                                       _width, constant)) :
                               constant;
                });
        });
}

void ScheduleEmitter::_accel_instance_write(
    const schedule::Instruction &instruction) {
    if (!instruction.source_op || instruction.operands.size() != 3u) {
        _fail("acceleration-structure instance write is malformed");
        return;
    }
    auto op = static_cast<xir::ResourceWriteOp>(
        *instruction.source_op);
    auto transform =
        op == xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_TRANSFORM;
    auto visibility =
        op == xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_VISIBILITY_MASK;
    auto user_id =
        op == xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID;
    auto opacity =
        op == xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_OPACITY;
    if (!transform && !visibility && !user_id && !opacity) {
        _fail("unsupported acceleration-structure instance write");
        return;
    }

    auto *accel_value = _source.value(instruction.operands[0u]);
    auto *index_value = _source.value(instruction.operands[1u]);
    auto *written_value = _source.value(instruction.operands[2u]);
    auto *accel = _load_value(instruction.operands[0u]);
    auto *value = _load_value(instruction.operands[2u]);
    if (accel_value == nullptr || index_value == nullptr ||
        written_value == nullptr || accel == nullptr || value == nullptr ||
        accel_value->type == nullptr || !accel_value->type->is_accel() ||
        (transform ? !is_float4x4_type(written_value->type) :
         opacity   ? !written_value->type->is_bool() :
                     !written_value->type->is_uint32())) {
        _fail("acceleration-structure instance write has invalid types");
        return;
    }
    auto varying =
        !schedule::is_uniform(index_value->value_class) ||
        !schedule::is_uniform(written_value->value_class);
    if (varying) {
        value = _as_lane_vector(value, *written_value);
        auto *zero = value == nullptr ? nullptr :
                                        ::llvm::Constant::getNullValue(
                                            value->getType());
        value = zero == nullptr ? nullptr :
                                  _masked_merge(
                                      value, zero,
                                      written_value->type, _active_mask);
        if (value == nullptr) { return; }
    }
    auto address = _accel_instance_address(
        accel, instruction.operands[1u], varying);
    if (address.data == nullptr) { return; }

    auto store = [&](::llvm::Value *stored, size_t offset,
                     size_t alignment) {
        if (varying) {
            auto *pointers = _leaf_pointers(
                address.data, address.offsets, offset);
            _builder.CreateMaskedScatter(
                stored, pointers, ::llvm::Align{alignment},
                _active_mask);
            return;
        }
        auto *pointer = _byte_pointer(address.scalar, offset);
        if (auto *pointer_instruction =
                ::llvm::dyn_cast<::llvm::Instruction>(pointer)) {
            pointer_instruction->setName(
                "accel.instance.scalar.store");
        }
        auto *write = _builder.CreateStore(stored, pointer);
        write->setAlignment(::llvm::Align{alignment});
    };

    if (transform) {
        for (auto row = uint32_t{0u}; row < 3u; row++) {
            for (auto column = uint32_t{0u}; column < 4u; column++) {
                auto *column_type = _child_type(
                    written_value->type, column);
                auto *column_value = _extract_child(
                    value, written_value->type, column, varying);
                auto *component = _extract_child(
                    column_value, column_type, row, varying);
                store(
                    component,
                    offsetof(SIMDHostAccelInstance, affine) +
                        (row * 4u + column) * sizeof(float),
                    alignof(float));
            }
        }
    } else if (visibility) {
        auto *i8 = varying ?
                       static_cast<::llvm::Type *>(::llvm::FixedVectorType::get(
                           _builder.getInt8Ty(), _width)) :
                       _builder.getInt8Ty();
        store(
            _builder.CreateTrunc(value, i8),
            offsetof(SIMDHostAccelInstance, mask),
            alignof(uint8_t));
    } else if (opacity) {
        auto *i8 = varying ?
                       static_cast<::llvm::Type *>(::llvm::FixedVectorType::get(
                           _builder.getInt8Ty(), _width)) :
                       _builder.getInt8Ty();
        auto *byte = _builder.CreateZExt(
            value, i8, "accel.instance.opacity.byte");
        store(
            byte, offsetof(SIMDHostAccelInstance, opaque),
            alignof(uint8_t));
    } else {
        store(
            value, offsetof(SIMDHostAccelInstance, user_id),
            alignof(uint32_t));
    }

    auto *dirty = varying ?
                      static_cast<::llvm::Value *>(
                          _builder.CreateVectorSplat(
                              _width, _builder.getInt8(1u))) :
                      _builder.getInt8(1u);
    store(
        dirty, offsetof(SIMDHostAccelInstance, dirty),
        alignof(uint8_t));
}

[[nodiscard]] ScheduleEmitter::AccelMotionAddress
ScheduleEmitter::_accel_motion_address(
    ::llvm::Value *accel, schedule::ValueId instance_id,
    schedule::ValueId keyframe_id, bool varying,
    uint32_t expected_mode) {
    auto *instance_value = _source.value(instance_id);
    auto *keyframe_value = _source.value(keyframe_id);
    auto *raw_keyframe = _load_value(keyframe_id);
    if (instance_value == nullptr || keyframe_value == nullptr ||
        raw_keyframe == nullptr ||
        keyframe_value->type == nullptr ||
        (!keyframe_value->type->is_int32() &&
         !keyframe_value->type->is_uint32())) {
        _fail("acceleration-structure motion keyframe index is invalid");
        return {};
    }
    auto varying_instance = varying &&
                            !schedule::is_uniform(
                                instance_value->value_class);
    auto instance = _accel_instance_address(
        accel, instance_id, varying_instance);
    if (instance.data == nullptr) { return {}; }

    auto &context = _module.getContext();
    auto *pointer_type = ::llvm::PointerType::getUnqual(context);
    auto *has_active = _builder.CreateOrReduce(_active_mask);
    auto load_field = [&](::llvm::Type *type, size_t offset,
                          size_t alignment) {
        if (varying_instance) {
            auto *pointers = _leaf_pointers(
                instance.data, instance.offsets, offset);
            auto *lanes = ::llvm::FixedVectorType::get(type, _width);
            return static_cast<::llvm::Value *>(
                _builder.CreateMaskedGather(
                    lanes, pointers, ::llvm::Align{alignment},
                    _active_mask,
                    ::llvm::Constant::getNullValue(lanes)));
        }
        auto *pointer = _byte_pointer(instance.scalar, offset);
        auto *value = _builder.CreateLoad(type, pointer);
        value->setAlignment(::llvm::Align{alignment});
        auto *loaded = static_cast<::llvm::Value *>(value);
        return varying ? _builder.CreateVectorSplat(_width, loaded) :
                         loaded;
    };
    auto *frames = load_field(
        pointer_type,
        offsetof(SIMDHostAccelInstance, motion_frames),
        alignof(void *));
    auto *count = load_field(
        _builder.getInt32Ty(),
        offsetof(SIMDHostAccelInstance, motion_keyframe_count),
        alignof(uint32_t));
    auto *mode = load_field(
        _builder.getInt32Ty(),
        offsetof(SIMDHostAccelInstance, motion_mode),
        alignof(uint32_t));

    if (varying) {
        auto *pointer_lanes = ::llvm::FixedVectorType::get(
            pointer_type, _width);
        auto *i32_lanes = ::llvm::FixedVectorType::get(
            _builder.getInt32Ty(), _width);
        auto *i64_lanes = ::llvm::FixedVectorType::get(
            _builder.getInt64Ty(), _width);
        auto *null_frames = _builder.CreateICmpEQ(
            frames, ::llvm::Constant::getNullValue(pointer_lanes));
        auto *wrong_mode = _builder.CreateICmpNE(
            mode, _builder.CreateVectorSplat(
                      _width, _builder.getInt32(expected_mode)));
        _trap_if(
            _builder.CreateOrReduce(
                _builder.CreateAnd(
                    _active_mask,
                    _builder.CreateOr(null_frames, wrong_mode))),
            "accel.motion.metadata.invalid");

        auto *keyframe = _as_lane_vector(
            raw_keyframe, *keyframe_value);
        if (keyframe == nullptr) { return {}; }
        keyframe = _builder.CreateZExtOrTrunc(
            keyframe, i32_lanes);
        keyframe = _builder.CreateSelect(
            _active_mask, keyframe,
            ::llvm::Constant::getNullValue(i32_lanes),
            "accel.safe.motion.keyframe");
        auto *invalid = _builder.CreateAnd(
            _active_mask,
            _builder.CreateICmpUGE(keyframe, count));
        _trap_if(
            _builder.CreateOrReduce(invalid),
            "accel.motion.keyframe.out.of.bounds");
        auto *offsets = _builder.CreateMul(
            _builder.CreateZExt(keyframe, i64_lanes),
            _builder.CreateVectorSplat(
                _width,
                _builder.getInt64(simd_host_accel_motion_frame_size)));
        return {
            .instance = instance,
            .frame = _builder.CreateGEP(
                _builder.getInt8Ty(), frames, offsets),
        };
    }

    auto *null_pointer =
        ::llvm::ConstantPointerNull::get(pointer_type);
    auto *metadata_invalid = _builder.CreateOr(
        _builder.CreateICmpEQ(frames, null_pointer),
        _builder.CreateICmpNE(
            mode, _builder.getInt32(expected_mode)));
    _trap_if(
        _builder.CreateAnd(has_active, metadata_invalid),
        "accel.motion.metadata.invalid");
    auto *keyframe = _builder.CreateZExtOrTrunc(
        raw_keyframe, _builder.getInt32Ty());
    _trap_if(
        _builder.CreateAnd(
            has_active,
            _builder.CreateICmpUGE(keyframe, count)),
        "accel.motion.keyframe.out.of.bounds");
    auto *offset = _builder.CreateMul(
        _builder.CreateZExt(keyframe, _builder.getInt64Ty()),
        _builder.getInt64(simd_host_accel_motion_frame_size));
    return {
        .instance = instance,
        .frame = _builder.CreateGEP(
            _builder.getInt8Ty(), frames, offset,
            "accel.motion.scalar.frame"),
    };
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_accel_motion_query(
    const schedule::Instruction &instruction) {
    if (!instruction.result || !instruction.source_op ||
        instruction.operands.size() != 3u) {
        _fail("acceleration-structure motion query is malformed");
        return nullptr;
    }
    auto op = static_cast<xir::ResourceQueryOp>(
        *instruction.source_op);
    auto matrix =
        op == xir::ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_MATRIX;
    auto srt =
        op == xir::ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_SRT;
    auto *result = _source.value(*instruction.result);
    auto *accel_value = _source.value(instruction.operands[0u]);
    auto *instance_value = _source.value(instruction.operands[1u]);
    auto *keyframe_value = _source.value(instruction.operands[2u]);
    auto *accel = _load_value(instruction.operands[0u]);
    if ((!matrix && !srt) || result == nullptr || accel_value == nullptr ||
        instance_value == nullptr || keyframe_value == nullptr ||
        accel == nullptr || accel_value->type == nullptr ||
        !accel_value->type->is_accel() ||
        (matrix ? !is_float4x4_type(result->type) :
                  !is_motion_srt_type(result->type))) {
        _fail("acceleration-structure motion query has invalid types");
        return nullptr;
    }
    auto varying =
        result->value_class == schedule::ValueClass::varying;
    if (!varying &&
        (!schedule::is_uniform(instance_value->value_class) ||
         !schedule::is_uniform(keyframe_value->value_class))) {
        _fail("uniform motion query requires uniform indices");
        return nullptr;
    }
    auto address = _accel_motion_address(
        accel, instruction.operands[1u], instruction.operands[2u],
        varying,
        static_cast<uint32_t>(
            matrix ? SIMDHostAccelMotionMode::matrix :
                     SIMDHostAccelMotionMode::srt));
    if (address.frame == nullptr) { return nullptr; }
    if (varying) {
        auto *offsets = ::llvm::Constant::getNullValue(
            ::llvm::FixedVectorType::get(
                _builder.getInt64Ty(), _width));
        return _gather_data(
            address.frame, offsets, result->type);
    }
    return _load_uniform_data(address.frame, result->type);
}

void ScheduleEmitter::_accel_motion_write(
    const schedule::Instruction &instruction) {
    if (!instruction.source_op || instruction.operands.size() != 4u) {
        _fail("acceleration-structure motion write is malformed");
        return;
    }
    auto op = static_cast<xir::ResourceWriteOp>(
        *instruction.source_op);
    auto matrix =
        op == xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX;
    auto srt =
        op == xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT;
    auto *accel_value = _source.value(instruction.operands[0u]);
    auto *instance_value = _source.value(instruction.operands[1u]);
    auto *keyframe_value = _source.value(instruction.operands[2u]);
    auto *written_value = _source.value(instruction.operands[3u]);
    auto *accel = _load_value(instruction.operands[0u]);
    auto *value = _load_value(instruction.operands[3u]);
    if ((!matrix && !srt) || accel_value == nullptr ||
        instance_value == nullptr || keyframe_value == nullptr ||
        written_value == nullptr || accel == nullptr || value == nullptr ||
        accel_value->type == nullptr || !accel_value->type->is_accel() ||
        (matrix ? !is_float4x4_type(written_value->type) :
                  !is_motion_srt_type(written_value->type))) {
        _fail("acceleration-structure motion write has invalid types");
        return;
    }
    auto varying =
        !schedule::is_uniform(instance_value->value_class) ||
        !schedule::is_uniform(keyframe_value->value_class) ||
        !schedule::is_uniform(written_value->value_class);
    if (varying) {
        value = _as_lane_vector(value, *written_value);
        auto *zero = value == nullptr ? nullptr :
                                        ::llvm::Constant::getNullValue(
                                            value->getType());
        value = zero == nullptr ? nullptr :
                                  _masked_merge(
                                      value, zero,
                                      written_value->type, _active_mask);
        if (value == nullptr) { return; }
    }
    auto address = _accel_motion_address(
        accel, instruction.operands[1u], instruction.operands[2u],
        varying,
        static_cast<uint32_t>(
            matrix ? SIMDHostAccelMotionMode::matrix :
                     SIMDHostAccelMotionMode::srt));
    if (address.frame == nullptr) { return; }
    if (varying) {
        auto *offsets = ::llvm::Constant::getNullValue(
            ::llvm::FixedVectorType::get(
                _builder.getInt64Ty(), _width));
        _scatter_data(
            address.frame, offsets,
            written_value->type, value);
        if (schedule::is_uniform(instance_value->value_class)) {
            auto scalar_instance = _accel_instance_address(
                accel, instruction.operands[1u], false);
            if (scalar_instance.scalar == nullptr) { return; }
            auto *dirty_pointer = _byte_pointer(
                scalar_instance.scalar,
                offsetof(SIMDHostAccelInstance, dirty));
            auto *dirty = _builder.CreateStore(
                _builder.getInt8(1u), dirty_pointer);
            dirty->setAlignment(::llvm::Align{alignof(uint8_t)});
        } else {
            auto *dirty_pointers = _leaf_pointers(
                address.instance.data, address.instance.offsets,
                offsetof(SIMDHostAccelInstance, dirty));
            _builder.CreateMaskedScatter(
                _builder.CreateVectorSplat(
                    _width, _builder.getInt8(1u)),
                dirty_pointers, ::llvm::Align{alignof(uint8_t)},
                _active_mask);
        }
        return;
    }

    std::function<void(
        ::llvm::Value *, const Type *, ::llvm::Value *, size_t)>
        store_uniform;
    store_uniform = [&](::llvm::Value *base, const Type *type,
                        ::llvm::Value *stored, size_t offset) {
        if (_is_scalar_data(type)) {
            auto *pointer = _byte_pointer(base, offset);
            if (type->is_bool()) {
                stored = _builder.CreateZExt(
                    stored, _builder.getInt8Ty());
            }
            auto *write = _builder.CreateStore(stored, pointer);
            write->setAlignment(::llvm::Align{type->alignment()});
            return;
        }
        for (auto i = uint32_t{0u}; i < _child_count(type); i++) {
            store_uniform(
                base, _child_type(type, i),
                _extract_child(stored, type, i, false),
                offset + _child_offset(type, i));
        }
    };
    store_uniform(
        address.frame, written_value->type, value, 0u);
    auto *dirty_pointer = _byte_pointer(
        address.instance.scalar,
        offsetof(SIMDHostAccelInstance, dirty));
    auto *dirty = _builder.CreateStore(
        _builder.getInt8(1u), dirty_pointer);
    dirty->setAlignment(::llvm::Align{alignof(uint8_t)});
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_accel_query(
    const schedule::Instruction &instruction) {
    if (!instruction.result || !instruction.source_op) {
        _fail("acceleration-structure query instruction is malformed");
        return nullptr;
    }
    if (_width > 16u) {
        _fail("Embree packet callbacks support SIMD widths up to 16");
        return nullptr;
    }
    auto op = static_cast<xir::ResourceQueryOp>(
        *instruction.source_op);
    auto closest =
        op == xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST ||
        op == xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR;
    auto any =
        op == xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY ||
        op == xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR;
    auto motion =
        op == xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR ||
        op == xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR;
    auto expected_operand_count = motion ? 4u : 3u;
    if ((!closest && !any) ||
        instruction.operands.size() != expected_operand_count) {
        _fail("unsupported SIMD acceleration-structure query");
        return nullptr;
    }

    auto mask_operand = motion ? 3u : 2u;
    auto *result = _source.value(*instruction.result);
    auto *accel_value = _source.value(instruction.operands[0u]);
    auto *ray_value = _source.value(instruction.operands[1u]);
    auto *time_value = motion ?
                           _source.value(instruction.operands[2u]) :
                           nullptr;
    auto *mask_value = _source.value(instruction.operands[mask_operand]);
    auto *accel = _load_value(instruction.operands[0u]);
    auto *ray = ray_value == nullptr ? nullptr :
                                       _as_lane_vector(
                                           _load_value(instruction.operands[1u]), *ray_value);
    auto *time = time_value == nullptr ? nullptr :
                                         _as_lane_vector(
                                             _load_value(instruction.operands[2u]), *time_value);
    auto *visibility = mask_value == nullptr ? nullptr :
                                               _as_lane_vector(
                                                   _load_value(instruction.operands[mask_operand]), *mask_value);
    if (result == nullptr || accel_value == nullptr ||
        ray_value == nullptr || mask_value == nullptr ||
        accel == nullptr || ray == nullptr || visibility == nullptr ||
        (motion &&
         (time_value == nullptr || time == nullptr ||
          time_value->type == nullptr ||
          !time_value->type->is_float32())) ||
        accel_value->type == nullptr ||
        !accel_value->type->is_accel() ||
        !is_ray_type(ray_value->type) ||
        mask_value->type == nullptr ||
        !mask_value->type->is_uint32() ||
        (closest ? !is_surface_hit_type(result->type) :
                   !result->type->is_bool())) {
        _fail("acceleration-structure query has invalid operands or result");
        return nullptr;
    }

    auto &context = _module.getContext();
    auto *pointer_type = ::llvm::PointerType::getUnqual(context);
    auto *float_lanes = ::llvm::FixedVectorType::get(
        _builder.getFloatTy(), _width);
    auto *i32_lanes = ::llvm::FixedVectorType::get(
        _builder.getInt32Ty(), _width);
    auto *null_pointer =
        ::llvm::ConstantPointerNull::get(pointer_type);
    visibility = _builder.CreateZExtOrTrunc(visibility, i32_lanes);
    visibility = _builder.CreateSelect(
        _active_mask, visibility,
        ::llvm::Constant::getNullValue(i32_lanes),
        "accel.safe.visibility");

    auto *origin_type = _child_type(ray_value->type, 0u);
    auto *direction_type = _child_type(ray_value->type, 2u);
    auto *origin = _extract_child(
        ray, ray_value->type, 0u, true);
    auto *direction = _extract_child(
        ray, ray_value->type, 2u, true);
    std::array<::llvm::Value *, 8u> ray_components{
        _extract_child(origin, origin_type, 0u, true),
        _extract_child(origin, origin_type, 1u, true),
        _extract_child(origin, origin_type, 2u, true),
        _extract_child(ray, ray_value->type, 1u, true),
        _extract_child(direction, direction_type, 0u, true),
        _extract_child(direction, direction_type, 1u, true),
        _extract_child(direction, direction_type, 2u, true),
        _extract_child(ray, ray_value->type, 3u, true),
    };
    auto *zero_float = ::llvm::Constant::getNullValue(float_lanes);
    auto *one_float = ::llvm::ConstantVector::getSplat(
        ::llvm::ElementCount::getFixed(_width),
        ::llvm::ConstantFP::get(_builder.getFloatTy(), 1.0));
    auto *ray_scratch_type = ::llvm::ArrayType::get(
        float_lanes, ray_components.size());
    auto *ray_scratch = _entry_scratch(
        ray_scratch_type,
        "accel.rays." +
            std::to_string(instruction.result->value));
    _builder.CreateStore(
        ::llvm::Constant::getNullValue(ray_scratch_type), ray_scratch);
    for (auto component = uint32_t{0u};
         component < ray_components.size(); component++) {
        auto *fallback = component == 6u ? one_float : zero_float;
        auto *safe = _builder.CreateSelect(
            _active_mask, ray_components[component], fallback,
            "accel.safe.ray.component");
        auto *pointer = _builder.CreateGEP(
            ray_scratch_type, ray_scratch,
            {_builder.getInt32(0u), _builder.getInt32(component)});
        _builder.CreateStore(safe, pointer);
    }
    auto *mask_scratch = _entry_scratch(
        i32_lanes,
        "accel.masks." +
            std::to_string(instruction.result->value));
    _builder.CreateStore(visibility, mask_scratch);
    auto *time_scratch = static_cast<::llvm::Value *>(null_pointer);
    if (motion) {
        auto *safe_time = _builder.CreateSelect(
            _active_mask, time, zero_float,
            "accel.safe.ray.time");
        auto *scratch = _entry_scratch(
            float_lanes,
            "accel.times." +
                std::to_string(instruction.result->value));
        _builder.CreateStore(safe_time, scratch);
        time_scratch = scratch;
    }

    auto *object = _builder.CreateExtractValue(accel, {0u});
    auto callback_index = closest ? 1u : 2u;
    auto *callback = _builder.CreateExtractValue(
        accel, {callback_index});
    auto *missing_callback = _builder.CreateOr(
        _builder.CreateICmpEQ(object, null_pointer),
        _builder.CreateICmpEQ(callback, null_pointer));
    _trap_if(
        _builder.CreateAnd(
            _builder.CreateOrReduce(_active_mask), missing_callback),
        closest ? "accel.trace.closest.callback.null" :
                  "accel.trace.any.callback.null");
    auto *active_mask_bits = _bindless_callback_mask(
        result->value_class == schedule::ValueClass::varying);

    if (closest) {
        auto *ids_type = ::llvm::ArrayType::get(i32_lanes, 2u);
        auto *values_type = ::llvm::ArrayType::get(float_lanes, 3u);
        auto *ids = _entry_scratch(
            ids_type,
            "accel.closest.ids." +
                std::to_string(instruction.result->value));
        auto *values = _entry_scratch(
            values_type,
            "accel.closest.values." +
                std::to_string(instruction.result->value));
        auto *invalid_ids = ::llvm::ConstantArray::get(
            ids_type,
            {::llvm::Constant::getAllOnesValue(i32_lanes),
             ::llvm::Constant::getAllOnesValue(i32_lanes)});
        _builder.CreateStore(invalid_ids, ids);
        _builder.CreateStore(
            ::llvm::Constant::getNullValue(values_type), values);
        auto *callback_type = ::llvm::FunctionType::get(
            _builder.getVoidTy(),
            {pointer_type, _builder.getInt32Ty(),
             _builder.getInt64Ty(), pointer_type,
             pointer_type, pointer_type, pointer_type,
             pointer_type},
            false);
        _builder.CreateCall(
            callback_type, callback,
            {object, _builder.getInt32(_width), active_mask_bits,
             ray_scratch, mask_scratch, time_scratch,
             ids, values});

        auto load_ids = [&](uint32_t component) {
            auto *pointer = _builder.CreateGEP(
                ids_type, ids,
                {_builder.getInt32(0u), _builder.getInt32(component)});
            return _builder.CreateLoad(i32_lanes, pointer);
        };
        auto load_values = [&](uint32_t component) {
            auto *pointer = _builder.CreateGEP(
                values_type, values,
                {_builder.getInt32(0u), _builder.getInt32(component)});
            return _builder.CreateLoad(float_lanes, pointer);
        };
        auto *bary_type = _child_type(result->type, 2u);
        auto *bary = _assemble(
            bary_type, true,
            [&](uint32_t component) { return load_values(component); });
        auto *hits = static_cast<::llvm::Value *>(
            ::llvm::PoisonValue::get(_data_type(result->type, true)));
        hits = _insert_child(
            hits, load_ids(0u), result->type, 0u, true);
        hits = _insert_child(
            hits, load_ids(1u), result->type, 1u, true);
        hits = _insert_child(
            hits, bary, result->type, 2u, true);
        hits = _insert_child(
            hits, load_values(2u), result->type, 3u, true);
        return result->value_class == schedule::ValueClass::varying ?
                   hits :
                   _extract_lane(
                       hits, result->type,
                       _safe_first_lane(_active_mask));
    }

    auto *occluded = _entry_scratch(
        i32_lanes,
        "accel.any.result." +
            std::to_string(instruction.result->value));
    _builder.CreateStore(
        ::llvm::Constant::getNullValue(i32_lanes), occluded);
    auto *callback_type = ::llvm::FunctionType::get(
        _builder.getVoidTy(),
        {pointer_type, _builder.getInt32Ty(),
         _builder.getInt64Ty(), pointer_type,
         pointer_type, pointer_type, pointer_type},
        false);
    _builder.CreateCall(
        callback_type, callback,
        {object, _builder.getInt32(_width), active_mask_bits,
         ray_scratch, mask_scratch, time_scratch, occluded});
    auto *bits = _builder.CreateICmpNE(
        _builder.CreateLoad(i32_lanes, occluded),
        ::llvm::Constant::getNullValue(i32_lanes));
    return result->value_class == schedule::ValueClass::varying ?
               bits :
               _builder.CreateExtractElement(
                   bits, _safe_first_lane(_active_mask));
}

}// namespace luisa::compute::simd::detail
