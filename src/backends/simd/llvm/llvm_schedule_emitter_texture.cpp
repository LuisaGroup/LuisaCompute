#include "llvm_schedule_emitter.h"

#include <array>

#include <luisa/runtime/rhi/pixel.h>

#include "../../common/llvm_native_math.h"

namespace luisa::compute::simd::detail {

[[nodiscard]] ScheduleEmitter::NativeTexturePacketInfo
ScheduleEmitter::_native_texture_packet_info(
    ::llvm::Value *texture,
    const std::array<::llvm::Value *, 3u> &coordinates,
    bool floating, bool allow_byte4_write) {
    NativeTexturePacketInfo info{};
    if ((_width != 8u && _width != 16u) || texture == nullptr ||
        coordinates[0u] == nullptr || coordinates[1u] == nullptr) {
        return info;
    }
    auto &context = _module.getContext();
    auto *pointer_type = ::llvm::PointerType::getUnqual(context);
    auto *null_pointer = ::llvm::ConstantPointerNull::get(pointer_type);
    auto *i32_lanes = ::llvm::FixedVectorType::get(
        _builder.getInt32Ty(), _width);
    auto *zero_lanes = ::llvm::Constant::getNullValue(i32_lanes);
    auto *safe_x = _builder.CreateSelect(
        _active_mask, coordinates[0u], zero_lanes,
        "texture.native.safe.x");
    auto *safe_y = _builder.CreateSelect(
        _active_mask, coordinates[1u], zero_lanes,
        "texture.native.safe.y");
    auto *x = _builder.CreateExtractElement(safe_x, uint64_t{0u});
    auto *y = _builder.CreateExtractElement(safe_y, uint64_t{0u});
    auto *expected_x = _builder.CreateAdd(
        _builder.CreateVectorSplat(_width, x), _lane_ids());
    auto *same_x = _builder.CreateAndReduce(
        _builder.CreateICmpEQ(safe_x, expected_x));
    auto *same_y = _builder.CreateAndReduce(
        _builder.CreateICmpEQ(
            safe_y, _builder.CreateVectorSplat(_width, y)));
    auto *all_active = _builder.CreateAndReduce(_active_mask);
    auto *data = _builder.CreateExtractValue(texture, {9u});
    auto *width = _builder.CreateExtractValue(texture, {10u});
    auto *height = _builder.CreateExtractValue(texture, {11u});
    auto *storage = _builder.CreateExtractValue(texture, {13u});
    auto *dimension = _builder.CreateExtractValue(texture, {7u});
    auto *has_data = _builder.CreateICmpNE(data, null_pointer);
    auto *native_storage = _builder.getInt32(static_cast<uint32_t>(
        floating ? PixelStorage::FLOAT4 : PixelStorage::INT4));
    auto *storage_matches = _builder.CreateICmpEQ(
        storage, native_storage);
    auto *dimension_matches = _builder.CreateICmpEQ(
        dimension, _builder.getInt32(2u));
    auto *x_inside = _builder.CreateICmpULT(x, width);
    auto *row_remaining = _builder.CreateSub(width, x);
    auto *packet_inside = _builder.CreateICmpUGE(
        row_remaining, _builder.getInt32(_width));
    auto *y_inside = _builder.CreateICmpULT(y, height);
    auto *common_guard = _builder.CreateAnd(all_active, same_x);
    common_guard = _builder.CreateAnd(common_guard, same_y);
    common_guard = _builder.CreateAnd(common_guard, has_data);
    common_guard = _builder.CreateAnd(
        common_guard, dimension_matches);
    common_guard = _builder.CreateAnd(common_guard, x_inside);
    common_guard = _builder.CreateAnd(common_guard, packet_inside);
    common_guard = _builder.CreateAnd(common_guard, y_inside);
    info.guard = _builder.CreateAnd(
        common_guard, storage_matches,
        "texture.native.packet.guard");
    if (allow_byte4_write && floating) {
        auto *capabilities = _builder.CreateExtractValue(
            texture, {14u});
        auto *has_byte4_capability = _builder.CreateICmpNE(
            _builder.CreateAnd(
                capabilities,
                _builder.getInt32(
                    simd_host_texture_capability_byte4_float_write)),
            _builder.getInt32(0u));
        auto *byte4_storage = _builder.CreateICmpEQ(
            storage,
            _builder.getInt32(static_cast<uint32_t>(
                PixelStorage::BYTE4)));
        auto *byte4_guard = _builder.CreateAnd(
            common_guard, has_byte4_capability);
        info.byte4_guard = _builder.CreateAnd(
            byte4_guard, byte4_storage,
            "texture.byte4.packet.guard");
    }
    info.data = data;
    info.width = width;
    info.x = x;
    info.y = y;
    return info;
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
        op == xir::ResourceReadOp::TEXTURE3D_READ ? 3u :
                                                    0u;
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
    auto *result_type = _data_type(result->type, true);
    auto emit_callback = [&]() -> ::llvm::Value * {
        auto *scratch_type = ::llvm::ArrayType::get(lane_type, 4u);
        auto *scratch = _entry_scratch(
            scratch_type,
            "texture.read.packet." +
                std::to_string(instruction.result->value));
        auto *read = _builder.CreateExtractValue(
            texture, {floating ? 1u : 2u});
        auto *object = _builder.CreateExtractValue(texture, {0u});
        auto *level = _builder.CreateExtractValue(texture, {6u});
        auto *pointer_type = ::llvm::PointerType::getUnqual(
            _module.getContext());
        auto *read_type = ::llvm::FunctionType::get(
            _builder.getVoidTy(),
            {pointer_type, _builder.getInt32Ty(),
             _builder.getInt32Ty(), _builder.getInt64Ty(),
             pointer_type, pointer_type, pointer_type, pointer_type},
            false);
        std::array<::llvm::AllocaInst *, 3u> coordinate_scratch{};
        auto *coordinate_type = ::llvm::FixedVectorType::get(
            _builder.getInt32Ty(), _width);
        auto *zero_coordinates = ::llvm::Constant::getNullValue(
            coordinate_type);
        for (auto axis = uint32_t{0u}; axis < 3u; axis++) {
            coordinate_scratch[axis] = _entry_scratch(
                coordinate_type,
                "texture.read.coordinate." + std::to_string(axis));
            auto *safe_coordinate = _builder.CreateSelect(
                _active_mask, coordinates[axis], zero_coordinates,
                "texture.read.safe.coordinate");
            _builder.CreateStore(
                safe_coordinate, coordinate_scratch[axis]);
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
    };

    if (expected_dimension != 2u ||
        (_width != 8u && _width != 16u)) {
        return emit_callback();
    }
    auto native = _native_texture_packet_info(
        texture, coordinates, floating, false);
    if (native.guard == nullptr) { return emit_callback(); }
    auto &context = _module.getContext();
    auto *fast_block = ::llvm::BasicBlock::Create(
        context, "texture.read.native", _entry);
    auto *callback_block = ::llvm::BasicBlock::Create(
        context, "texture.read.callback", _entry);
    auto *merge_block = ::llvm::BasicBlock::Create(
        context, "texture.read.merge", _entry);
    _builder.CreateCondBr(native.guard, fast_block, callback_block);

    _builder.SetInsertPoint(fast_block);
    auto *i64 = _builder.getInt64Ty();
    auto *pixel_index = _builder.CreateAdd(
        _builder.CreateMul(
            _builder.CreateZExt(native.y, i64),
            _builder.CreateZExt(native.width, i64)),
        _builder.CreateZExt(native.x, i64));
    auto *byte_offset = _builder.CreateShl(
        pixel_index, _builder.getInt64(4u));
    auto *pixel_pointer = _builder.CreateGEP(
        _builder.getInt8Ty(), native.data, byte_offset,
        "texture.read.native.pointer");
    auto *physical_type = ::llvm::FixedVectorType::get(
        scalar_type, 4u * _width);
    auto *loaded = _builder.CreateAlignedLoad(
        physical_type, pixel_pointer, ::llvm::Align{1u},
        "texture.read.native.aos");
    auto *native_pixels = static_cast<::llvm::Value *>(
        ::llvm::PoisonValue::get(result_type));
    for (auto component = uint32_t{0u}; component < 4u;
         component++) {
        std::vector<int> mask;
        mask.reserve(_width);
        for (auto lane = uint32_t{0u}; lane < _width; lane++) {
            mask.emplace_back(static_cast<int>(4u * lane + component));
        }
        auto *lanes = _builder.CreateShuffleVector(
            loaded, ::llvm::PoisonValue::get(physical_type), mask,
            "texture.read.native.soa");
        native_pixels = _insert_child(
            native_pixels, lanes, result->type, component, true);
    }
    auto *fast_exit = _builder.GetInsertBlock();
    _builder.CreateBr(merge_block);

    _builder.SetInsertPoint(callback_block);
    auto *callback_pixels = emit_callback();
    auto *callback_exit = _builder.GetInsertBlock();
    _builder.CreateBr(merge_block);

    _builder.SetInsertPoint(merge_block);
    auto *pixels = _builder.CreatePHI(
        result_type, 2u, "texture.read.packet");
    pixels->addIncoming(native_pixels, fast_exit);
    pixels->addIncoming(callback_pixels, callback_exit);
    _result.guarded_native_texture_read_count++;
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
        op == xir::ResourceWriteOp::TEXTURE3D_WRITE ? 3u :
                                                      0u;
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
    std::array<::llvm::Value *, 3u> coordinates{
        nullptr, nullptr,
        _builder.CreateVectorSplat(
            _width, _builder.getInt32(0u))};
    for (auto axis = uint32_t{0u}; axis < expected_dimension; axis++) {
        coordinates[axis] = _extract_child(
            coordinate, coordinate_value->type, axis, true);
    }
    auto emit_callback = [&] {
        auto *scratch_type = ::llvm::ArrayType::get(lane_type, 4u);
        auto *scratch = _entry_scratch(
            scratch_type,
            "texture.write.packet." + std::to_string(
                                          instruction.operands[2u].value));
        auto *write = _builder.CreateExtractValue(
            texture, {floating ? 3u : 4u});
        auto *object = _builder.CreateExtractValue(texture, {0u});
        auto *level = _builder.CreateExtractValue(texture, {6u});
        auto *pointer_type = ::llvm::PointerType::getUnqual(
            _module.getContext());
        auto *write_type = ::llvm::FunctionType::get(
            _builder.getVoidTy(),
            {pointer_type, _builder.getInt32Ty(),
             _builder.getInt32Ty(), _builder.getInt64Ty(),
             pointer_type, pointer_type, pointer_type, pointer_type},
            false);
        std::array<::llvm::AllocaInst *, 3u> coordinate_scratch{};
        auto *coordinate_type = ::llvm::FixedVectorType::get(
            _builder.getInt32Ty(), _width);
        auto *zero_coordinates = ::llvm::Constant::getNullValue(
            coordinate_type);
        for (auto axis = uint32_t{0u}; axis < 3u; axis++) {
            coordinate_scratch[axis] = _entry_scratch(
                coordinate_type,
                "texture.write.coordinate." + std::to_string(axis));
            auto *safe_coordinate = _builder.CreateSelect(
                _active_mask, coordinates[axis], zero_coordinates,
                "texture.write.safe.coordinate");
            _builder.CreateStore(
                safe_coordinate, coordinate_scratch[axis]);
        }
        auto *zero_lanes = ::llvm::Constant::getNullValue(lane_type);
        for (auto component = uint32_t{0u}; component < 4u;
             component++) {
            auto *component_pointer = _builder.CreateGEP(
                scratch_type, scratch,
                {_builder.getInt32(0u),
                 _builder.getInt32(component)});
            auto *lanes = _extract_child(
                written, written_value->type, component, true);
            lanes = _builder.CreateSelect(
                _active_mask, lanes, zero_lanes,
                "texture.write.safe.value");
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
    };

    if (expected_dimension != 2u ||
        (_width != 8u && _width != 16u)) {
        emit_callback();
        return;
    }
    auto native = _native_texture_packet_info(
        texture, coordinates, floating, floating);
    if (native.guard == nullptr) {
        emit_callback();
        return;
    }
    std::array<::llvm::Value *, 4u> components{};
    for (auto component = uint32_t{0u}; component < 4u;
         component++) {
        components[component] = _extract_child(
            written, written_value->type, component, true);
    }
    std::array<::llvm::Value *, 4u> byte4_components{};

    std::vector<int> concatenate_mask;
    concatenate_mask.reserve(2u * _width);
    for (auto i = uint32_t{0u}; i < 2u * _width; i++) {
        concatenate_mask.emplace_back(static_cast<int>(i));
    }
    std::vector<int> interleave_mask;
    interleave_mask.reserve(4u * _width);
    for (auto lane = uint32_t{0u}; lane < _width; lane++) {
        interleave_mask.emplace_back(static_cast<int>(lane));
        interleave_mask.emplace_back(static_cast<int>(_width + lane));
        interleave_mask.emplace_back(static_cast<int>(2u * _width + lane));
        interleave_mask.emplace_back(static_cast<int>(3u * _width + lane));
    }
    auto interleave_components = [&concatenate_mask, &interleave_mask, this](
                                     const std::array<::llvm::Value *, 4u> &values,
                                     const char *low_name,
                                     const char *high_name,
                                     const char *result_name) {
        auto *low = _builder.CreateShuffleVector(
            values[0u], values[1u], concatenate_mask, low_name);
        auto *high = _builder.CreateShuffleVector(
            values[2u], values[3u], concatenate_mask, high_name);
        return _builder.CreateShuffleVector(
            low, high, interleave_mask, result_name);
    };

    auto &context = _module.getContext();
    auto *fast_block = ::llvm::BasicBlock::Create(
        context, "texture.write.native", _entry);
    auto *byte4_route_block = native.byte4_guard == nullptr ?
                                  nullptr :
                                  ::llvm::BasicBlock::Create(
                                      context,
                                      "texture.write.byte4.route", _entry);
    auto *byte4_value_block = native.byte4_guard == nullptr ?
                                  nullptr :
                                  ::llvm::BasicBlock::Create(
                                      context,
                                      "texture.write.byte4.value", _entry);
    auto *byte4_block = native.byte4_guard == nullptr ?
                            nullptr :
                            ::llvm::BasicBlock::Create(
                                context, "texture.write.byte4", _entry);
    auto *callback_block = ::llvm::BasicBlock::Create(
        context, "texture.write.callback", _entry);
    auto *merge_block = ::llvm::BasicBlock::Create(
        context, "texture.write.merge", _entry);
    auto *i64 = _builder.getInt64Ty();
    auto *pixel_index = _builder.CreateAdd(
        _builder.CreateMul(
            _builder.CreateZExt(native.y, i64),
            _builder.CreateZExt(native.width, i64)),
        _builder.CreateZExt(native.x, i64));
    _builder.CreateCondBr(
        native.guard, fast_block,
        byte4_route_block == nullptr ? callback_block :
                                       byte4_route_block);
    if (byte4_route_block != nullptr) {
        _builder.SetInsertPoint(byte4_route_block);
        _builder.CreateCondBr(
            native.byte4_guard, byte4_value_block, callback_block);

        _builder.SetInsertPoint(byte4_value_block);
        auto *zero_lanes = ::llvm::Constant::getNullValue(lane_type);
        auto *all_ordered = static_cast<::llvm::Value *>(
            _builder.getTrue());
        for (auto component = uint32_t{0u}; component < 4u;
             component++) {
            auto *safe = _builder.CreateSelect(
                _active_mask, components[component], zero_lanes,
                "texture.write.byte4.safe.active");
            auto *ordered = _builder.CreateFCmpORD(safe, safe);
            all_ordered = _builder.CreateAnd(
                all_ordered, _builder.CreateAndReduce(ordered));
            byte4_components[component] = _builder.CreateSelect(
                ordered, safe, zero_lanes,
                "texture.write.byte4.safe.nan");
        }
        _builder.CreateCondBr(
            all_ordered, byte4_block, callback_block);
    }

    _builder.SetInsertPoint(fast_block);
    auto *byte_offset = _builder.CreateShl(
        pixel_index, _builder.getInt64(4u));
    auto *pixel_pointer = _builder.CreateGEP(
        _builder.getInt8Ty(), native.data, byte_offset,
        "texture.write.native.pointer");
    auto *stored = interleave_components(
        components, "texture.write.native.low",
        "texture.write.native.high",
        "texture.write.native.aos");
    _builder.CreateAlignedStore(
        stored, pixel_pointer, ::llvm::Align{1u});
    _builder.CreateBr(merge_block);

    if (byte4_block != nullptr) {
        _builder.SetInsertPoint(byte4_block);
        auto *float_lanes = ::llvm::FixedVectorType::get(
            _builder.getFloatTy(), _width);
        auto *i32_lanes = ::llvm::FixedVectorType::get(
            _builder.getInt32Ty(), _width);
        auto splat_float = [this](float value) {
            return ::llvm::ConstantVector::getSplat(
                ::llvm::ElementCount::getFixed(_width),
                ::llvm::ConstantFP::get(
                    _builder.getFloatTy(), value));
        };
        auto *zero = ::llvm::Constant::getNullValue(float_lanes);
        auto *one = splat_float(1.0f);
        auto *scale = splat_float(255.0f);
        auto *round_bias = splat_float(0.5f);
        std::array<::llvm::Value *, 4u> converted{};
        for (auto component = uint32_t{0u}; component < 4u;
             component++) {
            auto *clamped = _builder.CreateMaxNum(
                byte4_components[component], zero);
            clamped = _builder.CreateMinNum(clamped, one);
            auto *scaled = _builder.CreateFAdd(
                _builder.CreateFMul(clamped, scale), round_bias);
            converted[component] = _builder.CreateFPToUI(
                scaled, i32_lanes,
                "texture.write.byte4.converted");
        }
        auto *packed_i32 = interleave_components(
            converted, "texture.write.byte4.low",
            "texture.write.byte4.high",
            "texture.write.byte4.i32.aos");
        auto *packed_byte_type = ::llvm::FixedVectorType::get(
            _builder.getInt8Ty(), 4u * _width);
        auto *packed_bytes = _builder.CreateTrunc(
            packed_i32, packed_byte_type,
            "texture.write.byte4.aos");
        auto *byte4_offset = _builder.CreateShl(
            pixel_index, _builder.getInt64(2u));
        auto *byte4_pointer = _builder.CreateGEP(
            _builder.getInt8Ty(), native.data, byte4_offset,
            "texture.write.byte4.pointer");
        _builder.CreateAlignedStore(
            packed_bytes, byte4_pointer, ::llvm::Align{1u});
        _builder.CreateBr(merge_block);
    }

    _builder.SetInsertPoint(callback_block);
    emit_callback();
    _builder.CreateBr(merge_block);

    _builder.SetInsertPoint(merge_block);
    _result.guarded_native_texture_write_count++;
    if (native.byte4_guard != nullptr) {
        _result.guarded_byte4_texture_write_count++;
    }
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_direct_texture_sample(
    const schedule::Instruction &instruction) {
    if (!instruction.result || !instruction.source_op) {
        _fail("direct texture sample instruction is malformed");
        return nullptr;
    }
    auto op = static_cast<xir::ResourceQueryOp>(
        *instruction.source_op);
    auto dimension =
        op == xir::ResourceQueryOp::TEXTURE2D_SAMPLE ||
                op == xir::ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL ||
                op == xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD ||
                op == xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL ?
            2u :
        op == xir::ResourceQueryOp::TEXTURE3D_SAMPLE ||
                op == xir::ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL ||
                op == xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD ||
                op == xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL ?
            3u :
            0u;
    auto has_level =
        op == xir::ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL ||
        op == xir::ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL;
    auto has_gradient =
        op == xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD ||
        op == xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD ||
        op == xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL ||
        op == xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL;
    auto has_gradient_level =
        op == xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL ||
        op == xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL;
    auto expected_operands =
        4u + (has_gradient ? 2u : 0u) +
        ((has_level || has_gradient_level) ? 1u : 0u);
    if (dimension == 0u ||
        instruction.operands.size() != expected_operands) {
        _fail("unsupported direct texture sample operation");
        return nullptr;
    }
    if (_width > 64u) {
        _fail("direct texture packet callbacks support widths up to 64 lanes");
        return nullptr;
    }

    auto *result = _source.value(*instruction.result);
    auto *texture_value = _source.value(instruction.operands[0u]);
    auto *coordinate_value = _source.value(instruction.operands[1u]);
    auto *texture = texture_value == nullptr ? nullptr :
                                               _load_value(
                                                   instruction.operands[0u]);
    auto *coordinate = coordinate_value == nullptr ? nullptr :
                                                     _as_lane_vector(
                                                         _load_value(instruction.operands[1u]),
                                                         *coordinate_value);
    if (result == nullptr || result->type == nullptr ||
        !result->type->is_vector() || result->type->dimension() != 4u ||
        !result->type->element()->is_float32() ||
        texture_value == nullptr || texture_value->type == nullptr ||
        !texture_value->type->is_texture() || texture == nullptr ||
        coordinate_value == nullptr || coordinate_value->type == nullptr ||
        !coordinate_value->type->is_vector() ||
        coordinate_value->type->dimension() != dimension ||
        !coordinate_value->type->element()->is_float32() ||
        coordinate == nullptr) {
        _fail("direct texture sample has invalid operands or result type");
        return nullptr;
    }

    auto &context = _module.getContext();
    auto *pointer_type = ::llvm::PointerType::getUnqual(context);
    auto *null_pointer = ::llvm::ConstantPointerNull::get(pointer_type);
    auto *i32_lanes = ::llvm::FixedVectorType::get(
        _builder.getInt32Ty(), _width);
    auto *float_lanes = ::llvm::FixedVectorType::get(
        _builder.getFloatTy(), _width);
    auto *zero_i32 = ::llvm::Constant::getNullValue(i32_lanes);
    auto *zero_float = ::llvm::Constant::getNullValue(float_lanes);
    auto *one_float = ::llvm::ConstantVector::getSplat(
        ::llvm::ElementCount::getFixed(_width),
        ::llvm::ConstantFP::get(_builder.getFloatTy(), 1.0));
    auto varying_result = result->value_class ==
                          schedule::ValueClass::varying;

    std::array<::llvm::AllocaInst *, 3u> coordinate_scratch{};
    for (auto axis = uint32_t{0u}; axis < 3u; axis++) {
        auto *lanes = axis < dimension ?
                          _extract_child(
                              coordinate, coordinate_value->type,
                              axis, true) :
                          zero_float;
        lanes = _builder.CreateSelect(
            _active_mask, lanes, zero_float,
            "texture.sample.safe.coordinate");
        coordinate_scratch[axis] = _entry_scratch(
            float_lanes,
            "texture.sample.coordinate." + std::to_string(axis));
        _builder.CreateStore(lanes, coordinate_scratch[axis]);
    }

    auto *object = _builder.CreateExtractValue(texture, {0u});
    auto *size_callback = _builder.CreateExtractValue(texture, {5u});
    auto *base_level = _builder.CreateExtractValue(texture, {6u});
    auto *sample_callback = _builder.CreateExtractValue(texture, {8u});
    auto *any_active = _builder.CreateOrReduce(_active_mask);
    auto *missing_sample = _builder.CreateICmpEQ(
        sample_callback, null_pointer);
    _trap_if(
        _builder.CreateAnd(any_active, missing_sample),
        "texture.sample.callback.null");

    auto operand_index = size_t{2u};
    ::llvm::Value *gradient_level = nullptr;
    ::llvm::Value *uniform_gradient_level = nullptr;
    ::llvm::Value *gradient_mask = nullptr;
    auto gradient_level_is_uniform = false;
    if (has_gradient) {
        std::array<const schedule::Value *, 2u> gradient_values{};
        auto uniform_gradient_lod = true;
        for (auto derivative = uint32_t{0u}; derivative < 2u;
             derivative++) {
            auto *gradient_value = _source.value(
                instruction.operands[operand_index + derivative]);
            gradient_values[derivative] = gradient_value;
            if (gradient_value == nullptr ||
                gradient_value->type == nullptr ||
                !gradient_value->type->is_vector() ||
                gradient_value->type->dimension() != dimension ||
                !gradient_value->type->element()->is_float32()) {
                _fail("direct texture sample has an invalid gradient");
                return nullptr;
            }
            uniform_gradient_lod &=
                schedule::is_uniform(gradient_value->value_class);
        }

        auto *first_active_mask = _builder.CreateAnd(
            _active_mask,
            _builder.CreateICmpEQ(
                _lane_ids(),
                _builder.CreateVectorSplat(
                    _width, _safe_first_lane(_active_mask))));
        gradient_mask = varying_result ? _active_mask :
                                         first_active_mask;

        auto *missing_size = _builder.CreateICmpEQ(
            size_callback, null_pointer);
        _trap_if(
            _builder.CreateAnd(any_active, missing_size),
            "texture.sample.size.callback.null");
        auto *size_type = ::llvm::FunctionType::get(
            _builder.getInt32Ty(),
            {pointer_type, _builder.getInt32Ty(),
             _builder.getInt32Ty()},
            false);
        std::array<::llvm::Value *, 3u> extents{};
        for (auto axis = uint32_t{0u}; axis < dimension; axis++) {
            auto *extent = _builder.CreateCall(
                size_type, size_callback,
                {object, base_level, _builder.getInt32(axis)},
                "texture.sample.extent." + std::to_string(axis));
            extents[axis] = _builder.CreateUIToFP(
                extent, _builder.getFloatTy());
            if (!uniform_gradient_lod) {
                extents[axis] = _builder.CreateVectorSplat(
                    _width, extents[axis]);
            }
        }

        if (uniform_gradient_lod) {
            gradient_level_is_uniform = true;
            auto *zero_scalar = ::llvm::ConstantFP::get(
                _builder.getFloatTy(), 0.0);
            auto *one_scalar = ::llvm::ConstantFP::get(
                _builder.getFloatTy(), 1.0);
            std::array<::llvm::Value *, 2u> gradient_norms{
                zero_scalar, zero_scalar};
            ::llvm::Value *gradient_has_nan = _builder.getFalse();
            for (auto derivative = uint32_t{0u}; derivative < 2u;
                 derivative++) {
                auto *gradient = _load_value(
                    instruction.operands[operand_index + derivative]);
                if (gradient == nullptr) { return nullptr; }
                for (auto axis = uint32_t{0u}; axis < dimension;
                     axis++) {
                    auto *component = _extract_child(
                        gradient, gradient_values[derivative]->type,
                        axis, false);
                    gradient_has_nan = _builder.CreateOr(
                        gradient_has_nan,
                        _builder.CreateFCmpUNO(component, component));
                    auto *scaled = _builder.CreateFMul(
                        component, extents[axis]);
                    gradient_norms[derivative] = _builder.CreateFAdd(
                        gradient_norms[derivative],
                        _builder.CreateFMul(scaled, scaled));
                }
            }
            auto *rho2 = _builder.CreateMaxNum(
                gradient_norms[0u], gradient_norms[1u]);
            rho2 = _builder.CreateMaxNum(rho2, one_scalar);
            rho2 = _builder.CreateSelect(
                gradient_has_nan, one_scalar, rho2,
                "texture.sample.uniform.nan.gradient.lod");
            uniform_gradient_level = _builder.CreateUnaryIntrinsic(
                ::llvm::Intrinsic::log2, rho2, nullptr,
                "texture.sample.uniform.gradient.log2");
            uniform_gradient_level = _builder.CreateFMul(
                uniform_gradient_level,
                ::llvm::ConstantFP::get(
                    _builder.getFloatTy(), 0.5));
        } else {
            std::array<::llvm::Value *, 2u> gradient_norms{
                zero_float, zero_float};
            ::llvm::Value *gradient_has_nan =
                ::llvm::Constant::getNullValue(
                    ::llvm::FixedVectorType::get(
                        _builder.getInt1Ty(), _width));
            for (auto derivative = uint32_t{0u}; derivative < 2u;
                 derivative++) {
                auto *gradient = _as_lane_vector(
                    _load_value(
                        instruction.operands[operand_index + derivative]),
                    *gradient_values[derivative]);
                if (gradient == nullptr) { return nullptr; }
                for (auto axis = uint32_t{0u}; axis < dimension;
                     axis++) {
                    auto *lanes = _extract_child(
                        gradient, gradient_values[derivative]->type,
                        axis, true);
                    lanes = _builder.CreateSelect(
                        gradient_mask, lanes, zero_float,
                        "texture.sample.safe.gradient");
                    gradient_has_nan = _builder.CreateOr(
                        gradient_has_nan,
                        _builder.CreateFCmpUNO(lanes, lanes));
                    auto *scaled = _builder.CreateFMul(
                        lanes, extents[axis]);
                    gradient_norms[derivative] = _builder.CreateFAdd(
                        gradient_norms[derivative],
                        _builder.CreateFMul(scaled, scaled));
                }
            }
            auto *rho2 = _builder.CreateMaxNum(
                gradient_norms[0u], gradient_norms[1u]);
            rho2 = _builder.CreateMaxNum(rho2, one_float);
            rho2 = _builder.CreateSelect(
                gradient_has_nan, one_float, rho2,
                "texture.sample.nan.gradient.lod");
            auto native_math_mode = _enable_fast_math ?
                                        cpu::LLVMNativeMathMode::fast :
                                        cpu::LLVMNativeMathMode::precise;
            gradient_level = cpu::LLVMNativeMath::emit_log2_f32(
                _module, _builder, rho2, native_math_mode);
            if (gradient_level == nullptr) {
                _fail("direct texture gradient LOD requires fixed f32 vectors");
                return nullptr;
            }
            gradient_level = _builder.CreateFMul(
                gradient_level,
                ::llvm::ConstantVector::getSplat(
                    ::llvm::ElementCount::getFixed(_width),
                    ::llvm::ConstantFP::get(
                        _builder.getFloatTy(), 0.5)));
            gradient_level = _builder.CreateSelect(
                gradient_mask, gradient_level, zero_float,
                "texture.sample.safe.gradient.level");
        }
        operand_index += 2u;
    }

    ::llvm::Value *level_pointer = null_pointer;
    if (has_level || has_gradient_level) {
        auto *level_value = _source.value(
            instruction.operands[operand_index]);
        auto *raw_level = level_value == nullptr ? nullptr :
                                                   _load_value(
                                                       instruction.operands[operand_index]);
        if (level_value == nullptr || raw_level == nullptr ||
            level_value->type == nullptr ||
            !level_value->type->is_float32()) {
            _fail(
                has_gradient_level ?
                    "direct texture sample has an invalid minimum mip level" :
                    "direct texture sample has an invalid mip level");
            return nullptr;
        }
        ::llvm::Value *level = nullptr;
        auto uniform_minimum_level =
            has_gradient_level && gradient_level_is_uniform &&
            schedule::is_uniform(level_value->value_class);
        if (uniform_minimum_level) {
            auto *scalar_level = _builder.CreateMaxNum(
                uniform_gradient_level, raw_level);
            level = _builder.CreateVectorSplat(
                _width, scalar_level,
                "texture.sample.uniform.gradient.minimum.splat");
            level = _builder.CreateSelect(
                gradient_mask, level, zero_float);
        } else {
            level = _as_lane_vector(raw_level, *level_value);
            level = _builder.CreateSelect(
                has_gradient_level ? gradient_mask : _active_mask,
                level, zero_float);
            if (has_gradient_level) {
                if (gradient_level_is_uniform) {
                    gradient_level = _builder.CreateVectorSplat(
                        _width, uniform_gradient_level,
                        "texture.sample.uniform.gradient.splat");
                }
                level = _builder.CreateMaxNum(gradient_level, level);
                level = _builder.CreateSelect(
                    gradient_mask, level, zero_float);
            }
        }
        auto *level_scratch = _entry_scratch(
            float_lanes, "texture.sample.levels");
        _builder.CreateStore(level, level_scratch);
        level_pointer = level_scratch;
        operand_index++;
    } else if (has_gradient) {
        if (gradient_level_is_uniform) {
            gradient_level = _builder.CreateVectorSplat(
                _width, uniform_gradient_level,
                "texture.sample.uniform.gradient.splat");
            gradient_level = _builder.CreateSelect(
                gradient_mask, gradient_level, zero_float,
                "texture.sample.safe.gradient.level");
        }
        auto *level_scratch = _entry_scratch(
            float_lanes, "texture.sample.gradient.levels");
        _builder.CreateStore(gradient_level, level_scratch);
        level_pointer = level_scratch;
    }

    auto *filter_value = _source.value(
        instruction.operands[operand_index]);
    auto *address_value = _source.value(
        instruction.operands[operand_index + 1u]);
    auto *filter = filter_value == nullptr ? nullptr :
                                             _as_lane_vector(
                                                 _load_value(instruction.operands[operand_index]),
                                                 *filter_value);
    auto *address = address_value == nullptr ? nullptr :
                                               _as_lane_vector(
                                                   _load_value(instruction.operands[operand_index + 1u]),
                                                   *address_value);
    if (filter_value == nullptr || address_value == nullptr ||
        filter == nullptr || address == nullptr ||
        filter_value->type == nullptr || address_value->type == nullptr ||
        !filter_value->type->is_scalar() ||
        !address_value->type->is_scalar() ||
        filter_value->type->is_float() ||
        address_value->type->is_float()) {
        _fail("direct texture sample has an invalid sampler");
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
        _active_mask, sampler, zero_i32,
        "texture.sample.safe.sampler");
    auto *sampler_scratch = _entry_scratch(
        i32_lanes, "texture.sample.samplers");
    _builder.CreateStore(sampler, sampler_scratch);

    auto *scratch_type = ::llvm::ArrayType::get(float_lanes, 4u);
    auto *scratch = _entry_scratch(
        scratch_type,
        "texture.sample.result." +
            std::to_string(instruction.result->value));
    _builder.CreateStore(
        ::llvm::Constant::getNullValue(scratch_type), scratch);
    auto *callback_type = ::llvm::FunctionType::get(
        _builder.getVoidTy(),
        {pointer_type, _builder.getInt32Ty(),
         _builder.getInt32Ty(), _builder.getInt32Ty(),
         _builder.getInt64Ty(), pointer_type, pointer_type,
         pointer_type, pointer_type, pointer_type, pointer_type},
        false);
    auto *active_mask_bits = _bindless_callback_mask(varying_result);
    _builder.CreateCall(
        callback_type, sample_callback,
        {object, base_level, _builder.getInt32(dimension),
         _builder.getInt32(_width), active_mask_bits,
         sampler_scratch, coordinate_scratch[0u],
         coordinate_scratch[1u], coordinate_scratch[2u],
         level_pointer, scratch});

    auto *pixels = static_cast<::llvm::Value *>(
        ::llvm::PoisonValue::get(_data_type(result->type, true)));
    for (auto component = uint32_t{0u}; component < 4u;
         component++) {
        auto *component_pointer = _builder.CreateGEP(
            scratch_type, scratch,
            {_builder.getInt32(0u), _builder.getInt32(component)});
        pixels = _insert_child(
            pixels, _builder.CreateLoad(float_lanes, component_pointer),
            result->type, component, true);
    }
    return varying_result ?
               pixels :
               _extract_lane(
                   pixels, result->type,
                   _safe_first_lane(_active_mask));
}

}// namespace luisa::compute::simd::detail
