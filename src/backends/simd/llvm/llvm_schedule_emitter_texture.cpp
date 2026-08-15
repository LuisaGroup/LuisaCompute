#include "llvm_schedule_emitter.h"

#include <array>

#include "../../common/llvm_native_math.h"

namespace luisa::compute::simd::detail {

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
