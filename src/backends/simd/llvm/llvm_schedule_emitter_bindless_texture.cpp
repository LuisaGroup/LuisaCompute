#include "llvm_schedule_emitter.h"

#include <limits>

#include <llvm/IR/MDBuilder.h>

#include "../../common/env_flag.h"
#include "../../common/llvm_native_math.h"
#include <luisa/core/stl/string.h>

namespace luisa::compute::simd::detail {

void emit_byte1_linear_mirror_sample(
    ::llvm::IRBuilder<> &builder, uint32_t width,
    ::llvm::Value *active_mask, ::llvm::Value *data,
    ::llvm::Value *width_value, ::llvm::Value *height_value,
    ::llvm::Value *u, ::llvm::Value *v,
    ::llvm::AllocaInst *scratch,
    ::llvm::ArrayType *scratch_type,
    bool enable_wide_gathers, bool little_endian);

[[nodiscard]] ::llvm::Function *get_or_create_w1_rgba_linear_sample_helper(
    ::llvm::Module &module, bool is_float4, bool enable_fast_math);

[[nodiscard]] ::llvm::Function *get_or_create_wide_rgba_linear_sample_helper(
    ::llvm::Module &module, uint32_t width, bool is_float4, bool little_endian);

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
    auto has_gradient_level =
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL ||
        op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL ||
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
    auto *zero_i32 = ::llvm::Constant::getNullValue(i32_lanes);

    if (size_query) {
        auto *slot_scratch = _entry_scratch(
            i32_lanes, "bindless.texture.query.slots");
        _builder.CreateStore(safe_slot_indices, slot_scratch);
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
        3u + (has_gradient ? 2u : 0u) +
        ((has_level || has_gradient_level) ? 1u : 0u) +
        (explicit_sampler ? 2u : 0u);
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

    auto varying_result = result->value_class ==
                          schedule::ValueClass::varying;
    auto *float_lanes = ::llvm::FixedVectorType::get(
        _builder.getFloatTy(), _width);
    auto *zero_float = ::llvm::Constant::getNullValue(float_lanes);
    auto *one_float = ::llvm::ConstantVector::getSplat(
        ::llvm::ElementCount::getFixed(_width),
        ::llvm::ConstantFP::get(_builder.getFloatTy(), 1.0));
    auto *slot_value = _source.value(instruction.operands[1u]);
    const bool direct_mip0_candidate =
        dimension == 2u && !has_level && !has_gradient &&
        (!explicit_sampler || _width == 1u) && varying_result &&
        slot_value != nullptr &&
        !luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_IR_BYTE1_TEXTURE_SAMPLING");
    if (_width == 1u && direct_mip0_candidate) {
        auto *u_lanes = _extract_child(
            coordinate, coordinate_value->type, 0u, true);
        auto *v_lanes = _extract_child(
            coordinate, coordinate_value->type, 1u, true);
        ::llvm::Value *sampler = nullptr;
        ::llvm::Value *explicit_sampler_scalar = nullptr;
        if (explicit_sampler) {
            auto *filter_value = _source.value(instruction.operands[3u]);
            auto *address_value = _source.value(instruction.operands[4u]);
            auto *filter = filter_value == nullptr ?
                               nullptr :
                               _as_lane_vector(
                                   _load_value(instruction.operands[3u]),
                                   *filter_value);
            auto *address = address_value == nullptr ?
                                nullptr :
                                _as_lane_vector(
                                    _load_value(instruction.operands[4u]),
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
            sampler = _builder.CreateOr(
                _builder.CreateShl(
                    filter,
                    _builder.CreateVectorSplat(_width, _builder.getInt32(2u))),
                address);
            sampler = _builder.CreateSelect(_active_mask, sampler, zero_i32);
            explicit_sampler_scalar = _builder.CreateExtractElement(
                sampler, _builder.getInt32(0u),
                "bindless.texture.explicit.sampler.scalar");
        }
        auto *any_active = _builder.CreateExtractElement(
            _active_mask, _builder.getInt32(0u));
        auto *inspect_bb = ::llvm::BasicBlock::Create(
            context, "bindless.texture.w1.mip0.inspect", _entry);
        auto *direct_byte4 = ::llvm::BasicBlock::Create(
            context, "bindless.texture.w1.byte4.direct", _entry);
        auto *check_float4 = ::llvm::BasicBlock::Create(
            context, "bindless.texture.w1.float4.check", _entry);
        auto *direct_float4 = ::llvm::BasicBlock::Create(
            context, "bindless.texture.w1.float4.direct", _entry);
        auto *fallback = ::llvm::BasicBlock::Create(
            context, "bindless.texture.w1.mip0.callback", _entry);
        auto *merge = ::llvm::BasicBlock::Create(
            context, "bindless.texture.w1.mip0.merge", _entry);

        auto *pre_active_bb = _builder.GetInsertBlock();
        auto *active_br = _builder.CreateCondBr(any_active, inspect_bb, merge);
        active_br->setMetadata(
            ::llvm::LLVMContext::MD_prof,
            ::llvm::MDBuilder{context}.createBranchWeights(1000u, 1u));

        _builder.SetInsertPoint(inspect_bb);
        auto *scalar_slot = _builder.CreateExtractElement(
            array.slot_indices, _builder.getInt32(0u));
        auto *slot_base = _builder.CreateGEP(
            _builder.getInt8Ty(), array.slots,
            _builder.CreateMul(
                scalar_slot,
                _builder.getInt64(sizeof(SIMDHostBindlessSlot))));
        auto *descriptor = _builder.CreateGEP(
            _builder.getInt8Ty(), slot_base,
            _builder.getInt64(offsetof(SIMDHostBindlessSlot, texture2d)));
        auto *data = _builder.CreateLoad(
            pointer_type,
            _builder.CreateGEP(
                _builder.getInt8Ty(), descriptor,
                _builder.getInt64(
                    offsetof(SIMDHostBindlessTextureSlot, byte1_mip0))),
            "bindless.texture.mip0.data");
        auto *metadata = _builder.CreateLoad(
            _builder.getInt64Ty(),
            _builder.CreateGEP(
                _builder.getInt8Ty(), descriptor,
                _builder.getInt64(
                    offsetof(SIMDHostBindlessTextureSlot, metadata))),
            "bindless.texture.mip0.metadata");
        auto *sampler_code =
            explicit_sampler_scalar != nullptr ?
                explicit_sampler_scalar :
                _builder.CreateTrunc(
                    _builder.CreateAnd(metadata, _builder.getInt64(0x0fu)),
                    _builder.getInt32Ty());
        auto *effective_metadata =
            explicit_sampler_scalar != nullptr ?
                _builder.CreateOr(
                    _builder.CreateAnd(
                        metadata, _builder.getInt64(~uint64_t{0x0fu})),
                    _builder.CreateZExt(
                        _builder.CreateAnd(
                            explicit_sampler_scalar, _builder.getInt32(0x0fu)),
                        _builder.getInt64Ty())) :
                metadata;
        auto *mip0_format = _builder.CreateTrunc(
            _builder.CreateLShr(
                metadata,
                _builder.getInt64(simd_bindless_texture_mip0_format_shift)),
            _builder.getInt32Ty());
        auto *address_mode = _builder.CreateAnd(
            sampler_code, _builder.getInt32(0x03u));
        auto *linear_non_zero = _builder.CreateAnd(
            _builder.CreateICmpNE(data, null_pointer),
            _builder.CreateAnd(
                _builder.CreateICmpUGE(sampler_code, _builder.getInt32(4u)),
                _builder.CreateICmpNE(address_mode, _builder.getInt32(3u))));
        auto *byte4_eligible = _builder.CreateAnd(
            linear_non_zero,
            _builder.CreateICmpEQ(
                mip0_format,
                _builder.getInt32(static_cast<uint32_t>(
                    simd_bindless_texture_mip0_format_byte4))),
            "bindless.texture.byte4.eligible");
        auto *float4_eligible = _builder.CreateAnd(
            linear_non_zero,
            _builder.CreateICmpEQ(
                mip0_format,
                _builder.getInt32(static_cast<uint32_t>(
                    simd_bindless_texture_mip0_format_float4))),
            "bindless.texture.float4.eligible");
        auto *byte4_br = _builder.CreateCondBr(
            byte4_eligible, direct_byte4, check_float4);
        byte4_br->setMetadata(
            ::llvm::LLVMContext::MD_prof,
            ::llvm::MDBuilder{context}.createBranchWeights(1000000u, 1u));

        auto emit_w1_helper_call = [&](bool is_float4) noexcept {
            auto *helper = get_or_create_w1_rgba_linear_sample_helper(
                _module, is_float4, _enable_fast_math);
            auto *u_scalar = _builder.CreateExtractElement(
                u_lanes, _builder.getInt32(0u));
            auto *v_scalar = _builder.CreateExtractElement(
                v_lanes, _builder.getInt32(0u));
            auto *call = _builder.CreateCall(
                helper, {data, effective_metadata, u_scalar, v_scalar});
            call->setCallingConv(::llvm::CallingConv::Fast);
            return call;
        };

        _builder.SetInsertPoint(direct_byte4);
        auto *byte4_res = emit_w1_helper_call(false);
        _builder.CreateBr(merge);

        _builder.SetInsertPoint(check_float4);
        auto *float4_br = _builder.CreateCondBr(
            float4_eligible, direct_float4, fallback);
        float4_br->setMetadata(
            ::llvm::LLVMContext::MD_prof,
            ::llvm::MDBuilder{context}.createBranchWeights(1000000u, 1u));

        _builder.SetInsertPoint(direct_float4);
        auto *float4_res = emit_w1_helper_call(true);
        _builder.CreateBr(merge);

        _builder.SetInsertPoint(fallback);
        if (_w1_bindless_slot_scratch == nullptr) {
            _w1_bindless_slot_scratch = _entry_scratch(
                i32_lanes, "w1.bindless.texture.query.slots");
        }
        for (uint32_t axis = 0u; axis < 3u; ++axis) {
            if (_w1_bindless_coord_scratch[axis] == nullptr) {
                _w1_bindless_coord_scratch[axis] = _entry_scratch(
                    float_lanes,
                    "w1.bindless.texture.sample.coordinate." +
                        std::to_string(axis));
            }
        }
        ::llvm::Value *fallback_sampler_ptr = null_pointer;
        if (explicit_sampler) {
            if (_w1_bindless_sampler_scratch == nullptr) {
                _w1_bindless_sampler_scratch = _entry_scratch(
                    i32_lanes, "w1.bindless.texture.sample.samplers");
            }
            _builder.CreateStore(sampler, _w1_bindless_sampler_scratch);
            fallback_sampler_ptr = _w1_bindless_sampler_scratch;
        }
        auto *scratch_type = ::llvm::ArrayType::get(float_lanes, 4u);
        if (_w1_bindless_result_scratch == nullptr) {
            _w1_bindless_result_scratch = _entry_scratch(
                scratch_type, "w1.bindless.texture.sample.result");
        }
        _builder.CreateStore(safe_slot_indices, _w1_bindless_slot_scratch);
        _builder.CreateStore(
            _builder.CreateSelect(_active_mask, u_lanes, zero_float),
            _w1_bindless_coord_scratch[0u]);
        _builder.CreateStore(
            _builder.CreateSelect(_active_mask, v_lanes, zero_float),
            _w1_bindless_coord_scratch[1u]);
        _builder.CreateStore(zero_float, _w1_bindless_coord_scratch[2u]);
        _builder.CreateStore(
            ::llvm::Constant::getNullValue(scratch_type),
            _w1_bindless_result_scratch);

        auto *callback = _builder.CreateExtractValue(array.view, {2u});
        auto *callback_type = ::llvm::FunctionType::get(
            _builder.getVoidTy(),
            {pointer_type, _builder.getInt64Ty(),
             _builder.getInt32Ty(), _builder.getInt32Ty(),
             _builder.getInt64Ty(), pointer_type, pointer_type,
             pointer_type, pointer_type, pointer_type,
             pointer_type, pointer_type},
            false);
        auto *missing_callback = _builder.CreateICmpEQ(callback, null_pointer);
        _trap_if(
            _builder.CreateAnd(
                _builder.CreateOrReduce(_active_mask), missing_callback),
            "bindless.texture.sample.callback.null");
        auto *active_mask_bits = _bindless_callback_mask(true);
        _builder.CreateCall(
            callback_type, callback,
            {array.slots, array.slot_count,
             _builder.getInt32(dimension), _builder.getInt32(_width),
             active_mask_bits, _w1_bindless_slot_scratch,
             fallback_sampler_ptr,
             _w1_bindless_coord_scratch[0u],
             _w1_bindless_coord_scratch[1u],
             _w1_bindless_coord_scratch[2u],
             null_pointer, _w1_bindless_result_scratch});
        auto *f32x4_ty = ::llvm::FixedVectorType::get(
            _builder.getFloatTy(), 4u);
        ::llvm::Value *fallback_res = ::llvm::PoisonValue::get(f32x4_ty);
        for (uint32_t ch = 0u; ch < 4u; ++ch) {
            auto *comp_ptr = _builder.CreateGEP(
                scratch_type, _w1_bindless_result_scratch,
                {_builder.getInt32(0u), _builder.getInt32(ch)});
            auto *comp_v1 = _builder.CreateLoad(float_lanes, comp_ptr);
            auto *comp_scalar = _builder.CreateExtractElement(
                comp_v1, _builder.getInt32(0u));
            fallback_res = _builder.CreateInsertElement(
                fallback_res, comp_scalar, _builder.getInt32(ch));
        }
        auto *fallback_end_bb = _builder.GetInsertBlock();
        _builder.CreateBr(merge);

        _builder.SetInsertPoint(merge);
        auto *res_phi = _builder.CreatePHI(
            f32x4_ty, 4u, "bindless.texture.mip0.w1.res");
        res_phi->addIncoming(
            ::llvm::Constant::getNullValue(f32x4_ty), pre_active_bb);
        res_phi->addIncoming(byte4_res, direct_byte4);
        res_phi->addIncoming(float4_res, direct_float4);
        res_phi->addIncoming(fallback_res, fallback_end_bb);

        auto *pixels = static_cast<::llvm::Value *>(
            ::llvm::PoisonValue::get(_data_type(result->type, true)));
        for (uint32_t ch = 0u; ch < 4u; ++ch) {
            auto *elem = _builder.CreateExtractElement(
                res_phi, _builder.getInt32(ch));
            auto *v1 = _builder.CreateInsertElement(
                ::llvm::PoisonValue::get(float_lanes), elem,
                _builder.getInt32(0u));
            pixels = _insert_child(pixels, v1, result->type, ch, true);
        }
        return pixels;
    }

    auto *slot_scratch = _entry_scratch(
        i32_lanes, "bindless.texture.query.slots");
    _builder.CreateStore(safe_slot_indices, slot_scratch);
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

    auto operand_index = size_t{3u};
    ::llvm::Value *gradient_level = nullptr;
    ::llvm::Value *uniform_gradient_level = nullptr;
    ::llvm::Value *gradient_mask = nullptr;
    auto gradient_level_is_uniform = false;
    if (has_gradient) {
        auto *slot_value = _source.value(instruction.operands[1u]);
        std::array<const schedule::Value *, 2u> gradient_values{};
        auto uniform_gradient_lod =
            slot_value != nullptr &&
            schedule::is_uniform(slot_value->value_class);
        for (auto derivative = 0u; derivative < 2u; derivative++) {
            auto *gradient_value = _source.value(
                instruction.operands[operand_index + derivative]);
            gradient_values[derivative] = gradient_value;
            if (gradient_value == nullptr ||
                gradient_value->type == nullptr ||
                !gradient_value->type->is_vector() ||
                gradient_value->type->dimension() != dimension ||
                !gradient_value->type->element()->is_float32()) {
                _fail("bindless texture sample has an invalid gradient");
                return nullptr;
            }
            uniform_gradient_lod &=
                schedule::is_uniform(gradient_value->value_class);
        }
        constexpr auto texture_offset =
            offsetof(SIMDHostBindlessSlot, texture2d);
        constexpr auto texture_stride =
            offsetof(SIMDHostBindlessSlot, texture3d) -
            offsetof(SIMDHostBindlessSlot, texture2d);
        auto descriptor_offset = texture_offset +
                                 (dimension - 2u) * texture_stride;
        auto *first_active_mask = _builder.CreateAnd(
            _active_mask,
            _builder.CreateICmpEQ(
                _lane_ids(), _builder.CreateVectorSplat(
                                 _width, _safe_first_lane(_active_mask))));
        gradient_mask = varying_result ? _active_mask : first_active_mask;
        auto slot_uniform = schedule::is_uniform(slot_value->value_class);
        std::array<::llvm::Value *, 3u> extents{};
        auto *i64_lanes = ::llvm::FixedVectorType::get(
            _builder.getInt64Ty(), _width);
        auto metadata_offset = descriptor_offset +
                               offsetof(SIMDHostBindlessTextureSlot, metadata);
        ::llvm::Value *metadata = nullptr;
        if (slot_uniform) {
            auto *scalar_slot = _builder.CreateExtractElement(
                array.slot_indices, _safe_first_lane(_active_mask));
            auto *metadata_pointer = _builder.CreateGEP(
                _builder.getInt8Ty(), array.slots,
                _builder.CreateAdd(
                    _builder.CreateMul(
                        scalar_slot,
                        _builder.getInt64(
                            sizeof(SIMDHostBindlessSlot))),
                    _builder.getInt64(metadata_offset)));
            auto *zero_metadata = _entry_scratch(
                _builder.getInt64Ty(),
                "bindless.texture.sample.uniform.metadata.zero");
            _builder.CreateStore(
                _builder.getInt64(0u), zero_metadata);
            auto *safe_metadata_pointer = _builder.CreateSelect(
                _builder.CreateOrReduce(_active_mask),
                metadata_pointer, zero_metadata,
                "bindless.texture.sample.uniform.metadata.pointer");
            auto *metadata_load = _builder.CreateLoad(
                _builder.getInt64Ty(), safe_metadata_pointer,
                "bindless.texture.sample.uniform.metadata");
            metadata_load->setAlignment(
                ::llvm::Align{alignof(uint64_t)});
            metadata = metadata_load;
        } else {
            auto *slot_offsets = _builder.CreateMul(
                array.slot_indices,
                _builder.CreateVectorSplat(
                    _width,
                    _builder.getInt64(sizeof(SIMDHostBindlessSlot))));
            auto *slot_bases = _builder.CreateGEP(
                _builder.getInt8Ty(),
                _builder.CreateVectorSplat(_width, array.slots),
                slot_offsets);
            auto *extent_pointers = _builder.CreateGEP(
                _builder.getInt8Ty(), slot_bases,
                _builder.CreateVectorSplat(
                    _width, _builder.getInt64(metadata_offset)));
            metadata = _builder.CreateMaskedGather(
                i64_lanes, extent_pointers,
                ::llvm::Align{alignof(uint64_t)}, gradient_mask,
                ::llvm::Constant::getNullValue(i64_lanes));
        }
        for (auto axis = 0u; axis < dimension; axis++) {
            auto shift = 4u +
                         axis * simd_bindless_texture_extent_bits;
            if (slot_uniform) {
                extents[axis] = _builder.CreateUIToFP(
                    _builder.CreateTrunc(
                        _builder.CreateAnd(
                            _builder.CreateLShr(
                                metadata, _builder.getInt64(shift)),
                            _builder.getInt64(
                                simd_bindless_texture_extent_mask)),
                        _builder.getInt32Ty()),
                    _builder.getFloatTy());
                if (!uniform_gradient_lod) {
                    extents[axis] = _builder.CreateVectorSplat(
                        _width, extents[axis]);
                }
            } else {
                extents[axis] = _builder.CreateUIToFP(
                    _builder.CreateTrunc(
                        _builder.CreateAnd(
                            _builder.CreateLShr(
                                metadata,
                                _builder.CreateVectorSplat(
                                    _width, _builder.getInt64(shift))),
                            _builder.CreateVectorSplat(
                                _width, _builder.getInt64(
                                            simd_bindless_texture_extent_mask))),
                        i32_lanes),
                    float_lanes);
            }
        }
        auto native_math_mode = _enable_fast_math ?
                                    cpu::LLVMNativeMathMode::fast :
                                    cpu::LLVMNativeMathMode::precise;
        if (uniform_gradient_lod) {
            gradient_level_is_uniform = true;
            auto *zero_scalar = ::llvm::ConstantFP::get(
                _builder.getFloatTy(), 0.0);
            auto *one_scalar = ::llvm::ConstantFP::get(
                _builder.getFloatTy(), 1.0);
            std::array<::llvm::Value *, 2u> gradient_norms{
                zero_scalar, zero_scalar};
            ::llvm::Value *gradient_has_nan =
                _builder.getFalse();
            for (auto derivative = 0u; derivative < 2u; derivative++) {
                auto *gradient = _load_value(
                    instruction.operands[operand_index + derivative]);
                if (gradient == nullptr) { return nullptr; }
                for (auto axis = 0u; axis < dimension; axis++) {
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
                "bindless.texture.sample.uniform.nan.gradient.lod");
            uniform_gradient_level = _builder.CreateUnaryIntrinsic(
                ::llvm::Intrinsic::log2, rho2,
                nullptr, "bindless.texture.sample.uniform.gradient.log2");
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
            for (auto derivative = 0u; derivative < 2u; derivative++) {
                auto *gradient = _as_lane_vector(
                    _load_value(instruction.operands[operand_index + derivative]),
                    *gradient_values[derivative]);
                if (gradient == nullptr) { return nullptr; }
                for (auto axis = 0u; axis < dimension; axis++) {
                    auto *lanes = _extract_child(
                        gradient, gradient_values[derivative]->type,
                        axis, true);
                    lanes = _builder.CreateSelect(
                        gradient_mask, lanes, zero_float,
                        "bindless.texture.sample.safe.gradient");
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
                "bindless.texture.sample.nan.gradient.lod");
            gradient_level = cpu::LLVMNativeMath::emit_log2_f32(
                _module, _builder, rho2, native_math_mode);
            if (gradient_level == nullptr) {
                _fail("bindless texture gradient LOD requires fixed f32 vectors");
                return nullptr;
            }
            gradient_level = _builder.CreateFMul(
                gradient_level,
                ::llvm::ConstantVector::getSplat(
                    ::llvm::ElementCount::getFixed(_width),
                    ::llvm::ConstantFP::get(
                        _builder.getFloatTy(), 0.5)));
        }
        if (!gradient_level_is_uniform) {
            gradient_level = _builder.CreateSelect(
                gradient_mask, gradient_level, zero_float,
                "bindless.texture.sample.safe.gradient.level");
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
                    "bindless texture sample has an invalid minimum mip level" :
                    "bindless texture sample has an invalid mip level");
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
                "bindless.texture.sample.uniform.gradient.minimum.splat");
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
                        "bindless.texture.sample.uniform.gradient.splat");
                }
                level = _builder.CreateMaxNum(
                    gradient_level, level);
                level = _builder.CreateSelect(
                    gradient_mask, level, zero_float);
            }
        }
        auto *level_scratch = _entry_scratch(
            float_lanes, "bindless.texture.sample.levels");
        _builder.CreateStore(level, level_scratch);
        level_pointer = level_scratch;
        operand_index++;
    } else if (has_gradient) {
        if (gradient_level_is_uniform) {
            gradient_level = _builder.CreateVectorSplat(
                _width, uniform_gradient_level,
                "bindless.texture.sample.uniform.gradient.splat");
            gradient_level = _builder.CreateSelect(
                gradient_mask, gradient_level, zero_float,
                "bindless.texture.sample.safe.gradient.level");
        }
        auto *level_scratch = _entry_scratch(
            float_lanes, "bindless.texture.sample.gradient.levels");
        _builder.CreateStore(gradient_level, level_scratch);
        level_pointer = level_scratch;
    }

    ::llvm::Value *sampler_pointer = null_pointer;
    ::llvm::Value *explicit_sampler_scalar = nullptr;
    if (explicit_sampler) {
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
        if (_width == 1u) {
            explicit_sampler_scalar = _builder.CreateExtractElement(
                sampler, _builder.getInt32(0u),
                "bindless.texture.explicit.sampler.scalar");
        }
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
    auto *callback_type = ::llvm::FunctionType::get(
        _builder.getVoidTy(),
        {pointer_type, _builder.getInt64Ty(),
         _builder.getInt32Ty(), _builder.getInt32Ty(),
         _builder.getInt64Ty(), pointer_type, pointer_type,
         pointer_type, pointer_type, pointer_type,
         pointer_type, pointer_type},
        false);
    auto emit_callback = [&] {
        auto *missing_callback = _builder.CreateICmpEQ(
            callback, null_pointer);
        _trap_if(
            _builder.CreateAnd(
                _builder.CreateOrReduce(_active_mask),
                missing_callback),
            "bindless.texture.sample.callback.null");
        auto *active_mask_bits = _bindless_callback_mask(
            result->value_class == schedule::ValueClass::varying);
        _builder.CreateCall(
            callback_type, callback,
            {array.slots, array.slot_count,
             _builder.getInt32(dimension), _builder.getInt32(_width),
             active_mask_bits, slot_scratch, sampler_pointer,
             coordinate_scratch[0u], coordinate_scratch[1u],
             coordinate_scratch[2u], level_pointer, scratch});
    };

    const auto slot_is_uniform =
        _width == 1u ||
        (slot_value != nullptr && schedule::is_uniform(slot_value->value_class));
    if (direct_mip0_candidate) {
        auto *any_active = _width == 1u ?
                               _builder.CreateExtractElement(_active_mask, _builder.getInt32(0u)) :
                               _builder.CreateOrReduce(_active_mask);
        auto *inspect_bb = ::llvm::BasicBlock::Create(
            _module.getContext(), "bindless.texture.mip0.inspect", _entry);
        auto *direct_byte4 = ::llvm::BasicBlock::Create(
            _module.getContext(), "bindless.texture.byte4.direct", _entry);
        auto *check_float4 = ::llvm::BasicBlock::Create(
            _module.getContext(), "bindless.texture.float4.check", _entry);
        auto *direct_float4 = ::llvm::BasicBlock::Create(
            _module.getContext(), "bindless.texture.float4.direct", _entry);
        auto *check_byte1 = ::llvm::BasicBlock::Create(
            _module.getContext(), "bindless.texture.byte1.check", _entry);
        auto *direct_byte1 = ::llvm::BasicBlock::Create(
            _module.getContext(), "bindless.texture.byte1.direct", _entry);
        auto *fallback = ::llvm::BasicBlock::Create(
            _module.getContext(), "bindless.texture.mip0.callback", _entry);
        auto *merge = ::llvm::BasicBlock::Create(
            _module.getContext(), "bindless.texture.mip0.merge", _entry);

        auto *active_br = _builder.CreateCondBr(any_active, inspect_bb, merge);
        active_br->setMetadata(
            ::llvm::LLVMContext::MD_prof,
            ::llvm::MDBuilder{_module.getContext()}.createBranchWeights(1000u, 1u));

        _builder.SetInsertPoint(inspect_bb);
        auto *first_active_lane = _safe_first_lane(_active_mask);
        auto *scalar_slot = _builder.CreateExtractElement(
            array.slot_indices, first_active_lane);
        ::llvm::Value *all_slots_equal = _builder.getTrue();
        if (!slot_is_uniform) {
            auto *splat_slot = _builder.CreateVectorSplat(_width, scalar_slot);
            auto *same_slot = _builder.CreateICmpEQ(
                array.slot_indices, splat_slot);
            auto *all_true_lanes = ::llvm::ConstantVector::getSplat(
                ::llvm::ElementCount::getFixed(_width), _builder.getTrue());
            auto *active_same = _builder.CreateSelect(
                _active_mask, same_slot, all_true_lanes,
                "bindless.texture.mip0.slots.active.same");
            all_slots_equal = _builder.CreateAndReduce(active_same);
        }
        auto *slot_base = _builder.CreateGEP(
            _builder.getInt8Ty(), array.slots,
            _builder.CreateMul(
                scalar_slot,
                _builder.getInt64(sizeof(SIMDHostBindlessSlot))));
        auto *descriptor = _builder.CreateGEP(
            _builder.getInt8Ty(), slot_base,
            _builder.getInt64(
                offsetof(SIMDHostBindlessSlot, texture2d)));
        auto *data_pointer = _builder.CreateGEP(
            _builder.getInt8Ty(), descriptor,
            _builder.getInt64(offsetof(
                SIMDHostBindlessTextureSlot, byte1_mip0)));
        auto *data = _builder.CreateLoad(
            pointer_type, data_pointer,
            "bindless.texture.mip0.data");
        auto *metadata_pointer = _builder.CreateGEP(
            _builder.getInt8Ty(), descriptor,
            _builder.getInt64(offsetof(
                SIMDHostBindlessTextureSlot, metadata)));
        auto *metadata = _builder.CreateLoad(
            _builder.getInt64Ty(), metadata_pointer,
            "bindless.texture.mip0.metadata");
        auto extent = [&](uint32_t shift,
                          luisa::string_view name) {
            return _builder.CreateTrunc(
                _builder.CreateAnd(
                    _builder.CreateLShr(
                        metadata, _builder.getInt64(shift)),
                    _builder.getInt64(
                        simd_bindless_texture_extent_mask)),
                _builder.getInt32Ty(), name);
        };
        auto *width = extent(
            4u, "bindless.texture.mip0.width");
        auto *height = extent(
            24u, "bindless.texture.mip0.height");
        auto *sampler_code =
            explicit_sampler_scalar != nullptr ?
                explicit_sampler_scalar :
                _builder.CreateTrunc(
                    _builder.CreateAnd(metadata, _builder.getInt64(0x0fu)),
                    _builder.getInt32Ty());
        auto *effective_metadata =
            explicit_sampler_scalar != nullptr ?
                _builder.CreateOr(
                    _builder.CreateAnd(
                        metadata, _builder.getInt64(~uint64_t{0x0fu})),
                    _builder.CreateZExt(
                        _builder.CreateAnd(
                            explicit_sampler_scalar, _builder.getInt32(0x0fu)),
                        _builder.getInt64Ty())) :
                metadata;
        auto *mip0_format = _builder.CreateTrunc(
            _builder.CreateLShr(
                metadata,
                _builder.getInt64(simd_bindless_texture_mip0_format_shift)),
            _builder.getInt32Ty());
        auto *has_valid_mip0 = _builder.CreateAnd(
            all_slots_equal,
            _builder.CreateAnd(
                _builder.CreateICmpNE(data, null_pointer),
                _builder.CreateAnd(
                    _builder.CreateICmpNE(width, _builder.getInt32(0u)),
                    _builder.CreateICmpNE(height, _builder.getInt32(0u)))));
        auto *address_mode = _builder.CreateAnd(
            sampler_code, _builder.getInt32(0x03u));
        auto *linear_non_zero = _builder.CreateAnd(
            has_valid_mip0,
            _builder.CreateAnd(
                _builder.CreateICmpUGE(sampler_code, _builder.getInt32(4u)),
                _builder.CreateICmpNE(address_mode, _builder.getInt32(3u))));
        auto *byte4_eligible = _builder.CreateAnd(
            linear_non_zero,
            _builder.CreateICmpEQ(
                mip0_format,
                _builder.getInt32(static_cast<uint32_t>(
                    simd_bindless_texture_mip0_format_byte4))),
            "bindless.texture.byte4.eligible");
        auto *float4_eligible = _builder.CreateAnd(
            linear_non_zero,
            _builder.CreateICmpEQ(
                mip0_format,
                _builder.getInt32(static_cast<uint32_t>(
                    simd_bindless_texture_mip0_format_float4))),
            "bindless.texture.float4.eligible");
        auto *byte1_eligible = _builder.CreateAnd(
            has_valid_mip0,
            _builder.CreateAnd(
                _builder.CreateICmpEQ(
                    mip0_format,
                    _builder.getInt32(static_cast<uint32_t>(
                        simd_bindless_texture_mip0_format_byte1))),
                _builder.CreateICmpEQ(
                    sampler_code,
                    _builder.getInt32(
                        simd_bindless_linear_point_mirror_sampler_code))),
            "bindless.texture.byte1.eligible");

        auto emit_direct_rgba = [&](bool is_float4) {
            auto *u_lanes = _builder.CreateLoad(float_lanes, coordinate_scratch[0u]);
            auto *v_lanes = _builder.CreateLoad(float_lanes, coordinate_scratch[1u]);
            if (_width == 1u) {
                auto *helper = get_or_create_w1_rgba_linear_sample_helper(
                    _module, is_float4, _enable_fast_math);
                auto *u_scalar = _builder.CreateExtractElement(
                    u_lanes, _builder.getInt32(0u));
                auto *v_scalar = _builder.CreateExtractElement(
                    v_lanes, _builder.getInt32(0u));
                auto *call = _builder.CreateCall(
                    helper,
                    {data, effective_metadata, u_scalar, v_scalar});
                call->setCallingConv(::llvm::CallingConv::Fast);
                for (uint32_t ch = 0u; ch < 4u; ++ch) {
                    auto *elem = _builder.CreateExtractElement(
                        call, _builder.getInt32(ch));
                    auto *v1 = _builder.CreateInsertElement(
                        ::llvm::PoisonValue::get(float_lanes), elem,
                        _builder.getInt32(0u));
                    auto *slot_ptr = _builder.CreateGEP(
                        scratch_type, scratch,
                        {_builder.getInt32(0u), _builder.getInt32(ch)});
                    _builder.CreateStore(v1, slot_ptr);
                }
            } else {
                auto *helper = get_or_create_wide_rgba_linear_sample_helper(
                    _module, _width, is_float4,
                    _module.getDataLayout().isLittleEndian());
                auto *call = _builder.CreateCall(
                    helper,
                    {data, width, height, address_mode,
                     u_lanes, v_lanes, _active_mask, scratch});
                call->setCallingConv(::llvm::CallingConv::Fast);
            }
        };

        _builder.CreateCondBr(byte4_eligible, direct_byte4, check_float4);

        _builder.SetInsertPoint(direct_byte4);
        emit_direct_rgba(false);
        _builder.CreateBr(merge);

        _builder.SetInsertPoint(check_float4);
        _builder.CreateCondBr(float4_eligible, direct_float4, check_byte1);

        _builder.SetInsertPoint(direct_float4);
        emit_direct_rgba(true);
        _builder.CreateBr(merge);

        _builder.SetInsertPoint(check_byte1);
        _builder.CreateCondBr(byte1_eligible, direct_byte1, fallback);

        _builder.SetInsertPoint(direct_byte1);
        emit_byte1_linear_mirror_sample(
            _builder, _width, _active_mask,
            data, width, height,
            _builder.CreateLoad(
                float_lanes, coordinate_scratch[0u]),
            _builder.CreateLoad(
                float_lanes, coordinate_scratch[1u]),
            scratch, scratch_type,
            (_width == 4u || _width == 8u) &&
                !luisa::compute::detail::env_flag(
                    "LUISA_SIMD_DISABLE_WIDE_BYTE1_GATHERS"),
            _module.getDataLayout().isLittleEndian());
        _builder.CreateBr(merge);

        _builder.SetInsertPoint(fallback);
        emit_callback();
        _builder.CreateBr(merge);
        _builder.SetInsertPoint(merge);
    } else {
        emit_callback();
    }
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
