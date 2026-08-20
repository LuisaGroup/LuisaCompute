//
// Created by mike on 3/18/26.
//

#include "hip_codegen_llvm_impl.h"
#include "../hip_bindless_array.h"
#include "../hip_buffer.h"
#include "../hip_motion_instance.h"
#include "../hip_texture.h"

namespace luisa::compute::hip {

llvm::Value *HIPCodegenLLVMImpl::_get_direct_texture_descriptor_pointer(
    IB &b, llvm::Value *texture) noexcept {
    auto encoded = b.CreateExtractValue(
        texture, llvm_texture_type_descriptor_index,
        "texture.descriptor.tagged");
    auto address = b.CreateAnd(
        encoded,
        b.getInt64(~HIPTexture::direct_descriptor_mip_tag_mask),
        "texture.descriptor.address");
    return b.CreateIntToPtr(
        address,
        llvm::PointerType::get(
            _llvm_context, amdgpu_address_space_constant),
        "texture.descriptor");
}

llvm::Value *HIPCodegenLLVMImpl::_get_direct_texture_base_level(
    IB &b, llvm::Value *texture) noexcept {
    auto encoded = b.CreateExtractValue(
        texture, llvm_texture_type_descriptor_index,
        "texture.descriptor.tagged");
    return b.CreateAnd(
        encoded,
        b.getInt64(HIPTexture::direct_descriptor_mip_tag_mask),
        "texture.base.level");
}

llvm::Value *HIPCodegenLLVMImpl::_get_direct_texture_storage(
    IB &b, llvm::Value *texture) noexcept {
    auto descriptor = _get_direct_texture_descriptor_pointer(b, texture);
    auto storage_ptr = b.CreateInBoundsGEP(
        b.getInt8Ty(), descriptor,
        b.getInt64(offsetof(HIPDirectTextureDescriptor, storage)),
        "texture.storage.ptr");
    return b.CreateLoad(b.getInt64Ty(), storage_ptr, "texture.storage");
}

llvm::Value *HIPCodegenLLVMImpl::_sample_packed_r10g10b10a2(
    IB &b, llvm::Value *resource, llvm::Value *coord,
    llvm::ArrayRef<llvm::Value *> sizes, llvm::Value *filter,
    llvm::Value *address) noexcept {
    LUISA_DEBUG_ASSERT(sizes.size() == 2u || sizes.size() == 3u);
    auto is_2d = sizes.size() == 2u;
    auto llvm_i32_type = b.getInt32Ty();
    auto llvm_f32_type = b.getFloatTy();
    auto llvm_v4i32_type = llvm::FixedVectorType::get(llvm_i32_type, 4u);
    auto llvm_v4f32_type = llvm::FixedVectorType::get(llvm_f32_type, 4u);
    auto llvm_v8i32_type = llvm::FixedVectorType::get(llvm_i32_type, 8u);

    auto address_coord = [&](llvm::Value *integer_coord,
                             llvm::Value *size) noexcept {
        auto zero = b.getInt32(0);
        auto one = b.getInt32(1);
        auto size_minus_one = b.CreateSub(size, one);
        auto in_bounds = b.CreateAnd(
            b.CreateICmpSGE(integer_coord, zero),
            b.CreateICmpSLT(integer_coord, size));
        auto clamped = b.CreateSelect(
            b.CreateICmpSLT(integer_coord, zero), zero,
            b.CreateSelect(b.CreateICmpSGT(integer_coord, size_minus_one),
                           size_minus_one, integer_coord));

        auto repeated = b.CreateSRem(integer_coord, size);
        repeated = b.CreateSelect(
            b.CreateICmpSLT(repeated, zero),
            b.CreateAdd(repeated, size), repeated);

        auto period = b.CreateShl(size, one);
        auto mirrored = b.CreateSRem(integer_coord, period);
        mirrored = b.CreateSelect(
            b.CreateICmpSLT(mirrored, zero),
            b.CreateAdd(mirrored, period), mirrored);
        mirrored = b.CreateSelect(
            b.CreateICmpSGE(mirrored, size),
            b.CreateSub(b.CreateSub(period, one), mirrored), mirrored);

        auto is_repeat = b.CreateICmpEQ(
            address, b.getInt32(to_underlying(Sampler::Address::REPEAT)));
        auto is_mirror = b.CreateICmpEQ(
            address, b.getInt32(to_underlying(Sampler::Address::MIRROR)));
        auto is_zero = b.CreateICmpEQ(
            address, b.getInt32(to_underlying(Sampler::Address::ZERO)));
        auto adjusted = b.CreateSelect(
            is_repeat, repeated,
            b.CreateSelect(is_mirror, mirrored, clamped));
        auto valid = b.CreateSelect(is_zero, in_bounds, b.getTrue());
        return std::pair{adjusted, valid};
    };

    auto fetch = [&](llvm::ArrayRef<llvm::Value *> integer_coords) noexcept {
        LUISA_DEBUG_ASSERT(integer_coords.size() == sizes.size());
        llvm::SmallVector<llvm::Value *, 3u> adjusted_coords;
        auto valid = static_cast<llvm::Value *>(b.getTrue());
        for (auto i = 0u; i < sizes.size(); i++) {
            auto [adjusted, coord_valid] = address_coord(
                integer_coords[i], sizes[i]);
            adjusted_coords.emplace_back(adjusted);
            valid = b.CreateAnd(valid, coord_valid);
        }
        llvm::SmallVector<llvm::Value *, 8u> args;
        args.emplace_back(b.getInt32(15));
        for (auto adjusted : adjusted_coords) {
            args.emplace_back(adjusted);
        }
        args.emplace_back(resource);
        args.emplace_back(b.getInt32(0));
        args.emplace_back(b.getInt32(0));
        auto intrinsic = is_2d ?
                             llvm::Intrinsic::amdgcn_image_load_2d :
                             llvm::Intrinsic::amdgcn_image_load_3d;
        auto raw = b.CreateIntrinsic(
            intrinsic, {llvm_v4f32_type, llvm_i32_type, llvm_v8i32_type}, args);
        auto raw_i32 = b.CreateBitCast(raw, llvm_v4i32_type);
        auto packed = b.CreateExtractElement(raw_i32, b.getInt64(0u));
        auto value = _unpack_r10g10b10a2(
            b, packed, llvm_v4f32_type);
        return b.CreateSelect(
            valid, value,
            llvm::Constant::getNullValue(llvm_v4f32_type));
    };

    llvm::SmallVector<llvm::Value *, 3u> point_coords;
    llvm::SmallVector<llvm::Value *, 3u> linear_coords;
    llvm::SmallVector<llvm::Value *, 3u> linear_weights;
    for (auto i = 0u; i < sizes.size(); i++) {
        auto size_f = b.CreateUIToFP(sizes[i], llvm_f32_type);
        auto scaled = b.CreateFMul(
            b.CreateExtractElement(coord, b.getInt64(i)), size_f);
        auto point_floor = b.CreateUnaryIntrinsic(
            llvm::Intrinsic::floor, scaled);
        point_coords.emplace_back(b.CreateFPToSI(point_floor, llvm_i32_type));
        auto linear_position = b.CreateFSub(
            scaled, llvm::ConstantFP::get(llvm_f32_type, 0.5));
        auto linear_floor = b.CreateUnaryIntrinsic(
            llvm::Intrinsic::floor, linear_position);
        linear_coords.emplace_back(b.CreateFPToSI(linear_floor, llvm_i32_type));
        linear_weights.emplace_back(b.CreateFSub(linear_position, linear_floor));
    }

    auto llvm_func = b.GetInsertBlock()->getParent();
    auto llvm_point_block = llvm::BasicBlock::Create(
        _llvm_context, "packed.sample.point", llvm_func);
    auto llvm_linear_block = llvm::BasicBlock::Create(
        _llvm_context, "packed.sample.linear", llvm_func);
    auto llvm_merge_block = llvm::BasicBlock::Create(
        _llvm_context, "packed.sample.merge", llvm_func);
    b.CreateCondBr(
        b.CreateICmpEQ(
            filter, b.getInt32(to_underlying(Sampler::Filter::POINT))),
        llvm_point_block, llvm_linear_block);

    b.SetInsertPoint(llvm_point_block);
    auto point_value = fetch(point_coords);
    auto point_result_block = b.GetInsertBlock();
    b.CreateBr(llvm_merge_block);

    auto lerp = [&](llvm::Value *x, llvm::Value *y,
                    llvm::Value *weight) noexcept {
        auto weights = _create_llvm_vector(
            b, {weight, weight, weight, weight});
        return b.CreateFAdd(
            x, b.CreateFMul(b.CreateFSub(y, x), weights));
    };
    auto offset_coord = [&](uint32_t mask) noexcept {
        llvm::SmallVector<llvm::Value *, 3u> result;
        for (auto i = 0u; i < linear_coords.size(); i++) {
            result.emplace_back((mask & (1u << i)) == 0u ?
                                    linear_coords[i] :
                                    b.CreateAdd(linear_coords[i], b.getInt32(1u)));
        }
        return result;
    };

    b.SetInsertPoint(llvm_linear_block);
    auto v000 = fetch(offset_coord(0u));
    auto v100 = fetch(offset_coord(1u));
    auto v010 = fetch(offset_coord(2u));
    auto v110 = fetch(offset_coord(3u));
    auto v00 = lerp(v000, v100, linear_weights[0]);
    auto v10 = lerp(v010, v110, linear_weights[0]);
    auto linear_value = lerp(v00, v10, linear_weights[1]);
    if (!is_2d) {
        auto v001 = fetch(offset_coord(4u));
        auto v101 = fetch(offset_coord(5u));
        auto v011 = fetch(offset_coord(6u));
        auto v111 = fetch(offset_coord(7u));
        auto v01 = lerp(v001, v101, linear_weights[0]);
        auto v11 = lerp(v011, v111, linear_weights[0]);
        auto vz1 = lerp(v01, v11, linear_weights[1]);
        linear_value = lerp(linear_value, vz1, linear_weights[2]);
    }
    auto linear_result_block = b.GetInsertBlock();
    b.CreateBr(llvm_merge_block);

    b.SetInsertPoint(llvm_merge_block);
    auto result = b.CreatePHI(llvm_v4f32_type, 2u, "packed.sample");
    result->addIncoming(point_value, point_result_block);
    result->addIncoming(linear_value, linear_result_block);
    return result;
}

llvm::Value *HIPCodegenLLVMImpl::_sample_texture_level(
    IB &b, bool is_2d, llvm::Value *resource, llvm::Value *sampler,
    llvm::Value *coord, llvm::ArrayRef<llvm::Value *> sizes,
    llvm::Value *filter, llvm::Value *address,
    llvm::Value *is_packed_r10g10b10a2) noexcept {
    LUISA_DEBUG_ASSERT(sizes.size() == (is_2d ? 2u : 3u));
    auto llvm_i32_type = b.getInt32Ty();
    auto llvm_f32_type = b.getFloatTy();
    auto llvm_v4i32_type = llvm::FixedVectorType::get(llvm_i32_type, 4u);
    auto llvm_v4f32_type = llvm::FixedVectorType::get(llvm_f32_type, 4u);
    auto llvm_v8i32_type = llvm::FixedVectorType::get(llvm_i32_type, 8u);

    auto llvm_func = b.GetInsertBlock()->getParent();
    auto llvm_native_block = llvm::BasicBlock::Create(
        _llvm_context, "texture.sample.native", llvm_func);
    auto llvm_packed_block = llvm::BasicBlock::Create(
        _llvm_context, "texture.sample.packed", llvm_func);
    auto llvm_merge_block = llvm::BasicBlock::Create(
        _llvm_context, "texture.sample.merge", llvm_func);
    b.CreateCondBr(is_packed_r10g10b10a2,
                   llvm_packed_block, llvm_native_block);

    b.SetInsertPoint(llvm_native_block);
    llvm::SmallVector<llvm::Value *, 12u> args;
    args.emplace_back(b.getInt32(15));
    for (auto i = 0u; i < sizes.size(); i++) {
        args.emplace_back(b.CreateExtractElement(coord, b.getInt64(i)));
    }
    args.emplace_back(llvm::ConstantFP::get(llvm_f32_type, 0.0));
    args.emplace_back(resource);
    args.emplace_back(sampler);
    args.emplace_back(b.getInt1(false));
    args.emplace_back(b.getInt32(0));
    args.emplace_back(b.getInt32(0));
    auto intrinsic = is_2d ?
                         llvm::Intrinsic::amdgcn_image_sample_l_2d :
                         llvm::Intrinsic::amdgcn_image_sample_l_3d;
    auto native_value = b.CreateIntrinsic(
        intrinsic,
        {llvm_v4f32_type, llvm_f32_type,
         llvm_v8i32_type, llvm_v4i32_type},
        args);
    auto native_result_block = b.GetInsertBlock();
    b.CreateBr(llvm_merge_block);

    b.SetInsertPoint(llvm_packed_block);
    auto packed_value = _sample_packed_r10g10b10a2(
        b, resource, coord, sizes, filter, address);
    auto packed_result_block = b.GetInsertBlock();
    b.CreateBr(llvm_merge_block);

    b.SetInsertPoint(llvm_merge_block);
    auto result = b.CreatePHI(llvm_v4f32_type, 2u, "texture.sample");
    result->addIncoming(native_value, native_result_block);
    result->addIncoming(packed_value, packed_result_block);
    return result;
}

llvm::Value *HIPCodegenLLVMImpl::_translate_resource_query_inst(IB &b, FunctionContext &func_ctx, const xir::ResourceQueryInst *inst) noexcept {
    auto sample_mip_levels = [&](auto &&sample_level,
                                 llvm::Value *level0,
                                 llvm::Value *level1,
                                 llvm::Value *mip_linear,
                                 llvm::Value *mip_weight) noexcept {
        auto value0 = sample_level(level0);
        auto value0_block = b.GetInsertBlock();
        auto needs_level1 = b.CreateAnd(
            mip_linear,
            b.CreateAnd(
                b.CreateICmpNE(level0, level1),
                b.CreateFCmpONE(
                    mip_weight,
                    llvm::ConstantFP::get(b.getFloatTy(), 0.0))));

        auto llvm_func = b.GetInsertBlock()->getParent();
        auto llvm_level1_block = llvm::BasicBlock::Create(
            _llvm_context, "texture.sample.mip.level1", llvm_func);
        auto llvm_merge_block = llvm::BasicBlock::Create(
            _llvm_context, "texture.sample.mip.merge", llvm_func);
        b.CreateCondBr(needs_level1,
                       llvm_level1_block, llvm_merge_block);

        b.SetInsertPoint(llvm_level1_block);
        auto value1 = sample_level(level1);
        auto weight = _create_llvm_vector(
            b, {mip_weight, mip_weight, mip_weight, mip_weight});
        auto blended = b.CreateFAdd(
            value0,
            b.CreateFMul(b.CreateFSub(value1, value0), weight));
        auto level1_result_block = b.GetInsertBlock();
        b.CreateBr(llvm_merge_block);

        b.SetInsertPoint(llvm_merge_block);
        auto result = b.CreatePHI(
            value0->getType(), 2u, "texture.sample.mip");
        result->addIncoming(value0, value0_block);
        result->addIncoming(blended, level1_result_block);
        return result;
    };
    switch (auto op = inst->op()) {
        case xir::ResourceQueryOp::BUFFER_SIZE: {
            auto llvm_buffer = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_size_bytes = b.CreateExtractValue(llvm_buffer, llvm_buffer_type_size_index);
            auto elem_type = inst->operand(0)->type()->element();
            auto llvm_size_elements = b.CreateUDiv(llvm_size_bytes, b.getInt64(elem_type->size()));
            auto llvm_result_type = _get_llvm_type(inst->type())->reg_type;
            return b.CreateZExtOrTrunc(llvm_size_elements, llvm_result_type);
        }
        case xir::ResourceQueryOp::BYTE_BUFFER_SIZE: {
            auto llvm_buffer = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_size = b.CreateExtractValue(llvm_buffer, llvm_buffer_type_size_index);
            auto llvm_result_type = _get_llvm_type(inst->type())->reg_type;
            return b.CreateZExtOrTrunc(llvm_size, llvm_result_type);
        }
        case xir::ResourceQueryOp::TEXTURE2D_SIZE: {
            LUISA_DEBUG_ASSERT(inst->type() == Type::of<luisa::int2>() || inst->type() == Type::of<luisa::uint2>());
            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_descriptor = _get_direct_texture_descriptor_pointer(b, llvm_texture);
            auto llvm_base_level = _get_direct_texture_base_level(b, llvm_texture);
            auto llvm_size_ptr = b.CreateInBoundsGEP(
                b.getInt8Ty(), llvm_descriptor,
                b.getInt64(offsetof(HIPDirectTextureDescriptor, size_xy)));
            auto llvm_size_xy = b.CreateLoad(b.getInt64Ty(), llvm_size_ptr);
            auto llvm_width = b.CreateTrunc(llvm_size_xy, b.getInt32Ty());
            auto llvm_height = b.CreateTrunc(
                b.CreateLShr(llvm_size_xy, b.getInt64(32u)), b.getInt32Ty());
            auto llvm_shift = b.CreateTrunc(llvm_base_level, b.getInt32Ty());
            llvm_width = b.CreateLShr(llvm_width, llvm_shift);
            llvm_height = b.CreateLShr(llvm_height, llvm_shift);
            llvm_width = b.CreateSelect(
                b.CreateICmpUGT(llvm_width, b.getInt32(1u)),
                llvm_width, b.getInt32(1u));
            llvm_height = b.CreateSelect(
                b.CreateICmpUGT(llvm_height, b.getInt32(1u)),
                llvm_height, b.getInt32(1u));
            return _create_llvm_vector(b, {llvm_width, llvm_height});
        }
        case xir::ResourceQueryOp::TEXTURE3D_SIZE: {
            LUISA_DEBUG_ASSERT(inst->type() == Type::of<luisa::int3>() || inst->type() == Type::of<luisa::uint3>());
            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_descriptor = _get_direct_texture_descriptor_pointer(b, llvm_texture);
            auto llvm_base_level = _get_direct_texture_base_level(b, llvm_texture);
            auto llvm_size_xy_ptr = b.CreateInBoundsGEP(
                b.getInt8Ty(), llvm_descriptor,
                b.getInt64(offsetof(HIPDirectTextureDescriptor, size_xy)));
            auto llvm_size_z_ptr = b.CreateInBoundsGEP(
                b.getInt8Ty(), llvm_descriptor,
                b.getInt64(offsetof(HIPDirectTextureDescriptor, size_z)));
            auto llvm_size_xy = b.CreateLoad(b.getInt64Ty(), llvm_size_xy_ptr);
            auto llvm_size_z = b.CreateLoad(b.getInt64Ty(), llvm_size_z_ptr);
            auto llvm_width = b.CreateTrunc(llvm_size_xy, b.getInt32Ty());
            auto llvm_height = b.CreateTrunc(
                b.CreateLShr(llvm_size_xy, b.getInt64(32u)), b.getInt32Ty());
            auto llvm_depth = b.CreateTrunc(llvm_size_z, b.getInt32Ty());
            auto llvm_shift = b.CreateTrunc(llvm_base_level, b.getInt32Ty());
            llvm_width = b.CreateLShr(llvm_width, llvm_shift);
            llvm_height = b.CreateLShr(llvm_height, llvm_shift);
            llvm_depth = b.CreateLShr(llvm_depth, llvm_shift);
            llvm_width = b.CreateSelect(
                b.CreateICmpUGT(llvm_width, b.getInt32(1u)),
                llvm_width, b.getInt32(1u));
            llvm_height = b.CreateSelect(
                b.CreateICmpUGT(llvm_height, b.getInt32(1u)),
                llvm_height, b.getInt32(1u));
            llvm_depth = b.CreateSelect(
                b.CreateICmpUGT(llvm_depth, b.getInt32(1u)),
                llvm_depth, b.getInt32(1u));
            return _create_llvm_vector(b, {llvm_width, llvm_height, llvm_depth});
        }
        case xir::ResourceQueryOp::BINDLESS_BUFFER_SIZE: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE: {
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_slot_ptr = _get_bindless_array_slot_pointer(b, llvm_bindless_array, llvm_index);
            // Use byte offset to access buffer_size field
            auto llvm_slot_type = _get_llvm_bindless_array_slot_type();
            auto &dl = _llvm_module->getDataLayout();
            auto slot_struct_type = llvm::cast<llvm::StructType>(llvm_slot_type);
            auto buffer_size_offset = dl.getStructLayout(slot_struct_type)->getElementOffset(llvm_bindless_array_slot_type_buffer_size_index);
            auto llvm_buffer_size_ptr = b.CreateInBoundsGEP(b.getInt8Ty(), llvm_slot_ptr, b.getInt64(buffer_size_offset));
            auto llvm_buffer_size = static_cast<llvm::Value *>(b.CreateLoad(
                llvm::Type::getInt64Ty(_llvm_context), llvm_buffer_size_ptr));
            if (op == xir::ResourceQueryOp::BINDLESS_BUFFER_SIZE) {
                auto elem_stride = b.CreateZExt(_get_llvm_value(b, func_ctx, inst->operand(2)), llvm_buffer_size->getType());
                llvm_buffer_size = b.CreateUDiv(llvm_buffer_size, elem_stride);
            }
            auto llvm_result_type = _get_llvm_type(inst->type())->reg_type;
            return b.CreateZExtOrTrunc(llvm_buffer_size, llvm_result_type);
        }
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL: {
            LUISA_DEBUG_ASSERT(inst->type() == Type::of<luisa::int2>() || inst->type() == Type::of<luisa::uint2>());
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_slot_ptr = _get_bindless_array_slot_pointer(b, llvm_bindless_array, llvm_index);
            auto llvm_slot_type = _get_llvm_bindless_array_slot_type();
            auto &dl = _llvm_module->getDataLayout();
            auto slot_struct_type = llvm::cast<llvm::StructType>(llvm_slot_type);
            auto levels_offset = dl.getStructLayout(slot_struct_type)->getElementOffset(llvm_bindless_array_slot_type_texture2d_levels_index);
            auto size_offset = dl.getStructLayout(slot_struct_type)->getElementOffset(llvm_bindless_array_slot_type_texture2d_size_index);
            auto llvm_level_count_ptr = b.CreateInBoundsGEP(b.getInt8Ty(), llvm_slot_ptr, b.getInt64(levels_offset));
            auto llvm_size_ptr = b.CreateInBoundsGEP(b.getInt8Ty(), llvm_slot_ptr, b.getInt64(size_offset));
            auto llvm_level_count = b.CreateAnd(
                b.CreateLoad(b.getInt64Ty(), llvm_level_count_ptr),
                b.getInt64(0xffu));
            auto llvm_size_xy = b.CreateLoad(b.getInt64Ty(), llvm_size_ptr);
            _create_assertion_with_message(b, b.CreateICmpUGT(llvm_level_count, b.getInt64(0)), "Bindless texture slot has no mip levels.");
            auto llvm_level = static_cast<llvm::Value *>(b.getInt64(0));
            if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL) {
                auto level = _get_llvm_value(b, func_ctx, inst->operand(2));
                llvm_level = level->getType()->isFloatingPointTy() ? b.CreateFPToUI(level, b.getInt64Ty()) : b.CreateZExtOrTrunc(level, b.getInt64Ty());
            }
            auto llvm_max_level = b.CreateSub(llvm_level_count, b.getInt64(1));
            llvm_level = b.CreateSelect(b.CreateICmpUGT(llvm_level, llvm_max_level), llvm_max_level, llvm_level);
            auto llvm_width = b.CreateTrunc(llvm_size_xy, b.getInt32Ty());
            auto llvm_height = b.CreateTrunc(b.CreateLShr(llvm_size_xy, b.getInt64(32u)), b.getInt32Ty());
            auto llvm_shift = b.CreateTrunc(llvm_level, b.getInt32Ty());
            llvm_width = b.CreateLShr(llvm_width, llvm_shift);
            llvm_height = b.CreateLShr(llvm_height, llvm_shift);
            llvm_width = b.CreateSelect(b.CreateICmpUGT(llvm_width, b.getInt32(1u)), llvm_width, b.getInt32(1u));
            llvm_height = b.CreateSelect(b.CreateICmpUGT(llvm_height, b.getInt32(1u)), llvm_height, b.getInt32(1u));
            return _create_llvm_vector(b, {llvm_width, llvm_height});
        }
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL: {
            LUISA_DEBUG_ASSERT(inst->type() == Type::of<luisa::int3>() || inst->type() == Type::of<luisa::uint3>());
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_slot_ptr = _get_bindless_array_slot_pointer(b, llvm_bindless_array, llvm_index);
            auto llvm_slot_type = _get_llvm_bindless_array_slot_type();
            auto &dl = _llvm_module->getDataLayout();
            auto slot_struct_type = llvm::cast<llvm::StructType>(llvm_slot_type);
            auto levels_offset = dl.getStructLayout(slot_struct_type)->getElementOffset(llvm_bindless_array_slot_type_texture3d_levels_index);
            auto size_xy_offset = dl.getStructLayout(slot_struct_type)->getElementOffset(llvm_bindless_array_slot_type_texture3d_size_xy_index);
            auto size_z_offset = dl.getStructLayout(slot_struct_type)->getElementOffset(llvm_bindless_array_slot_type_texture3d_size_z_index);
            auto llvm_level_count_ptr = b.CreateInBoundsGEP(b.getInt8Ty(), llvm_slot_ptr, b.getInt64(levels_offset));
            auto llvm_size_xy_ptr = b.CreateInBoundsGEP(b.getInt8Ty(), llvm_slot_ptr, b.getInt64(size_xy_offset));
            auto llvm_size_z_ptr = b.CreateInBoundsGEP(b.getInt8Ty(), llvm_slot_ptr, b.getInt64(size_z_offset));
            auto llvm_level_count = b.CreateAnd(
                b.CreateLoad(b.getInt64Ty(), llvm_level_count_ptr),
                b.getInt64(0xffu));
            auto llvm_size_xy = b.CreateLoad(b.getInt64Ty(), llvm_size_xy_ptr);
            auto llvm_size_z = b.CreateLoad(b.getInt64Ty(), llvm_size_z_ptr);
            _create_assertion_with_message(b, b.CreateICmpUGT(llvm_level_count, b.getInt64(0)), "Bindless texture slot has no mip levels.");
            auto llvm_level = static_cast<llvm::Value *>(b.getInt64(0));
            if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL) {
                auto level = _get_llvm_value(b, func_ctx, inst->operand(2));
                llvm_level = level->getType()->isFloatingPointTy() ? b.CreateFPToUI(level, b.getInt64Ty()) : b.CreateZExtOrTrunc(level, b.getInt64Ty());
            }
            auto llvm_max_level = b.CreateSub(llvm_level_count, b.getInt64(1));
            llvm_level = b.CreateSelect(b.CreateICmpUGT(llvm_level, llvm_max_level), llvm_max_level, llvm_level);
            auto llvm_width = b.CreateTrunc(llvm_size_xy, b.getInt32Ty());
            auto llvm_height = b.CreateTrunc(b.CreateLShr(llvm_size_xy, b.getInt64(32u)), b.getInt32Ty());
            auto llvm_depth = b.CreateTrunc(llvm_size_z, b.getInt32Ty());
            auto llvm_shift = b.CreateTrunc(llvm_level, b.getInt32Ty());
            llvm_width = b.CreateLShr(llvm_width, llvm_shift);
            llvm_height = b.CreateLShr(llvm_height, llvm_shift);
            llvm_depth = b.CreateLShr(llvm_depth, llvm_shift);
            llvm_width = b.CreateSelect(b.CreateICmpUGT(llvm_width, b.getInt32(1u)), llvm_width, b.getInt32(1u));
            llvm_height = b.CreateSelect(b.CreateICmpUGT(llvm_height, b.getInt32(1u)), llvm_height, b.getInt32(1u));
            llvm_depth = b.CreateSelect(b.CreateICmpUGT(llvm_depth, b.getInt32(1u)), llvm_depth, b.getInt32(1u));
            return _create_llvm_vector(b, {llvm_width, llvm_height, llvm_depth});
        }
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE: [[fallthrough]];
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD: [[fallthrough]];
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE: [[fallthrough]];
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD: [[fallthrough]];
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL: {
            auto is_2d = op == xir::ResourceQueryOp::TEXTURE2D_SAMPLE ||
                         op == xir::ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL ||
                         op == xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD ||
                         op == xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL;
            auto has_explicit_level =
                op == xir::ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL ||
                op == xir::ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL;
            auto has_grad =
                op == xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD ||
                op == xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD ||
                op == xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL ||
                op == xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL;
            auto has_min_mip =
                op == xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL ||
                op == xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL;

            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_descriptor =
                _get_direct_texture_descriptor_pointer(b, llvm_texture);
            auto llvm_base_level =
                _get_direct_texture_base_level(b, llvm_texture);
            auto llvm_storage = _get_direct_texture_storage(b, llvm_texture);
            auto llvm_is_packed_r10g10b10a2 = b.CreateICmpEQ(
                llvm_storage,
                b.getInt64(to_underlying(PixelStorage::R10G10B10A2)));
            auto llvm_level_count_ptr = b.CreateInBoundsGEP(
                b.getInt8Ty(), llvm_descriptor,
                b.getInt64(offsetof(
                    HIPDirectTextureDescriptor, level_count)),
                "texture.level.count.ptr");
            auto llvm_total_level_count = b.CreateLoad(
                b.getInt64Ty(), llvm_level_count_ptr,
                "texture.level.count");
            _create_assertion_with_message(
                b,
                b.CreateICmpULT(
                    llvm_base_level, llvm_total_level_count),
                "Direct texture binding has an invalid base mip level.");
            // A bound ImageView exposes its selected level as LOD zero and
            // all remaining levels after it, matching the other backends.
            auto llvm_view_level_count = b.CreateSub(
                llvm_total_level_count, llvm_base_level,
                "texture.view.level.count");

            auto llvm_filter = b.CreateZExtOrTrunc(
                _get_llvm_value(
                    b, func_ctx,
                    inst->operand(inst->operand_count() - 2u)),
                b.getInt32Ty());
            auto llvm_address = b.CreateZExtOrTrunc(
                _get_llvm_value(
                    b, func_ctx,
                    inst->operand(inst->operand_count() - 1u)),
                b.getInt32Ty());
            _create_assertion_with_message(
                b, b.CreateICmpULT(llvm_filter, b.getInt32(4u)),
                "Invalid direct texture sampler filter.");
            _create_assertion_with_message(
                b, b.CreateICmpULT(llvm_address, b.getInt32(4u)),
                "Invalid direct texture sampler address mode.");
            auto llvm_sampler_code = b.CreateOr(
                b.CreateShl(llvm_filter, b.getInt32(2u)),
                llvm_address);
            auto llvm_mip_linear = b.CreateICmpUGE(
                llvm_filter, b.getInt32(2u));

            auto llvm_sampler_offset = b.CreateAdd(
                b.getInt64(offsetof(
                    HIPDirectTextureDescriptor, samplers)),
                b.CreateMul(
                    b.CreateZExt(
                        llvm_sampler_code, b.getInt64Ty()),
                    b.getInt64(sizeof(HIPSamplerDescriptor))));
            auto llvm_sampler_ptr = b.CreateInBoundsGEP(
                b.getInt8Ty(), llvm_descriptor,
                llvm_sampler_offset, "texture.sampler.ptr");
            auto llvm_v4i32_type =
                llvm::FixedVectorType::get(b.getInt32Ty(), 4);
            auto llvm_sampler = b.CreateLoad(
                llvm_v4i32_type, llvm_sampler_ptr,
                "texture.sampler");

            auto llvm_lod = static_cast<llvm::Value *>(
                llvm::ConstantFP::get(b.getFloatTy(), 0.0));
            if (has_explicit_level) {
                llvm_lod = _safe_fp_cast(
                    b,
                    _get_llvm_value(b, func_ctx, inst->operand(2)),
                    b.getFloatTy());
            } else if (has_grad) {
                auto llvm_ddx = _get_llvm_value(
                    b, func_ctx, inst->operand(2));
                auto llvm_ddy = _get_llvm_value(
                    b, func_ctx, inst->operand(3));
                auto llvm_size_xy_ptr = b.CreateInBoundsGEP(
                    b.getInt8Ty(), llvm_descriptor,
                    b.getInt64(offsetof(
                        HIPDirectTextureDescriptor, size_xy)));
                auto llvm_size_xy = b.CreateLoad(
                    b.getInt64Ty(), llvm_size_xy_ptr);
                auto llvm_shift = b.CreateTrunc(
                    llvm_base_level, b.getInt32Ty());
                auto llvm_width = b.CreateLShr(
                    b.CreateTrunc(
                        llvm_size_xy, b.getInt32Ty()),
                    llvm_shift);
                auto llvm_height = b.CreateLShr(
                    b.CreateTrunc(
                        b.CreateLShr(
                            llvm_size_xy, b.getInt64(32u)),
                        b.getInt32Ty()),
                    llvm_shift);
                llvm::SmallVector<llvm::Value *, 3u> llvm_sizes;
                llvm_sizes.emplace_back(b.CreateUIToFP(
                    llvm_width, b.getFloatTy()));
                llvm_sizes.emplace_back(b.CreateUIToFP(
                    llvm_height, b.getFloatTy()));
                if (!is_2d) {
                    auto llvm_size_z_ptr = b.CreateInBoundsGEP(
                        b.getInt8Ty(), llvm_descriptor,
                        b.getInt64(offsetof(
                            HIPDirectTextureDescriptor, size_z)));
                    auto llvm_size_z = b.CreateLoad(
                        b.getInt64Ty(), llvm_size_z_ptr);
                    auto llvm_depth = b.CreateLShr(
                        b.CreateTrunc(
                            llvm_size_z, b.getInt32Ty()),
                        llvm_shift);
                    llvm_sizes.emplace_back(b.CreateUIToFP(
                        llvm_depth, b.getFloatTy()));
                }
                auto llvm_rho_x2 = static_cast<llvm::Value *>(
                    llvm::ConstantFP::get(b.getFloatTy(), 0.0));
                auto llvm_rho_y2 = static_cast<llvm::Value *>(
                    llvm::ConstantFP::get(b.getFloatTy(), 0.0));
                for (auto i = 0u; i < llvm_sizes.size(); i++) {
                    auto llvm_dx = b.CreateFMul(
                        b.CreateExtractElement(
                            llvm_ddx, b.getInt64(i)),
                        llvm_sizes[i]);
                    auto llvm_dy = b.CreateFMul(
                        b.CreateExtractElement(
                            llvm_ddy, b.getInt64(i)),
                        llvm_sizes[i]);
                    llvm_rho_x2 = b.CreateFAdd(
                        llvm_rho_x2,
                        b.CreateFMul(llvm_dx, llvm_dx));
                    llvm_rho_y2 = b.CreateFAdd(
                        llvm_rho_y2,
                        b.CreateFMul(llvm_dy, llvm_dy));
                }
                auto llvm_rho2 = b.CreateMaxNum(
                    llvm_rho_x2, llvm_rho_y2);
                llvm_rho2 = b.CreateMaxNum(
                    llvm_rho2,
                    llvm::ConstantFP::get(
                        b.getFloatTy(), 1.0));
                llvm_lod = b.CreateFMul(
                    b.CreateUnaryIntrinsic(
                        llvm::Intrinsic::log2, llvm_rho2),
                    llvm::ConstantFP::get(
                        b.getFloatTy(), 0.5));
                if (has_min_mip) {
                    auto llvm_min_mip = _safe_fp_cast(
                        b,
                        _get_llvm_value(
                            b, func_ctx, inst->operand(4)),
                        b.getFloatTy());
                    llvm_lod = b.CreateMaxNum(
                        llvm_lod, llvm_min_mip);
                }
            }

            auto llvm_level0 = static_cast<llvm::Value *>(b.getInt64(0u));
            auto llvm_level1 = static_cast<llvm::Value *>(b.getInt64(0u));
            auto llvm_mip_weight = static_cast<llvm::Value *>(
                llvm::ConstantFP::get(b.getFloatTy(), 0.0));
            if (has_explicit_level || has_grad) {
                auto llvm_max_level = b.CreateSub(
                    llvm_view_level_count, b.getInt64(1u));
                auto llvm_max_level_f = b.CreateUIToFP(
                    llvm_max_level, b.getFloatTy());
                llvm_lod = b.CreateMinNum(
                    b.CreateMaxNum(
                        llvm_lod,
                        llvm::ConstantFP::get(
                            b.getFloatTy(), 0.0)),
                    llvm_max_level_f);
                auto llvm_lod_floor = b.CreateUnaryIntrinsic(
                    llvm::Intrinsic::floor, llvm_lod);
                auto llvm_linear_level = b.CreateFPToUI(
                    llvm_lod_floor, b.getInt64Ty());
                auto llvm_nearest_level = b.CreateFPToUI(
                    b.CreateUnaryIntrinsic(
                        llvm::Intrinsic::floor,
                        b.CreateFAdd(
                            llvm_lod,
                            llvm::ConstantFP::get(
                                b.getFloatTy(), 0.5))),
                    b.getInt64Ty());
                llvm_level0 = b.CreateSelect(
                    llvm_mip_linear,
                    llvm_linear_level, llvm_nearest_level);
                llvm_level1 = b.CreateAdd(
                    llvm_level0, b.getInt64(1u));
                llvm_level1 = b.CreateSelect(
                    b.CreateICmpUGT(
                        llvm_level1, llvm_max_level),
                    llvm_max_level, llvm_level1);
                llvm_mip_weight = b.CreateSelect(
                    llvm_mip_linear,
                    b.CreateFSub(llvm_lod, llvm_lod_floor),
                    llvm::ConstantFP::get(
                        b.getFloatTy(), 0.0));
            }

            auto llvm_v8i32_type =
                llvm::FixedVectorType::get(b.getInt32Ty(), 8);
            auto load_resource = [&](llvm::Value *relative_level) noexcept {
                auto llvm_absolute_level = b.CreateAdd(
                    llvm_base_level, relative_level);
                auto llvm_image_offset = b.CreateAdd(
                    b.getInt64(offsetof(
                        HIPDirectTextureDescriptor, images)),
                    b.CreateMul(
                        llvm_absolute_level,
                        b.getInt64(sizeof(HIPImageDescriptor))));
                auto llvm_image_ptr = b.CreateInBoundsGEP(
                    b.getInt8Ty(), llvm_descriptor,
                    llvm_image_offset, "texture.image.ptr");
                return b.CreateLoad(
                    llvm_v8i32_type, llvm_image_ptr,
                    "texture.image");
            };
            auto sample_level = [&](llvm::Value *level) noexcept {
                auto llvm_resource = load_resource(level);
                auto llvm_absolute_level = b.CreateAdd(
                    llvm_base_level, level);
                auto llvm_shift = b.CreateTrunc(
                    llvm_absolute_level, b.getInt32Ty());
                auto llvm_size_xy_ptr = b.CreateInBoundsGEP(
                    b.getInt8Ty(), llvm_descriptor,
                    b.getInt64(offsetof(
                        HIPDirectTextureDescriptor, size_xy)));
                auto llvm_size_xy = b.CreateLoad(
                    b.getInt64Ty(), llvm_size_xy_ptr);
                auto llvm_width = b.CreateLShr(
                    b.CreateTrunc(llvm_size_xy, b.getInt32Ty()),
                    llvm_shift);
                auto llvm_height = b.CreateLShr(
                    b.CreateTrunc(
                        b.CreateLShr(llvm_size_xy, b.getInt64(32u)),
                        b.getInt32Ty()),
                    llvm_shift);
                llvm_width = b.CreateSelect(
                    b.CreateICmpUGT(llvm_width, b.getInt32(1u)),
                    llvm_width, b.getInt32(1u));
                llvm_height = b.CreateSelect(
                    b.CreateICmpUGT(llvm_height, b.getInt32(1u)),
                    llvm_height, b.getInt32(1u));
                llvm::SmallVector<llvm::Value *, 3u> llvm_sizes{
                    llvm_width, llvm_height};
                if (!is_2d) {
                    auto llvm_size_z_ptr = b.CreateInBoundsGEP(
                        b.getInt8Ty(), llvm_descriptor,
                        b.getInt64(offsetof(
                            HIPDirectTextureDescriptor, size_z)));
                    auto llvm_depth = b.CreateLShr(
                        b.CreateTrunc(
                            b.CreateLoad(b.getInt64Ty(), llvm_size_z_ptr),
                            b.getInt32Ty()),
                        llvm_shift);
                    llvm_depth = b.CreateSelect(
                        b.CreateICmpUGT(llvm_depth, b.getInt32(1u)),
                        llvm_depth, b.getInt32(1u));
                    llvm_sizes.emplace_back(llvm_depth);
                }
                return _sample_texture_level(
                    b, is_2d, llvm_resource, llvm_sampler, llvm_coord,
                    llvm_sizes, llvm_filter, llvm_address,
                    llvm_is_packed_r10g10b10a2);
            };
            auto llvm_value = has_explicit_level || has_grad ?
                                  sample_mip_levels(
                                      sample_level, llvm_level0, llvm_level1,
                                      llvm_mip_linear, llvm_mip_weight) :
                                  sample_level(llvm_level0);
            return _safe_fp_cast(
                b, llvm_value,
                _get_llvm_type(inst->type())->reg_type);
        }
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER: {
            auto is_2d = op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE ||
                         op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL ||
                         op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD ||
                         op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL ||
                         op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER ||
                         op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER ||
                         op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER ||
                         op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER;
            auto has_explicit_level = op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL ||
                                      op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL ||
                                      op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER ||
                                      op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER;
            auto has_grad = op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD ||
                            op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD ||
                            op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL ||
                            op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL ||
                            op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER ||
                            op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER ||
                            op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER ||
                            op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER;
            auto has_min_mip = op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL ||
                               op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL ||
                               op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER ||
                               op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER;
            auto has_custom_sampler = op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER ||
                                      op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER ||
                                      op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER ||
                                      op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER ||
                                      op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER ||
                                      op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER ||
                                      op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER ||
                                      op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER;

            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_slot_ptr = _get_bindless_array_slot_pointer(b, llvm_bindless_array, llvm_index);
            auto llvm_slot_type = llvm::cast<llvm::StructType>(_get_llvm_bindless_array_slot_type());
            auto &dl = _llvm_module->getDataLayout();
            auto levels_i = is_2d ? llvm_bindless_array_slot_type_texture2d_levels_index :
                                    llvm_bindless_array_slot_type_texture3d_levels_index;
            auto size_xy_i = is_2d ? llvm_bindless_array_slot_type_texture2d_size_index :
                                     llvm_bindless_array_slot_type_texture3d_size_xy_index;
            auto levels_offset = dl.getStructLayout(llvm_slot_type)->getElementOffset(levels_i);
            auto size_xy_offset = dl.getStructLayout(llvm_slot_type)->getElementOffset(size_xy_i);
            auto llvm_levels_sampler = b.CreateLoad(
                b.getInt64Ty(), b.CreateInBoundsGEP(
                                    b.getInt8Ty(), llvm_slot_ptr,
                                    b.getInt64(levels_offset)));
            auto llvm_level_count = b.CreateAnd(
                llvm_levels_sampler,
                b.getInt64(HIPBindlessArray::texture_level_count_mask));
            auto llvm_storage = b.CreateAnd(
                b.CreateLShr(
                    llvm_levels_sampler,
                    b.getInt64(HIPBindlessArray::texture_storage_shift)),
                b.getInt64(HIPBindlessArray::texture_storage_mask));
            auto llvm_is_packed_r10g10b10a2 = b.CreateICmpEQ(
                llvm_storage,
                b.getInt64(to_underlying(PixelStorage::R10G10B10A2)));
            _create_assertion_with_message(
                b, b.CreateICmpUGT(llvm_level_count, b.getInt64(0)),
                "Bindless texture slot has no mip levels.");

            auto llvm_sampler_code = b.CreateTrunc(
                b.CreateAnd(
                    b.CreateLShr(
                        llvm_levels_sampler,
                        b.getInt64(HIPBindlessArray::texture_sampler_shift)),
                    b.getInt64(HIPBindlessArray::texture_sampler_mask)),
                b.getInt32Ty());
            if (has_custom_sampler) {
                auto llvm_filter = b.CreateZExtOrTrunc(
                    _get_llvm_value(b, func_ctx, inst->operand(inst->operand_count() - 2u)),
                    b.getInt32Ty());
                auto llvm_address = b.CreateZExtOrTrunc(
                    _get_llvm_value(b, func_ctx, inst->operand(inst->operand_count() - 1u)),
                    b.getInt32Ty());
                _create_assertion_with_message(
                    b, b.CreateICmpULT(llvm_filter, b.getInt32(4u)),
                    "Invalid bindless texture sampler filter.");
                _create_assertion_with_message(
                    b, b.CreateICmpULT(llvm_address, b.getInt32(4u)),
                    "Invalid bindless texture sampler address mode.");
                llvm_sampler_code = b.CreateOr(
                    b.CreateShl(llvm_filter, b.getInt32(2u)), llvm_address);
            }
            auto llvm_filter = b.CreateLShr(llvm_sampler_code, b.getInt32(2u));
            auto llvm_address = b.CreateAnd(llvm_sampler_code, b.getInt32(0x03u));
            auto llvm_mip_linear = b.CreateICmpUGE(llvm_filter, b.getInt32(2u));

            auto llvm_sampler_table = b.CreateExtractValue(
                llvm_bindless_array, llvm_bindless_array_type_samplers_index);
            _create_assertion_with_message(
                b, b.CreateIsNotNull(llvm_sampler_table),
                "HIP bindless sampler table is not initialized.");
            auto llvm_sampler_offset = b.CreateMul(
                b.CreateZExt(llvm_sampler_code, b.getInt64Ty()),
                b.getInt64(sizeof(HIPSamplerDescriptor)));
            auto llvm_sampler_ptr = b.CreateInBoundsGEP(
                b.getInt8Ty(), llvm_sampler_table, llvm_sampler_offset,
                "tex.sampler.ptr");
            auto llvm_v4i32_type = llvm::FixedVectorType::get(b.getInt32Ty(), 4);
            auto llvm_sampler = b.CreateLoad(
                llvm_v4i32_type, llvm_sampler_ptr, "tex.sampler");

            auto llvm_lod = static_cast<llvm::Value *>(
                llvm::ConstantFP::get(b.getFloatTy(), 0.0));
            if (has_explicit_level) {
                llvm_lod = _safe_fp_cast(
                    b, _get_llvm_value(b, func_ctx, inst->operand(3)),
                    b.getFloatTy());
            } else if (has_grad) {
                auto llvm_ddx = _get_llvm_value(b, func_ctx, inst->operand(3));
                auto llvm_ddy = _get_llvm_value(b, func_ctx, inst->operand(4));
                auto llvm_size_xy = b.CreateLoad(
                    b.getInt64Ty(), b.CreateInBoundsGEP(
                                        b.getInt8Ty(), llvm_slot_ptr,
                                        b.getInt64(size_xy_offset)));
                llvm::SmallVector<llvm::Value *, 3u> llvm_sizes;
                llvm_sizes.emplace_back(b.CreateUIToFP(
                    b.CreateTrunc(llvm_size_xy, b.getInt32Ty()), b.getFloatTy()));
                llvm_sizes.emplace_back(b.CreateUIToFP(
                    b.CreateTrunc(b.CreateLShr(llvm_size_xy, b.getInt64(32u)),
                                  b.getInt32Ty()),
                    b.getFloatTy()));
                if (!is_2d) {
                    auto size_z_offset = dl.getStructLayout(llvm_slot_type)->getElementOffset(llvm_bindless_array_slot_type_texture3d_size_z_index);
                    auto llvm_size_z = b.CreateLoad(
                        b.getInt64Ty(), b.CreateInBoundsGEP(
                                            b.getInt8Ty(), llvm_slot_ptr,
                                            b.getInt64(size_z_offset)));
                    llvm_sizes.emplace_back(b.CreateUIToFP(
                        b.CreateTrunc(llvm_size_z, b.getInt32Ty()),
                        b.getFloatTy()));
                }
                auto llvm_rho_x2 = static_cast<llvm::Value *>(
                    llvm::ConstantFP::get(b.getFloatTy(), 0.0));
                auto llvm_rho_y2 = static_cast<llvm::Value *>(
                    llvm::ConstantFP::get(b.getFloatTy(), 0.0));
                for (auto i = 0u; i < llvm_sizes.size(); i++) {
                    auto llvm_dx = b.CreateFMul(
                        b.CreateExtractElement(llvm_ddx, b.getInt64(i)),
                        llvm_sizes[i]);
                    auto llvm_dy = b.CreateFMul(
                        b.CreateExtractElement(llvm_ddy, b.getInt64(i)),
                        llvm_sizes[i]);
                    llvm_rho_x2 = b.CreateFAdd(
                        llvm_rho_x2, b.CreateFMul(llvm_dx, llvm_dx));
                    llvm_rho_y2 = b.CreateFAdd(
                        llvm_rho_y2, b.CreateFMul(llvm_dy, llvm_dy));
                }
                auto llvm_rho2 = b.CreateMaxNum(llvm_rho_x2, llvm_rho_y2);
                llvm_rho2 = b.CreateMaxNum(
                    llvm_rho2, llvm::ConstantFP::get(b.getFloatTy(), 1.0));
                llvm_lod = b.CreateFMul(
                    b.CreateUnaryIntrinsic(llvm::Intrinsic::log2, llvm_rho2),
                    llvm::ConstantFP::get(b.getFloatTy(), 0.5));
                if (has_min_mip) {
                    auto llvm_min_mip = _safe_fp_cast(
                        b, _get_llvm_value(b, func_ctx, inst->operand(5)),
                        b.getFloatTy());
                    llvm_lod = b.CreateMaxNum(llvm_lod, llvm_min_mip);
                }
            }

            auto llvm_level0 = static_cast<llvm::Value *>(b.getInt64(0u));
            auto llvm_level1 = static_cast<llvm::Value *>(b.getInt64(0u));
            auto llvm_mip_weight = static_cast<llvm::Value *>(
                llvm::ConstantFP::get(b.getFloatTy(), 0.0));
            if (has_explicit_level || has_grad) {
                auto llvm_max_level = b.CreateSub(
                    llvm_level_count, b.getInt64(1u));
                auto llvm_max_level_f = b.CreateUIToFP(
                    llvm_max_level, b.getFloatTy());
                llvm_lod = b.CreateMinNum(
                    b.CreateMaxNum(
                        llvm_lod,
                        llvm::ConstantFP::get(b.getFloatTy(), 0.0)),
                    llvm_max_level_f);
                auto llvm_lod_floor = b.CreateUnaryIntrinsic(
                    llvm::Intrinsic::floor, llvm_lod);
                auto llvm_linear_level = b.CreateFPToUI(
                    llvm_lod_floor, b.getInt64Ty());
                auto llvm_nearest_level = b.CreateFPToUI(
                    b.CreateUnaryIntrinsic(
                        llvm::Intrinsic::floor,
                        b.CreateFAdd(
                            llvm_lod,
                            llvm::ConstantFP::get(
                                b.getFloatTy(), 0.5))),
                    b.getInt64Ty());
                llvm_level0 = b.CreateSelect(
                    llvm_mip_linear,
                    llvm_linear_level, llvm_nearest_level);
                llvm_level1 = b.CreateAdd(
                    llvm_level0, b.getInt64(1u));
                llvm_level1 = b.CreateSelect(
                    b.CreateICmpUGT(
                        llvm_level1, llvm_max_level),
                    llvm_max_level, llvm_level1);
                llvm_mip_weight = b.CreateSelect(
                    llvm_mip_linear,
                    b.CreateFSub(llvm_lod, llvm_lod_floor),
                    llvm::ConstantFP::get(b.getFloatTy(), 0.0));
            }

            auto llvm_v8i32_type = llvm::FixedVectorType::get(b.getInt32Ty(), 8);
            auto load_resource = [&](llvm::Value *level) noexcept {
                auto llvm_descriptor = _get_bindless_array_texture_handle(
                    b, llvm_bindless_array, llvm_index,
                    is_2d ? 2 : 3, level);
                auto llvm_descriptor_ptr = b.CreateIntToPtr(
                    llvm_descriptor,
                    llvm::PointerType::get(
                        _llvm_context, amdgpu_address_space_global),
                    "tex.image.ptr");
                return b.CreateLoad(
                    llvm_v8i32_type, llvm_descriptor_ptr, "tex.image");
            };
            auto sample_level = [&](llvm::Value *level) noexcept {
                auto llvm_resource = load_resource(level);
                auto llvm_shift = b.CreateTrunc(level, b.getInt32Ty());
                auto llvm_size_xy = b.CreateLoad(
                    b.getInt64Ty(), b.CreateInBoundsGEP(
                                        b.getInt8Ty(), llvm_slot_ptr,
                                        b.getInt64(size_xy_offset)));
                auto llvm_width = b.CreateLShr(
                    b.CreateTrunc(llvm_size_xy, b.getInt32Ty()),
                    llvm_shift);
                auto llvm_height = b.CreateLShr(
                    b.CreateTrunc(
                        b.CreateLShr(llvm_size_xy, b.getInt64(32u)),
                        b.getInt32Ty()),
                    llvm_shift);
                llvm_width = b.CreateSelect(
                    b.CreateICmpUGT(llvm_width, b.getInt32(1u)),
                    llvm_width, b.getInt32(1u));
                llvm_height = b.CreateSelect(
                    b.CreateICmpUGT(llvm_height, b.getInt32(1u)),
                    llvm_height, b.getInt32(1u));
                llvm::SmallVector<llvm::Value *, 3u> llvm_sizes{
                    llvm_width, llvm_height};
                if (!is_2d) {
                    auto size_z_offset = dl.getStructLayout(llvm_slot_type)->getElementOffset(llvm_bindless_array_slot_type_texture3d_size_z_index);
                    auto llvm_depth = b.CreateLShr(
                        b.CreateTrunc(
                            b.CreateLoad(
                                b.getInt64Ty(), b.CreateInBoundsGEP(
                                                    b.getInt8Ty(), llvm_slot_ptr,
                                                    b.getInt64(size_z_offset))),
                            b.getInt32Ty()),
                        llvm_shift);
                    llvm_depth = b.CreateSelect(
                        b.CreateICmpUGT(llvm_depth, b.getInt32(1u)),
                        llvm_depth, b.getInt32(1u));
                    llvm_sizes.emplace_back(llvm_depth);
                }
                return _sample_texture_level(
                    b, is_2d, llvm_resource, llvm_sampler, llvm_coord,
                    llvm_sizes, llvm_filter, llvm_address,
                    llvm_is_packed_r10g10b10a2);
            };
            auto llvm_value = has_explicit_level || has_grad ?
                                  sample_mip_levels(
                                      sample_level, llvm_level0, llvm_level1,
                                      llvm_mip_linear, llvm_mip_weight) :
                                  sample_level(llvm_level0);
            return _safe_fp_cast(
                b, llvm_value, _get_llvm_type(inst->type())->reg_type);
        }
        case xir::ResourceQueryOp::BUFFER_DEVICE_ADDRESS: {
            auto llvm_buffer = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_result_type = _get_llvm_type(inst->type())->reg_type;
            return b.CreatePtrToInt(b.CreateExtractValue(llvm_buffer, llvm_buffer_type_ptr_index), llvm_result_type);
        }
        case xir::ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS: {
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_slot_ptr = _get_bindless_array_slot_pointer(b, llvm_bindless_array, llvm_index);
            // Use byte offset to access buffer pointer field
            auto llvm_slot_type = _get_llvm_bindless_array_slot_type();
            auto &dl = _llvm_module->getDataLayout();
            auto slot_struct_type = llvm::cast<llvm::StructType>(llvm_slot_type);
            auto buffer_ptr_offset = dl.getStructLayout(slot_struct_type)->getElementOffset(llvm_bindless_array_slot_type_buffer_ptr_index);
            auto llvm_buffer_ptr_addr = b.CreateInBoundsGEP(b.getInt8Ty(), llvm_slot_ptr, b.getInt64(buffer_ptr_offset));
            auto llvm_buffer_ptr = b.CreateLoad(llvm::PointerType::get(_llvm_context, amdgpu_address_space_global), llvm_buffer_ptr_addr);
            auto llvm_result_type = _get_llvm_type(inst->type())->reg_type;
            return b.CreatePtrToInt(llvm_buffer_ptr, llvm_result_type);
        }
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM: {
            LUISA_DEBUG_ASSERT(inst->type() == Type::of<luisa::float4x4>());
            auto llvm_accel = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_instance_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_instance_ptr = _get_accel_instance_pointer(b, llvm_accel, llvm_instance_index);
            auto llvm_affine_ptr = b.CreateStructGEP(_get_llvm_accel_instance_type(), llvm_instance_ptr, llvm_accel_instance_type_affine_index);
            return _load_accel_affine_matrix(b, llvm_affine_ptr);
        }
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID: {
            auto llvm_accel = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_instance_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_instance_ptr = _get_accel_instance_pointer(b, llvm_accel, llvm_instance_index);
            auto llvm_instance_type = _get_llvm_accel_instance_type();
            auto llvm_user_id_ptr = b.CreateStructGEP(llvm_instance_type, llvm_instance_ptr, llvm_accel_instance_type_user_id_index);
            auto llvm_user_id = b.CreateLoad(llvm_instance_type->getStructElementType(llvm_accel_instance_type_user_id_index), llvm_user_id_ptr);
            auto llvm_result_type = _get_llvm_type(inst->type())->reg_type;
            return b.CreateZExtOrTrunc(llvm_user_id, llvm_result_type);
        }
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK: {
            auto llvm_accel = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_instance_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_instance_ptr = _get_accel_instance_pointer(b, llvm_accel, llvm_instance_index);
            auto llvm_instance_type = _get_llvm_accel_instance_type();
            auto llvm_mask_ptr = b.CreateStructGEP(llvm_instance_type, llvm_instance_ptr, llvm_accel_instance_type_mask_index);
            auto llvm_mask = b.CreateLoad(llvm_instance_type->getStructElementType(llvm_accel_instance_type_mask_index), llvm_mask_ptr);
            auto llvm_result_type = _get_llvm_type(inst->type())->reg_type;
            return b.CreateZExtOrTrunc(
                b.CreateAnd(
                    llvm_mask,
                    llvm_accel_instance_visibility_mask_bits),
                llvm_result_type);
        }
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_MATRIX: {
            LUISA_DEBUG_ASSERT(inst->type() == Type::of<luisa::float4x4>());
            auto llvm_frame = _get_accel_instance_motion_frame(
                b,
                _get_llvm_value(b, func_ctx, inst->operand(0)),
                _get_llvm_value(b, func_ctx, inst->operand(1)),
                _get_llvm_value(b, func_ctx, inst->operand(2)),
                AccelMotionMode::MATRIX);
            return _load_accel_affine_matrix(b, llvm_frame);
        }
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_SRT: {
            LUISA_DEBUG_ASSERT(
                inst->type() == Type::of<MotionInstanceTransformSRT>());
            auto llvm_frame = _get_accel_instance_motion_frame(
                b,
                _get_llvm_value(b, func_ctx, inst->operand(0)),
                _get_llvm_value(b, func_ctx, inst->operand(1)),
                _get_llvm_value(b, func_ctx, inst->operand(2)),
                AccelMotionMode::SRT);
            auto llvm_srt = static_cast<llvm::Value *>(llvm::PoisonValue::get(
                _get_llvm_type(inst->type())->reg_type));
            auto load_component = [&](size_t offset) noexcept {
                auto address = b.CreateInBoundsGEP(
                    b.getInt8Ty(), llvm_frame, b.getInt64(offset));
                return b.CreateAlignedLoad(
                    b.getFloatTy(), address, llvm::Align{alignof(float)});
            };
            auto insert_component = [&](size_t offset,
                                        unsigned field,
                                        unsigned component) noexcept {
                llvm_srt = b.CreateInsertValue(
                    llvm_srt, load_component(offset), {field, component});
            };
            constexpr auto rotation = offsetof(hiprtFrameSRTQuaternion, rotation);
            constexpr auto pivot = offsetof(hiprtFrameSRTQuaternion, pivot);
            constexpr auto scale = offsetof(hiprtFrameSRTQuaternion, scale);
            constexpr auto shear = offsetof(hiprtFrameSRTQuaternion, shear);
            constexpr auto translation = offsetof(hiprtFrameSRTQuaternion, translation);
            for (auto i = 0u; i < 3u; i++) {
                insert_component(pivot + i * sizeof(float), 0u, i);
                insert_component(scale + i * sizeof(float), 2u, i);
                insert_component(shear + i * sizeof(float), 3u, i);
                insert_component(translation + i * sizeof(float), 4u, i);
            }
            for (auto i = 0u; i < 4u; i++) {
                insert_component(rotation + i * sizeof(float), 1u, i);
            }
            return llvm_srt;
        }
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST: {
            auto llvm_accel = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_ray = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_mask = _get_llvm_value(b, func_ctx, inst->operand(2));
            return _accel_trace_closest(
                b, func_ctx, llvm_accel, llvm_ray, nullptr, llvm_mask);
        }
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY: {
            auto llvm_accel = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_ray = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_mask = _get_llvm_value(b, func_ctx, inst->operand(2));
            return _accel_trace_any(
                b, func_ctx, llvm_accel, llvm_ray, nullptr, llvm_mask);
        }
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR: {
            auto llvm_accel = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_ray = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_time = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_mask = _get_llvm_value(b, func_ctx, inst->operand(3));
            return _accel_trace_closest(
                b, func_ctx, llvm_accel, llvm_ray, llvm_time, llvm_mask);
        }
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR: {
            auto llvm_accel = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_ray = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_time = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_mask = _get_llvm_value(b, func_ctx, inst->operand(3));
            return _accel_trace_any(
                b, func_ctx, llvm_accel, llvm_ray, llvm_time, llvm_mask);
        }
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL: [[fallthrough]];
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY: [[fallthrough]];
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR: [[fallthrough]];
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: {
            auto is_motion_query =
                op == xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR ||
                op == xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR;
            auto is_any =
                op == xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY ||
                op == xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR;
            LUISA_DEBUG_ASSERT(
                is_any ? inst->type() == Type::of<RayQueryAny>() :
                         inst->type() == Type::of<RayQueryAll>());
            auto llvm_accel = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_ray = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_time = is_motion_query ?
                                 _get_llvm_value(b, func_ctx, inst->operand(2)) :
                                 llvm::ConstantFP::get(b.getFloatTy(), 0.0);
            auto llvm_mask = _get_llvm_value(
                b, func_ctx, inst->operand(is_motion_query ? 3u : 2u));
            auto llvm_accel_handle = b.CreateExtractValue(llvm_accel, llvm_accel_type_handle_index);
            auto llvm_accel_instances = b.CreateExtractValue(llvm_accel, llvm_accel_type_instances_index);
            auto llvm_instance_data = b.CreateAddrSpaceCast(llvm_accel_instances, b.getPtrTy(0));
            auto llvm_ray_t_min = b.CreateExtractValue(llvm_ray, llvm_ray_type_t_min_index);
            auto llvm_ray_t_max = b.CreateExtractValue(llvm_ray, llvm_ray_type_t_max_index);
            auto llvm_ox = b.CreateExtractValue(llvm_ray, {llvm_ray_type_origin_index, 0u});
            auto llvm_oy = b.CreateExtractValue(llvm_ray, {llvm_ray_type_origin_index, 1u});
            auto llvm_oz = b.CreateExtractValue(llvm_ray, {llvm_ray_type_origin_index, 2u});
            auto llvm_dx = b.CreateExtractValue(llvm_ray, {llvm_ray_type_direction_index, 0u});
            auto llvm_dy = b.CreateExtractValue(llvm_ray, {llvm_ray_type_direction_index, 1u});
            auto llvm_dz = b.CreateExtractValue(llvm_ray, {llvm_ray_type_direction_index, 2u});
            // flags: 1 = terminate-on-first-hit (for "any" queries)
            auto llvm_flags = b.getInt32(is_any ? 1u : 0u);
            auto llvm_state_address = b.CreatePtrToInt(
                func_ctx.llvm_rq_state, _get_llvm_ray_query_type(),
                "ray.query.state.address");
            llvm::SmallVector<llvm::Value *, 16> llvm_initialize_args{
                llvm_accel_handle,
                llvm_instance_data,
                llvm_ox, llvm_oy, llvm_oz,
                llvm_dx, llvm_dy, llvm_dz,
                llvm_ray_t_min, llvm_ray_t_max};
            if (!_uses_hardware_rt_stack ||
                !func_ctx.llvm_rq_state_uses_resumable_abi) {
                llvm_initialize_args.emplace_back(llvm_time);
            }
            llvm_initialize_args.emplace_back(llvm_mask);
            llvm_initialize_args.emplace_back(llvm_flags);
            llvm_initialize_args.emplace_back(func_ctx.llvm_rt_stack_size);
            llvm_initialize_args.emplace_back(func_ctx.llvm_rt_stack_count);
            llvm_initialize_args.emplace_back(func_ctx.llvm_rt_stack_data);
            (void)_call_ray_query_intrinsic(
                b, func_ctx, llvm_ray_query_intrinsic_name_initialize, b.getVoidTy(),
                llvm_initialize_args);
            // Ray-query operations must follow the object operand instead of
            // implicitly using the current function's state. In particular,
            // lower_ray_query_loop outlines candidate handlers into separate
            // callables, each of which has its own local state allocation.
            // Encode this query's actual private state pointer in the opaque
            // object so a reference passed to an outlined handler still refers
            // to the traversal initialized above.
            return llvm_state_address;
        }
        default: LUISA_NOT_IMPLEMENTED();
    }
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_translate_resource_read_inst(IB &b, const FunctionContext &func_ctx, const xir::ResourceReadInst *inst) noexcept {
    switch (auto op = inst->op()) {
        case xir::ResourceReadOp::BUFFER_READ: [[fallthrough]];
        case xir::ResourceReadOp::BUFFER_VOLATILE_READ: {
            auto is_volatile = op == xir::ResourceReadOp::BUFFER_VOLATILE_READ;
            if (is_volatile) {
                b.CreateFence(llvm::AtomicOrdering::AcquireRelease,
                              _llvm_context.getOrInsertSyncScopeID("agent"));
            }
            auto llvm_buffer = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            LUISA_DEBUG_ASSERT(inst->type() == inst->operand(0)->type()->element());
            auto elem_type = inst->type();
            auto llvm_elem_ptr = _get_buffer_element_pointer(b, llvm_buffer, llvm_index, elem_type->size(), elem_type->size());
            return _load_llvm_value(b, llvm_elem_ptr, elem_type, is_volatile);
        }
        case xir::ResourceReadOp::BYTE_BUFFER_READ: [[fallthrough]];
        case xir::ResourceReadOp::BYTE_BUFFER_VOLATILE_READ: {
            auto is_volatile = op == xir::ResourceReadOp::BYTE_BUFFER_VOLATILE_READ;
            if (is_volatile) {
                b.CreateFence(llvm::AtomicOrdering::AcquireRelease,
                              _llvm_context.getOrInsertSyncScopeID("agent"));
            }
            auto llvm_buffer = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_byte_offset = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto elem_type = inst->type();
            auto llvm_elem_ptr = _get_buffer_element_pointer(b, llvm_buffer, llvm_byte_offset, 1, elem_type->size());
            return _load_llvm_value(b, llvm_elem_ptr, elem_type, is_volatile);
        }
        case xir::ResourceReadOp::TEXTURE2D_READ: {
            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_texture_handle = b.CreateExtractValue(llvm_texture, llvm_texture_type_handle_index);
            auto llvm_texture_storage = _get_direct_texture_storage(b, llvm_texture);
            auto llvm_texture_is_signed = b.getInt1(inst->type()->is_int_vector());
            auto llvm_result_type = _get_llvm_type(inst->type())->reg_type;
            LUISA_DEBUG_ASSERT(llvm_result_type->isVectorTy());
            auto llvm_func = _get_texture2d_read_function(llvm::cast<llvm::VectorType>(llvm_result_type));
            return b.CreateCall(llvm_func, {llvm_texture_handle, llvm_texture_storage, llvm_texture_is_signed, llvm_coord});
        }
        case xir::ResourceReadOp::TEXTURE3D_READ: {
            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_texture_handle = b.CreateExtractValue(llvm_texture, llvm_texture_type_handle_index);
            auto llvm_texture_storage = _get_direct_texture_storage(b, llvm_texture);
            auto llvm_texture_is_signed = b.getInt1(inst->type()->is_int_vector());
            auto llvm_result_type = _get_llvm_type(inst->type())->reg_type;
            LUISA_DEBUG_ASSERT(llvm_result_type->isVectorTy());
            auto llvm_func = _get_texture3d_read_function(llvm::cast<llvm::VectorType>(llvm_result_type));
            return b.CreateCall(llvm_func, {llvm_texture_handle, llvm_texture_storage, llvm_texture_is_signed, llvm_coord});
        }
        case xir::ResourceReadOp::BINDLESS_BUFFER_READ: [[fallthrough]];
        case xir::ResourceReadOp::BINDLESS_BYTE_BUFFER_READ: {
            auto llvm_bindless = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_slot_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_slot_ptr = _get_bindless_array_slot_pointer(b, llvm_bindless, llvm_slot_index);
            // Load buffer struct fields using byte offsets
            auto llvm_slot_type = _get_llvm_bindless_array_slot_type();
            auto &dl = _llvm_module->getDataLayout();
            auto slot_struct_type = llvm::cast<llvm::StructType>(llvm_slot_type);
            auto ptr_offset = dl.getStructLayout(slot_struct_type)->getElementOffset(llvm_bindless_array_slot_type_buffer_ptr_index);
            auto size_offset = dl.getStructLayout(slot_struct_type)->getElementOffset(llvm_bindless_array_slot_type_buffer_size_index);
            auto ptr_addr = b.CreateInBoundsGEP(b.getInt8Ty(), llvm_slot_ptr, b.getInt64(ptr_offset));
            auto size_addr = b.CreateInBoundsGEP(b.getInt8Ty(), llvm_slot_ptr, b.getInt64(size_offset));
            auto llvm_buffer_ptr = b.CreateLoad(llvm::PointerType::get(_llvm_context, amdgpu_address_space_global), ptr_addr);
            auto llvm_buffer_size = b.CreateLoad(llvm::Type::getInt64Ty(_llvm_context), size_addr);
            // Construct buffer value
            auto llvm_buffer_type = _get_llvm_buffer_type();
            auto llvm_buffer = llvm::cast<llvm::Value>(llvm::Constant::getNullValue(llvm_buffer_type));
            llvm_buffer = b.CreateInsertValue(llvm_buffer, llvm_buffer_ptr, llvm_buffer_type_ptr_index);
            llvm_buffer = b.CreateInsertValue(llvm_buffer, llvm_buffer_size, llvm_buffer_type_size_index);
            auto llvm_index_or_offset = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto elem_type = inst->type();
            auto index_stride = (op == xir::ResourceReadOp::BINDLESS_BUFFER_READ) ? elem_type->size() : 1;
            auto llvm_elem_ptr = _get_buffer_element_pointer(b, llvm_buffer, llvm_index_or_offset, index_stride, elem_type->size());
            return _load_llvm_value(b, llvm_elem_ptr, elem_type);
        }
        case xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ: {
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_handle = _get_bindless_array_texture_handle(b, llvm_bindless_array, llvm_index, 2);
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_coord_x = b.CreateExtractElement(llvm_coord, b.getInt64(0));
            auto llvm_coord_y = b.CreateExtractElement(llvm_coord, b.getInt64(1));
            auto llvm_v8i32_type = llvm::FixedVectorType::get(b.getInt32Ty(), 8);
            auto llvm_f32x4_type = llvm::FixedVectorType::get(b.getFloatTy(), 4);
            auto llvm_const_ptr = b.CreateIntToPtr(llvm_handle, llvm::PointerType::get(_llvm_context, amdgpu_address_space_global), "tex.ptr");
            auto llvm_rsrc = b.CreateLoad(llvm_v8i32_type, llvm_const_ptr, "tex.rsrc");
            auto llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_load_2d,
                                                 {llvm_f32x4_type, b.getInt32Ty(), llvm_v8i32_type},
                                                 {b.getInt32(15), llvm_coord_x, llvm_coord_y, llvm_rsrc, b.getInt32(0), b.getInt32(0)});
            auto llvm_result_x = b.CreateExtractElement(llvm_result, b.getInt64(0));
            auto llvm_result_y = b.CreateExtractElement(llvm_result, b.getInt64(1));
            auto llvm_result_z = b.CreateExtractElement(llvm_result, b.getInt64(2));
            auto llvm_result_w = b.CreateExtractElement(llvm_result, b.getInt64(3));
            auto llvm_value = _create_llvm_vector(b, {llvm_result_x, llvm_result_y, llvm_result_z, llvm_result_w});
            auto llvm_result_type = llvm::cast<llvm::VectorType>(
                _get_llvm_type(inst->type())->reg_type);
            auto llvm_regular = _safe_fp_cast(b, llvm_value, llvm_result_type);
            auto llvm_raw_i32 = b.CreateBitCast(
                llvm_result, llvm::FixedVectorType::get(b.getInt32Ty(), 4u));
            auto llvm_packed = b.CreateExtractElement(llvm_raw_i32, b.getInt64(0u));
            auto llvm_decoded = _unpack_r10g10b10a2(
                b, llvm_packed, llvm_result_type);
            auto llvm_storage = _get_bindless_array_texture_storage(
                b, llvm_bindless_array, llvm_index, 2);
            return b.CreateSelect(
                b.CreateICmpEQ(
                    llvm_storage,
                    b.getInt64(to_underlying(PixelStorage::R10G10B10A2))),
                llvm_decoded, llvm_regular);
        }
        case xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL: {
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_level = _get_llvm_value(b, func_ctx, inst->operand(3));
            auto llvm_handle = _get_bindless_array_texture_handle(b, llvm_bindless_array, llvm_index, 2, llvm_level);
            auto llvm_coord_x = b.CreateExtractElement(llvm_coord, b.getInt64(0));
            auto llvm_coord_y = b.CreateExtractElement(llvm_coord, b.getInt64(1));
            auto llvm_v8i32_type = llvm::FixedVectorType::get(b.getInt32Ty(), 8);
            auto llvm_f32x4_type = llvm::FixedVectorType::get(b.getFloatTy(), 4);
            auto llvm_const_ptr = b.CreateIntToPtr(llvm_handle, llvm::PointerType::get(_llvm_context, amdgpu_address_space_global), "tex.ptr");
            auto llvm_rsrc = b.CreateLoad(llvm_v8i32_type, llvm_const_ptr, "tex.rsrc");
            auto llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_load_2d,
                                                 {llvm_f32x4_type, b.getInt32Ty(), llvm_v8i32_type},
                                                 {b.getInt32(15), llvm_coord_x, llvm_coord_y, llvm_rsrc, b.getInt32(0), b.getInt32(0)});
            auto llvm_result_x = b.CreateExtractElement(llvm_result, b.getInt64(0));
            auto llvm_result_y = b.CreateExtractElement(llvm_result, b.getInt64(1));
            auto llvm_result_z = b.CreateExtractElement(llvm_result, b.getInt64(2));
            auto llvm_result_w = b.CreateExtractElement(llvm_result, b.getInt64(3));
            auto llvm_value = _create_llvm_vector(b, {llvm_result_x, llvm_result_y, llvm_result_z, llvm_result_w});
            auto llvm_result_type = llvm::cast<llvm::VectorType>(
                _get_llvm_type(inst->type())->reg_type);
            auto llvm_regular = _safe_fp_cast(b, llvm_value, llvm_result_type);
            auto llvm_raw_i32 = b.CreateBitCast(
                llvm_result, llvm::FixedVectorType::get(b.getInt32Ty(), 4u));
            auto llvm_packed = b.CreateExtractElement(llvm_raw_i32, b.getInt64(0u));
            auto llvm_decoded = _unpack_r10g10b10a2(
                b, llvm_packed, llvm_result_type);
            auto llvm_storage = _get_bindless_array_texture_storage(
                b, llvm_bindless_array, llvm_index, 2);
            return b.CreateSelect(
                b.CreateICmpEQ(
                    llvm_storage,
                    b.getInt64(to_underlying(PixelStorage::R10G10B10A2))),
                llvm_decoded, llvm_regular);
        }
        case xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ: {
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_handle = _get_bindless_array_texture_handle(b, llvm_bindless_array, llvm_index, 3);
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_coord_x = b.CreateExtractElement(llvm_coord, b.getInt64(0));
            auto llvm_coord_y = b.CreateExtractElement(llvm_coord, b.getInt64(1));
            auto llvm_coord_z = b.CreateExtractElement(llvm_coord, b.getInt64(2));
            auto llvm_v8i32_type = llvm::FixedVectorType::get(b.getInt32Ty(), 8);
            auto llvm_f32x4_type = llvm::FixedVectorType::get(b.getFloatTy(), 4);
            auto llvm_const_ptr = b.CreateIntToPtr(llvm_handle, llvm::PointerType::get(_llvm_context, amdgpu_address_space_global), "tex.ptr");
            auto llvm_rsrc = b.CreateLoad(llvm_v8i32_type, llvm_const_ptr, "tex.rsrc");
            auto llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_load_3d,
                                                 {llvm_f32x4_type, b.getInt32Ty(), llvm_v8i32_type},
                                                 {b.getInt32(15), llvm_coord_x, llvm_coord_y, llvm_coord_z, llvm_rsrc, b.getInt32(0), b.getInt32(0)});
            auto llvm_result_x = b.CreateExtractElement(llvm_result, b.getInt64(0));
            auto llvm_result_y = b.CreateExtractElement(llvm_result, b.getInt64(1));
            auto llvm_result_z = b.CreateExtractElement(llvm_result, b.getInt64(2));
            auto llvm_result_w = b.CreateExtractElement(llvm_result, b.getInt64(3));
            auto llvm_value = _create_llvm_vector(b, {llvm_result_x, llvm_result_y, llvm_result_z, llvm_result_w});
            auto llvm_result_type = llvm::cast<llvm::VectorType>(
                _get_llvm_type(inst->type())->reg_type);
            auto llvm_regular = _safe_fp_cast(b, llvm_value, llvm_result_type);
            auto llvm_raw_i32 = b.CreateBitCast(
                llvm_result, llvm::FixedVectorType::get(b.getInt32Ty(), 4u));
            auto llvm_packed = b.CreateExtractElement(llvm_raw_i32, b.getInt64(0u));
            auto llvm_decoded = _unpack_r10g10b10a2(
                b, llvm_packed, llvm_result_type);
            auto llvm_storage = _get_bindless_array_texture_storage(
                b, llvm_bindless_array, llvm_index, 3);
            return b.CreateSelect(
                b.CreateICmpEQ(
                    llvm_storage,
                    b.getInt64(to_underlying(PixelStorage::R10G10B10A2))),
                llvm_decoded, llvm_regular);
        }
        case xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL: {
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_level = _get_llvm_value(b, func_ctx, inst->operand(3));
            auto llvm_handle = _get_bindless_array_texture_handle(b, llvm_bindless_array, llvm_index, 3, llvm_level);
            auto llvm_coord_x = b.CreateExtractElement(llvm_coord, b.getInt64(0));
            auto llvm_coord_y = b.CreateExtractElement(llvm_coord, b.getInt64(1));
            auto llvm_coord_z = b.CreateExtractElement(llvm_coord, b.getInt64(2));
            auto llvm_v8i32_type = llvm::FixedVectorType::get(b.getInt32Ty(), 8);
            auto llvm_f32x4_type = llvm::FixedVectorType::get(b.getFloatTy(), 4);
            auto llvm_const_ptr = b.CreateIntToPtr(llvm_handle, llvm::PointerType::get(_llvm_context, amdgpu_address_space_global), "tex.ptr");
            auto llvm_rsrc = b.CreateLoad(llvm_v8i32_type, llvm_const_ptr, "tex.rsrc");
            auto llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_load_3d,
                                                 {llvm_f32x4_type, b.getInt32Ty(), llvm_v8i32_type},
                                                 {b.getInt32(15), llvm_coord_x, llvm_coord_y, llvm_coord_z, llvm_rsrc, b.getInt32(0), b.getInt32(0)});
            auto llvm_result_x = b.CreateExtractElement(llvm_result, b.getInt64(0));
            auto llvm_result_y = b.CreateExtractElement(llvm_result, b.getInt64(1));
            auto llvm_result_z = b.CreateExtractElement(llvm_result, b.getInt64(2));
            auto llvm_result_w = b.CreateExtractElement(llvm_result, b.getInt64(3));
            auto llvm_value = _create_llvm_vector(b, {llvm_result_x, llvm_result_y, llvm_result_z, llvm_result_w});
            auto llvm_result_type = llvm::cast<llvm::VectorType>(
                _get_llvm_type(inst->type())->reg_type);
            auto llvm_regular = _safe_fp_cast(b, llvm_value, llvm_result_type);
            auto llvm_raw_i32 = b.CreateBitCast(
                llvm_result, llvm::FixedVectorType::get(b.getInt32Ty(), 4u));
            auto llvm_packed = b.CreateExtractElement(llvm_raw_i32, b.getInt64(0u));
            auto llvm_decoded = _unpack_r10g10b10a2(
                b, llvm_packed, llvm_result_type);
            auto llvm_storage = _get_bindless_array_texture_storage(
                b, llvm_bindless_array, llvm_index, 3);
            return b.CreateSelect(
                b.CreateICmpEQ(
                    llvm_storage,
                    b.getInt64(to_underlying(PixelStorage::R10G10B10A2))),
                llvm_decoded, llvm_regular);
        }
        case xir::ResourceReadOp::DEVICE_ADDRESS_READ: {
            auto llvm_address = b.CreateZExt(_get_llvm_value(b, func_ctx, inst->operand(0)), b.getInt64Ty(), "", true);
            auto llvm_ptr = b.CreateIntToPtr(llvm_address, b.getPtrTy());
            return _load_llvm_value(b, llvm_ptr, inst->type());
        }
        case xir::ResourceReadOp::COOPERATIVE_MUL_ADD:
        case xir::ResourceReadOp::BINDLESS_COOPERATIVE_MUL_ADD:
        case xir::ResourceReadOp::COOPERATIVE_MUL:
        case xir::ResourceReadOp::BINDLESS_COOPERATIVE_MUL:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_LOAD:
        case xir::ResourceReadOp::BINDLESS_COOPERATIVE_VECTOR_LOAD:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_SPLAT:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_CAST:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_WORKGROUP_LOAD:
        // Future cooperative-vector element-wise operations — TODO: implement
        // in the HIP LLVM backend.
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_DOT:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_ABS:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_SIGN:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_FLOOR:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_CEIL:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_FRACT:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_TRUNC:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_ROUND:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_RINT:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_SQRT:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_RSQRT:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_EXP2:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_EXP10:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_LOG2:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_LOG10:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_SATURATE:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_ISINF:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_ISNAN:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_SIN:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_COS:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_TAN:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_ASIN:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_ACOS:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_SINH:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_COSH:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_ASINH:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_ACOSH:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_ATANH:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_MIX:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_LERP:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_POW:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_STEP:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_SMOOTHSTEP:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_ADD:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_SUB:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_MUL:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_DIV:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_LESS:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_LESS_EQUAL:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_GREATER:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_GREATER_EQUAL:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_EQUAL:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_NOT_EQUAL: break;
    }
    LUISA_NOT_IMPLEMENTED();
}

void HIPCodegenLLVMImpl::_translate_resource_write_inst(IB &b, FunctionContext &func_ctx, const xir::ResourceWriteInst *inst) noexcept {
    switch (auto op = inst->op()) {
        case xir::ResourceWriteOp::BUFFER_WRITE: [[fallthrough]];
        case xir::ResourceWriteOp::BUFFER_VOLATILE_WRITE: {
            auto is_volatile = op == xir::ResourceWriteOp::BUFFER_VOLATILE_WRITE;
            auto llvm_buffer = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(2));
            LUISA_DEBUG_ASSERT(inst->operand(2)->type() == inst->operand(0)->type()->element());
            auto elem_type = inst->operand(2)->type();
            auto llvm_elem_ptr = _get_buffer_element_pointer(b, llvm_buffer, llvm_index, elem_type->size(), elem_type->size());
            _store_llvm_value(b, llvm_elem_ptr, llvm_value, elem_type, is_volatile);
            if (is_volatile) {
                b.CreateFence(llvm::AtomicOrdering::AcquireRelease,
                              _llvm_context.getOrInsertSyncScopeID("agent"));
            }
            return;
        }
        case xir::ResourceWriteOp::BYTE_BUFFER_WRITE: [[fallthrough]];
        case xir::ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE: {
            auto is_volatile = op == xir::ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE;
            auto llvm_buffer = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_byte_offset = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto elem_type = inst->operand(2)->type();
            auto llvm_elem_ptr = _get_buffer_element_pointer(b, llvm_buffer, llvm_byte_offset, 1, elem_type->size());
            _store_llvm_value(b, llvm_elem_ptr, llvm_value, elem_type, is_volatile);
            if (is_volatile) {
                b.CreateFence(llvm::AtomicOrdering::AcquireRelease,
                              _llvm_context.getOrInsertSyncScopeID("agent"));
            }
            return;
        }
        case xir::ResourceWriteOp::TEXTURE2D_WRITE: {
            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_texture_handle = b.CreateExtractValue(llvm_texture, llvm_texture_type_handle_index);
            auto llvm_texture_storage = _get_direct_texture_storage(b, llvm_texture);
            auto llvm_texture_is_signed = b.getInt1(
                inst->operand(2)->type()->is_int_vector());
            LUISA_DEBUG_ASSERT(llvm_value->getType()->isVectorTy());
            auto llvm_func = _get_texture2d_write_function(llvm::cast<llvm::VectorType>(llvm_value->getType()));
            b.CreateCall(llvm_func, {llvm_texture_handle, llvm_texture_storage, llvm_texture_is_signed, llvm_coord, llvm_value});
            return;
        }
        case xir::ResourceWriteOp::TEXTURE3D_WRITE: {
            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_texture_handle = b.CreateExtractValue(llvm_texture, llvm_texture_type_handle_index);
            auto llvm_texture_storage = _get_direct_texture_storage(b, llvm_texture);
            auto llvm_texture_is_signed = b.getInt1(
                inst->operand(2)->type()->is_int_vector());
            LUISA_DEBUG_ASSERT(llvm_value->getType()->isVectorTy());
            auto llvm_func = _get_texture3d_write_function(llvm::cast<llvm::VectorType>(llvm_value->getType()));
            b.CreateCall(llvm_func, {llvm_texture_handle, llvm_texture_storage, llvm_texture_is_signed, llvm_coord, llvm_value});
            return;
        }
        case xir::ResourceWriteOp::BINDLESS_BUFFER_WRITE: [[fallthrough]];
        case xir::ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE: {
            auto llvm_bindless = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_slot_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_slot_ptr = _get_bindless_array_slot_pointer(b, llvm_bindless, llvm_slot_index);
            // Load buffer struct fields using byte offsets
            auto llvm_slot_type = _get_llvm_bindless_array_slot_type();
            auto &dl = _llvm_module->getDataLayout();
            auto slot_struct_type = llvm::cast<llvm::StructType>(llvm_slot_type);
            auto ptr_offset = dl.getStructLayout(slot_struct_type)->getElementOffset(llvm_bindless_array_slot_type_buffer_ptr_index);
            auto size_offset = dl.getStructLayout(slot_struct_type)->getElementOffset(llvm_bindless_array_slot_type_buffer_size_index);
            auto ptr_addr = b.CreateInBoundsGEP(b.getInt8Ty(), llvm_slot_ptr, b.getInt64(ptr_offset));
            auto size_addr = b.CreateInBoundsGEP(b.getInt8Ty(), llvm_slot_ptr, b.getInt64(size_offset));
            auto llvm_buffer_ptr = b.CreateLoad(llvm::PointerType::get(_llvm_context, amdgpu_address_space_global), ptr_addr);
            auto llvm_buffer_size = b.CreateLoad(llvm::Type::getInt64Ty(_llvm_context), size_addr);
            // Construct buffer value
            auto llvm_buffer_type = _get_llvm_buffer_type();
            auto llvm_buffer = llvm::cast<llvm::Value>(llvm::Constant::getNullValue(llvm_buffer_type));
            llvm_buffer = b.CreateInsertValue(llvm_buffer, llvm_buffer_ptr, llvm_buffer_type_ptr_index);
            llvm_buffer = b.CreateInsertValue(llvm_buffer, llvm_buffer_size, llvm_buffer_type_size_index);
            auto llvm_index_or_offset = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto value = inst->operand(3);
            auto elem_type = value->type();
            auto index_stride = (op == xir::ResourceWriteOp::BINDLESS_BUFFER_WRITE) ? elem_type->size() : 1;
            auto llvm_elem_ptr = _get_buffer_element_pointer(b, llvm_buffer, llvm_index_or_offset, index_stride, elem_type->size());
            auto llvm_value = _get_llvm_value(b, func_ctx, value);
            return _store_llvm_value(b, llvm_elem_ptr, llvm_value, elem_type);
        }
        case xir::ResourceWriteOp::INDIRECT_DISPATCH_SET_COUNT: {
            auto llvm_indirect = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_data = b.CreateExtractValue(
                llvm_indirect, llvm_buffer_type_ptr_index, "indirect.data");
            auto llvm_count = _get_llvm_value(b, func_ctx, inst->operand(1));
            llvm_count = b.CreateZExtOrTrunc(llvm_count, b.getInt32Ty());
            b.CreateStore(llvm_count, llvm_data);
            return;
        }
        case xir::ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL: {
            auto llvm_indirect = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_data = b.CreateExtractValue(
                llvm_indirect, llvm_buffer_type_ptr_index, "indirect.data");
            auto llvm_range = b.CreateExtractValue(
                llvm_indirect, llvm_buffer_type_size_index, "indirect.range");
            auto llvm_offset = b.CreateTrunc(llvm_range, b.getInt32Ty());
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            llvm_index = b.CreateZExtOrTrunc(llvm_index, b.getInt32Ty());
            auto llvm_record_index = b.CreateAdd(llvm_index, llvm_offset, "indirect.index");
            auto llvm_record_offset = b.CreateAdd(
                b.CreateMul(b.CreateZExt(llvm_record_index, b.getInt64Ty()),
                            b.getInt64(sizeof(HIPBuffer::IndirectDispatch))),
                b.getInt64(sizeof(HIPBuffer::IndirectHeader)),
                "indirect.record.offset");
            auto store_i32 = [&](size_t byte_offset, llvm::Value *value) noexcept {
                auto llvm_offset_in_record = b.CreateAdd(
                    llvm_record_offset, b.getInt64(byte_offset));
                auto llvm_ptr = b.CreateInBoundsGEP(
                    b.getInt8Ty(), llvm_data, llvm_offset_in_record);
                b.CreateStore(b.CreateZExtOrTrunc(value, b.getInt32Ty()), llvm_ptr);
            };
            auto llvm_block_size = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_dispatch_size = _get_llvm_value(b, func_ctx, inst->operand(3));
            for (auto i = 0u; i < 3u; i++) {
                store_i32(offsetof(HIPBuffer::IndirectDispatch, block_size) +
                              sizeof(uint32_t) * i,
                          b.CreateExtractElement(llvm_block_size, b.getInt64(i)));
                store_i32(offsetof(HIPBuffer::IndirectDispatch, dispatch_size_and_kernel_id) +
                              sizeof(uint32_t) * i,
                          b.CreateExtractElement(llvm_dispatch_size, b.getInt64(i)));
            }
            store_i32(offsetof(HIPBuffer::IndirectDispatch, dispatch_size_and_kernel_id) +
                          sizeof(uint32_t) * 3u,
                      _get_llvm_value(b, func_ctx, inst->operand(4)));
            return;
        }
        case xir::ResourceWriteOp::DEVICE_ADDRESS_WRITE: {
            auto llvm_address = b.CreateZExt(_get_llvm_value(b, func_ctx, inst->operand(0)), b.getInt64Ty(), "", true);
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_ptr = b.CreateIntToPtr(llvm_address, b.getPtrTy());
            _store_llvm_value(b, llvm_ptr, llvm_value, inst->operand(1)->type());
            return;
        }
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_TRANSFORM: {
            auto llvm_accel = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_instance_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_transform = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_instance_ptr = _get_accel_instance_pointer(b, llvm_accel, llvm_instance_index);
            auto llvm_affine_ptr = b.CreateStructGEP(_get_llvm_accel_instance_type(), llvm_instance_ptr, llvm_accel_instance_type_affine_index);
            _store_accel_affine_matrix(b, llvm_affine_ptr, llvm_transform);
            return;
        }
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_VISIBILITY_MASK: {
            auto llvm_accel = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_instance_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_instance_type = _get_llvm_accel_instance_type();
            auto llvm_mask_type = llvm_instance_type->getStructElementType(llvm_accel_instance_type_mask_index);
            auto llvm_instance_ptr = _get_accel_instance_pointer(b, llvm_accel, llvm_instance_index);
            auto llvm_mask_ptr = b.CreateStructGEP(llvm_instance_type, llvm_instance_ptr, llvm_accel_instance_type_mask_index);
            auto llvm_old_mask = b.CreateLoad(llvm_mask_type, llvm_mask_ptr);
            auto llvm_public_mask = b.CreateAnd(
                b.CreateZExtOrTrunc(
                    _get_llvm_value(b, func_ctx, inst->operand(2)),
                    llvm_mask_type),
                llvm_accel_instance_visibility_mask_bits);
            auto llvm_packed_opacity = b.CreateAnd(
                llvm_old_mask,
                llvm_accel_instance_packed_opacity_bit);
            b.CreateStore(
                b.CreateOr(llvm_packed_opacity, llvm_public_mask),
                llvm_mask_ptr);
            return;
        }
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_OPACITY: {
            auto llvm_accel = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_instance_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_is_opaque = _get_llvm_value(b, func_ctx, inst->operand(2));
            _set_accel_instance_opacity(b, llvm_accel, llvm_instance_index, llvm_is_opaque);
            return;
        }
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID: {
            auto llvm_accel = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_instance_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_instance_type = _get_llvm_accel_instance_type();
            auto llvm_user_id_type = llvm_instance_type->getStructElementType(llvm_accel_instance_type_user_id_index);
            auto llvm_user_id = b.CreateZExtOrTrunc(_get_llvm_value(b, func_ctx, inst->operand(2)), llvm_user_id_type);
            auto llvm_instance_ptr = _get_accel_instance_pointer(b, llvm_accel, llvm_instance_index);
            auto llvm_user_id_ptr = b.CreateStructGEP(llvm_instance_type, llvm_instance_ptr, llvm_accel_instance_type_user_id_index);
            b.CreateStore(llvm_user_id, llvm_user_id_ptr);
            return;
        }
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX: {
            auto llvm_frame = _get_accel_instance_motion_frame(
                b,
                _get_llvm_value(b, func_ctx, inst->operand(0)),
                _get_llvm_value(b, func_ctx, inst->operand(1)),
                _get_llvm_value(b, func_ctx, inst->operand(2)),
                AccelMotionMode::MATRIX);
            _store_accel_affine_matrix(
                b, llvm_frame,
                _get_llvm_value(b, func_ctx, inst->operand(3)));
            return;
        }
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT: {
            auto llvm_frame = _get_accel_instance_motion_frame(
                b,
                _get_llvm_value(b, func_ctx, inst->operand(0)),
                _get_llvm_value(b, func_ctx, inst->operand(1)),
                _get_llvm_value(b, func_ctx, inst->operand(2)),
                AccelMotionMode::SRT);
            auto llvm_srt = _get_llvm_value(b, func_ctx, inst->operand(3));
            auto store_component = [&](size_t offset,
                                       unsigned field,
                                       unsigned component) noexcept {
                auto address = b.CreateInBoundsGEP(
                    b.getInt8Ty(), llvm_frame, b.getInt64(offset));
                auto value = b.CreateExtractValue(
                    llvm_srt, {field, component});
                b.CreateAlignedStore(
                    value, address, llvm::Align{alignof(float)});
            };
            constexpr auto rotation = offsetof(hiprtFrameSRTQuaternion, rotation);
            constexpr auto pivot = offsetof(hiprtFrameSRTQuaternion, pivot);
            constexpr auto scale = offsetof(hiprtFrameSRTQuaternion, scale);
            constexpr auto shear = offsetof(hiprtFrameSRTQuaternion, shear);
            constexpr auto translation = offsetof(hiprtFrameSRTQuaternion, translation);
            for (auto i = 0u; i < 3u; i++) {
                store_component(pivot + i * sizeof(float), 0u, i);
                store_component(scale + i * sizeof(float), 2u, i);
                store_component(shear + i * sizeof(float), 3u, i);
                store_component(translation + i * sizeof(float), 4u, i);
            }
            for (auto i = 0u; i < 4u; i++) {
                store_component(rotation + i * sizeof(float), 1u, i);
            }
            return;
        }
        case xir::ResourceWriteOp::COOPERATIVE_OUTER_PRODUCT_ACCUMULATE:
        case xir::ResourceWriteOp::COOPERATIVE_VECTOR_ACCUMULATE:
        case xir::ResourceWriteOp::COOPERATIVE_VECTOR_STORE:
        case xir::ResourceWriteOp::BINDLESS_COOPERATIVE_VECTOR_STORE:
        case xir::ResourceWriteOp::COOPERATIVE_VECTOR_WORKGROUP_STORE: break;
    }
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_get_buffer_element_pointer(IB &b, llvm::Value *buffer, llvm::Value *index, size_t index_stride, size_t element_size) noexcept {
    auto buffer_data_ptr = b.CreateExtractValue(buffer, llvm_buffer_type_ptr_index);
    auto buffer_size_bytes = b.CreateExtractValue(buffer, llvm_buffer_type_size_index);
    auto size_type = buffer_size_bytes->getType();
    LUISA_DEBUG_ASSERT(size_type->isIntegerTy(64));
    index = b.CreateZExt(index, size_type, "", true);
    auto offset_bytes = index_stride == 1 ? index : b.CreateMul(index, b.getInt64(index_stride), "", true, true);
    return b.CreateInBoundsGEP(b.getInt8Ty(), buffer_data_ptr, offset_bytes);
}

llvm::Value *HIPCodegenLLVMImpl::_get_bindless_array_slot_pointer(IB &b, llvm::Value *bindless_array, llvm::Value *slot_index) noexcept {
    auto slots = b.CreateExtractValue(bindless_array, llvm_bindless_array_type_slots_index);
    auto slot_count = b.CreateExtractValue(bindless_array, llvm_bindless_array_type_size_index);
    slot_index = b.CreateZExt(slot_index, slot_count->getType(), "", true);
    // Bounds check: slot_index < slot_count
    auto slot_index_in_bounds = b.CreateICmpULT(slot_index, slot_count);
    _create_assertion_with_message(b, slot_index_in_bounds, "Bindless array slot index out of bounds.");
    // Use byte offset calculation to avoid LLVM opaque pointer issues with struct GEP
    auto slot_type = _get_llvm_bindless_array_slot_type();
    auto slot_size = _llvm_module->getDataLayout().getTypeAllocSize(slot_type);
    auto offset_bytes = b.CreateMul(slot_index, b.getInt64(slot_size), "", true, true);
    return b.CreateInBoundsGEP(b.getInt8Ty(), slots, offset_bytes);
}

llvm::Value *HIPCodegenLLVMImpl::_get_bindless_array_texture_storage(
    IB &b, llvm::Value *bindless_array,
    llvm::Value *slot_index, int dim) noexcept {
    LUISA_DEBUG_ASSERT(dim == 2 || dim == 3);
    auto slot_ptr = _get_bindless_array_slot_pointer(
        b, bindless_array, slot_index);
    auto slot_type = llvm::cast<llvm::StructType>(
        _get_llvm_bindless_array_slot_type());
    auto levels_i = dim == 2 ?
                        llvm_bindless_array_slot_type_texture2d_levels_index :
                        llvm_bindless_array_slot_type_texture3d_levels_index;
    auto levels_offset = _llvm_module->getDataLayout()
                             .getStructLayout(slot_type)
                             ->getElementOffset(levels_i);
    auto levels_sampler = b.CreateLoad(
        b.getInt64Ty(), b.CreateInBoundsGEP(
                            b.getInt8Ty(), slot_ptr,
                            b.getInt64(levels_offset)));
    return b.CreateAnd(
        b.CreateLShr(
            levels_sampler,
            b.getInt64(HIPBindlessArray::texture_storage_shift)),
        b.getInt64(HIPBindlessArray::texture_storage_mask));
}

llvm::Value *HIPCodegenLLVMImpl::_get_bindless_array_texture_handle(IB &b, llvm::Value *bindless_array,
                                                                    llvm::Value *slot_index, int dim,
                                                                    llvm::Value *level) noexcept {
    auto slot_ptr = _get_bindless_array_slot_pointer(b, bindless_array, slot_index);
    auto slot_type = _get_llvm_bindless_array_slot_type();
    auto handle_i = dim == 2 ? llvm_bindless_array_slot_type_texture2d_handle_index :
                               llvm_bindless_array_slot_type_texture3d_handle_index;
    auto levels_i = dim == 2 ? llvm_bindless_array_slot_type_texture2d_levels_index :
                               llvm_bindless_array_slot_type_texture3d_levels_index;
    auto &dl = _llvm_module->getDataLayout();
    auto slot_struct_type = llvm::cast<llvm::StructType>(slot_type);
    auto handle_offset = dl.getStructLayout(slot_struct_type)->getElementOffset(handle_i);
    auto levels_offset = dl.getStructLayout(slot_struct_type)->getElementOffset(levels_i);
    auto handle_table_ptr = b.CreateInBoundsGEP(b.getInt8Ty(), slot_ptr, b.getInt64(handle_offset));
    auto level_count_ptr = b.CreateInBoundsGEP(b.getInt8Ty(), slot_ptr, b.getInt64(levels_offset));
    auto handle_table = b.CreateLoad(llvm::Type::getInt64Ty(_llvm_context), handle_table_ptr);
    auto level_count = b.CreateAnd(
        b.CreateLoad(llvm::Type::getInt64Ty(_llvm_context), level_count_ptr),
        b.getInt64(0xffu));
    _create_assertion_with_message(b, b.CreateICmpNE(handle_table, b.getInt64(0)), "Bindless texture slot is empty.");
    _create_assertion_with_message(b, b.CreateICmpUGT(level_count, b.getInt64(0)), "Bindless texture slot has no mip levels.");
    auto selected_level = static_cast<llvm::Value *>(b.getInt64(0));
    if (level != nullptr) {
        selected_level = level->getType()->isFloatingPointTy() ?
                             b.CreateFPToUI(level, b.getInt64Ty()) :
                             b.CreateZExtOrTrunc(level, b.getInt64Ty());
    }
    auto max_level = b.CreateSub(level_count, b.getInt64(1));
    selected_level = b.CreateSelect(b.CreateICmpUGT(selected_level, max_level), max_level, selected_level);
    auto table_ptr = b.CreateIntToPtr(handle_table, llvm::PointerType::get(_llvm_context, amdgpu_address_space_global), "tex.descriptor.table");
    auto descriptor_offset = b.CreateMul(
        selected_level, b.getInt64(sizeof(HIPImageDescriptor)),
        "tex.descriptor.offset", true, true);
    auto descriptor_ptr = b.CreateInBoundsGEP(
        b.getInt8Ty(), table_ptr, descriptor_offset,
        "tex.descriptor.ptr");
    return b.CreatePtrToInt(descriptor_ptr, b.getInt64Ty());
}

llvm::Value *HIPCodegenLLVMImpl::_get_accel_instance_pointer(IB &b, llvm::Value *accel, llvm::Value *instance_index) noexcept {
    auto instances = b.CreateExtractValue(accel, llvm_accel_type_instances_index);
    instance_index = b.CreateZExt(instance_index, b.getInt64Ty(), "", true);
    return b.CreateInBoundsGEP(_get_llvm_accel_instance_type(), instances, instance_index);
}

llvm::Value *HIPCodegenLLVMImpl::_get_accel_instance_motion_frame(
    IB &b, llvm::Value *accel, llvm::Value *instance_index,
    llvm::Value *key_index, AccelMotionMode expected_mode) noexcept {
    auto instance = _get_accel_instance_pointer(
        b, accel, instance_index);
    auto instance_type = _get_llvm_accel_instance_type();
    auto metadata_address = b.CreateLoad(
        b.getInt64Ty(),
        b.CreateStructGEP(
            instance_type, instance,
            llvm_accel_instance_type_motion_data_index));
    _create_assertion_with_message(
        b, b.CreateICmpNE(metadata_address, b.getInt64(0u)),
        "Acceleration-structure instance is not a motion instance.");
    auto metadata = b.CreateIntToPtr(
        metadata_address,
        llvm::PointerType::get(
            _llvm_context, amdgpu_address_space_global));
    auto load_i32 = [&](size_t offset) noexcept {
        auto address = b.CreateInBoundsGEP(
            b.getInt8Ty(), metadata, b.getInt64(offset));
        return b.CreateAlignedLoad(
            b.getInt32Ty(), address, llvm::Align{alignof(uint32_t)});
    };
    auto frames_address_ptr = b.CreateInBoundsGEP(
        b.getInt8Ty(), metadata,
        b.getInt64(offsetof(HIPMotionInstanceDeviceData, frames)));
    auto frames_address = b.CreateAlignedLoad(
        b.getInt64Ty(), frames_address_ptr,
        llvm::Align{alignof(uint64_t)});
    auto keyframe_count = load_i32(
        offsetof(HIPMotionInstanceDeviceData, keyframe_count));
    auto frame_stride = load_i32(
        offsetof(HIPMotionInstanceDeviceData, frame_stride));
    auto mode = load_i32(
        offsetof(HIPMotionInstanceDeviceData, mode));
    _create_assertion_with_message(
        b, b.CreateICmpEQ(
               mode, b.getInt32(static_cast<uint32_t>(expected_mode))),
        expected_mode == AccelMotionMode::MATRIX ?
            "Motion instance does not contain matrix keyframes." :
            "Motion instance does not contain SRT keyframes.");
    key_index = b.CreateZExtOrTrunc(key_index, b.getInt32Ty());
    _create_assertion_with_message(
        b, b.CreateICmpULT(key_index, keyframe_count),
        "Motion-instance keyframe index is out of bounds.");
    auto offset = b.CreateMul(
        b.CreateZExt(key_index, b.getInt64Ty()),
        b.CreateZExt(frame_stride, b.getInt64Ty()),
        "motion.frame.offset", true, true);
    auto frames = b.CreateIntToPtr(
        frames_address,
        llvm::PointerType::get(
            _llvm_context, amdgpu_address_space_global));
    return b.CreateInBoundsGEP(
        b.getInt8Ty(), frames, offset, "motion.frame");
}

llvm::Value *HIPCodegenLLVMImpl::_load_accel_affine_matrix(IB &b, llvm::Value *affine_ptr) noexcept {
    auto llvm_f32_type = b.getFloatTy();
    auto llvm_f32x4_type = llvm::VectorType::get(llvm_f32_type, 4, false);
    auto llvm_align = llvm::Align{alignof(float4)};
    auto llvm_a0 = b.CreateAlignedLoad(llvm_f32x4_type, b.CreateInBoundsGEP(llvm_f32x4_type, affine_ptr, b.getInt64(0)), llvm_align);
    auto llvm_a1 = b.CreateAlignedLoad(llvm_f32x4_type, b.CreateInBoundsGEP(llvm_f32x4_type, affine_ptr, b.getInt64(1)), llvm_align);
    auto llvm_a2 = b.CreateAlignedLoad(llvm_f32x4_type, b.CreateInBoundsGEP(llvm_f32x4_type, affine_ptr, b.getInt64(2)), llvm_align);
    auto llvm_one = llvm::ConstantFP::get(llvm_f32_type, 1.);
    auto llvm_a3 = b.CreateInsertElement(llvm::Constant::getNullValue(llvm_f32x4_type), llvm_one, b.getInt64(3));
    auto llvm_transform = static_cast<llvm::Value *>(llvm::PoisonValue::get(_get_llvm_type(Type::of<float4x4>())->reg_type));
    llvm_transform = b.CreateInsertValue(llvm_transform, llvm_a0, 0);
    llvm_transform = b.CreateInsertValue(llvm_transform, llvm_a1, 1);
    llvm_transform = b.CreateInsertValue(llvm_transform, llvm_a2, 2);
    llvm_transform = b.CreateInsertValue(llvm_transform, llvm_a3, 3);
    return _translate_matrix_transpose(b, llvm_transform);
}

void HIPCodegenLLVMImpl::_store_accel_affine_matrix(IB &b, llvm::Value *affine_ptr, llvm::Value *matrix) noexcept {
    auto llvm_transform = _translate_matrix_transpose(b, matrix);
    auto llvm_a0 = b.CreateExtractValue(llvm_transform, 0);
    auto llvm_a1 = b.CreateExtractValue(llvm_transform, 1);
    auto llvm_a2 = b.CreateExtractValue(llvm_transform, 2);
    auto llvm_align = llvm::Align{alignof(float4)};
    b.CreateAlignedStore(llvm_a0, b.CreateInBoundsGEP(llvm_a0->getType(), affine_ptr, b.getInt64(0)), llvm_align);
    b.CreateAlignedStore(llvm_a1, b.CreateInBoundsGEP(llvm_a1->getType(), affine_ptr, b.getInt64(1)), llvm_align);
    b.CreateAlignedStore(llvm_a2, b.CreateInBoundsGEP(llvm_a2->getType(), affine_ptr, b.getInt64(2)), llvm_align);
}

void HIPCodegenLLVMImpl::_set_accel_instance_opacity(IB &b, llvm::Value *accel, llvm::Value *instance_index, llvm::Value *is_opaque) noexcept {
    LUISA_DEBUG_ASSERT(is_opaque->getType()->isIntegerTy(1));
    auto instance_ptr = _get_accel_instance_pointer(b, accel, instance_index);
    auto instances = b.CreateExtractValue(
        accel, llvm_accel_type_instances_index);
    auto metadata_ptr = b.CreateInBoundsGEP(
        b.getInt8Ty(), instances,
        b.getInt64(-static_cast<int64_t>(llvm_accel_metadata_size)),
        "accel.metadata");
    using namespace std::string_view_literals;
    auto name = "luisa.accel.set.instance.opacity"sv;
    auto f = _llvm_module->getFunction(name);
    if (f == nullptr) {
        auto void_type = llvm::Type::getVoidTy(_llvm_context);
        auto f_type = llvm::FunctionType::get(
            void_type,
            {instance_ptr->getType(), metadata_ptr->getType(),
             is_opaque->getType()}, false);
        f = llvm::Function::Create(f_type, llvm::Function::PrivateLinkage, name, *_llvm_module);
        auto entry = llvm::BasicBlock::Create(_llvm_context, "entry", f);
        IB fb{entry};
        // The certificate is monotone. An atomic OR linearizes concurrent
        // transitions to opaque; transitions to non-opaque deliberately leave
        // it set. Codegen rejects the native effect-only route for any module
        // that writes opacity, while later kernels still observe this durable
        // proof invalidation.
        auto mark_opaque = llvm::BasicBlock::Create(
            _llvm_context, "mark.opaque", f);
        auto update_instance = llvm::BasicBlock::Create(
            _llvm_context, "update.instance", f);
        fb.CreateCondBr(f->getArg(2), mark_opaque, update_instance);
        fb.SetInsertPoint(mark_opaque);
        auto certificate_ptr = fb.CreateInBoundsGEP(
            fb.getInt8Ty(), f->getArg(1),
            fb.getInt64(
                llvm_accel_metadata_opacity_may_be_present_offset));
        auto certificate = fb.CreateAtomicRMW(
            llvm::AtomicRMWInst::Or, certificate_ptr,
            fb.getInt32(1u), llvm::MaybeAlign{alignof(uint32_t)},
            llvm::AtomicOrdering::Monotonic);
        certificate->setSyncScopeID(
            _llvm_context.getOrInsertSyncScopeID("agent"));
        fb.CreateBr(update_instance);

        fb.SetInsertPoint(update_instance);
        auto flags_ptr = fb.CreateStructGEP(_get_llvm_accel_instance_type(), f->getArg(0), llvm_accel_instance_type_flags_index);
        auto flags = fb.CreateLoad(fb.getInt32Ty(), flags_ptr);
        // HIP's codegen-visible instance ABI reserves bit 0 for opacity for
        // every geometry kind. Do not use the unrelated OptiX instance bits:
        // HIP ray-query traversal reads this compact flag word directly.
        constexpr auto instance_flag_opaque = 1u << 0u;
        auto cleared_flags = fb.CreateAnd(flags, ~instance_flag_opaque);
        auto new_flag_bit = fb.CreateSelect(f->getArg(2),
                                            fb.getInt32(instance_flag_opaque),
                                            fb.getInt32(0u));
        fb.CreateStore(fb.CreateOr(cleared_flags, new_flag_bit), flags_ptr);
        // Mirror opacity into the private high bit of the packed visibility
        // field. HIPAccel copies that field into HIPRT's instance node on the
        // next build/refit; public visibility reads and writes mask it out.
        auto mask_ptr = fb.CreateStructGEP(
            _get_llvm_accel_instance_type(), f->getArg(0),
            llvm_accel_instance_type_mask_index);
        auto mask = fb.CreateLoad(fb.getInt32Ty(), mask_ptr);
        auto cleared_mask = fb.CreateAnd(
            mask, ~llvm_accel_instance_packed_opacity_bit);
        auto new_packed_opacity = fb.CreateSelect(
            f->getArg(2),
            fb.getInt32(llvm_accel_instance_packed_opacity_bit),
            fb.getInt32(0u));
        fb.CreateStore(
            fb.CreateOr(cleared_mask, new_packed_opacity),
            mask_ptr);
        fb.CreateRetVoid();
    }
    b.CreateCall(f, {instance_ptr, metadata_ptr, is_opaque});
}

llvm::Value *HIPCodegenLLVMImpl::_accel_trace_closest(
    IB &b, const FunctionContext &func_ctx, llvm::Value *accel,
    llvm::Value *ray, llvm::Value *time, llvm::Value *mask) noexcept {
    auto handle = b.CreateExtractValue(accel, llvm_accel_type_handle_index);
    auto instances = b.CreateExtractValue(accel, llvm_accel_type_instances_index);
    auto ox = b.CreateExtractValue(ray, {llvm_ray_type_origin_index, 0});
    auto oy = b.CreateExtractValue(ray, {llvm_ray_type_origin_index, 1});
    auto oz = b.CreateExtractValue(ray, {llvm_ray_type_origin_index, 2});
    auto dx = b.CreateExtractValue(ray, {llvm_ray_type_direction_index, 0});
    auto dy = b.CreateExtractValue(ray, {llvm_ray_type_direction_index, 1});
    auto dz = b.CreateExtractValue(ray, {llvm_ray_type_direction_index, 2});
    auto tmin = b.CreateExtractValue(ray, llvm_ray_type_t_min_index);
    auto tmax = b.CreateExtractValue(ray, llvm_ray_type_t_max_index);
    mask = b.CreateAnd(b.CreateZExtOrTrunc(mask, b.getInt32Ty()), 0xffu);

    auto is_motion_blur = time != nullptr;
    LUISA_DEBUG_ASSERT(!is_motion_blur || time->getType()->isFloatTy());
    auto use_hwstack = _uses_hardware_rt_stack && !is_motion_blur;
    using namespace std::string_view_literals;
    auto wrapper_name = is_motion_blur ?
                            "luisa_hiprt_trace_closest_motion_blur"sv :
                        use_hwstack ? "luisa_hiprt_trace_closest_hwstack"sv :
                                      "luisa_hiprt_trace_closest"sv;
    auto wrapper_f = _llvm_module->getFunction(wrapper_name);
    if (wrapper_f == nullptr) {
        auto void_type = llvm::Type::getVoidTy(_llvm_context);
        auto f32 = b.getFloatTy();
        auto i32 = b.getInt32Ty();
        auto i64 = b.getInt64Ty();
        auto generic_ptr = b.getPtrTy(0);
        llvm::FunctionType *f_type;
        if (is_motion_blur) {
            f_type = llvm::FunctionType::get(void_type,
                                             {i64, generic_ptr,
                                              f32, f32, f32, f32, f32, f32, f32, f32, f32,
                                              i32,
                                              generic_ptr, generic_ptr, generic_ptr, generic_ptr, generic_ptr},
                                             false);
        } else if (use_hwstack) {
            f_type = llvm::FunctionType::get(void_type,
                                             {i64, generic_ptr, f32, f32, f32, f32, f32, f32, f32, f32, i32,
                                              generic_ptr, generic_ptr, generic_ptr, generic_ptr, generic_ptr},
                                             false);
        } else {
            f_type = llvm::FunctionType::get(void_type,
                                             {i64, generic_ptr, f32, f32, f32, f32, f32, f32, f32, f32, i32,
                                              i32, i32, generic_ptr,
                                              generic_ptr, generic_ptr, generic_ptr, generic_ptr, generic_ptr},
                                             false);
        }
        wrapper_f = llvm::Function::Create(f_type, llvm::Function::ExternalLinkage, wrapper_name, *_llvm_module);
    }

    auto alloca_inst_id = b.CreateAlloca(b.getInt32Ty());
    auto alloca_prim_id = b.CreateAlloca(b.getInt32Ty());
    auto alloca_u = b.CreateAlloca(b.getFloatTy());
    auto alloca_v = b.CreateAlloca(b.getFloatTy());
    auto alloca_t = b.CreateAlloca(b.getFloatTy());

    auto generic_ptr_type = b.getPtrTy(0);
    auto cast_instances = b.CreateAddrSpaceCast(instances, generic_ptr_type);
    auto cast_inst_id = b.CreateAddrSpaceCast(alloca_inst_id, generic_ptr_type);
    auto cast_prim_id = b.CreateAddrSpaceCast(alloca_prim_id, generic_ptr_type);
    auto cast_u = b.CreateAddrSpaceCast(alloca_u, generic_ptr_type);
    auto cast_v = b.CreateAddrSpaceCast(alloca_v, generic_ptr_type);
    auto cast_t = b.CreateAddrSpaceCast(alloca_t, generic_ptr_type);

    if (is_motion_blur) {
        b.CreateCall(wrapper_f, {handle, cast_instances,
                                 ox, oy, oz, dx, dy, dz, tmin, tmax,
                                 time, mask,
                                 cast_inst_id, cast_prim_id,
                                 cast_u, cast_v, cast_t});
    } else if (use_hwstack) {
        b.CreateCall(wrapper_f, {handle, cast_instances, ox, oy, oz, dx, dy, dz, tmin, tmax, mask,
                                 cast_inst_id, cast_prim_id, cast_u, cast_v, cast_t});
    } else {
        b.CreateCall(wrapper_f, {handle, cast_instances, ox, oy, oz, dx, dy, dz, tmin, tmax, mask,
                                 func_ctx.llvm_rt_stack_size, func_ctx.llvm_rt_stack_count, func_ctx.llvm_rt_stack_data,
                                 cast_inst_id, cast_prim_id, cast_u, cast_v, cast_t});
    }

    auto inst_id = b.CreateLoad(b.getInt32Ty(), alloca_inst_id);
    auto prim_id = b.CreateLoad(b.getInt32Ty(), alloca_prim_id);
    auto u = b.CreateLoad(b.getFloatTy(), alloca_u);
    auto v = b.CreateLoad(b.getFloatTy(), alloca_v);
    auto t = b.CreateLoad(b.getFloatTy(), alloca_t);

    auto bary = _create_llvm_vector(b, {u, v});

    auto result_type = _get_llvm_surface_hit_type();
    auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(result_type));
    result = b.CreateInsertValue(result, inst_id, llvm_surface_hit_type_inst_id_index);
    result = b.CreateInsertValue(result, prim_id, llvm_surface_hit_type_prim_id_index);
    result = b.CreateInsertValue(result, bary, llvm_surface_hit_type_bary_index);
    result = b.CreateInsertValue(result, t, llvm_surface_hit_type_t_index);
    return result;
}

llvm::Value *HIPCodegenLLVMImpl::_accel_trace_any(
    IB &b, const FunctionContext &func_ctx, llvm::Value *accel,
    llvm::Value *ray, llvm::Value *time, llvm::Value *mask) noexcept {
    auto handle = b.CreateExtractValue(accel, llvm_accel_type_handle_index);
    auto instances = b.CreateExtractValue(accel, llvm_accel_type_instances_index);
    auto ox = b.CreateExtractValue(ray, {llvm_ray_type_origin_index, 0});
    auto oy = b.CreateExtractValue(ray, {llvm_ray_type_origin_index, 1});
    auto oz = b.CreateExtractValue(ray, {llvm_ray_type_origin_index, 2});
    auto dx = b.CreateExtractValue(ray, {llvm_ray_type_direction_index, 0});
    auto dy = b.CreateExtractValue(ray, {llvm_ray_type_direction_index, 1});
    auto dz = b.CreateExtractValue(ray, {llvm_ray_type_direction_index, 2});
    auto tmin = b.CreateExtractValue(ray, llvm_ray_type_t_min_index);
    auto tmax = b.CreateExtractValue(ray, llvm_ray_type_t_max_index);
    mask = b.CreateAnd(b.CreateZExtOrTrunc(mask, b.getInt32Ty()), 0xffu);

    auto is_motion_blur = time != nullptr;
    LUISA_DEBUG_ASSERT(!is_motion_blur || time->getType()->isFloatTy());
    auto use_hwstack = _uses_hardware_rt_stack && !is_motion_blur;
    using namespace std::string_view_literals;
    auto wrapper_name = is_motion_blur ?
                            "luisa_hiprt_trace_any_motion_blur"sv :
                        use_hwstack ? "luisa_hiprt_trace_any_hwstack"sv :
                                      "luisa_hiprt_trace_any"sv;
    auto wrapper_f = _llvm_module->getFunction(wrapper_name);
    if (wrapper_f == nullptr) {
        auto f32 = b.getFloatTy();
        auto i32 = b.getInt32Ty();
        auto i64 = b.getInt64Ty();
        auto i1 = b.getInt1Ty();
        auto generic_ptr = b.getPtrTy(0);
        llvm::FunctionType *f_type;
        if (is_motion_blur) {
            f_type = llvm::FunctionType::get(i1,
                                             {i64, generic_ptr,
                                              f32, f32, f32, f32, f32, f32, f32, f32, f32,
                                              i32},
                                             false);
        } else if (use_hwstack) {
            f_type = llvm::FunctionType::get(i1,
                                             {i64, generic_ptr, f32, f32, f32, f32, f32, f32, f32, f32, i32},
                                             false);
        } else {
            f_type = llvm::FunctionType::get(i1,
                                             {i64, generic_ptr, f32, f32, f32, f32, f32, f32, f32, f32, i32,
                                              i32, i32, generic_ptr},
                                             false);
        }
        wrapper_f = llvm::Function::Create(f_type, llvm::Function::ExternalLinkage, wrapper_name, *_llvm_module);
    }

    auto cast_instances = b.CreateAddrSpaceCast(instances, b.getPtrTy(0));
    if (is_motion_blur) {
        return b.CreateCall(wrapper_f, {handle, cast_instances,
                                        ox, oy, oz, dx, dy, dz, tmin, tmax,
                                        time, mask});
    }
    if (use_hwstack) {
        return b.CreateCall(wrapper_f, {handle, cast_instances, ox, oy, oz, dx, dy, dz, tmin, tmax, mask});
    } else {
        return b.CreateCall(wrapper_f, {handle, cast_instances, ox, oy, oz, dx, dy, dz, tmin, tmax, mask,
                                        func_ctx.llvm_rt_stack_size, func_ctx.llvm_rt_stack_count, func_ctx.llvm_rt_stack_data});
    }
}

}// namespace luisa::compute::hip
