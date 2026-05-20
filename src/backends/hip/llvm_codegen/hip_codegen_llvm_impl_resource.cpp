//
// Created by mike on 3/18/26.
//

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

llvm::Value *HIPCodegenLLVMImpl::_translate_resource_query_inst(IB &b, FunctionContext &func_ctx, const xir::ResourceQueryInst *inst) noexcept {
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
            auto llvm_handle = b.CreateExtractValue(llvm_texture, llvm_texture_type_handle_index);
            auto llvm_width = static_cast<llvm::Value *>(b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_getresinfo_2d, {b.getFloatTy()}, {llvm_handle}));
            auto llvm_height = b.CreateExtractValue(llvm_width, 1);
            llvm_width = b.CreateExtractValue(llvm_width, 0);
            return _create_llvm_vector(b, {llvm_width, llvm_height});
        }
        case xir::ResourceQueryOp::TEXTURE3D_SIZE: {
            LUISA_DEBUG_ASSERT(inst->type() == Type::of<luisa::int3>() || inst->type() == Type::of<luisa::uint3>());
            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_handle = b.CreateExtractValue(llvm_texture, llvm_texture_type_handle_index);
            auto llvm_resinfo = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_getresinfo_3d, {b.getFloatTy()}, {llvm_handle});
            auto llvm_width = b.CreateExtractValue(llvm_resinfo, 0);
            auto llvm_height = b.CreateExtractValue(llvm_resinfo, 1);
            auto llvm_depth = b.CreateExtractValue(llvm_resinfo, 2);
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
            auto llvm_level_count = b.CreateLoad(b.getInt64Ty(), llvm_level_count_ptr);
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
            auto llvm_level_count = b.CreateLoad(b.getInt64Ty(), llvm_level_count_ptr);
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
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE: break;
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL: break;
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD: break;
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL: break;
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE: break;
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL: break;
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD: break;
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL: {
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_level_operand = op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL ?
                                          _get_llvm_value(b, func_ctx, inst->operand(3)) :
                                          nullptr;
            auto llvm_handle = _get_bindless_array_texture_handle(b, llvm_bindless_array, llvm_index, 2, llvm_level_operand);
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_coord_x = b.CreateExtractElement(llvm_coord, b.getInt64(0));
            auto llvm_coord_y = b.CreateExtractElement(llvm_coord, b.getInt64(1));
            // Load texture descriptors from constant address space
            auto llvm_v8i32_type = llvm::FixedVectorType::get(b.getInt32Ty(), 8);
            auto llvm_v4i32_type = llvm::FixedVectorType::get(b.getInt32Ty(), 4);
            auto llvm_f32x4_type = llvm::FixedVectorType::get(b.getFloatTy(), 4);
            auto llvm_const_ptr = b.CreateIntToPtr(llvm_handle, llvm::PointerType::get(_llvm_context, amdgpu_address_space_constant), "tex.ptr");
            auto llvm_rsrc = b.CreateLoad(llvm_v8i32_type, llvm_const_ptr, "tex.rsrc");
            auto llvm_samp_ptr = b.CreateInBoundsGEP(b.getInt8Ty(), llvm_const_ptr, b.getInt64(llvm_texture_object_sampler_offset), "tex.samp.ptr");
            auto llvm_samp = b.CreateLoad(llvm_v4i32_type, llvm_samp_ptr, "tex.samp");
            auto llvm_result = static_cast<llvm::Value *>(nullptr);
            if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE) {
                llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_sample_2d,
                                                {llvm_f32x4_type, b.getFloatTy(), llvm_v8i32_type, llvm_v4i32_type},
                                                {b.getInt32(15), llvm_coord_x, llvm_coord_y, llvm_rsrc, llvm_samp, b.getInt1(false), b.getInt32(0), b.getInt32(0)});
            } else if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL) {
                llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_sample_2d,
                                                {llvm_f32x4_type, b.getFloatTy(), llvm_v8i32_type, llvm_v4i32_type},
                                                {b.getInt32(15), llvm_coord_x, llvm_coord_y, llvm_rsrc, llvm_samp, b.getInt1(false), b.getInt32(0), b.getInt32(0)});
            } else if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD ||
                       op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL) {
                if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL) {
                    LUISA_WARNING_WITH_LOCATION("Level parameter in BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL is ignored in HIP backend.");
                }
                auto llvm_ddx = _get_llvm_value(b, func_ctx, inst->operand(3));
                auto llvm_ddy = _get_llvm_value(b, func_ctx, inst->operand(4));
                auto llvm_ddx_x = b.CreateExtractElement(llvm_ddx, b.getInt64(0));
                auto llvm_ddx_y = b.CreateExtractElement(llvm_ddx, b.getInt64(1));
                auto llvm_ddy_x = b.CreateExtractElement(llvm_ddy, b.getInt64(0));
                auto llvm_ddy_y = b.CreateExtractElement(llvm_ddy, b.getInt64(1));
                llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_sample_d_2d,
                                                {llvm_f32x4_type, b.getFloatTy(), b.getFloatTy(), b.getFloatTy(), b.getFloatTy(), b.getFloatTy(), llvm_v8i32_type, llvm_v4i32_type},
                                                {b.getInt32(15), llvm_coord_x, llvm_coord_y, llvm_ddx_x, llvm_ddx_y, llvm_ddy_x, llvm_ddy_y, llvm_rsrc, llvm_samp, b.getInt1(false), b.getInt32(0), b.getInt32(0)});
            }
            auto llvm_result_x = b.CreateExtractElement(llvm_result, b.getInt64(0));
            auto llvm_result_y = b.CreateExtractElement(llvm_result, b.getInt64(1));
            auto llvm_result_z = b.CreateExtractElement(llvm_result, b.getInt64(2));
            auto llvm_result_w = b.CreateExtractElement(llvm_result, b.getInt64(3));
            auto llvm_value = _create_llvm_vector(b, {llvm_result_x, llvm_result_y, llvm_result_z, llvm_result_w});
            return _safe_fp_cast(b, llvm_value, _get_llvm_type(inst->type())->reg_type);
        }
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL: {
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_level_operand = op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL ?
                                          _get_llvm_value(b, func_ctx, inst->operand(3)) :
                                          nullptr;
            auto llvm_handle = _get_bindless_array_texture_handle(b, llvm_bindless_array, llvm_index, 3, llvm_level_operand);
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_coord_x = b.CreateExtractElement(llvm_coord, b.getInt64(0));
            auto llvm_coord_y = b.CreateExtractElement(llvm_coord, b.getInt64(1));
            auto llvm_coord_z = b.CreateExtractElement(llvm_coord, b.getInt64(2));
            auto llvm_v8i32_type = llvm::FixedVectorType::get(b.getInt32Ty(), 8);
            auto llvm_v4i32_type = llvm::FixedVectorType::get(b.getInt32Ty(), 4);
            auto llvm_f32x4_type = llvm::FixedVectorType::get(b.getFloatTy(), 4);
            auto llvm_const_ptr = b.CreateIntToPtr(llvm_handle, llvm::PointerType::get(_llvm_context, amdgpu_address_space_constant), "tex.ptr");
            auto llvm_rsrc = b.CreateLoad(llvm_v8i32_type, llvm_const_ptr, "tex.rsrc");
            auto llvm_samp_ptr = b.CreateInBoundsGEP(b.getInt8Ty(), llvm_const_ptr, b.getInt64(llvm_texture_object_sampler_offset), "tex.samp.ptr");
            auto llvm_samp = b.CreateLoad(llvm_v4i32_type, llvm_samp_ptr, "tex.samp");
            auto llvm_result = static_cast<llvm::Value *>(nullptr);
            if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE) {
                llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_sample_3d,
                                                {llvm_f32x4_type, b.getFloatTy(), llvm_v8i32_type, llvm_v4i32_type},
                                                {b.getInt32(15), llvm_coord_x, llvm_coord_y, llvm_coord_z, llvm_rsrc, llvm_samp, b.getInt1(false), b.getInt32(0), b.getInt32(0)});
            } else if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL) {
                llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_sample_3d,
                                                {llvm_f32x4_type, b.getFloatTy(), llvm_v8i32_type, llvm_v4i32_type},
                                                {b.getInt32(15), llvm_coord_x, llvm_coord_y, llvm_coord_z, llvm_rsrc, llvm_samp, b.getInt1(false), b.getInt32(0), b.getInt32(0)});
            } else if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD ||
                       op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL) {
                if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL) {
                    LUISA_WARNING_WITH_LOCATION("Level parameter in BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL is ignored in HIP backend.");
                }
                auto llvm_ddx = _get_llvm_value(b, func_ctx, inst->operand(3));
                auto llvm_ddy = _get_llvm_value(b, func_ctx, inst->operand(4));
                auto llvm_ddx_x = b.CreateExtractElement(llvm_ddx, b.getInt64(0));
                auto llvm_ddx_y = b.CreateExtractElement(llvm_ddx, b.getInt64(1));
                auto llvm_ddx_z = b.CreateExtractElement(llvm_ddx, b.getInt64(2));
                auto llvm_ddy_x = b.CreateExtractElement(llvm_ddy, b.getInt64(0));
                auto llvm_ddy_y = b.CreateExtractElement(llvm_ddy, b.getInt64(1));
                auto llvm_ddy_z = b.CreateExtractElement(llvm_ddy, b.getInt64(2));
                llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_sample_d_3d,
                                                {llvm_f32x4_type, b.getFloatTy(), b.getFloatTy(), b.getFloatTy(), b.getFloatTy(), b.getFloatTy(), b.getFloatTy(), llvm_v8i32_type, llvm_v4i32_type},
                                                {b.getInt32(15), llvm_coord_x, llvm_coord_y, llvm_coord_z, llvm_ddx_x, llvm_ddx_y, llvm_ddx_z, llvm_ddy_x, llvm_ddy_y, llvm_ddy_z, llvm_rsrc, llvm_samp, b.getInt1(false), b.getInt32(0), b.getInt32(0)});
            }
            auto llvm_result_x = b.CreateExtractElement(llvm_result, b.getInt64(0));
            auto llvm_result_y = b.CreateExtractElement(llvm_result, b.getInt64(1));
            auto llvm_result_z = b.CreateExtractElement(llvm_result, b.getInt64(2));
            auto llvm_result_w = b.CreateExtractElement(llvm_result, b.getInt64(3));
            auto llvm_value = _create_llvm_vector(b, {llvm_result_x, llvm_result_y, llvm_result_z, llvm_result_w});
            return _safe_fp_cast(b, llvm_value, _get_llvm_type(inst->type())->reg_type);
        }
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER: {
            LUISA_WARNING_WITH_LOCATION("BINDLESS_TEXTURE2D_SAMPLE*_SAMPLER uses stored sampler; custom sampler ignored in HIP backend.");
            // Fall through to non-sampler implementation using stored sampler
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_level_operand = op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER ?
                                          _get_llvm_value(b, func_ctx, inst->operand(3)) :
                                          nullptr;
            auto llvm_handle = _get_bindless_array_texture_handle(b, llvm_bindless_array, llvm_index, 2, llvm_level_operand);
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_coord_x = b.CreateExtractElement(llvm_coord, b.getInt64(0));
            auto llvm_coord_y = b.CreateExtractElement(llvm_coord, b.getInt64(1));
            auto llvm_v8i32_type = llvm::FixedVectorType::get(b.getInt32Ty(), 8);
            auto llvm_v4i32_type = llvm::FixedVectorType::get(b.getInt32Ty(), 4);
            auto llvm_f32x4_type = llvm::FixedVectorType::get(b.getFloatTy(), 4);
            auto llvm_const_ptr = b.CreateIntToPtr(llvm_handle, llvm::PointerType::get(_llvm_context, amdgpu_address_space_constant), "tex.ptr");
            auto llvm_rsrc = b.CreateLoad(llvm_v8i32_type, llvm_const_ptr, "tex.rsrc");
            auto llvm_samp_ptr = b.CreateInBoundsGEP(b.getInt8Ty(), llvm_const_ptr, b.getInt64(llvm_texture_object_sampler_offset), "tex.samp.ptr");
            auto llvm_samp = b.CreateLoad(llvm_v4i32_type, llvm_samp_ptr, "tex.samp");
            auto llvm_result = static_cast<llvm::Value *>(nullptr);
            if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER) {
                llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_sample_2d,
                                                {llvm_f32x4_type, b.getFloatTy(), llvm_v8i32_type, llvm_v4i32_type},
                                                {b.getInt32(15), llvm_coord_x, llvm_coord_y, llvm_rsrc, llvm_samp, b.getInt1(false), b.getInt32(0), b.getInt32(0)});
            } else if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER) {
                llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_sample_2d,
                                                {llvm_f32x4_type, b.getFloatTy(), llvm_v8i32_type, llvm_v4i32_type},
                                                {b.getInt32(15), llvm_coord_x, llvm_coord_y, llvm_rsrc, llvm_samp, b.getInt1(false), b.getInt32(0), b.getInt32(0)});
            } else if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER ||
                       op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER) {
                auto llvm_ddx = _get_llvm_value(b, func_ctx, inst->operand(3));
                auto llvm_ddy = _get_llvm_value(b, func_ctx, inst->operand(4));
                auto llvm_ddx_x = b.CreateExtractElement(llvm_ddx, b.getInt64(0));
                auto llvm_ddx_y = b.CreateExtractElement(llvm_ddx, b.getInt64(1));
                auto llvm_ddy_x = b.CreateExtractElement(llvm_ddy, b.getInt64(0));
                auto llvm_ddy_y = b.CreateExtractElement(llvm_ddy, b.getInt64(1));
                llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_sample_d_2d,
                                                {llvm_f32x4_type, b.getFloatTy(), b.getFloatTy(), b.getFloatTy(), b.getFloatTy(), b.getFloatTy(), llvm_v8i32_type, llvm_v4i32_type},
                                                {b.getInt32(15), llvm_coord_x, llvm_coord_y, llvm_ddx_x, llvm_ddx_y, llvm_ddy_x, llvm_ddy_y, llvm_rsrc, llvm_samp, b.getInt1(false), b.getInt32(0), b.getInt32(0)});
            }
            auto llvm_result_x = b.CreateExtractElement(llvm_result, b.getInt64(0));
            auto llvm_result_y = b.CreateExtractElement(llvm_result, b.getInt64(1));
            auto llvm_result_z = b.CreateExtractElement(llvm_result, b.getInt64(2));
            auto llvm_result_w = b.CreateExtractElement(llvm_result, b.getInt64(3));
            auto llvm_value = _create_llvm_vector(b, {llvm_result_x, llvm_result_y, llvm_result_z, llvm_result_w});
            return _safe_fp_cast(b, llvm_value, _get_llvm_type(inst->type())->reg_type);
        }
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER: {
            LUISA_WARNING_WITH_LOCATION("BINDLESS_TEXTURE3D_SAMPLE*_SAMPLER uses stored sampler; custom sampler ignored in HIP backend.");
            // Fall through to non-sampler implementation using stored sampler
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_level_operand = op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER ?
                                          _get_llvm_value(b, func_ctx, inst->operand(3)) :
                                          nullptr;
            auto llvm_handle = _get_bindless_array_texture_handle(b, llvm_bindless_array, llvm_index, 3, llvm_level_operand);
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_coord_x = b.CreateExtractElement(llvm_coord, b.getInt64(0));
            auto llvm_coord_y = b.CreateExtractElement(llvm_coord, b.getInt64(1));
            auto llvm_coord_z = b.CreateExtractElement(llvm_coord, b.getInt64(2));
            auto llvm_v8i32_type = llvm::FixedVectorType::get(b.getInt32Ty(), 8);
            auto llvm_v4i32_type = llvm::FixedVectorType::get(b.getInt32Ty(), 4);
            auto llvm_f32x4_type = llvm::FixedVectorType::get(b.getFloatTy(), 4);
            auto llvm_const_ptr = b.CreateIntToPtr(llvm_handle, llvm::PointerType::get(_llvm_context, amdgpu_address_space_constant), "tex.ptr");
            auto llvm_rsrc = b.CreateLoad(llvm_v8i32_type, llvm_const_ptr, "tex.rsrc");
            auto llvm_samp_ptr = b.CreateInBoundsGEP(b.getInt8Ty(), llvm_const_ptr, b.getInt64(llvm_texture_object_sampler_offset), "tex.samp.ptr");
            auto llvm_samp = b.CreateLoad(llvm_v4i32_type, llvm_samp_ptr, "tex.samp");
            auto llvm_result = static_cast<llvm::Value *>(nullptr);
            if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER) {
                llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_sample_3d,
                                                {llvm_f32x4_type, b.getFloatTy(), llvm_v8i32_type, llvm_v4i32_type},
                                                {b.getInt32(15), llvm_coord_x, llvm_coord_y, llvm_coord_z, llvm_rsrc, llvm_samp, b.getInt1(false), b.getInt32(0), b.getInt32(0)});
            } else if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER) {
                llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_sample_3d,
                                                {llvm_f32x4_type, b.getFloatTy(), llvm_v8i32_type, llvm_v4i32_type},
                                                {b.getInt32(15), llvm_coord_x, llvm_coord_y, llvm_coord_z, llvm_rsrc, llvm_samp, b.getInt1(false), b.getInt32(0), b.getInt32(0)});
            } else if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER ||
                       op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER) {
                auto llvm_ddx = _get_llvm_value(b, func_ctx, inst->operand(3));
                auto llvm_ddy = _get_llvm_value(b, func_ctx, inst->operand(4));
                auto llvm_ddx_x = b.CreateExtractElement(llvm_ddx, b.getInt64(0));
                auto llvm_ddx_y = b.CreateExtractElement(llvm_ddx, b.getInt64(1));
                auto llvm_ddx_z = b.CreateExtractElement(llvm_ddx, b.getInt64(2));
                auto llvm_ddy_x = b.CreateExtractElement(llvm_ddy, b.getInt64(0));
                auto llvm_ddy_y = b.CreateExtractElement(llvm_ddy, b.getInt64(1));
                auto llvm_ddy_z = b.CreateExtractElement(llvm_ddy, b.getInt64(2));
                llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_sample_d_3d,
                                                {llvm_f32x4_type, b.getFloatTy(), b.getFloatTy(), b.getFloatTy(), b.getFloatTy(), b.getFloatTy(), b.getFloatTy(), llvm_v8i32_type, llvm_v4i32_type},
                                                {b.getInt32(15), llvm_coord_x, llvm_coord_y, llvm_coord_z, llvm_ddx_x, llvm_ddx_y, llvm_ddx_z, llvm_ddy_x, llvm_ddy_y, llvm_ddy_z, llvm_rsrc, llvm_samp, b.getInt1(false), b.getInt32(0), b.getInt32(0)});
            }
            auto llvm_result_x = b.CreateExtractElement(llvm_result, b.getInt64(0));
            auto llvm_result_y = b.CreateExtractElement(llvm_result, b.getInt64(1));
            auto llvm_result_z = b.CreateExtractElement(llvm_result, b.getInt64(2));
            auto llvm_result_w = b.CreateExtractElement(llvm_result, b.getInt64(3));
            auto llvm_value = _create_llvm_vector(b, {llvm_result_x, llvm_result_y, llvm_result_z, llvm_result_w});
            return _safe_fp_cast(b, llvm_value, _get_llvm_type(inst->type())->reg_type);
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
            return b.CreateZExtOrTrunc(llvm_mask, llvm_result_type);
        }
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST: {
            auto llvm_accel = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_ray = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_mask = _get_llvm_value(b, func_ctx, inst->operand(2));
            return _accel_trace_closest(b, func_ctx, llvm_accel, llvm_ray, llvm_mask);
        }
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY: {
            auto llvm_accel = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_ray = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_mask = _get_llvm_value(b, func_ctx, inst->operand(2));
            return _accel_trace_any(b, func_ctx, llvm_accel, llvm_ray, llvm_mask);
        }
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL: [[fallthrough]];
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY: {
            auto is_any = (op == xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY);
            auto llvm_accel = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_ray = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_mask = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_accel_handle = b.CreateExtractValue(llvm_accel, llvm_accel_type_handle_index);
            auto llvm_ray_origin = b.CreateExtractValue(llvm_ray, llvm_ray_type_origin_index);
            auto llvm_ray_t_min = b.CreateExtractValue(llvm_ray, llvm_ray_type_t_min_index);
            auto llvm_ray_direction = b.CreateExtractValue(llvm_ray, llvm_ray_type_direction_index);
            auto llvm_ray_t_max = b.CreateExtractValue(llvm_ray, llvm_ray_type_t_max_index);
            auto llvm_ox = b.CreateExtractValue(llvm_ray, {llvm_ray_type_origin_index, 0u});
            auto llvm_oy = b.CreateExtractValue(llvm_ray, {llvm_ray_type_origin_index, 1u});
            auto llvm_oz = b.CreateExtractValue(llvm_ray, {llvm_ray_type_origin_index, 2u});
            auto llvm_dx = b.CreateExtractValue(llvm_ray, {llvm_ray_type_direction_index, 0u});
            auto llvm_dy = b.CreateExtractValue(llvm_ray, {llvm_ray_type_direction_index, 1u});
            auto llvm_dz = b.CreateExtractValue(llvm_ray, {llvm_ray_type_direction_index, 2u});
            // flags: 1 = terminate-on-first-hit (for "any" queries)
            auto llvm_flags = b.getInt32(is_any ? 1u : 0u);
            _call_ray_query_intrinsic(b, func_ctx, llvm_ray_query_intrinsic_name_initialize, b.getVoidTy(),
                                      {llvm_accel_handle,
                                       llvm_ox, llvm_oy, llvm_oz,
                                       llvm_dx, llvm_dy, llvm_dz,
                                       llvm_ray_t_min, llvm_ray_t_max,
                                       llvm_mask, llvm_flags,
                                       func_ctx.llvm_rt_stack_size,
                                       func_ctx.llvm_rt_stack_count,
                                       func_ctx.llvm_rt_stack_data});
            return llvm::Constant::getNullValue(_get_llvm_ray_query_type());
        }
        default: LUISA_NOT_IMPLEMENTED();
    }
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_translate_resource_read_inst(IB &b, const FunctionContext &func_ctx, const xir::ResourceReadInst *inst) noexcept {
    switch (auto op = inst->op()) {
        case xir::ResourceReadOp::BUFFER_READ: [[fallthrough]];
        case xir::ResourceReadOp::BUFFER_VOLATILE_READ: {
            auto llvm_buffer = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            LUISA_DEBUG_ASSERT(inst->type() == inst->operand(0)->type()->element());
            auto elem_type = inst->type();
            auto llvm_elem_ptr = _get_buffer_element_pointer(b, llvm_buffer, llvm_index, elem_type->size(), elem_type->size());
            return _load_llvm_value(b, llvm_elem_ptr, elem_type);
        }
        case xir::ResourceReadOp::BYTE_BUFFER_READ: [[fallthrough]];
        case xir::ResourceReadOp::BYTE_BUFFER_VOLATILE_READ: {
            auto llvm_buffer = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_byte_offset = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto elem_type = inst->type();
            auto llvm_elem_ptr = _get_buffer_element_pointer(b, llvm_buffer, llvm_byte_offset, 1, elem_type->size());
            return _load_llvm_value(b, llvm_elem_ptr, elem_type);
        }
        case xir::ResourceReadOp::TEXTURE2D_READ: {
            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_texture_handle = b.CreateExtractValue(llvm_texture, llvm_texture_type_handle_index);
            auto llvm_texture_storage = b.CreateExtractValue(llvm_texture, llvm_texture_type_storage_index);
            auto llvm_result_type = _get_llvm_type(inst->type())->reg_type;
            LUISA_DEBUG_ASSERT(llvm_result_type->isVectorTy());
            auto llvm_func = _get_texture2d_read_function(llvm::cast<llvm::VectorType>(llvm_result_type));
            return b.CreateCall(llvm_func, {llvm_texture_handle, llvm_texture_storage, llvm_coord});
        }
        case xir::ResourceReadOp::TEXTURE3D_READ: {
            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_texture_handle = b.CreateExtractValue(llvm_texture, llvm_texture_type_handle_index);
            auto llvm_texture_storage = b.CreateExtractValue(llvm_texture, llvm_texture_type_storage_index);
            auto llvm_result_type = _get_llvm_type(inst->type())->reg_type;
            LUISA_DEBUG_ASSERT(llvm_result_type->isVectorTy());
            auto llvm_func = _get_texture3d_read_function(llvm::cast<llvm::VectorType>(llvm_result_type));
            return b.CreateCall(llvm_func, {llvm_texture_handle, llvm_texture_storage, llvm_coord});
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
            auto llvm_const_ptr = b.CreateIntToPtr(llvm_handle, llvm::PointerType::get(_llvm_context, amdgpu_address_space_constant), "tex.ptr");
            auto llvm_rsrc = b.CreateLoad(llvm_v8i32_type, llvm_const_ptr, "tex.rsrc");
            auto llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_load_2d,
                                                 {llvm_f32x4_type, b.getInt32Ty(), llvm_v8i32_type},
                                                 {b.getInt32(15), llvm_coord_x, llvm_coord_y, llvm_rsrc, b.getInt32(0), b.getInt32(0)});
            auto llvm_result_x = b.CreateExtractElement(llvm_result, b.getInt64(0));
            auto llvm_result_y = b.CreateExtractElement(llvm_result, b.getInt64(1));
            auto llvm_result_z = b.CreateExtractElement(llvm_result, b.getInt64(2));
            auto llvm_result_w = b.CreateExtractElement(llvm_result, b.getInt64(3));
            auto llvm_value = _create_llvm_vector(b, {llvm_result_x, llvm_result_y, llvm_result_z, llvm_result_w});
            return _safe_fp_cast(b, llvm_value, _get_llvm_type(inst->type())->reg_type);
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
            auto llvm_const_ptr = b.CreateIntToPtr(llvm_handle, llvm::PointerType::get(_llvm_context, amdgpu_address_space_constant), "tex.ptr");
            auto llvm_rsrc = b.CreateLoad(llvm_v8i32_type, llvm_const_ptr, "tex.rsrc");
            auto llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_load_2d,
                                                 {llvm_f32x4_type, b.getInt32Ty(), llvm_v8i32_type},
                                                 {b.getInt32(15), llvm_coord_x, llvm_coord_y, llvm_rsrc, b.getInt32(0), b.getInt32(0)});
            auto llvm_result_x = b.CreateExtractElement(llvm_result, b.getInt64(0));
            auto llvm_result_y = b.CreateExtractElement(llvm_result, b.getInt64(1));
            auto llvm_result_z = b.CreateExtractElement(llvm_result, b.getInt64(2));
            auto llvm_result_w = b.CreateExtractElement(llvm_result, b.getInt64(3));
            auto llvm_value = _create_llvm_vector(b, {llvm_result_x, llvm_result_y, llvm_result_z, llvm_result_w});
            return _safe_fp_cast(b, llvm_value, _get_llvm_type(inst->type())->reg_type);
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
            auto llvm_const_ptr = b.CreateIntToPtr(llvm_handle, llvm::PointerType::get(_llvm_context, amdgpu_address_space_constant), "tex.ptr");
            auto llvm_rsrc = b.CreateLoad(llvm_v8i32_type, llvm_const_ptr, "tex.rsrc");
            auto llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_load_3d,
                                                 {llvm_f32x4_type, b.getInt32Ty(), llvm_v8i32_type},
                                                 {b.getInt32(15), llvm_coord_x, llvm_coord_y, llvm_coord_z, llvm_rsrc, b.getInt32(0), b.getInt32(0)});
            auto llvm_result_x = b.CreateExtractElement(llvm_result, b.getInt64(0));
            auto llvm_result_y = b.CreateExtractElement(llvm_result, b.getInt64(1));
            auto llvm_result_z = b.CreateExtractElement(llvm_result, b.getInt64(2));
            auto llvm_result_w = b.CreateExtractElement(llvm_result, b.getInt64(3));
            auto llvm_value = _create_llvm_vector(b, {llvm_result_x, llvm_result_y, llvm_result_z, llvm_result_w});
            return _safe_fp_cast(b, llvm_value, _get_llvm_type(inst->type())->reg_type);
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
            auto llvm_const_ptr = b.CreateIntToPtr(llvm_handle, llvm::PointerType::get(_llvm_context, amdgpu_address_space_constant), "tex.ptr");
            auto llvm_rsrc = b.CreateLoad(llvm_v8i32_type, llvm_const_ptr, "tex.rsrc");
            auto llvm_result = b.CreateIntrinsic(llvm::Intrinsic::amdgcn_image_load_3d,
                                                 {llvm_f32x4_type, b.getInt32Ty(), llvm_v8i32_type},
                                                 {b.getInt32(15), llvm_coord_x, llvm_coord_y, llvm_coord_z, llvm_rsrc, b.getInt32(0), b.getInt32(0)});
            auto llvm_result_x = b.CreateExtractElement(llvm_result, b.getInt64(0));
            auto llvm_result_y = b.CreateExtractElement(llvm_result, b.getInt64(1));
            auto llvm_result_z = b.CreateExtractElement(llvm_result, b.getInt64(2));
            auto llvm_result_w = b.CreateExtractElement(llvm_result, b.getInt64(3));
            auto llvm_value = _create_llvm_vector(b, {llvm_result_x, llvm_result_y, llvm_result_z, llvm_result_w});
            return _safe_fp_cast(b, llvm_value, _get_llvm_type(inst->type())->reg_type);
        }
        case xir::ResourceReadOp::DEVICE_ADDRESS_READ: {
            auto llvm_address = b.CreateZExt(_get_llvm_value(b, func_ctx, inst->operand(0)), b.getInt64Ty(), "", true);
            auto llvm_ptr = b.CreateIntToPtr(llvm_address, b.getPtrTy());
            return _load_llvm_value(b, llvm_ptr, inst->type());
        }
    }
    LUISA_NOT_IMPLEMENTED();
}

void HIPCodegenLLVMImpl::_translate_resource_write_inst(IB &b, FunctionContext &func_ctx, const xir::ResourceWriteInst *inst) noexcept {
    switch (auto op = inst->op()) {
        case xir::ResourceWriteOp::BUFFER_WRITE: [[fallthrough]];
        case xir::ResourceWriteOp::BUFFER_VOLATILE_WRITE: {
            auto llvm_buffer = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(2));
            LUISA_DEBUG_ASSERT(inst->operand(2)->type() == inst->operand(0)->type()->element());
            auto elem_type = inst->operand(2)->type();
            auto llvm_elem_ptr = _get_buffer_element_pointer(b, llvm_buffer, llvm_index, elem_type->size(), elem_type->size());
            _store_llvm_value(b, llvm_elem_ptr, llvm_value, elem_type);
            return;
        }
        case xir::ResourceWriteOp::BYTE_BUFFER_WRITE: [[fallthrough]];
        case xir::ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE: {
            auto llvm_buffer = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_byte_offset = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto elem_type = inst->operand(2)->type();
            auto llvm_elem_ptr = _get_buffer_element_pointer(b, llvm_buffer, llvm_byte_offset, 1, elem_type->size());
            _store_llvm_value(b, llvm_elem_ptr, llvm_value, elem_type);
            return;
        }
        case xir::ResourceWriteOp::TEXTURE2D_WRITE: {
            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_texture_handle = b.CreateExtractValue(llvm_texture, llvm_texture_type_handle_index);
            auto llvm_texture_storage = b.CreateExtractValue(llvm_texture, llvm_texture_type_storage_index);
            LUISA_DEBUG_ASSERT(llvm_value->getType()->isVectorTy());
            auto llvm_func = _get_texture2d_write_function(llvm::cast<llvm::VectorType>(llvm_value->getType()));
            b.CreateCall(llvm_func, {llvm_texture_handle, llvm_texture_storage, llvm_coord, llvm_value});
            return;
        }
        case xir::ResourceWriteOp::TEXTURE3D_WRITE: {
            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_texture_handle = b.CreateExtractValue(llvm_texture, llvm_texture_type_handle_index);
            auto llvm_texture_storage = b.CreateExtractValue(llvm_texture, llvm_texture_type_storage_index);
            LUISA_DEBUG_ASSERT(llvm_value->getType()->isVectorTy());
            auto llvm_func = _get_texture3d_write_function(llvm::cast<llvm::VectorType>(llvm_value->getType()));
            b.CreateCall(llvm_func, {llvm_texture_handle, llvm_texture_storage, llvm_coord, llvm_value});
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
            auto llvm_mask = b.CreateZExtOrTrunc(_get_llvm_value(b, func_ctx, inst->operand(2)), llvm_mask_type);
            auto llvm_instance_ptr = _get_accel_instance_pointer(b, llvm_accel, llvm_instance_index);
            auto llvm_mask_ptr = b.CreateStructGEP(llvm_instance_type, llvm_instance_ptr, llvm_accel_instance_type_mask_index);
            b.CreateStore(llvm_mask, llvm_mask_ptr);
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
    auto level_count = b.CreateLoad(llvm::Type::getInt64Ty(_llvm_context), level_count_ptr);
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
    auto table_ptr = b.CreateIntToPtr(handle_table, llvm::PointerType::get(_llvm_context, amdgpu_address_space_global), "tex.handle.table");
    auto handle_ptr = b.CreateInBoundsGEP(llvm::Type::getInt64Ty(_llvm_context), table_ptr, selected_level);
    return b.CreateLoad(llvm::Type::getInt64Ty(_llvm_context), handle_ptr);
}

llvm::Value *HIPCodegenLLVMImpl::_get_accel_instance_pointer(IB &b, llvm::Value *accel, llvm::Value *instance_index) noexcept {
    auto instances = b.CreateExtractValue(accel, llvm_accel_type_instances_index);
    instance_index = b.CreateZExt(instance_index, b.getInt64Ty(), "", true);
    return b.CreateInBoundsGEP(_get_llvm_accel_instance_type(), instances, instance_index);
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
    using namespace std::string_view_literals;
    auto name = "luisa.accel.set.instance.opacity"sv;
    auto f = _llvm_module->getFunction(name);
    if (f == nullptr) {
        auto void_type = llvm::Type::getVoidTy(_llvm_context);
        auto f_type = llvm::FunctionType::get(void_type, {instance_ptr->getType(), is_opaque->getType()}, false);
        f = llvm::Function::Create(f_type, llvm::Function::PrivateLinkage, name, *_llvm_module);
        f->addFnAttr(llvm::Attribute::AlwaysInline);
        auto entry = llvm::BasicBlock::Create(_llvm_context, "entry", f);
        IB fb{entry};
        auto flags_ptr = fb.CreateStructGEP(_get_llvm_accel_instance_type(), f->getArg(0), llvm_accel_instance_type_flags_index);
        auto flags = fb.CreateLoad(fb.getInt32Ty(), flags_ptr);
        // Instance flag constants (same as OptiX/CUDA)
        constexpr auto INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING = 1u << 0u;
        constexpr auto INSTANCE_FLAG_DISABLE_ANYHIT = 1u << 2u;
        constexpr auto INSTANCE_FLAG_ENFORCE_ANYHIT = 1u << 3u;
        auto is_triangle = fb.CreateAnd(flags, INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING);
        is_triangle = fb.CreateICmpNE(is_triangle, fb.getInt32(0));
        auto body = llvm::BasicBlock::Create(_llvm_context, "body", f);
        auto exit = llvm::BasicBlock::Create(_llvm_context, "exit", f);
        fb.CreateCondBr(is_triangle, body, exit);
        fb.SetInsertPoint(body);
        auto cleared_flags = fb.CreateAnd(flags, ~(INSTANCE_FLAG_DISABLE_ANYHIT | INSTANCE_FLAG_ENFORCE_ANYHIT));
        auto new_flag_bit = fb.CreateSelect(f->getArg(1),
                                            fb.getInt32(INSTANCE_FLAG_DISABLE_ANYHIT),
                                            fb.getInt32(INSTANCE_FLAG_ENFORCE_ANYHIT));
        fb.CreateStore(fb.CreateOr(cleared_flags, new_flag_bit), flags_ptr);
        fb.CreateBr(exit);
        fb.SetInsertPoint(exit);
        fb.CreateRetVoid();
    }
    b.CreateCall(f, {instance_ptr, is_opaque});
}

llvm::Value *HIPCodegenLLVMImpl::_accel_trace_closest(IB &b, const FunctionContext &func_ctx, llvm::Value *accel, llvm::Value *ray, llvm::Value *mask) noexcept {
    auto handle = b.CreateExtractValue(accel, llvm_accel_type_handle_index);
    auto ox = b.CreateExtractValue(ray, {llvm_ray_type_origin_index, 0});
    auto oy = b.CreateExtractValue(ray, {llvm_ray_type_origin_index, 1});
    auto oz = b.CreateExtractValue(ray, {llvm_ray_type_origin_index, 2});
    auto dx = b.CreateExtractValue(ray, {llvm_ray_type_direction_index, 0});
    auto dy = b.CreateExtractValue(ray, {llvm_ray_type_direction_index, 1});
    auto dz = b.CreateExtractValue(ray, {llvm_ray_type_direction_index, 2});
    auto tmin = b.CreateExtractValue(ray, llvm_ray_type_t_min_index);
    auto tmax = b.CreateExtractValue(ray, llvm_ray_type_t_max_index);
    mask = b.CreateAnd(b.CreateZExtOrTrunc(mask, b.getInt32Ty()), 0xffu);

    auto use_hwstack = _config.amdgpu_arch >= 1200;
    using namespace std::string_view_literals;
    auto wrapper_name = use_hwstack ? "luisa_hiprt_trace_closest_hwstack"sv : "luisa_hiprt_trace_closest"sv;
    auto wrapper_f = _llvm_module->getFunction(wrapper_name);
    if (wrapper_f == nullptr) {
        auto void_type = llvm::Type::getVoidTy(_llvm_context);
        auto f32 = b.getFloatTy();
        auto i32 = b.getInt32Ty();
        auto i64 = b.getInt64Ty();
        auto generic_ptr = b.getPtrTy(0);
        llvm::FunctionType *f_type;
        if (use_hwstack) {
            f_type = llvm::FunctionType::get(void_type,
                                             {i64, f32, f32, f32, f32, f32, f32, f32, f32, i32,
                                              generic_ptr, generic_ptr, generic_ptr, generic_ptr, generic_ptr},
                                             false);
        } else {
            f_type = llvm::FunctionType::get(void_type,
                                             {i64, f32, f32, f32, f32, f32, f32, f32, f32, i32,
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
    auto cast_inst_id = b.CreateAddrSpaceCast(alloca_inst_id, generic_ptr_type);
    auto cast_prim_id = b.CreateAddrSpaceCast(alloca_prim_id, generic_ptr_type);
    auto cast_u = b.CreateAddrSpaceCast(alloca_u, generic_ptr_type);
    auto cast_v = b.CreateAddrSpaceCast(alloca_v, generic_ptr_type);
    auto cast_t = b.CreateAddrSpaceCast(alloca_t, generic_ptr_type);

    if (use_hwstack) {
        b.CreateCall(wrapper_f, {handle, ox, oy, oz, dx, dy, dz, tmin, tmax, mask,
                                 cast_inst_id, cast_prim_id, cast_u, cast_v, cast_t});
    } else {
        b.CreateCall(wrapper_f, {handle, ox, oy, oz, dx, dy, dz, tmin, tmax, mask,
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

llvm::Value *HIPCodegenLLVMImpl::_accel_trace_any(IB &b, const FunctionContext &func_ctx, llvm::Value *accel, llvm::Value *ray, llvm::Value *mask) noexcept {
    auto handle = b.CreateExtractValue(accel, llvm_accel_type_handle_index);
    auto ox = b.CreateExtractValue(ray, {llvm_ray_type_origin_index, 0});
    auto oy = b.CreateExtractValue(ray, {llvm_ray_type_origin_index, 1});
    auto oz = b.CreateExtractValue(ray, {llvm_ray_type_origin_index, 2});
    auto dx = b.CreateExtractValue(ray, {llvm_ray_type_direction_index, 0});
    auto dy = b.CreateExtractValue(ray, {llvm_ray_type_direction_index, 1});
    auto dz = b.CreateExtractValue(ray, {llvm_ray_type_direction_index, 2});
    auto tmin = b.CreateExtractValue(ray, llvm_ray_type_t_min_index);
    auto tmax = b.CreateExtractValue(ray, llvm_ray_type_t_max_index);
    mask = b.CreateAnd(b.CreateZExtOrTrunc(mask, b.getInt32Ty()), 0xffu);

    auto use_hwstack = _config.amdgpu_arch >= 1200;
    using namespace std::string_view_literals;
    auto wrapper_name = use_hwstack ? "luisa_hiprt_trace_any_hwstack"sv : "luisa_hiprt_trace_any"sv;
    auto wrapper_f = _llvm_module->getFunction(wrapper_name);
    if (wrapper_f == nullptr) {
        auto f32 = b.getFloatTy();
        auto i32 = b.getInt32Ty();
        auto i64 = b.getInt64Ty();
        auto i1 = b.getInt1Ty();
        auto generic_ptr = b.getPtrTy(0);
        llvm::FunctionType *f_type;
        if (use_hwstack) {
            f_type = llvm::FunctionType::get(i1,
                                             {i64, f32, f32, f32, f32, f32, f32, f32, f32, i32},
                                             false);
        } else {
            f_type = llvm::FunctionType::get(i1,
                                             {i64, f32, f32, f32, f32, f32, f32, f32, f32, i32,
                                              i32, i32, generic_ptr},
                                             false);
        }
        wrapper_f = llvm::Function::Create(f_type, llvm::Function::ExternalLinkage, wrapper_name, *_llvm_module);
    }

    if (use_hwstack) {
        return b.CreateCall(wrapper_f, {handle, ox, oy, oz, dx, dy, dz, tmin, tmax, mask});
    } else {
        return b.CreateCall(wrapper_f, {handle, ox, oy, oz, dx, dy, dz, tmin, tmax, mask,
                                        func_ctx.llvm_rt_stack_size, func_ctx.llvm_rt_stack_count, func_ctx.llvm_rt_stack_data});
    }
}

}// namespace luisa::compute::hip
