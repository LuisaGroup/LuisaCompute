//
// Created by mike on 11/1/25.
//

#include <luisa/runtime/rtx/hit.h>
#include <luisa/runtime/rtx/motion_transform.h>
#include <luisa/dsl/rtx/ray_query.h>

#include "../optix_api.h"
#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

llvm::Value *CUDACodegenLLVMImpl::_translate_resource_query_inst(IB &b, FunctionContext &func_ctx, const xir::ResourceQueryInst *inst) noexcept {
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
            auto llvm_width = b.CreateIntrinsic(llvm::Intrinsic::nvvm_suq_width, {llvm_handle});
            auto llvm_height = b.CreateIntrinsic(llvm::Intrinsic::nvvm_suq_height, {llvm_handle});
            return _create_llvm_vector(b, {llvm_width, llvm_height});
        }
        case xir::ResourceQueryOp::TEXTURE3D_SIZE: {
            LUISA_DEBUG_ASSERT(inst->type() == Type::of<luisa::int3>() || inst->type() == Type::of<luisa::uint3>());
            auto llvm_texture = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_handle = b.CreateExtractValue(llvm_texture, llvm_texture_type_handle_index);
            auto llvm_width = b.CreateIntrinsic(llvm::Intrinsic::nvvm_suq_width, {llvm_handle});
            auto llvm_height = b.CreateIntrinsic(llvm::Intrinsic::nvvm_suq_height, {llvm_handle});
            auto llvm_depth = b.CreateIntrinsic(llvm::Intrinsic::nvvm_suq_depth, {llvm_handle});
            return _create_llvm_vector(b, {llvm_width, llvm_height, llvm_depth});
        }
        case xir::ResourceQueryOp::BINDLESS_BUFFER_SIZE: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE: {
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_slot_ptr = _get_bindless_array_slot_pointer(b, llvm_bindless_array, llvm_index);
            auto llvm_slot_type = _get_llvm_bindless_array_slot_type();
            auto llvm_buffer_size_ptr = b.CreateStructGEP(llvm_slot_type, llvm_slot_ptr, llvm_bindless_array_slot_type_buffer_size_index);
            auto llvm_buffer_size = static_cast<llvm::Value *>(b.CreateLoad(
                llvm_slot_type->getStructElementType(llvm_bindless_array_slot_type_buffer_size_index), llvm_buffer_size_ptr));
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
            auto llvm_handle = _get_bindless_array_texture_handle(b, llvm_bindless_array, llvm_index, 2);
            auto llvm_width = b.CreateIntrinsic(llvm::Intrinsic::nvvm_suq_width, {llvm_handle});
            auto llvm_height = b.CreateIntrinsic(llvm::Intrinsic::nvvm_suq_height, {llvm_handle});
            auto llvm_size = _create_llvm_vector(b, {llvm_width, llvm_height});
            if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL) {
                auto llvm_level = b.CreateVectorSplat(2, _get_llvm_value(b, func_ctx, inst->operand(2)));
                llvm_size = b.CreateLShr(llvm_size, llvm_level);
            }
            return llvm_size;
        }
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL: {
            LUISA_DEBUG_ASSERT(inst->type() == Type::of<luisa::int3>() || inst->type() == Type::of<luisa::uint3>());
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_handle = _get_bindless_array_texture_handle(b, llvm_bindless_array, llvm_index, 3);
            auto llvm_width = b.CreateIntrinsic(llvm::Intrinsic::nvvm_suq_width, {llvm_handle});
            auto llvm_height = b.CreateIntrinsic(llvm::Intrinsic::nvvm_suq_height, {llvm_handle});
            auto llvm_depth = b.CreateIntrinsic(llvm::Intrinsic::nvvm_suq_depth, {llvm_handle});
            auto llvm_size = _create_llvm_vector(b, {llvm_width, llvm_height, llvm_depth});
            if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL) {
                auto llvm_level = b.CreateVectorSplat(3, _get_llvm_value(b, func_ctx, inst->operand(2)));
                llvm_size = b.CreateLShr(llvm_size, llvm_level);
            }
            return llvm_size;
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
            auto llvm_handle = _get_bindless_array_texture_handle(b, llvm_bindless_array, llvm_index, 2);
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_coord_x = b.CreateExtractElement(llvm_coord, b.getInt64(0));
            auto llvm_coord_y = b.CreateExtractElement(llvm_coord, b.getInt64(1));
            auto llvm_result = static_cast<llvm::Value *>(nullptr);
            if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE) {
                llvm_result = b.CreateIntrinsic(llvm::Intrinsic::nvvm_tex_unified_2d_v4f32_f32,
                                                {llvm_handle, llvm_coord_x, llvm_coord_y});
            } else if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL) {
                auto llvm_level = _get_llvm_value(b, func_ctx, inst->operand(3));
                llvm_result = b.CreateIntrinsic(llvm::Intrinsic::nvvm_tex_unified_2d_level_v4f32_f32,
                                                {llvm_handle, llvm_coord_x, llvm_coord_y, llvm_level});
            } else {
                if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL) {
                    LUISA_WARNING_WITH_LOCATION("Level parameter in BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL is ignored in CUDA backend.");
                }
                auto llvm_ddx = _get_llvm_value(b, func_ctx, inst->operand(3));
                auto llvm_ddy = _get_llvm_value(b, func_ctx, inst->operand(4));
                auto llvm_ddx_x = b.CreateExtractElement(llvm_ddx, b.getInt64(0));
                auto llvm_ddx_y = b.CreateExtractElement(llvm_ddx, b.getInt64(1));
                auto llvm_ddy_x = b.CreateExtractElement(llvm_ddy, b.getInt64(0));
                auto llvm_ddy_y = b.CreateExtractElement(llvm_ddy, b.getInt64(1));
                llvm_result = b.CreateIntrinsic(llvm::Intrinsic::nvvm_tex_unified_2d_grad_v4f32_f32,
                                                {llvm_handle, llvm_coord_x, llvm_coord_y,
                                                 llvm_ddx_x, llvm_ddx_y, llvm_ddy_x, llvm_ddy_y});
            }
            auto llvm_result_x = b.CreateExtractValue(llvm_result, 0);
            auto llvm_result_y = b.CreateExtractValue(llvm_result, 1);
            auto llvm_result_z = b.CreateExtractValue(llvm_result, 2);
            auto llvm_result_w = b.CreateExtractValue(llvm_result, 3);
            auto llvm_value = _create_llvm_vector(b, {llvm_result_x, llvm_result_y, llvm_result_z, llvm_result_w});
            return _safe_fp_cast(b, llvm_value, _get_llvm_type(inst->type())->reg_type);
        }
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL: {
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_handle = _get_bindless_array_texture_handle(b, llvm_bindless_array, llvm_index, 3);
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_coord_x = b.CreateExtractElement(llvm_coord, b.getInt64(0));
            auto llvm_coord_y = b.CreateExtractElement(llvm_coord, b.getInt64(1));
            auto llvm_coord_z = b.CreateExtractElement(llvm_coord, b.getInt64(2));
            auto llvm_result = static_cast<llvm::Value *>(nullptr);
            if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE) {
                llvm_result = b.CreateIntrinsic(llvm::Intrinsic::nvvm_tex_unified_3d_v4f32_f32,
                                                {llvm_handle, llvm_coord_x, llvm_coord_y, llvm_coord_z});
            } else if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL) {
                auto llvm_level = _get_llvm_value(b, func_ctx, inst->operand(3));
                llvm_result = b.CreateIntrinsic(llvm::Intrinsic::nvvm_tex_unified_3d_level_v4f32_f32,
                                                {llvm_handle, llvm_coord_x, llvm_coord_y, llvm_coord_z, llvm_level});
            } else {
                if (op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL) {
                    LUISA_WARNING_WITH_LOCATION("Level parameter in BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL is ignored in CUDA backend.");
                }
                auto llvm_ddx = _get_llvm_value(b, func_ctx, inst->operand(3));
                auto llvm_ddy = _get_llvm_value(b, func_ctx, inst->operand(4));
                auto llvm_ddx_x = b.CreateExtractElement(llvm_ddx, b.getInt64(0));
                auto llvm_ddx_y = b.CreateExtractElement(llvm_ddx, b.getInt64(1));
                auto llvm_ddx_z = b.CreateExtractElement(llvm_ddx, b.getInt64(2));
                auto llvm_ddy_x = b.CreateExtractElement(llvm_ddy, b.getInt64(0));
                auto llvm_ddy_y = b.CreateExtractElement(llvm_ddy, b.getInt64(1));
                auto llvm_ddy_z = b.CreateExtractElement(llvm_ddy, b.getInt64(2));
                llvm_result = b.CreateIntrinsic(llvm::Intrinsic::nvvm_tex_unified_3d_grad_v4f32_f32,
                                                {llvm_handle, llvm_coord_x, llvm_coord_y, llvm_coord_z,
                                                 llvm_ddx_x, llvm_ddx_y, llvm_ddx_z,
                                                 llvm_ddy_x, llvm_ddy_y, llvm_ddy_z});
            }
            auto llvm_result_x = b.CreateExtractValue(llvm_result, 0);
            auto llvm_result_y = b.CreateExtractValue(llvm_result, 1);
            auto llvm_result_z = b.CreateExtractValue(llvm_result, 2);
            auto llvm_result_w = b.CreateExtractValue(llvm_result, 3);
            auto llvm_value = _create_llvm_vector(b, {llvm_result_x, llvm_result_y, llvm_result_z, llvm_result_w});
            return _safe_fp_cast(b, llvm_value, _get_llvm_type(inst->type())->reg_type);
        }
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER: break;
        case xir::ResourceQueryOp::BUFFER_DEVICE_ADDRESS: {
            auto llvm_buffer = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_result_type = _get_llvm_type(inst->type())->reg_type;
            return b.CreatePtrToInt(b.CreateExtractValue(llvm_buffer, llvm_buffer_type_ptr_index), llvm_result_type);
        }
        case xir::ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS: {
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_slot_ptr = _get_bindless_array_slot_pointer(b, llvm_bindless_array, llvm_index);
            auto llvm_slot_type = _get_llvm_bindless_array_slot_type();
            auto llvm_buffer_ptr = b.CreateLoad(llvm_slot_type->getStructElementType(llvm_bindless_array_slot_type_buffer_ptr_index), llvm_slot_ptr);
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
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST: [[fallthrough]];
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY: {
            auto llvm_accel = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_ray = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_mask = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_zero = llvm::ConstantFP::getZero(b.getFloatTy());
            constexpr auto flags_closest = optix::RAY_FLAG_DISABLE_ANYHIT | optix::RAY_FLAG_DISABLE_CLOSESTHIT;
            constexpr auto flags_any = optix::RAY_FLAG_DISABLE_ANYHIT | optix::RAY_FLAG_DISABLE_CLOSESTHIT | optix::RAY_FLAG_TERMINATE_ON_FIRST_HIT;
            return op == xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST ?
                       _accel_trace_closest(b, flags_closest, llvm_accel, llvm_ray, llvm_zero, llvm_mask) :
                       _accel_trace_any(b, flags_any, llvm_accel, llvm_ray, llvm_zero, llvm_mask);
        }
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_MATRIX: {
            LUISA_DEBUG_ASSERT(inst->type() == Type::of<luisa::float4x4>());
            auto llvm_accel = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_instance_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_key_index = _get_llvm_value(b, func_ctx, inst->operand(2));
            struct alignas(16) MatrixTransform {
                float affine[12];
            };
            static_assert(sizeof(MatrixTransform) == 48);
            auto llvm_motion_matrix_ptr = _get_accel_instance_motion_data(b, llvm_accel, llvm_instance_index, llvm_key_index, sizeof(MatrixTransform));
            return _load_accel_affine_matrix(b, llvm_motion_matrix_ptr);
        }
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_SRT: {
            LUISA_DEBUG_ASSERT(inst->type() == Type::of<luisa::compute::MotionInstanceTransformSRT>());
            auto llvm_accel = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_instance_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_key_index = _get_llvm_value(b, func_ctx, inst->operand(2));
            static_assert(sizeof(optix::SRTData) == 64);
            auto llvm_motion_srt_ptr = _get_accel_instance_motion_data(b, llvm_accel, llvm_instance_index, llvm_key_index, sizeof(optix::SRTData));
            b.CreateAlignmentAssumption(*_data_layout, llvm_motion_srt_ptr, 16);
            auto llvm_srt = llvm::PoisonValue::get(_get_llvm_type(inst->type())->reg_type);
            auto llvm_f32_type = b.getFloatTy();
#define LUISA_MAKE_SRT_FIELD_LOAD(field, i0, i1) \
    b.CreateInsertValue(llvm_srt, b.CreateLoad(llvm_f32_type, b.CreateInBoundsPtrAdd(llvm_motion_srt_ptr, b.getInt64(offsetof(optix::SRTData, field)))), {i0, i1});
            LUISA_MAKE_SRT_FIELD_LOAD(pvx, 0, 0);
            LUISA_MAKE_SRT_FIELD_LOAD(pvy, 0, 1);
            LUISA_MAKE_SRT_FIELD_LOAD(pvz, 0, 2);
            LUISA_MAKE_SRT_FIELD_LOAD(qx, 1, 0);
            LUISA_MAKE_SRT_FIELD_LOAD(qy, 1, 1);
            LUISA_MAKE_SRT_FIELD_LOAD(qz, 1, 2);
            LUISA_MAKE_SRT_FIELD_LOAD(qw, 1, 3);
            LUISA_MAKE_SRT_FIELD_LOAD(sx, 2, 0);
            LUISA_MAKE_SRT_FIELD_LOAD(sy, 2, 1);
            LUISA_MAKE_SRT_FIELD_LOAD(sz, 2, 2);
            LUISA_MAKE_SRT_FIELD_LOAD(a, 3, 0);
            LUISA_MAKE_SRT_FIELD_LOAD(b, 3, 1);
            LUISA_MAKE_SRT_FIELD_LOAD(c, 3, 2);
            LUISA_MAKE_SRT_FIELD_LOAD(tx, 4, 0);
            LUISA_MAKE_SRT_FIELD_LOAD(ty, 4, 1);
            LUISA_MAKE_SRT_FIELD_LOAD(tz, 4, 2);
#undef LUISA_MAKE_SRT_FIELD_LOAD
            return llvm_srt;
        }
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR: [[fallthrough]];
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR: {
            auto llvm_accel = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_ray = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_time = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_mask = _get_llvm_value(b, func_ctx, inst->operand(3));
            constexpr auto flags_closest = optix::RAY_FLAG_DISABLE_ANYHIT | optix::RAY_FLAG_DISABLE_CLOSESTHIT;
            constexpr auto flags_any = optix::RAY_FLAG_DISABLE_ANYHIT | optix::RAY_FLAG_DISABLE_CLOSESTHIT | optix::RAY_FLAG_TERMINATE_ON_FIRST_HIT;
            return op == xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR ?
                       _accel_trace_closest(b, flags_closest, llvm_accel, llvm_ray, llvm_time, llvm_mask) :
                       _accel_trace_any(b, flags_any, llvm_accel, llvm_ray, llvm_time, llvm_mask);
        }
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL: [[fallthrough]];
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY: [[fallthrough]];
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR: [[fallthrough]];
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: {
            auto is_any = (op == xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY ||
                           op == xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR);
            LUISA_DEBUG_ASSERT(is_any ? inst->type() == Type::of<RayQueryAny>() : inst->type() == Type::of<RayQueryAll>());
            auto uses_motion_blur = (op == xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR ||
                                     op == xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR);
            auto llvm_accel = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_ray = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_time = uses_motion_blur ? _get_llvm_value(b, func_ctx, inst->operand(2)) :
                                                llvm::ConstantFP::getZero(b.getFloatTy());
            auto llvm_mask = uses_motion_blur ? _get_llvm_value(b, func_ctx, inst->operand(3)) :
                                                _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_flags = is_any ? b.getInt32(optix::RAY_FLAG_DISABLE_CLOSESTHIT | optix::RAY_FLAG_TERMINATE_ON_FIRST_HIT) :
                                       b.getInt32(optix::RAY_FLAG_DISABLE_CLOSESTHIT);
            _call_ray_query_intrinsic(b, llvm_ray_query_intrinsic_name_initialize, b.getVoidTy(),
                                      {llvm_accel, llvm_ray, llvm_time, llvm_mask, llvm_flags});
            return llvm::Constant::getNullValue(_get_llvm_ray_query_type());
        }
    }
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *CUDACodegenLLVMImpl::_translate_resource_read_inst(IB &b, const FunctionContext &func_ctx, const xir::ResourceReadInst *inst) noexcept {
    switch (auto op = inst->op()) {
        case xir::ResourceReadOp::BUFFER_READ: [[fallthrough]];
        case xir::ResourceReadOp::BUFFER_VOLATILE_READ: {
            // create a fence before volatile read if volatile read is requested
            if (op == xir::ResourceReadOp::BUFFER_VOLATILE_READ) { b.CreateIntrinsic(llvm::Intrinsic::nvvm_membar_gl, {}); }
            // load from buffer with element index
            auto llvm_buffer = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            LUISA_DEBUG_ASSERT(inst->type() == inst->operand(0)->type()->element());
            auto elem_type = inst->type();
            auto llvm_elem_ptr = _get_buffer_element_pointer(b, llvm_buffer, llvm_index, elem_type->size(), elem_type->size());
            return _load_llvm_value(b, llvm_elem_ptr, elem_type);
        }
        case xir::ResourceReadOp::BYTE_BUFFER_READ: [[fallthrough]];
        case xir::ResourceReadOp::BYTE_BUFFER_VOLATILE_READ: {
            // create a fence before volatile read if volatile read is requested
            if (op == xir::ResourceReadOp::BYTE_BUFFER_VOLATILE_READ) { b.CreateIntrinsic(llvm::Intrinsic::nvvm_membar_gl, {}); }
            // load from byte buffer with byte offset
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
            auto llvm_buffer_type = _get_llvm_buffer_type();
            LUISA_DEBUG_ASSERT(llvm_buffer_type->getStructNumElements() == 2 &&
                               llvm_buffer_type->getStructElementType(llvm_buffer_type_ptr_index) ==
                                   _get_llvm_bindless_array_slot_type()->getStructElementType(llvm_buffer_type_ptr_index) &&
                               llvm_buffer_type->getStructElementType(llvm_buffer_type_size_index) ==
                                   _get_llvm_bindless_array_slot_type()->getStructElementType(llvm_buffer_type_size_index));
            auto llvm_buffer = b.CreateLoad(llvm_buffer_type, llvm_slot_ptr);
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
            auto llvm_result = b.CreateIntrinsic(llvm::Intrinsic::nvvm_tex_unified_2d_v4f32_s32,
                                                 {llvm_handle, llvm_coord_x, llvm_coord_y});
            auto llvm_result_x = b.CreateExtractValue(llvm_result, 0);
            auto llvm_result_y = b.CreateExtractValue(llvm_result, 1);
            auto llvm_result_z = b.CreateExtractValue(llvm_result, 2);
            auto llvm_result_w = b.CreateExtractValue(llvm_result, 3);
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
            auto llvm_result = b.CreateIntrinsic(llvm::Intrinsic::nvvm_tex_unified_3d_v4f32_s32,
                                                 {llvm_handle, llvm_coord_x, llvm_coord_y, llvm_coord_z});
            auto llvm_result_x = b.CreateExtractValue(llvm_result, 0);
            auto llvm_result_y = b.CreateExtractValue(llvm_result, 1);
            auto llvm_result_z = b.CreateExtractValue(llvm_result, 2);
            auto llvm_result_w = b.CreateExtractValue(llvm_result, 3);
            auto llvm_value = _create_llvm_vector(b, {llvm_result_x, llvm_result_y, llvm_result_z, llvm_result_w});
            return _safe_fp_cast(b, llvm_value, _get_llvm_type(inst->type())->reg_type);
        }
        case xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL: {
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_handle = _get_bindless_array_texture_handle(b, llvm_bindless_array, llvm_index, 2);
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_level = _get_llvm_value(b, func_ctx, inst->operand(3));
            auto llvm_coord_x = b.CreateExtractElement(llvm_coord, b.getInt64(0));
            auto llvm_coord_y = b.CreateExtractElement(llvm_coord, b.getInt64(1));
            auto llvm_asm = _get_inline_asm("tex.level.2d.v4.f32.s32 {$0, $1, $2, $3}, [$4, {$5, $6}], $7;",
                                            "=f,=f,=f,=f,l,r,r,r", false);
            auto llvm_i32_type = b.getInt32Ty();
            auto llvm_result = b.CreateCall(llvm_asm, {llvm_handle,
                                                       b.CreateZExtOrTrunc(llvm_coord_x, llvm_i32_type),
                                                       b.CreateZExtOrTrunc(llvm_coord_y, llvm_i32_type),
                                                       b.CreateZExtOrTrunc(llvm_level, llvm_i32_type)});
            auto llvm_result_x = b.CreateExtractValue(llvm_result, 0);
            auto llvm_result_y = b.CreateExtractValue(llvm_result, 1);
            auto llvm_result_z = b.CreateExtractValue(llvm_result, 2);
            auto llvm_result_w = b.CreateExtractValue(llvm_result, 3);
            auto llvm_value = _create_llvm_vector(b, {llvm_result_x, llvm_result_y, llvm_result_z, llvm_result_w});
            return _safe_fp_cast(b, llvm_value, _get_llvm_type(inst->type())->reg_type);
        }
        case xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL: {
            auto llvm_bindless_array = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_handle = _get_bindless_array_texture_handle(b, llvm_bindless_array, llvm_index, 3);
            auto llvm_coord = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_level = _get_llvm_value(b, func_ctx, inst->operand(3));
            auto llvm_coord_x = b.CreateExtractElement(llvm_coord, b.getInt64(0));
            auto llvm_coord_y = b.CreateExtractElement(llvm_coord, b.getInt64(1));
            auto llvm_coord_z = b.CreateExtractElement(llvm_coord, b.getInt64(2));
            auto llvm_asm = _get_inline_asm("tex.level.3d.v4.f32.s32 {$0, $1, $2, $3}, [$4, {$5, $6, $7, $8}], $9;",
                                            "=f,=f,=f,=f,l,r,r,r,r,r", false);
            auto llvm_i32_type = b.getInt32Ty();
            auto llvm_result = b.CreateCall(llvm_asm, {llvm_handle,
                                                       b.CreateZExtOrTrunc(llvm_coord_x, llvm_i32_type),
                                                       b.CreateZExtOrTrunc(llvm_coord_y, llvm_i32_type),
                                                       b.CreateZExtOrTrunc(llvm_coord_z, llvm_i32_type),
                                                       b.getInt32(0),
                                                       b.CreateZExtOrTrunc(llvm_level, llvm_i32_type)});
            auto llvm_result_x = b.CreateExtractValue(llvm_result, 0);
            auto llvm_result_y = b.CreateExtractValue(llvm_result, 1);
            auto llvm_result_z = b.CreateExtractValue(llvm_result, 2);
            auto llvm_result_w = b.CreateExtractValue(llvm_result, 3);
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

void CUDACodegenLLVMImpl::_translate_resource_write_inst(IB &b, FunctionContext &func_ctx, const xir::ResourceWriteInst *inst) noexcept {
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
            // create a fence after volatile write if volatile write is requested
            if (op == xir::ResourceWriteOp::BUFFER_VOLATILE_WRITE) { b.CreateIntrinsic(llvm::Intrinsic::nvvm_membar_gl, {}); }
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
            // create a fence after volatile write if volatile write is requested
            if (op == xir::ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE) { b.CreateIntrinsic(llvm::Intrinsic::nvvm_membar_gl, {}); }
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
            auto llvm_buffer_type = _get_llvm_buffer_type();
            LUISA_DEBUG_ASSERT(llvm_buffer_type->getStructNumElements() == 2 &&
                               llvm_buffer_type->getStructElementType(llvm_buffer_type_ptr_index) ==
                                   _get_llvm_bindless_array_slot_type()->getStructElementType(llvm_buffer_type_ptr_index) &&
                               llvm_buffer_type->getStructElementType(llvm_buffer_type_size_index) ==
                                   _get_llvm_bindless_array_slot_type()->getStructElementType(llvm_buffer_type_size_index));
            auto llvm_buffer = b.CreateLoad(llvm_buffer_type, llvm_slot_ptr);
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
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX: {
            auto llvm_accel = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_instance_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_key_index = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_transform = _get_llvm_value(b, func_ctx, inst->operand(3));
            struct alignas(16) MatrixTransform {
                float affine[12];
            };
            static_assert(sizeof(MatrixTransform) == 48);
            auto llvm_motion_matrix_ptr = _get_accel_instance_motion_data(b, llvm_accel, llvm_instance_index, llvm_key_index, sizeof(MatrixTransform));
            _store_accel_affine_matrix(b, llvm_motion_matrix_ptr, llvm_transform);
            return;
        }
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT: {
            auto llvm_accel = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_instance_index = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_key_index = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto llvm_srt = _get_llvm_value(b, func_ctx, inst->operand(3));
            static_assert(sizeof(optix::SRTData) == 64);
            auto llvm_motion_srt_ptr = _get_accel_instance_motion_data(b, llvm_accel, llvm_instance_index, llvm_key_index, sizeof(optix::SRTData));
            b.CreateAlignmentAssumption(*_data_layout, llvm_motion_srt_ptr, 16);
            auto llvm_srt_pvx = b.CreateExtractValue(llvm_srt, {0, 0});
            auto llvm_srt_pvy = b.CreateExtractValue(llvm_srt, {0, 1});
            auto llvm_srt_pvz = b.CreateExtractValue(llvm_srt, {0, 2});
            auto llvm_srt_qx = b.CreateExtractValue(llvm_srt, {1, 0});
            auto llvm_srt_qy = b.CreateExtractValue(llvm_srt, {1, 1});
            auto llvm_srt_qz = b.CreateExtractValue(llvm_srt, {1, 2});
            auto llvm_srt_qw = b.CreateExtractValue(llvm_srt, {1, 3});
            auto llvm_srt_sx = b.CreateExtractValue(llvm_srt, {2, 0});
            auto llvm_srt_sy = b.CreateExtractValue(llvm_srt, {2, 1});
            auto llvm_srt_sz = b.CreateExtractValue(llvm_srt, {2, 2});
            auto llvm_srt_a = b.CreateExtractValue(llvm_srt, {3, 0});
            auto llvm_srt_b = b.CreateExtractValue(llvm_srt, {3, 1});
            auto llvm_srt_c = b.CreateExtractValue(llvm_srt, {3, 2});
            auto llvm_srt_tx = b.CreateExtractValue(llvm_srt, {4, 0});
            auto llvm_srt_ty = b.CreateExtractValue(llvm_srt, {4, 1});
            auto llvm_srt_tz = b.CreateExtractValue(llvm_srt, {4, 2});
            // store
#define LUISA_MAKE_SRT_FIELD_STORE(field) \
    b.CreateStore(llvm_srt_##field, b.CreateInBoundsPtrAdd(llvm_motion_srt_ptr, b.getInt64(offsetof(optix::SRTData, field))));
            LUISA_MAKE_SRT_FIELD_STORE(pvx)
            LUISA_MAKE_SRT_FIELD_STORE(pvy)
            LUISA_MAKE_SRT_FIELD_STORE(pvz)
            LUISA_MAKE_SRT_FIELD_STORE(qx)
            LUISA_MAKE_SRT_FIELD_STORE(qy)
            LUISA_MAKE_SRT_FIELD_STORE(qz)
            LUISA_MAKE_SRT_FIELD_STORE(qw)
            LUISA_MAKE_SRT_FIELD_STORE(sx)
            LUISA_MAKE_SRT_FIELD_STORE(sy)
            LUISA_MAKE_SRT_FIELD_STORE(sz)
            LUISA_MAKE_SRT_FIELD_STORE(a)
            LUISA_MAKE_SRT_FIELD_STORE(b)
            LUISA_MAKE_SRT_FIELD_STORE(c)
            LUISA_MAKE_SRT_FIELD_STORE(tx)
            LUISA_MAKE_SRT_FIELD_STORE(ty)
            LUISA_MAKE_SRT_FIELD_STORE(tz)
#undef LUISA_MAKE_SRT_FIELD_STORE
            return;
        }
        case xir::ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL: break;
        case xir::ResourceWriteOp::INDIRECT_DISPATCH_SET_COUNT: break;
    }
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *CUDACodegenLLVMImpl::_get_buffer_element_pointer(IB &b, llvm::Value *buffer, llvm::Value *index, size_t index_stride, size_t element_size) noexcept {
    auto buffer_data_ptr = b.CreateExtractValue(buffer, llvm_buffer_type_ptr_index);
    auto buffer_size_bytes = b.CreateExtractValue(buffer, llvm_buffer_type_size_index);
    auto size_type = buffer_size_bytes->getType();
    LUISA_DEBUG_ASSERT(size_type->isIntegerTy(64));
    index = b.CreateZExt(index, size_type, "", true);
    auto offset_bytes = index_stride == 1 ? index : b.CreateMul(index, b.getInt64(index_stride), "", true, true);
    // check oob if debug mode
    if (_config.enable_debug_info) {
        auto element_end = b.CreateAdd(offset_bytes, b.getInt64(element_size), "", true, true);
        auto inbounds = b.CreateICmpULE(element_end, buffer_size_bytes);
        _create_assertion_with_message(b, inbounds, "Buffer access out of bounds");
    }
    // get element pointer
    return b.CreateInBoundsGEP(b.getInt8Ty(), buffer_data_ptr, offset_bytes);
}

llvm::Value *CUDACodegenLLVMImpl::_get_bindless_array_slot_pointer(IB &b, llvm::Value *bindless_array, llvm::Value *slot_index) noexcept {
    auto slots = b.CreateExtractValue(bindless_array, llvm_bindless_array_type_slots_index);
    auto slot_count = b.CreateExtractValue(bindless_array, llvm_bindless_array_type_size_index);
    slot_index = b.CreateZExt(slot_index, slot_count->getType(), "", true);
    auto slot_index_in_bounds = b.CreateICmpULT(slot_index, slot_count);
    _create_assertion_with_message(b, slot_index_in_bounds, "Bindless array slot index out of bounds.");
    auto slot_type = _get_llvm_bindless_array_slot_type();
    return b.CreateInBoundsGEP(slot_type, slots, slot_index);
}

llvm::Value *CUDACodegenLLVMImpl::_get_bindless_array_texture_handle(IB &b, llvm::Value *bindless_array, llvm::Value *slot_index, int dim) noexcept {
    auto slot_ptr = _get_bindless_array_slot_pointer(b, bindless_array, slot_index);
    auto slot_type = _get_llvm_bindless_array_slot_type();
    auto i = dim == 2 ? llvm_bindless_array_slot_type_texture2d_handle_index :
                        llvm_bindless_array_slot_type_texture3d_handle_index;
    auto handle_ptr = b.CreateStructGEP(slot_type, slot_ptr, i);
    return b.CreateLoad(slot_type->getStructElementType(i), handle_ptr);
}

llvm::Value *CUDACodegenLLVMImpl::_get_accel_instance_pointer(IB &b, llvm::Value *accel, llvm::Value *instance_index) noexcept {
    auto instances = b.CreateExtractValue(accel, llvm_accel_type_instances_index);
    instance_index = b.CreateZExt(instance_index, b.getInt64Ty(), "", true);
    return b.CreateInBoundsGEP(_get_llvm_accel_instance_type(), instances, instance_index);
}

llvm::Value *CUDACodegenLLVMImpl::_get_accel_instance_motion_data(IB &b, llvm::Value *accel, llvm::Value *instance_index, llvm::Value *key_index, size_t stride) noexcept {
    auto instance_ptr = _get_accel_instance_pointer(b, accel, instance_index);
    auto instance_type = _get_llvm_accel_instance_type();
    auto instance_handle_type = instance_type->getStructElementType(llvm_accel_instance_type_handle_index);
    auto instance_handle_ptr = b.CreateStructGEP(instance_type, instance_ptr, llvm_accel_instance_type_handle_index);
    auto instance_handle = b.CreateLoad(instance_handle_type, instance_handle_ptr);
    constexpr auto optix_instance_address_mask = ~0x0full;
    auto instance_address = b.CreateAnd(instance_handle, optix_instance_address_mask);
    auto instance_data = b.CreateIntToPtr(instance_address, b.getPtrTy(nvptx_address_space_global));
    struct alignas(16) MotionDataHeader {
        uint64_t child;
        optix::MotionOptions options;
        uint32_t pad[3];
    };
    static_assert(sizeof(MotionDataHeader) == 32);
    key_index = b.CreateZExt(key_index, b.getInt64Ty(), "", true);
    auto offset = b.CreateAdd(b.getInt64(sizeof(MotionDataHeader)),
                              b.CreateMul(key_index, b.getInt64(stride), "", true, true),
                              "", true, true);
    return b.CreateInBoundsPtrAdd(instance_data, offset);
}

void CUDACodegenLLVMImpl::_set_accel_instance_opacity(IB &b, llvm::Value *accel, llvm::Value *instance_index, llvm::Value *is_opaque) noexcept {
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
        auto is_triangle = fb.CreateAnd(flags, optix::INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING);
        is_triangle = fb.CreateICmpNE(is_triangle, fb.getInt32(0));
        auto body = llvm::BasicBlock::Create(_llvm_context, "body", f);
        auto exit = llvm::BasicBlock::Create(_llvm_context, "exit", f);
        fb.CreateCondBr(is_triangle, body, exit);
        fb.SetInsertPoint(body);
        auto cleared_flags = fb.CreateAnd(flags, ~(optix::INSTANCE_FLAG_DISABLE_ANYHIT | optix::INSTANCE_FLAG_ENFORCE_ANYHIT));
        auto new_flag_bit = fb.CreateSelect(f->getArg(1),
                                            fb.getInt32(optix::INSTANCE_FLAG_DISABLE_ANYHIT),
                                            b.getInt32(optix::INSTANCE_FLAG_ENFORCE_ANYHIT));
        fb.CreateStore(fb.CreateOr(cleared_flags, new_flag_bit), flags_ptr);
        fb.CreateBr(exit);
        fb.SetInsertPoint(exit);
        fb.CreateRetVoid();
    }
    b.CreateCall(f, {instance_ptr, is_opaque});
}

llvm::Value *CUDACodegenLLVMImpl::_load_accel_affine_matrix(IB &b, llvm::Value *affine_ptr) noexcept {
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

void CUDACodegenLLVMImpl::_store_accel_affine_matrix(IB &b, llvm::Value *affine_ptr, llvm::Value *matrix) noexcept {
    auto llvm_transform = _translate_matrix_transpose(b, matrix);
    auto llvm_a0 = b.CreateExtractValue(llvm_transform, 0);
    auto llvm_a1 = b.CreateExtractValue(llvm_transform, 1);
    auto llvm_a2 = b.CreateExtractValue(llvm_transform, 2);
    auto llvm_align = llvm::Align{alignof(float4)};
    b.CreateAlignedStore(llvm_a0, b.CreateInBoundsGEP(llvm_a0->getType(), affine_ptr, b.getInt64(0)), llvm_align);
    b.CreateAlignedStore(llvm_a1, b.CreateInBoundsGEP(llvm_a1->getType(), affine_ptr, b.getInt64(1)), llvm_align);
    b.CreateAlignedStore(llvm_a2, b.CreateInBoundsGEP(llvm_a2->getType(), affine_ptr, b.getInt64(2)), llvm_align);
}

llvm::Value *CUDACodegenLLVMImpl::_accel_trace_closest(IB &b, uint32_t flags, llvm::Value *accel, llvm::Value *ray, llvm::Value *time, llvm::Value *mask) noexcept {
    _call_optix_trace(b, optix::PAYLOAD_TYPE_ID_0, 0u, flags, accel, ray, time, mask, {});
    auto is_hit = _call_optix_hit_object_is_hit(b);
    auto invalid_id = b.getInt32(~0u);
    auto inst_id = b.CreateSelect(is_hit, _call_optix_hit_object_instance_index(b), invalid_id);
    auto prim_id = _call_optix_hit_object_primitive_index(b);
    auto bary = _call_optix_hit_object_triangle_barycentrics(b);
    if (_rt_analysis.curve_basis_set.any()) {// might be a curve
        auto hit_kind = _call_optix_hit_object_hit_kind(b);
        auto is_triangle_front = b.CreateICmpEQ(hit_kind, b.getInt32(optix::HIT_KIND_TRIANGLE_FRONT_FACE));
        auto is_triangle_back = b.CreateICmpEQ(hit_kind, b.getInt32(optix::HIT_KIND_TRIANGLE_BACK_FACE));
        auto is_triangle = b.CreateOr(is_triangle_front, is_triangle_back);
        auto curve_param = b.CreateInsertElement(bary, llvm::ConstantFP::get(b.getFloatTy(), -1.), b.getInt64(1));
        bary = b.CreateSelect(is_triangle, bary, curve_param);
    }
    auto t = _call_optix_hit_object_ray_t_max(b);
    _call_optix_hit_object_reset(b);
    auto result_type = _get_llvm_surface_hit_type();
    auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(result_type));
    result = b.CreateInsertValue(result, inst_id, llvm_surface_hit_type_inst_id_index);
    result = b.CreateInsertValue(result, prim_id, llvm_surface_hit_type_prim_id_index);
    result = b.CreateInsertValue(result, bary, llvm_surface_hit_type_bary_index);
    result = b.CreateInsertValue(result, t, llvm_surface_hit_type_t_index);
    return result;
}

llvm::Value *CUDACodegenLLVMImpl::_accel_trace_any(IB &b, uint32_t flags, llvm::Value *accel, llvm::Value *ray, llvm::Value *time, llvm::Value *mask) noexcept {
    _call_optix_trace(b, optix::PAYLOAD_TYPE_ID_0, 0u, flags, accel, ray, time, mask, {});
    auto is_hit = _call_optix_hit_object_is_hit(b);
    _call_optix_hit_object_reset(b);
    return is_hit;
}

void CUDACodegenLLVMImpl::_call_optix_trace(IB &b, uint32_t payload_type, uint32_t sbt_offset, uint32_t flags,
                                            llvm::Value *accel, llvm::Value *ray, llvm::Value *time, llvm::Value *mask,
                                            llvm::ArrayRef<llvm::Value *> registers) noexcept {
    LUISA_DEBUG_ASSERT(registers.size() <= 2);
    auto handle = b.CreateExtractValue(accel, llvm_accel_type_handle_index);
    auto ox = b.CreateExtractValue(ray, {llvm_ray_type_origin_index, 0});
    auto oy = b.CreateExtractValue(ray, {llvm_ray_type_origin_index, 1});
    auto oz = b.CreateExtractValue(ray, {llvm_ray_type_origin_index, 2});
    auto dx = b.CreateExtractValue(ray, {llvm_ray_type_direction_index, 0});
    auto dy = b.CreateExtractValue(ray, {llvm_ray_type_direction_index, 1});
    auto dz = b.CreateExtractValue(ray, {llvm_ray_type_direction_index, 2});
    auto tmin = b.CreateExtractValue(ray, llvm_ray_type_t_min_index);
    auto tmax = b.CreateExtractValue(ray, llvm_ray_type_t_max_index);
    auto undef = _call_optix_undef(b);
    time = _safe_fp_cast(b, time, b.getFloatTy());
    mask = b.CreateAnd(b.CreateZExtOrTrunc(mask, b.getInt32Ty()), 0xffu);
    auto llvm_asm = _get_inline_asm("{\n"// work around OptiX's trash implementation of ptx2llvm, which stupidly complains that b32 registers cannot be used as f32...
                                    "\t\t.reg .f32 luisa_ray_ox;\n"
                                    "\t\t.reg .f32 luisa_ray_oy;\n"
                                    "\t\t.reg .f32 luisa_ray_oz;\n"
                                    "\t\t.reg .f32 luisa_ray_dx;\n"
                                    "\t\t.reg .f32 luisa_ray_dy;\n"
                                    "\t\t.reg .f32 luisa_ray_dz;\n"
                                    "\t\t.reg .f32 luisa_ray_tmin;\n"
                                    "\t\t.reg .f32 luisa_ray_tmax;\n"
                                    "\t\t.reg .f32 luisa_ray_time;\n"
                                    "\t\tmov.b32 luisa_ray_ox, $34;\n"
                                    "\t\tmov.b32 luisa_ray_oy, $35;\n"
                                    "\t\tmov.b32 luisa_ray_oz, $36;\n"
                                    "\t\tmov.b32 luisa_ray_dx, $37;\n"
                                    "\t\tmov.b32 luisa_ray_dy, $38;\n"
                                    "\t\tmov.b32 luisa_ray_dz, $39;\n"
                                    "\t\tmov.b32 luisa_ray_tmin, $40;\n"
                                    "\t\tmov.b32 luisa_ray_tmax, $41;\n"
                                    "\t\tmov.b32 luisa_ray_time, $42;\n"
                                    "\t\tcall($0,$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30,$31),_optix_hitobject_traverse,"
                                    "($32,$33,luisa_ray_ox,luisa_ray_oy,luisa_ray_oz,luisa_ray_dx,luisa_ray_dy,luisa_ray_dz,luisa_ray_tmin,luisa_ray_tmax,luisa_ray_time,$43,$44,$45,$46,$47,$48,$49,$50,$51,$52,$53,$54,$55,$56,$57,$58,$59,$60,$61,$62,$63,$64,$65,$66,$67,$68,$69,$70,$71,$72,$73,$74,$75,$76,$77,$78,$79,$80);\n"
                                    "\t}",
                                    "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,r,l,f,f,f,f,f,f,f,f,f,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r",
                                    true);
    llvm::SmallVector<llvm::Value *, 80 - 31> args;
    args.emplace_back(b.getInt32(payload_type));
    args.emplace_back(handle);
    args.emplace_back(ox);
    args.emplace_back(oy);
    args.emplace_back(oz);
    args.emplace_back(dx);
    args.emplace_back(dy);
    args.emplace_back(dz);
    args.emplace_back(tmin);
    args.emplace_back(tmax);
    args.emplace_back(time);
    args.emplace_back(mask);
    args.emplace_back(b.getInt32(flags));
    args.emplace_back(b.getInt32(sbt_offset));
    args.emplace_back(b.getInt32(0u));
    args.emplace_back(b.getInt32(0u));
    args.emplace_back(b.getInt32(static_cast<uint32_t>(registers.size())));
    for (auto i = 0; i < 32; i++) {
        if (i < registers.size()) {
            args.emplace_back(registers[i]);
        } else {
            args.emplace_back(undef);
        }
    }
    b.CreateCall(llvm_asm, args);
}

llvm::Value *CUDACodegenLLVMImpl::_call_optix_undef(IB &b) noexcept {
    auto llvm_asm = _get_inline_asm("call ($0), _optix_undef_value, ();", "=r", false);
    return b.CreateCall(llvm_asm, {});
}

void CUDACodegenLLVMImpl::_call_optix_hit_object_reset(IB &b) noexcept {
    auto llvm_asm = _get_inline_asm("call (), _optix_hitobject_make_nop, ();", "", true);
    b.CreateCall(llvm_asm, {});
}

llvm::Value *CUDACodegenLLVMImpl::_call_optix_hit_object_is_hit(IB &b) noexcept {
    auto llvm_asm = _get_inline_asm("call ($0), _optix_hitobject_is_hit, ();", "=r", true);
    auto llvm_result = b.CreateCall(llvm_asm, {});
    return b.CreateTrunc(llvm_result, b.getInt1Ty());
}

llvm::Value *CUDACodegenLLVMImpl::_call_optix_hit_object_triangle_barycentrics(IB &b) noexcept {
    auto llvm_asm = _get_inline_asm("call ($0), _optix_hitobject_get_attribute, ($1);", "=r,r", true);
    auto llvm_u = b.CreateBitCast(b.CreateCall(llvm_asm, {b.getInt32(0)}), b.getFloatTy());
    auto llvm_v = b.CreateBitCast(b.CreateCall(llvm_asm, {b.getInt32(1)}), b.getFloatTy());
    return _create_llvm_vector(b, {llvm_u, llvm_v});
}

llvm::Value *CUDACodegenLLVMImpl::_call_optix_hit_object_curve_parameter(IB &b) noexcept {
    auto llvm_asm = _get_inline_asm("call ($0), _optix_hitobject_get_attribute, ($1);", "=r,r", true);
    return b.CreateBitCast(b.CreateCall(llvm_asm, {b.getInt32(0)}), b.getFloatTy());
}

llvm::Value *CUDACodegenLLVMImpl::_call_optix_hit_object_instance_index(IB &b) noexcept {
    auto llvm_asm = _get_inline_asm("call ($0), _optix_hitobject_get_instance_idx, ();", "=r", true);
    return b.CreateCall(llvm_asm, {});
}

llvm::Value *CUDACodegenLLVMImpl::_call_optix_hit_object_primitive_index(IB &b) noexcept {
    auto llvm_asm = _get_inline_asm("call ($0), _optix_hitobject_get_primitive_idx, ();", "=r", true);
    return b.CreateCall(llvm_asm, {});
}

llvm::Value *CUDACodegenLLVMImpl::_call_optix_hit_object_ray_t_max(IB &b) noexcept {
    auto llvm_asm = _get_inline_asm("call ($0), _optix_hitobject_get_ray_tmax, ();", "=f", true);
    return b.CreateCall(llvm_asm, {});
}

llvm::Value *CUDACodegenLLVMImpl::_call_optix_hit_object_hit_kind(IB &b) noexcept {
    auto llvm_asm = _get_inline_asm("call ($0), _optix_hitobject_get_hitkind, ();", "=r", true);
    return b.CreateCall(llvm_asm, {});
}

llvm::Value *CUDACodegenLLVMImpl::_call_optix_read_instance_index(IB &b) noexcept {
    auto llvm_asm = _get_inline_asm("call ($0), _optix_read_instance_idx, ();", "=r", false);
    return b.CreateCall(llvm_asm, {});
}

llvm::Value *CUDACodegenLLVMImpl::_call_optix_read_primitive_index(IB &b) noexcept {
    auto llvm_asm = _get_inline_asm("call ($0), _optix_read_primitive_idx, ();", "=r", false);
    return b.CreateCall(llvm_asm, {});
}

llvm::Value *CUDACodegenLLVMImpl::_call_optix_get_triangle_barycentrics(IB &b) noexcept {
    auto llvm_asm = _get_inline_asm("call ($0, $1), _optix_get_triangle_barycentrics, ();", "=f,=f", false);
    auto llvm_uv = b.CreateCall(llvm_asm, {});
    auto llvm_u = b.CreateExtractValue(llvm_uv, 0);
    auto llvm_v = b.CreateExtractValue(llvm_uv, 1);
    return _create_llvm_vector(b, {llvm_u, llvm_v});
}

llvm::Value *CUDACodegenLLVMImpl::_call_optix_get_curve_parameter(IB &b) noexcept {
    auto llvm_asm = _get_inline_asm("call ($0), _optix_get_curve_parameter, ();", "=f", false);
    return b.CreateCall(llvm_asm, {});
}

llvm::Value *CUDACodegenLLVMImpl::_call_optix_get_hit_distance(IB &b) noexcept {
    auto llvm_asm = _get_inline_asm("call ($0), _optix_get_ray_tmax, ();", "=f", false);
    return b.CreateCall(llvm_asm, {});
}

llvm::Value *CUDACodegenLLVMImpl::_call_optix_get_hit_kind(IB &b) noexcept {
    auto llvm_asm = _get_inline_asm("call ($0), _optix_get_hit_kind, ();", "=r", false);
    return b.CreateCall(llvm_asm, {});
}

llvm::Value *CUDACodegenLLVMImpl::_call_optix_get_world_space_ray(IB &b) noexcept {
    auto f = [&](std::string_view component) {
        auto asm_str = fmt::format("call ($0), _optix_get_{}, ();", component);
        auto llvm_asm = _get_inline_asm(asm_str, "=f", false);
        return b.CreateCall(llvm_asm, {});
    };
    auto ox = f("world_ray_origin_x");
    auto oy = f("world_ray_origin_y");
    auto oz = f("world_ray_origin_z");
    auto dx = f("world_ray_direction_x");
    auto dy = f("world_ray_direction_y");
    auto dz = f("world_ray_direction_z");
    auto tmin = f("ray_tmin");
    auto tmax = f("ray_tmax");
    auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(_get_llvm_ray_type()));
    result = b.CreateInsertValue(result, ox, {llvm_ray_type_origin_index, 0});
    result = b.CreateInsertValue(result, oy, {llvm_ray_type_origin_index, 1});
    result = b.CreateInsertValue(result, oz, {llvm_ray_type_origin_index, 2});
    result = b.CreateInsertValue(result, dx, {llvm_ray_type_direction_index, 0});
    result = b.CreateInsertValue(result, dy, {llvm_ray_type_direction_index, 1});
    result = b.CreateInsertValue(result, dz, {llvm_ray_type_direction_index, 2});
    result = b.CreateInsertValue(result, tmin, llvm_ray_type_t_min_index);
    result = b.CreateInsertValue(result, tmax, llvm_ray_type_t_max_index);
    return result;
}

void CUDACodegenLLVMImpl::_call_optix_report_intersection(IB &b, llvm::Value *hit_kind, llvm::Value *t) noexcept {
    auto llvm_asm = _get_inline_asm("call ($0), _optix_report_intersection_0, ($1, $2);", "=r,f,r", true);
    b.CreateCall(llvm_asm, {t, hit_kind});
}

void CUDACodegenLLVMImpl::_call_optix_ignore_intersection(IB &b) noexcept {
    auto llvm_asm = _get_inline_asm("call (), _optix_ignore_intersection, ();", "", true);
    b.CreateCall(llvm_asm, {});
}

void CUDACodegenLLVMImpl::_call_optix_terminate_ray(IB &b) noexcept {
    auto llvm_asm = _get_inline_asm("call (), _optix_terminate_ray, ();", "", true);
    b.CreateCall(llvm_asm, {});
}

llvm::Value *CUDACodegenLLVMImpl::_call_optix_get_payload(IB &b, llvm::Value *index) noexcept {
    auto llvm_asm = _get_inline_asm("call ($0), _optix_get_payload, ($1);", "=r,r", false);
    return b.CreateCall(llvm_asm, {index});
}

}// namespace luisa::compute::cuda
