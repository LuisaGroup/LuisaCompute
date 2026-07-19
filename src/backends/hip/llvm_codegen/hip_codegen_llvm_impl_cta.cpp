//
// Created by mike on 3/19/26.
//

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

llvm::Value *HIPCodegenLLVMImpl::_translate_thread_group_inst(IB &b, FunctionContext &func_ctx, const xir::ThreadGroupInst *inst) noexcept {

    auto wave_size = _config.wave_size;
    auto mask_type = wave_size == 64 ? static_cast<llvm::Type *>(b.getInt64Ty()) : static_cast<llvm::Type *>(b.getInt32Ty());
    auto mask_zero = llvm::Constant::getNullValue(mask_type);
    auto mask_constant = [&](uint64_t value) noexcept -> llvm::Constant * {
        return wave_size == 64 ? static_cast<llvm::Constant *>(b.getInt64(value)) :
                                 static_cast<llvm::Constant *>(b.getInt32(static_cast<uint32_t>(value)));
    };

    auto shuffle_idx = [&](llvm::Value *value, llvm::Value *src_lane) noexcept -> llvm::Value * {
        // readlane broadcasts one lane selected by a scalar index. Warp
        // shuffles have a different source lane in every thread, so using it
        // here lets the backend scalarize a divergent index and produces
        // incorrect reductions. ds_bpermute is AMDGPU's divergent gather;
        // its index is a byte address within the wave.
        auto byte_index = b.CreateShl(src_lane, b.getInt32(2u));
        return b.CreateIntrinsic(b.getInt32Ty(), llvm::Intrinsic::amdgcn_ds_bpermute,
                                 {byte_index, value});
    };

    auto pack_into_i32_vector = [&](llvm::Value *v) noexcept {
        LUISA_DEBUG_ASSERT(v->getType()->isIntOrIntVectorTy() || v->getType()->isFPOrFPVectorTy());
        auto bitwidth = _data_layout->getTypeSizeInBits(v->getType()).getFixedValue();
        auto n = static_cast<unsigned>((bitwidth + 31u) / 32u);
        auto packed_bitwidth = n * 32u;
        v = b.CreateBitCast(v, b.getIntNTy(static_cast<unsigned>(bitwidth)));
        if (bitwidth < packed_bitwidth) {
            v = b.CreateZExt(v, b.getIntNTy(packed_bitwidth));
        }
        return std::make_pair(b.CreateBitCast(v, llvm::VectorType::get(b.getInt32Ty(), n, false)), n);
    };

    auto unpack_from_i32_vector = [&](llvm::Value *v, llvm::Type *target_type) noexcept {
        LUISA_DEBUG_ASSERT(v->getType()->isIntOrIntVectorTy(32));
        LUISA_DEBUG_ASSERT(target_type->isIntOrIntVectorTy() || target_type->isFPOrFPVectorTy());
        auto bitwidth = _data_layout->getTypeSizeInBits(target_type).getFixedValue();
        auto packed_bitwidth = _data_layout->getTypeSizeInBits(v->getType()).getFixedValue();
        v = b.CreateBitCast(v, b.getIntNTy(static_cast<unsigned>(packed_bitwidth)));
        if (bitwidth < packed_bitwidth) {
            v = b.CreateTrunc(v, b.getIntNTy(static_cast<unsigned>(bitwidth)));
        }
        return b.CreateBitCast(v, target_type);
    };

    auto shuffle_arbitrary_value = [&](auto &&self, llvm::Value *value, llvm::Value *src_lane) noexcept -> llvm::Value * {
        if (auto array_type = llvm::dyn_cast<llvm::ArrayType>(value->getType())) {
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(array_type));
            for (auto i = 0u; i < array_type->getNumElements(); i++) {
                auto elem = b.CreateExtractValue(value, {i});
                auto shuffled_elem = self(self, elem, src_lane);
                result = b.CreateInsertValue(result, shuffled_elem, {i});
            }
            return result;
        }
        auto [packed_value, packed_i32_count] = pack_into_i32_vector(value);
        auto shuffled_packed = static_cast<llvm::Value *>(llvm::PoisonValue::get(packed_value->getType()));
        for (auto i = 0u; i < packed_i32_count; i++) {
            auto elem = b.CreateExtractElement(packed_value, i);
            auto shuffled_elem = shuffle_idx(elem, src_lane);
            shuffled_packed = b.CreateInsertElement(shuffled_packed, shuffled_elem, i);
        }
        return unpack_from_i32_vector(shuffled_packed, value->getType());
    };

    auto popcount_mask = [&](llvm::Value *mask) noexcept -> llvm::Value * {
        auto count = b.CreateUnaryIntrinsic(llvm::Intrinsic::ctpop, mask);
        return wave_size == 64 ? b.CreateTrunc(count, b.getInt32Ty()) : count;
    };

    // Returns the physical lane containing the zero-based active-lane rank.
    // This is a broadword select: each step chooses the lower or upper half of
    // the remaining mask. Callers mask off out-of-range ranks before consuming
    // the shuffled value, so this helper never needs an undefined shift/index.
    auto select_active_lane = [&](llvm::Value *mask, llvm::Value *rank) noexcept -> llvm::Value * {
        auto selected_lane = static_cast<llvm::Value *>(b.getInt32(0));
        auto relative_rank = rank;
        auto remaining_mask = mask;
        for (auto half_width = wave_size / 2u; half_width >= 1u; half_width /= 2u) {
            auto lower_bits = (uint64_t{1} << half_width) - 1u;
            auto lower_mask = b.CreateAnd(remaining_mask, mask_constant(lower_bits));
            auto lower_count = popcount_mask(lower_mask);
            auto select_upper = b.CreateICmpUGE(relative_rank, lower_count);
            relative_rank = b.CreateSelect(select_upper, b.CreateSub(relative_rank, lower_count), relative_rank);
            selected_lane = b.CreateSelect(select_upper,
                                           b.CreateAdd(selected_lane, b.getInt32(half_width)),
                                           selected_lane);
            remaining_mask = b.CreateSelect(select_upper,
                                            b.CreateLShr(remaining_mask, mask_constant(half_width)),
                                            lower_mask);
        }
        return selected_lane;
    };

    auto shuffle_value = [&](llvm::Value *value, llvm::Value *src_lane) noexcept -> llvm::Value * {
        auto shuffled_value = static_cast<llvm::Value *>(nullptr);
        if (auto vt = llvm::dyn_cast<llvm::VectorType>(value->getType())) {
            llvm::SmallVector<llvm::Value *, 8> shuffled_values;
            auto dim = vt->getElementCount().getFixedValue();
            for (auto i = 0u; i < dim; i++) {
                auto elem = b.CreateExtractElement(value, i);
                shuffled_values.emplace_back(shuffle_idx(elem, src_lane));
            }
            shuffled_value = _create_llvm_vector(b, shuffled_values);
        } else {
            shuffled_value = shuffle_idx(value, src_lane);
        }
        return shuffled_value;
    };

    auto reduce_active = [&](llvm::Value *mask, llvm::Value *value, auto binary_op) noexcept -> llvm::Value * {
        LUISA_DEBUG_ASSERT(value->getType()->isIntOrIntVectorTy(32));
        auto prefix_mask = _read_warp_prefix_lane_mask(b, func_ctx);
        auto rank = popcount_mask(b.CreateAnd(mask, prefix_mask));
        auto active_count = popcount_mask(mask);
        for (auto offset = 1u; offset < wave_size; offset *= 2u) {
            auto partner_rank = b.CreateAdd(rank, b.getInt32(offset));
            auto is_group_root = b.CreateICmpEQ(
                b.CreateAnd(rank, b.getInt32(2u * offset - 1u)), b.getInt32(0));
            auto partner_exists = b.CreateICmpULT(partner_rank, active_count);
            auto partner_lane = select_active_lane(mask, partner_rank);
            auto shuffled_value = shuffle_value(value, partner_lane);
            auto combine = b.CreateAnd(is_group_root, partner_exists);
            value = b.CreateSelect(combine, binary_op(value, shuffled_value), value);
        }
        auto first_active_lane = select_active_lane(mask, b.getInt32(0));
        return shuffle_value(value, first_active_lane);
    };

    auto reduce_prefix = [&](llvm::Value *mask, llvm::Value *unit, llvm::Value *value, auto binary_op) noexcept {
        LUISA_DEBUG_ASSERT(value->getType()->isIntOrIntVectorTy(32));
        auto prefix_mask = _read_warp_prefix_lane_mask(b, func_ctx);
        auto rank = popcount_mask(b.CreateAnd(mask, prefix_mask));
        for (auto offset = 1u; offset < wave_size; offset *= 2u) {
            auto has_predecessor = b.CreateICmpUGE(rank, b.getInt32(offset));
            auto predecessor_rank = b.CreateSub(rank, b.getInt32(offset));
            auto predecessor_lane = select_active_lane(mask, predecessor_rank);
            auto shuffled_value = shuffle_value(value, predecessor_lane);
            value = b.CreateSelect(has_predecessor, binary_op(value, shuffled_value), value);
        }
        auto is_first_active = b.CreateICmpEQ(rank, b.getInt32(0));
        auto previous_rank = b.CreateSub(rank, b.getInt32(1));
        auto previous_lane = select_active_lane(mask, previous_rank);
        auto exclusive_value = shuffle_value(value, previous_lane);
        return b.CreateSelect(is_first_active, unit, exclusive_value);
    };

    auto ballot_type = mask_type;
    auto ballot = [&](llvm::Value *pred) noexcept {
        return b.CreateIntrinsic(ballot_type, llvm::Intrinsic::amdgcn_ballot, {pred});
    };

    switch (auto op = inst->op()) {
        case xir::ThreadGroupOp::SHADER_EXECUTION_REORDER: {
            return nullptr;
        }
        case xir::ThreadGroupOp::RASTER_QUAD_DDX: LUISA_ERROR_WITH_LOCATION("RASTER_QUAD_DDX is not supported in HIP backend.");
        case xir::ThreadGroupOp::RASTER_QUAD_DDY: LUISA_ERROR_WITH_LOCATION("RASTER_QUAD_DDY is not supported in HIP backend.");
        case xir::ThreadGroupOp::WARP_IS_FIRST_ACTIVE_LANE: {
            LUISA_DEBUG_ASSERT(inst->type()->is_bool());
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto llvm_first_wide = b.CreateBinaryIntrinsic(llvm::Intrinsic::cttz, llvm_active_mask, b.getInt1(true));
            auto llvm_first = wave_size == 64 ? b.CreateTrunc(llvm_first_wide, b.getInt32Ty()) : llvm_first_wide;
            auto llvm_lane_id = _read_warp_lane_id(b, func_ctx);
            return b.CreateICmpEQ(llvm_lane_id, llvm_first);
        }
        case xir::ThreadGroupOp::WARP_FIRST_ACTIVE_LANE: {
            LUISA_DEBUG_ASSERT(inst->type()->is_int32() || inst->type()->is_uint32());
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto result_wide = b.CreateBinaryIntrinsic(llvm::Intrinsic::cttz, llvm_active_mask, b.getInt1(true));
            return wave_size == 64 ? b.CreateTrunc(result_wide, b.getInt32Ty()) : result_wide;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_ALL_EQUAL: {
            LUISA_DEBUG_ASSERT(inst->type()->is_bool_or_bool_vector());
            LUISA_DEBUG_ASSERT(inst->operand_count() == 1 && inst->operand(0)->type()->is_scalar_or_vector());
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_value_type = llvm_value->getType();
            auto llvm_first_wide = b.CreateBinaryIntrinsic(llvm::Intrinsic::cttz, llvm_active_mask, b.getInt1(true));
            auto llvm_first_active_lane_id = wave_size == 64 ? b.CreateTrunc(llvm_first_wide, b.getInt32Ty()) : llvm_first_wide;
            auto [llvm_packed_value, packed_i32_count] = pack_into_i32_vector(llvm_value);
            auto llvm_packed_value_from_first = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_packed_value->getType()));
            for (auto i = 0; i < packed_i32_count; i++) {
                auto llvm_local_elem = b.CreateExtractElement(llvm_packed_value, i);
                auto llvm_elem_from_first = shuffle_idx(llvm_local_elem, llvm_first_active_lane_id);
                llvm_packed_value_from_first = b.CreateInsertElement(llvm_packed_value_from_first, llvm_elem_from_first, i);
            }
            auto llvm_value_from_first = unpack_from_i32_vector(llvm_packed_value_from_first, llvm_value_type);
            auto llvm_cmp = llvm_value_type->isFPOrFPVectorTy() ?
                                b.CreateFCmpOEQ(llvm_value, llvm_value_from_first) :
                                b.CreateICmpEQ(llvm_value, llvm_value_from_first);
            if (inst->type()->is_bool()) {
                auto llvm_ballot = ballot(llvm_cmp);
                return b.CreateICmpEQ(llvm_ballot, llvm_active_mask);
            }
            auto llvm_result = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_cmp->getType()));
            for (auto i = 0u; i < inst->type()->dimension(); i++) {
                auto llvm_elem = b.CreateExtractElement(llvm_cmp, b.getInt32(i));
                auto llvm_ballot_elem = ballot(llvm_elem);
                auto llvm_elem_voted = b.CreateICmpEQ(llvm_ballot_elem, llvm_active_mask);
                llvm_result = b.CreateInsertElement(llvm_result, llvm_elem_voted, i);
            }
            return llvm_result;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_AND: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_OR: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_XOR: {
            LUISA_DEBUG_ASSERT(inst->operand_count() == 1 && inst->type() == inst->operand(0)->type());
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(0));
            LUISA_DEBUG_ASSERT(llvm_value->getType()->isIntOrIntVectorTy());
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto [llvm_packed_value, packed_i32_count] = pack_into_i32_vector(llvm_value);
            auto llvm_result_packed = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_packed_value->getType()));
            auto handle_one_i32 = [&](llvm::Value *llvm_local_i32) noexcept -> llvm::Value * {
                return reduce_active(llvm_active_mask, llvm_local_i32, [&](auto x, auto y) noexcept {
                    switch (op) {
                        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_AND: return b.CreateAnd(x, y);
                        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_OR: return b.CreateOr(x, y);
                        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_XOR: return b.CreateXor(x, y);
                        default: break;
                    }
                    LUISA_ERROR_WITH_LOCATION("Invalid bitwise warp reduction op.");
                });
            };
            for (auto i = 0; i < packed_i32_count; i++) {
                auto llvm_local_elem = b.CreateExtractElement(llvm_packed_value, i);
                auto llvm_reduced_elem = handle_one_i32(llvm_local_elem);
                llvm_result_packed = b.CreateInsertElement(llvm_result_packed, llvm_reduced_elem, i);
            }
            return unpack_from_i32_vector(llvm_result_packed, llvm_value->getType());
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_COUNT_BITS: {
            LUISA_DEBUG_ASSERT(inst->type()->is_int32() || inst->type()->is_uint32());
            auto llvm_pred = _get_llvm_value(b, func_ctx, inst->operand(0));
            LUISA_DEBUG_ASSERT(llvm_pred->getType()->isIntegerTy(1));
            auto llvm_ballot_val = ballot(llvm_pred);
            auto pop = b.CreateUnaryIntrinsic(llvm::Intrinsic::ctpop, llvm_ballot_val);
            return wave_size == 64 ? b.CreateTrunc(pop, b.getInt32Ty()) : pop;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_MAX: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_MIN: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_PRODUCT: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_SUM: {
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_value_type = llvm_value->getType();
            LUISA_DEBUG_ASSERT(llvm_value_type->isIntOrIntVectorTy() || llvm_value_type->isFPOrFPVectorTy());
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto llvm_packed_value = pack_into_i32_vector(llvm_value).first;
            auto llvm_result_packed = reduce_active(llvm_active_mask, llvm_packed_value, [&](auto x, auto y) noexcept {
                x = unpack_from_i32_vector(x, llvm_value_type);
                y = unpack_from_i32_vector(y, llvm_value_type);
                auto reduced = [&] {
                    switch (op) {
                        case xir::ThreadGroupOp::WARP_ACTIVE_MAX:
                            return inst->type()->is_int_or_int_vector()   ? b.CreateBinaryIntrinsic(llvm::Intrinsic::smax, x, y) :
                                   inst->type()->is_uint_or_uint_vector() ? b.CreateBinaryIntrinsic(llvm::Intrinsic::umax, x, y) :
                                                                            b.CreateBinaryIntrinsic(llvm::Intrinsic::maxnum, x, y);
                        case xir::ThreadGroupOp::WARP_ACTIVE_MIN:
                            return inst->type()->is_int_or_int_vector()   ? b.CreateBinaryIntrinsic(llvm::Intrinsic::smin, x, y) :
                                   inst->type()->is_uint_or_uint_vector() ? b.CreateBinaryIntrinsic(llvm::Intrinsic::umin, x, y) :
                                                                            b.CreateBinaryIntrinsic(llvm::Intrinsic::minnum, x, y);
                        case xir::ThreadGroupOp::WARP_ACTIVE_PRODUCT:
                            return inst->type()->is_int_or_int_vector()   ? b.CreateNSWMul(x, y) :
                                   inst->type()->is_uint_or_uint_vector() ? b.CreateMul(x, y) :
                                                                            b.CreateFMul(x, y);
                        case xir::ThreadGroupOp::WARP_ACTIVE_SUM:
                            return inst->type()->is_int_or_int_vector()   ? b.CreateNSWAdd(x, y) :
                                   inst->type()->is_uint_or_uint_vector() ? b.CreateAdd(x, y) :
                                                                            b.CreateFAdd(x, y);
                        default: break;
                    }
                    LUISA_ERROR_WITH_LOCATION("Invalid warp reduction op.");
                }();
                return pack_into_i32_vector(reduced).first;
            });
            return unpack_from_i32_vector(llvm_result_packed, llvm_value_type);
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_ALL: {
            LUISA_DEBUG_ASSERT(inst->type()->is_bool());
            auto llvm_pred = _get_llvm_value(b, func_ctx, inst->operand(0));
            LUISA_DEBUG_ASSERT(llvm_pred->getType()->isIntegerTy(1));
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto llvm_ballot_val = ballot(llvm_pred);
            return b.CreateICmpEQ(llvm_ballot_val, llvm_active_mask);
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_ANY: {
            LUISA_DEBUG_ASSERT(inst->type()->is_bool());
            auto llvm_pred = _get_llvm_value(b, func_ctx, inst->operand(0));
            LUISA_DEBUG_ASSERT(llvm_pred->getType()->isIntegerTy(1));
            auto llvm_ballot_val = ballot(llvm_pred);
            return b.CreateICmpNE(llvm_ballot_val, mask_zero);
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_MASK: {
            LUISA_DEBUG_ASSERT(inst->type() == Type::of<int4>() || inst->type() == Type::of<uint4>());
            auto llvm_pred = _get_llvm_value(b, func_ctx, inst->operand(0));
            LUISA_DEBUG_ASSERT(llvm_pred->getType()->isIntegerTy(1));
            auto llvm_ballot_val = ballot(llvm_pred);
            auto llvm_zero = llvm::Constant::getNullValue(llvm::VectorType::get(b.getInt32Ty(), 4, false));
            if (wave_size == 64) {
                auto lo = b.CreateTrunc(llvm_ballot_val, b.getInt32Ty());
                auto hi = b.CreateTrunc(b.CreateLShr(llvm_ballot_val, b.getInt64(32)), b.getInt32Ty());
                auto result = b.CreateInsertElement(llvm_zero, lo, b.getInt64(0));
                return b.CreateInsertElement(result, hi, b.getInt64(1));
            }
            return b.CreateInsertElement(llvm_zero, llvm_ballot_val, b.getInt64(0));
        }
        case xir::ThreadGroupOp::WARP_PREFIX_COUNT_BITS: {
            LUISA_DEBUG_ASSERT(inst->type()->is_int32() || inst->type()->is_uint32());
            auto llvm_pred = _get_llvm_value(b, func_ctx, inst->operand(0));
            LUISA_DEBUG_ASSERT(llvm_pred->getType()->isIntegerTy(1));
            auto llvm_ballot_val = ballot(llvm_pred);
            auto llvm_prefix_mask = _read_warp_prefix_lane_mask(b, func_ctx);
            auto llvm_ballot_and_prefix = b.CreateAnd(llvm_ballot_val, llvm_prefix_mask);
            auto pop = b.CreateUnaryIntrinsic(llvm::Intrinsic::ctpop, llvm_ballot_and_prefix);
            return wave_size == 64 ? b.CreateTrunc(pop, b.getInt32Ty()) : pop;
        }
        case xir::ThreadGroupOp::WARP_PREFIX_SUM: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_PREFIX_PRODUCT: {
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_value_type = llvm_value->getType();
            LUISA_DEBUG_ASSERT(llvm_value_type->isIntOrIntVectorTy() || llvm_value_type->isFPOrFPVectorTy());
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto llvm_packed_value = pack_into_i32_vector(llvm_value).first;
            auto llvm_result_packed = static_cast<llvm::Value *>(nullptr);
            if (op == xir::ThreadGroupOp::WARP_PREFIX_SUM) {
                auto llvm_unit = pack_into_i32_vector(llvm::Constant::getNullValue(llvm_value_type)).first;
                llvm_result_packed = reduce_prefix(llvm_active_mask, llvm_unit, llvm_packed_value, [&](auto x, auto y) noexcept {
                    x = unpack_from_i32_vector(x, llvm_value_type);
                    y = unpack_from_i32_vector(y, llvm_value_type);
                    auto result = inst->type()->is_int_or_int_vector()   ? b.CreateNSWAdd(x, y) :
                                  inst->type()->is_uint_or_uint_vector() ? b.CreateAdd(x, y) :
                                                                           b.CreateFAdd(x, y);
                    return pack_into_i32_vector(result).first;
                });
            } else if (op == xir::ThreadGroupOp::WARP_PREFIX_PRODUCT) {
                auto llvm_unit = pack_into_i32_vector(
                                     llvm_value_type->isIntOrIntVectorTy() ?
                                         llvm::ConstantInt::get(llvm_value_type, 1) :
                                         llvm::ConstantFP::get(llvm_value_type, 1.))
                                     .first;
                llvm_result_packed = reduce_prefix(llvm_active_mask, llvm_unit, llvm_packed_value, [&](auto x, auto y) noexcept {
                    x = unpack_from_i32_vector(x, llvm_value_type);
                    y = unpack_from_i32_vector(y, llvm_value_type);
                    auto result = inst->type()->is_int_or_int_vector()   ? b.CreateNSWMul(x, y) :
                                  inst->type()->is_uint_or_uint_vector() ? b.CreateMul(x, y) :
                                                                           b.CreateFMul(x, y);
                    return pack_into_i32_vector(result).first;
                });
            } else {
                LUISA_ERROR_WITH_LOCATION("Invalid warp prefix op.");
            }
            return unpack_from_i32_vector(llvm_result_packed, llvm_value_type);
        }
        case xir::ThreadGroupOp::WARP_READ_LANE: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE: {
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(0));
            llvm::Value *llvm_lane_id;
            if (op == xir::ThreadGroupOp::WARP_READ_LANE) {
                llvm_lane_id = _get_llvm_value(b, func_ctx, inst->operand(1));
            } else {
                auto first_wide = b.CreateBinaryIntrinsic(llvm::Intrinsic::cttz, llvm_active_mask, b.getInt1(true));
                llvm_lane_id = wave_size == 64 ? b.CreateTrunc(first_wide, b.getInt32Ty()) : first_wide;
            }
            LUISA_DEBUG_ASSERT(llvm_lane_id->getType()->isIntegerTy(32));
            return shuffle_arbitrary_value(shuffle_arbitrary_value, llvm_value, llvm_lane_id);
        }
        case xir::ThreadGroupOp::SYNCHRONIZE_BLOCK: {
            return b.CreateIntrinsic(b.getVoidTy(), llvm::Intrinsic::amdgcn_s_barrier, {});
        }
    }
    LUISA_NOT_IMPLEMENTED();
}

}// namespace luisa::compute::hip
