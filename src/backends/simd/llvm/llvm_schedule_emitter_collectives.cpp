#include "llvm_schedule_emitter.h"

#include "../../common/env_flag.h"

namespace luisa::compute::simd::detail {

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_collective(
    const schedule::Instruction &instruction) {
    if (!instruction.result || !instruction.source_op ||
        !instruction.participant_mask) {
        _fail("warp collective is missing result, operation, or participant mask");
        return nullptr;
    }
    auto *participants = _load_value(*instruction.participant_mask);
    auto *result_value = _source.value(*instruction.result);
    if (participants == nullptr) { return nullptr; }
    std::vector<::llvm::Value *> operands;
    std::vector<const schedule::Value *> operand_values;
    operands.reserve(instruction.operands.size());
    operand_values.reserve(instruction.operands.size());
    for (auto operand_id : instruction.operands) {
        auto *operand = _source.value(operand_id);
        if (operand == nullptr) {
            _fail("warp collective references an invalid operand");
            return nullptr;
        }
        auto *llvm_operand = _as_lane_vector(
            _load_value(operand_id), *operand);
        if (llvm_operand == nullptr) { return nullptr; }
        operands.emplace_back(llvm_operand);
        operand_values.emplace_back(operand);
    }
    auto require = [&](size_t count) {
        if (operands.size() != count) {
            _fail("warp collective has an invalid operand count");
            return false;
        }
        return true;
    };
    auto op = static_cast<xir::ThreadGroupOp>(*instruction.source_op);
    if (instruction.cohort_uniform_operand_index &&
        (op != xir::ThreadGroupOp::WARP_READ_LANE || *instruction.cohort_uniform_operand_index != 1u)) {
        _fail("warp cohort-uniform operand must identify a read-lane source index");
        return nullptr;
    }
    auto cohort_scalar = [&](::llvm::Value *lanes) {
        if (lanes == nullptr || result_value == nullptr ||
            result_value->value_class !=
                schedule::ValueClass::cohort_uniform) {
            return lanes;
        }
        return _extract_lane(
            lanes, result_value->type,
            _safe_first_lane(participants));
    };
    auto scalar_result = [&](::llvm::Value *scalar) {
        // Compute the collective once in this cohort, then form the declared
        // value shape. Escaping results need masked lane-wise snapshots, not
        // a scalar reload from the first lane after distinct epochs join.
        if (scalar == nullptr || result_value == nullptr ||
            result_value->value_class != schedule::ValueClass::varying) {
            return scalar;
        }
        // This also handles aggregate reductions and the uint4 ballot value;
        // their uniform representation is not necessarily an LLVM scalar.
        return _splat_data(scalar, result_value->type);
    };
    auto reduce_components = [&](const UnaryLeaf &leaf) {
        if (!require(1u) || result_value == nullptr) {
            return static_cast<::llvm::Value *>(nullptr);
        }
        return scalar_result(_componentwise_varying_to_uniform(
            result_value->type, operands[0u],
            operand_values[0u]->type, leaf));
    };
    auto scan_components = [&](const UnaryLeaf &leaf) {
        if (!require(1u) || result_value == nullptr) {
            return static_cast<::llvm::Value *>(nullptr);
        }
        return _componentwise_unary(
            result_value->type, operands[0u],
            operand_values[0u]->type, true, leaf);
    };
    switch (op) {
        case xir::ThreadGroupOp::WARP_IS_FIRST_ACTIVE_LANE:
            if (!require(0u)) { return nullptr; }
            return _collectives.is_first_active_lane(
                _builder, participants);
        case xir::ThreadGroupOp::WARP_FIRST_ACTIVE_LANE:
            if (!require(0u)) { return nullptr; }
            return scalar_result(_collectives.first_active_lane(
                _builder, participants));
        case xir::ThreadGroupOp::WARP_ACTIVE_ALL_EQUAL:
            return reduce_components(
                [&](::llvm::Value *value, const Type *) {
                    return _collectives.active_all_equal(
                        _builder, value, participants);
                });
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_AND:
            return reduce_components(
                [&](::llvm::Value *value, const Type *) {
                    return _collectives.active_bit_and(
                        _builder, value, participants);
                });
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_OR:
            return reduce_components(
                [&](::llvm::Value *value, const Type *) {
                    return _collectives.active_bit_or(
                        _builder, value, participants);
                });
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_XOR:
            return reduce_components(
                [&](::llvm::Value *value, const Type *) {
                    return _collectives.active_bit_xor(
                        _builder, value, participants);
                });
        case xir::ThreadGroupOp::WARP_ACTIVE_COUNT_BITS:
            if (!require(1u)) { return nullptr; }
            return scalar_result(_collectives.active_count_bits(
                _builder, operands[0u], participants));
        case xir::ThreadGroupOp::WARP_ACTIVE_MAX:
            return reduce_components(
                [&](::llvm::Value *value, const Type *type) {
                    return _collectives.active_max(
                        _builder, value, participants,
                        type->is_int());
                });
        case xir::ThreadGroupOp::WARP_ACTIVE_MIN:
            return reduce_components(
                [&](::llvm::Value *value, const Type *type) {
                    return _collectives.active_min(
                        _builder, value, participants,
                        type->is_int());
                });
        case xir::ThreadGroupOp::WARP_ACTIVE_PRODUCT:
            return reduce_components(
                [&](::llvm::Value *value, const Type *) {
                    return _collectives.active_product(
                        _builder, value, participants);
                });
        case xir::ThreadGroupOp::WARP_ACTIVE_SUM:
            return reduce_components(
                [&](::llvm::Value *value, const Type *) {
                    return _collectives.active_sum(
                        _builder, value, participants);
                });
        case xir::ThreadGroupOp::WARP_ACTIVE_ALL:
            if (!require(1u)) { return nullptr; }
            return scalar_result(_collectives.active_all(
                _builder, operands[0u], participants));
        case xir::ThreadGroupOp::WARP_ACTIVE_ANY:
            if (!require(1u)) { return nullptr; }
            return scalar_result(_collectives.active_any(
                _builder, operands[0u], participants));
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_MASK:
            if (!require(1u)) { return nullptr; }
            return scalar_result(_collectives.active_bit_mask(
                _builder, operands[0u], participants));
        case xir::ThreadGroupOp::WARP_PREFIX_COUNT_BITS:
            if (!require(1u)) { return nullptr; }
            return _collectives.prefix_count_bits(
                _builder, operands[0u], participants);
        case xir::ThreadGroupOp::WARP_PREFIX_SUM:
            return scan_components(
                [&](::llvm::Value *value, const Type *) {
                    return _collectives.prefix_sum(
                        _builder, value, participants);
                });
        case xir::ThreadGroupOp::WARP_PREFIX_PRODUCT:
            return scan_components(
                [&](::llvm::Value *value, const Type *) {
                    return _collectives.prefix_product(
                        _builder, value, participants);
                });
        case xir::ThreadGroupOp::WARP_READ_LANE:
            if (!require(2u)) { return nullptr; }
            if (result_value == nullptr) { return nullptr; }
            if (instruction.cohort_uniform_operand_index == 1u &&
                !luisa::compute::detail::env_flag("LUISA_SIMD_DISABLE_UNIFORM_READ_LANE")) {
                // Read the saved source index in this use's participant
                // cohort, not the value's defining block or another epoch.
                // An empty mask may leave every source lane poison: discard
                // that extraction before it can become a dynamic index.
                auto *source = _builder.CreateExtractElement(operands[1u], _safe_first_lane(participants));
                source = _builder.CreateSelect(_builder.CreateOrReduce(participants), source,
                                               ::llvm::Constant::getNullValue(source->getType()), "warp.source.uniform");
                operands[1u] = _builder.CreateVectorSplat(_width, source);
            }
            return cohort_scalar(_componentwise_unary(
                result_value->type, operands[0u],
                operand_values[0u]->type, true,
                [&](::llvm::Value *value, const Type *) {
                    return _collectives.read_lane(
                        _builder, value, operands[1u], participants)
                        .values;
                }));
        case xir::ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE:
            if (!require(1u) || result_value == nullptr) {
                return nullptr;
            }
            return cohort_scalar(_componentwise_unary(
                result_value->type, operands[0u],
                operand_values[0u]->type, true,
                [&](::llvm::Value *value, const Type *) {
                    return _collectives.read_first_active_lane(
                        _builder, value, participants)
                        .values;
                }));
        default:
            _fail("Phase-2 LLVM packet codegen encountered a non-warp thread-group operation");
            return nullptr;
    }
}

}// namespace luisa::compute::simd::detail
