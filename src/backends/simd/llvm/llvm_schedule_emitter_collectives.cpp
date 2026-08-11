#include "llvm_schedule_emitter.h"

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
    auto cohort_scalar = [&](::llvm::Value *lanes) {
        if (lanes == nullptr || result_value == nullptr ||
            result_value->value_class !=
                schedule::ValueClass::cohort_uniform) {
            return lanes;
        }
        auto *first = _collectives.first_active_lane(
            _builder, participants);
        auto *safe = _builder.CreateSelect(
            _builder.CreateOrReduce(participants), first,
            _builder.getInt32(0u));
        return _extract_lane(lanes, result_value->type, safe);
    };
    auto reduce_components = [&](const UnaryLeaf &leaf) {
        if (!require(1u) || result_value == nullptr) {
            return static_cast<::llvm::Value *>(nullptr);
        }
        return _componentwise_varying_to_uniform(
            result_value->type, operands[0u],
            operand_values[0u]->type, leaf);
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
            return _collectives.first_active_lane(
                _builder, participants);
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
            return _collectives.active_count_bits(
                _builder, operands[0u], participants);
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
            return _collectives.active_all(
                _builder, operands[0u], participants);
        case xir::ThreadGroupOp::WARP_ACTIVE_ANY:
            if (!require(1u)) { return nullptr; }
            return _collectives.active_any(
                _builder, operands[0u], participants);
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_MASK:
            if (!require(1u)) { return nullptr; }
            return _collectives.active_bit_mask(
                _builder, operands[0u], participants);
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
