#include "llvm_schedule_emitter.h"

#include "../../common/llvm_native_math.h"

namespace luisa::compute::simd::detail {

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_select_data(
    ::llvm::Value *condition, ::llvm::Value *true_value,
    ::llvm::Value *false_value, const Type *type, bool varying) {
    if (_is_scalar_data(type)) {
        return _builder.CreateSelect(
            condition, true_value, false_value);
    }
    return _assemble(type, varying, [&](uint32_t i) {
        return _select_data(
            condition,
            _extract_child(true_value, type, i, varying),
            _extract_child(false_value, type, i, varying),
            _child_type(type, i), varying);
    });
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_componentwise_unary(
    const Type *result_type, ::llvm::Value *operand,
    const Type *operand_type, bool varying, const UnaryLeaf &leaf) {
    if (_is_scalar_data(result_type)) {
        return leaf(operand, operand_type);
    }
    return _assemble(result_type, varying, [&](uint32_t i) {
        auto scalar_operand = _is_scalar_data(operand_type);
        auto *child_operand_type = scalar_operand ? operand_type :
                                                    _child_type(operand_type, i);
        auto *child_operand = scalar_operand ? operand :
                                               _extract_child(operand, operand_type, i, varying);
        return _componentwise_unary(
            _child_type(result_type, i), child_operand,
            child_operand_type, varying, leaf);
    });
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_componentwise_binary(
    const Type *result_type, ::llvm::Value *lhs, const Type *lhs_type,
    ::llvm::Value *rhs, const Type *rhs_type, bool varying,
    const BinaryLeaf &leaf) {
    if (_is_scalar_data(result_type)) {
        return leaf(lhs, rhs, lhs_type, rhs_type);
    }
    return _assemble(result_type, varying, [&](uint32_t i) {
        auto lhs_scalar = _is_scalar_data(lhs_type);
        auto rhs_scalar = _is_scalar_data(rhs_type);
        auto *lhs_child_type = lhs_scalar ? lhs_type :
                                            _child_type(lhs_type, i);
        auto *rhs_child_type = rhs_scalar ? rhs_type :
                                            _child_type(rhs_type, i);
        auto *lhs_child = lhs_scalar ? lhs :
                                       _extract_child(lhs, lhs_type, i, varying);
        auto *rhs_child = rhs_scalar ? rhs :
                                       _extract_child(rhs, rhs_type, i, varying);
        return _componentwise_binary(
            _child_type(result_type, i), lhs_child, lhs_child_type,
            rhs_child, rhs_child_type, varying, leaf);
    });
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_componentwise_ternary(
    const Type *result_type,
    ::llvm::Value *a, const Type *a_type,
    ::llvm::Value *b, const Type *b_type,
    ::llvm::Value *c, const Type *c_type, bool varying,
    const TernaryLeaf &leaf) {
    if (_is_scalar_data(result_type)) {
        return leaf(a, b, c, a_type, b_type, c_type);
    }
    return _assemble(result_type, varying, [&](uint32_t i) {
        auto a_scalar = _is_scalar_data(a_type);
        auto b_scalar = _is_scalar_data(b_type);
        auto c_scalar = _is_scalar_data(c_type);
        auto *a_child_type = a_scalar ? a_type :
                                        _child_type(a_type, i);
        auto *b_child_type = b_scalar ? b_type :
                                        _child_type(b_type, i);
        auto *c_child_type = c_scalar ? c_type :
                                        _child_type(c_type, i);
        auto *a_child = a_scalar ? a :
                                   _extract_child(a, a_type, i, varying);
        auto *b_child = b_scalar ? b :
                                   _extract_child(b, b_type, i, varying);
        auto *c_child = c_scalar ? c :
                                   _extract_child(c, c_type, i, varying);
        return _componentwise_ternary(
            _child_type(result_type, i),
            a_child, a_child_type, b_child, b_child_type,
            c_child, c_child_type, varying, leaf);
    });
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_componentwise_varying_to_uniform(
    const Type *result_type, ::llvm::Value *operand,
    const Type *operand_type, const UnaryLeaf &leaf) {
    if (_is_scalar_data(result_type)) {
        if (!_is_scalar_data(operand_type)) {
            _fail("aggregate warp collective result shape does not match its operand");
            return nullptr;
        }
        return leaf(operand, operand_type);
    }
    if (_is_scalar_data(operand_type) ||
        _child_count(result_type) != _child_count(operand_type)) {
        _fail("aggregate warp collective result shape does not match its operand");
        return nullptr;
    }
    return _assemble(result_type, false, [&](uint32_t i) {
        return _componentwise_varying_to_uniform(
            _child_type(result_type, i),
            _extract_child(operand, operand_type, i, true),
            _child_type(operand_type, i), leaf);
    });
}

[[nodiscard]] std::optional<uint64_t> ScheduleEmitter::_constant_index(
    ::llvm::Value *value) noexcept {
    if (auto *integer = ::llvm::dyn_cast<::llvm::ConstantInt>(value)) {
        return integer->getZExtValue();
    }
    if (auto *constant = ::llvm::dyn_cast<::llvm::Constant>(value)) {
        if (auto *splat = constant->getSplatValue()) {
            if (auto *integer = ::llvm::dyn_cast<::llvm::ConstantInt>(splat)) {
                return integer->getZExtValue();
            }
        }
    }
    return std::nullopt;
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_index_constant_like(
    ::llvm::Value *index, uint64_t value) {
    auto *type = index->getType();
    if (auto *vector = ::llvm::dyn_cast<::llvm::VectorType>(type)) {
        auto *element = ::llvm::cast<::llvm::IntegerType>(
            vector->getElementType());
        return _builder.CreateVectorSplat(
            vector->getElementCount(),
            ::llvm::ConstantInt::get(element, value));
    }
    return ::llvm::ConstantInt::get(
        ::llvm::cast<::llvm::IntegerType>(type), value);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_extract_indexed(
    ::llvm::Value *aggregate, const Type *type,
    const std::vector<::llvm::Value *> &indices, size_t depth,
    bool varying) {
    if (depth == indices.size()) { return aggregate; }
    auto *index = indices[depth];
    auto count = _child_count(type);
    if (count == 0u) {
        _fail("aggregate extract has too many indices");
        return nullptr;
    }
    ::llvm::Value *selected = nullptr;
    auto *child_type = _child_type(type, 0u);
    if (auto constant = _constant_index(index)) {
        if (*constant >= count) {
            _fail("aggregate extract index is out of range");
            return nullptr;
        }
        selected = _extract_child(
            aggregate, type, static_cast<uint32_t>(*constant), varying);
        child_type = _child_type(
            type, static_cast<uint32_t>(*constant));
    } else {
        if (type->is_structure()) {
            _fail("dynamic structure member extraction is invalid");
            return nullptr;
        }
        selected = _extract_child(aggregate, type, 0u, varying);
        for (auto i = uint32_t{1u}; i < count; i++) {
            auto *candidate = _extract_child(
                aggregate, type, i, varying);
            auto *condition = _builder.CreateICmpEQ(
                index, _index_constant_like(index, i));
            selected = _select_data(
                condition, candidate, selected, child_type, varying);
        }
    }
    return _extract_indexed(
        selected, child_type, indices, depth + 1u, varying);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_insert_indexed(
    ::llvm::Value *aggregate, const Type *type,
    ::llvm::Value *replacement,
    const std::vector<::llvm::Value *> &indices, size_t depth,
    bool varying) {
    if (depth == indices.size()) { return replacement; }
    auto *index = indices[depth];
    auto count = _child_count(type);
    if (count == 0u) {
        _fail("aggregate insert has too many indices");
        return nullptr;
    }
    if (auto constant = _constant_index(index)) {
        if (*constant >= count) {
            _fail("aggregate insert index is out of range");
            return nullptr;
        }
        auto i = static_cast<uint32_t>(*constant);
        auto *child_type = _child_type(type, i);
        auto *old_child = _extract_child(
            aggregate, type, i, varying);
        auto *new_child = _insert_indexed(
            old_child, child_type, replacement,
            indices, depth + 1u, varying);
        return new_child == nullptr ? nullptr :
                                      _insert_child(aggregate, new_child, type, i, varying);
    }
    if (type->is_structure()) {
        _fail("dynamic structure member insertion is invalid");
        return nullptr;
    }
    auto *result = aggregate;
    for (auto i = uint32_t{0u}; i < count; i++) {
        auto *child_type = _child_type(type, i);
        auto *old_child = _extract_child(
            aggregate, type, i, varying);
        auto *updated = _insert_indexed(
            old_child, child_type, replacement,
            indices, depth + 1u, varying);
        if (updated == nullptr) { return nullptr; }
        auto *condition = _builder.CreateICmpEQ(
            index, _index_constant_like(index, i));
        auto *selected = _select_data(
            condition, updated, old_child, child_type, varying);
        result = _insert_child(
            result, selected, type, i, varying);
    }
    return result;
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_aggregate_operation(
    const schedule::Value &result,
    const schedule::Instruction &instruction,
    const std::vector<::llvm::Value *> &operands, bool varying) {
    if (operands.size() != _child_count(result.type)) {
        _fail("aggregate construction operand count mismatch");
        return nullptr;
    }
    return _assemble(result.type, varying, [&](uint32_t i) {
        return operands[i];
    });
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_arithmetic(
    const schedule::Instruction &instruction) {
    if (!instruction.result || !instruction.source_op) {
        _fail("arithmetic instruction is missing result or source operation");
        return nullptr;
    }
    auto *result = _source.value(*instruction.result);
    if (result == nullptr || !_is_data(result->type)) {
        _fail("arithmetic requires a supported Luisa data result type");
        return nullptr;
    }
    auto varying = result->value_class == schedule::ValueClass::varying;
    std::vector<::llvm::Value *> operands;
    std::vector<const Type *> operand_types;
    operands.reserve(instruction.operands.size());
    operand_types.reserve(instruction.operands.size());
    for (auto operand_id : instruction.operands) {
        auto *operand = _source.value(operand_id);
        auto *llvm_operand = _load_value(operand_id);
        if (varying) {
            llvm_operand = _as_lane_vector(llvm_operand, *operand);
        }
        if (llvm_operand == nullptr) { return nullptr; }
        operands.emplace_back(llvm_operand);
        operand_types.emplace_back(operand->type);
    }
    auto require = [&](size_t count) {
        if (operands.size() != count) {
            _fail("arithmetic operation has an invalid operand count");
            return false;
        }
        return true;
    };
    auto op = static_cast<xir::ArithmeticOp>(*instruction.source_op);
    auto native_math_mode = _enable_fast_math ?
                                cpu::LLVMNativeMathMode::fast :
                                cpu::LLVMNativeMathMode::precise;
    if (op == xir::ArithmeticOp::AGGREGATE) {
        return _aggregate_operation(
            *result, instruction, operands, varying);
    }
    if (op == xir::ArithmeticOp::EXTRACT ||
        op == xir::ArithmeticOp::INSERT ||
        op == xir::ArithmeticOp::SHUFFLE) {
        if (operands.size() < 2u) {
            _fail("aggregate extraction requires an aggregate and indices");
            return nullptr;
        }
        if (op == xir::ArithmeticOp::SHUFFLE) {
            return _assemble(result->type, varying, [&](uint32_t i) {
                std::vector<::llvm::Value *> index{operands[i + 1u]};
                return _extract_indexed(
                    operands[0u], operand_types[0u], index, 0u,
                    varying);
            });
        }
        if (op == xir::ArithmeticOp::INSERT) {
            if (operands.size() < 3u) {
                _fail("aggregate insertion requires a base, value, and indices");
                return nullptr;
            }
            std::vector<::llvm::Value *> indices{
                operands.begin() + 2, operands.end()};
            return _insert_indexed(
                operands[0u], operand_types[0u], operands[1u],
                indices, 0u, varying);
        }
        std::vector<::llvm::Value *> indices{
            operands.begin() + 1, operands.end()};
        return _extract_indexed(
            operands[0u], operand_types[0u], indices, 0u, varying);
    }

    auto unary = [&](const UnaryLeaf &leaf) -> ::llvm::Value * {
        if (!require(1u)) { return nullptr; }
        return _componentwise_unary(
            result->type, operands[0u], operand_types[0u],
            varying, leaf);
    };
    auto binary = [&](const BinaryLeaf &leaf) -> ::llvm::Value * {
        if (!require(2u)) { return nullptr; }
        return _componentwise_binary(
            result->type, operands[0u], operand_types[0u],
            operands[1u], operand_types[1u], varying, leaf);
    };
    auto ternary = [&](const TernaryLeaf &leaf) -> ::llvm::Value * {
        if (!require(3u)) { return nullptr; }
        return _componentwise_ternary(
            result->type,
            operands[0u], operand_types[0u],
            operands[1u], operand_types[1u],
            operands[2u], operand_types[2u], varying, leaf);
    };
    auto intrinsic = [&](::llvm::Intrinsic::ID id,
                         std::initializer_list<::llvm::Value *> args) {
        std::vector<::llvm::Value *> values{args};
        std::array<::llvm::Type *, 1u> overloads{
            values.front()->getType()};
#if LLVM_VERSION_MAJOR >= 22
        auto *function = ::llvm::Intrinsic::getOrInsertDeclaration(
#else
        auto *function = ::llvm::Intrinsic::getDeclaration(
#endif
            &_module, id, overloads);
        return _builder.CreateCall(function, values);
    };
    auto unary_intrinsic = [&](::llvm::Intrinsic::ID id) {
        return unary([&](::llvm::Value *value, const Type *) {
            return intrinsic(id, {value});
        });
    };
    auto binary_intrinsic = [&](::llvm::Intrinsic::ID id) {
        return binary([&](::llvm::Value *lhs, ::llvm::Value *rhs,
                          const Type *, const Type *) {
            return intrinsic(id, {lhs, rhs});
        });
    };
    auto float_constant_like = [&](::llvm::Value *value, double x) {
        auto *scalar = ::llvm::ConstantFP::get(
            value->getType()->getScalarType(), x);
        if (auto *vector = ::llvm::dyn_cast<::llvm::VectorType>(
                value->getType())) {
            return static_cast<::llvm::Constant *>(
                ::llvm::ConstantVector::getSplat(
                    vector->getElementCount(), scalar));
        }
        return scalar;
    };
    auto integer_constant_like = [&](::llvm::Value *value, uint64_t x) {
        auto *scalar = ::llvm::ConstantInt::get(
            value->getType()->getScalarType(), x);
        if (auto *vector = ::llvm::dyn_cast<::llvm::VectorType>(
                value->getType())) {
            return static_cast<::llvm::Constant *>(
                ::llvm::ConstantVector::getSplat(
                    vector->getElementCount(), scalar));
        }
        return static_cast<::llvm::Constant *>(scalar);
    };
    auto sanitize_inactive_integer = [&](::llvm::Value *value,
                                         uint64_t neutral) {
        return varying ?
                   _builder.CreateSelect(
                       _active_mask, value,
                       integer_constant_like(value, neutral)) :
                   value;
    };
    auto minmax_leaf = [&](::llvm::Value *lhs, ::llvm::Value *rhs,
                           const Type *type, bool maximum)
        -> ::llvm::Value * {
        if (type->is_float16() || type->is_float32() ||
            type->is_float64()) {
            return intrinsic(
                maximum ? ::llvm::Intrinsic::maxnum :
                          ::llvm::Intrinsic::minnum,
                {lhs, rhs});
        }
        auto predicate = maximum ?
                             type->is_int() ? ::llvm::CmpInst::ICMP_SGT :
                                              ::llvm::CmpInst::ICMP_UGT :
                         type->is_int() ? ::llvm::CmpInst::ICMP_SLT :
                                          ::llvm::CmpInst::ICMP_ULT;
        return _builder.CreateSelect(
            _builder.CreateICmp(predicate, lhs, rhs), lhs, rhs);
    };
    auto dot = [&](::llvm::Value *lhs, ::llvm::Value *rhs,
                   const Type *type) -> ::llvm::Value * {
        if (!type->is_vector()) {
            _fail("dot product requires vector operands");
            return nullptr;
        }
        ::llvm::Value *sum = nullptr;
        for (auto i = uint32_t{0u}; i < type->dimension(); i++) {
            auto *a = _extract_child(lhs, type, i, varying);
            auto *b = _extract_child(rhs, type, i, varying);
            auto *product = _builder.CreateFMul(a, b);
            sum = sum == nullptr ? product :
                                   _builder.CreateFAdd(sum, product);
        }
        return sum;
    };
    auto binary_leaf = [&](::llvm::Value *lhs, ::llvm::Value *rhs,
                           const Type *lhs_type,
                           const Type *) -> ::llvm::Value * {
        auto is_float = lhs_type->is_float16() ||
                        lhs_type->is_float32() ||
                        lhs_type->is_float64();
        auto is_signed = lhs_type->is_int();
        switch (op) {
            case xir::ArithmeticOp::BINARY_ADD:
            case xir::ArithmeticOp::MATRIX_COMP_ADD:
                return is_float ? _builder.CreateFAdd(lhs, rhs) :
                                  _builder.CreateAdd(lhs, rhs);
            case xir::ArithmeticOp::BINARY_SUB:
            case xir::ArithmeticOp::MATRIX_COMP_SUB:
                return is_float ? _builder.CreateFSub(lhs, rhs) :
                                  _builder.CreateSub(lhs, rhs);
            case xir::ArithmeticOp::BINARY_MUL:
            case xir::ArithmeticOp::MATRIX_COMP_MUL:
                return is_float ? _builder.CreateFMul(lhs, rhs) :
                                  _builder.CreateMul(lhs, rhs);
            case xir::ArithmeticOp::BINARY_DIV:
            case xir::ArithmeticOp::MATRIX_COMP_DIV: {
                if (is_float) { return _builder.CreateFDiv(lhs, rhs); }
                // Integer division/remainder may lower to trapping host
                // instructions. Inactive vector lanes are semantically
                // absent, so make both operands defined before executing the
                // instruction instead of relying on a later masked merge.
                lhs = sanitize_inactive_integer(lhs, 0u);
                rhs = sanitize_inactive_integer(rhs, 1u);
                return is_signed ? _builder.CreateSDiv(lhs, rhs) :
                                   _builder.CreateUDiv(lhs, rhs);
            }
            case xir::ArithmeticOp::BINARY_MOD: {
                if (is_float) { return _builder.CreateFRem(lhs, rhs); }
                lhs = sanitize_inactive_integer(lhs, 0u);
                rhs = sanitize_inactive_integer(rhs, 1u);
                return is_signed ? _builder.CreateSRem(lhs, rhs) :
                                   _builder.CreateURem(lhs, rhs);
            }
            case xir::ArithmeticOp::BINARY_BIT_AND:
                return _builder.CreateAnd(lhs, rhs);
            case xir::ArithmeticOp::BINARY_BIT_OR:
                return _builder.CreateOr(lhs, rhs);
            case xir::ArithmeticOp::BINARY_BIT_XOR:
                return _builder.CreateXor(lhs, rhs);
            case xir::ArithmeticOp::BINARY_SHIFT_LEFT:
                return _builder.CreateShl(
                    sanitize_inactive_integer(lhs, 0u),
                    sanitize_inactive_integer(rhs, 0u));
            case xir::ArithmeticOp::BINARY_SHIFT_RIGHT:
                lhs = sanitize_inactive_integer(lhs, 0u);
                rhs = sanitize_inactive_integer(rhs, 0u);
                return is_signed ? _builder.CreateAShr(lhs, rhs) :
                                   _builder.CreateLShr(lhs, rhs);
            case xir::ArithmeticOp::BINARY_LESS:
            case xir::ArithmeticOp::BINARY_GREATER:
            case xir::ArithmeticOp::BINARY_LESS_EQUAL:
            case xir::ArithmeticOp::BINARY_GREATER_EQUAL:
            case xir::ArithmeticOp::BINARY_EQUAL:
            case xir::ArithmeticOp::BINARY_NOT_EQUAL: {
                if (is_float) {
                    auto predicate = ::llvm::CmpInst::FCMP_FALSE;
                    switch (op) {
                        case xir::ArithmeticOp::BINARY_LESS: predicate = ::llvm::CmpInst::FCMP_OLT; break;
                        case xir::ArithmeticOp::BINARY_GREATER: predicate = ::llvm::CmpInst::FCMP_OGT; break;
                        case xir::ArithmeticOp::BINARY_LESS_EQUAL: predicate = ::llvm::CmpInst::FCMP_OLE; break;
                        case xir::ArithmeticOp::BINARY_GREATER_EQUAL: predicate = ::llvm::CmpInst::FCMP_OGE; break;
                        case xir::ArithmeticOp::BINARY_EQUAL: predicate = ::llvm::CmpInst::FCMP_OEQ; break;
                        case xir::ArithmeticOp::BINARY_NOT_EQUAL: predicate = ::llvm::CmpInst::FCMP_UNE; break;
                        default: break;
                    }
                    return _builder.CreateFCmp(predicate, lhs, rhs);
                }
                auto predicate = ::llvm::CmpInst::BAD_ICMP_PREDICATE;
                switch (op) {
                    case xir::ArithmeticOp::BINARY_LESS: predicate = is_signed ? ::llvm::CmpInst::ICMP_SLT : ::llvm::CmpInst::ICMP_ULT; break;
                    case xir::ArithmeticOp::BINARY_GREATER: predicate = is_signed ? ::llvm::CmpInst::ICMP_SGT : ::llvm::CmpInst::ICMP_UGT; break;
                    case xir::ArithmeticOp::BINARY_LESS_EQUAL: predicate = is_signed ? ::llvm::CmpInst::ICMP_SLE : ::llvm::CmpInst::ICMP_ULE; break;
                    case xir::ArithmeticOp::BINARY_GREATER_EQUAL: predicate = is_signed ? ::llvm::CmpInst::ICMP_SGE : ::llvm::CmpInst::ICMP_UGE; break;
                    case xir::ArithmeticOp::BINARY_EQUAL: predicate = ::llvm::CmpInst::ICMP_EQ; break;
                    case xir::ArithmeticOp::BINARY_NOT_EQUAL: predicate = ::llvm::CmpInst::ICMP_NE; break;
                    default: break;
                }
                return _builder.CreateICmp(predicate, lhs, rhs);
            }
            default: return nullptr;
        }
    };
    switch (op) {
        case xir::ArithmeticOp::UNARY_MINUS:
        case xir::ArithmeticOp::MATRIX_COMP_NEG:
            return unary([&](::llvm::Value *value, const Type *type)
                             -> ::llvm::Value * {
                return type->is_float16() || type->is_float32() ||
                               type->is_float64() ?
                           _builder.CreateFNeg(value) :
                           _builder.CreateNeg(value);
            });
        case xir::ArithmeticOp::UNARY_BIT_NOT:
            return unary([&](::llvm::Value *value, const Type *) {
                return _builder.CreateNot(value);
            });
        case xir::ArithmeticOp::BINARY_ADD:
        case xir::ArithmeticOp::MATRIX_COMP_ADD:
        case xir::ArithmeticOp::BINARY_SUB:
        case xir::ArithmeticOp::MATRIX_COMP_SUB:
        case xir::ArithmeticOp::BINARY_MUL:
        case xir::ArithmeticOp::MATRIX_COMP_MUL:
        case xir::ArithmeticOp::BINARY_DIV:
        case xir::ArithmeticOp::MATRIX_COMP_DIV:
        case xir::ArithmeticOp::BINARY_MOD:
        case xir::ArithmeticOp::BINARY_BIT_AND:
        case xir::ArithmeticOp::BINARY_BIT_OR:
        case xir::ArithmeticOp::BINARY_BIT_XOR:
        case xir::ArithmeticOp::BINARY_SHIFT_LEFT:
        case xir::ArithmeticOp::BINARY_SHIFT_RIGHT:
        case xir::ArithmeticOp::BINARY_LESS:
        case xir::ArithmeticOp::BINARY_GREATER:
        case xir::ArithmeticOp::BINARY_LESS_EQUAL:
        case xir::ArithmeticOp::BINARY_GREATER_EQUAL:
        case xir::ArithmeticOp::BINARY_EQUAL:
        case xir::ArithmeticOp::BINARY_NOT_EQUAL:
            return binary(binary_leaf);
        case xir::ArithmeticOp::SELECT:
            return ternary(
                [&](::llvm::Value *if_false,
                    ::llvm::Value *if_true,
                    ::llvm::Value *condition,
                    const Type *, const Type *, const Type *) {
                    return _builder.CreateSelect(
                        condition, if_true, if_false);
                });
        case xir::ArithmeticOp::CLAMP:
            return ternary(
                [&](::llvm::Value *value, ::llvm::Value *low,
                    ::llvm::Value *high, const Type *type,
                    const Type *, const Type *) {
                    return minmax_leaf(
                        minmax_leaf(value, low, type, true),
                        high, type, false);
                });
        case xir::ArithmeticOp::SATURATE:
            return unary([&](::llvm::Value *value, const Type *type) {
                auto *zero = float_constant_like(value, 0.0);
                auto *one = float_constant_like(value, 1.0);
                return minmax_leaf(
                    minmax_leaf(value, zero, type, true),
                    one, type, false);
            });
        case xir::ArithmeticOp::LERP:
            return ternary(
                [&](::llvm::Value *a, ::llvm::Value *b,
                    ::llvm::Value *t, const Type *,
                    const Type *, const Type *) {
                    return _builder.CreateFAdd(
                        a, _builder.CreateFMul(
                               _builder.CreateFSub(b, a), t));
                });
        case xir::ArithmeticOp::ABS:
            return unary([&](::llvm::Value *value, const Type *type)
                             -> ::llvm::Value * {
                if (type->is_float16() || type->is_float32() ||
                    type->is_float64()) {
                    return intrinsic(::llvm::Intrinsic::fabs, {value});
                }
                if (!type->is_int()) { return value; }
                auto *zero = ::llvm::Constant::getNullValue(
                    value->getType());
                return _builder.CreateSelect(
                    _builder.CreateICmpSLT(value, zero),
                    _builder.CreateNeg(value), value);
            });
        case xir::ArithmeticOp::MIN:
        case xir::ArithmeticOp::MAX:
            return binary([&](::llvm::Value *lhs, ::llvm::Value *rhs,
                              const Type *type, const Type *) {
                return minmax_leaf(
                    lhs, rhs, type,
                    op == xir::ArithmeticOp::MAX);
            });
        case xir::ArithmeticOp::ISNAN:
            return unary([&](::llvm::Value *value, const Type *) {
                return _builder.CreateFCmpUNO(value, value);
            });
        case xir::ArithmeticOp::ISINF:
            return unary([&](::llvm::Value *value, const Type *) {
                auto *absolute = intrinsic(
                    ::llvm::Intrinsic::fabs, {value});
                auto *infinity = ::llvm::ConstantFP::getInfinity(
                    value->getType()->getScalarType());
                if (auto *vector = ::llvm::dyn_cast<::llvm::VectorType>(
                        value->getType())) {
                    infinity = ::llvm::ConstantVector::getSplat(
                        vector->getElementCount(), infinity);
                }
                return _builder.CreateFCmpOEQ(absolute, infinity);
            });
        case xir::ArithmeticOp::ACOS:
            return unary([&](::llvm::Value *value, const Type *type)
                             -> ::llvm::Value * {
                if (varying && type->is_float32()) {
                    auto *safe = _builder.CreateSelect(
                        _active_mask, value,
                        float_constant_like(value, 0.0));
                    auto *native = cpu::LLVMNativeMath::emit_acos_f32(
                        _module, _builder, safe,
                        native_math_mode);
                    if (native == nullptr) {
                        _fail("native SIMD acos requires fixed f32 vectors");
                    }
                    return native;
                }
                return intrinsic(::llvm::Intrinsic::acos, {value});
            });
        case xir::ArithmeticOp::ASIN:
            return unary([&](::llvm::Value *value, const Type *type)
                             -> ::llvm::Value * {
                if (varying && type->is_float32()) {
                    auto *safe = _builder.CreateSelect(
                        _active_mask, value,
                        float_constant_like(value, 0.0));
                    auto *native = cpu::LLVMNativeMath::emit_asin_f32(
                        _module, _builder, safe,
                        native_math_mode);
                    if (native == nullptr) {
                        _fail("native SIMD asin requires fixed f32 vectors");
                    }
                    return native;
                }
                return intrinsic(::llvm::Intrinsic::asin, {value});
            });
        case xir::ArithmeticOp::ATAN:
            return unary([&](::llvm::Value *value, const Type *type)
                             -> ::llvm::Value * {
                if (varying && type->is_float32()) {
                    auto *safe = _builder.CreateSelect(
                        _active_mask, value,
                        float_constant_like(value, 0.0));
                    auto *native = cpu::LLVMNativeMath::emit_atan_f32(
                        _module, _builder, safe,
                        native_math_mode);
                    if (native == nullptr) {
                        _fail("native SIMD atan requires fixed f32 vectors");
                    }
                    return native;
                }
                return intrinsic(::llvm::Intrinsic::atan, {value});
            });
        case xir::ArithmeticOp::ATAN2:
            return binary_intrinsic(::llvm::Intrinsic::atan2);
        case xir::ArithmeticOp::COS:
            return unary([&](::llvm::Value *value, const Type *type)
                             -> ::llvm::Value * {
                if (varying && type->is_float32()) {
                    auto *safe = _builder.CreateSelect(
                        _active_mask, value,
                        float_constant_like(value, 0.0));
                    auto *native = cpu::LLVMNativeMath::emit_cos_f32(
                        _module, _builder, safe,
                        native_math_mode);
                    if (native == nullptr) {
                        _fail("native SIMD cos requires fixed f32 vectors");
                    }
                    return native;
                }
                return intrinsic(::llvm::Intrinsic::cos, {value});
            });
        case xir::ArithmeticOp::SIN:
            return unary([&](::llvm::Value *value, const Type *type)
                             -> ::llvm::Value * {
                if (varying && type->is_float32()) {
                    // Native math may contain float-to-int conversions and
                    // table gathers. Inactive lanes are semantically absent,
                    // so neutralize them before entering the implementation.
                    auto *safe = _builder.CreateSelect(
                        _active_mask, value,
                        float_constant_like(value, 0.0));
                    auto *native = cpu::LLVMNativeMath::emit_sin_f32(
                        _module, _builder, safe,
                        native_math_mode);
                    if (native == nullptr) {
                        _fail("native SIMD sin requires fixed f32 vectors");
                    }
                    return native;
                }
                return intrinsic(::llvm::Intrinsic::sin, {value});
            });
        case xir::ArithmeticOp::TAN:
            return unary([&](::llvm::Value *value, const Type *type)
                             -> ::llvm::Value * {
                if (varying && type->is_float32()) {
                    auto *safe = _builder.CreateSelect(
                        _active_mask, value,
                        float_constant_like(value, 0.0));
                    auto *native = cpu::LLVMNativeMath::emit_tan_f32(
                        _module, _builder, safe,
                        native_math_mode);
                    if (native == nullptr) {
                        _fail("native SIMD tan requires fixed f32 vectors");
                    }
                    return native;
                }
                return intrinsic(::llvm::Intrinsic::tan, {value});
            });
        case xir::ArithmeticOp::COSH:
            return unary_intrinsic(::llvm::Intrinsic::cosh);
        case xir::ArithmeticOp::SINH:
            return unary_intrinsic(::llvm::Intrinsic::sinh);
        case xir::ArithmeticOp::TANH:
            return unary_intrinsic(::llvm::Intrinsic::tanh);
        case xir::ArithmeticOp::EXP:
            return unary([&](::llvm::Value *value, const Type *type)
                             -> ::llvm::Value * {
                if (varying && type->is_float32()) {
                    auto *safe = _builder.CreateSelect(
                        _active_mask, value,
                        float_constant_like(value, 0.0));
                    auto *native = cpu::LLVMNativeMath::emit_exp_f32(
                        _module, _builder, safe,
                        native_math_mode);
                    if (native == nullptr) {
                        _fail("native SIMD exp requires fixed f32 vectors");
                    }
                    return native;
                }
                return intrinsic(::llvm::Intrinsic::exp, {value});
            });
        case xir::ArithmeticOp::EXP2:
            return unary([&](::llvm::Value *value, const Type *type)
                             -> ::llvm::Value * {
                if (varying && type->is_float32()) {
                    auto *safe = _builder.CreateSelect(
                        _active_mask, value,
                        float_constant_like(value, 0.0));
                    auto *native = cpu::LLVMNativeMath::emit_exp2_f32(
                        _module, _builder, safe,
                        native_math_mode);
                    if (native == nullptr) {
                        _fail("native SIMD exp2 requires fixed f32 vectors");
                    }
                    return native;
                }
                return intrinsic(::llvm::Intrinsic::exp2, {value});
            });
        case xir::ArithmeticOp::EXP10:
            return unary([&](::llvm::Value *value, const Type *type)
                             -> ::llvm::Value * {
                if (varying && type->is_float32()) {
                    auto *safe = _builder.CreateSelect(
                        _active_mask, value,
                        float_constant_like(value, 0.0));
                    auto *native = cpu::LLVMNativeMath::emit_exp10_f32(
                        _module, _builder, safe,
                        native_math_mode);
                    if (native == nullptr) {
                        _fail("native SIMD exp10 requires fixed f32 vectors");
                    }
                    return native;
                }
                return intrinsic(::llvm::Intrinsic::exp10, {value});
            });
        case xir::ArithmeticOp::LOG:
            return unary([&](::llvm::Value *value, const Type *type)
                             -> ::llvm::Value * {
                if (varying && type->is_float32()) {
                    auto *safe = _builder.CreateSelect(
                        _active_mask, value,
                        float_constant_like(value, 1.0));
                    auto *native = cpu::LLVMNativeMath::emit_log_f32(
                        _module, _builder, safe,
                        native_math_mode);
                    if (native == nullptr) {
                        _fail("native SIMD log requires fixed f32 vectors");
                    }
                    return native;
                }
                return intrinsic(::llvm::Intrinsic::log, {value});
            });
        case xir::ArithmeticOp::LOG2:
            return unary([&](::llvm::Value *value, const Type *type)
                             -> ::llvm::Value * {
                if (varying && type->is_float32()) {
                    auto *safe = _builder.CreateSelect(
                        _active_mask, value,
                        float_constant_like(value, 1.0));
                    auto *native = cpu::LLVMNativeMath::emit_log2_f32(
                        _module, _builder, safe,
                        native_math_mode);
                    if (native == nullptr) {
                        _fail("native SIMD log2 requires fixed f32 vectors");
                    }
                    return native;
                }
                return intrinsic(::llvm::Intrinsic::log2, {value});
            });
        case xir::ArithmeticOp::LOG10:
            return unary([&](::llvm::Value *value, const Type *type)
                             -> ::llvm::Value * {
                if (varying && type->is_float32()) {
                    auto *safe = _builder.CreateSelect(
                        _active_mask, value,
                        float_constant_like(value, 1.0));
                    auto *native = cpu::LLVMNativeMath::emit_log10_f32(
                        _module, _builder, safe,
                        native_math_mode);
                    if (native == nullptr) {
                        _fail("native SIMD log10 requires fixed f32 vectors");
                    }
                    return native;
                }
                return intrinsic(::llvm::Intrinsic::log10, {value});
            });
        case xir::ArithmeticOp::POW:
            return binary_intrinsic(::llvm::Intrinsic::pow);
        case xir::ArithmeticOp::SQRT:
            return unary_intrinsic(::llvm::Intrinsic::sqrt);
        case xir::ArithmeticOp::RSQRT:
            return unary([&](::llvm::Value *value, const Type *) {
                auto *root = intrinsic(
                    ::llvm::Intrinsic::sqrt, {value});
                return _builder.CreateFDiv(
                    float_constant_like(value, 1.0), root);
            });
        case xir::ArithmeticOp::CEIL:
            return unary_intrinsic(::llvm::Intrinsic::ceil);
        case xir::ArithmeticOp::FLOOR:
            return unary_intrinsic(::llvm::Intrinsic::floor);
        case xir::ArithmeticOp::TRUNC:
            return unary_intrinsic(::llvm::Intrinsic::trunc);
        case xir::ArithmeticOp::ROUND:
            return unary_intrinsic(::llvm::Intrinsic::round);
        case xir::ArithmeticOp::RINT:
            return unary_intrinsic(::llvm::Intrinsic::rint);
        case xir::ArithmeticOp::FRACT:
            return unary([&](::llvm::Value *value, const Type *) {
                return _builder.CreateFSub(
                    value, intrinsic(
                               ::llvm::Intrinsic::floor, {value}));
            });
        case xir::ArithmeticOp::FMA:
            return ternary(
                [&](::llvm::Value *a, ::llvm::Value *b,
                    ::llvm::Value *c, const Type *,
                    const Type *, const Type *) {
                    return intrinsic(
                        ::llvm::Intrinsic::fma, {a, b, c});
                });
        case xir::ArithmeticOp::COPYSIGN:
            return binary_intrinsic(::llvm::Intrinsic::copysign);
        case xir::ArithmeticOp::DOT:
            if (!require(2u)) { return nullptr; }
            return dot(
                operands[0u], operands[1u], operand_types[0u]);
        case xir::ArithmeticOp::LENGTH_SQUARED:
            if (!require(1u)) { return nullptr; }
            return dot(
                operands[0u], operands[0u], operand_types[0u]);
        case xir::ArithmeticOp::LENGTH: {
            if (!require(1u)) { return nullptr; }
            auto *squared = dot(
                operands[0u], operands[0u], operand_types[0u]);
            return squared == nullptr ? nullptr :
                                        intrinsic(::llvm::Intrinsic::sqrt, {squared});
        }
        case xir::ArithmeticOp::NORMALIZE: {
            if (!require(1u) || !operand_types[0u]->is_vector()) {
                return nullptr;
            }
            auto *squared = dot(
                operands[0u], operands[0u], operand_types[0u]);
            if (squared == nullptr) { return nullptr; }
            auto *length = intrinsic(
                ::llvm::Intrinsic::sqrt, {squared});
            return _componentwise_unary(
                result->type, operands[0u], operand_types[0u],
                varying,
                [&](::llvm::Value *value, const Type *) {
                    return _builder.CreateFDiv(value, length);
                });
        }
        case xir::ArithmeticOp::CROSS: {
            if (!require(2u) || !result->type->is_vector() ||
                result->type->dimension() != 3u) {
                return nullptr;
            }
            std::array<::llvm::Value *, 3u> a{};
            std::array<::llvm::Value *, 3u> b{};
            for (auto i = uint32_t{0u}; i < 3u; i++) {
                a[i] = _extract_child(
                    operands[0u], operand_types[0u], i, varying);
                b[i] = _extract_child(
                    operands[1u], operand_types[1u], i, varying);
            }
            return _assemble(result->type, varying, [&](uint32_t i) {
                auto j = (i + 1u) % 3u;
                auto k = (i + 2u) % 3u;
                return _builder.CreateFSub(
                    _builder.CreateFMul(a[j], b[k]),
                    _builder.CreateFMul(a[k], b[j]));
            });
        }
        default:
            _fail("LLVM packet codegen does not implement arithmetic operation '" +
                  std::string{xir::to_string(op)} + "' yet");
            return nullptr;
    }
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_cast(
    const schedule::Instruction &instruction) {
    if (!instruction.result || !instruction.source_op ||
        instruction.operands.size() != 1u) {
        _fail("cast instruction is malformed");
        return nullptr;
    }
    auto *result = _source.value(*instruction.result);
    auto *source = _source.value(instruction.operands.front());
    auto *value = _load_value(instruction.operands.front());
    if (result == nullptr || source == nullptr || value == nullptr ||
        !_is_data(result->type) || !_is_data(source->type)) {
        _fail("cast requires supported data types");
        return nullptr;
    }
    auto varying = result->value_class == schedule::ValueClass::varying;
    if (varying) { value = _as_lane_vector(value, *source); }
    auto op = static_cast<xir::CastOp>(*instruction.source_op);
    return _componentwise_unary(
        result->type, value, source->type, varying,
        [&](::llvm::Value *scalar, const Type *source_type) {
            auto *destination_type = result->type;
            while (!_is_scalar_data(destination_type)) {
                destination_type = destination_type->element();
            }
            if (op == xir::CastOp::BITWISE_CAST) {
                return _builder.CreateBitCast(
                    scalar,
                    scalar->getType()->isVectorTy() ?
                        ::llvm::FixedVectorType::get(
                            _data_type(destination_type, false),
                            _width) :
                        _data_type(destination_type, false));
            }
            auto destination_is_float =
                destination_type->is_float16() ||
                destination_type->is_float32() ||
                destination_type->is_float64();
            auto source_is_float = source_type->is_float16() ||
                                   source_type->is_float32() ||
                                   source_type->is_float64();
            auto *destination = scalar->getType()->isVectorTy() ?
                                    static_cast<::llvm::Type *>(::llvm::FixedVectorType::get(
                                        _data_type(destination_type, false), _width)) :
                                    _data_type(destination_type, false);
            if (destination_type->is_bool()) {
                auto *zero = ::llvm::Constant::getNullValue(
                    scalar->getType());
                return source_is_float ?
                           _builder.CreateFCmpUNE(scalar, zero) :
                           _builder.CreateICmpNE(scalar, zero);
            }
            if (source_type->is_bool()) {
                return destination_is_float ?
                           _builder.CreateUIToFP(scalar, destination) :
                           _builder.CreateZExtOrTrunc(scalar, destination);
            }
            if (source_is_float && destination_is_float) {
                return _builder.CreateFPCast(scalar, destination);
            }
            if (source_is_float) {
                return destination_type->is_int() ?
                           _builder.CreateFPToSI(scalar, destination) :
                           _builder.CreateFPToUI(scalar, destination);
            }
            if (destination_is_float) {
                return source_type->is_int() ?
                           _builder.CreateSIToFP(scalar, destination) :
                           _builder.CreateUIToFP(scalar, destination);
            }
            return source_type->is_int() ?
                       _builder.CreateSExtOrTrunc(scalar, destination) :
                       _builder.CreateZExtOrTrunc(scalar, destination);
        });
}

}// namespace luisa::compute::simd::detail
