#include "llvm_schedule_emitter.h"

#include "../../common/llvm_native_math.h"
#include <llvm/IR/MDBuilder.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>

namespace luisa::compute::simd::detail {

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
    // Intrinsics with more than one overload type (llvm.ldexp is
    // `llvm_anyfloat_ty (LLVMMatchType<0>, llvm_anyint_ty)`) must be
    // requested with *all* of their overload types: Intrinsic::getType
    // resolves each overloaded parameter through
    // Tys[D.getArgumentNumber()], so a one-element overload array is an
    // out-of-bounds read for the second overloaded type.
    auto intrinsic_with_overloads = [&](
                                            ::llvm::Intrinsic::ID id,
                                            ::llvm::ArrayRef<::llvm::Type *> overloads,
                                            ::llvm::ArrayRef<::llvm::Value *> args) {
#if LLVM_VERSION_MAJOR >= 22
            auto *function = ::llvm::Intrinsic::getOrInsertDeclaration(
#else
            auto *function = ::llvm::Intrinsic::getDeclaration(
#endif
                &_module, id, overloads);
            return _builder.CreateCall(function, args);
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
    auto scalar_libm = [&](luisa::string_view operation,
                           ::llvm::Value *value) -> ::llvm::Value * {
        auto *source_type = value->getType();
        auto half = source_type->isHalfTy();
        auto *call_type = half ? _builder.getFloatTy() : source_type;
        if (!call_type->isFloatTy() && !call_type->isDoubleTy()) {
            _fail("scalar libm operation requires a floating-point scalar");
            return nullptr;
        }
        auto name = std::string{operation} +
                    (call_type->isFloatTy() ? "f" : "");
        auto callee = _module.getOrInsertFunction(
            name, ::llvm::FunctionType::get(
                      call_type, {call_type}, false));
        if (auto *function = ::llvm::dyn_cast<::llvm::Function>(
                callee.getCallee())) {
            function->setDoesNotAccessMemory();
            function->setDoesNotThrow();
            function->setWillReturn();
        }
        auto *argument = half ?
                             _builder.CreateFPExt(value, call_type) :
                             value;
        auto *call = _builder.CreateCall(callee, {argument});
        return half ? _builder.CreateFPTrunc(call, source_type) : call;
    };
    auto float_constant_like = [&](::llvm::Value *value, double x) -> ::llvm::Constant * {
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
    auto matrix_element = [&](::llvm::Value *matrix,
                              const Type *matrix_type,
                              uint32_t column,
                              uint32_t row) -> ::llvm::Value * {
        auto *column_type = _child_type(matrix_type, column);
        auto *column_value = _extract_child(
            matrix, matrix_type, column, varying);
        return _extract_child(
            column_value, column_type, row, varying);
    };
    auto determinant_minor =
        [&](auto &&self, ::llvm::Value *matrix,
            const Type *matrix_type, uint32_t row_mask,
            uint32_t column_mask,
            std::array<::llvm::Value *, 256u> &cache)
        -> ::llvm::Value * {
        auto key = (row_mask << 4u) | column_mask;
        if (cache[key] != nullptr) { return cache[key]; }
        auto first_row = uint32_t{0u};
        while ((row_mask & (1u << first_row)) == 0u) {
            first_row++;
        }
        if ((row_mask & (row_mask - 1u)) == 0u) {
            auto first_column = uint32_t{0u};
            while ((column_mask & (1u << first_column)) == 0u) {
                first_column++;
            }
            return cache[key] = matrix_element(
                       matrix, matrix_type, first_column, first_row);
        }
        auto remaining_rows = row_mask & ~(1u << first_row);
        auto term_index = uint32_t{0u};
        ::llvm::Value *sum = nullptr;
        for (auto column = uint32_t{0u};
             column < matrix_type->dimension(); column++) {
            if ((column_mask & (1u << column)) == 0u) { continue; }
            auto *minor = self(
                self, matrix, matrix_type, remaining_rows,
                column_mask & ~(1u << column), cache);
            auto *term = _builder.CreateFMul(
                matrix_element(
                    matrix, matrix_type, column, first_row),
                minor);
            if (sum == nullptr) {
                sum = (term_index & 1u) == 0u ?
                          term :
                          _builder.CreateFNeg(term);
            } else {
                sum = (term_index & 1u) == 0u ?
                          _builder.CreateFAdd(sum, term) :
                          _builder.CreateFSub(sum, term);
            }
            term_index++;
        }
        return cache[key] = sum;
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
#if LLVM_VERSION_MAJOR >= 23
                if (varying) {
                    auto *value = intrinsic(
                        is_signed ? ::llvm::Intrinsic::masked_sdiv :
                                    ::llvm::Intrinsic::masked_udiv,
                        {lhs, rhs, _active_mask});
                    return sanitize_inactive_integer(value, 0u);
                }
#endif
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
                if (is_float) { return _emit_float_fmod(lhs, rhs, varying); }
#if LLVM_VERSION_MAJOR >= 23
                if (varying) {
                    auto *value = intrinsic(
                        is_signed ? ::llvm::Intrinsic::masked_srem :
                                    ::llvm::Intrinsic::masked_urem,
                        {lhs, rhs, _active_mask});
                    return sanitize_inactive_integer(value, 0u);
                }
#endif
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
            case xir::ArithmeticOp::BINARY_SHIFT_RIGHT: {
                auto is_safe_constant_shift = [&](::llvm::Value *shift_val) noexcept {
                    auto *c = ::llvm::dyn_cast<::llvm::Constant>(shift_val);
                    if (c == nullptr) { return false; }
                    auto *ci = ::llvm::dyn_cast<::llvm::ConstantInt>(c);
                    if (ci == nullptr) {
                        ci = ::llvm::dyn_cast_or_null<::llvm::ConstantInt>(
                            c->getSplatValue());
                    }
                    if (ci == nullptr) { return false; }
                    auto bits = shift_val->getType()->getScalarSizeInBits();
                    return ci->getZExtValue() < bits;
                };
                if (!is_safe_constant_shift(rhs)) {
                    rhs = sanitize_inactive_integer(rhs, 0u);
                }
                if (op == xir::ArithmeticOp::BINARY_SHIFT_LEFT) {
                    return _builder.CreateShl(lhs, rhs);
                }
                return is_signed ? _builder.CreateAShr(lhs, rhs) :
                                   _builder.CreateLShr(lhs, rhs);
            }
            case xir::ArithmeticOp::BINARY_ROTATE_LEFT:
            case xir::ArithmeticOp::BINARY_ROTATE_RIGHT:
                return intrinsic(
                    op == xir::ArithmeticOp::BINARY_ROTATE_LEFT ?
                        ::llvm::Intrinsic::fshl :
                        ::llvm::Intrinsic::fshr,
                    {lhs, lhs, rhs});
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
        case xir::ArithmeticOp::ALL:
        case xir::ArithmeticOp::ANY: {
            if (!require(1u) || !result->type->is_bool() ||
                !operand_types[0u]->is_bool_vector()) {
                _fail("all/any requires one boolean-vector operand and a boolean result");
                return nullptr;
            }
            auto combine = [&](::llvm::Value *lhs,
                               ::llvm::Value *rhs) {
                return op == xir::ArithmeticOp::ALL ?
                           _builder.CreateAnd(lhs, rhs) :
                           _builder.CreateOr(lhs, rhs);
            };
            if (!varying) {
                return op == xir::ArithmeticOp::ALL ?
                           _builder.CreateAndReduce(operands[0u]) :
                           _builder.CreateOrReduce(operands[0u]);
            }
            // Varying aggregates use an SoA layout: each boolean component
            // is a <W x i1>. Reduce components here while preserving the W
            // physical lanes; LLVM's vector-reduce intrinsics would instead
            // collapse distinct logical threads into one scalar.
            auto *reduced = _extract_child(
                operands[0u], operand_types[0u], 0u, true);
            for (auto i = uint32_t{1u};
                 i < operand_types[0u]->dimension(); i++) {
                reduced = combine(
                    reduced,
                    _extract_child(
                        operands[0u], operand_types[0u], i, true));
            }
            return reduced;
        }
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
        case xir::ArithmeticOp::BINARY_ROTATE_LEFT:
        case xir::ArithmeticOp::BINARY_ROTATE_RIGHT:
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
        case xir::ArithmeticOp::SMOOTHSTEP:
            return ternary(
                [&](::llvm::Value *edge0, ::llvm::Value *edge1,
                    ::llvm::Value *x, const Type *,
                    const Type *, const Type *x_type) {
                    auto *zero = float_constant_like(x, 0.0);
                    auto *one = float_constant_like(x, 1.0);
                    auto *minus_two = float_constant_like(x, -2.0);
                    auto *three = float_constant_like(x, 3.0);
                    auto *t = _builder.CreateFDiv(
                        _builder.CreateFSub(x, edge0),
                        _builder.CreateFSub(edge1, edge0));
                    t = minmax_leaf(t, zero, x_type, true);
                    t = minmax_leaf(t, one, x_type, false);
                    auto *tt = _builder.CreateFMul(t, t);
                    auto *polynomial = intrinsic(
                        ::llvm::Intrinsic::fma,
                        {minus_two, t, three});
                    return _builder.CreateFMul(tt, polynomial);
                });
        case xir::ArithmeticOp::STEP:
            return binary(
                [&](::llvm::Value *edge, ::llvm::Value *x,
                    const Type *, const Type *) {
                    auto *less = _builder.CreateFCmpOLT(x, edge);
                    return _builder.CreateSelect(
                        less,
                        float_constant_like(x, 0.0),
                        float_constant_like(x, 1.0));
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
        case xir::ArithmeticOp::CLZ:
        case xir::ArithmeticOp::CTZ:
            return unary([&](::llvm::Value *value, const Type *) {
                value = sanitize_inactive_integer(value, 0u);
                return intrinsic(
                    op == xir::ArithmeticOp::CLZ ?
                        ::llvm::Intrinsic::ctlz :
                        ::llvm::Intrinsic::cttz,
                    {value, _builder.getFalse()});
            });
        case xir::ArithmeticOp::POPCOUNT:
            return unary([&](::llvm::Value *value, const Type *) {
                return intrinsic(
                    ::llvm::Intrinsic::ctpop,
                    {sanitize_inactive_integer(value, 0u)});
            });
        case xir::ArithmeticOp::REVERSE:
            return unary([&](::llvm::Value *value, const Type *) {
                return intrinsic(
                    ::llvm::Intrinsic::bitreverse,
                    {sanitize_inactive_integer(value, 0u)});
            });
        case xir::ArithmeticOp::ISNAN:
            return unary([&](::llvm::Value *value, const Type *) {
                return _builder.CreateFCmpUNO(value, value);
            });
        case xir::ArithmeticOp::ISINF:
            return unary([&](::llvm::Value *value, const Type *) {
                auto *absolute = intrinsic(
                    ::llvm::Intrinsic::fabs, {value});
                ::llvm::Constant *infinity = ::llvm::ConstantFP::getInfinity(
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
            return binary([&](::llvm::Value *y, ::llvm::Value *x,
                              const Type *y_type, const Type *)
                              -> ::llvm::Value * {
                if (varying && y_type->is_float32()) {
                    auto *safe_y = _builder.CreateSelect(
                        _active_mask, y,
                        float_constant_like(y, 0.0));
                    auto *safe_x = _builder.CreateSelect(
                        _active_mask, x,
                        float_constant_like(x, 1.0));
                    auto *native = cpu::LLVMNativeMath::emit_atan2_f32(
                        _module, _builder, safe_y, safe_x,
                        native_math_mode);
                    if (native == nullptr) {
                        _fail("native SIMD atan2 requires fixed f32 vectors");
                    }
                    return native;
                }
                return intrinsic(::llvm::Intrinsic::atan2, {y, x});
            });
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
            return unary([&](::llvm::Value *value, const Type *type)
                             -> ::llvm::Value * {
                if (varying && type->is_float32()) {
                    auto *safe = _builder.CreateSelect(
                        _active_mask, value,
                        float_constant_like(value, 0.0));
                    auto *native = cpu::LLVMNativeMath::emit_cosh_f32(
                        _module, _builder, safe, native_math_mode);
                    if (native == nullptr) {
                        _fail("native SIMD cosh requires fixed f32 vectors");
                    }
                    return native;
                }
                return intrinsic(::llvm::Intrinsic::cosh, {value});
            });
        case xir::ArithmeticOp::SINH:
            return unary([&](::llvm::Value *value, const Type *type)
                             -> ::llvm::Value * {
                if (varying && type->is_float32()) {
                    auto *safe = _builder.CreateSelect(
                        _active_mask, value,
                        float_constant_like(value, 0.0));
                    auto *native = cpu::LLVMNativeMath::emit_sinh_f32(
                        _module, _builder, safe, native_math_mode);
                    if (native == nullptr) {
                        _fail("native SIMD sinh requires fixed f32 vectors");
                    }
                    return native;
                }
                return intrinsic(::llvm::Intrinsic::sinh, {value});
            });
        case xir::ArithmeticOp::TANH:
            return unary([&](::llvm::Value *value, const Type *type)
                             -> ::llvm::Value * {
                if (varying && type->is_float32()) {
                    auto *safe = _builder.CreateSelect(
                        _active_mask, value,
                        float_constant_like(value, 0.0));
                    auto *native = cpu::LLVMNativeMath::emit_tanh_f32(
                        _module, _builder, safe, native_math_mode);
                    if (native == nullptr) {
                        _fail("native SIMD tanh requires fixed f32 vectors");
                    }
                    return native;
                }
                return intrinsic(::llvm::Intrinsic::tanh, {value});
            });
        case xir::ArithmeticOp::ASINH:
            return unary([&](::llvm::Value *value, const Type *type)
                             -> ::llvm::Value * {
                if (varying && type->is_float32()) {
                    auto *safe = _builder.CreateSelect(
                        _active_mask, value,
                        float_constant_like(value, 0.0));
                    auto *native = cpu::LLVMNativeMath::emit_asinh_f32(
                        _module, _builder, safe, native_math_mode);
                    if (native == nullptr) {
                        _fail("native SIMD asinh requires fixed f32 vectors");
                    }
                    return native;
                }
                return scalar_libm("asinh", value);
            });
        case xir::ArithmeticOp::ACOSH:
            return unary([&](::llvm::Value *value, const Type *type)
                             -> ::llvm::Value * {
                if (varying && type->is_float32()) {
                    auto *safe = _builder.CreateSelect(
                        _active_mask, value,
                        float_constant_like(value, 1.0));
                    auto *native = cpu::LLVMNativeMath::emit_acosh_f32(
                        _module, _builder, safe, native_math_mode);
                    if (native == nullptr) {
                        _fail("native SIMD acosh requires fixed f32 vectors");
                    }
                    return native;
                }
                return scalar_libm("acosh", value);
            });
        case xir::ArithmeticOp::ATANH:
            return unary([&](::llvm::Value *value, const Type *type)
                             -> ::llvm::Value * {
                if (varying && type->is_float32()) {
                    auto *safe = _builder.CreateSelect(
                        _active_mask, value,
                        float_constant_like(value, 0.0));
                    auto *native = cpu::LLVMNativeMath::emit_atanh_f32(
                        _module, _builder, safe, native_math_mode);
                    if (native == nullptr) {
                        _fail("native SIMD atanh requires fixed f32 vectors");
                    }
                    return native;
                }
                return scalar_libm("atanh", value);
            });
        case xir::ArithmeticOp::ERF:
            // LLVM has no llvm.erf intrinsic, so this always takes
            // the C library entry point (there is no native f32 body for
            // it in LLVMNativeMath).
            return unary([&](::llvm::Value *value, const Type *)
                             -> ::llvm::Value * {
                return scalar_libm("erf", value);
            });
        case xir::ArithmeticOp::ERFC:
            // LLVM has no llvm.erfc intrinsic, so this always takes
            // the C library entry point (there is no native f32 body for
            // it in LLVMNativeMath).
            return unary([&](::llvm::Value *value, const Type *)
                             -> ::llvm::Value * {
                return scalar_libm("erfc", value);
            });
        case xir::ArithmeticOp::EXPM1:
            // LLVM has no llvm.expm1 intrinsic, so this always takes
            // the C library entry point (there is no native f32 body for
            // it in LLVMNativeMath).
            return unary([&](::llvm::Value *value, const Type *)
                             -> ::llvm::Value * {
                return scalar_libm("expm1", value);
            });
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
            return binary([&](::llvm::Value *base,
                              ::llvm::Value *exponent,
                              const Type *base_type, const Type *)
                              -> ::llvm::Value * {
                if (auto *c = ::llvm::dyn_cast<::llvm::Constant>(exponent)) {
                    auto *cfp = ::llvm::dyn_cast<::llvm::ConstantFP>(c);
                    if (cfp == nullptr) {
                        cfp = ::llvm::dyn_cast_or_null<::llvm::ConstantFP>(
                            c->getSplatValue());
                    }
                    if (cfp != nullptr) {
                        const double exp_val = cfp->getValueAPF().convertToDouble();
                        if (exp_val == 2.0) {
                            return _builder.CreateFMul(base, base);
                        }
                        if (exp_val == 3.0) {
                            return _builder.CreateFMul(_builder.CreateFMul(base, base), base);
                        }
                        if (exp_val == 4.0) {
                            auto *sq = _builder.CreateFMul(base, base);
                            return _builder.CreateFMul(sq, sq);
                        }
                        if (exp_val == 5.0) {
                            auto *sq = _builder.CreateFMul(base, base);
                            return _builder.CreateFMul(_builder.CreateFMul(sq, sq), base);
                        }
                        if (exp_val == 0.5 && _enable_fast_math) {
                            return intrinsic(::llvm::Intrinsic::sqrt, {base});
                        }
                    }
                }
                if (varying && base_type->is_float32()) {
                    // POW contains float-to-int conversion for its integer
                    // exponent classification. Neutralize inactive lanes
                    // before any operation that could otherwise create poison.
                    auto *safe_base = _builder.CreateSelect(
                        _active_mask, base,
                        float_constant_like(base, 1.0));
                    auto *safe_exponent = _builder.CreateSelect(
                        _active_mask, exponent,
                        float_constant_like(exponent, 0.0));
                    auto *native = cpu::LLVMNativeMath::emit_pow_f32(
                        _module, _builder, safe_base, safe_exponent,
                        native_math_mode);
                    if (native == nullptr) {
                        _fail("native SIMD pow requires fixed f32 vectors");
                    }
                    return native;
                }
                return intrinsic(
                    ::llvm::Intrinsic::pow, {base, exponent});
            });
        case xir::ArithmeticOp::POW_INT:
            return binary([&](::llvm::Value *base,
                              ::llvm::Value *exponent,
                              const Type *, const Type *exponent_type)
                              -> ::llvm::Value * {
                if (varying) {
                    // A parked physical lane has no semantic exponent. Make
                    // it the exact multiplicative identity before entering
                    // the vector loop, including the INT_MIN magnitude path.
                    base = _builder.CreateSelect(
                        _active_mask, base,
                        float_constant_like(base, 1.0));
                    exponent = _builder.CreateSelect(
                        _active_mask, exponent,
                        integer_constant_like(exponent, 0u));
                }
                auto *base_llvm_type = base->getType();
                auto *exponent_llvm_type = exponent->getType();
                auto *base_scalar_type = base_llvm_type->getScalarType();
                auto *exponent_scalar_type =
                    ::llvm::dyn_cast<::llvm::IntegerType>(
                        exponent_llvm_type->getScalarType());
                if (!base_scalar_type->isFloatingPointTy() ||
                    exponent_scalar_type == nullptr ||
                    exponent_scalar_type->isIntegerTy(1)) {
                    _fail("pow_int requires floating-point bases and non-boolean integer exponents");
                    return nullptr;
                }
                auto vector_width = uint32_t{0u};
                if (auto *vector = ::llvm::dyn_cast<::llvm::FixedVectorType>(
                        base_llvm_type)) {
                    vector_width = vector->getNumElements();
                }
                auto base_code = base_scalar_type->isHalfTy()  ? "f16" :
                                 base_scalar_type->isFloatTy() ? "f32" :
                                                                 "f64";
                auto helper_name = std::string{"luisa.simd.pow_int."} +
                                   (exponent_type->is_int() ? "s" : "u") +
                                   std::to_string(
                                       exponent_scalar_type->getBitWidth()) +
                                   "." + base_code +
                                   (vector_width == 0u ? ".scalar" :
                                                         ".v" + std::to_string(vector_width));
                auto *helper = _module.getFunction(helper_name);
                if (helper == nullptr) {
                    auto *helper_type = ::llvm::FunctionType::get(
                        base_llvm_type,
                        {base_llvm_type, exponent_llvm_type}, false);
                    helper = ::llvm::Function::Create(
                        helper_type, ::llvm::Function::PrivateLinkage,
                        helper_name, &_module);
                    helper->addFnAttr(::llvm::Attribute::AlwaysInline);
                    helper->setDoesNotAccessMemory();
                    helper->setDoesNotThrow();
                    helper->setWillReturn();

                    auto &context = _module.getContext();
                    auto *entry = ::llvm::BasicBlock::Create(
                        context, "entry", helper);
                    auto *loop = ::llvm::BasicBlock::Create(
                        context, "loop", helper);
                    auto *body = ::llvm::BasicBlock::Create(
                        context, "body", helper);
                    auto *exit = ::llvm::BasicBlock::Create(
                        context, "exit", helper);
                    ::llvm::IRBuilder<> helper_builder{entry};
                    helper_builder.setFastMathFlags(
                        _builder.getFastMathFlags());
                    auto *zero = ::llvm::Constant::getNullValue(
                        exponent_llvm_type);
                    auto *one = float_constant_like(
                        helper->getArg(0u), 1.0);
                    auto *condition_type = static_cast<::llvm::Type *>(
                        helper_builder.getInt1Ty());
                    if (vector_width != 0u) {
                        condition_type = ::llvm::FixedVectorType::get(
                            helper_builder.getInt1Ty(), vector_width);
                    }
                    auto *negative = exponent_type->is_int() ?
                                         helper_builder.CreateICmpSLT(
                                             helper->getArg(1u), zero) :
                                         ::llvm::Constant::getNullValue(
                                             condition_type);
                    // Unsigned two's-complement magnitude is defined even
                    // for the most-negative signed exponent; do not attach
                    // NSW to this subtraction.
                    auto *magnitude = exponent_type->is_int() ?
                                          helper_builder.CreateSelect(
                                              negative,
                                              helper_builder.CreateSub(
                                                  zero, helper->getArg(1u)),
                                              helper->getArg(1u)) :
                                          helper->getArg(1u);
                    auto *reciprocal = helper_builder.CreateFDiv(
                        one, helper->getArg(0u));
                    auto *factor = helper_builder.CreateSelect(
                        negative, reciprocal, helper->getArg(0u));
                    helper_builder.CreateBr(loop);

                    helper_builder.SetInsertPoint(loop);
                    auto *current_magnitude = helper_builder.CreatePHI(
                        exponent_llvm_type, 2u, "magnitude");
                    auto *current_result = helper_builder.CreatePHI(
                        base_llvm_type, 2u, "result");
                    auto *current_factor = helper_builder.CreatePHI(
                        base_llvm_type, 2u, "factor");
                    current_magnitude->addIncoming(magnitude, entry);
                    current_result->addIncoming(one, entry);
                    current_factor->addIncoming(factor, entry);
                    auto *nonzero = helper_builder.CreateICmpNE(
                        current_magnitude, zero);
                    auto *any_nonzero = vector_width == 0u ?
                                            nonzero :
                                            helper_builder.CreateOrReduce(
                                                nonzero);
                    helper_builder.CreateCondBr(
                        any_nonzero, body, exit);

                    helper_builder.SetInsertPoint(body);
                    auto *odd = helper_builder.CreateTrunc(
                        current_magnitude, condition_type);
                    auto *multiplied = helper_builder.CreateFMul(
                        current_result, current_factor);
                    auto *next_result = helper_builder.CreateSelect(
                        odd, multiplied, current_result);
                    auto *next_magnitude = helper_builder.CreateLShr(
                        current_magnitude, 1u);
                    auto *next_factor = helper_builder.CreateFMul(
                        current_factor, current_factor);
                    helper_builder.CreateBr(loop);
                    current_magnitude->addIncoming(
                        next_magnitude, body);
                    current_result->addIncoming(next_result, body);
                    current_factor->addIncoming(next_factor, body);

                    helper_builder.SetInsertPoint(exit);
                    helper_builder.CreateRet(current_result);
                }
                return _builder.CreateCall(helper, {base, exponent});
            });
        case xir::ArithmeticOp::LDEXP:
            return binary([&](::llvm::Value *base, ::llvm::Value *exponent,
                              const Type *, const Type *exponent_type_xir)
                              -> ::llvm::Value * {
                // llvm.ldexp takes an anyfloat mantissa and an anyint exponent
                // whose element count must match the mantissa, and every target
                // lowers it to the 32-bit libm entry point. Broadcast a scalar
                // exponent and narrow it to i32 here.
                auto *base_type = base->getType();
                if (auto *vector_type = ::llvm::dyn_cast<
                        ::llvm::FixedVectorType>(base_type)) {
                    if (!exponent->getType()->isVectorTy()) {
                        exponent = _builder.CreateVectorSplat(
                            vector_type->getNumElements(), exponent);
                    }
                }
                // Signedness contract: LLVM IR drops the XIR signedness, so it is
                // recovered from the operand type. A signed exponent is
                // sign-extended and clamped to [INT32_MIN, INT32_MAX]; an unsigned
                // exponent is zero-extended and clamped to INT32_MAX only, because
                // an unsigned exponent above INT32_MAX already saturates the
                // mantissa to zero or +inf, so keeping its value stays exact while
                // sign-extending it would turn a huge exponent into a hugely
                // negative one.
                auto *exponent_element_type = exponent_type_xir->is_vector() ?
                                                  exponent_type_xir->element() :
                                                  exponent_type_xir;
                auto exponent_is_signed = exponent_element_type->is_int();
                auto *i32_type = _builder.getInt32Ty();
                auto *exponent_scalar_type = exponent->getType()->getScalarType();
                        auto exponent_bits = exponent_scalar_type->getIntegerBitWidth();
                        auto *i32_exponent_type =
                            exponent->getType()->isVectorTy() ?
                                static_cast<::llvm::Type *>(
                                    ::llvm::FixedVectorType::get(
                                        i32_type,
                                        ::llvm::cast<::llvm::FixedVectorType>(
                                            exponent->getType())
                                            ->getNumElements())) :
                                static_cast<::llvm::Type *>(i32_type);
                        auto lanes = exponent->getType()->isVectorTy() ?
                                         ::llvm::cast<::llvm::FixedVectorType>(
                                             exponent->getType())
                                             ->getNumElements() :
                                         0u;
                        auto splat = [&](::llvm::Constant *constant) -> ::llvm::Value * {
                            return lanes == 0u ?
                                       static_cast<::llvm::Value *>(constant) :
                                       _builder.CreateVectorSplat(lanes, constant);
                        };
                        if (exponent_bits < 32u) {
                            exponent = exponent_is_signed ?
                                           _builder.CreateSExt(exponent, i32_exponent_type) :
                                           _builder.CreateZExt(exponent, i32_exponent_type);
                        } else if (!(exponent_is_signed && exponent_bits == 32u)) {
                            // Clamp in the source type *before* truncating: truncation
                            // alone would wrap an out-of-range exponent back into range.
                            auto *high = splat(::llvm::ConstantInt::get(
                                exponent_scalar_type, 0x7fffffffu));
                            exponent = exponent_is_signed ?
                                           _builder.CreateSelect(
                                               _builder.CreateICmpSGT(exponent, high),
                                               high, exponent) :
                                           _builder.CreateSelect(
                                               _builder.CreateICmpUGT(exponent, high),
                                               high, exponent);
                            if (exponent_is_signed) {
                                auto *low = splat(::llvm::ConstantInt::getSigned(
                                    exponent_scalar_type, -2147483648ll));
                                exponent = _builder.CreateSelect(
                                    _builder.CreateICmpSLT(exponent, low), low, exponent);
                            }
                            if (exponent_bits > 32u) {
                                exponent = _builder.CreateTrunc(exponent, i32_exponent_type);
                            }
                        }
                return intrinsic_with_overloads(
                    ::llvm::Intrinsic::ldexp,
                    {base_type, exponent->getType()}, {base, exponent});
            });
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
            return unary([&](::llvm::Value *value, const Type *type) {
                if (type->is_float32() || type->is_float64()) {
                    // LLVM lowers @llvm.round on x86 AVX2/AVX-512 vectors to
                    // scalar libm roundf/round calls. trunc(x + copysign(0.5 - ulp, x))
                    // is bit-exact IEEE round-half-away-from-zero for all float32/64
                    // values and lowers to 3 native SIMD instructions on x86 & AArch64.
                    ::llvm::IRBuilder<>::FastMathFlagGuard strict_guard{_builder};
                    _builder.clearFastMathFlags();
                    auto *half_down = type->is_float32() ?
                                          float_constant_like(value, 0x1.fffffep-2) :
                                          float_constant_like(value, 0x1.fffffffffffffp-2);
                    auto *signed_half = intrinsic(
                        ::llvm::Intrinsic::copysign, {half_down, value});
                    return intrinsic(
                        ::llvm::Intrinsic::trunc,
                        {_builder.CreateFAdd(value, signed_half)});
                }
                return intrinsic(::llvm::Intrinsic::round, {value});
            });
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
        case xir::ArithmeticOp::REDUCE_SUM:
        case xir::ArithmeticOp::REDUCE_PRODUCT:
        case xir::ArithmeticOp::REDUCE_MIN:
        case xir::ArithmeticOp::REDUCE_MAX: {
            if (!require(1u) || !operand_types[0u]->is_vector()) {
                _fail("vector reduction requires one vector operand");
                return nullptr;
            }
            auto *vector_type = operand_types[0u];
            auto *element_type = vector_type->element();
            auto *first = _extract_child(
                operands[0u], vector_type, 0u, varying);
            ::llvm::Value *reduced = nullptr;
            auto first_component = uint32_t{0u};
            if (op == xir::ArithmeticOp::REDUCE_MIN ||
                op == xir::ArithmeticOp::REDUCE_MAX) {
                reduced = first;
                first_component = 1u;
            } else if (element_type->is_float16() ||
                       element_type->is_float32() ||
                       element_type->is_float64()) {
                reduced = float_constant_like(
                    first,
                    op == xir::ArithmeticOp::REDUCE_SUM ?
                        -0.0 :
                        1.0);
            } else {
                reduced = integer_constant_like(
                    first,
                    op == xir::ArithmeticOp::REDUCE_SUM ?
                        0u :
                        1u);
            }
            for (auto i = first_component;
                 i < vector_type->dimension(); i++) {
                auto *component = _extract_child(
                    operands[0u], vector_type, i, varying);
                if (op == xir::ArithmeticOp::REDUCE_SUM) {
                    reduced = element_type->is_float16() ||
                                      element_type->is_float32() ||
                                      element_type->is_float64() ?
                                  _builder.CreateFAdd(
                                      reduced, component) :
                                  _builder.CreateAdd(
                                      reduced, component);
                } else if (op == xir::ArithmeticOp::REDUCE_PRODUCT) {
                    reduced = element_type->is_float16() ||
                                      element_type->is_float32() ||
                                      element_type->is_float64() ?
                                  _builder.CreateFMul(
                                      reduced, component) :
                                  _builder.CreateMul(
                                      reduced, component);
                } else {
                    reduced = minmax_leaf(
                        reduced, component, element_type,
                        op == xir::ArithmeticOp::REDUCE_MAX);
                }
            }
            return reduced;
        }
        case xir::ArithmeticOp::OUTER_PRODUCT: {
            if (!require(2u) || !result->type->is_matrix()) {
                _fail("outer product requires two vectors or two matrices");
                return nullptr;
            }
            auto *lhs_type = operand_types[0u];
            auto *rhs_type = operand_types[1u];
            auto dimension = result->type->dimension();
            if (lhs_type->dimension() != dimension ||
                rhs_type->dimension() != dimension ||
                lhs_type->is_matrix() != rhs_type->is_matrix()) {
                _fail("outer product operand dimensions do not match its result");
                return nullptr;
            }
            return _assemble(
                result->type, varying, [&](uint32_t column) {
                    auto *column_type = _child_type(
                        result->type, column);
                    return _assemble(
                        column_type, varying, [&](uint32_t row) {
                            if (lhs_type->is_vector()) {
                                return _builder.CreateFMul(
                                    _extract_child(
                                        operands[0u], lhs_type,
                                        row, varying),
                                    _extract_child(
                                        operands[1u], rhs_type,
                                        column, varying));
                            }
                            ::llvm::Value *sum = nullptr;
                            for (auto k = uint32_t{0u};
                                 k < dimension; k++) {
                                auto *term = _builder.CreateFMul(
                                    matrix_element(
                                        operands[0u], lhs_type,
                                        k, row),
                                    matrix_element(
                                        operands[1u], rhs_type,
                                        k, column));
                                sum = sum == nullptr ?
                                          term :
                                          _builder.CreateFAdd(sum, term);
                            }
                            return sum;
                        });
                });
        }
        case xir::ArithmeticOp::MATRIX_TRANSPOSE: {
            if (!require(1u) ||
                !operand_types[0u]->is_matrix() ||
                operand_types[0u] != result->type) {
                _fail("matrix transpose requires a same-type matrix result");
                return nullptr;
            }
            return _assemble(
                result->type, varying, [&](uint32_t column) {
                    auto *column_type = _child_type(
                        result->type, column);
                    return _assemble(
                        column_type, varying, [&](uint32_t row) {
                            return matrix_element(
                                operands[0u], operand_types[0u],
                                row, column);
                        });
                });
        }
        case xir::ArithmeticOp::MATRIX_DETERMINANT: {
            if (!require(1u) || !operand_types[0u]->is_matrix()) {
                _fail("matrix determinant requires one matrix operand");
                return nullptr;
            }
            auto dimension = operand_types[0u]->dimension();
            if (dimension < 2u || dimension > 4u) {
                _fail("matrix determinant supports dimensions two through four");
                return nullptr;
            }
            auto full_mask = (1u << dimension) - 1u;
            std::array<::llvm::Value *, 256u> cache{};
            return determinant_minor(
                determinant_minor, operands[0u], operand_types[0u],
                full_mask, full_mask, cache);
        }
        case xir::ArithmeticOp::MATRIX_INVERSE: {
            if (!require(1u) ||
                !operand_types[0u]->is_matrix() ||
                operand_types[0u] != result->type) {
                _fail("matrix inverse requires a same-type matrix result");
                return nullptr;
            }
            auto dimension = result->type->dimension();
            if (dimension < 2u || dimension > 4u) {
                _fail("matrix inverse supports dimensions two through four");
                return nullptr;
            }
            auto full_mask = (1u << dimension) - 1u;
            std::array<::llvm::Value *, 256u> cache{};
            auto *determinant = determinant_minor(
                determinant_minor, operands[0u], operand_types[0u],
                full_mask, full_mask, cache);
            auto *inverse_determinant = _builder.CreateFDiv(
                float_constant_like(determinant, 1.0), determinant);
            return _assemble(
                result->type, varying, [&](uint32_t column) {
                    auto *column_type = _child_type(
                        result->type, column);
                    return _assemble(
                        column_type, varying, [&](uint32_t row) {
                            // inverse[row][column] is the transposed
                            // cofactor C[column][row]. With column-major
                            // storage this excludes source row=column and
                            // source column=row.
                            auto *cofactor = determinant_minor(
                                determinant_minor,
                                operands[0u], operand_types[0u],
                                full_mask & ~(1u << column),
                                full_mask & ~(1u << row), cache);
                            if (((column + row) & 1u) != 0u) {
                                cofactor = _builder.CreateFNeg(cofactor);
                            }
                            return _builder.CreateFMul(
                                cofactor, inverse_determinant);
                        });
                });
        }
        case xir::ArithmeticOp::MATRIX_LINALG_MUL: {
            if (!require(2u)) { return nullptr; }
            auto *lhs_type = operand_types[0u];
            auto *rhs_type = operand_types[1u];
            auto lhs_matrix = lhs_type->is_matrix();
            auto rhs_matrix = rhs_type->is_matrix();
            auto lhs_vector = lhs_type->is_float_vector();
            auto rhs_vector = rhs_type->is_float_vector();
            if ((!lhs_matrix && !lhs_vector) ||
                (!rhs_matrix && !rhs_vector) ||
                (!lhs_matrix && !rhs_matrix) ||
                lhs_type->dimension() != rhs_type->dimension()) {
                _fail("matrix linear-algebra multiply has invalid operand types");
                return nullptr;
            }
            auto dimension = lhs_type->dimension();
            auto vector_component = [&](::llvm::Value *value,
                                        const Type *type,
                                        uint32_t index) {
                return _extract_child(
                    value, type, index, varying);
            };
            auto matrix_element = [&](::llvm::Value *value,
                                      const Type *type,
                                      uint32_t column,
                                      uint32_t row) {
                auto *column_type = _child_type(type, column);
                auto *column_value = _extract_child(
                    value, type, column, varying);
                return _extract_child(
                    column_value, column_type, row, varying);
            };
            auto sum_products = [&](auto &&term) {
                ::llvm::Value *sum = nullptr;
                for (auto k = uint32_t{0u}; k < dimension; k++) {
                    auto [a, b] = term(k);
                    auto *product = _builder.CreateFMul(a, b);
                    sum = sum == nullptr ? product :
                                           _builder.CreateFAdd(sum, product);
                }
                return sum;
            };
            if (lhs_matrix && rhs_vector) {
                if (!result->type->is_float_vector() ||
                    result->type->dimension() != dimension) {
                    _fail("matrix-vector multiply has an invalid result type");
                    return nullptr;
                }
                return _assemble(
                    result->type, varying, [&](uint32_t row) {
                        return sum_products([&](uint32_t k) {
                            return std::pair{
                                matrix_element(
                                    operands[0u], lhs_type, k, row),
                                vector_component(
                                    operands[1u], rhs_type, k)};
                        });
                    });
            }
            if (lhs_vector && rhs_matrix) {
                if (!result->type->is_float_vector() ||
                    result->type->dimension() != dimension) {
                    _fail("vector-matrix multiply has an invalid result type");
                    return nullptr;
                }
                return _assemble(
                    result->type, varying, [&](uint32_t column) {
                        return sum_products([&](uint32_t k) {
                            return std::pair{
                                vector_component(
                                    operands[0u], lhs_type, k),
                                matrix_element(
                                    operands[1u], rhs_type, column, k)};
                        });
                    });
            }
            if (!lhs_matrix || !rhs_matrix ||
                !result->type->is_matrix() ||
                result->type->dimension() != dimension) {
                _fail("matrix-matrix multiply has an invalid result type");
                return nullptr;
            }
            return _assemble(
                result->type, varying, [&](uint32_t column) {
                    auto *column_type = _child_type(
                        result->type, column);
                    return _assemble(
                        column_type, varying, [&](uint32_t row) {
                            return sum_products([&](uint32_t k) {
                                return std::pair{
                                    matrix_element(
                                        operands[0u], lhs_type, k, row),
                                    matrix_element(
                                        operands[1u], rhs_type, column, k)};
                            });
                        });
                });
        }
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
        case xir::ArithmeticOp::FACEFORWARD: {
            if (!require(3u) || !result->type->is_float_vector() ||
                operand_types[0u] != result->type ||
                operand_types[1u] != result->type ||
                operand_types[2u] != result->type) {
                return nullptr;
            }
            auto *orientation = dot(
                operands[1u], operands[2u], result->type);
            if (orientation == nullptr) { return nullptr; }
            auto *forward = _builder.CreateFCmpOLT(
                orientation,
                ::llvm::Constant::getNullValue(
                    orientation->getType()));
            return _assemble(result->type, varying, [&](uint32_t i) {
                auto *normal = _extract_child(
                    operands[0u], result->type, i, varying);
                return _builder.CreateSelect(
                    forward, normal, _builder.CreateFNeg(normal));
            });
        }
        case xir::ArithmeticOp::REFLECT: {
            if (!require(2u) || !result->type->is_float_vector() ||
                operand_types[0u] != result->type ||
                operand_types[1u] != result->type) {
                return nullptr;
            }
            auto *projection = dot(
                operands[1u], operands[0u], result->type);
            if (projection == nullptr) { return nullptr; }
            auto *scale = _builder.CreateFMul(
                float_constant_like(projection, -2.0), projection);
            return _assemble(result->type, varying, [&](uint32_t i) {
                auto *incident = _extract_child(
                    operands[0u], result->type, i, varying);
                auto *normal = _extract_child(
                    operands[1u], result->type, i, varying);
                return intrinsic(
                    ::llvm::Intrinsic::fma,
                    {scale, normal, incident});
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

}// namespace luisa::compute::simd::detail
