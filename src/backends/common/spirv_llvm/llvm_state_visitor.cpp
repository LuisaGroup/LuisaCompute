#include "llvm_state_visitor.h"
#include "llvm_codegen_utility.h"
#include "llvm_codegen_stack_data.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/InlineAsm.h>

namespace lc::llvm_codegen {

// ============================================================================
// LLVMStateVisitor — Construction
// ============================================================================

LLVMStateVisitor::LLVMStateVisitor(Function f, LLVMCodegenUtility &util)
    : f(f), _util(&util),
      _ctx(util.context()),
      _module(util.module()),
      _builder(util.builder()) {}

void LLVMStateVisitor::VisitFunction(Function func) {
    auto *llvm_func = _util->current_function();
    if (!llvm_func) {
        LUISA_ERROR_WITH_LOCATION("No current LLVM function set.");
    }
    _entry_block = &llvm_func->getEntryBlock();

    // Visit body
    auto *body = func.body();
    if (body) {
        visit(body);
    }
}

// ============================================================================
// Helpers
// ============================================================================

llvm::Value *LLVMStateVisitor::EvalExpr(Expression const *expr) {
    expr->accept(*this);
    return _last_value;
}

llvm::Type *LLVMStateVisitor::ToLLVMType(Type const &type) {
    return _util->ToLLVMType(type);
}

llvm::Value *LLVMStateVisitor::GetVariable(uint32_t uid, Type const *type) {
    auto it = _util->opt->variables.find(uid);
    if (it != _util->opt->variables.end()) {
        auto *alloca = it->second;
        // If it's an alloca (has a type), load from it
        if (llvm::isa<llvm::AllocaInst>(alloca)) {
            return _builder.CreateLoad(ToLLVMType(*type), alloca);
        }
        // Otherwise it's a direct value (e.g., function parameter for resources)
        return alloca;
    }
    LUISA_ERROR_WITH_LOCATION("Variable {} not found.", uid);
    return nullptr;
}

void LLVMStateVisitor::StoreVariable(uint32_t uid, llvm::Value *value) {
    auto it = _util->opt->variables.find(uid);
    if (it != _util->opt->variables.end()) {
        auto *alloca = it->second;
        if (llvm::isa<llvm::AllocaInst>(alloca)) {
            _builder.CreateStore(value, alloca);
            return;
        }
    }
    LUISA_ERROR_WITH_LOCATION("Cannot store to variable {}.", uid);
}

void LLVMStateVisitor::_push_loop(llvm::BasicBlock *break_target, llvm::BasicBlock *continue_target) {
    _util->opt->loop_stack.push_back({break_target, continue_target});
}

void LLVMStateVisitor::_pop_loop() {
    if (!_util->opt->loop_stack.empty()) {
        _util->opt->loop_stack.pop_back();
    }
}

// ============================================================================
// Expression Visitors
// ============================================================================

void LLVMStateVisitor::visit(const LiteralExpr *expr) {
    auto *type = ToLLVMType(*expr->type());

    // Dispatch based on the element type stored in the variant
    luisa::visit(
        [&]<typename T>(T const &value) -> void {
            if constexpr (std::is_same_v<T, bool>) {
                _last_value = llvm::ConstantInt::get(type, value ? 1ULL : 0ULL);
            } else if constexpr (std::is_same_v<T, float>) {
                _last_value = llvm::ConstantFP::get(type, static_cast<double>(value));
            } else if constexpr (std::is_same_v<T, double>) {
                _last_value = llvm::ConstantFP::get(type, value);
            } else if constexpr (std::is_same_v<T, half>) {
                uint16_t raw = *reinterpret_cast<uint16_t const *>(&value);
                _last_value = llvm::ConstantFP::get(type, llvm::APFloat(llvm::APFloat::IEEEhalf(), llvm::APInt(16, raw)));
            } else if constexpr (luisa::is_vector_v<T>) {
                // Vector literal: build ConstantVector from elements
                auto *vec_ty = llvm::cast<llvm::FixedVectorType>(type);
                auto *elem_ty = vec_ty->getElementType();
                auto n = vec_ty->getNumElements();
                luisa::vector<llvm::Constant *> elems;
                for (unsigned i = 0; i < n; ++i) {
                    auto elem = value[i];
                    if (elem_ty->isFloatTy() || elem_ty->isHalfTy() || elem_ty->isDoubleTy()) {
                        double d = 0.0;
                        if constexpr (std::is_same_v<decltype(elem), float>) d = static_cast<double>(elem);
                        else if constexpr (std::is_same_v<decltype(elem), double>) d = elem;
                        else if constexpr (std::is_same_v<decltype(elem), half>) d = static_cast<double>(elem);
                        elems.push_back(llvm::ConstantFP::get(elem_ty, d));
                    } else {
                        elems.push_back(llvm::ConstantInt::get(elem_ty, static_cast<uint64_t>(elem)));
                    }
                }
                _last_value = llvm::ConstantVector::get(elems);
            } else if constexpr (luisa::is_matrix_v<T>) {
                // Matrix literal: build ConstantArray from row vectors
                auto *mat_ty = llvm::cast<llvm::ArrayType>(type);
                auto *row_ty = llvm::cast<llvm::FixedVectorType>(mat_ty->getElementType());
                auto *float_ty = row_ty->getElementType();
                auto n = row_ty->getNumElements();
                luisa::vector<llvm::Constant *> rows;
                for (unsigned i = 0; i < n; ++i) {
                    luisa::vector<llvm::Constant *> elems;
                    for (unsigned j = 0; j < n; ++j) {
                        elems.push_back(llvm::ConstantFP::get(float_ty, static_cast<double>(value[i][j])));
                    }
                    rows.push_back(llvm::ConstantVector::get(elems));
                }
                _last_value = llvm::ConstantArray::get(mat_ty, rows);
            } else {
                // Integer types: int, uint, long, ulong, short, ushort, etc.
                _last_value = llvm::ConstantInt::get(type, static_cast<uint64_t>(value));
            }
        },
        expr->value());
}

void LLVMStateVisitor::visit(const RefExpr *expr) {
    auto uid = expr->variable().uid();
    _last_value = GetVariable(uid, expr->type());
}

void LLVMStateVisitor::visit(const UnaryExpr *expr) {
    auto *operand = EvalExpr(expr->operand());
    auto *type = expr->type();
    auto op = expr->op();

    switch (op) {
        case UnaryOp::PLUS:
            _last_value = operand;
            break;
        case UnaryOp::MINUS: {
            if (type->is_float() || type->is_float_vector() || type->is_matrix()) {
                _last_value = _builder.CreateFNeg(operand);
            } else {
                _last_value = _builder.CreateNeg(operand);
            }
            break;
        }
        case UnaryOp::NOT: {
            // Logical NOT: xor with true
            auto *true_val = llvm::ConstantInt::get(operand->getType(), 1);
            _last_value = _builder.CreateXor(operand, true_val);
            break;
        }
        case UnaryOp::BIT_NOT: {
            _last_value = _builder.CreateNot(operand);
            break;
        }
    }
}

void LLVMStateVisitor::visit(const BinaryExpr *expr) {
    auto *lhs = EvalExpr(expr->lhs());
    auto *rhs = EvalExpr(expr->rhs());
    auto *type = expr->type();
    auto op = expr->op();

    bool is_float = type->is_float() || type->is_float_vector() || type->is_matrix();
    bool is_int = type->is_int() || type->is_int_vector() || type->is_uint() || type->is_uint_vector();

    switch (op) {
        // --- Arithmetic ---
        case BinaryOp::ADD:
            _last_value = is_float ? _builder.CreateFAdd(lhs, rhs) : _builder.CreateAdd(lhs, rhs);
            break;
        case BinaryOp::SUB:
            _last_value = is_float ? _builder.CreateFSub(lhs, rhs) : _builder.CreateSub(lhs, rhs);
            break;
        case BinaryOp::MUL:
            _last_value = is_float ? _builder.CreateFMul(lhs, rhs) : _builder.CreateMul(lhs, rhs);
            break;
        case BinaryOp::DIV:
            if (is_float) {
                _last_value = _builder.CreateFDiv(lhs, rhs);
            } else if (type->is_int() || type->is_int_vector()) {
                _last_value = _builder.CreateSDiv(lhs, rhs);
            } else {
                _last_value = _builder.CreateUDiv(lhs, rhs);
            }
            break;
        case BinaryOp::MOD:
            if (is_float) {
                _last_value = _builder.CreateFRem(lhs, rhs);
            } else if (type->is_int() || type->is_int_vector()) {
                _last_value = _builder.CreateSRem(lhs, rhs);
            } else {
                _last_value = _builder.CreateURem(lhs, rhs);
            }
            break;

        // --- Bitwise ---
        case BinaryOp::BIT_AND:
            _last_value = _builder.CreateAnd(lhs, rhs);
            break;
        case BinaryOp::BIT_OR:
            _last_value = _builder.CreateOr(lhs, rhs);
            break;
        case BinaryOp::BIT_XOR:
            _last_value = _builder.CreateXor(lhs, rhs);
            break;
        case BinaryOp::SHL:
            _last_value = _builder.CreateShl(lhs, rhs);
            break;
        case BinaryOp::SHR:
            if (type->is_int() || type->is_int_vector()) {
                _last_value = _builder.CreateAShr(lhs, rhs);
            } else {
                _last_value = _builder.CreateLShr(lhs, rhs);
            }
            break;

        // --- Logical (short-circuit not done here; we use bitwise for scalar bool) ---
        case BinaryOp::AND:
            _last_value = _builder.CreateAnd(lhs, rhs);
            break;
        case BinaryOp::OR:
            _last_value = _builder.CreateOr(lhs, rhs);
            break;

        // --- Relational ---
        case BinaryOp::LESS:
            _last_value = is_float ? _builder.CreateFCmpOLT(lhs, rhs) :
                          (type->is_int() || type->is_int_vector()) ? _builder.CreateICmpSLT(lhs, rhs) :
                          _builder.CreateICmpULT(lhs, rhs);
            break;
        case BinaryOp::GREATER:
            _last_value = is_float ? _builder.CreateFCmpOGT(lhs, rhs) :
                          (type->is_int() || type->is_int_vector()) ? _builder.CreateICmpSGT(lhs, rhs) :
                          _builder.CreateICmpUGT(lhs, rhs);
            break;
        case BinaryOp::LESS_EQUAL:
            _last_value = is_float ? _builder.CreateFCmpOLE(lhs, rhs) :
                          (type->is_int() || type->is_int_vector()) ? _builder.CreateICmpSLE(lhs, rhs) :
                          _builder.CreateICmpULE(lhs, rhs);
            break;
        case BinaryOp::GREATER_EQUAL:
            _last_value = is_float ? _builder.CreateFCmpOGE(lhs, rhs) :
                          (type->is_int() || type->is_int_vector()) ? _builder.CreateICmpSGE(lhs, rhs) :
                          _builder.CreateICmpUGE(lhs, rhs);
            break;
        case BinaryOp::EQUAL:
            _last_value = is_float ? _builder.CreateFCmpOEQ(lhs, rhs) :
                          _builder.CreateICmpEQ(lhs, rhs);
            break;
        case BinaryOp::NOT_EQUAL:
            _last_value = is_float ? _builder.CreateFCmpONE(lhs, rhs) :
                          _builder.CreateICmpNE(lhs, rhs);
            break;

        default:
            LUISA_ERROR_WITH_LOCATION("Unsupported binary op.");
    }
}

void LLVMStateVisitor::visit(const MemberExpr *expr) {
    auto *self = EvalExpr(expr->self());
    auto *self_type = expr->self()->type();

    if (expr->is_swizzle()) {
        // Swizzle: extract + combine elements
        auto swizzle_size = expr->swizzle_size();
        if (swizzle_size == 1) {
            _last_value = _builder.CreateExtractElement(self, expr->swizzle_index(0));
        } else {
            auto *result_type = llvm::FixedVectorType::get(
                llvm::cast<llvm::FixedVectorType>(self->getType())->getElementType(),
                swizzle_size);
            llvm::Value *result = llvm::UndefValue::get(result_type);
            for (uint32_t i = 0; i < swizzle_size; ++i) {
                auto *elem = _builder.CreateExtractElement(self, expr->swizzle_index(i));
                result = _builder.CreateInsertElement(result, elem, i);
            }
            _last_value = result;
        }
    } else {
        // Struct/array member access
        auto member_idx = expr->member_index();
        if (self_type->is_structure()) {
            _last_value = _builder.CreateExtractValue(self, {member_idx});
        } else if (self_type->is_array()) {
            _last_value = _builder.CreateExtractValue(self, {member_idx});
        } else if (self_type->is_vector()) {
            _last_value = _builder.CreateExtractElement(self, member_idx);
        } else if (self_type->is_matrix()) {
            // Matrix: array of vectors; extract the row (vector)
            _last_value = _builder.CreateExtractValue(self, {member_idx});
        } else {
            LUISA_ERROR_WITH_LOCATION("MemberExpr on unsupported type.");
        }
    }
}

void LLVMStateVisitor::visit(const AccessExpr *expr) {
    auto *range = EvalExpr(expr->range());
    auto *index = EvalExpr(expr->index());
    auto *range_type = expr->range()->type();

    if (range_type->is_vector()) {
        _last_value = _builder.CreateExtractElement(range, index);
    } else if (range_type->is_matrix()) {
        // Matrix is array of vectors; extract row then column if needed
        // For now: extract the row (vector)
        _last_value = _builder.CreateExtractValue(range, {static_cast<unsigned>(
            llvm::cast<llvm::ConstantInt>(index)->getZExtValue())});
    } else if (range_type->is_array()) {
        // Array loaded into register: use extractvalue
        _last_value = _builder.CreateExtractValue(range, {static_cast<unsigned>(
            llvm::cast<llvm::ConstantInt>(index)->getZExtValue())});
    } else if (range_type->is_buffer()) {
        // Buffer access: GEP from the pointer
        luisa::vector<llvm::Value *> idx_list = {index};
        auto *elem_type = ToLLVMType(*range_type->element());
        auto *gep = _builder.CreateInBoundsGEP(elem_type, range, idx_list);
        _last_value = _builder.CreateLoad(elem_type, gep);
    } else {
        LUISA_ERROR_WITH_LOCATION("AccessExpr on unsupported type: {}",
                                  static_cast<uint32_t>(range_type->tag()));
    }
}

void LLVMStateVisitor::visit(const CastExpr *expr) {
    auto *operand = EvalExpr(expr->expression());
    auto *src_type = expr->expression()->type();
    auto *dst_type = expr->type();
    auto op = expr->op();

    if (op == CastOp::STATIC) {
        auto *llvm_dst = ToLLVMType(*dst_type);

        // Float → Int
        if (src_type->is_float() && (dst_type->is_int() || dst_type->is_uint())) {
            if (dst_type->is_int()) {
                _last_value = _builder.CreateFPToSI(operand, llvm_dst);
            } else {
                _last_value = _builder.CreateFPToUI(operand, llvm_dst);
            }
        }
        // Int → Float
        else if (dst_type->is_float() && (src_type->is_int() || src_type->is_uint())) {
            if (src_type->is_int()) {
                _last_value = _builder.CreateSIToFP(operand, llvm_dst);
            } else {
                _last_value = _builder.CreateUIToFP(operand, llvm_dst);
            }
        }
        // Same kind, different width
        else if (dst_type->is_int() || dst_type->is_uint()) {
            auto src_bits = operand->getType()->getIntegerBitWidth();
            auto dst_bits = llvm_dst->getIntegerBitWidth();
            if (dst_bits > src_bits) {
                _last_value = (src_type->is_int())
                    ? _builder.CreateSExt(operand, llvm_dst)
                    : _builder.CreateZExt(operand, llvm_dst);
            } else {
                _last_value = _builder.CreateTrunc(operand, llvm_dst);
            }
        }
        // Float → Float
        else if (src_type->is_float() && dst_type->is_float()) {
            if (operand->getType()->isHalfTy()) {
                _last_value = _builder.CreateFPExt(operand, llvm_dst);
            } else if (llvm_dst->isHalfTy()) {
                _last_value = _builder.CreateFPTrunc(operand, llvm_dst);
            } else {
                _last_value = _builder.CreateFPExt(operand, llvm_dst);
            }
        }
        // Bool ↔ Int
        else if (dst_type->is_bool()) {
            _last_value = _builder.CreateICmpNE(operand,
                llvm::ConstantInt::get(operand->getType(), 0));
        } else if (src_type->is_bool()) {
            auto *llvm_src = ToLLVMType(*src_type);
            _last_value = _builder.CreateZExt(operand, llvm_dst);
        }
        // Vector casts
        else if (src_type->is_vector() && dst_type->is_vector()) {
            auto src_dim = src_type->dimension();
            auto dst_dim = dst_type->dimension();
            if (src_dim == dst_dim) {
                // Element-wise cast
                auto *elem_dst = ToLLVMType(*dst_type->element());
                llvm::Value *result = llvm::UndefValue::get(llvm_dst);
                for (size_t i = 0; i < src_dim; ++i) {
                    auto *elem = _builder.CreateExtractElement(operand, i);
                    // Recurse via cast on element
                    // Simpler: just bitcast if same size
                    if (operand->getType()->getPrimitiveSizeInBits() == llvm_dst->getPrimitiveSizeInBits()) {
                        result = _builder.CreateBitCast(operand, llvm_dst);
                        break;
                    }
                }
                _last_value = result;
            } else {
                _last_value = _builder.CreateBitCast(operand, llvm_dst);
            }
        } else {
            // Fallback: bitcast
            _last_value = _builder.CreateBitCast(operand, llvm_dst);
        }
    } else if (op == CastOp::BITWISE) {
        _last_value = _builder.CreateBitCast(operand, ToLLVMType(*dst_type));
    } else {
        LUISA_ERROR_WITH_LOCATION("Unsupported cast op.");
    }
}

void LLVMStateVisitor::visit(const ConstantExpr *expr) {
    auto *type = ToLLVMType(*expr->type());
    auto data = expr->data();
    auto *constant = _util->CreateConstant(data, type);
    _last_value = constant;
}

void LLVMStateVisitor::visit(const TypeIDExpr *expr) {
    auto hash = expr->type()->hash();
    _last_value = llvm::ConstantInt::get(_builder.getInt64Ty(), hash);
}

void LLVMStateVisitor::visit(const StringIDExpr *expr) {
    _last_value = llvm::ConstantInt::get(_builder.getInt64Ty(), expr->hash());
}

// ============================================================================
// CallExpr (builtin + custom dispatch)
// ============================================================================

void LLVMStateVisitor::_codegen_builtin_call(CallOp op, const CallExpr *expr) {
    auto args = expr->arguments();
    auto *ret_type = expr->type();

    switch (op) {
        // --- SELECT ---
        case CallOp::SELECT: {
            auto *cond = EvalExpr(args[0]);
            auto *tval = EvalExpr(args[1]);
            auto *fval = EvalExpr(args[2]);
            _last_value = _builder.CreateSelect(cond, tval, fval);
            break;
        }

        // --- ABS ---
        case CallOp::ABS: {
            auto *v = EvalExpr(args[0]);
            _last_value = _emit_abs(v, *args[0]->type());
            break;
        }

        // --- MIN / MAX ---
        case CallOp::MIN: {
            auto *a = EvalExpr(args[0]);
            auto *b = EvalExpr(args[1]);
            _last_value = _emit_min(a, b, *ret_type);
            break;
        }
        case CallOp::MAX: {
            auto *a = EvalExpr(args[0]);
            auto *b = EvalExpr(args[1]);
            _last_value = _emit_max(a, b, *ret_type);
            break;
        }

        // --- CLAMP ---
        case CallOp::CLAMP: {
            auto *v = EvalExpr(args[0]);
            auto *lo = EvalExpr(args[1]);
            auto *hi = EvalExpr(args[2]);
            _last_value = _emit_clamp(v, lo, hi, *ret_type);
            break;
        }

        // --- SATURATE ---
        case CallOp::SATURATE: {
            auto *v = EvalExpr(args[0]);
            auto *zero = llvm::Constant::getNullValue(v->getType());
            auto *one = llvm::ConstantFP::get(v->getType(), 1.0);
            _last_value = _emit_clamp(v, zero, one, *ret_type);
            break;
        }

        // --- LERP ---
        case CallOp::LERP: {
            auto *a = EvalExpr(args[0]);
            auto *b = EvalExpr(args[1]);
            auto *t = EvalExpr(args[2]);
            _last_value = _emit_lerp(a, b, t, *ret_type);
            break;
        }

        // --- SQRT ---
        case CallOp::SQRT: {
            auto *v = EvalExpr(args[0]);
            auto *intrinsic = llvm::Intrinsic::getDeclarationIfExists(
                &_module, llvm::Intrinsic::sqrt, {v->getType()});
            _last_value = _builder.CreateCall(intrinsic, {v});
            break;
        }

        // --- RSQRT ---
        case CallOp::RSQRT: {
            auto *v = EvalExpr(args[0]);
            auto *sqrt_intr = llvm::Intrinsic::getDeclarationIfExists(
                &_module, llvm::Intrinsic::sqrt, {v->getType()});
            auto *sqrt_val = _builder.CreateCall(sqrt_intr, {v});
            auto *one = llvm::ConstantFP::get(v->getType(), 1.0);
            _last_value = _builder.CreateFDiv(one, sqrt_val);
            break;
        }

        // --- SIN ---
        case CallOp::SIN: {
            auto *v = EvalExpr(args[0]);
            auto *intrinsic = llvm::Intrinsic::getDeclarationIfExists(
                &_module, llvm::Intrinsic::sin, {v->getType()});
            _last_value = _builder.CreateCall(intrinsic, {v});
            break;
        }

        // --- COS ---
        case CallOp::COS: {
            auto *v = EvalExpr(args[0]);
            auto *intrinsic = llvm::Intrinsic::getDeclarationIfExists(
                &_module, llvm::Intrinsic::cos, {v->getType()});
            _last_value = _builder.CreateCall(intrinsic, {v});
            break;
        }

        // --- EXP ---
        case CallOp::EXP: {
            auto *v = EvalExpr(args[0]);
            auto *intrinsic = llvm::Intrinsic::getDeclarationIfExists(
                &_module, llvm::Intrinsic::exp, {v->getType()});
            _last_value = _builder.CreateCall(intrinsic, {v});
            break;
        }

        // --- EXP2 ---
        case CallOp::EXP2: {
            auto *v = EvalExpr(args[0]);
            auto *intrinsic = llvm::Intrinsic::getDeclarationIfExists(
                &_module, llvm::Intrinsic::exp2, {v->getType()});
            _last_value = _builder.CreateCall(intrinsic, {v});
            break;
        }

        // --- LOG ---
        case CallOp::LOG: {
            auto *v = EvalExpr(args[0]);
            auto *intrinsic = llvm::Intrinsic::getDeclarationIfExists(
                &_module, llvm::Intrinsic::log, {v->getType()});
            _last_value = _builder.CreateCall(intrinsic, {v});
            break;
        }

        // --- LOG2 ---
        case CallOp::LOG2: {
            auto *v = EvalExpr(args[0]);
            auto *intrinsic = llvm::Intrinsic::getDeclarationIfExists(
                &_module, llvm::Intrinsic::log2, {v->getType()});
            _last_value = _builder.CreateCall(intrinsic, {v});
            break;
        }

        // --- POW ---
        case CallOp::POW: {
            auto *a = EvalExpr(args[0]);
            auto *b = EvalExpr(args[1]);
            auto *intrinsic = llvm::Intrinsic::getDeclarationIfExists(
                &_module, llvm::Intrinsic::pow, {a->getType()});
            _last_value = _builder.CreateCall(intrinsic, {a, b});
            break;
        }

        // --- FMA ---
        case CallOp::FMA: {
            auto *a = EvalExpr(args[0]);
            auto *b = EvalExpr(args[1]);
            auto *c = EvalExpr(args[2]);
            auto *intrinsic = llvm::Intrinsic::getDeclarationIfExists(
                &_module, llvm::Intrinsic::fma, {a->getType()});
            _last_value = _builder.CreateCall(intrinsic, {a, b, c});
            break;
        }

        // --- COPYSIGN ---
        case CallOp::COPYSIGN: {
            auto *a = EvalExpr(args[0]);
            auto *b = EvalExpr(args[1]);
            auto *intrinsic = llvm::Intrinsic::getDeclarationIfExists(
                &_module, llvm::Intrinsic::copysign, {a->getType()});
            _last_value = _builder.CreateCall(intrinsic, {a, b});
            break;
        }

        // --- FLOOR ---
        case CallOp::FLOOR: {
            auto *v = EvalExpr(args[0]);
            auto *intrinsic = llvm::Intrinsic::getDeclarationIfExists(
                &_module, llvm::Intrinsic::floor, {v->getType()});
            _last_value = _builder.CreateCall(intrinsic, {v});
            break;
        }

        // --- CEIL ---
        case CallOp::CEIL: {
            auto *v = EvalExpr(args[0]);
            auto *intrinsic = llvm::Intrinsic::getDeclarationIfExists(
                &_module, llvm::Intrinsic::ceil, {v->getType()});
            _last_value = _builder.CreateCall(intrinsic, {v});
            break;
        }

        // --- TRUNC ---
        case CallOp::TRUNC: {
            auto *v = EvalExpr(args[0]);
            auto *intrinsic = llvm::Intrinsic::getDeclarationIfExists(
                &_module, llvm::Intrinsic::trunc, {v->getType()});
            _last_value = _builder.CreateCall(intrinsic, {v});
            break;
        }

        // --- ROUND ---
        case CallOp::ROUND: {
            auto *v = EvalExpr(args[0]);
            auto *intrinsic = llvm::Intrinsic::getDeclarationIfExists(
                &_module, llvm::Intrinsic::round, {v->getType()});
            _last_value = _builder.CreateCall(intrinsic, {v});
            break;
        }

        // --- ISINF ---
        case CallOp::ISINF: {
            auto *v = EvalExpr(args[0]);
            auto *inf = llvm::ConstantFP::getInfinity(v->getType());
            _last_value = _builder.CreateFCmpOEQ(
                _builder.CreateFCmpOEQ(v, inf),
                llvm::ConstantInt::getTrue(_ctx));
            // Actually: isinf = abs(x) == infinity
            auto *abs = _emit_abs(v, *args[0]->type());
            _last_value = _builder.CreateFCmpOEQ(abs, inf);
            break;
        }

        // --- ISNAN ---
        case CallOp::ISNAN: {
            auto *v = EvalExpr(args[0]);
            _last_value = _builder.CreateFCmpUNO(v, v);
            break;
        }

        // --- DOT ---
        case CallOp::DOT: {
            auto *a = EvalExpr(args[0]);
            auto *b = EvalExpr(args[1]);
            _last_value = _emit_dot(a, b);
            break;
        }

        // --- CROSS ---
        case CallOp::CROSS: {
            auto *a = EvalExpr(args[0]);
            auto *b = EvalExpr(args[1]);
            _last_value = _emit_cross(a, b);
            break;
        }

        // --- LENGTH ---
        case CallOp::LENGTH: {
            auto *v = EvalExpr(args[0]);
            _last_value = _emit_length(v);
            break;
        }

        // --- LENGTH_SQUARED ---
        case CallOp::LENGTH_SQUARED: {
            auto *v = EvalExpr(args[0]);
            auto *dot = _emit_dot(v, v);
            _last_value = dot;
            break;
        }

        // --- NORMALIZE ---
        case CallOp::NORMALIZE: {
            auto *v = EvalExpr(args[0]);
            _last_value = _emit_normalize(v);
            break;
        }

        // --- ALL / ANY ---
        case CallOp::ALL: {
            auto *v = EvalExpr(args[0]);
            _last_value = _emit_all(v);
            break;
        }
        case CallOp::ANY: {
            auto *v = EvalExpr(args[0]);
            _last_value = _emit_any(v);
            break;
        }

        // --- CLZ / CTZ / POPCOUNT / REVERSE ---
        case CallOp::CLZ: {
            auto *v = EvalExpr(args[0]);
            auto *intrinsic = llvm::Intrinsic::getDeclarationIfExists(
                &_module, llvm::Intrinsic::ctlz, {v->getType()});
            _last_value = _builder.CreateCall(intrinsic, {v, _builder.getInt1(false)});
            break;
        }
        case CallOp::CTZ: {
            auto *v = EvalExpr(args[0]);
            auto *intrinsic = llvm::Intrinsic::getDeclarationIfExists(
                &_module, llvm::Intrinsic::cttz, {v->getType()});
            _last_value = _builder.CreateCall(intrinsic, {v, _builder.getInt1(false)});
            break;
        }
        case CallOp::POPCOUNT: {
            auto *v = EvalExpr(args[0]);
            auto *intrinsic = llvm::Intrinsic::getDeclarationIfExists(
                &_module, llvm::Intrinsic::ctpop, {v->getType()});
            _last_value = _builder.CreateCall(intrinsic, {v});
            break;
        }
        case CallOp::REVERSE: {
            auto *v = EvalExpr(args[0]);
            auto *intrinsic = llvm::Intrinsic::getDeclarationIfExists(
                &_module, llvm::Intrinsic::bitreverse, {v->getType()});
            _last_value = _builder.CreateCall(intrinsic, {v});
            break;
        }

        // --- DETERMINANT ---
        case CallOp::DETERMINANT: {
            auto *v = EvalExpr(args[0]);
            auto dim = args[0]->type()->dimension();
            if (dim == 2) {
                // det([[a,b],[c,d]]) = a*d - b*c
                auto *a = _builder.CreateExtractValue(v, {0});
                auto *b = _builder.CreateExtractValue(v, {1});
                auto *a0 = _builder.CreateExtractElement(a, uint64_t(0));
                auto *a1 = _builder.CreateExtractElement(a, uint64_t(1));
                auto *b0 = _builder.CreateExtractElement(b, uint64_t(0));
                auto *b1 = _builder.CreateExtractElement(b, uint64_t(1));
                _last_value = _builder.CreateFSub(
                    _builder.CreateFMul(a0, b1),
                    _builder.CreateFMul(a1, b0));
            } else {
                LUISA_NOT_IMPLEMENTED();
                _last_value = llvm::ConstantFP::get(_builder.getFloatTy(), 0.0);
            }
            break;
        }

        // --- TRANSPOSE ---
        case CallOp::TRANSPOSE: {
            LUISA_NOT_IMPLEMENTED();
            _last_value = llvm::UndefValue::get(ToLLVMType(*ret_type));
            break;
        }

        // --- INVERSE ---
        case CallOp::INVERSE: {
            LUISA_NOT_IMPLEMENTED();
            _last_value = llvm::UndefValue::get(ToLLVMType(*ret_type));
            break;
        }

        // --- MAKE_* vector makers ---
        case CallOp::MAKE_FLOAT2:
        case CallOp::MAKE_FLOAT3:
        case CallOp::MAKE_FLOAT4:
        case CallOp::MAKE_INT2:
        case CallOp::MAKE_INT3:
        case CallOp::MAKE_INT4:
        case CallOp::MAKE_UINT2:
        case CallOp::MAKE_UINT3:
        case CallOp::MAKE_UINT4:
        case CallOp::MAKE_BOOL2:
        case CallOp::MAKE_BOOL3:
        case CallOp::MAKE_BOOL4:
        case CallOp::MAKE_SHORT2:
        case CallOp::MAKE_SHORT3:
        case CallOp::MAKE_SHORT4:
        case CallOp::MAKE_USHORT2:
        case CallOp::MAKE_USHORT3:
        case CallOp::MAKE_USHORT4:
        case CallOp::MAKE_LONG2:
        case CallOp::MAKE_LONG3:
        case CallOp::MAKE_LONG4:
        case CallOp::MAKE_ULONG2:
        case CallOp::MAKE_ULONG3:
        case CallOp::MAKE_ULONG4:
        case CallOp::MAKE_HALF2:
        case CallOp::MAKE_HALF3:
        case CallOp::MAKE_HALF4:
        case CallOp::MAKE_DOUBLE2:
        case CallOp::MAKE_DOUBLE3:
        case CallOp::MAKE_DOUBLE4:
        case CallOp::MAKE_BYTE2:
        case CallOp::MAKE_BYTE3:
        case CallOp::MAKE_BYTE4:
        case CallOp::MAKE_UBYTE2:
        case CallOp::MAKE_UBYTE3:
        case CallOp::MAKE_UBYTE4: {
            auto *vec_type = llvm::cast<llvm::FixedVectorType>(ToLLVMType(*ret_type));
            llvm::Value *result = llvm::UndefValue::get(vec_type);
            for (size_t i = 0; i < args.size(); ++i) {
                auto *elem = EvalExpr(args[i]);
                // If element is same type as vector element
                if (elem->getType() == vec_type->getElementType()) {
                    result = _builder.CreateInsertElement(result, elem, i);
                } else {
                    // Scalar broadcast or type mismatch — insert first element
                    if (auto *vec_elem = llvm::dyn_cast<llvm::FixedVectorType>(elem->getType())) {
                        elem = _builder.CreateExtractElement(elem, uint64_t(0));
                    }
                    result = _builder.CreateInsertElement(result, elem, i);
                }
            }
            _last_value = result;
            break;
        }

        // --- MAKE_FLOAT2X2/3X3/4X4 ---
        case CallOp::MAKE_FLOAT2X2:
        case CallOp::MAKE_FLOAT3X3:
        case CallOp::MAKE_FLOAT4X4: {
            auto *mat_type = llvm::cast<llvm::ArrayType>(ToLLVMType(*ret_type));
            auto *row_type = mat_type->getElementType();
            llvm::Value *result = llvm::UndefValue::get(mat_type);
            for (size_t i = 0; i < args.size(); ++i) {
                auto *row_val = EvalExpr(args[i]);
                if (row_val->getType() != row_type) {
                    // Try bitcast or element-wise conversion
                    row_val = _builder.CreateBitCast(row_val, row_type);
                }
                result = _builder.CreateInsertValue(result, row_val, {static_cast<unsigned>(i)});
            }
            _last_value = result;
            break;
        }

        // --- STEP ---
        case CallOp::STEP: {
            auto *edge = EvalExpr(args[0]);
            auto *x = EvalExpr(args[1]);
            // step(edge, x) = (x >= edge) ? 1 : 0
            auto *cmp = _builder.CreateFCmpOGE(x, edge);
            _last_value = _builder.CreateUIToFP(
                _builder.CreateZExt(cmp, _builder.getInt32Ty()),
                edge->getType());
            break;
        }

        // --- SMOOTHSTEP ---
        case CallOp::SMOOTHSTEP: {
            auto *edge0 = EvalExpr(args[0]);
            auto *edge1 = EvalExpr(args[1]);
            auto *x = EvalExpr(args[2]);
            // t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
            // return t * t * (3.0 - 2.0 * t)
            auto *diff = _builder.CreateFSub(edge1, edge0);
            auto *t = _builder.CreateFDiv(_builder.CreateFSub(x, edge0), diff);
            auto *zero = llvm::Constant::getNullValue(t->getType());
            auto *one = llvm::ConstantFP::get(t->getType(), 1.0);
            t = _emit_clamp(t, zero, one, *ret_type);
            auto *t2 = _builder.CreateFMul(t, t);
            auto *two = llvm::ConstantFP::get(t->getType(), 2.0);
            auto *three = llvm::ConstantFP::get(t->getType(), 3.0);
            _last_value = _builder.CreateFMul(t2,
                _builder.CreateFSub(three, _builder.CreateFMul(two, t)));
            break;
        }

        // --- SYNCHRONIZE_BLOCK ---
        case CallOp::SYNCHRONIZE_BLOCK: {
            // Emit a barrier call. Try NVVM barrier or fall back to a fence.
            auto *barrier_func = _module.getFunction("llvm.nvvm.barrier0");
            if (!barrier_func) {
                auto *void_ty = llvm::FunctionType::get(_builder.getVoidTy(), false);
                barrier_func = llvm::Function::Create(
                    void_ty, llvm::Function::ExternalLinkage,
                    "llvm.nvvm.barrier0", &_module);
            }
            _builder.CreateCall(barrier_func);
            _last_value = nullptr;
            break;
        }

        // --- FRACT ---
        case CallOp::FRACT: {
            auto *v = EvalExpr(args[0]);
            auto *floor_val = _builder.CreateCall(
                llvm::Intrinsic::getDeclarationIfExists(&_module, llvm::Intrinsic::floor, {v->getType()}),
                {v});
            _last_value = _builder.CreateFSub(v, floor_val);
            break;
        }

        // --- ACOS ---
        case CallOp::ACOS: {
            LUISA_NOT_IMPLEMENTED();
            _last_value = llvm::Constant::getNullValue(ToLLVMType(*ret_type));
            break;
        }
        // --- ASIN ---
        case CallOp::ASIN: {
            LUISA_NOT_IMPLEMENTED();
            _last_value = llvm::Constant::getNullValue(ToLLVMType(*ret_type));
            break;
        }
        // --- ATAN ---
        case CallOp::ATAN: {
            LUISA_NOT_IMPLEMENTED();
            _last_value = llvm::Constant::getNullValue(ToLLVMType(*ret_type));
            break;
        }
        // --- ATAN2 ---
        case CallOp::ATAN2: {
            LUISA_NOT_IMPLEMENTED();
            _last_value = llvm::Constant::getNullValue(ToLLVMType(*ret_type));
            break;
        }
        // --- TAN ---
        case CallOp::TAN: {
            LUISA_NOT_IMPLEMENTED();
            _last_value = llvm::Constant::getNullValue(ToLLVMType(*ret_type));
            break;
        }
        // --- COSH, SINH, TANH, ACOSH, ASINH, ATANH ---
        case CallOp::COSH:
        case CallOp::SINH:
        case CallOp::TANH:
        case CallOp::ACOSH:
        case CallOp::ASINH:
        case CallOp::ATANH:
        case CallOp::EXP10:
        case CallOp::LOG10: {
            LUISA_NOT_IMPLEMENTED();
            _last_value = llvm::Constant::getNullValue(ToLLVMType(*ret_type));
            break;
        }

        // --- REDUCE_* ---
        case CallOp::REDUCE_SUM: {
            auto *v = EvalExpr(args[0]);
            auto *vec_ty = llvm::dyn_cast<llvm::FixedVectorType>(v->getType());
            if (!vec_ty) {
                _last_value = v;
                break;
            }
            auto *result = _builder.CreateExtractElement(v, uint64_t(0));
            for (unsigned i = 1; i < vec_ty->getNumElements(); ++i) {
                auto *elem = _builder.CreateExtractElement(v, i);
                result = _builder.CreateFAdd(result, elem);
            }
            _last_value = result;
            break;
        }

        // --- OUTER_PRODUCT ---
        case CallOp::OUTER_PRODUCT: {
            LUISA_NOT_IMPLEMENTED();
            _last_value = llvm::UndefValue::get(ToLLVMType(*ret_type));
            break;
        }

        // --- MATRIX_COMPONENT_WISE_MULTIPLICATION ---
        case CallOp::MATRIX_COMPONENT_WISE_MULTIPLICATION: {
            LUISA_NOT_IMPLEMENTED();
            _last_value = llvm::UndefValue::get(ToLLVMType(*ret_type));
            break;
        }

        // --- FACEFORWARD ---
        case CallOp::FACEFORWARD: {
            LUISA_NOT_IMPLEMENTED();
            _last_value = llvm::Constant::getNullValue(ToLLVMType(*ret_type));
            break;
        }

        // --- REFLECT ---
        case CallOp::REFLECT: {
            auto *i = EvalExpr(args[0]);
            auto *n = EvalExpr(args[1]);
            // i - 2 * dot(n, i) * n
            auto *dot = _emit_dot(n, i);
            auto *two = llvm::ConstantFP::get(dot->getType(), 2.0);
            auto *two_dot = _builder.CreateFMul(two, dot);

            // Broadcast scalar to vector if needed
            llvm::Value *scaled_n;
            if (auto *vec_ty = llvm::dyn_cast<llvm::FixedVectorType>(n->getType())) {
                llvm::Value *broadcast = llvm::UndefValue::get(vec_ty);
                for (unsigned j = 0; j < vec_ty->getNumElements(); ++j) {
                    broadcast = _builder.CreateInsertElement(broadcast, two_dot, j);
                }
                scaled_n = _builder.CreateFMul(broadcast, n);
            } else {
                scaled_n = _builder.CreateFMul(two_dot, n);
            }
            _last_value = _builder.CreateFSub(i, scaled_n);
            break;
        }

        // --- BUFFER_READ / BUFFER_WRITE (including volatile aliases) ---
        case CallOp::BUFFER_READ:
        case CallOp::BUFFER_VOLATILE_READ: {
            auto *buffer_ptr = EvalExpr(args[0]);
            auto *index = EvalExpr(args[1]);
            auto *elem_type = ToLLVMType(*ret_type);
            // Track bindless usage by checking if the buffer argument is a bindless array variable
            if (args[0]->tag() == Expression::Tag::REF) {
                auto *ref = static_cast<RefExpr const *>(args[0]);
                if (ref->variable().tag() == Variable::Tag::BINDLESS_ARRAY) {
                    _util->opt->useBufferBindless = true;
                }
            }
            luisa::vector<llvm::Value *> idx_list = {index};
            auto *gep = _builder.CreateInBoundsGEP(elem_type, buffer_ptr, idx_list);
            _last_value = _builder.CreateLoad(elem_type, gep);
            break;
        }
        case CallOp::BUFFER_WRITE:
        case CallOp::BUFFER_VOLATILE_WRITE: {
            auto *buffer_ptr = EvalExpr(args[0]);
            auto *index = EvalExpr(args[1]);
            auto *value = EvalExpr(args[2]);
            auto *elem_type = value->getType();
            // Track bindless usage
            if (args[0]->tag() == Expression::Tag::REF) {
                auto *ref = static_cast<RefExpr const *>(args[0]);
                if (ref->variable().tag() == Variable::Tag::BINDLESS_ARRAY) {
                    _util->opt->useBufferBindless = true;
                }
            }
            luisa::vector<llvm::Value *> idx_list = {index};
            auto *gep = _builder.CreateInBoundsGEP(elem_type, buffer_ptr, idx_list);
            _builder.CreateStore(value, gep);
            _last_value = nullptr;
            break;
        }

        // --- BYTE_BUFFER_READ / BYTE_BUFFER_WRITE ---
        case CallOp::BYTE_BUFFER_READ:
        case CallOp::BYTE_BUFFER_VOLATILE_READ: {
            auto *buffer_ptr = EvalExpr(args[0]);
            auto *byte_index = EvalExpr(args[1]);
            auto *elem_type = ToLLVMType(*ret_type);
            auto *i8_ptr = _builder.CreateBitCast(buffer_ptr, _builder.getPtrTy(0));
            auto *gep = _builder.CreateInBoundsGEP(_builder.getInt8Ty(), i8_ptr, {byte_index});
            auto *cast_ptr = _builder.CreateBitCast(gep, llvm::PointerType::get(elem_type, 0));
            _last_value = _builder.CreateLoad(elem_type, cast_ptr);
            break;
        }
        case CallOp::BYTE_BUFFER_WRITE:
        case CallOp::BYTE_BUFFER_VOLATILE_WRITE: {
            auto *buffer_ptr = EvalExpr(args[0]);
            auto *byte_index = EvalExpr(args[1]);
            auto *value = EvalExpr(args[2]);
            auto *i8_ptr = _builder.CreateBitCast(buffer_ptr, _builder.getPtrTy(0));
            auto *gep = _builder.CreateInBoundsGEP(_builder.getInt8Ty(), i8_ptr, {byte_index});
            auto *cast_ptr = _builder.CreateBitCast(gep, llvm::PointerType::get(value->getType(), 0));
            _builder.CreateStore(value, cast_ptr);
            _last_value = nullptr;
            break;
        }

        // --- BUFFER_SIZE ---
        case CallOp::BUFFER_SIZE: {
            LUISA_NOT_IMPLEMENTED();
            _last_value = _builder.getInt32(0);
            break;
        }

        // --- ATOMIC_* ---
        case CallOp::ATOMIC_EXCHANGE: {
            auto *ptr_val = EvalExpr(args[0]); // should be a pointer
            auto *val = EvalExpr(args[1]);
            _last_value = _builder.CreateAtomicRMW(
                llvm::AtomicRMWInst::Xchg, ptr_val, val,
                llvm::MaybeAlign(),
                llvm::AtomicOrdering::SequentiallyConsistent);
            break;
        }
        case CallOp::ATOMIC_FETCH_ADD: {
            auto *ptr_val = EvalExpr(args[0]);
            auto *val = EvalExpr(args[1]);
            _last_value = _builder.CreateAtomicRMW(
                llvm::AtomicRMWInst::Add, ptr_val, val,
                llvm::MaybeAlign(),
                llvm::AtomicOrdering::SequentiallyConsistent);
            break;
        }
        case CallOp::ATOMIC_FETCH_SUB: {
            auto *ptr_val = EvalExpr(args[0]);
            auto *val = EvalExpr(args[1]);
            _last_value = _builder.CreateAtomicRMW(
                llvm::AtomicRMWInst::Sub, ptr_val, val,
                llvm::MaybeAlign(),
                llvm::AtomicOrdering::SequentiallyConsistent);
            break;
        }
        case CallOp::ATOMIC_FETCH_AND: {
            auto *ptr_val = EvalExpr(args[0]);
            auto *val = EvalExpr(args[1]);
            _last_value = _builder.CreateAtomicRMW(
                llvm::AtomicRMWInst::And, ptr_val, val,
                llvm::MaybeAlign(),
                llvm::AtomicOrdering::SequentiallyConsistent);
            break;
        }
        case CallOp::ATOMIC_FETCH_OR: {
            auto *ptr_val = EvalExpr(args[0]);
            auto *val = EvalExpr(args[1]);
            _last_value = _builder.CreateAtomicRMW(
                llvm::AtomicRMWInst::Or, ptr_val, val,
                llvm::MaybeAlign(),
                llvm::AtomicOrdering::SequentiallyConsistent);
            break;
        }
        case CallOp::ATOMIC_FETCH_XOR: {
            auto *ptr_val = EvalExpr(args[0]);
            auto *val = EvalExpr(args[1]);
            _last_value = _builder.CreateAtomicRMW(
                llvm::AtomicRMWInst::Xor, ptr_val, val,
                llvm::MaybeAlign(),
                llvm::AtomicOrdering::SequentiallyConsistent);
            break;
        }
        case CallOp::ATOMIC_FETCH_MIN: {
            auto *ptr_val = EvalExpr(args[0]);
            auto *val = EvalExpr(args[1]);
            auto *cmp_type = val->getType();
            _last_value = _builder.CreateAtomicRMW(
                cmp_type->isIntegerTy() ? llvm::AtomicRMWInst::Min : llvm::AtomicRMWInst::FMin,
                ptr_val, val,
                llvm::MaybeAlign(),
                llvm::AtomicOrdering::SequentiallyConsistent);
            break;
        }
        case CallOp::ATOMIC_FETCH_MAX: {
            auto *ptr_val = EvalExpr(args[0]);
            auto *val = EvalExpr(args[1]);
            auto *cmp_type = val->getType();
            _last_value = _builder.CreateAtomicRMW(
                cmp_type->isIntegerTy() ? llvm::AtomicRMWInst::Max : llvm::AtomicRMWInst::FMax,
                ptr_val, val,
                llvm::MaybeAlign(),
                llvm::AtomicOrdering::SequentiallyConsistent);
            break;
        }
        case CallOp::ATOMIC_COMPARE_EXCHANGE: {
            auto *ptr_val = EvalExpr(args[0]);
            auto *expected = EvalExpr(args[1]);
            auto *desired = EvalExpr(args[2]);
            auto *pair = _builder.CreateAtomicCmpXchg(
                ptr_val, expected, desired,
                llvm::MaybeAlign(),
                llvm::AtomicOrdering::SequentiallyConsistent,
                llvm::AtomicOrdering::SequentiallyConsistent);
            _last_value = _builder.CreateExtractValue(pair, {0}); // old value
            break;
        }

        // --- WARP_* (wave intrinsics) ---
        case CallOp::WARP_IS_FIRST_ACTIVE_LANE:
        case CallOp::WARP_FIRST_ACTIVE_LANE:
        case CallOp::WARP_ACTIVE_ALL_EQUAL:
        case CallOp::WARP_ACTIVE_BIT_AND:
        case CallOp::WARP_ACTIVE_BIT_OR:
        case CallOp::WARP_ACTIVE_BIT_XOR:
        case CallOp::WARP_ACTIVE_COUNT_BITS:
        case CallOp::WARP_ACTIVE_MAX:
        case CallOp::WARP_ACTIVE_MIN:
        case CallOp::WARP_ACTIVE_PRODUCT:
        case CallOp::WARP_ACTIVE_SUM:
        case CallOp::WARP_ACTIVE_ALL:
        case CallOp::WARP_ACTIVE_ANY:
        case CallOp::WARP_ACTIVE_BIT_MASK:
        case CallOp::WARP_PREFIX_COUNT_BITS:
        case CallOp::WARP_PREFIX_SUM:
        case CallOp::WARP_PREFIX_PRODUCT:
        case CallOp::WARP_READ_LANE:
        case CallOp::WARP_READ_FIRST_ACTIVE_LANE: {
            LUISA_NOT_IMPLEMENTED();
            _last_value = llvm::Constant::getNullValue(ToLLVMType(*ret_type));
            break;
        }

        // --- DDX / DDY ---
        case CallOp::DDX:
        case CallOp::DDY: {
            LUISA_NOT_IMPLEMENTED();
            _last_value = llvm::Constant::getNullValue(ToLLVMType(*ret_type));
            break;
        }

        // --- ASSUME / UNREACHABLE ---
        case CallOp::UNREACHABLE: {
            _builder.CreateUnreachable();
            _last_value = nullptr;
            break;
        }
        case CallOp::ASSUME: {
            // no-op for now
            _last_value = nullptr;
            break;
        }

        // --- CLOCK ---
        case CallOp::CLOCK: {
            LUISA_NOT_IMPLEMENTED();
            _last_value = _builder.getInt64(0);
            break;
        }

        // --- ZERO / ONE ---
        case CallOp::ZERO: {
            _last_value = llvm::Constant::getNullValue(ToLLVMType(*ret_type));
            break;
        }
        case CallOp::ONE: {
            auto *ty = ToLLVMType(*ret_type);
            if (ty->isFloatTy()) {
                _last_value = llvm::ConstantFP::get(ty, 1.0);
            } else if (ty->isIntegerTy()) {
                _last_value = llvm::ConstantInt::get(ty, 1);
            } else {
                _last_value = llvm::Constant::getNullValue(ty); // fallback
            }
            break;
        }

        // --- Texture / Bindless / Ray Tracing / Cooperative ---
        default: {
            LUISA_NOT_IMPLEMENTED();
            if (ret_type->tag() == Type::Tag::BOOL) {
                _last_value = _builder.getFalse();
            } else if (!ret_type) {
                _last_value = nullptr;
            } else {
                _last_value = llvm::Constant::getNullValue(ToLLVMType(*ret_type));
            }
            break;
        }
    }
}

void LLVMStateVisitor::visit(const CallExpr *expr) {
    if (expr->is_builtin()) {
        _codegen_builtin_call(expr->op(), expr);
    } else if (expr->is_custom()) {
        auto callee = expr->custom();
        auto *llvm_callee = _util->GetOrDeclareFunction(callee);
        luisa::vector<llvm::Value *> call_args;
        for (auto arg : expr->arguments()) {
            call_args.push_back(EvalExpr(arg));
        }
        auto *call = _builder.CreateCall(llvm_callee, call_args);
        _last_value = call;
    } else {
        LUISA_ERROR_WITH_LOCATION("Unknown call expr type.");
    }
}

// ============================================================================
// Math Helpers
// ============================================================================

llvm::Value *LLVMStateVisitor::_emit_abs(llvm::Value *v, Type const &type) {
    if (type.is_float() || type.is_float_vector()) {
        auto *intrinsic = llvm::Intrinsic::getDeclarationIfExists(
            &_module, llvm::Intrinsic::fabs, {v->getType()});
        return _builder.CreateCall(intrinsic, {v});
    } else {
        // Integer abs: (x < 0) ? -x : x
        auto *zero = llvm::ConstantInt::get(v->getType(), 0);
        auto *neg = _builder.CreateNeg(v);
        auto *cmp = _builder.CreateICmpSLT(v, zero);
        return _builder.CreateSelect(cmp, neg, v);
    }
}

llvm::Value *LLVMStateVisitor::_emit_min(llvm::Value *a, llvm::Value *b, Type const &type) {
    if (type.is_float() || type.is_float_vector()) {
        auto *intrinsic = llvm::Intrinsic::getDeclarationIfExists(
            &_module, llvm::Intrinsic::minnum, {a->getType()});
        return _builder.CreateCall(intrinsic, {a, b});
    } else if (type.is_int() || type.is_int_vector()) {
        auto *cmp = _builder.CreateICmpSLT(a, b);
        return _builder.CreateSelect(cmp, a, b);
    } else {
        auto *cmp = _builder.CreateICmpULT(a, b);
        return _builder.CreateSelect(cmp, a, b);
    }
}

llvm::Value *LLVMStateVisitor::_emit_max(llvm::Value *a, llvm::Value *b, Type const &type) {
    if (type.is_float() || type.is_float_vector()) {
        auto *intrinsic = llvm::Intrinsic::getDeclarationIfExists(
            &_module, llvm::Intrinsic::maxnum, {a->getType()});
        return _builder.CreateCall(intrinsic, {a, b});
    } else if (type.is_int() || type.is_int_vector()) {
        auto *cmp = _builder.CreateICmpSGT(a, b);
        return _builder.CreateSelect(cmp, a, b);
    } else {
        auto *cmp = _builder.CreateICmpUGT(a, b);
        return _builder.CreateSelect(cmp, a, b);
    }
}

llvm::Value *LLVMStateVisitor::_emit_clamp(llvm::Value *v, llvm::Value *lo, llvm::Value *hi, Type const &type) {
    return _emit_max(_emit_min(v, hi, type), lo, type);
}

llvm::Value *LLVMStateVisitor::_emit_lerp(llvm::Value *a, llvm::Value *b, llvm::Value *t, Type const &type) {
    // a + t * (b - a)
    auto *diff = _builder.CreateFSub(b, a);
    auto *scaled = _builder.CreateFMul(t, diff);
    return _builder.CreateFAdd(a, scaled);
}

llvm::Value *LLVMStateVisitor::_emit_dot(llvm::Value *a, llvm::Value *b) {
    auto *vec_ty = llvm::dyn_cast<llvm::FixedVectorType>(a->getType());
    if (!vec_ty) {
        return _builder.CreateFMul(a, b);
    }
    auto *result = _builder.CreateFMul(
        _builder.CreateExtractElement(a, uint64_t(0)),
        _builder.CreateExtractElement(b, uint64_t(0)));
    for (unsigned i = 1; i < vec_ty->getNumElements(); ++i) {
        auto *prod = _builder.CreateFMul(
            _builder.CreateExtractElement(a, i),
            _builder.CreateExtractElement(b, i));
        result = _builder.CreateFAdd(result, prod);
    }
    return result;
}

llvm::Value *LLVMStateVisitor::_emit_length(llvm::Value *v) {
    auto *dot = _emit_dot(v, v);
    auto *intrinsic = llvm::Intrinsic::getDeclarationIfExists(
        &_module, llvm::Intrinsic::sqrt, {dot->getType()});
    return _builder.CreateCall(intrinsic, {dot});
}

llvm::Value *LLVMStateVisitor::_emit_normalize(llvm::Value *v) {
    auto *len = _emit_length(v);
    // Broadcast scalar length to vector
    if (auto *vec_ty = llvm::dyn_cast<llvm::FixedVectorType>(v->getType())) {
        llvm::Value *broadcast_len = llvm::UndefValue::get(vec_ty);
        for (unsigned i = 0; i < vec_ty->getNumElements(); ++i) {
            broadcast_len = _builder.CreateInsertElement(broadcast_len, len, i);
        }
        return _builder.CreateFDiv(v, broadcast_len);
    }
    return _builder.CreateFDiv(v, len);
}

llvm::Value *LLVMStateVisitor::_emit_cross(llvm::Value *a, llvm::Value *b) {
    // cross product for float3
    auto *a0 = _builder.CreateExtractElement(a, uint64_t(0));
    auto *a1 = _builder.CreateExtractElement(a, uint64_t(1));
    auto *a2 = _builder.CreateExtractElement(a, uint64_t(2));
    auto *b0 = _builder.CreateExtractElement(b, uint64_t(0));
    auto *b1 = _builder.CreateExtractElement(b, uint64_t(1));
    auto *b2 = _builder.CreateExtractElement(b, uint64_t(2));

    auto *x = _builder.CreateFSub(_builder.CreateFMul(a1, b2), _builder.CreateFMul(a2, b1));
    auto *y = _builder.CreateFSub(_builder.CreateFMul(a2, b0), _builder.CreateFMul(a0, b2));
    auto *z = _builder.CreateFSub(_builder.CreateFMul(a0, b1), _builder.CreateFMul(a1, b0));

    llvm::Value *result = llvm::UndefValue::get(llvm::FixedVectorType::get(_builder.getFloatTy(), 3));
    result = _builder.CreateInsertElement(result, x, uint64_t(0));
    result = _builder.CreateInsertElement(result, y, uint64_t(1));
    result = _builder.CreateInsertElement(result, z, uint64_t(2));
    return result;
}

llvm::Value *LLVMStateVisitor::_emit_all(llvm::Value *v) {
    // Reduce vector<bool> via AND chain
    auto *vec_ty = llvm::dyn_cast<llvm::FixedVectorType>(v->getType());
    if (!vec_ty) return v;
    auto *result = _builder.CreateExtractElement(v, uint64_t(0));
    for (unsigned i = 1; i < vec_ty->getNumElements(); ++i) {
        result = _builder.CreateAnd(result, _builder.CreateExtractElement(v, i));
    }
    return result;
}

llvm::Value *LLVMStateVisitor::_emit_any(llvm::Value *v) {
    auto *vec_ty = llvm::dyn_cast<llvm::FixedVectorType>(v->getType());
    if (!vec_ty) return v;
    auto *result = _builder.CreateExtractElement(v, uint64_t(0));
    for (unsigned i = 1; i < vec_ty->getNumElements(); ++i) {
        result = _builder.CreateOr(result, _builder.CreateExtractElement(v, i));
    }
    return result;
}

// ============================================================================
// Statement Visitors
// ============================================================================

void LLVMStateVisitor::visit(const BreakStmt *) {
    if (_util->opt->loop_stack.empty()) {
        LUISA_ERROR_WITH_LOCATION("Break outside loop.");
    }
    _builder.CreateBr(_util->opt->loop_stack.back().break_target);
}

void LLVMStateVisitor::visit(const ContinueStmt *) {
    if (_util->opt->loop_stack.empty()) {
        LUISA_ERROR_WITH_LOCATION("Continue outside loop.");
    }
    _builder.CreateBr(_util->opt->loop_stack.back().continue_target);
}

void LLVMStateVisitor::visit(const ReturnStmt *stmt) {
    if (stmt->expression()) {
        auto *val = EvalExpr(stmt->expression());
        _builder.CreateRet(val);
    } else {
        _builder.CreateRetVoid();
    }
}

void LLVMStateVisitor::visit(const ScopeStmt *stmt) {
    for (auto *s : stmt->statements()) {
        s->accept(*this);
        // If the current block has a terminator, stop
        if (_builder.GetInsertBlock()->getTerminator()) {
            break;
        }
    }
}

void LLVMStateVisitor::visit(const ExprStmt *stmt) {
    EvalExpr(stmt->expression()); // discard result
}

void LLVMStateVisitor::visit(const AssignStmt *stmt) {
    auto *rhs = EvalExpr(stmt->rhs());
    // LHS must be a RefExpr (reference to a variable)
    auto *lhs = stmt->lhs();
    if (lhs->tag() == Expression::Tag::REF) {
        auto *ref = static_cast<RefExpr const *>(lhs);
        auto uid = ref->variable().uid();
        StoreVariable(uid, rhs);
    } else {
        LUISA_ERROR_WITH_LOCATION("Assignment LHS must be a reference to a variable.");
    }
    _last_value = rhs;
}

void LLVMStateVisitor::visit(const IfStmt *stmt) {
    auto *cond = EvalExpr(stmt->condition());

    // Ensure cond is i1
    if (!cond->getType()->isIntegerTy(1)) {
        cond = _builder.CreateICmpNE(cond,
            llvm::ConstantInt::get(cond->getType(), 0));
    }

    auto *func = _builder.GetInsertBlock()->getParent();
    auto *then_bb = llvm::BasicBlock::Create(_ctx, "if_then", func);
    auto *else_bb = stmt->false_branch() && !stmt->false_branch()->statements().empty()
        ? llvm::BasicBlock::Create(_ctx, "if_else", func)
        : nullptr;
    auto *merge_bb = llvm::BasicBlock::Create(_ctx, "if_merge", func);

    if (else_bb) {
        _builder.CreateCondBr(cond, then_bb, else_bb);
    } else {
        _builder.CreateCondBr(cond, then_bb, merge_bb);
    }

    // Then branch
    _builder.SetInsertPoint(then_bb);
    stmt->true_branch()->accept(*this);
    if (!_builder.GetInsertBlock()->getTerminator()) {
        _builder.CreateBr(merge_bb);
    }

    // Else branch
    if (else_bb) {
        _builder.SetInsertPoint(else_bb);
        stmt->false_branch()->accept(*this);
        if (!_builder.GetInsertBlock()->getTerminator()) {
            _builder.CreateBr(merge_bb);
        }
    }

    // Merge
    _builder.SetInsertPoint(merge_bb);
}

void LLVMStateVisitor::visit(const LoopStmt *stmt) {
    auto *func = _builder.GetInsertBlock()->getParent();
    auto *loop_header = llvm::BasicBlock::Create(_ctx, "loop_header", func);
    auto *loop_body = llvm::BasicBlock::Create(_ctx, "loop_body", func);
    auto *loop_exit = llvm::BasicBlock::Create(_ctx, "loop_exit", func);

    _push_loop(loop_exit, loop_header);

    _builder.CreateBr(loop_header);

    _builder.SetInsertPoint(loop_header);
    _builder.CreateBr(loop_body);

    _builder.SetInsertPoint(loop_body);
    stmt->body()->accept(*this);
    if (!_builder.GetInsertBlock()->getTerminator()) {
        _builder.CreateBr(loop_header);
    }

    _pop_loop();
    _builder.SetInsertPoint(loop_exit);
}

void LLVMStateVisitor::visit(const ForStmt *stmt) {
    auto *func = _builder.GetInsertBlock()->getParent();
    auto *for_cond = llvm::BasicBlock::Create(_ctx, "for_cond", func);
    auto *for_body = llvm::BasicBlock::Create(_ctx, "for_body", func);
    auto *for_step = llvm::BasicBlock::Create(_ctx, "for_step", func);
    auto *for_exit = llvm::BasicBlock::Create(_ctx, "for_exit", func);

    // Extract loop variable from the variable expression (should be a RefExpr)
    auto *var_expr = stmt->variable();
    uint32_t loop_var_uid = 0;
    if (var_expr->tag() == Expression::Tag::REF) {
        loop_var_uid = static_cast<RefExpr const *>(var_expr)->variable().uid();
    }

    _push_loop(for_exit, for_step);

    _builder.CreateBr(for_cond);

    // Condition
    _builder.SetInsertPoint(for_cond);
    auto *cond = EvalExpr(stmt->condition());
    if (!cond->getType()->isIntegerTy(1)) {
        cond = _builder.CreateICmpNE(cond,
            llvm::ConstantInt::get(cond->getType(), 0));
    }
    _builder.CreateCondBr(cond, for_body, for_exit);

    // Body
    _builder.SetInsertPoint(for_body);
    stmt->body()->accept(*this);
    if (!_builder.GetInsertBlock()->getTerminator()) {
        _builder.CreateBr(for_step);
    }

    // Step: evaluate and assign back to loop variable
    _builder.SetInsertPoint(for_step);
    auto *step_val = EvalExpr(stmt->step());
    if (loop_var_uid != 0) {
        StoreVariable(loop_var_uid, step_val);
    }
    _builder.CreateBr(for_cond);

    _pop_loop();
    _builder.SetInsertPoint(for_exit);
}

void LLVMStateVisitor::visit(const SwitchStmt *stmt) {
    auto *expr = EvalExpr(stmt->expression());
    auto *func = _builder.GetInsertBlock()->getParent();
    auto *merge_bb = llvm::BasicBlock::Create(_ctx, "switch_merge", func);

    // Collect case information
    struct CaseInfo {
        SwitchCaseStmt const *case_stmt;
        llvm::ConstantInt *value;
        llvm::BasicBlock *block;
    };
    luisa::vector<CaseInfo> cases;
    SwitchDefaultStmt const *default_stmt = nullptr;
    llvm::BasicBlock *default_bb = nullptr;

    for (auto *s : stmt->body()->statements()) {
        if (s->tag() == Statement::Tag::SWITCH_CASE) {
            auto *case_stmt = static_cast<SwitchCaseStmt const *>(s);
            auto *case_val = EvalExpr(case_stmt->expression());
            auto *case_const = llvm::dyn_cast<llvm::ConstantInt>(case_val);
            if (!case_const) {
                LUISA_ERROR_WITH_LOCATION("Switch case value must be a constant integer.");
            }
            auto *case_bb = llvm::BasicBlock::Create(_ctx, "switch_case", func);
            cases.push_back({case_stmt, case_const, case_bb});
        } else if (s->tag() == Statement::Tag::SWITCH_DEFAULT) {
            default_stmt = static_cast<SwitchDefaultStmt const *>(s);
            default_bb = llvm::BasicBlock::Create(_ctx, "switch_default", func);
        }
    }

    // If no explicit default, use merge block as default
    if (!default_stmt) {
        default_bb = merge_bb;
    }

    // Create the switch instruction
    auto *sw = _builder.CreateSwitch(expr, default_bb, static_cast<unsigned>(cases.size()));
    for (auto &ci : cases) {
        sw->addCase(ci.value, ci.block);
    }

    _switch_merge_block = merge_bb;
    _current_switch = sw;

    // Visit each case body
    for (auto &ci : cases) {
        _builder.SetInsertPoint(ci.block);
        ci.case_stmt->accept(*this);
        if (!_builder.GetInsertBlock()->getTerminator()) {
            _builder.CreateBr(merge_bb);
        }
    }

    // Visit default body
    if (default_stmt) {
        _builder.SetInsertPoint(default_bb);
        default_stmt->accept(*this);
        if (!_builder.GetInsertBlock()->getTerminator()) {
            _builder.CreateBr(merge_bb);
        }
    }

    _current_switch = nullptr;
    _switch_merge_block = nullptr;
    _builder.SetInsertPoint(merge_bb);
}

void LLVMStateVisitor::visit(const SwitchCaseStmt *stmt) {
    for (auto *s : stmt->body()->statements()) {
        s->accept(*this);
        if (_builder.GetInsertBlock()->getTerminator()) break;
    }
}

void LLVMStateVisitor::visit(const SwitchDefaultStmt *stmt) {
    for (auto *s : stmt->body()->statements()) {
        s->accept(*this);
        if (_builder.GetInsertBlock()->getTerminator()) break;
    }
}

void LLVMStateVisitor::visit(const CommentStmt *) {
    // No-op for now; could emit debug metadata later
}

void LLVMStateVisitor::visit(const RayQueryStmt *) {
    LUISA_NOT_IMPLEMENTED();
}

void LLVMStateVisitor::visit(const AutoDiffStmt *stmt) {
    // Autodiff handled at IR level; just visit body
    stmt->body()->accept(*this);
}

void LLVMStateVisitor::visit(const PrintStmt *) {
    LUISA_NOT_IMPLEMENTED();
}

void LLVMStateVisitor::visit(const DebugBreakStmt *) {
    LUISA_NOT_IMPLEMENTED();
}

} // namespace lc::llvm_codegen
