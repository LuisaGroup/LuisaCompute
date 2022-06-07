//
// Created by Mike Smith on 2022/5/23.
//

#include <backends/llvm/llvm_codegen.h>

namespace luisa::compute::llvm {

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-static-cast-downcast"
::llvm::Value *LLVMCodegen::_create_expr(const Expression *expr) noexcept {
    switch (expr->tag()) {
        case Expression::Tag::UNARY:
            return _create_unary_expr(static_cast<const UnaryExpr *>(expr));
        case Expression::Tag::BINARY:
            return _create_binary_expr(static_cast<const BinaryExpr *>(expr));
        case Expression::Tag::MEMBER:
            return _create_member_expr(static_cast<const MemberExpr *>(expr));
        case Expression::Tag::ACCESS:
            return _create_access_expr(static_cast<const AccessExpr *>(expr));
        case Expression::Tag::LITERAL:
            return _create_literal_expr(static_cast<const LiteralExpr *>(expr));
        case Expression::Tag::REF:
            return _create_ref_expr(static_cast<const RefExpr *>(expr));
        case Expression::Tag::CONSTANT:
            return _create_constant_expr(static_cast<const ConstantExpr *>(expr));
        case Expression::Tag::CALL:
            return _create_call_expr(static_cast<const CallExpr *>(expr));
        case Expression::Tag::CAST:
            return _create_cast_expr(static_cast<const CastExpr *>(expr));
    }
    LUISA_ERROR_WITH_LOCATION("Invalid expression tag: {}.", to_underlying(expr->tag()));
}
#pragma clang diagnostic pop

::llvm::Value *LLVMCodegen::_create_unary_expr(const UnaryExpr *expr) noexcept {
    auto x = _create_expr(expr->operand());
    switch (expr->op()) {
        case UnaryOp::PLUS: return _builtin_unary_plus(expr->operand()->type(), x);
        case UnaryOp::MINUS: return _builtin_unary_minus(expr->operand()->type(), x);
        case UnaryOp::NOT: return _builtin_unary_not(expr->operand()->type(), x);
        case UnaryOp::BIT_NOT: return _builtin_unary_bit_not(expr->operand()->type(), x);
    }
    LUISA_ERROR_WITH_LOCATION("Invalid unary operator.");
}

::llvm::Value *LLVMCodegen::_short_circuit_and(const Expression *lhs, const Expression *rhs) noexcept {
    LUISA_ASSERT(lhs->type()->tag() == Type::Tag::BOOL &&
                     rhs->type()->tag() == Type::Tag::BOOL,
                 "Expected (bool && bool) but got ({} && {}).",
                 lhs->type()->description(), rhs->type()->description());
    auto ctx = _current_context();
    auto value = _create_expr(lhs);
    auto lhs_is_true = ctx->builder->CreateICmpEQ(
        ctx->builder->CreateLoad(_create_type(lhs->type()), value, "load"),
        ctx->builder->getInt8(0), "cmp");
    auto next_block = ::llvm::BasicBlock::Create(_context, "and.next", ctx->ir);
    auto out_block = ::llvm::BasicBlock::Create(_context, "and.out", ctx->ir);
    next_block->moveAfter(ctx->builder->GetInsertBlock());
    ctx->builder->CreateCondBr(lhs_is_true, next_block, out_block);
    ctx->builder->SetInsertPoint(next_block);
    auto rhs_value = _create_expr(rhs);
    _create_assignment(Type::of<bool>(), rhs->type(), value, _create_expr(rhs));
    out_block->moveAfter(ctx->builder->GetInsertBlock());
    ctx->builder->CreateBr(out_block);
    ctx->builder->SetInsertPoint(out_block);
    return value;
}

::llvm::Value *LLVMCodegen::_short_circuit_or(const Expression *lhs, const Expression *rhs) noexcept {
    LUISA_ASSERT(lhs->type()->tag() == Type::Tag::BOOL &&
                     rhs->type()->tag() == Type::Tag::BOOL,
                 "Expected (bool || bool) but got ({} || {}).",
                 lhs->type()->description(), rhs->type()->description());
    auto ctx = _current_context();
    auto value = _create_expr(lhs);
    auto lhs_is_true = ctx->builder->CreateICmpEQ(
        ctx->builder->CreateLoad(_create_type(lhs->type()), value, "load"),
        ctx->builder->getInt8(0), "cmp");
    auto next_block = ::llvm::BasicBlock::Create(_context, "or.next", ctx->ir);
    auto out_block = ::llvm::BasicBlock::Create(_context, "or.out", ctx->ir);
    next_block->moveAfter(ctx->builder->GetInsertBlock());
    ctx->builder->CreateCondBr(lhs_is_true, out_block, next_block);
    ctx->builder->SetInsertPoint(next_block);
    auto rhs_value = _create_expr(rhs);
    _create_assignment(Type::of<bool>(), rhs->type(), value, _create_expr(rhs));
    out_block->moveAfter(ctx->builder->GetInsertBlock());
    ctx->builder->CreateBr(out_block);
    ctx->builder->SetInsertPoint(out_block);
    return value;
}

::llvm::Value *LLVMCodegen::_create_binary_expr(const BinaryExpr *expr) noexcept {
    auto ctx = _current_context();
    auto lhs_type = expr->lhs()->type();
    auto rhs_type = expr->rhs()->type();
    // logical and/or should be short-circuit.
    if (lhs_type->is_scalar() && rhs_type->is_scalar()) {
        if (expr->op() == BinaryOp::AND) { return _short_circuit_and(expr->lhs(), expr->rhs()); }
        if (expr->op() == BinaryOp::OR) { return _short_circuit_or(expr->lhs(), expr->rhs()); }
    }
    // matrices have to be handled separately
    auto p_lhs = _create_expr(expr->lhs());
    auto p_rhs = _create_expr(expr->rhs());
    if (lhs_type->is_matrix() && rhs_type->is_matrix()) {
        switch (expr->op()) {
            case BinaryOp::ADD: return _builtin_add_matrix_matrix(lhs_type, rhs_type, p_lhs, p_rhs);
            case BinaryOp::SUB: return _builtin_sub_matrix_matrix(lhs_type, rhs_type, p_lhs, p_rhs);
            case BinaryOp::MUL: return _builtin_mul_matrix_matrix(lhs_type, rhs_type, p_lhs, p_rhs);
            default: LUISA_ERROR_WITH_LOCATION("Invalid binary matrix-matrix operator.");
        }
    }
    if (lhs_type->is_matrix() && rhs_type->is_scalar()) {
        p_rhs = _scalar_to_float(rhs_type, p_rhs);
        switch (expr->op()) {
            case BinaryOp::ADD: return _builtin_add_matrix_scalar(lhs_type, rhs_type, p_lhs, p_rhs);
            case BinaryOp::SUB: return _builtin_sub_matrix_scalar(lhs_type, rhs_type, p_lhs, p_rhs);
            case BinaryOp::MUL: return _builtin_mul_matrix_scalar(lhs_type, rhs_type, p_lhs, p_rhs);
            case BinaryOp::DIV: return _builtin_div_matrix_scalar(lhs_type, rhs_type, p_lhs, p_rhs);
            default: LUISA_ERROR_WITH_LOCATION("Invalid binary matrix-scalar operator.");
        }
    }
    if (lhs_type->is_scalar() && rhs_type->is_matrix()) {
        p_lhs = _scalar_to_float(lhs_type, p_lhs);
        switch (expr->op()) {
            case BinaryOp::ADD: return _builtin_add_scalar_matrix(lhs_type, rhs_type, p_lhs, p_rhs);
            case BinaryOp::SUB: return _builtin_sub_scalar_matrix(lhs_type, rhs_type, p_lhs, p_rhs);
            case BinaryOp::MUL: return _builtin_mul_scalar_matrix(lhs_type, rhs_type, p_lhs, p_rhs);
            case BinaryOp::DIV: return _builtin_div_scalar_matrix(lhs_type, rhs_type, p_lhs, p_rhs);
            default: LUISA_ERROR_WITH_LOCATION("Invalid binary scalar-matrix operator.");
        }
    }
    if (lhs_type->is_matrix() && rhs_type->is_vector()) {
        LUISA_ASSERT(expr->op() == BinaryOp::MUL, "Invalid binary matrix-vector operator.");
        LUISA_ASSERT(lhs_type->dimension() == rhs_type->dimension(),
                     "Dimensions mismatch in matrix-vector multiplication: {} vs {}.",
                     lhs_type->dimension(), rhs_type->dimension());
        p_rhs = _vector_to_float_vector(rhs_type, p_rhs);
        return _builtin_mul_matrix_vector(lhs_type, rhs_type, p_lhs, p_rhs);
    }
    // scalar/scalar or vector/vector or vector/scalar
    LUISA_ASSERT((lhs_type->is_scalar() || lhs_type->is_vector()) &&
                     (rhs_type->is_scalar() || rhs_type->is_vector()),
                 "Expected (scalar op vector) or (scalar op vector) but got ({} op {}).",
                 lhs_type->description(), rhs_type->description());
    auto lhs_elem_type = lhs_type->is_scalar() ? lhs_type : lhs_type->element();
    auto rhs_elem_type = rhs_type->is_scalar() ? rhs_type : rhs_type->element();
    auto promoted_elem_type = [&] {
        switch (lhs_elem_type->tag()) {
            case Type::Tag::BOOL: return rhs_elem_type;
            case Type::Tag::FLOAT: return lhs_elem_type;
            case Type::Tag::INT:
                return rhs_elem_type->tag() == Type::Tag::UINT ||
                               rhs_elem_type->tag() == Type::Tag::FLOAT ?
                           rhs_elem_type :
                           lhs_elem_type;
            case Type::Tag::UINT:
                return rhs_elem_type->tag() == Type::Tag::FLOAT ?
                           rhs_elem_type :
                           lhs_elem_type;
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION(
            "Invalid types '{}' and '{}' for binary operator.",
            lhs_elem_type->description(), rhs_elem_type->description());
    }();
    auto promoted_type = expr->type()->is_vector() ?
                             Type::from(luisa::format(
                                 "vector<{},{}>",
                                 promoted_elem_type->description(),
                                 expr->type()->dimension())) :
                             promoted_elem_type;
    auto lhs_v = _builtin_static_cast(promoted_type, lhs_type, p_lhs);
    auto rhs_v = _builtin_static_cast(promoted_type, rhs_type, p_rhs);
    switch (expr->op()) {
        case BinaryOp::ADD: return _builtin_add(promoted_type, lhs_v, rhs_v);
        case BinaryOp::SUB: return _builtin_sub(promoted_type, lhs_v, rhs_v);
        case BinaryOp::MUL: return _builtin_mul(promoted_type, lhs_v, rhs_v);
        case BinaryOp::DIV: return _builtin_div(promoted_type, lhs_v, rhs_v);
        case BinaryOp::MOD: return _builtin_mod(promoted_type, lhs_v, rhs_v);
        case BinaryOp::BIT_AND: return _builtin_and(promoted_type, lhs_v, rhs_v);
        case BinaryOp::BIT_OR: return _builtin_or(promoted_type, lhs_v, rhs_v);
        case BinaryOp::BIT_XOR: return _builtin_xor(promoted_type, lhs_v, rhs_v);
        case BinaryOp::SHL: return _builtin_shl(promoted_type, lhs_v, rhs_v);
        case BinaryOp::SHR: return _builtin_shr(promoted_type, lhs_v, rhs_v);
        case BinaryOp::AND: return _builtin_and(promoted_type, lhs_v, rhs_v);
        case BinaryOp::OR: return _builtin_or(promoted_type, lhs_v, rhs_v);
        case BinaryOp::LESS: return _builtin_lt(promoted_type, lhs_v, rhs_v);
        case BinaryOp::GREATER: return _builtin_gt(promoted_type, lhs_v, rhs_v);
        case BinaryOp::LESS_EQUAL: return _builtin_le(promoted_type, lhs_v, rhs_v);
        case BinaryOp::GREATER_EQUAL: return _builtin_ge(promoted_type, lhs_v, rhs_v);
        case BinaryOp::EQUAL: return _builtin_eq(promoted_type, lhs_v, rhs_v);
        case BinaryOp::NOT_EQUAL: return _builtin_ne(promoted_type, lhs_v, rhs_v);
    }
    LUISA_ERROR_WITH_LOCATION("Invalid binary operator.");
}

::llvm::Value *LLVMCodegen::_create_member_expr(const MemberExpr *expr) noexcept {
    auto ctx = _current_context();
    auto self = _create_expr(expr->self());
    auto self_type = expr->self()->type();
    if (self_type->is_structure()) {
        auto member_index = _struct_types.at(self_type->hash())
                                .member_indices.at(expr->member_index());
        return ctx->builder->CreateStructGEP(
            _create_type(self_type), self, member_index,
            "struct.member.addr");
    }
    LUISA_ASSERT(expr->self()->type()->is_vector() && expr->is_swizzle(),
                 "Invalid member expression. Vector swizzling expected.");
    switch (expr->swizzle_size()) {
        case 1u: {
            auto idx = static_cast<uint>(expr->swizzle_index(0u));
            auto elem_type = _create_type(expr->type());
            auto ptr_type = ::llvm::PointerType::get(elem_type, 0);
            auto ptr = ctx->builder->CreateBitOrPointerCast(self, ptr_type, "vector.member.ptr");
            return ctx->builder->CreateConstInBoundsGEP1_32(elem_type, ptr, idx, "vector.member.addr");
        }
        case 2u: return _create_stack_variable(
            ctx->builder->CreateShuffleVector(
                ctx->builder->CreateLoad(
                    _create_type(self_type), self, "vector.member.load"),
                {static_cast<int>(expr->swizzle_index(0u)),
                 static_cast<int>(expr->swizzle_index(1u))},
                "vector.swizzle"),
            "vector.swizzle.addr");
        case 3u: return _create_stack_variable(
            ctx->builder->CreateShuffleVector(
                ctx->builder->CreateLoad(
                    _create_type(self_type), self, "vector.member.load"),
                {static_cast<int>(expr->swizzle_index(0u)),
                 static_cast<int>(expr->swizzle_index(1u)),
                 static_cast<int>(expr->swizzle_index(2u)), 0},
                "vector.swizzle"),
            "vector.swizzle.addr");
        case 4u: return _create_stack_variable(
            ctx->builder->CreateShuffleVector(
                ctx->builder->CreateLoad(
                    _create_type(self_type), self, "vector.member.load"),
                {static_cast<int>(expr->swizzle_index(0u)),
                 static_cast<int>(expr->swizzle_index(1u)),
                 static_cast<int>(expr->swizzle_index(2u)),
                 static_cast<int>(expr->swizzle_index(3u))},
                "vector.swizzle"),
            "vector.swizzle.addr");
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid swizzle size: {}.", expr->swizzle_size());
}

::llvm::Value *LLVMCodegen::_create_access_expr(const AccessExpr *expr) noexcept {
    auto ctx = _current_context();
    auto range = _create_expr(expr->range());
    auto index = ctx->builder->CreateLoad(
        _create_type(Type::of<uint>()),
        _builtin_static_cast(
            Type::of<uint>(), expr->index()->type(),
            _create_expr(expr->index())),
        "access.index");
    auto elem_type = _create_type(expr->type());
    auto ptr_type = ::llvm::PointerType::get(elem_type, 0);
    auto range_type = expr->range()->type();
    if (range_type->is_buffer()) {
        return ctx->builder->CreateInBoundsGEP(elem_type, range, index, "access.addr");
    }
    if (range_type->is_vector()) {
        auto ptr = ctx->builder->CreateBitOrPointerCast(range, ptr_type, "vector.ptr");
        return ctx->builder->CreateInBoundsGEP(elem_type, ptr, index, "access.addr");
    }
    LUISA_ASSERT(range_type->is_array() || range_type->is_matrix(),
                 "Invalid range type '{}'.", range_type->description());
    return ctx->builder->CreateInBoundsGEP(
        _create_type(range_type), range, {_literal(0u), index}, "access.addr");
}

::llvm::Value *LLVMCodegen::_create_literal_expr(const LiteralExpr *expr) noexcept {
    return luisa::visit(
        [this](auto x) noexcept {
            return _create_stack_variable(
                _literal(x), "const.addr");
        },
        expr->value());
}

::llvm::Value *LLVMCodegen::_create_ref_expr(const RefExpr *expr) noexcept {
    return _current_context()->variables.at(expr->variable().uid());
}

::llvm::Value *LLVMCodegen::_create_constant_expr(const ConstantExpr *expr) noexcept {
    return _create_constant(expr->data());
}

::llvm::Value *LLVMCodegen::_create_call_expr(const CallExpr *expr) noexcept {
    if (expr->is_builtin()) {
        return _create_builtin_call_expr(
            expr->type(), expr->op(), expr->arguments());
    }
    // custom
    auto f = _create_function(expr->custom());
    auto ctx = _current_context();
    luisa::vector<::llvm::Value *> args;
    for (auto i = 0u; i < expr->arguments().size(); i++) {
        auto arg = expr->arguments()[i];
        auto call_arg = _create_expr(arg);
        if (auto expected_arg = expr->custom().arguments()[i];
            expected_arg.type()->is_basic() ||
            expected_arg.type()->is_array() ||
            expected_arg.type()->is_structure()) {
            call_arg = _builtin_static_cast(expected_arg.type(), arg->type(), _create_expr(arg));
            if (auto usage = expr->custom().variable_usage(expected_arg.uid());
                expected_arg.tag() != Variable::Tag::REFERENCE ||
                (usage == Usage::NONE || usage == Usage::READ)) {
                call_arg = ctx->builder->CreateLoad(
                    _create_type(expected_arg.type()), call_arg, "load");
            }
        }
        args.emplace_back(call_arg);
    }
    ::llvm::ArrayRef<::llvm::Value *> args_ref{args.data(), args.size()};
    auto call = ctx->builder->CreateCall(f->getFunctionType(), f, args_ref);
    if (expr->type() == nullptr) { return call; }
    call->setName("result");
    return _create_stack_variable(call, "result.addr");
}

::llvm::Value *LLVMCodegen::_create_cast_expr(const CastExpr *expr) noexcept {
    if (expr->op() == CastOp::STATIC) {
        return _builtin_static_cast(
            expr->type(), expr->expression()->type(),
            _create_expr(expr->expression()));
    }
    return _builtin_bitwise_cast(
        expr->type(), expr->expression()->type(),
        _create_expr(expr->expression()));
}

}// namespace luisa::compute::llvm
