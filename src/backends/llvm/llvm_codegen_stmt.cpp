//
// Created by Mike Smith on 2022/6/6.
//

#include <backends/llvm/llvm_codegen.h>

namespace luisa::compute::llvm {

void LLVMCodegen::visit(const BreakStmt *stmt) {
    auto ctx = _current_context();
    LUISA_ASSERT(!ctx->break_targets.empty(),
                 "Invalid break statement.");
    auto br = ctx->builder->CreateBr(ctx->break_targets.back());
}

void LLVMCodegen::visit(const ContinueStmt *stmt) {
    auto ctx = _current_context();
    LUISA_ASSERT(!ctx->continue_targets.empty(),
                 "Invalid continue statement.");
    auto br = ctx->builder->CreateBr(ctx->continue_targets.back());
}

void LLVMCodegen::visit(const ReturnStmt *stmt) {
    auto ctx = _current_context();
    if (auto ret_val = stmt->expression()) {
        _create_assignment(
            ctx->function.return_type(), stmt->expression()->type(),
            ctx->ret, _create_expr(stmt->expression()));
    }
    auto br = ctx->builder->CreateBr(ctx->exit_block);
}

void LLVMCodegen::visit(const ScopeStmt *stmt) {
    for (auto s : stmt->statements()) {
        s->accept(*this);
        if (auto block = _current_context()->builder->GetInsertBlock();
            block->getTerminator() != nullptr /* terminated */) {
            break;// remaining code is dead
        }
    }
}

void LLVMCodegen::visit(const IfStmt *stmt) {
    auto ctx = _current_context();
    auto cond = _create_expr(stmt->condition());
    cond = _scalar_to_bool(stmt->condition()->type(), cond);
    cond = ctx->builder->CreateICmpNE(
        ctx->builder->CreateLoad(_create_type(Type::of<bool>()), cond, "cond"),
        ctx->builder->getInt8(0), "cond.cmp");
    auto then_block = ::llvm::BasicBlock::Create(_context, "if.then", ctx->ir);
    auto else_block = ::llvm::BasicBlock::Create(_context, "if.else", ctx->ir);
    auto end_block = ::llvm::BasicBlock::Create(_context, "if.end", ctx->ir);
    ctx->builder->CreateCondBr(cond, then_block, else_block);
    // true branch
    then_block->moveAfter(ctx->builder->GetInsertBlock());
    ctx->builder->SetInsertPoint(then_block);
    stmt->true_branch()->accept(*this);
    if (ctx->builder->GetInsertBlock()->getTerminator() == nullptr) {
        ctx->builder->CreateBr(end_block);
    }
    // false branch
    else_block->moveAfter(ctx->builder->GetInsertBlock());
    ctx->builder->SetInsertPoint(else_block);
    stmt->false_branch()->accept(*this);
    end_block->moveAfter(ctx->builder->GetInsertBlock());
    if (ctx->builder->GetInsertBlock()->getTerminator() == nullptr) {
        ctx->builder->CreateBr(end_block);
    }
    // end
    ctx->builder->SetInsertPoint(end_block);
}

void LLVMCodegen::visit(const LoopStmt *stmt) {
    auto ctx = _current_context();
    auto loop_block = ::llvm::BasicBlock::Create(_context, "loop", ctx->ir);
    auto loop_exit_block = ::llvm::BasicBlock::Create(_context, "loop.exit", ctx->ir);
    loop_block->moveAfter(ctx->builder->GetInsertBlock());
    ctx->continue_targets.emplace_back(loop_block);
    ctx->break_targets.emplace_back(loop_exit_block);
    ctx->builder->CreateBr(loop_block);
    ctx->builder->SetInsertPoint(loop_block);
    stmt->body()->accept(*this);
    if (ctx->builder->GetInsertBlock()->getTerminator() == nullptr) {
        ctx->builder->CreateBr(loop_block);
    }
    ctx->continue_targets.pop_back();
    ctx->break_targets.pop_back();
    loop_exit_block->moveAfter(ctx->builder->GetInsertBlock());
    ctx->builder->SetInsertPoint(loop_exit_block);
}

void LLVMCodegen::visit(const ExprStmt *stmt) {
    static_cast<void>(_create_expr(stmt->expression()));
}

void LLVMCodegen::visit(const SwitchStmt *stmt) {
    auto ctx = _current_context();
    auto t = ctx->builder->getInt32Ty();
    auto cond = ctx->builder->CreateLoad(t, _create_expr(stmt->expression()), "switch.value");
    auto end_block = ::llvm::BasicBlock::Create(_context, "switch.end", ctx->ir);
    auto inst = ctx->builder->CreateSwitch(cond, end_block);
    ctx->break_targets.emplace_back(end_block);
    ctx->switch_stack.emplace_back(inst);
    for (auto c : stmt->body()->statements()) { c->accept(*this); }
    ctx->break_targets.pop_back();
    ctx->switch_stack.pop_back();
    end_block->moveAfter(ctx->builder->GetInsertBlock());
    ctx->builder->SetInsertPoint(end_block);
}

void LLVMCodegen::visit(const SwitchCaseStmt *stmt) {
    LUISA_ASSERT(stmt->expression()->tag() == Expression::Tag::LITERAL,
                 "Switch case expression must be a literal.");
    auto v = static_cast<const LiteralExpr *>(stmt->expression())->value();
    LUISA_ASSERT(luisa::holds_alternative<int>(v) || luisa::holds_alternative<uint>(v),
                 "Switch case expression must be an integer.");
    auto value = luisa::holds_alternative<int>(v) ? luisa::get<int>(v) : luisa::get<uint>(v);
    auto ctx = _current_context();
    LUISA_ASSERT(!ctx->switch_stack.empty(), "Case outside of switch.");
    auto case_block = ::llvm::BasicBlock::Create(_context, "switch.case", ctx->ir);
    case_block->moveAfter(ctx->builder->GetInsertBlock());
    ctx->switch_stack.back()->addCase(ctx->builder->getInt32(value), case_block);
    ctx->builder->SetInsertPoint(case_block);
    stmt->body()->accept(*this);
    if (ctx->builder->GetInsertBlock()->getTerminator() == nullptr) {
        ctx->builder->CreateBr(ctx->break_targets.back());
    }
}

void LLVMCodegen::visit(const SwitchDefaultStmt *stmt) {
    auto ctx = _current_context();
    LUISA_ASSERT(!ctx->switch_stack.empty(), "Default case outside of switch.");
    auto default_block = ::llvm::BasicBlock::Create(_context, "switch.default", ctx->ir);
    ctx->switch_stack.back()->setDefaultDest(default_block);
    default_block->moveAfter(ctx->builder->GetInsertBlock());
    ctx->builder->SetInsertPoint(default_block);
    stmt->body()->accept(*this);
    if (ctx->builder->GetInsertBlock()->getTerminator() == nullptr) {
        ctx->builder->CreateBr(ctx->break_targets.back());
    }
}

void LLVMCodegen::visit(const AssignStmt *stmt) {
    _create_assignment(
        stmt->lhs()->type(), stmt->rhs()->type(),
        _create_expr(stmt->lhs()), _create_expr(stmt->rhs()));
}

void LLVMCodegen::visit(const ForStmt *stmt) {
    auto var = _create_expr(stmt->variable());
    auto ctx = _current_context();
    auto loop_test = ::llvm::BasicBlock::Create(_context, "for.test", ctx->ir);
    auto loop_body = ::llvm::BasicBlock::Create(_context, "for.body", ctx->ir);
    auto loop_update = ::llvm::BasicBlock::Create(_context, "for.update", ctx->ir);
    auto loop_exit = ::llvm::BasicBlock::Create(_context, "for.exit", ctx->ir);
    loop_test->moveAfter(ctx->builder->GetInsertBlock());
    ctx->builder->CreateBr(loop_test);
    ctx->builder->SetInsertPoint(loop_test);
    auto cond = _create_expr(stmt->condition());
    cond = _scalar_to_bool(stmt->condition()->type(), cond);
    cond = ctx->builder->CreateICmpNE(
        ctx->builder->CreateLoad(ctx->builder->getInt8Ty(), cond, "for.cond"),
        ctx->builder->getInt8(0), "for.cond.cmp");
    ctx->builder->CreateCondBr(cond, loop_body, loop_exit);
    loop_body->moveAfter(ctx->builder->GetInsertBlock());
    ctx->builder->SetInsertPoint(loop_body);
    ctx->continue_targets.emplace_back(loop_update);
    ctx->break_targets.emplace_back(loop_exit);
    stmt->body()->accept(*this);
    if (ctx->builder->GetInsertBlock()->getTerminator() == nullptr) {
        ctx->builder->CreateBr(loop_update);
    }
    ctx->continue_targets.pop_back();
    ctx->break_targets.pop_back();
    loop_update->moveAfter(ctx->builder->GetInsertBlock());
    ctx->builder->SetInsertPoint(loop_update);
    auto vt = stmt->variable()->type();
    auto st = stmt->step()->type();
    auto step = _builtin_static_cast(vt, st, _create_expr(stmt->step()));
    auto t = _create_type(vt);
    auto next = [&] {
        switch (vt->tag()) {
            case Type::Tag::FLOAT:
                return ctx->builder->CreateFAdd(
                    ctx->builder->CreateLoad(t, var, "for.var"),
                    ctx->builder->CreateLoad(t, step, "for.var.step"), "for.var.next");
            case Type::Tag::INT:
                return ctx->builder->CreateNSWAdd(
                    ctx->builder->CreateLoad(t, var, "for.var"),
                    ctx->builder->CreateLoad(t, step, "for.var.step"), "for.var.next");
            case Type::Tag::UINT:
                return ctx->builder->CreateAdd(
                    ctx->builder->CreateLoad(t, var, "for.var"),
                    ctx->builder->CreateLoad(t, step, "for.var.step"), "for.var.next");
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION(
            "Invalid loop variable type: {}.",
            vt->description());
    }();
    ctx->builder->CreateStore(next, var);
    ctx->builder->CreateBr(loop_test);
    loop_exit->moveAfter(ctx->builder->GetInsertBlock());
    ctx->builder->SetInsertPoint(loop_exit);
}

void LLVMCodegen::visit(const CommentStmt *stmt) { /* do nothing */ }

void LLVMCodegen::visit(const MetaStmt *stmt) {
    auto ctx = _current_context();
    for (auto v : stmt->variables()) {
        if (v.tag() == Variable::Tag::LOCAL) {
            auto p = _create_alloca(_create_type(v.type()), _variable_name(v));
            ctx->variables.emplace(v.uid(), p);
            ctx->builder->CreateMemSet(
                p, ctx->builder->getInt8(0),
                v.type()->size(), ::llvm::Align{16});
        }
    }
    stmt->scope()->accept(*this);
}

void LLVMCodegen::_create_assignment(const Type *dst_type, const Type *src_type, ::llvm::Value *p_dst, ::llvm::Value *p_src) noexcept {
    auto p_rhs = _builtin_static_cast(dst_type, src_type, p_src);
    auto builder = _current_context()->builder.get();
    auto dst = builder->CreateLoad(_create_type(dst_type), p_rhs, "load");
    builder->CreateStore(dst, p_dst);
}

}// namespace luisa::compute::llvm
