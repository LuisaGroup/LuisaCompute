//
// Created by Mike Smith on 2022/5/23.
//

#include <backends/llvm/llvm_codegen.h>

namespace luisa::compute::llvm {

LLVMCodegen::LLVMCodegen(::llvm::LLVMContext &ctx) noexcept
    : _context{ctx} {}

void LLVMCodegen::visit(const BreakStmt *stmt) {
    auto ctx = _current_context();
    LUISA_ASSERT(!ctx->break_targets.empty(),
                 "Invalid break statement.");
    ctx->builder->CreateBr(ctx->break_targets.back());
}

void LLVMCodegen::visit(const ContinueStmt *stmt) {
    auto ctx = _current_context();
    LUISA_ASSERT(!ctx->continue_targets.empty(),
                 "Invalid continue statement.");
    ctx->builder->CreateBr(ctx->continue_targets.back());
}

void LLVMCodegen::visit(const ReturnStmt *stmt) {
    auto ctx = _current_context();
    if (auto ret_val = stmt->expression()) {
        _create_assignment(
            ctx->function.return_type(), stmt->expression()->type(),
            ctx->ret, _create_expr(stmt->expression()));
    }
    ctx->builder->CreateBr(ctx->exit_block);
}

void LLVMCodegen::visit(const ScopeStmt *stmt) {
    for (auto s : stmt->statements()) { s->accept(*this); }
}

void LLVMCodegen::visit(const IfStmt *stmt) {
}

void LLVMCodegen::visit(const LoopStmt *stmt) {
}

void LLVMCodegen::visit(const ExprStmt *stmt) {
    static_cast<void>(_create_expr(stmt->expression()));
}

void LLVMCodegen::visit(const SwitchStmt *stmt) {
}

void LLVMCodegen::visit(const SwitchCaseStmt *stmt) {
}

void LLVMCodegen::visit(const SwitchDefaultStmt *stmt) {
}

void LLVMCodegen::visit(const AssignStmt *stmt) {
    _create_assignment(
        stmt->lhs()->type(), stmt->rhs()->type(),
        _create_expr(stmt->lhs()), _create_expr(stmt->rhs()));
}

void LLVMCodegen::visit(const ForStmt *stmt) {
}

void LLVMCodegen::visit(const CommentStmt *stmt) {
    // do nothing
}

void LLVMCodegen::visit(const MetaStmt *stmt) {
    auto ctx = _current_context();
    for (auto v : stmt->variables()) {
        if (v.tag() == Variable::Tag::LOCAL) {
            auto p = ctx->builder->CreateAlloca(
                _create_type(v.type()), nullptr,
                luisa::string_view{_variable_name(v)});
            ctx->variables.emplace(v.uid(), p);
        }
    }
    stmt->scope()->accept(*this);
}

void LLVMCodegen::_emit_function() noexcept {
    _function_stack.back()->function.body()->accept(*this);
}

luisa::unique_ptr<::llvm::Module> LLVMCodegen::emit(Function f) noexcept {
    auto module_name = luisa::format("module_{:016x}", f.hash());
    auto module = luisa::make_unique<::llvm::Module>(
        ::llvm::StringRef{module_name.data(), module_name.size()}, _context);
    _module = module.get();
    auto _ = _create_function(f);
    _module = nullptr;
    ::llvm::verifyModule(*module);
    return module;
}

unique_ptr<LLVMCodegen::FunctionContext> LLVMCodegen::_create_kernel_context(Function f) noexcept {
    auto create_argument_struct = [&] {
        auto member_index = 0u;
        luisa::vector<::llvm::Type *> field_types;
        luisa::vector<uint> field_indices;
        auto size = 0ul;
        static constexpr auto alignment = 16ul;
        for (auto &arg : f.arguments()) {
            auto aligned_offset = luisa::align(size, alignment);
            if (aligned_offset > size) {
                auto padding = ::llvm::ArrayType::get(
                    ::llvm::Type::getInt8Ty(_context), aligned_offset - size);
                field_types.emplace_back(padding);
                member_index++;
            }
            auto arg_size = [arg]() noexcept -> size_t {
                switch (arg.tag()) {
                    case Variable::Tag::REFERENCE: return 8u;
                    case Variable::Tag::BUFFER: return buffer_argument_size;
                    case Variable::Tag::TEXTURE: return texture_argument_size;
                    case Variable::Tag::BINDLESS_ARRAY: return bindless_array_argument_size;
                    case Variable::Tag::ACCEL: return accel_argument_size;
                    default: break;
                }
                return arg.type()->size();
            }();
            field_types.emplace_back(_create_type(arg.type()));
            field_indices.emplace_back(member_index++);
            size = aligned_offset + arg_size;
        }
        auto aligned_size = luisa::align(size, alignment);
        if (aligned_size > size) {// last padding
            auto padding = ::llvm::ArrayType::get(
                ::llvm::Type::getInt8Ty(_context), aligned_size - size);
            field_types.emplace_back(padding);
            member_index++;
        }
        // launch size
        field_types.emplace_back(_create_type(Type::of<uint3>()));
        field_indices.emplace_back(member_index++);
        ::llvm::ArrayRef<::llvm::Type *> fields_ref{field_types.data(), field_types.size()};
        return std::make_pair(
            ::llvm::StructType::get(_context, fields_ref),
            std::move(field_indices));
    };
    auto [arg_struct_type, arg_struct_indices] = create_argument_struct();
    auto arg_buffer_type = ::llvm::PointerType::get(arg_struct_type, 0);
    ::llvm::SmallVector<::llvm::Type *, 4u> arg_types;
    arg_types.emplace_back(arg_buffer_type);
    arg_types.emplace_back(::llvm::Type::getInt32Ty(_context));
    arg_types.emplace_back(::llvm::Type::getInt32Ty(_context));
    arg_types.emplace_back(::llvm::Type::getInt32Ty(_context));
    LUISA_ASSERT(f.return_type() == nullptr,
                 "Invalid return type '{}' for kernel. Only void is allowed.",
                 f.return_type()->description());
    auto function_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(_context), arg_types, false);
    auto name = _function_name(f).append("_driver");
    auto ir = ::llvm::Function::Create(
        function_type, ::llvm::Function::ExternalLinkage,
        ::llvm::StringRef{name.data(), name.size()}, _module);
    auto builder = luisa::make_unique<::llvm::IRBuilder<>>(_context);
    auto body_block = ::llvm::BasicBlock::Create(_context, "entry", ir);
    builder->SetInsertPoint(body_block);
    luisa::vector<::llvm::Value *> arguments;
    for (auto arg_id = 0u; arg_id < f.arguments().size(); arg_id++) {
        auto &arg = f.arguments()[arg_id];
        switch (arg.tag()) {
            case Variable::Tag::LOCAL:
            case Variable::Tag::BUFFER:
            case Variable::Tag::TEXTURE:
            case Variable::Tag::BINDLESS_ARRAY:
            case Variable::Tag::ACCEL: {
                auto pr = builder->CreateStructGEP(arg_struct_type, ir->getArg(0u), arg_struct_indices[arg_id]);
                auto r = builder->CreateAlignedLoad(_create_type(arg.type()), pr, ::llvm::Align{16ull});
                auto arg_name = _variable_name(arg);
                r->setName(::llvm::StringRef{arg_name.data(), arg_name.size()});
                arguments.emplace_back(r);
                break;
            }
            default: LUISA_ERROR_WITH_LOCATION("Invalid kernel argument type.");
        }
    }
    auto exit_block = ::llvm::BasicBlock::Create(_context, "exit", ir);
    builder->SetInsertPoint(exit_block);
    builder->CreateRetVoid();

    // create work function...
    auto ctx = _create_kernel_program(f);
    // create simple schedule...
    builder->SetInsertPoint(body_block);
    // block id
    auto block_id = static_cast<::llvm::Value *>(
        ::llvm::UndefValue::get(_create_type(Type::of<uint3>())));
    block_id = builder->CreateInsertElement(block_id, ir->getArg(1), 0ull);
    block_id = builder->CreateInsertElement(block_id, ir->getArg(2), 1ull);
    block_id = builder->CreateInsertElement(block_id, ir->getArg(3), 2ull, "block_id");
    // dispatch size
    auto p_dispatch_size = builder->CreateStructGEP(
        arg_struct_type, ir->getArg(0), arg_struct_indices.back());
    auto dispatch_size = builder->CreateAlignedLoad(
        _create_type(Type::of<uint3>()), p_dispatch_size,
        ::llvm::Align{16u}, "dispatch_size");
    // loop
    auto p_index = builder->CreateAlloca(::llvm::Type::getInt32Ty(_context), nullptr, "index");
    builder->CreateStore(builder->getInt32(0u), p_index);
    auto loop_block = ::llvm::BasicBlock::Create(_context, "loop", ir, exit_block);
    builder->CreateBr(loop_block);
    builder->SetInsertPoint(loop_block);
    auto index = builder->CreateLoad(::llvm::Type::getInt32Ty(_context), p_index, "index");
    auto thread_x = builder->CreateBinOp(::llvm::Instruction::URem, index, builder->getInt32(f.block_size().x), "thread_x");
    auto thread_yz = builder->CreateBinOp(::llvm::Instruction::UDiv, index, builder->getInt32(f.block_size().x));
    auto thread_y = builder->CreateBinOp(::llvm::Instruction::URem, thread_yz, builder->getInt32(f.block_size().y), "thread_y");
    auto thread_z = builder->CreateBinOp(::llvm::Instruction::UDiv, thread_yz, builder->getInt32(f.block_size().y), "thread_z");
    auto thread_id = static_cast<::llvm::Value *>(
        ::llvm::UndefValue::get(_create_type(Type::of<uint3>())));
    thread_id = builder->CreateInsertElement(thread_id, thread_x, 0ull);
    thread_id = builder->CreateInsertElement(thread_id, thread_y, 1ull);
    thread_id = builder->CreateInsertElement(thread_id, thread_z, 2ull, "thread_id");
    auto dispatch_x = builder->CreateNUWAdd(thread_x, builder->CreateNUWMul(ir->getArg(1), builder->getInt32(f.block_size().x)), "dispatch_x");
    auto dispatch_y = builder->CreateNUWAdd(thread_y, builder->CreateNUWMul(ir->getArg(2), builder->getInt32(f.block_size().y)), "dispatch_y");
    auto dispatch_z = builder->CreateNUWAdd(thread_z, builder->CreateNUWMul(ir->getArg(3), builder->getInt32(f.block_size().z)), "dispatch_z");
    auto dispatch_id = static_cast<::llvm::Value *>(
        ::llvm::UndefValue::get(_create_type(Type::of<uint3>())));
    dispatch_id = builder->CreateInsertElement(dispatch_id, dispatch_x, 0ull);
    dispatch_id = builder->CreateInsertElement(dispatch_id, dispatch_y, 1ull);
    dispatch_id = builder->CreateInsertElement(dispatch_id, dispatch_z, 2ull, "dispatch_id");
    auto valid_thread_xyz = builder->CreateICmpULT(thread_id, dispatch_size);
    auto valid_thread = builder->CreateLogicalAnd(
        builder->CreateLogicalAnd(
            builder->CreateExtractElement(valid_thread_xyz, 0ull),
            builder->CreateExtractElement(valid_thread_xyz, 1ull)),
        builder->CreateExtractElement(valid_thread_xyz, 2ull));
    auto call_block = ::llvm::BasicBlock::Create(_context, "call_kernel", ir, exit_block);
    auto loop_update_block = ::llvm::BasicBlock::Create(_context, "loop_update", ir, exit_block);
    builder->CreateCondBr(valid_thread, call_block, loop_update_block);
    builder->SetInsertPoint(call_block);
    auto call_args = arguments;
    call_args.emplace_back(thread_id);
    call_args.emplace_back(block_id);
    call_args.emplace_back(dispatch_id);
    call_args.emplace_back(dispatch_size);
    builder->CreateCall(ctx->ir->getFunctionType(), ctx->ir, ::llvm::ArrayRef<::llvm::Value *>{call_args.data(), call_args.size()});
    // update
    builder->CreateBr(loop_update_block);
    builder->SetInsertPoint(loop_update_block);
    auto next_index = builder->CreateAdd(index, builder->getInt32(1u), "next_index");
    builder->CreateStore(next_index, p_index);
    auto thread_count = f.block_size().x * f.block_size().y * f.block_size().z;
    auto should_continue = builder->CreateICmpULT(next_index, builder->getInt32(thread_count), "should_continue");
    builder->CreateCondBr(should_continue, loop_block, exit_block);
    return ctx;
}

luisa::unique_ptr<LLVMCodegen::FunctionContext> LLVMCodegen::_create_kernel_program(Function f) noexcept {
    luisa::vector<::llvm::Type *> arg_types;
    for (auto &&arg : f.arguments()) {
        switch (arg.tag()) {
            case Variable::Tag::LOCAL:
            case Variable::Tag::BUFFER:
            case Variable::Tag::TEXTURE:
            case Variable::Tag::BINDLESS_ARRAY:
            case Variable::Tag::ACCEL:
                arg_types.emplace_back(_create_type(arg.type()));
                break;
            default: LUISA_ERROR_WITH_LOCATION("Invalid kernel argument type.");
        }
    }
    // thread_id/block_id/dispatch_id/dispatch_size
    for (auto i = 0u; i < 4u; i++) {
        arg_types.emplace_back(_create_type(Type::of<uint3>()));
    }
    auto return_type = _create_type(f.return_type());
    ::llvm::ArrayRef<::llvm::Type *> arg_types_ref{arg_types.data(), arg_types.size()};
    auto function_type = ::llvm::FunctionType::get(return_type, arg_types_ref, false);
    auto name = _function_name(f);
    auto ir = ::llvm::Function::Create(
        function_type, ::llvm::Function::InternalLinkage,
        ::llvm::StringRef{name.data(), name.size()}, _module);
    auto builder = luisa::make_unique<::llvm::IRBuilder<>>(_context);
    auto body_block = ::llvm::BasicBlock::Create(_context, "entry", ir);
    auto exit_block = ::llvm::BasicBlock::Create(_context, "exit", ir);
    builder->SetInsertPoint(exit_block);
    builder->CreateRetVoid();
    builder->SetInsertPoint(body_block);
    luisa::unordered_map<uint, ::llvm::Value *> variables;
    auto make_alloca = [&](::llvm::Value *x, luisa::string_view name = "") noexcept {
        auto p = builder->CreateAlloca(x->getType(), nullptr, name);
        builder->CreateStore(x, p);
        return p;
    };
    for (auto i = 0u; i < f.arguments().size(); i++) {
        auto lc_arg = f.arguments()[i];
        auto arg_name = _variable_name(lc_arg);
        auto arg = static_cast<::llvm::Value *>(ir->getArg(i));
        if (lc_arg.tag() == Variable::Tag::LOCAL) { arg = make_alloca(arg); }
        arg->setName(luisa::string_view{_variable_name(lc_arg)});
        arg->setName(::llvm::StringRef{arg_name.data(), arg_name.size()});
        variables.emplace(lc_arg.uid(), arg);
    }
    auto builtin_offset = f.arguments().size();
    for (auto arg : f.builtin_variables()) {
        switch (arg.tag()) {
            case Variable::Tag::THREAD_ID:
                variables.emplace(arg.uid(), make_alloca(ir->getArg(builtin_offset + 0), _variable_name(arg)));
                break;
            case Variable::Tag::BLOCK_ID:
                variables.emplace(arg.uid(), make_alloca(ir->getArg(builtin_offset + 1), _variable_name(arg)));
                break;
            case Variable::Tag::DISPATCH_ID:
                variables.emplace(arg.uid(), make_alloca(ir->getArg(builtin_offset + 2), _variable_name(arg)));
                break;
            case Variable::Tag::DISPATCH_SIZE:
                variables.emplace(arg.uid(), make_alloca(ir->getArg(builtin_offset + 3), _variable_name(arg)));
                break;
            default: LUISA_ERROR_WITH_LOCATION("Invalid kernel argument type.");
        }
    }
    return luisa::make_unique<FunctionContext>(
        f, ir, nullptr, exit_block, std::move(builder), std::move(variables));
}

unique_ptr<LLVMCodegen::FunctionContext> LLVMCodegen::_create_callable_context(Function f) noexcept {
    auto is_out_reference = [&f](auto v) noexcept {
        return v.tag() == Variable::Tag::REFERENCE &&
               (f.variable_usage(v.uid()) == Usage::WRITE ||
                f.variable_usage(v.uid()) == Usage::READ_WRITE);
    };
    luisa::vector<::llvm::Type *> arg_types;
    for (auto &&arg : f.arguments()) {
        auto arg_type = _create_type(arg.type());
        if (is_out_reference(arg)) {
            arg_type = ::llvm::PointerType::get(arg_type, 0);
        }
        arg_types.emplace_back(arg_type);
    }
    auto return_type = _create_type(f.return_type());
    ::llvm::ArrayRef<::llvm::Type *> arg_types_ref{arg_types.data(), arg_types.size()};
    auto function_type = ::llvm::FunctionType::get(return_type, arg_types_ref, false);
    auto name = _function_name(f);
    auto ir = ::llvm::Function::Create(
        function_type, ::llvm::Function::InternalLinkage,
        ::llvm::StringRef{name.data(), name.size()}, _module);
    auto builder = luisa::make_unique<::llvm::IRBuilder<>>(_context);
    auto body_block = ::llvm::BasicBlock::Create(_context, "entry", ir);
    auto exit_block = ::llvm::BasicBlock::Create(_context, "exit", ir);
    builder->SetInsertPoint(body_block);
    auto i = 0u;
    luisa::unordered_map<uint, ::llvm::Value *> variables;
    for (auto &&arg : ir->args()) {
        auto lc_arg = f.arguments()[i++];
        auto arg_name = _variable_name(lc_arg);
        arg.setName(::llvm::StringRef{arg_name.data(), arg_name.size()});
        if (is_out_reference(lc_arg)) {
            variables.emplace(lc_arg.uid(), &arg);
        } else {
            auto p_arg = builder->CreateAlloca(arg.getType());
            builder->CreateStore(&arg, p_arg);
            variables.emplace(lc_arg.uid(), p_arg);
        }
    }
    ::llvm::Value *ret = nullptr;
    if (auto ret_type = f.return_type()) {
        builder->SetInsertPoint(body_block);
        ret = builder->CreateAlloca(return_type, nullptr, "retval");
        builder->SetInsertPoint(exit_block);
        builder->CreateRet(builder->CreateLoad(return_type, ret));
    } else {// return void
        builder->SetInsertPoint(exit_block);
        builder->CreateRetVoid();
    }
    builder->SetInsertPoint(body_block);
    return luisa::make_unique<FunctionContext>(
        f, ir, ret, exit_block, std::move(builder), std::move(variables));
}

::llvm::Function *LLVMCodegen::_create_function(Function f) noexcept {
    auto name = _function_name(f);
    ::llvm::StringRef name_ref{name.data(), name.size()};
    if (auto ir = _module->getFunction(name_ref)) { return ir; }
    _function_stack.emplace_back(
        f.tag() == Function::Tag::KERNEL ?
            _create_kernel_context(f) :
            _create_callable_context(f));
    _emit_function();
    auto ctx = std::move(_function_stack.back());
    _function_stack.pop_back();
    ctx->builder->CreateBr(ctx->exit_block);
    return ctx->ir;
}

::llvm::Type *LLVMCodegen::_create_type(const Type *t) noexcept {
    if (t == nullptr) { return ::llvm::Type::getVoidTy(_context); }
    switch (t->tag()) {
        case Type::Tag::BOOL: return ::llvm::Type::getInt8Ty(_context);
        case Type::Tag::FLOAT: return ::llvm::Type::getFloatTy(_context);
        case Type::Tag::INT: [[fallthrough]];
        case Type::Tag::UINT: return ::llvm::Type::getInt32Ty(_context);
        case Type::Tag::VECTOR: return ::llvm::VectorType::get(
            _create_type(t->element()),
            t->dimension() == 3u ? 4u : t->dimension(), false);
        case Type::Tag::MATRIX: return ::llvm::ArrayType::get(
            _create_type(Type::from(luisa::format(
                "vector<{},{}>", t->element()->description(), t->dimension()))),
            t->dimension());
        case Type::Tag::ARRAY: return ::llvm::ArrayType::get(
            _create_type(t->element()), t->dimension());
        case Type::Tag::STRUCTURE: {
            if (auto iter = _struct_types.find(t->hash()); iter != _struct_types.end()) {
                return iter->second.type;
            }
            auto member_index = 0u;
            luisa::vector<::llvm::Type *> field_types;
            luisa::vector<uint> field_indices;
            auto size = 0ul;
            for (auto &member : t->members()) {
                auto aligned_offset = luisa::align(size, member->alignment());
                if (aligned_offset > size) {
                    auto padding = ::llvm::ArrayType::get(
                        ::llvm::Type::getInt8Ty(_context), aligned_offset - size);
                    field_types.emplace_back(padding);
                    member_index++;
                }
                field_types.emplace_back(_create_type(member));
                field_indices.emplace_back(member_index++);
                size = aligned_offset + member->size();
            }
            if (t->size() > size) {// last padding
                auto padding = ::llvm::ArrayType::get(
                    ::llvm::Type::getInt8Ty(_context), t->size() - size);
                field_types.emplace_back(padding);
            }
            ::llvm::ArrayRef<::llvm::Type *> fields_ref{field_types.data(), field_types.size()};
            auto struct_type = ::llvm::StructType::get(_context, fields_ref);
            _struct_types.emplace(t->hash(), LLVMStruct{struct_type, std::move(field_indices)});
            return struct_type;
        }
        case Type::Tag::BUFFER: return ::llvm::PointerType::get(_create_type(t->element()), 0);
        case Type::Tag::TEXTURE: /* TODO: implement */ break;
        case Type::Tag::BINDLESS_ARRAY: /* TODO: implement */ break;
        case Type::Tag::ACCEL: /* TODO: implement */ break;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid type: {}.", t->description());
}

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

::llvm::Value *LLVMCodegen::_create_unary_expr(const UnaryExpr *expr) noexcept {

    return nullptr;
}

::llvm::Value *LLVMCodegen::_create_binary_expr(const BinaryExpr *expr) noexcept {
    return nullptr;
}

::llvm::Value *LLVMCodegen::_create_member_expr(const MemberExpr *expr) noexcept {
    return nullptr;
}

::llvm::Value *LLVMCodegen::_create_access_expr(const AccessExpr *expr) noexcept {
    return nullptr;
}

::llvm::Value *LLVMCodegen::_create_literal_expr(const LiteralExpr *expr) noexcept {
    return nullptr;
}

::llvm::Value *LLVMCodegen::_create_ref_expr(const RefExpr *expr) noexcept {
    return _current_context()->variables.at(expr->variable().uid());
}

::llvm::Value *LLVMCodegen::_create_constant_expr(const ConstantExpr *expr) noexcept {
    return nullptr;
}

::llvm::Value *LLVMCodegen::_create_call_expr(const CallExpr *expr) noexcept {
    return nullptr;
}

::llvm::Value *LLVMCodegen::_create_cast_expr(const CastExpr *expr) noexcept {
    return nullptr;
}

luisa::string LLVMCodegen::_variable_name(Variable v) const noexcept {
    switch (v.tag()) {
        case Variable::Tag::LOCAL: return luisa::format("v{}", v.uid());
        case Variable::Tag::SHARED: return luisa::format("s{}", v.uid());
        case Variable::Tag::REFERENCE: return luisa::format("r{}", v.uid());
        case Variable::Tag::BUFFER: return luisa::format("b{}", v.uid());
        case Variable::Tag::TEXTURE: return luisa::format("t{}", v.uid());
        case Variable::Tag::BINDLESS_ARRAY: return luisa::format("h{}", v.uid());
        case Variable::Tag::ACCEL: return luisa::format("a{}", v.uid());
        case Variable::Tag::THREAD_ID: return "tid";
        case Variable::Tag::BLOCK_ID: return "bid";
        case Variable::Tag::DISPATCH_ID: return "did";
        case Variable::Tag::DISPATCH_SIZE: return "ls";
    }
    LUISA_ERROR_WITH_LOCATION("Invalid variable.");
}

luisa::string LLVMCodegen::_function_name(Function f) const noexcept {
    return luisa::format("func_{:016x}", f.hash());
}

LLVMCodegen::FunctionContext *LLVMCodegen::_current_context() noexcept {
    LUISA_ASSERT(!_function_stack.empty(), "Empty function context stack.");
    return _function_stack.back().get();
}

::llvm::Value *LLVMCodegen::_convert(const Type *dst_type, const Type *src_type, ::llvm::Value *p_src) noexcept {
    switch (dst_type->tag()) {
        case Type::Tag::BOOL: return _scalar_to_bool(src_type, p_src);
        case Type::Tag::FLOAT: return _scalar_to_float(src_type, p_src);
        case Type::Tag::INT: return _scalar_to_int(src_type, p_src);
        case Type::Tag::UINT: return _scalar_to_uint(src_type, p_src);
        case Type::Tag::VECTOR: {
            if (src_type->is_vector()) { break; }
            LUISA_ASSERT(*src_type == *dst_type->element(),
                         "Invalid conversion from '{}' to '{}'.",
                         src_type->description(), dst_type->description());
            return _scalar_to_vector(dst_type, dst_type->dimension(), p_src);
        }
        default: break;
    }
    LUISA_ASSERT(*dst_type == *src_type, "Cannot convert '{}' to '{}'.",
                 src_type->description(), dst_type->description());
    return p_src;
}

::llvm::Value *LLVMCodegen::_scalar_to_bool(const Type *src_type, ::llvm::Value *p_src) noexcept {
    LUISA_ASSERT(src_type->is_scalar(), "Invalid source type: {}.", src_type->description());
    auto builder = _current_context()->builder.get();
    switch (src_type->tag()) {
        case Type::Tag::BOOL: return p_src;
        case Type::Tag::FLOAT: return _create_stack_variable(builder->CreateFCmpONE(p_src, ::llvm::ConstantFP::get(p_src->getType(), 0.0)));
        case Type::Tag::INT:
        case Type::Tag::UINT: return _create_stack_variable(builder->CreateICmpNE(p_src, ::llvm::ConstantInt::get(p_src->getType(), 0)));
        default: break;
    }
    return nullptr;
}

::llvm::Value *LLVMCodegen::_scalar_to_float(const Type *src_type, ::llvm::Value *p_src) noexcept {
    LUISA_ASSERT(src_type->is_scalar(), "Invalid source type: {}.", src_type->description());
    auto builder = _current_context()->builder.get();
    switch (src_type->tag()) {
        case Type::Tag::BOOL: return _create_stack_variable(builder->CreateSelect(p_src, ::llvm::ConstantFP::get(p_src->getType(), 1.0), ::llvm::ConstantFP::get(p_src->getType(), 0.0)));
        case Type::Tag::FLOAT: return p_src;
        case Type::Tag::INT: return _create_stack_variable(builder->CreateSIToFP(p_src, ::llvm::Type::getFloatTy(_context)));
        case Type::Tag::UINT: return _create_stack_variable(builder->CreateUIToFP(p_src, ::llvm::Type::getFloatTy(_context)));
        default: break;
    }
    return nullptr;
}

::llvm::Value *LLVMCodegen::_scalar_to_int(const Type *src_type, ::llvm::Value *p_src) noexcept {
    LUISA_ASSERT(src_type->is_scalar(), "Invalid source type: {}.", src_type->description());
    auto builder = _current_context()->builder.get();
    switch (src_type->tag()) {
        case Type::Tag::BOOL: return _create_stack_variable(builder->CreateZExt(p_src, ::llvm::Type::getInt32Ty(_context)));
        case Type::Tag::FLOAT: return _create_stack_variable(builder->CreateFPToSI(p_src, ::llvm::Type::getInt32Ty(_context)));
        case Type::Tag::INT:
        case Type::Tag::UINT: return p_src;
        default: break;
    }
    return nullptr;
}

::llvm::Value *LLVMCodegen::_scalar_to_uint(const Type *src_type, ::llvm::Value *p_src) noexcept {
    LUISA_ASSERT(src_type->is_scalar(), "Invalid source type: {}.", src_type->description());
    auto builder = _current_context()->builder.get();
    switch (src_type->tag()) {
        case Type::Tag::BOOL: return _create_stack_variable(builder->CreateZExt(p_src, ::llvm::Type::getInt32Ty(_context)));
        case Type::Tag::FLOAT: return _create_stack_variable(builder->CreateFPToUI(p_src, ::llvm::Type::getInt32Ty(_context)));
        case Type::Tag::INT:
        case Type::Tag::UINT: return p_src;
        default: break;
    }
    return nullptr;
}

::llvm::Value *LLVMCodegen::_scalar_to_vector(const Type *src_type, uint dst_dim, ::llvm::Value *p_src) noexcept {
    LUISA_ASSERT(src_type->is_scalar(), "Invalid source type: {}.", src_type->description());
    if (dst_dim == 3u) { dst_dim = 4u; }
    auto builder = _current_context()->builder.get();
    switch (src_type->tag()) {
        case Type::Tag::BOOL:
        case Type::Tag::FLOAT:
        case Type::Tag::INT:
        case Type::Tag::UINT: return _create_stack_variable(builder->CreateVectorSplat(dst_dim, p_src));
        default: break;
    }
    return nullptr;
}

::llvm::Value *LLVMCodegen::_create_stack_variable(::llvm::Value *x, luisa::string_view name) noexcept {
    auto builder = _current_context()->builder.get();
    if (x->getType()->isIntegerTy(1)) {// special handling for int1
        return _create_stack_variable(
            builder->CreateZExt(x, builder->getInt8Ty()), name);
    }
    auto p = builder->CreateAlloca(x->getType(), nullptr, name);
    builder->CreateStore(x, p);
    return p;
}

void LLVMCodegen::_create_assignment(const Type *dst_type, const Type *src_type, ::llvm::Value *p_dst, ::llvm::Value *p_src) noexcept {
    auto p_rhs = _convert(dst_type, src_type, p_src);
    auto builder = _current_context()->builder.get();
    auto dst = builder->CreateLoad(_create_type(dst_type), p_rhs);
    builder->CreateStore(dst, p_dst);
}

}// namespace luisa::compute::llvm
