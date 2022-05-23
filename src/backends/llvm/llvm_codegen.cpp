//
// Created by Mike Smith on 2022/5/23.
//

#include <backends/llvm/llvm_codegen.h>

namespace luisa::compute::llvm {

LLVMCodegen::LLVMCodegen(::llvm::LLVMContext &ctx) noexcept
    : _context{ctx}, _builder{luisa::make_unique<::llvm::IRBuilder<>>(ctx)} {}

void LLVMCodegen::visit(const UnaryExpr *expr) {
}

void LLVMCodegen::visit(const BinaryExpr *expr) {
}
void LLVMCodegen::visit(const MemberExpr *expr) {
}
void LLVMCodegen::visit(const AccessExpr *expr) {
}
void LLVMCodegen::visit(const LiteralExpr *expr) {
}
void LLVMCodegen::visit(const RefExpr *expr) {
}
void LLVMCodegen::visit(const ConstantExpr *expr) {
}
void LLVMCodegen::visit(const CallExpr *expr) {
}
void LLVMCodegen::visit(const CastExpr *expr) {
}
void LLVMCodegen::visit(const BreakStmt *stmt) {
}
void LLVMCodegen::visit(const ContinueStmt *stmt) {
}
void LLVMCodegen::visit(const ReturnStmt *stmt) {
}
void LLVMCodegen::visit(const ScopeStmt *stmt) {
}
void LLVMCodegen::visit(const IfStmt *stmt) {
}
void LLVMCodegen::visit(const LoopStmt *stmt) {
}
void LLVMCodegen::visit(const ExprStmt *stmt) {
}
void LLVMCodegen::visit(const SwitchStmt *stmt) {
}
void LLVMCodegen::visit(const SwitchCaseStmt *stmt) {
}
void LLVMCodegen::visit(const SwitchDefaultStmt *stmt) {
}
void LLVMCodegen::visit(const AssignStmt *stmt) {
}
void LLVMCodegen::visit(const ForStmt *stmt) {
}
void LLVMCodegen::visit(const CommentStmt *stmt) {
}
void LLVMCodegen::visit(const MetaStmt *stmt) {
}

::llvm::Function *LLVMCodegen::_emit(Function f) noexcept {
    // check if function already defined
    auto function_name = luisa::format("func_{:016x}", f.hash());
    if (auto function = _module->getFunction(
            ::llvm::StringRef{function_name.data(), function_name.size()})) {
        return function;
    }
    // codegen the function
    auto last_function = _function;
    auto last_insertion_block = _builder->GetInsertBlock();
    auto last_insertion_point = _builder->GetInsertPoint();
    _function = f;

    _function = last_function;
    _builder->SetInsertPoint(last_insertion_block, last_insertion_point);
    return nullptr;
}

luisa::unique_ptr<::llvm::Module> LLVMCodegen::emit(Function f) noexcept {
    auto module_name = luisa::format("module_{:016x}", f.hash());
    auto module = luisa::make_unique<::llvm::Module>(
        ::llvm::StringRef{module_name.data(), module_name.size()}, _context);
    _module = module.get();
    _emit(f);
    _module = nullptr;
    _builder->ClearInsertionPoint();
    return module;
}

::llvm::FunctionType *LLVMCodegen::_create_function_type(Function f) noexcept {
    luisa::vector<::llvm::Type *> arg_types;
    ::llvm::Type *return_type = nullptr;
    if (f.tag() == Function::Tag::KERNEL) {
        luisa::string arg_struct{"struct<16"};
        for (auto &arg : f.arguments()) { arg_struct.append(",").append(arg.type()->description()); }
        arg_struct.append(",").append(Type::of<uint3>()->description()).append(">");
        auto arg_struct_type = _create_type(Type::from(arg_struct));
        auto arg_buffer_type = ::llvm::PointerType::get(arg_struct_type, 0);
        arg_types.emplace_back(arg_buffer_type);
        arg_types.emplace_back(::llvm::Type::getInt32Ty(_context));
        arg_types.emplace_back(::llvm::Type::getInt32Ty(_context));
        arg_types.emplace_back(::llvm::Type::getInt32Ty(_context));
    } else {
        return_type = _create_type(f.return_type());
        for (auto &&arg : f.arguments()) {
            arg_types.emplace_back(_create_type(arg.type()));
        }
    }
    ::llvm::ArrayRef<::llvm::Type *> args{arg_types.data(), arg_types.size()};
    auto ft = ::llvm::FunctionType::get(return_type, args, false);
    return ft;
}

::llvm::Type *LLVMCodegen::_create_type(const Type *t) noexcept {
    if (t == nullptr) { return nullptr; }
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
                "vector<{},{}>", t->element(), t->dimension()))),
            t->dimension());
        case Type::Tag::ARRAY: return ::llvm::ArrayType::get(
            _create_type(t->element()), t->dimension());
        case Type::Tag::STRUCTURE: {
            if (auto iter = _struct_types.find(t->hash()); iter != _struct_types.end()) {
                return iter->second.type;
            }

            // TODO...
            return nullptr;
        }
        case Type::Tag::BUFFER: return ::llvm::PointerType::get(
            _create_type(t->element()), 0);
        case Type::Tag::TEXTURE: /* TODO: implement */ break;
        case Type::Tag::BINDLESS_ARRAY: /* TODO: implement */ break;
        case Type::Tag::ACCEL: /* TODO: implement */ break;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid type: {}.", t->description());
}

}// namespace luisa::compute::llvm
