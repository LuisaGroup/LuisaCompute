//
// Created by Mike Smith on 2022/10/17.
//

#include <ir/ast2ir.h>

namespace luisa::compute {

ir::Module AST2IR::convert(Function function) noexcept {
}

//ir::NodeRef AST2IR::_convert_expr(const Expression *expr) noexcept {
//    switch (expr->tag()) {
//        case Expression::Tag::UNARY: return _convert(static_cast<const UnaryExpr *>(expr));
//        case Expression::Tag::BINARY: return _convert(static_cast<const BinaryExpr *>(expr));
//        case Expression::Tag::MEMBER: return _convert(static_cast<const MemberExpr *>(expr));
//        case Expression::Tag::ACCESS: return _convert(static_cast<const AccessExpr *>(expr));
//        case Expression::Tag::LITERAL: return _convert(static_cast<const LiteralExpr *>(expr));
//        case Expression::Tag::REF: return _convert(static_cast<const RefExpr *>(expr));
//        case Expression::Tag::CONSTANT: return _convert(static_cast<const ConstantExpr *>(expr));
//        case Expression::Tag::CALL: return _convert(static_cast<const CallExpr *>(expr));
//        case Expression::Tag::CAST: return _convert(static_cast<const CastExpr *>(expr));
//        case Expression::Tag::CPUCUSTOM: return _convert(static_cast<const CpuCustomOpExpr *>(expr));
//        case Expression::Tag::GPUCUSTOM: return _convert(static_cast<const GpuCustomOpExpr *>(expr));
//        case Expression::Tag::PHI: return _convert(static_cast<const PhiExpr *>(expr));
//        case Expression::Tag::REPLACE_MEMBER: return _convert(static_cast<const ReplaceMemberExpr *>(expr));
//    }
//    LUISA_ERROR_WITH_LOCATION("Invalid expression tag.");
//}
//
//void AST2IR::_convert_stmt(const Statement *stmt) noexcept {
//    switch (stmt->tag()) {
//        case Statement::Tag::BREAK: _convert(static_cast<const BreakStmt *>(stmt)); break;
//        case Statement::Tag::CONTINUE: _convert(static_cast<const ContinueStmt *>(stmt)); break;
//        case Statement::Tag::RETURN: _convert(static_cast<const ReturnStmt *>(stmt)); break;
//        case Statement::Tag::SCOPE: _convert(static_cast<const ScopeStmt *>(stmt)); break;
//        case Statement::Tag::IF: _convert(static_cast<const IfStmt *>(stmt)); break;
//        case Statement::Tag::LOOP: _convert(static_cast<const LoopStmt *>(stmt)); break;
//        case Statement::Tag::EXPR: _convert(static_cast<const ExprStmt *>(stmt)); break;
//        case Statement::Tag::SWITCH: _convert(static_cast<const SwitchStmt *>(stmt)); break;
//        case Statement::Tag::SWITCH_CASE: _convert(static_cast<const SwitchCaseStmt *>(stmt)); break;
//        case Statement::Tag::SWITCH_DEFAULT: _convert(static_cast<const SwitchDefaultStmt *>(stmt)); break;
//        case Statement::Tag::ASSIGN: _convert(static_cast<const AssignStmt *>(stmt)); break;
//        case Statement::Tag::FOR: _convert(static_cast<const ForStmt *>(stmt)); break;
//        case Statement::Tag::COMMENT: _convert(static_cast<const CommentStmt *>(stmt)); break;
//    }
//}
//
//ir::NodeRef AST2IR::_convert(const ConstantData &c) noexcept {
//    if (auto iter = _constants.find(c.hash()); iter != _constants.end()) {
//        return iter->second;
//    }
//    auto b = _current_builder();
//}
//
//ir::IrBuilder *AST2IR::_current_builder() noexcept {
//    LUISA_ASSERT(!_builder_stack.empty(), "Builder stack is empty.");
//    return _builder_stack.back().get();
//}

const ir::Type *AST2IR::_convert(const Type *type) noexcept {
    ir::Type t{};
    switch (type->tag()) {
        case Type::Tag::BOOL:
            t = {.tag = ir::Type::Tag::Primitive,
                 .primitive = {ir::Primitive::Bool}};
            break;
        case Type::Tag::FLOAT:
            t = {.tag = ir::Type::Tag::Primitive,
                 .primitive = {ir::Primitive::Float32}};
            break;
        case Type::Tag::INT:
            t = {.tag = ir::Type::Tag::Primitive,
                 .primitive = {ir::Primitive::Int32}};
            break;
        case Type::Tag::UINT:
            t = {.tag = ir::Type::Tag::Primitive,
                 .primitive = {ir::Primitive::Uint32}};
            break;
        case Type::Tag::VECTOR: {

        }
        case Type::Tag::MATRIX: break;
        case Type::Tag::ARRAY: break;
        case Type::Tag::STRUCTURE: break;
        case Type::Tag::BUFFER: break;
        case Type::Tag::TEXTURE: break;
        case Type::Tag::BINDLESS_ARRAY: break;
        case Type::Tag::ACCEL: break;
    }
    return t;
}

}// namespace luisa::compute
