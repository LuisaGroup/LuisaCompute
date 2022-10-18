//
// Created by Mike Smith on 2022/10/17.
//

#include <ir/ast2ir.h>

namespace luisa::compute {

ir::Module AST2IR::convert(Function function) noexcept {
    // TODO
}

ir::NodeRef AST2IR::_convert_expr(const Expression *expr) noexcept {
    switch (expr->tag()) {
        case Expression::Tag::UNARY: return _convert(static_cast<const UnaryExpr *>(expr));
        case Expression::Tag::BINARY: return _convert(static_cast<const BinaryExpr *>(expr));
        case Expression::Tag::MEMBER: return _convert(static_cast<const MemberExpr *>(expr));
        case Expression::Tag::ACCESS: return _convert(static_cast<const AccessExpr *>(expr));
        case Expression::Tag::LITERAL: return _convert(static_cast<const LiteralExpr *>(expr));
        case Expression::Tag::REF: return _convert(static_cast<const RefExpr *>(expr));
        case Expression::Tag::CONSTANT: return _convert(static_cast<const ConstantExpr *>(expr));
        case Expression::Tag::CALL: return _convert(static_cast<const CallExpr *>(expr));
        case Expression::Tag::CAST: return _convert(static_cast<const CastExpr *>(expr));
        case Expression::Tag::CPUCUSTOM: return _convert(static_cast<const CpuCustomOpExpr *>(expr));
        case Expression::Tag::GPUCUSTOM: return _convert(static_cast<const GpuCustomOpExpr *>(expr));
        case Expression::Tag::PHI: return _convert(static_cast<const PhiExpr *>(expr));
        case Expression::Tag::REPLACE_MEMBER: return _convert(static_cast<const ReplaceMemberExpr *>(expr));
    }
    LUISA_ERROR_WITH_LOCATION("Invalid expression tag.");
}

ir::NodeRef AST2IR::_convert_stmt(const Statement *stmt) noexcept {
    switch (stmt->tag()) {
        case Statement::Tag::BREAK: return _convert(static_cast<const BreakStmt *>(stmt)); break;
        case Statement::Tag::CONTINUE: return _convert(static_cast<const ContinueStmt *>(stmt)); break;
        case Statement::Tag::RETURN: return _convert(static_cast<const ReturnStmt *>(stmt)); break;
        case Statement::Tag::SCOPE: return _convert(static_cast<const ScopeStmt *>(stmt)); break;
        case Statement::Tag::IF: return _convert(static_cast<const IfStmt *>(stmt)); break;
        case Statement::Tag::LOOP: return _convert(static_cast<const LoopStmt *>(stmt)); break;
        case Statement::Tag::EXPR: return _convert(static_cast<const ExprStmt *>(stmt)); break;
        case Statement::Tag::SWITCH: return _convert(static_cast<const SwitchStmt *>(stmt)); break;
        case Statement::Tag::SWITCH_CASE: return _convert(static_cast<const SwitchCaseStmt *>(stmt)); break;
        case Statement::Tag::SWITCH_DEFAULT: return _convert(static_cast<const SwitchDefaultStmt *>(stmt)); break;
        case Statement::Tag::ASSIGN: return _convert(static_cast<const AssignStmt *>(stmt)); break;
        case Statement::Tag::FOR: return _convert(static_cast<const ForStmt *>(stmt)); break;
        case Statement::Tag::COMMENT: return _convert(static_cast<const CommentStmt *>(stmt)); break;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid statement tag.");
}

ir::NodeRef AST2IR::_convert(const ConstantData &c) noexcept {
    if (auto iter = _constants.find(c.hash()); iter != _constants.end()) {
        return iter->second;
    }
    auto b = _current_builder();
}

ir::IrBuilder *AST2IR::_current_builder() noexcept {
    LUISA_ASSERT(!_builder_stack.empty(), "Builder stack is empty.");
    return _builder_stack.back().get();
}

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
            auto dim = static_cast<uint>(type->dimension());
            switch (auto elem = type->element(); elem->tag()) {
                case Type::Tag::BOOL:
                    t = {.tag = ir::Type::Tag::Vector,
                         .vector = {{.element = {.tag = ir::VectorElementType::Tag::Scalar,
                                                 .scalar = {ir::Primitive::Bool}},
                                     .length = dim}}};
                    break;
                case Type::Tag::FLOAT:
                    t = {.tag = ir::Type::Tag::Vector,
                         .vector = {{.element = {.tag = ir::VectorElementType::Tag::Scalar,
                                                 .scalar = {ir::Primitive::Float32}},
                                     .length = dim}}};
                    break;
                case Type::Tag::INT:
                    t = {.tag = ir::Type::Tag::Vector,
                         .vector = {{.element = {.tag = ir::VectorElementType::Tag::Scalar,
                                                 .scalar = {ir::Primitive::Int32}},
                                     .length = dim}}};
                    break;
                case Type::Tag::UINT:
                    t = {.tag = ir::Type::Tag::Vector,
                         .vector = {{.element = {.tag = ir::VectorElementType::Tag::Scalar,
                                                 .scalar = {ir::Primitive::Uint32}},
                                     .length = dim}}};
                    break;
                default: LUISA_ERROR_WITH_LOCATION(
                    "Invalid vector element type: {}.",
                    elem->description());
            }
            break;
        }
        case Type::Tag::MATRIX:
            t = {.tag = ir::Type::Tag::Matrix,
                 .matrix = {{.element = {.tag = ir::VectorElementType::Tag::Scalar,
                                         .scalar = {ir::Primitive::Float32}},
                             .dimension = static_cast<uint>(type->dimension())}}};
            break;
        case Type::Tag::ARRAY:
            t = {.tag = ir::Type::Tag::Array,
                 .array = {{.element = _convert(type->element()),
                            .length = type->dimension()}}};
            break;
        case Type::Tag::STRUCTURE:
        case Type::Tag::BUFFER:
        case Type::Tag::TEXTURE:
        case Type::Tag::BINDLESS_ARRAY:
        case Type::Tag::ACCEL:
            return nullptr;
    }
    return ir::luisa_compute_ir_register_type(&t);
}

ir::NodeRef AST2IR::_convert(const UnaryExpr *expr) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const BinaryExpr *expr) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const MemberExpr *expr) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const AccessExpr *expr) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const LiteralExpr *expr) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const RefExpr *expr) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const ConstantExpr *expr) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const CallExpr *expr) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const CastExpr *expr) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const PhiExpr *expr) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const CpuCustomOpExpr *expr) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const GpuCustomOpExpr *expr) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const ReplaceMemberExpr *expr) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const BreakStmt *stmt) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const ContinueStmt *stmt) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const ReturnStmt *stmt) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const ScopeStmt *stmt) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const IfStmt *stmt) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const LoopStmt *stmt) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const ExprStmt *stmt) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const SwitchStmt *stmt) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const SwitchCaseStmt *stmt) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const SwitchDefaultStmt *stmt) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const AssignStmt *stmt) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const ForStmt *stmt) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const CommentStmt *stmt) noexcept {
    return ir::NodeRef();
}

}// namespace luisa::compute
