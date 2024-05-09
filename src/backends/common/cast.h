#pragma once
#include <luisa/ast/statement.h>
namespace luisa::compute {
template<typename T>
struct AstNodeTypeDecl;
template<>
struct AstNodeTypeDecl<BreakStmt> {
    static constexpr auto value = Statement::Tag::BREAK;
};
template<>
struct AstNodeTypeDecl<ContinueStmt> {
    static constexpr auto value = Statement::Tag::CONTINUE;
};
template<>
struct AstNodeTypeDecl<ReturnStmt> {
    static constexpr auto value = Statement::Tag::RETURN;
};
template<>
struct AstNodeTypeDecl<ScopeStmt> {
    static constexpr auto value = Statement::Tag::SCOPE;
};
template<>
struct AstNodeTypeDecl<IfStmt> {
    static constexpr auto value = Statement::Tag::IF;
};
template<>
struct AstNodeTypeDecl<LoopStmt> {
    static constexpr auto value = Statement::Tag::LOOP;
};
template<>
struct AstNodeTypeDecl<ExprStmt> {
    static constexpr auto value = Statement::Tag::EXPR;
};
template<>
struct AstNodeTypeDecl<SwitchStmt> {
    static constexpr auto value = Statement::Tag::SWITCH;
};
template<>
struct AstNodeTypeDecl<SwitchCaseStmt> {
    static constexpr auto value = Statement::Tag::SWITCH_CASE;
};
template<>
struct AstNodeTypeDecl<SwitchDefaultStmt> {
    static constexpr auto value = Statement::Tag::SWITCH_DEFAULT;
};
template<>
struct AstNodeTypeDecl<AssignStmt> {
    static constexpr auto value = Statement::Tag::ASSIGN;
};
template<>
struct AstNodeTypeDecl<ForStmt> {
    static constexpr auto value = Statement::Tag::FOR;
};
template<>
struct AstNodeTypeDecl<CommentStmt> {
    static constexpr auto value = Statement::Tag::COMMENT;
};
template<>
struct AstNodeTypeDecl<RayQueryStmt> {
    static constexpr auto value = Statement::Tag::RAY_QUERY;
};
template<>
struct AstNodeTypeDecl<AutoDiffStmt> {
    static constexpr auto value = Statement::Tag::AUTO_DIFF;
};
template<>
struct AstNodeTypeDecl<PrintStmt> {
    static constexpr auto value = Statement::Tag::PRINT;
};
template<>
struct AstNodeTypeDecl<UnaryExpr> {
    static constexpr auto value = Expression::Tag::UNARY;
};
template<>
struct AstNodeTypeDecl<BinaryExpr> {
    static constexpr auto value = Expression::Tag::BINARY;
};
template<>
struct AstNodeTypeDecl<MemberExpr> {
    static constexpr auto value = Expression::Tag::MEMBER;
};
template<>
struct AstNodeTypeDecl<AccessExpr> {
    static constexpr auto value = Expression::Tag::ACCESS;
};
template<>
struct AstNodeTypeDecl<LiteralExpr> {
    static constexpr auto value = Expression::Tag::LITERAL;
};
template<>
struct AstNodeTypeDecl<RefExpr> {
    static constexpr auto value = Expression::Tag::REF;
};
template<>
struct AstNodeTypeDecl<ConstantExpr> {
    static constexpr auto value = Expression::Tag::CONSTANT;
};
template<>
struct AstNodeTypeDecl<CallExpr> {
    static constexpr auto value = Expression::Tag::CALL;
};
template<>
struct AstNodeTypeDecl<CastExpr> {
    static constexpr auto value = Expression::Tag::CAST;
};
template<>
struct AstNodeTypeDecl<TypeIDExpr> {
    static constexpr auto value = Expression::Tag::TYPE_ID;
};
template<>
struct AstNodeTypeDecl<StringIDExpr> {
    static constexpr auto value = Expression::Tag::STRING_ID;
};
template<>
struct AstNodeTypeDecl<CpuCustomOpExpr> {
    static constexpr auto value = Expression::Tag::CPUCUSTOM;
};
template<>
struct AstNodeTypeDecl<GpuCustomOpExpr> {
    static constexpr auto value = Expression::Tag::GPUCUSTOM;
};
template<typename T, typename Arg>
    requires(std::is_base_of_v<std::decay_t<Arg>, std::decay_t<T>>)
auto ast_cast_to(Arg *b) noexcept {
    using ReturnPtrType = decltype([]() {
        if constexpr (std::is_const_v<Arg>) {
            return (T const *)nullptr;
        } else {
            return (T *)nullptr;
        }
    }());
    if (b->tag() != AstNodeTypeDecl<T>::value) {
        return (ReturnPtrType)nullptr;
    }
    return static_cast<ReturnPtrType>(b);
}
}// namespace luisa::compute