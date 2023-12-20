#pragma once
#include "Utils/AttributeHelper.hpp"

namespace luisa::clangcxx {

struct TypeDatabase {
    using BuiltinCallCmd = luisa::variant<
        luisa::compute::CallOp,
        const compute::RefExpr *(*)(compute::detail::FunctionBuilder *)>;

    TypeDatabase();
    ~TypeDatabase();

    const luisa::compute::Type *RecordAsPrimitiveType(const clang::QualType Type);
    const luisa::compute::Type *RecordAsBuiltinType(const clang::QualType Ty);
    const luisa::compute::Type *RecordAsStuctureType(const clang::QualType Ty);
    const luisa::compute::Type *RecordType(const clang::QualType Ty);

    const luisa::compute::Type *FindOrAddType(const clang::QualType Ty);
    BuiltinCallCmd FindCallOp(const luisa::string_view &name);
    luisa::compute::BinaryOp FindBinOp(const luisa::string_view &name);

    [[nodiscard]] void SetASTContext(clang::ASTContext *ctx) { astContext = ctx; }
    [[nodiscard]] clang::ASTContext *GetASTContext() { return astContext; }
    [[nodiscard]] const clang::ASTContext *GetASTContext() const { return astContext; }

    luisa::shared_ptr<compute::detail::FunctionBuilder> kernel_builder;
    luisa::unordered_map<luisa::string, const luisa::compute::RefExpr *> globals;
    luisa::unordered_map<const clang::Decl *, luisa::shared_ptr<compute::detail::FunctionBuilder>> func_builders;
    luisa::unordered_map<const clang::Decl *, luisa::shared_ptr<compute::detail::FunctionBuilder>> lambda_builders;

protected:
    void commentSourceLoc(luisa::shared_ptr<compute::detail::FunctionBuilder> fb, const luisa::string &prefix, const clang::SourceLocation &loc);
    const luisa::compute::Type *findType(const clang::QualType Ty);
    bool tryEmplaceFieldType(const clang::QualType Ty, const clang::RecordDecl *decl, luisa::vector<const luisa::compute::Type *> &types);
    bool registerType(clang::QualType Ty, const luisa::compute::Type *type);

    clang::ASTContext *astContext = nullptr;
    vstd::HashMap<vstd::string, luisa::compute::BinaryOp> bin_ops_map;
    vstd::HashMap<vstd::string, BuiltinCallCmd> call_ops_map;
    luisa::unordered_map<luisa::string, const luisa::compute::Type *> type_map;

public:
    struct Commenter {
        Commenter(
            luisa::function<void()> &&Begin, luisa::function<void()> &&End = []() {})
            : Begin(Begin), End(End) {
            Begin();
        }
        ~Commenter() {
            End();
        }
        luisa::function<void()> Begin = []() {};
        luisa::function<void()> End = []() {};
    };
    [[nodiscard]] Commenter CommentStmt(luisa::shared_ptr<compute::detail::FunctionBuilder> fb, const clang::Stmt *stmt);
};

};