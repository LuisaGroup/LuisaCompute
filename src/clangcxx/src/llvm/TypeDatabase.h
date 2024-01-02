#pragma once
#include "Utils/AttributeHelper.hpp"

namespace luisa::clangcxx {

struct TypeDatabase {
    TypeDatabase();
    ~TypeDatabase();
    const luisa::compute::Type *RecordType(const clang::QualType Ty, bool isRestrict = false);

    const luisa::compute::Type *FindOrAddType(const clang::QualType Ty, const clang::SourceLocation &loc);
    luisa::compute::CallOp FindCallOp(const luisa::string_view &name);
    luisa::compute::BinaryOp FindBinOp(const luisa::string_view &name);

    [[nodiscard]] void SetASTContext(clang::ASTContext *ctx) { astContext = ctx; }
    [[nodiscard]] clang::ASTContext *GetASTContext() { return astContext; }
    [[nodiscard]] const clang::ASTContext *GetASTContext() const { return astContext; }

    void SetFunctionThis(const compute::detail::FunctionBuilder* _this, const compute::RefExpr* fb);
    const luisa::compute::RefExpr* GetFunctionThis(const compute::detail::FunctionBuilder* fb) const;

    void DumpWithLocation(const clang::Stmt* stmt);
    void DumpWithLocation(const clang::Decl* decl);

    // luisa::unordered_map<luisa::string, const luisa::compute::RefExpr *> globals;
    luisa::shared_ptr<compute::detail::FunctionBuilder> kernel_builder;
    luisa::unordered_map<const clang::Decl *, luisa::shared_ptr<compute::detail::FunctionBuilder>> func_builders;
    luisa::unordered_map<const clang::Decl *, luisa::shared_ptr<compute::detail::FunctionBuilder>> lambda_builders;

protected:
    const luisa::compute::Type *RecordAsPrimitiveType(const clang::QualType Type);
    const luisa::compute::Type *RecordAsBuiltinType(const clang::QualType Ty);
    const luisa::compute::Type *RecordAsStuctureType(const clang::QualType Ty);

    void commentSourceLoc(compute::detail::FunctionBuilder* fb, const luisa::string &prefix, const clang::SourceLocation &loc);
    const luisa::compute::Type *findType(const clang::QualType Ty);
    bool tryEmplaceFieldType(const clang::QualType Ty, const clang::RecordDecl *decl, luisa::vector<const luisa::compute::Type *> &types);
    bool registerType(clang::QualType Ty, const luisa::compute::Type *type);

    clang::ASTContext *astContext = nullptr;
    vstd::HashMap<vstd::string, luisa::compute::BinaryOp> bin_ops_map;
    vstd::HashMap<vstd::string, luisa::compute::CallOp> call_ops_map;
    luisa::unordered_map<luisa::string, const luisa::compute::Type *> type_map;
    luisa::unordered_map<const compute::detail::FunctionBuilder *, const luisa::compute::RefExpr *> this_map;

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
    [[nodiscard]] Commenter CommentStmt(compute::detail::FunctionBuilder* fb, const clang::Stmt *stmt);
};

};