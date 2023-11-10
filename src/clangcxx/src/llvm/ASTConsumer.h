#pragma once
#include <utility>
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecordLayout.h"

namespace luisa::clangcxx {

class ASTConsumer : public clang::ASTConsumer
{
public:
    explicit ASTConsumer(std::string OutputPath);
    void HandleTranslationUnit(clang::ASTContext& ctx) override;
    inline clang::ASTContext* GetContext() { return _astContext; }
    
protected:
    void HandleDecl(clang::NamedDecl* decl, const clang::ASTRecordLayout* layout);
    void HandleNamespace(clang::NamedDecl* decl, const clang::ASTRecordLayout* layout);
    void HandleRecord(clang::NamedDecl* decl, const clang::ASTRecordLayout* layout);
    void HandleFunction(clang::NamedDecl* decl, const clang::ASTRecordLayout* layout);

    std::string OutputPath;
    clang::ASTContext* _astContext = nullptr;
};

}// namespace lc::clangcxx