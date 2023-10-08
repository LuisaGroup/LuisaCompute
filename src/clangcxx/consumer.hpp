#pragma once
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecordLayout.h"
#include <unordered_set>

namespace luisa::clangcxx
{

class ASTConsumer : public clang::ASTConsumer
{
public:
    explicit ASTConsumer(std::string&& OutputPath)
        : OutputPath(OutputPath)
    {
        
    }
    void HandleTranslationUnit(clang::ASTContext& ctx) override;
    
protected:
    void HandleDecl(clang::NamedDecl* decl, const clang::ASTRecordLayout* layout);

    std::string OutputPath;
};

}