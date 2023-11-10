#include "ASTConsumer.h"
#include <iostream>

namespace luisa::clangcxx {

ASTConsumer::ASTConsumer(std::string OutputPath)
    : OutputPath(std::move(OutputPath)) {
}

void ASTConsumer::HandleTranslationUnit(clang::ASTContext &ctx) {
    // 1. collect
    _astContext = &ctx;
    auto tuDecl = ctx.getTranslationUnitDecl();
    for (clang::DeclContext::decl_iterator i = tuDecl->decls_begin();
         i != tuDecl->decls_end(); ++i) {
        auto *named_decl = llvm::dyn_cast<clang::NamedDecl>(*i);
        if (named_decl == nullptr)
            continue;

        // Filter out unsupported decls at the global namespace level
        clang::Decl::Kind kind = named_decl->getKind();
        std::vector<std::string> newStack;
        switch (kind) {
            case (clang::Decl::Namespace):
            case (clang::Decl::CXXRecord):
            case (clang::Decl::Function):
            case (clang::Decl::Enum):
            case (clang::Decl::ClassTemplate):
                HandleDecl(named_decl, nullptr);
                break;
            default:
                break;
        }
    }
}

void Remove(std::string &str, const std::string &remove_str) {
    for (size_t i; (i = str.find(remove_str)) != std::string::npos;)
        str.replace(i, remove_str.length(), "");
}

std::string GetTypeName(clang::QualType type, clang::ASTContext *ctx) {
    type = type.getCanonicalType();
    auto baseName = type.getAsString(ctx->getLangOpts());
    Remove(baseName, "struct ");
    Remove(baseName, "class ");
    return baseName;
}

void ASTConsumer::HandleDecl(clang::NamedDecl *decl, const clang::ASTRecordLayout *layout) {
    if (decl == nullptr)
        return;
    if (decl->isInvalidDecl())
        return;

    clang::ASTContext &ctx = *_astContext;
    // Filter out unsupported decls at the global namespace level
    clang::Decl::Kind kind = decl->getKind();
    switch (kind) {
        case (clang::Decl::Namespace): {
            HandleNamespace(decl, nullptr);
            clang::DeclContext *declContext = clang::NamedDecl::castToDeclContext(decl);
            std::vector<std::string> newStack;
            for (clang::DeclContext::decl_iterator i = declContext->decls_begin();
                 i != declContext->decls_end(); ++i) {
                if (auto *sub_decl = llvm::dyn_cast<clang::NamedDecl>(*i))
                    HandleDecl(sub_decl, nullptr);
            }
            return;
        } break;
        case (clang::Decl::CXXRecord):
            HandleRecord(decl, nullptr);
            break;
        case (clang::Decl::Function):
            HandleFunction(decl, nullptr);
            break;
        case (clang::Decl::ParmVar):
        case (clang::Decl::Enum):
        case (clang::Decl::ClassTemplate):
        case (clang::Decl::Field):
        case (clang::Decl::Var):
        case (clang::Decl::VarTemplateSpecialization):
        default:
            break;
    }
}

void ASTConsumer::HandleNamespace(clang::NamedDecl *decl, const clang::ASTRecordLayout *layout) {
    auto nspace = static_cast<clang::NamespaceDecl *>(decl);
    std::cout << "namespace: " << nspace->getDeclName().getAsString() << std::endl;
}

void ASTConsumer::HandleRecord(clang::NamedDecl *decl, const clang::ASTRecordLayout *layout) {
    auto record = static_cast<clang::CXXRecordDecl *>(decl);
    std::cout << "record: " << record->getQualifiedNameAsString() << std::endl;
}

void ASTConsumer::HandleFunction(clang::NamedDecl *decl, const clang::ASTRecordLayout *layout) {
    std::cout << "function: " << decl->getDeclName().getAsString() << std::endl;
}

}// namespace luisa::clangcxx