#pragma once
// Compat shim for clang::TagDecl::getTypeForDecl, which was deleted on
// derived classes (TagDecl/CXXRecordDecl) in newer Clang versions (>= 21).
// On older Clang it forwards to TagDecl::getTypeForDecl; on newer Clang it
// fetches the type via ASTContext::getTypeDeclType(d).getTypePtr().

#include <clang/AST/Decl.h>
#include <clang/AST/DeclCXX.h>
#include <clang/AST/ASTContext.h>
#include <clang/Basic/Version.h>
#include <type_traits>

namespace luisa::clangcxx::compat {

template<typename DeclT>
inline const clang::Type *get_type_for_decl(const DeclT *d) noexcept {
#if CLANG_VERSION_MAJOR >= 21
    if constexpr (std::is_base_of_v<clang::TagDecl, DeclT>) {
        return d->getASTContext().getCanonicalTagType(d).getTypePtr();
    } else {
        return d->getASTContext().getTypeDeclType(d).getTypePtr();
    }
#else
    return d->getTypeForDecl();
#endif
}

}// namespace luisa::clangcxx::compat
