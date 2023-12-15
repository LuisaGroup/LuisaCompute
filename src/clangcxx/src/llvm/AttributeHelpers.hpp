#pragma once
#include "clang/AST/Attr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Expr.h"
#include <luisa/core/logging.h>

namespace luisa::clangcxx {
inline static clang::RecordDecl *GetRecordDeclFromQualType(clang::QualType Ty, luisa::string parent = {}) {
    clang::RecordDecl *recordDecl = Ty->getAsRecordDecl();
    if (!recordDecl) {
        if (const auto TPT = Ty->getAs<clang::PointerType>()) {
            Ty = TPT->getPointeeType();
            recordDecl = Ty->getAsRecordDecl();
        } else if (const auto TRT = Ty->getAs<clang::ReferenceType>()) {
            Ty = TRT->getPointeeType();
            recordDecl = Ty->getAsRecordDecl();
        } else if (const auto *TDT = Ty->getAs<clang::TypedefType>()) {
            Ty = TDT->getDecl()->getUnderlyingType();
            recordDecl = Ty->getAsRecordDecl();
        } else if (const auto *TST = Ty->getAs<clang::TemplateSpecializationType>()) {
            recordDecl = TST->getAsRecordDecl();
        } else if (const auto *AT = Ty->getAsArrayTypeUnsafe()) {
            recordDecl = AT->getAsRecordDecl();
            if (!parent.empty()) {
                luisa::log_error("array type is not supported: [{}] in type [{}]", Ty.getAsString(), parent);
            } else {
                luisa::log_error("array type is not supported: [{}]", Ty.getAsString());
            }
        } else {
            Ty.dump();
        }
    }
    return recordDecl;
}

inline static bool isLuisaAttribute(const clang::AnnotateAttr *Anno) {
    return Anno->getAnnotation() == "luisa-shader";
}

inline static bool isIgnore(const clang::AnnotateAttr *Anno) {
    if (!isLuisaAttribute(Anno))
        return false;
    if (Anno->args_size() == 1) {
        auto arg = Anno->args_begin();
        if (auto TypeLiterial = llvm::dyn_cast<clang::StringLiteral>((*arg)->IgnoreParenCasts())) {
            return (TypeLiterial->getString() == "ignore");
        }
    }
    return false;
}

inline static bool isByPass(const clang::AnnotateAttr *Anno) {
    if (!isLuisaAttribute(Anno))
        return false;
    if (Anno->args_size() >= 1) {
        auto arg = Anno->args_begin();
        if (auto TypeLiterial = llvm::dyn_cast<clang::StringLiteral>((*arg)->IgnoreParenCasts())) {
            return (TypeLiterial->getString() == "bypass");
        }
    }
    return false;
}

inline static bool isByPass(const clang::Decl *decl) {
    if (auto cxxField = llvm::dyn_cast<clang::FieldDecl>(decl)) {
        if (cxxField->isAnonymousStructOrUnion())
            return true;
    }
    for (auto Anno = decl->specific_attr_begin<clang::AnnotateAttr>();
        Anno != decl->specific_attr_end<clang::AnnotateAttr>(); ++Anno) {
        if (isByPass(*Anno))
            return true;
    }
    return false;
}

inline static bool isKernel(const clang::AnnotateAttr *Anno) {
    if (!isLuisaAttribute(Anno))
        return false;
    auto arg = Anno->args_begin();
    if (auto TypeLiterial = llvm::dyn_cast<clang::StringLiteral>((*arg)->IgnoreParenCasts())) {
        return (TypeLiterial->getString().starts_with("kernel_"));
    }
    return false;
}

inline static bool getKernelSize(const clang::AnnotateAttr *Anno, uint32_t &x, uint32_t &y, uint32_t &z) {
    if (!isKernel(Anno))
        return false;
    x = y = z = 1;
    auto arg = Anno->args_begin();
    const auto N = Anno->args_size();
    if (N > 1) {
        arg++;
        if (auto IntLiteral = llvm::dyn_cast<clang::IntegerLiteral>((*arg)->IgnoreParenCasts())) {
            x = IntLiteral->getValue().getLimitedValue();
        }
    }
    if (N > 2) {
        arg++;
        if (auto IntLiteral = llvm::dyn_cast<clang::IntegerLiteral>((*arg)->IgnoreParenCasts())) {
            y = IntLiteral->getValue().getLimitedValue();
        }
    }
    if (N > 3) {
        arg++;
        if (auto IntLiteral = llvm::dyn_cast<clang::IntegerLiteral>((*arg)->IgnoreParenCasts())) {
            z = IntLiteral->getValue().getLimitedValue();
        }
    }
    return true;
}

inline static bool isBuiltinType(const clang::AnnotateAttr *Anno) {
    if (!isLuisaAttribute(Anno))
        return false;
    if (Anno->args_size() >= 1) {
        auto arg = Anno->args_begin();
        if (auto TypeLiterial = llvm::dyn_cast<clang::StringLiteral>((*arg)->IgnoreParenCasts())) {
            return (TypeLiterial->getString() == "builtin");
        }
    }
    return false;
}

inline static llvm::StringRef getBuiltinTypeName(const clang::AnnotateAttr *Anno) {
    if (!isBuiltinType(Anno))
        return {};
    if (Anno->args_size() >= 1) {
        auto arg = Anno->args_begin();
        arg++;
        if (auto TypeLiterial = llvm::dyn_cast<clang::StringLiteral>((*arg)->IgnoreParenCasts())) {
            return TypeLiterial->getString();
        }
    }
    return {};
}

inline static bool isCallop(const clang::AnnotateAttr *Anno) {
    if (!isLuisaAttribute(Anno))
        return false;
    if (Anno->args_size() >= 1) {
        auto arg = Anno->args_begin();
        if (auto TypeLiterial = llvm::dyn_cast<clang::StringLiteral>((*arg)->IgnoreParenCasts())) {
            return (TypeLiterial->getString() == "callop");
        }
    }
    return false;
}

inline static llvm::StringRef getCallopName(const clang::AnnotateAttr *Anno) {
    if (!isCallop(Anno))
        return {};
    if (Anno->args_size() >= 1) {
        auto arg = Anno->args_begin();
        arg++;
        if (auto TypeLiterial = llvm::dyn_cast<clang::StringLiteral>((*arg)->IgnoreParenCasts())) {
            return TypeLiterial->getString();
        }
    }
    return {};
}

inline static bool isSwizzle(const clang::AnnotateAttr *Anno) {
    if (!isLuisaAttribute(Anno))
        return false;
    if (Anno->args_size() >= 1) {
        auto arg = Anno->args_begin();
        if (auto TypeLiterial = llvm::dyn_cast<clang::StringLiteral>((*arg)->IgnoreParenCasts())) {
            return (TypeLiterial->getString() == "swizzle");
        }
    }
    return false;
}

inline static bool isSwizzle(const clang::FieldDecl *decl) {
    for (auto Anno = decl->specific_attr_begin<clang::AnnotateAttr>();
        Anno != decl->specific_attr_end<clang::AnnotateAttr>(); ++Anno) {
        if (isSwizzle(*Anno))
            return true;
    }
    if (auto typeDecl = GetRecordDeclFromQualType(decl->getType()))
    {
        for (auto Anno = typeDecl->specific_attr_begin<clang::AnnotateAttr>();
            Anno != typeDecl->specific_attr_end<clang::AnnotateAttr>(); ++Anno) {
            if (isSwizzle(*Anno))
                return true;
        }
    }
    return false;
}

}// namespace luisa::clangcxx