#pragma once
#include "clang/AST/Attr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Expr.h"
#include <luisa/core/dll_export.h>
#include <luisa/vstl/common.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/device.h>

namespace luisa::clangcxx {

using CXXBinOp = clang::BinaryOperator::Opcode;
using LCBinOp = luisa::compute::BinaryOp;
using CXXUnaryOp = clang::UnaryOperator::Opcode;
using LCUnaryOp = luisa::compute::UnaryOp;

inline static clang::RecordDecl *GetRecordDeclFromQualType(clang::QualType Ty, bool isRestrict = true) {
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
            return nullptr;
        } else if (isRestrict) {
            Ty.dump();
            luisa::log_error("Restrict GetRecordDeclFromQualType failed!!!");
        }
    }
    return recordDecl;
}

inline static clang::ClassTemplateSpecializationDecl *GetClassTemplateSpecializationDecl(clang::QualType Ty, bool isRestrict = true) {
    if (auto Record = Ty->getAs<clang::RecordType>())
        return llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(Record->getDecl());
    else if (auto Injected = Ty->getAs<clang::InjectedClassNameType>())
    {
        if (isRestrict)
        {
            Ty.dump();
            luisa::log_error("unsupportted, InjectedClassNameType!!!!");
        }
        return nullptr;
    }
    return nullptr;
}

inline static bool isLuisaAttribute(const clang::AnnotateAttr *Anno, const char* what) {
    if (Anno->getAnnotation() == "luisa-shader")
    {
        if (Anno->args_size() >= 1) {
            auto arg = Anno->args_begin();
            if (auto Literal = llvm::dyn_cast<clang::StringLiteral>((*arg)->IgnoreParenCasts())) {
                auto _what = Literal->getString();
                return (_what == what);
            }
        }
    }
    return false;
}

inline static bool isDump(const clang::AnnotateAttr *Anno) { return isLuisaAttribute(Anno, "dump"); }
inline static bool isIgnore(const clang::AnnotateAttr *Anno) { return isLuisaAttribute(Anno, "ignore"); }
inline static bool isByPass(const clang::AnnotateAttr *Anno) { return isLuisaAttribute(Anno, "bypass"); }
inline static bool isAccess(const clang::AnnotateAttr *Anno) { return isLuisaAttribute(Anno, "access"); }
inline static bool isFuncRef(const clang::AnnotateAttr *Anno) { return isLuisaAttribute(Anno, "funcref"); }
inline static bool isCallop(const clang::AnnotateAttr *Anno) { return isLuisaAttribute(Anno, "callop"); }
inline static bool isExtCall(const clang::AnnotateAttr *Anno) { return isLuisaAttribute(Anno, "ext_call"); }
inline static bool isBuiltinType(const clang::AnnotateAttr *Anno) { return isLuisaAttribute(Anno, "builtin"); }
inline static bool isNoignore(const clang::AnnotateAttr *Anno) { return isLuisaAttribute(Anno, "noignore"); }
inline static bool isBinop(const clang::AnnotateAttr *Anno) { return isLuisaAttribute(Anno, "binop"); }
inline static bool isUnaop(const clang::AnnotateAttr *Anno) { return isLuisaAttribute(Anno, "unaop"); }
inline static bool isExpr(const clang::AnnotateAttr *Anno) { return isLuisaAttribute(Anno, "expr"); }
inline static bool isSwizzle(const clang::AnnotateAttr *Anno) { return isLuisaAttribute(Anno, "swizzle"); }
inline static bool isExport(const clang::AnnotateAttr *Anno) { return isLuisaAttribute(Anno, "export"); }
inline static bool isVertex(const clang::AnnotateAttr *Anno) { return isLuisaAttribute(Anno, "vertex"); }
inline static bool isPixel(const clang::AnnotateAttr *Anno) { return isLuisaAttribute(Anno, "pixel"); }

inline static bool isKernel1D(const clang::AnnotateAttr *Anno) { return isLuisaAttribute(Anno, "kernel_1d"); }
inline static bool isKernel2D(const clang::AnnotateAttr *Anno) { return isLuisaAttribute(Anno, "kernel_2d"); }
inline static bool isKernel3D(const clang::AnnotateAttr *Anno) { return isLuisaAttribute(Anno, "kernel_3d"); }
inline static bool isKernel(const clang::AnnotateAttr *Anno) { return isKernel1D(Anno) || isKernel2D(Anno) || isKernel3D(Anno); }

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

inline static bool isAccess(const clang::Decl *decl) {
    if (auto cxxField = llvm::dyn_cast<clang::FieldDecl>(decl)) {
        if (cxxField->isAnonymousStructOrUnion())
            return true;
    }
    for (auto Anno = decl->specific_attr_begin<clang::AnnotateAttr>();
        Anno != decl->specific_attr_end<clang::AnnotateAttr>(); ++Anno) {
        if (isAccess(*Anno))
            return true;
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

inline static llvm::StringRef getExtCallName(const clang::AnnotateAttr *Anno) {
    if (!isExtCall(Anno))
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

inline static llvm::StringRef getBinopName(const clang::AnnotateAttr *Anno) {
    if (!isBinop(Anno))
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

inline static llvm::StringRef getUnaopName(const clang::AnnotateAttr *Anno) {
    if (!isUnaop(Anno))
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

inline static llvm::StringRef getExprName(const clang::AnnotateAttr *Anno) {
    if (!isExpr(Anno))
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

inline static bool isSwizzle(const clang::Decl *decl) {
    for (auto Anno = decl->specific_attr_begin<clang::AnnotateAttr>();
        Anno != decl->specific_attr_end<clang::AnnotateAttr>(); ++Anno) {
        if (isSwizzle(*Anno))
            return true;
    }
    if (auto cxxDeclare = dyn_cast<clang::DeclaratorDecl>(decl))
    {
        if (auto typeDecl = GetRecordDeclFromQualType(cxxDeclare->getType(), false))
        {
            for (auto Anno = typeDecl->specific_attr_begin<clang::AnnotateAttr>();
                Anno != typeDecl->specific_attr_end<clang::AnnotateAttr>(); ++Anno) {
                if (isSwizzle(*Anno))
                    return true;
            }
        }
    }
    return false;
}

}// namespace luisa::clangcxx