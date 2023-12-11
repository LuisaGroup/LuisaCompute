#pragma once
#include "clang/AST/Attr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Expr.h"

namespace luisa::clangcxx {
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
}// namespace luisa::clangcxx