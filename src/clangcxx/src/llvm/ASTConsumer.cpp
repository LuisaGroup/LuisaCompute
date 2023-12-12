#include "ASTConsumer.h"
#include "defer.hpp"
#include "clang/AST/Stmt.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "AttributeHelpers.hpp"
#include <iostream>
#include <luisa/core/magic_enum.h>
#include <luisa/ast/op.h>
#include <luisa/vstl/common.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/dsl/syntax.h>
#include <filesystem>

namespace luisa::clangcxx {

using namespace clang;
using namespace clang::ast_matchers;
using namespace luisa::compute;

using CXXBinOp = clang::BinaryOperator::Opcode;
using LCBinOp = luisa::compute::BinaryOp;
using CXXUnaryOp = clang::UnaryOperator::Opcode;
using LCUnaryOp = luisa::compute::UnaryOp;

inline luisa::compute::BinaryOp TranslateBinaryOp(clang::BinaryOperator::Opcode op) {
    switch (op) {
        case CXXBinOp::BO_Add: return LCBinOp::ADD;
        case CXXBinOp::BO_Sub: return LCBinOp::SUB;
        case CXXBinOp::BO_Mul: return LCBinOp::MUL;
        case CXXBinOp::BO_Div: return LCBinOp::DIV;
        case CXXBinOp::BO_Rem: return LCBinOp::MOD;
        case CXXBinOp::BO_And: return LCBinOp::BIT_AND;
        case CXXBinOp::BO_Or: return LCBinOp::BIT_OR;
        case CXXBinOp::BO_Xor: return LCBinOp::BIT_XOR;
        case CXXBinOp::BO_Shl: return LCBinOp::SHL;
        case CXXBinOp::BO_Shr: return LCBinOp::SHR;
        case CXXBinOp::BO_LAnd: return LCBinOp::AND;
        case CXXBinOp::BO_LOr: return LCBinOp::OR;

        case CXXBinOp::BO_LT: return LCBinOp::LESS;
        case CXXBinOp::BO_GT: return LCBinOp::GREATER;
        case CXXBinOp::BO_LE: return LCBinOp::LESS_EQUAL;
        case CXXBinOp::BO_GE: return LCBinOp::GREATER_EQUAL;
        case CXXBinOp::BO_EQ: return LCBinOp::EQUAL;
        case CXXBinOp::BO_NE: return LCBinOp::NOT_EQUAL;
        default:
            luisa::log_error("unsupportted binary op {}!", op);
            return LCBinOp::ADD;
    }
}

inline luisa::compute::BinaryOp TranslateBinaryAssignOp(clang::BinaryOperator::Opcode op) {
    switch (op) {
        case CXXBinOp::BO_AddAssign: return LCBinOp::ADD;
        case CXXBinOp::BO_SubAssign: return LCBinOp::SUB;
        case CXXBinOp::BO_MulAssign: return LCBinOp::MUL;
        case CXXBinOp::BO_DivAssign: return LCBinOp::DIV;
        case CXXBinOp::BO_RemAssign: return LCBinOp::MOD;
        case CXXBinOp::BO_AndAssign: return LCBinOp::BIT_AND;
        case CXXBinOp::BO_OrAssign: return LCBinOp::BIT_OR;
        case CXXBinOp::BO_XorAssign: return LCBinOp::BIT_XOR;
        case CXXBinOp::BO_ShlAssign: return LCBinOp::SHL;
        case CXXBinOp::BO_ShrAssign: return LCBinOp::SHR;
        default:
            luisa::log_error("unsupportted binary-assign op {}!", op);
            return LCBinOp::ADD;
    }
}

inline bool IsUnaryAssignOp(CXXUnaryOp op) {
    switch (op) {
        case CXXUnaryOp::UO_PreInc:
        case CXXUnaryOp::UO_PostInc:
        case CXXUnaryOp::UO_PreDec:
        case CXXUnaryOp::UO_PostDec:
            return true;
        default:
            return false;
    }
}

inline luisa::compute::UnaryOp TranslateUnaryOp(CXXUnaryOp op) {
    switch (op) {
        case CXXUnaryOp::UO_Plus: return LCUnaryOp::PLUS;
        case CXXUnaryOp::UO_Minus: return LCUnaryOp::MINUS;
        case CXXUnaryOp::UO_Not: return LCUnaryOp::BIT_NOT;
        case CXXUnaryOp::UO_LNot: return LCUnaryOp::NOT;
        default:
            luisa::log_error("unsupportted unary op {}!", op);
            return LCUnaryOp::PLUS;
    }
}

inline static void Remove(luisa::string &str, const luisa::string &remove_str) {
    for (size_t i; (i = str.find(remove_str)) != luisa::string::npos;)
        str.replace(i, remove_str.length(), "");
}

inline static luisa::string GetNonQualifiedTypeName(clang::QualType type, const clang::ASTContext *ctx) {
    type = type.getCanonicalType();
    type.removeLocalFastQualifiers();
    auto baseName = luisa::string(type.getAsString(ctx->getLangOpts()));
    Remove(baseName, "struct ");
    Remove(baseName, "class ");
    return baseName;
}

inline const luisa::compute::RefExpr *LC_ArgOrRef(clang::QualType qt, luisa::shared_ptr<compute::detail::FunctionBuilder> fb, const luisa::compute::Type *lc_type) {
    if (qt->isPointerType())
        luisa::log_error("pointer type is not supported: [{}]", qt.getAsString());
    else if (qt->isReferenceType())
        return fb->reference(lc_type);
    else
        return fb->argument(lc_type);
    return nullptr;
}

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

CXXBlackboard::CXXBlackboard() {
    using namespace luisa;
    using namespace luisa::compute;

    ops_map.reserve(call_op_count);
    for (auto i : vstd::range(call_op_count)) {
        const auto op = (CallOp)i;
        ops_map.emplace(luisa::to_string(op), op);
    }
}

CXXBlackboard::~CXXBlackboard() {
    for (auto &&[name, type] : type_map) {
        std::cout << name << " - ";
        std::cout << type->description() << std::endl;
    }
    for (auto &&[name, global] : globals) {
        std::cout << name << " - ";
        std::cout << global->type()->description() << std::endl;
    }
}

bool CXXBlackboard::registerType(clang::QualType Ty, const clang::ASTContext *astContext, const luisa::compute::Type *type) {
    auto name = GetNonQualifiedTypeName(Ty, astContext);
    type_map[name] = type;
    return true;
}

const luisa::compute::Type *CXXBlackboard::findType(const clang::QualType Ty, const clang::ASTContext *astContext) {
    auto name = GetNonQualifiedTypeName(Ty, astContext);
    auto iter = type_map.find(name);
    if (iter != type_map.end()) {
        return iter->second;
    }
    return nullptr;
}

const luisa::compute::Type *CXXBlackboard::FindOrAddType(const clang::QualType Ty, const clang::ASTContext *astContext) {
    auto name = GetNonQualifiedTypeName(Ty, astContext);
    auto iter = type_map.find(name);
    if (iter != type_map.end()) {
        return iter->second;
    } else if (auto _type = RecordType(Ty)) {
        type_map[name] = _type;
        return _type;
    } else {
        luisa::log_error("unfound type: {}", name);
    }
    return nullptr;
}

luisa::compute::CallOp CXXBlackboard::FindCallOp(const luisa::string_view &name) {
    auto iter = ops_map.find(name);
    if (iter) {
        return iter.value();
    }
    luisa::log_error("unfound call op: {}", name);
    return CallOp::ASIN;
}

void CXXBlackboard::commentSourceLoc(luisa::shared_ptr<compute::detail::FunctionBuilder> fb, const luisa::string &prefix, const clang::SourceLocation &loc) {
    const auto &SM = astContext->getSourceManager();
    auto RawLocString = loc.printToString(SM);
    luisa::string fmt = prefix + ", at {}";
    fb->comment_(luisa::format(fmt, RawLocString.data()));
}

static constexpr auto kUseComment = true;
CXXBlackboard::Commenter CXXBlackboard::CommentStmt_(luisa::shared_ptr<compute::detail::FunctionBuilder> fb, const clang::Stmt *x) {
    if (kUseComment) {
        if (auto cxxDecl = llvm::dyn_cast<clang::DeclStmt>(x)) {
            return Commenter(
                [=] {
                    const DeclGroupRef declGroup = cxxDecl->getDeclGroup();
                    for (auto decl : declGroup) {
                        if (!decl) continue;
                        if (auto *varDecl = dyn_cast<clang::VarDecl>(decl)) {
                            luisa::string what =
                                luisa::format("VarDecl: {} {}",
                                              varDecl->getType().getAsString(),
                                              varDecl->getNameAsString());
                            commentSourceLoc(fb, what, varDecl->getBeginLoc());
                        }
                    }
                });
        } else if (auto cxxBranch = llvm::dyn_cast<clang::IfStmt>(x)) {
            return Commenter(
                [=] { commentSourceLoc(fb, "BEGIN IF", x->getBeginLoc()); },
                [=] { commentSourceLoc(fb, "END IF", x->getBeginLoc()); });
        } else if (auto cxxSwitch = llvm::dyn_cast<clang::SwitchStmt>(x)) {
            return Commenter(
                [=] { commentSourceLoc(fb, "BEGIN SWITCH", x->getBeginLoc()); },
                [=] { commentSourceLoc(fb, "END SWITCH", x->getBeginLoc()); });
        } else if (auto cxxWhile = llvm::dyn_cast<clang::WhileStmt>(x)) {
            return Commenter(
                [=] { commentSourceLoc(fb, "BEGIN WHILE", x->getBeginLoc()); },
                [=] { commentSourceLoc(fb, "END WHILE", x->getBeginLoc()); });
        } else if (auto cxxCompound = llvm::dyn_cast<clang::CompoundStmt>(x)) {
            return Commenter(
                [=] { commentSourceLoc(fb, "BEGIN SCOPE", x->getBeginLoc()); },
                [=] { commentSourceLoc(fb, "END SCOPE", x->getBeginLoc()); });
        } else if (auto ret = llvm::dyn_cast<clang::ReturnStmt>(x)) {
            return Commenter([=] { commentSourceLoc(fb, "RETURN", x->getBeginLoc()); });
        } else if (auto ca = llvm::dyn_cast<CompoundAssignOperator>(x)) {
            return Commenter([=] { commentSourceLoc(fb, "COMPOUND ASSIGN", ca->getBeginLoc()); });
        } else if (auto bin = llvm::dyn_cast<BinaryOperator>(x)) {
            return Commenter([=] { if(bin->isAssignmentOp()) commentSourceLoc(fb, "ASSIGN", bin->getBeginLoc()); });
        } else if (auto call = llvm::dyn_cast<clang::CallExpr>(x)) {
            auto cxxFunc = call->getCalleeDecl()->getAsFunction();
            if (auto Method = llvm::dyn_cast<clang::CXXMethodDecl>(cxxFunc)) {
                if (Method->getParent()->isLambda())
                    return Commenter([=] { commentSourceLoc(fb, luisa::format("CALL LAMBDA: {}", Method->getParent()->getName().data()), x->getBeginLoc()); });
                else
                    return Commenter([=] { commentSourceLoc(fb, luisa::format("CALL METHOD: {}::{}", Method->getParent()->getName().data(), Method->getName().data()), x->getBeginLoc()); });
            } else
                return Commenter([=] { commentSourceLoc(fb, luisa::format("CALL FUNCTION: {}", cxxFunc->getName().data()), x->getBeginLoc()); });
        } else if (auto cxxCtor = llvm::dyn_cast<CXXConstructExpr>(x)) {
            auto cxxCtorDecl = cxxCtor->getConstructor();
            return Commenter([=] { commentSourceLoc(fb, luisa::format("CONSTRUCT: {}", cxxCtorDecl->getParent()->getName().data()), x->getBeginLoc()); });
        } else if (auto cxxDefaultArg = llvm::dyn_cast<clang::CXXDefaultArgExpr>(x)) {
            return Commenter([=] {
                auto funcDecl = llvm::dyn_cast<FunctionDecl>(cxxDefaultArg->getParam()->getParentFunctionOrMethod());
                commentSourceLoc(fb,
                                 luisa::format("DEFAULT ARG: {}::{}",
                                               funcDecl ? funcDecl->getQualifiedNameAsString() : "unknown-func",
                                               cxxDefaultArg->getParam()->getName().data()),
                                 x->getBeginLoc());
            });
        }
    }
    return {{}, {}};
}

const luisa::compute::Type *CXXBlackboard::RecordAsPrimitiveType(const clang::QualType Ty) {
    const luisa::compute::Type *_type = nullptr;
    if (auto builtin = Ty->getAs<clang::BuiltinType>()) {
        // clang-format off
        switch (builtin->getKind()) {
            /*
            case (BuiltinType::Kind::SChar): _type = Type::of<signed char>(); break;
            case (BuiltinType::Kind::Char_S): _type = Type::of<signed char>(); break;
            case (BuiltinType::Kind::Char8): _type = Type::of<signed char>(); break;

            case (BuiltinType::Kind::UChar): _type = Type::of<unsigned char>(); break;
            case (BuiltinType::Kind::Char_U): _type = Type::of<unsigned char>(); break;

            case (BuiltinType::Kind::Char16): _type = Type::of<char16_t>(); break;
            */
            case (BuiltinType::Kind::Bool): _type = Type::of<bool>(); break;

            case (BuiltinType::Kind::UShort): _type = Type::of<uint16_t>(); break;
            case (BuiltinType::Kind::UInt): _type = Type::of<uint32_t>(); break;
            case (BuiltinType::Kind::ULong): _type = Type::of<uint32_t>(); break;
            case (BuiltinType::Kind::ULongLong): _type = Type::of<uint64_t>(); break;

            case (BuiltinType::Kind::Short): _type = Type::of<int16_t>(); break;
            case (BuiltinType::Kind::Int): _type = Type::of<int32_t>(); break;
            case (BuiltinType::Kind::Long): _type = Type::of<int32_t>(); break;
            case (BuiltinType::Kind::LongLong): _type = Type::of<int64_t>(); break;

            case (BuiltinType::Kind::Float): _type = Type::of<float>(); break;
            case (BuiltinType::Kind::Double): _type = Type::of<double>(); break;

            default: 
            {
                luisa::log_error("unsupported field primitive type: [{}], kind [{}]",
                    Ty.getAsString(), builtin->getKind());
            }
            break;
        }
        // clang-format on
    }
    if (_type) {
        registerType(Ty, astContext, _type);
    }
    return _type;
}

const luisa::compute::Type *CXXBlackboard::RecordAsBuiltinType(const QualType Ty) {
    auto decl = GetRecordDeclFromQualType(Ty);
    const luisa::compute::Type *_type = nullptr;
    bool ext_builtin = false;
    llvm::StringRef builtin_type_name = {};
    for (auto Anno = decl->specific_attr_begin<clang::AnnotateAttr>();
         Anno != decl->specific_attr_end<clang::AnnotateAttr>(); ++Anno) {
        if (ext_builtin = isBuiltinType(*Anno)) {
            builtin_type_name = getBuiltinTypeName(*Anno);
        }
    }

    if (ext_builtin) {
        if (builtin_type_name == "vec") {
            if (auto TST = Ty->getAs<TemplateSpecializationType>()) {
                auto Arguments = TST->template_arguments();
                if (auto EType = Arguments[0].getAsType()->getAs<clang::BuiltinType>()) {
                    clang::Expr::EvalResult Result;
                    if (Arguments[1].getAsExpr()->EvaluateAsConstantExpr(Result, *astContext)) {
                        auto N = Result.Val.getInt().getExtValue();
                        // TST->dump();
                        // clang-format off
                        switch (EType->getKind()) {
#define CASE_VEC_TYPE(type)                                                                                    \
    switch (N) {                                                                                               \
        case 2: { _type = Type::of<type##2>(); } break;                                            \
        case 3: { _type = Type::of<type##3>(); } break;                                            \
        case 4: { _type = Type::of<type##4>(); } break;                                            \
        default: {                                                                                             \
            luisa::log_error("unsupported type: {}, kind {}, N {}", Ty.getAsString(), EType->getKind(), N);    \
        } break;                                                                                               \
    }
                            case (BuiltinType::Kind::Bool): { CASE_VEC_TYPE(bool) } break;
                            case (BuiltinType::Kind::Float): { CASE_VEC_TYPE(float) } break;
                            case (BuiltinType::Kind::Long): { CASE_VEC_TYPE(slong) } break;
                            case (BuiltinType::Kind::Int): { CASE_VEC_TYPE(int) } break;
                            case (BuiltinType::Kind::ULong): { CASE_VEC_TYPE(ulong) } break;
                            case (BuiltinType::Kind::UInt): { CASE_VEC_TYPE(uint) } break;
                            case (BuiltinType::Kind::Double): { CASE_VEC_TYPE(double) } break;
                            default: {
                                luisa::log_error("unsupported type: {}, kind {}", Ty.getAsString(), EType->getKind());
                            } break;
#undef CASE_VEC_TYPE
                        }
                        // clang-format on
                    }
                }
            } else {
                Ty->dump();
            }
        } else if (builtin_type_name == "array") {
            if (auto TST = Ty->getAs<TemplateSpecializationType>()) {
                auto Arguments = TST->template_arguments();
                clang::Expr::EvalResult Result;
                if (Arguments[1].getAsExpr()->EvaluateAsConstantExpr(Result, *astContext)) {
                    auto N = Result.Val.getInt().getExtValue();
                    if (auto lc_type = FindOrAddType(Arguments[0].getAsType(), astContext)) {
                        _type = Type::array(lc_type, N);
                    } else {
                        luisa::log_error("unfound array element type: {}", Arguments[0].getAsType().getAsString());
                    }
                }
            }
        } else if (builtin_type_name == "matrix") {
            if (auto TST = Ty->getAs<TemplateSpecializationType>()) {
                auto Arguments = TST->template_arguments();
                clang::Expr::EvalResult Result;
                if (Arguments[0].getAsExpr()->EvaluateAsConstantExpr(Result, *astContext)) {
                    auto N = Result.Val.getInt().getExtValue();
                    _type = Type::matrix(N);
                }
            }
        } else if (builtin_type_name == "ray_query_all") {
            _type = Type::custom("LC_RayQueryAll");
        } else if (builtin_type_name == "ray_query_any") {
            _type = Type::custom("LC_RayQueryAny");
        } else if (builtin_type_name == "accel") {
            _type = Type::of<luisa::compute::Accel>();
        } else if (builtin_type_name == "buffer") {
            if (auto TST = Ty->getAs<TemplateSpecializationType>()) {
                auto Arguments = TST->template_arguments();
                clang::Expr::EvalResult Result;
                if (auto lc_type = FindOrAddType(Arguments[0].getAsType(), astContext)) {
                    _type = Type::buffer(lc_type);
                } else {
                    luisa::log_error("unfound buffer element type: {}", Arguments[0].getAsType().getAsString());
                }
            }
        } else {
            luisa::log_error("unsupported builtin type: {}", luisa::string(builtin_type_name));
        }
    }
    if (_type) {
        registerType(Ty, astContext, _type);
    }
    return _type;
}

const luisa::compute::Type *CXXBlackboard::RecordAsStuctureType(const clang::QualType Ty) {
    if (const luisa::compute::Type *_type = findType(Ty, astContext)) {
        return _type;
    } else {
        auto S = GetRecordDeclFromQualType(Ty);
        bool ignore = false;
        for (auto Anno = S->specific_attr_begin<clang::AnnotateAttr>(); Anno != S->specific_attr_end<clang::AnnotateAttr>(); ++Anno) {
            ignore |= isIgnore(*Anno) || isBuiltinType(*Anno);
        }
        if (ignore)
            return nullptr;

        for (auto f = S->field_begin(); f != S->field_end(); f++) {
            if (f->getType()->isTemplateTypeParmType())
                return nullptr;
        }

        luisa::vector<const luisa::compute::Type *> types;
        if (!S->isLambda()) { // ignore lambda generated capture fields
            for (auto f = S->field_begin(); f != S->field_end(); f++) {
                auto Ty = f->getType();
                if (!tryEmplaceFieldType(Ty, S, types)) {
                    S->dump();
                    luisa::log_error("unsupported field type [{}] in type [{}]", Ty.getAsString(), S->getNameAsString());
                }
            }
        }
        // align
        uint64_t alignment = 4;
        for (auto ft : types) {
            alignment = std::max(alignment, ft->alignment());
        }
        auto lc_type = Type::structure(alignment, types);
        QualType Ty = S->getTypeForDecl()->getCanonicalTypeInternal();
        registerType(Ty, astContext, lc_type);
        return lc_type;
    }
}

const luisa::compute::Type *CXXBlackboard::RecordType(const clang::QualType Qt) {
    const luisa::compute::Type *_type = nullptr;
    clang::QualType Ty = Qt.getNonReferenceType();

    // 1. PRIMITIVE
    if (auto builtin = Ty->getAs<clang::BuiltinType>()) {
        _type = RecordAsPrimitiveType(Ty);
        if (!_type) {
            luisa::log_error("unsupported field primitive type: [{}], kind [{}]",
                             Ty.getAsString(), builtin->getKind());
        }
    } else {
        // 2. EMPLACE RECORD
        if (clang::RecordDecl *recordDecl = GetRecordDeclFromQualType(Ty)) {
            // 2.1 AS BUILTIN
            if (!_type) {
                _type = RecordAsBuiltinType(Ty);
            }
            // 2.2 AS STRUCTURE
            if (!_type) {
                _type = RecordAsStuctureType(Ty);
            }
        } else if (Ty->isTemplateTypeParmType()) {
            luisa::log_verbose("template type parameter type...");
            return nullptr;
        } else {
            recordDecl->dump();
            luisa::log_error("unsupported & unresolved type [{}]", Ty.getAsString());
        }
    }
    if (!_type) {
        luisa::log_error("unsupported type [{}]", Ty.getAsString());
    }
    return _type;
}

bool CXXBlackboard::tryEmplaceFieldType(const clang::QualType Qt, const clang::RecordDecl *decl, luisa::vector<const luisa::compute::Type *> &types) {
    if (auto _type = RecordType(Qt)) {
        types.emplace_back(_type);
        return true;
    }
    return false;
}

struct ExprTranslator : public clang::RecursiveASTVisitor<ExprTranslator> {
    const luisa::compute::Expression *caller = nullptr;
    clang::ForStmt *currentCxxForStmt = nullptr;

    bool TraverseStmt(clang::Stmt *x) {
        if (x == nullptr) return true;

        // CONTROL WALK
        if (auto cxxLambda = llvm::dyn_cast<LambdaExpr>(x)) {
            auto cxxCallee = cxxLambda->getLambdaClass()->getLambdaCallOperator();
            Stack newStack = *stack;
            FunctionBuilderBuilder bdbd(bb, newStack);
            bdbd.build(cxxCallee);
        } else if (auto cxxBranch = llvm::dyn_cast<clang::IfStmt>(x)) {
            auto _ = bb->CommentStmt_(fb, cxxBranch);

            auto cxxCond = cxxBranch->getCond();
            TraverseStmt(cxxCond);

            auto lc_if_ = fb->if_(stack->expr_map[cxxCond]);
            fb->push_scope(lc_if_->true_branch());
            if (cxxBranch->getThen())
                TraverseStmt(cxxBranch->getThen());
            fb->pop_scope(lc_if_->true_branch());

            fb->push_scope(lc_if_->false_branch());
            if (cxxBranch->getElse())
                TraverseStmt(cxxBranch->getElse());
            fb->pop_scope(lc_if_->false_branch());
        } else if (auto cxxSwitch = llvm::dyn_cast<clang::SwitchStmt>(x)) {
            auto _ = bb->CommentStmt_(fb, cxxSwitch);

            auto cxxCond = cxxSwitch->getCond();
            TraverseStmt(cxxCond);

            auto lc_switch_ = fb->switch_(stack->expr_map[cxxSwitch->getCond()]);
            fb->push_scope(lc_switch_->body());
            luisa::vector<clang::SwitchCase *> cxxCases;
            if (auto caseList = cxxSwitch->getSwitchCaseList()) {
                while (caseList) {
                    cxxCases.emplace_back(caseList);
                    caseList = caseList->getNextSwitchCase();
                }
                std::reverse(cxxCases.begin(), cxxCases.end());
                for (auto cxxCase : cxxCases)
                    TraverseStmt(cxxCase);
            }
            fb->pop_scope(lc_switch_->body());
        } else if (auto cxxCase = llvm::dyn_cast<clang::CaseStmt>(x)) {
            auto cxxCond = cxxCase->getLHS();
            TraverseStmt(cxxCond);

            auto lc_case_ = fb->case_(stack->expr_map[cxxCond]);
            fb->push_scope(lc_case_->body());
            if (auto cxxBody = cxxCase->getSubStmt())
                TraverseStmt(cxxBody);
            fb->pop_scope(lc_case_->body());
        } else if (auto cxxDefault = llvm::dyn_cast<clang::DefaultStmt>(x)) {
            auto lc_default_ = fb->default_();
            fb->push_scope(lc_default_->body());
            if (auto cxxBody = cxxDefault->getSubStmt())
                TraverseStmt(cxxBody);
            fb->pop_scope(lc_default_->body());
        } else if (auto cxxContinue = llvm::dyn_cast<clang::ContinueStmt>(x)) {
            if (currentCxxForStmt)
                TraverseStmt(currentCxxForStmt->getInc());
            fb->continue_();
        } else if (auto cxxBreak = llvm::dyn_cast<clang::BreakStmt>(x)) {
            fb->break_();
        } else if (auto cxxWhile = llvm::dyn_cast<clang::WhileStmt>(x)) {
            auto _ = bb->CommentStmt_(fb, cxxWhile);

            auto lc_while_ = fb->loop_();
            // while (cond)
            fb->push_scope(lc_while_->body());
            {
                auto cxxCond = cxxWhile->getCond();
                TraverseStmt(cxxCond);
                auto lc_cond_if_ = fb->if_(stack->expr_map[cxxCond]);
                // break
                fb->push_scope(lc_cond_if_->false_branch());
                fb->break_();
                fb->pop_scope(lc_cond_if_->false_branch());
                // body
                auto cxxBody = cxxWhile->getBody();
                TraverseStmt(cxxBody);
            }
            fb->pop_scope(lc_while_->body());
        } else if (auto cxxFor = llvm::dyn_cast<clang::ForStmt>(x)) {
            currentCxxForStmt = cxxFor;
            auto lc_while_ = fb->loop_();
            // i = 0
            auto cxxInit = cxxFor->getInit();
            TraverseStmt(cxxInit);
            // while (cond)
            fb->push_scope(lc_while_->body());
            {
                auto cxxCond = cxxFor->getCond();
                TraverseStmt(cxxCond);
                auto lc_cond_if_ = fb->if_(stack->expr_map[cxxCond]);
                // break
                fb->push_scope(lc_cond_if_->false_branch());
                fb->break_();
                fb->pop_scope(lc_cond_if_->false_branch());
                // body
                auto cxxBody = cxxFor->getBody();
                TraverseStmt(cxxBody);
                // i++
                auto cxxInc = cxxFor->getInc();
                TraverseStmt(cxxInc);
            }
            fb->pop_scope(lc_while_->body());
        } else if (auto cxxCompound = llvm::dyn_cast<clang::CompoundStmt>(x)) {
            auto _ = bb->CommentStmt_(fb, cxxCompound);

            for (auto sub : cxxCompound->body())
                TraverseStmt(sub);
        } else {
            RecursiveASTVisitor<ExprTranslator>::TraverseStmt(x);
        }

        // TRANSLATE
        const luisa::compute::Expression *current = nullptr;
        if (x) {
            if (auto cxxDecl = llvm::dyn_cast<clang::DeclStmt>(x)) {
                auto _ = bb->CommentStmt_(fb, cxxDecl);

                const DeclGroupRef declGroup = cxxDecl->getDeclGroup();
                for (auto decl : declGroup) {
                    if (!decl) continue;

                    if (auto *varDecl = dyn_cast<clang::VarDecl>(decl)) {
                        auto Ty = varDecl->getType();
                        if (auto lc_type = bb->FindOrAddType(Ty, bb->astContext)) {
                            auto lc_var = fb->local(lc_type);
                            stack->locals[varDecl] = lc_var;

                            auto init = varDecl->getInit();
                            if (auto lc_init = stack->expr_map[init]) {
                                fb->assign(lc_var, lc_init);
                                current = lc_var;
                            } else {
                                current = lc_var;
                            }
                        }
                    } else {
                        luisa::log_error("unsupported decl stmt: {}", cxxDecl->getStmtClassName());
                    }
                }
            } else if (auto cxxRet = llvm::dyn_cast<clang::ReturnStmt>(x)) {
                auto _ = bb->CommentStmt_(fb, cxxRet);

                auto cxx_ret = cxxRet->getRetValue();
                auto lc_ret = stack->expr_map[cxx_ret];
                if (fb->tag() != compute::Function::Tag::KERNEL) {
                    fb->return_(lc_ret);
                }
            } else if (auto ce = llvm::dyn_cast<clang::ConstantExpr>(x)) {
                const auto APK = ce->getResultAPValueKind();
                const auto &APV = ce->getAPValueResult();
                switch (APK) {
                    case clang::APValue::ValueKind::Int:
                        current = fb->literal(Type::of<int>(), (int)ce->getResultAsAPSInt().getLimitedValue());
                        break;
                    case clang::APValue::ValueKind::Float:
                        current = fb->literal(Type::of<float>(), (float)APV.getFloat().convertToFloat());
                        break;
                    default:
                        luisa::log_error("unsupportted ConstantExpr APValueKind {}", APK);
                        break;
                }
            } else if (auto il = llvm::dyn_cast<IntegerLiteral>(x)) {
                current = fb->literal(Type::of<int>(), (int)il->getValue().getLimitedValue());
            } else if (auto bl = llvm::dyn_cast<CXXBoolLiteralExpr>(x)) {
                current = fb->literal(Type::of<bool>(), (bool)bl->getValue());
            } else if (auto fl = llvm::dyn_cast<FloatingLiteral>(x)) {
                current = fb->literal(Type::of<float>(), (float)fl->getValue().convertToFloat());
            } else if (auto cxxCtor = llvm::dyn_cast<CXXConstructExpr>(x)) {
                auto _ = bb->CommentStmt_(fb, cxxCtor);

                // TODO: REFACTOR THIS
                auto calleeDecl = cxxCtor->getConstructor();
                if (!bb->func_builders.contains(calleeDecl)) {
                    auto funcDecl = calleeDecl->getAsFunction();
                    auto methodDecl = llvm::dyn_cast<clang::CXXMethodDecl>(funcDecl);
                    const auto isTemplateInstant = funcDecl->isTemplateInstantiation();
                    if (isTemplateInstant) {
                        FunctionBuilderBuilder fbfb(bb, *stack);
                        fbfb.build(calleeDecl->getAsFunction());
                    }
                }

                auto local = fb->local(bb->FindOrAddType(cxxCtor->getType(), bb->astContext));
                // args
                luisa::vector<const luisa::compute::Expression *> lc_args;
                lc_args.emplace_back(local);
                for (auto arg : cxxCtor->arguments()) {
                    if (auto lc_arg = stack->expr_map[arg])
                        lc_args.emplace_back(lc_arg);
                    else
                        luisa::log_error("unfound arg: {}", arg->getStmtClassName());
                }
                if (auto callable = bb->func_builders[cxxCtor->getConstructor()]) {
                    fb->call(luisa::compute::Function(callable.get()), lc_args);
                    current = local;
                } else if (cxxCtor->getConstructor()->getParent()->isLambda()) {
                    // ...IGNORE LAMBDA CTOR...
                    current = local;
                } else {
                    bool isBuiltin = false;
                    llvm::StringRef builtinName = {};
                    if (auto thisType = GetRecordDeclFromQualType(cxxCtor->getType())) {
                        for (auto Anno = thisType->specific_attr_begin<clang::AnnotateAttr>(); Anno != thisType->specific_attr_end<clang::AnnotateAttr>(); ++Anno) {
                            if (isBuiltinType(*Anno)) {
                                isBuiltin = true;
                                builtinName = getBuiltinTypeName(*Anno);
                                break;
                            }
                        }
                    }
                    if (isBuiltin) {
                        if (builtinName == "vec") {
                            auto Ty = cxxCtor->getType();
                            if (auto TST = Ty->getAs<TemplateSpecializationType>()) {
                                auto Arguments = TST->template_arguments();
                                if (auto EType = Arguments[0].getAsType()->getAs<clang::BuiltinType>()) {
                                    clang::Expr::EvalResult Result;
                                    if (Arguments[1].getAsExpr()->EvaluateAsConstantExpr(Result, *bb->astContext)) {
                                        auto N = Result.Val.getInt().getExtValue();
                                        // TST->dump();
                                        // clang-format off
                        switch (EType->getKind()) {
#define CASE_VEC_TYPE(stype, type)                                                                                    \
    switch (N) {                                                                                               \
        case 2: { auto lc_type = Type::of<stype##2>(); current = fb->call(lc_type, CallOp::MAKE_##type##2, { lc_args.begin() + 1, lc_args.end() }); } break;                                            \
        case 3: { auto lc_type = Type::of<stype##3>(); current = fb->call(lc_type, CallOp::MAKE_##type##3, { lc_args.begin() + 1, lc_args.end() }); } break;                                            \
        case 4: { auto lc_type = Type::of<stype##4>(); current = fb->call(lc_type, CallOp::MAKE_##type##4, { lc_args.begin() + 1, lc_args.end() }); } break;                                            \
        default: {                                                                                             \
            luisa::log_error("unsupported type: {}, kind {}, N {}", Ty.getAsString(), EType->getKind(), N);    \
        } break;                                                                                               \
    }
                            case (BuiltinType::Kind::Bool): { CASE_VEC_TYPE(bool, BOOL) } break;
                            case (BuiltinType::Kind::Float): { CASE_VEC_TYPE(float, FLOAT) } break;
                            case (BuiltinType::Kind::Long): { CASE_VEC_TYPE(slong, LONG) } break;
                            case (BuiltinType::Kind::Int): { CASE_VEC_TYPE(int, INT) } break;
                            case (BuiltinType::Kind::ULong): { CASE_VEC_TYPE(ulong, ULONG) } break;
                            case (BuiltinType::Kind::UInt): { CASE_VEC_TYPE(uint, UINT) } break;
                            case (BuiltinType::Kind::Double): { CASE_VEC_TYPE(double, DOUBLE) } break;
                            default: {
                                luisa::log_error("unsupported type: {}, kind {}", Ty.getAsString(), EType->getKind());
                            } break;
#undef CASE_VEC_TYPE
                        }
                                        // clang-format on
                                    }
                                }
                            } else
                                luisa::log_error("???");
                        } else if (builtinName == "matrix") {
                            auto Ty = cxxCtor->getType();
                            if (auto TST = Ty->getAs<TemplateSpecializationType>()) {
                                auto Arguments = TST->template_arguments();
                                clang::Expr::EvalResult Result;
                                if (Arguments[0].getAsExpr()->EvaluateAsConstantExpr(Result, *bb->astContext)) {
                                    auto N = Result.Val.getInt().getExtValue();
                                    auto lc_type = Type::matrix(N);
                                    const CallOp MATRIX_LUT[3] = {CallOp::MAKE_FLOAT2X2, CallOp::MAKE_FLOAT3X3, CallOp::MAKE_FLOAT4X4};
                                    current = fb->call(lc_type, MATRIX_LUT[N - 2], {lc_args.begin() + 1, lc_args.end()});
                                };
                            } else
                                luisa::log_error("???");
                        } else {
                            luisa::log_error("unhandled builtin constructor: {}", cxxCtor->getConstructor()->getNameAsString());
                        }
                    } else {
                        cxxCtor->dump();
                        luisa::log_error("unfound constructor: {}", cxxCtor->getConstructor()->getNameAsString());
                    }
                }
            } else if (auto unary = llvm::dyn_cast<UnaryOperator>(x)) {
                const auto cxx_op = unary->getOpcode();
                const auto lhs = stack->expr_map[unary->getSubExpr()];
                const auto lc_type = bb->FindOrAddType(unary->getType(), bb->astContext);
                if (!IsUnaryAssignOp(cxx_op)) {
                    current = fb->unary(lc_type, TranslateUnaryOp(cxx_op), lhs);
                } else {
                    auto one = fb->literal(Type::of<int>(), 1);
                    auto typed_one = one;
                    // auto typed_one = fb->cast(lc_type, CastOp::STATIC, one);
                    switch (cxx_op) {
                        case clang::UO_PreInc: {
                            auto ca_expr = fb->binary(lc_type, LCBinOp::ADD, lhs, typed_one);
                            fb->assign(lhs, ca_expr);
                            current = lhs;
                            break;
                        }
                        case clang::UO_PreDec: {
                            auto ca_expr = fb->binary(lc_type, LCBinOp::SUB, lhs, typed_one);
                            fb->assign(lhs, ca_expr);
                            current = lhs;
                            break;
                        }
                        case clang::UO_PostInc: {
                            auto old = fb->local(lc_type);
                            fb->assign(old, lhs);
                            auto ca_expr = fb->binary(lc_type, LCBinOp::ADD, lhs, typed_one);
                            fb->assign(lhs, ca_expr);
                            current = old;
                            break;
                        }
                        case clang::UO_PostDec: {
                            auto old = fb->local(lc_type);
                            fb->assign(old, lhs);
                            auto ca_expr = fb->binary(lc_type, LCBinOp::SUB, lhs, typed_one);
                            fb->assign(lhs, ca_expr);
                            current = old;
                            break;
                        }
                    }
                }
            } else if (auto bin = llvm::dyn_cast<BinaryOperator>(x)) {
                auto _ = bb->CommentStmt_(fb, bin);

                const auto cxx_op = bin->getOpcode();
                const auto lhs = stack->expr_map[bin->getLHS()];
                const auto rhs = stack->expr_map[bin->getRHS()];
                const auto lc_type = bb->FindOrAddType(bin->getType(), bb->astContext);
                if (auto ca = llvm::dyn_cast<CompoundAssignOperator>(x)) {
                    auto ca_expr = fb->binary(lc_type, TranslateBinaryAssignOp(cxx_op), lhs, rhs);
                    fb->assign(lhs, ca_expr);
                    current = lhs;
                } else if (cxx_op == CXXBinOp::BO_Assign) {
                    fb->assign(lhs, rhs);
                    current = lhs;
                } else {
                    current = fb->binary(lc_type, TranslateBinaryOp(cxx_op), lhs, rhs);
                }
            } else if (auto dref = llvm::dyn_cast<DeclRefExpr>(x)) {
                auto str = luisa::string(dref->getNameInfo().getName().getAsString());
                if (stack->locals[dref->getDecl()]) {
                    current = stack->locals[dref->getDecl()];
                } else if (auto value = dref->getDecl(); value && llvm::isa<clang::VarDecl>(value))// Value Ref
                {
                    if (auto var = value->getPotentiallyDecomposedVarDecl()) {
                        if (auto eval = var->getEvaluatedValue()) {
                            if (eval->isInt())
                                current = fb->literal(Type::of<int>(), (int)eval->getInt().getLimitedValue());
                            else if (eval->isFloat())
                                current = fb->literal(Type::of<float>(), (float)eval->getFloat().convertToFloat());
                            else
                                luisa::log_error("unsupportted eval type: {}", eval->getKind());
                        } else {
                            dref->dump();
                            luisa::log_error("unfound & unresolved ref: {}", str);
                        }
                    }
                } else if (auto value = dref->getDecl(); value && llvm::isa<clang::FunctionDecl>(value))// Func Ref
                    ;
                else
                    luisa::log_error("unfound var ref: {}", str);
            } else if (auto implicit_cast = llvm::dyn_cast<ImplicitCastExpr>(x)) {
                if (stack->expr_map[implicit_cast->getSubExpr()] != nullptr) {
                    const auto lc_type = bb->FindOrAddType(implicit_cast->getType(), bb->astContext);
                    if (!stack->expr_map[implicit_cast->getSubExpr()]) {
                        implicit_cast->getSubExpr()->dump();
                        luisa::log_error("!!!");
                    }
                    current = fb->cast(lc_type, CastOp::STATIC, stack->expr_map[implicit_cast->getSubExpr()]);
                }
            } else if (auto _c_cast = llvm::dyn_cast<CStyleCastExpr>(x)) {
                if (stack->expr_map[_c_cast->getSubExpr()] != nullptr) {
                    const auto lc_type = bb->FindOrAddType(_c_cast->getType(), bb->astContext);
                    if (!stack->expr_map[_c_cast->getSubExpr()]) {
                        _c_cast->getSubExpr()->dump();
                        luisa::log_error("!!!");
                    }
                    current = fb->cast(lc_type, CastOp::STATIC, stack->expr_map[_c_cast->getSubExpr()]);
                }
            } else if (auto _static_cast = llvm::dyn_cast<CXXStaticCastExpr>(x)) {
                if (stack->expr_map[_static_cast->getSubExpr()] != nullptr) {
                    const auto lc_type = bb->FindOrAddType(_static_cast->getType(), bb->astContext);
                    if (!stack->expr_map[_static_cast->getSubExpr()]) {
                        _static_cast->getSubExpr()->dump();
                        luisa::log_error("!!!");
                    }
                    current = fb->cast(lc_type, CastOp::STATIC, stack->expr_map[_static_cast->getSubExpr()]);
                }
            } else if (auto cxxDefaultArg = llvm::dyn_cast<clang::CXXDefaultArgExpr>(x)) {
                auto _ = bb->CommentStmt_(fb, cxxDefaultArg);

                const auto _value = fb->local(bb->FindOrAddType(cxxDefaultArg->getType(), bb->astContext));
                TraverseStmt(cxxDefaultArg->getExpr());
                fb->assign(_value, stack->expr_map[cxxDefaultArg->getExpr()]);
                current = _value;
            } else if (auto t = llvm::dyn_cast<clang::CXXThisExpr>(x)) {
                current = stack->locals[nullptr];
            } else if (auto cxxMember = llvm::dyn_cast<clang::MemberExpr>(x)) {
                if (cxxMember->isBoundMemberFunction(*bb->astContext)) {
                    auto lhs = stack->expr_map[cxxMember->getBase()];
                    caller = lhs;
                } else if (auto cxxField = llvm::dyn_cast<FieldDecl>(cxxMember->getMemberDecl())) {
                    auto lhs = stack->expr_map[cxxMember->getBase()];
                    const auto lcMemberType = bb->FindOrAddType(cxxField->getType(), bb->astContext);
                    current = fb->member(lcMemberType, lhs, cxxField->getFieldIndex());
                } else {
                    luisa::log_error("unsupported member expr: {}", cxxMember->getMemberDecl()->getNameAsString());
                }
            } else if (auto call = llvm::dyn_cast<clang::CallExpr>(x)) {
                auto _ = bb->CommentStmt_(fb, call);

                auto calleeDecl = call->getCalleeDecl();
                llvm::StringRef callopName = {};
                for (auto attr = calleeDecl->specific_attr_begin<clang::AnnotateAttr>();
                     attr != calleeDecl->specific_attr_end<clang::AnnotateAttr>(); attr++) {
                    if (callopName.empty())
                        callopName = getCallopName(*attr);
                }
                // args
                luisa::vector<const luisa::compute::Expression *> lc_args;
                if (auto mcall = llvm::dyn_cast<clang::CXXMemberCallExpr>(x)) {
                    lc_args.emplace_back(caller);// from -MemberExpr::isBoundMemberFunction
                    caller = nullptr;
                }
                for (auto arg : call->arguments()) {
                    if (auto lc_arg = stack->expr_map[arg])
                        lc_args.emplace_back(lc_arg);
                    else
                        luisa::log_error("unfound arg: {}", arg->getStmtClassName());
                }
                // call
                if (!callopName.empty()) {
                    auto op = bb->FindCallOp(callopName);
                    if (call->getCallReturnType(*bb->astContext)->isVoidType())
                        fb->call(op, lc_args);
                    else if (auto lc_type = bb->FindOrAddType(call->getCallReturnType(*bb->astContext), bb->astContext))
                        current = fb->call(lc_type, op, lc_args);
                    else
                        luisa::log_error("unfound return type: {}", call->getCallReturnType(*bb->astContext)->getCanonicalTypeInternal().getAsString());
                } else {
                    auto calleeDecl = call->getCalleeDecl();

                    // TODO: REFACTOR THIS
                    if (!bb->lambda_builders.contains(calleeDecl) && !bb->func_builders.contains(calleeDecl)) {
                        auto funcDecl = calleeDecl->getAsFunction();
                        auto methodDecl = llvm::dyn_cast<clang::CXXMethodDecl>(funcDecl);
                        const auto isTemplateInstant = funcDecl->isTemplateInstantiation();
                        const auto isLambda = methodDecl && methodDecl->getParent()->isLambda();
                        if (isTemplateInstant || isLambda) {
                            FunctionBuilderBuilder fbfb(bb, *stack);
                            fbfb.build(calleeDecl->getAsFunction());
                            calleeDecl = funcDecl;
                        }
                    }

                    if (auto func_callable = bb->func_builders[calleeDecl]) {
                        if (call->getCallReturnType(*bb->astContext)->isVoidType())
                            fb->call(luisa::compute::Function(func_callable.get()), lc_args);
                        else if (auto lc_type = bb->FindOrAddType(call->getCallReturnType(*bb->astContext), bb->astContext))
                            current = fb->call(lc_type, luisa::compute::Function(func_callable.get()), lc_args);
                        else
                            luisa::log_error("unfound return type in method/function: {}", call->getCallReturnType(*bb->astContext)->getCanonicalTypeInternal().getAsString());
                    } else if (auto lambda_callable = bb->lambda_builders[calleeDecl]) {
                        luisa::span<const luisa::compute::Expression *> lambda_args = {lc_args.begin() + 1, lc_args.end()};
                        if (call->getCallReturnType(*bb->astContext)->isVoidType())
                            fb->call(luisa::compute::Function(lambda_callable.get()), lambda_args);
                        else if (auto lc_type = bb->FindOrAddType(call->getCallReturnType(*bb->astContext), bb->astContext))
                            current = fb->call(lc_type, luisa::compute::Function(lambda_callable.get()), lambda_args);
                        else
                            luisa::log_error("unfound return type in lambda: {}", call->getCallReturnType(*bb->astContext)->getCanonicalTypeInternal().getAsString());
                    } else {
                        calleeDecl->dump();
                        luisa::log_error("unfound function!");
                    }
                }
            } else if (auto _init_expr = llvm::dyn_cast<clang::CXXDefaultInitExpr>(x)) {
                ExprTranslator v(stack, bb, fb, _init_expr->getExpr());
                if (!v.TraverseStmt(_init_expr->getExpr()))
                    luisa::log_error("untranslated member call expr: {}", _init_expr->getExpr()->getStmtClassName());
                current = v.translated;
            } else if (auto _exprWithCleanup = llvm::dyn_cast<clang::ExprWithCleanups>(x)) {// TODO
                luisa::log_warning("unsupportted ExprWithCleanups!");
                current = stack->expr_map[_exprWithCleanup->getSubExpr()];
            } else if (auto _matTemp = llvm::dyn_cast<clang::MaterializeTemporaryExpr>(x)) {// TODO
                luisa::log_warning("unsupportted MaterializeTemporaryExpr!");
                current = stack->expr_map[_matTemp->getSubExpr()];
            } else if (auto _init_list = llvm::dyn_cast<clang::InitListExpr>(x)) {// TODO
                luisa::log_warning("unsupportted InitListExpr!");
            } else if (auto _control_flow = llvm::dyn_cast<clang::IfStmt>(x)) {      // CONTROL FLOW
            } else if (auto _control_flow = llvm::dyn_cast<clang::ContinueStmt>(x)) {// CONTROL FLOW
            } else if (auto _control_flow = llvm::dyn_cast<clang::BreakStmt>(x)) {   // CONTROL FLOW
            } else if (auto _control_flow = llvm::dyn_cast<clang::WhileStmt>(x)) {   // CONTROL FLOW
            } else if (auto _control_flow = llvm::dyn_cast<clang::SwitchStmt>(x)) {  // CONTROL FLOW
            } else if (auto _control_flow = llvm::dyn_cast<clang::CaseStmt>(x)) {    // CONTROL FLOW
            } else if (auto _control_flow = llvm::dyn_cast<clang::DefaultStmt>(x)) { // CONTROL FLOW
            } else if (auto _control_flow = llvm::dyn_cast<clang::ForStmt>(x)) {     // CONTROL FLOW
            } else if (auto cxxLambda = llvm::dyn_cast<LambdaExpr>(x)) {             // LAMBDA TRANSLATED
            } else if (auto null = llvm::dyn_cast<NullStmt>(x)) {                    // EMPTY
            } else if (auto compound = llvm::dyn_cast<CompoundStmt>(x)) {            // EMPTY

            } else {
                x->dump();
                luisa::log_error("unsupportted expr!");
            }
        }

        stack->expr_map[x] = current;
        if (x == root) {
            translated = current;
        }
        return true;
    }

    ExprTranslator(Stack *stack, CXXBlackboard *bb, luisa::shared_ptr<compute::detail::FunctionBuilder> cur, clang::Stmt *root)
        : stack(stack), bb(bb), fb(cur), root(root) {
    }
    const luisa::compute::Expression *translated = nullptr;

protected:
    Stack *stack = nullptr;
    CXXBlackboard *bb = nullptr;
    luisa::shared_ptr<compute::detail::FunctionBuilder> fb = nullptr;
    clang::Stmt *root = nullptr;
};

void FunctionBuilderBuilder::build(const clang::FunctionDecl *S) {
    bool ignore = S->isTemplateDecl();
    bool is_kernel = false;
    uint3 kernelSize;
    bool is_method = false;
    bool is_lambda = false;
    QualType methodThisType;
    auto params = S->parameters();
    for (auto Anno = S->specific_attr_begin<clang::AnnotateAttr>(); Anno != S->specific_attr_end<clang::AnnotateAttr>(); ++Anno) {
        ignore |= isIgnore(*Anno);
        if (isKernel(*Anno)) {
            is_kernel = true;
            getKernelSize(*Anno, kernelSize.x, kernelSize.y, kernelSize.z);
        }
    }
    if (auto Method = llvm::dyn_cast<clang::CXXMethodDecl>(S)) {
        if (auto thisType = GetRecordDeclFromQualType(Method->getThisType()->getPointeeType())) {
            for (auto Anno = thisType->specific_attr_begin<clang::AnnotateAttr>(); Anno != thisType->specific_attr_end<clang::AnnotateAttr>(); ++Anno) {
                ignore |= isBuiltinType(*Anno);
            }
            for (auto f : thisType->fields())
                ignore |= f->getType()->isTemplateTypeParmType();
            if (thisType->isLambda())// ignore global lambda declares, we deal them on stacks only
                is_lambda = true;
        } else {
            // Method->getThisType()->dump();
            luisa::log_error("unfound this type [{}] in method [{}]",
                             Method->getThisType()->getPointeeType().getAsString(), S->getNameAsString());
        }
        is_method = !is_lambda;
        methodThisType = Method->getThisType()->getPointeeType();
    }
    for (auto param : params) {
        ignore |= param->getType()->isTemplateTypeParmType();
        ignore |= (param->getType()->getTypeClass() == clang::Type::PackExpansion);
        ignore |= param->isTemplateParameterPack();
        ignore |= param->isTemplateParameter();
    }
    if (!ignore) {
        // S->dump();

        if (auto Dtor = llvm::dyn_cast<clang::CXXDestructorDecl>(S)) {
            if (!Dtor->isDefaulted()) {
                S->dump();
                auto dtorName = Dtor->getQualifiedNameAsString();
                luisa::log_error("dtor {} is not allowed!", dtorName.c_str());
            }
        }

        luisa::shared_ptr<compute::detail::FunctionBuilder> builder;
        Stmt *body = S->getBody();
        {
            if (is_kernel)
                builder = bb->kernel_builder;
            else
                builder = luisa::make_shared<luisa::compute::detail::FunctionBuilder>(luisa::compute::Function::Tag::CALLABLE);

            if (is_lambda)
                bb->lambda_builders[S] = builder;
            else
                bb->func_builders[S] = builder;

            luisa::compute::detail::FunctionBuilder::push(builder.get());
            builder->push_scope(builder->body());
            {
                if (is_kernel) {
                    builder->set_block_size(kernelSize);
                }

                // comment name
                luisa::string name;
                if (kUseComment) {
                    if (auto Ctor = llvm::dyn_cast<clang::CXXConstructorDecl>(S))
                        name = "[Ctor] ";
                    else if (auto Method = llvm::dyn_cast<clang::CXXMethodDecl>(S))
                        name = "[Method] ";
                    else if (auto Dtor = llvm::dyn_cast<clang::CXXDestructorDecl>(S))
                        name = "[Dtor] ";
                    else
                        name = "[Function] ";
                    name += luisa::string(S->getQualifiedNameAsString());
                    builder->comment_(name);
                }
                // Stack stack;
                // this arg
                if (is_method) {
                    auto Method = llvm::dyn_cast<clang::CXXMethodDecl>(S);
                    if (auto lc_type = bb->FindOrAddType(methodThisType, bb->astContext)) {
                        auto this_local = builder->reference(lc_type);
                        stack.locals[nullptr] = this_local;
                    } else {
                        luisa::log_error("???");
                    }
                }

                // collect args
                for (auto param : params) {
                    auto Ty = param->getType();
                    if (auto lc_type = bb->FindOrAddType(Ty, bb->astContext)) {
                        const luisa::compute::RefExpr *local = nullptr;
                        switch (lc_type->tag()) {
                            case compute::Type::Tag::BUFFER:
                                local = builder->buffer(lc_type);
                                break;
                            case compute::Type::Tag::TEXTURE:
                                local = builder->texture(lc_type);
                                break;
                            case compute::Type::Tag::BINDLESS_ARRAY:
                                local = builder->bindless_array();
                                break;
                            case compute::Type::Tag::ACCEL:
                                local = builder->accel();
                                break;
                            default:
                                local = LC_ArgOrRef(Ty, builder, lc_type);
                                break;
                        }
                        stack.locals[param] = local;
                    }
                }

                // ctor initializers
                if (is_method) {
                    if (auto lc_type = bb->FindOrAddType(methodThisType, bb->astContext)) {
                        auto this_local = stack.locals[nullptr];
                        if (auto Ctor = llvm::dyn_cast<clang::CXXConstructorDecl>(S)) {
                            for (auto ctor_init : Ctor->inits()) {
                                auto init = ctor_init->getInit();
                                ExprTranslator v(&stack, bb, builder, init);
                                if (v.TraverseStmt(init)) {
                                    const auto cxxMember = ctor_init->getMember();
                                    const auto lcMemberType = bb->FindOrAddType(cxxMember->getType(), bb->astContext);
                                    const auto fid = cxxMember->getFieldIndex();
                                    auto mem = builder->member(lcMemberType, this_local, fid);
                                    builder->assign(mem, v.translated);
                                }
                            }
                        }
                    }
                }

                recursiveVisit(body, builder, stack);
            }
            builder->pop_scope(builder->body());
            luisa::compute::detail::FunctionBuilder::pop(builder.get());
        }
    }
}

bool FunctionBuilderBuilder::recursiveVisit(clang::Stmt *currStmt, luisa::shared_ptr<compute::detail::FunctionBuilder> cur, Stack &stack) {
    if (!currStmt)
        return true;

    ExprTranslator v(&stack, bb, cur, currStmt);
    if (!v.TraverseStmt(currStmt))
        luisa::log_error("untranslated member call expr: {}", currStmt->getStmtClassName());

    return true;
}

void RecordDeclStmtHandler::run(const MatchFinder::MatchResult &Result) {
    auto &kernel_builder = bb->kernel_builder;
    if (const auto *S = Result.Nodes.getNodeAs<clang::RecordDecl>("RecordDecl")) {
        QualType Ty = S->getTypeForDecl()->getCanonicalTypeInternal();
        bb->RecordAsStuctureType(Ty);
    }
}

void GlobalVarHandler::run(const MatchFinder::MatchResult &Result) {
    auto &kernel_builder = bb->kernel_builder;
    if (const auto *S = Result.Nodes.getNodeAs<clang::VarDecl>("VarDecl")) {
        bool ignore = false;
        for (auto Anno = S->specific_attr_begin<clang::AnnotateAttr>(); Anno != S->specific_attr_end<clang::AnnotateAttr>(); ++Anno) {
            ignore |= isIgnore(*Anno);
        }
        const auto isGlobal = S->isStaticLocal() || S->isStaticDataMember() || S->isFileVarDecl();
        const auto isConst = S->isConstexpr() || S->getType().isConstQualified();
        const auto isNonConstGlobal = isGlobal && !isConst;
        if (!ignore && isNonConstGlobal) {
            S->dump();
            luisa::log_error("global vars are banned!");
        }
    }
}

void FunctionDeclStmtHandler::run(const MatchFinder::MatchResult &Result) {
    // The matched 'if' statement was bound to 'ifStmt'.
    if (const auto *S = Result.Nodes.getNodeAs<clang::FunctionDecl>("FunctionDecl")) {
        bool isLambda = false;
        if (auto Method = llvm::dyn_cast<clang::CXXMethodDecl>(S)) {
            isLambda = Method->getParent()->isLambda();
        }
        if (!isLambda && !S->isTemplateInstantiation()) {
            auto stack = Stack();
            FunctionBuilderBuilder bdbd(bb, stack);
            bdbd.build(S);
        }
    }
}

ASTConsumer::ASTConsumer(std::string OutputPath, luisa::compute::Device *device, compute::ShaderOption option)
    : OutputPath(std::move(OutputPath)), device(device), option(option) {

    bb.kernel_builder = luisa::make_shared<luisa::compute::detail::FunctionBuilder>(luisa::compute::Function::Tag::KERNEL);

    HandlerForTypeDecl.bb = &bb;
    Matcher.addMatcher(recordDecl(
                           isDefinition(),
                           unless(isExpansionInSystemHeader()))
                           .bind("RecordDecl"),
                       &HandlerForTypeDecl);

    HandlerForFuncionDecl.bb = &bb;
    Matcher.addMatcher(functionDecl(
                           isDefinition(),
                           unless(isExpansionInSystemHeader()))
                           .bind("FunctionDecl"),
                       &HandlerForFuncionDecl);

    HandlerForGlobalVar.bb = &bb;
    Matcher.addMatcher(varDecl(
                           isDefinition(),
                           unless(isExpansionInSystemHeader()))
                           .bind("VarDecl"),
                       &HandlerForGlobalVar);
    // Matcher.addMatcher(stmt().bind("callExpr"), &HandlerForCallExpr);
}

ASTConsumer::~ASTConsumer() {
    device->impl()->create_shader(
        luisa::compute::ShaderOption{
            .compile_only = true,
            .name = "test.bin"},
        luisa::compute::Function{bb.kernel_builder.get()});
}

void ASTConsumer::HandleTranslationUnit(clang::ASTContext &Context) {
    // 1. collect
    bb.astContext = &Context;
    Matcher.matchAST(Context);
}

}// namespace luisa::clangcxx