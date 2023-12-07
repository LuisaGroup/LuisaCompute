#include "ASTConsumer.h"
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

namespace luisa::clangcxx {

using namespace clang;
using namespace clang::ast_matchers;
using namespace luisa::compute;

using CXXBinOp = clang::BinaryOperator::Opcode;
using LCBinOp = luisa::compute::BinaryOp;

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
                            case (BuiltinType::Kind::Long):
                            case (BuiltinType::Kind::Int): { CASE_VEC_TYPE(int) } break;
                            case (BuiltinType::Kind::ULong):
                            case (BuiltinType::Kind::UInt): { CASE_VEC_TYPE(uint) } break;
                            case (BuiltinType::Kind::Double):
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
    } 
    else
    {
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
        for (auto f = S->field_begin(); f != S->field_end(); f++) {
            auto Ty = f->getType();
            if (!tryEmplaceFieldType(Ty, S, types)) {
                S->dump();
                luisa::log_error("unsupported field type [{}] in type [{}]", Ty.getAsString(), S->getNameAsString());
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
        } 
        else if (Ty->isTemplateTypeParmType())
        {
            luisa::log_verbose("template type parameter type...");
            return nullptr;
        }
        else {
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

void RecordDeclStmtHandler::run(const MatchFinder::MatchResult &Result) {
    auto &kernel_builder = blackboard->kernel_builder;
    if (const auto *S = Result.Nodes.getNodeAs<clang::RecordDecl>("RecordDecl")) {
        QualType Ty = S->getTypeForDecl()->getCanonicalTypeInternal();
        blackboard->RecordAsStuctureType(Ty);
    }
}

void GlobalVarHandler::run(const MatchFinder::MatchResult &Result) {
    auto &kernel_builder = blackboard->kernel_builder;
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

struct ExprTranslator : public clang::RecursiveASTVisitor<ExprTranslator> {
    const luisa::compute::Expression *caller = nullptr;

    bool TraverseStmt(clang::Stmt *x) {

        bool last_ret = RecursiveASTVisitor<ExprTranslator>::TraverseStmt(x);

        const luisa::compute::Expression *current = nullptr;
        if (x)
        {
            if (auto il = llvm::dyn_cast<IntegerLiteral>(x)) {
                current = cur->literal(Type::of<int>(), (int)il->getValue().getLimitedValue());
            } else if (auto bin = llvm::dyn_cast<BinaryOperator>(x)) {
                const auto cxx_op = bin->getOpcode();
                const auto lhs = stack->expr_map[bin->getLHS()];
                const auto rhs = stack->expr_map[bin->getRHS()];
                const auto lc_type = blackboard->FindOrAddType(bin->getType(), blackboard->astContext);
                if (auto ca = llvm::dyn_cast<CompoundAssignOperator>(x)) {
                    auto ca_expr = cur->binary(lc_type, TranslateBinaryAssignOp(cxx_op), lhs, rhs);
                    cur->assign(lhs, ca_expr);
                    current = lhs;
                } else if (cxx_op == CXXBinOp::BO_Assign) {
                    cur->assign(lhs, rhs);
                    current = lhs;
                } else {
                    current = cur->binary(lc_type, TranslateBinaryOp(cxx_op), lhs, rhs);
                }
            } else if (auto dref = llvm::dyn_cast<DeclRefExpr>(x)) {
                auto str = luisa::string(dref->getNameInfo().getName().getAsString());
                current = stack->locals[str];
            } else if (auto implicit_cast = llvm::dyn_cast<ImplicitCastExpr>(x)) {
                if (stack->expr_map[last] != nullptr) {
                    const auto lc_type = blackboard->FindOrAddType(implicit_cast->getType(), blackboard->astContext);
                    current = cur->cast(lc_type, CastOp::STATIC, stack->expr_map[last]);
                }
            } else if (auto m = llvm::dyn_cast<clang::MemberExpr>(x)) {// TODO
                if (m->isBoundMemberFunction(*blackboard->astContext)) {
                    auto lhs = stack->expr_map[last];
                    caller = lhs;
                } else if (auto f = llvm::dyn_cast<FieldDecl>(m->getMemberDecl())) {
                    auto lhs = stack->expr_map[m->getBase()];
                    const auto lc_type = blackboard->FindOrAddType(m->getType(), blackboard->astContext);
                    current = cur->member(lc_type, lhs, f->getFieldIndex());
                } else {
                    luisa::log_error("unsupported member expr: {}", m->getMemberDecl()->getNameAsString());
                }
            } else if (auto mcall = llvm::dyn_cast<clang::CXXMemberCallExpr>(x)) {// TODO
                // std::cout << caller->type()->description() << std::endl;
                auto methodDecl = mcall->getMethodDecl();
                llvm::StringRef callopName = {};
                for (auto attr = methodDecl->specific_attr_begin<clang::AnnotateAttr>();
                    attr != methodDecl->specific_attr_end<clang::AnnotateAttr>(); attr++) {
                    if (callopName.empty())
                        callopName = getCallopName(*attr);
                }
                if (!callopName.empty()) {
                    auto op = blackboard->FindCallOp(callopName);
                    luisa::vector<const luisa::compute::Expression *> lc_args;
                    lc_args.emplace_back(caller);// from -MemberExpr::isBoundMemberFunction
                    caller = nullptr;
                    for (auto arg : mcall->arguments()) {
                        if (auto lc_arg = stack->expr_map[arg])
                            lc_args.emplace_back(lc_arg);
                        else
                            luisa::log_error("unfound arg: {}", arg->getStmtClassName());
                    }

                    if (mcall->getCallReturnType(*blackboard->astContext)->isVoidType())
                        cur->call(op, lc_args);
                    else if (auto lc_type = blackboard->FindOrAddType(mcall->getCallReturnType(*blackboard->astContext), blackboard->astContext))
                        current = cur->call(lc_type, op, lc_args);
                    else
                        luisa::log_error("unfound return type: {}", mcall->getCallReturnType(*blackboard->astContext)->getCanonicalTypeInternal().getAsString());
                }
                else {
                    luisa::log_warning("not callop!!!!! {}", luisa::string(methodDecl->getName()));
                }
            } else if (auto lambda = llvm::dyn_cast<LambdaExpr>(x)) {// TODO
                auto cap = lambda->capture_begin();
            }
        }

        last = x;
        stack->expr_map[x] = current;
        if (x == root) {
            translated = current;
        }
        return last_ret;
    }

    ExprTranslator(Stack *stack, CXXBlackboard *blackboard, luisa::shared_ptr<compute::detail::FunctionBuilder> cur, clang::Stmt *root)
        : stack(stack), blackboard(blackboard), cur(cur), root(root) {
    }
    const luisa::compute::Expression *translated = nullptr;

protected:
    Stack *stack = nullptr;
    CXXBlackboard *blackboard = nullptr;
    luisa::shared_ptr<compute::detail::FunctionBuilder> cur = nullptr;
    clang::Stmt *root = nullptr;
private:
    clang::Stmt *last = nullptr;
};

bool FunctionDeclStmtHandler::recursiveVisit(clang::Stmt *stmt, luisa::shared_ptr<compute::detail::FunctionBuilder> cur, Stack &stack) {
    if (!stmt)
        return true;

    for (Stmt::child_iterator i = stmt->child_begin(), e = stmt->child_end(); i != e; ++i) {
        Stmt *currStmt = *i;
        if (!currStmt)
            continue;

        if (auto declStmt = llvm::dyn_cast<clang::DeclStmt>(currStmt)) {
            const DeclGroupRef declGroup = declStmt->getDeclGroup();
            for (auto decl : declGroup) {
                if (!decl) continue;

                if (auto *varDecl = dyn_cast<clang::VarDecl>(decl)) {
                    auto Ty = varDecl->getType();
                    if (auto lc_type = blackboard->FindOrAddType(Ty, blackboard->astContext)) {
                        auto local = cur->local(lc_type);
                        auto str = luisa::string(varDecl->getName());
                        stack.locals[str] = local;

                        auto init = varDecl->getInit();
                        ExprTranslator v(&stack, blackboard, cur, init);
                        if (!v.TraverseStmt(init))
                            luisa::log_error("untranslated init expr: {}", init->getStmtClassName());
                        else if (v.translated)
                            cur->assign(local, v.translated);
                    }
                } else {
                    luisa::log_error("unsupported decl stmt: {}", declStmt->getStmtClassName());
                }
            }
        } else if (auto *mCall = dyn_cast<clang::CXXMemberCallExpr>(currStmt)) {
            auto body = mCall->getExprStmt();
            ExprTranslator v(&stack, blackboard, cur, body);
            if (!v.TraverseStmt(body))
                luisa::log_error("untranslated member call expr: {}", body->getStmtClassName());
        } else if (auto* compound = dyn_cast<clang::CompoundStmt>(currStmt)) {
            recursiveVisit(currStmt, cur, stack);
        }
    }
    return true;
}

void FunctionDeclStmtHandler::run(const MatchFinder::MatchResult &Result) {
    // The matched 'if' statement was bound to 'ifStmt'.
    if (const auto *S = Result.Nodes.getNodeAs<clang::FunctionDecl>("FunctionDecl")) {
        bool ignore = false;
        bool is_kernel = false;
        auto params = S->parameters();
        for (auto Anno = S->specific_attr_begin<clang::AnnotateAttr>(); Anno != S->specific_attr_end<clang::AnnotateAttr>(); ++Anno) {
            ignore |= isIgnore(*Anno);
            is_kernel |= isKernel(*Anno);
        }
        if (auto Method = llvm::dyn_cast<clang::CXXMethodDecl>(S)) {
            if (auto thisType = GetRecordDeclFromQualType(Method->getThisType()->getPointeeType())) {
                for (auto Anno = thisType->specific_attr_begin<clang::AnnotateAttr>(); Anno != thisType->specific_attr_end<clang::AnnotateAttr>(); ++Anno) {
                    ignore |= isBuiltinType(*Anno);
                }
            } else {
                Method->getThisType()->dump();
                luisa::log_error("unfound this type [{}] in method [{}]",
                                 Method->getThisType()->getPointeeType().getAsString(), S->getNameAsString());
            }
        }
        for (auto param : params) {
            ignore |= param->getType()->isTemplateTypeParmType();
        }
        if (!ignore) {
            S->dump();
            luisa::shared_ptr<compute::detail::FunctionBuilder> builder;
            Stmt *body = S->getBody();
            {
                if (is_kernel) {
                    builder = blackboard->kernel_builder;
                } else {
                    builder = luisa::make_shared<luisa::compute::detail::FunctionBuilder>(luisa::compute::Function::Tag::CALLABLE);
                }
                luisa::compute::detail::FunctionBuilder::push(builder.get());
                builder->push_scope(builder->body());
                {
                    if (is_kernel) {
                        builder->set_block_size(uint3(256, 1, 1));
                    }
                    Stack stack;
                    // collect args
                    for (auto param : params) {
                        auto Ty = param->getType();
                        if (auto lc_type = blackboard->FindOrAddType(Ty, blackboard->astContext)) {
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
                                    local = builder->argument(lc_type);
                                    break;
                            }
                            stack.locals[luisa::string(param->getName())] = local;
                        }
                    }
                    recursiveVisit(body, builder, stack);
                }
                builder->pop_scope(builder->body());
                luisa::compute::detail::FunctionBuilder::pop(builder.get());
            }
        }
    }
}

ASTConsumer::ASTConsumer(std::string OutputPath, luisa::compute::Device *device, compute::ShaderOption option)
    : OutputPath(std::move(OutputPath)), device(device), option(option) {

    blackboard.kernel_builder = luisa::make_shared<luisa::compute::detail::FunctionBuilder>(luisa::compute::Function::Tag::KERNEL);

    HandlerForTypeDecl.blackboard = &blackboard;
    Matcher.addMatcher(recordDecl(
                           isDefinition(),
                           unless(isExpansionInSystemHeader()))
                           .bind("RecordDecl"),
                       &HandlerForTypeDecl);

    HandlerForFuncionDecl.blackboard = &blackboard;
    Matcher.addMatcher(functionDecl(
                           isDefinition(),
                           unless(isExpansionInSystemHeader()))
                           .bind("FunctionDecl"),
                       &HandlerForFuncionDecl);

    HandlerForGlobalVar.blackboard = &blackboard;
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
        luisa::compute::Function{blackboard.kernel_builder.get()});
}

void ASTConsumer::HandleTranslationUnit(clang::ASTContext &Context) {
    // 1. collect
    blackboard.astContext = &Context;
    Matcher.matchAST(Context);
}

}// namespace luisa::clangcxx