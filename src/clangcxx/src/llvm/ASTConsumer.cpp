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
        } else if (builtin_type_name == "matrix") {
            if (auto TST = Ty->getAs<TemplateSpecializationType>()) {
                auto Arguments = TST->template_arguments();
                clang::Expr::EvalResult Result;
                if (Arguments[0].getAsExpr()->EvaluateAsConstantExpr(Result, *astContext)) {
                    auto N = Result.Val.getInt().getExtValue();
                    _type = Type::matrix(N);
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

    luisa::vector<const clang::IfStmt *> activeCXXBranches = {};
    luisa::vector<luisa::compute::IfStmt *> activeLuisaBranches = {};

    bool TraverseStmt(clang::Stmt *x) {
        if (x == nullptr)
            return true;
        // HANDLE SCOPE
        if (!activeCXXBranches.empty()) {
            auto cxxBranch = activeCXXBranches.back();
            auto lcBranch = activeLuisaBranches.back();
            if (x == cxxBranch->getThen()) { func_builder->push_scope(lcBranch->true_branch()); }
            if (x == cxxBranch->getElse()) { func_builder->push_scope(lcBranch->false_branch()); }
        }
        SKR_DEFER({
            if (!activeCXXBranches.empty()) {
                auto cxxBranch = activeCXXBranches.back();
                auto lcBranch = activeLuisaBranches.back();
                if (x == cxxBranch->getThen()) { func_builder->pop_scope(lcBranch->true_branch()); }
                if (x == cxxBranch->getElse()) { func_builder->pop_scope(lcBranch->false_branch()); }
            }
        });

        // CONTROL WALK
        if (auto cxxBranch = llvm::dyn_cast<clang::IfStmt>(x)) {
            auto cxxCond = cxxBranch->getCond();
            TraverseStmt(cxxCond);
            activeCXXBranches.emplace_back(cxxBranch);
            if (!stack->expr_map[cxxCond]) {
                cxxCond->dump();
                luisa::log_error("missing branch cond!");
            }
            activeLuisaBranches.emplace_back(func_builder->if_(stack->expr_map[cxxCond]));
            if (cxxBranch->getThen())
                TraverseStmt(cxxBranch->getThen());
            if (cxxBranch->getElse())
                TraverseStmt(cxxBranch->getElse());
            activeCXXBranches.pop_back();
            activeLuisaBranches.pop_back();
        } else {
            RecursiveASTVisitor<ExprTranslator>::TraverseStmt(x);
        }

        // TRANSLATE
        const luisa::compute::Expression *current = nullptr;
        if (x) {
            if (auto ret = llvm::dyn_cast<clang::ReturnStmt>(x)) {
                auto cxx_ret = ret->getRetValue();
                auto lc_ret = stack->expr_map[cxx_ret];
                func_builder->return_(lc_ret);
            } else if (auto ce = llvm::dyn_cast<clang::ConstantExpr>(x)) {
                const auto APK = ce->getResultAPValueKind();
                const auto &APV = ce->getAPValueResult();
                switch (APK) {
                    case clang::APValue::ValueKind::Int:
                        current = func_builder->literal(Type::of<int>(), (int)ce->getResultAsAPSInt().getLimitedValue());
                        break;
                    case clang::APValue::ValueKind::Float:
                        current = func_builder->literal(Type::of<float>(), (float)APV.getFloat().convertToFloat());
                        break;
                    default:
                        luisa::log_error("unsupportted ConstantExpr APValueKind {}", APK);
                        break;
                }
            } else if (auto il = llvm::dyn_cast<IntegerLiteral>(x)) {
                current = func_builder->literal(Type::of<int>(), (int)il->getValue().getLimitedValue());
            } else if (auto fl = llvm::dyn_cast<FloatingLiteral>(x)) {
                current = func_builder->literal(Type::of<float>(), (float)fl->getValue().convertToFloat());
            } else if (auto construct = llvm::dyn_cast<CXXConstructExpr>(x)) {// TODO
                auto local = func_builder->local(blackboard->FindOrAddType(construct->getType(), blackboard->astContext));
                // args
                luisa::vector<const luisa::compute::Expression *> lc_args;
                lc_args.emplace_back(local);
                for (auto arg : construct->arguments()) {
                    if (auto lc_arg = stack->expr_map[arg])
                        lc_args.emplace_back(lc_arg);
                    else
                        luisa::log_error("unfound arg: {}", arg->getStmtClassName());
                }
                auto callable = blackboard->builders[construct->getConstructor()];
                func_builder->call(luisa::compute::Function(callable.get()), lc_args);
                current = local;
            } else if (auto bin = llvm::dyn_cast<BinaryOperator>(x)) {
                const auto cxx_op = bin->getOpcode();
                const auto lhs = stack->expr_map[bin->getLHS()];
                const auto rhs = stack->expr_map[bin->getRHS()];
                const auto lc_type = blackboard->FindOrAddType(bin->getType(), blackboard->astContext);
                if (auto ca = llvm::dyn_cast<CompoundAssignOperator>(x)) {
                    auto ca_expr = func_builder->binary(lc_type, TranslateBinaryAssignOp(cxx_op), lhs, rhs);
                    func_builder->assign(lhs, ca_expr);
                    current = lhs;
                } else if (cxx_op == CXXBinOp::BO_Assign) {
                    func_builder->assign(lhs, rhs);
                    current = lhs;
                } else {
                    current = func_builder->binary(lc_type, TranslateBinaryOp(cxx_op), lhs, rhs);
                }
            } else if (auto dref = llvm::dyn_cast<DeclRefExpr>(x)) {
                auto str = luisa::string(dref->getNameInfo().getName().getAsString());
                if (stack->locals[str]) {
                    current = stack->locals[str];
                } else if (auto value = dref->getDecl(); value && llvm::isa<clang::VarDecl>(value))// Value Ref
                {
                    if (auto var = value->getPotentiallyDecomposedVarDecl()) {
                        if (auto eval = var->getEvaluatedValue()) {
                            if (eval->isInt())
                                current = func_builder->literal(Type::of<int>(), (int)eval->getInt().getLimitedValue());
                            else if (eval->isFloat())
                                current = func_builder->literal(Type::of<float>(), (float)eval->getFloat().convertToFloat());
                            else
                                luisa::log_error("unsupportted eval type: {}", eval->getKind());
                        } else
                            luisa::log_error("unfound & unresolved ref: {}", str);
                    }
                } else if (auto value = dref->getDecl(); value && llvm::isa<clang::FunctionDecl>(value))// Func Ref
                    ;
                else
                    luisa::log_error("unfound var ref: {}", str);
            } else if (auto implicit_cast = llvm::dyn_cast<ImplicitCastExpr>(x)) {
                if (stack->expr_map[implicit_cast->getSubExpr()] != nullptr) {
                    const auto lc_type = blackboard->FindOrAddType(implicit_cast->getType(), blackboard->astContext);
                    if (!stack->expr_map[implicit_cast->getSubExpr()]) {
                        implicit_cast->getSubExpr()->dump();
                        luisa::log_error("!!!");
                    }
                    current = func_builder->cast(lc_type, CastOp::STATIC, stack->expr_map[implicit_cast->getSubExpr()]);
                }
            } else if (auto _static_cast = llvm::dyn_cast<CXXStaticCastExpr>(x)) {
                if (stack->expr_map[_static_cast->getSubExpr()] != nullptr) {
                    const auto lc_type = blackboard->FindOrAddType(_static_cast->getType(), blackboard->astContext);
                    if (!stack->expr_map[_static_cast->getSubExpr()]) {
                        _static_cast->getSubExpr()->dump();
                        luisa::log_error("!!!");
                    }
                    current = func_builder->cast(lc_type, CastOp::STATIC, stack->expr_map[_static_cast->getSubExpr()]);
                }
            } else if (auto t = llvm::dyn_cast<clang::CXXThisExpr>(x)) {
                current = stack->locals["this"];
            } else if (auto m = llvm::dyn_cast<clang::MemberExpr>(x)) {
                if (m->isBoundMemberFunction(*blackboard->astContext)) {
                    auto lhs = stack->expr_map[m->getBase()];
                    caller = lhs;
                } else if (auto f = llvm::dyn_cast<FieldDecl>(m->getMemberDecl())) {
                    auto lhs = stack->expr_map[m->getBase()];
                    assert(lhs && "missing this!");
                    const auto lc_type = blackboard->FindOrAddType(m->getType(), blackboard->astContext);
                    current = func_builder->member(lc_type, lhs, f->getFieldIndex());
                } else {
                    luisa::log_error("unsupported member expr: {}", m->getMemberDecl()->getNameAsString());
                }
            } else if (auto call = llvm::dyn_cast<clang::CallExpr>(x)) {
                auto methodDecl = call->getCalleeDecl();
                llvm::StringRef callopName = {};
                for (auto attr = methodDecl->specific_attr_begin<clang::AnnotateAttr>();
                     attr != methodDecl->specific_attr_end<clang::AnnotateAttr>(); attr++) {
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
                    auto op = blackboard->FindCallOp(callopName);
                    if (call->getCallReturnType(*blackboard->astContext)->isVoidType())
                        func_builder->call(op, lc_args);
                    else if (auto lc_type = blackboard->FindOrAddType(call->getCallReturnType(*blackboard->astContext), blackboard->astContext))
                        current = func_builder->call(lc_type, op, lc_args);
                    else
                        luisa::log_error("unfound return type: {}", call->getCallReturnType(*blackboard->astContext)->getCanonicalTypeInternal().getAsString());
                } else {
                    auto callable = blackboard->builders[call->getCalleeDecl()];
                    if (call->getCallReturnType(*blackboard->astContext)->isVoidType())
                        func_builder->call(luisa::compute::Function(callable.get()), lc_args);
                    else if (auto lc_type = blackboard->FindOrAddType(call->getCallReturnType(*blackboard->astContext), blackboard->astContext))
                        current = func_builder->call(lc_type, luisa::compute::Function(callable.get()), lc_args);
                    else
                        luisa::log_error("unfound return type: {}", call->getCallReturnType(*blackboard->astContext)->getCanonicalTypeInternal().getAsString());
                }
            } else if (auto _init = llvm::dyn_cast<clang::InitListExpr>(x)) {// TODO
                luisa::log_warning("unsupportted init list expr!");

            } else if (auto _if = llvm::dyn_cast<clang::IfStmt>(x)) {    // FWD
            } else if (auto null = llvm::dyn_cast<NullStmt>(x)) {        // EMPTY
            } else if (auto compound = llvm::dyn_cast<CompoundStmt>(x)) {// EMPTY
            } else if (auto lambda = llvm::dyn_cast<LambdaExpr>(x)) {    // TODO
                auto cap = lambda->capture_begin();
                luisa::log_error("unsupportted lambda expr!");
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

    ExprTranslator(Stack *stack, CXXBlackboard *blackboard, luisa::shared_ptr<compute::detail::FunctionBuilder> cur, clang::Stmt *root)
        : stack(stack), blackboard(blackboard), func_builder(cur), root(root) {
    }
    const luisa::compute::Expression *translated = nullptr;

protected:
    Stack *stack = nullptr;
    CXXBlackboard *blackboard = nullptr;
    luisa::shared_ptr<compute::detail::FunctionBuilder> func_builder = nullptr;
    clang::Stmt *root = nullptr;
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
        } else if (auto *compound = dyn_cast<clang::CompoundStmt>(currStmt)) {
            recursiveVisit(currStmt, cur, stack);
        } else if (auto *ret = dyn_cast<clang::ReturnStmt>(currStmt)) {
            auto init = ret->getRetValue();
            ExprTranslator v(&stack, blackboard, cur, init);
            if (!v.TraverseStmt(init))
                luisa::log_error("untranslated return expr: {}", init->getStmtClassName());
            else if (cur->tag() != luisa::compute::Function::Tag::KERNEL) {
                if (v.translated)
                    cur->return_(v.translated);
                else
                    luisa::log_error("untranslated return expr: {}", init->getStmtClassName());
            }
        } else {
            ExprTranslator v(&stack, blackboard, cur, currStmt);
            if (!v.TraverseStmt(currStmt))
                luisa::log_error("untranslated member call expr: {}", currStmt->getStmtClassName());
        }
    }
    return true;
}

void FunctionDeclStmtHandler::run(const MatchFinder::MatchResult &Result) {
    // The matched 'if' statement was bound to 'ifStmt'.
    if (const auto *S = Result.Nodes.getNodeAs<clang::FunctionDecl>("FunctionDecl")) {
        bool ignore = false;
        bool is_kernel = false;
        bool is_method = false;
        QualType methodThisType;
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
                for (auto f : thisType->fields())
                    ignore |= f->getType()->isTemplateTypeParmType();
            } else {
                // Method->getThisType()->dump();
                luisa::log_error("unfound this type [{}] in method [{}]",
                                 Method->getThisType()->getPointeeType().getAsString(), S->getNameAsString());
            }
            is_method = true;
            methodThisType = Method->getThisType()->getPointeeType();
        }
        for (auto param : params) {
            ignore |= param->getType()->isTemplateTypeParmType();
        }
        if (!ignore) {
            // S->dump();
            luisa::shared_ptr<compute::detail::FunctionBuilder> builder;
            Stmt *body = S->getBody();
            {
                if (is_kernel) {
                    builder = blackboard->kernel_builder;
                } else {
                    builder = luisa::make_shared<luisa::compute::detail::FunctionBuilder>(luisa::compute::Function::Tag::CALLABLE);
                }
                blackboard->builders[S] = builder;
                luisa::compute::detail::FunctionBuilder::push(builder.get());
                builder->push_scope(builder->body());
                {
                    if (is_kernel) {
                        builder->set_block_size(uint3(256, 1, 1));
                    }

                    // comment name
                    if (true) {
                        luisa::string name = luisa::string(S->getName());
                        if (auto Ctor = llvm::dyn_cast<clang::CXXConstructorDecl>(S)) {
                            {
                                name += "Ctor: ";
                                name += Ctor->getQualifiedNameAsString();
                            }
                        }
                        builder->comment_(name);
                    }
                    Stack stack;
                    // this arg
                    if (is_method) {
                        auto Method = llvm::dyn_cast<clang::CXXMethodDecl>(S);
                        if (auto lc_type = blackboard->FindOrAddType(methodThisType, blackboard->astContext)) {
                            auto this_local = builder->reference(lc_type);
                            stack.locals["this"] = this_local;
                        } else {
                            luisa::log_error("???");
                        }
                    }

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

                    // ctor initializers
                    if (is_method) {
                        if (auto lc_type = blackboard->FindOrAddType(methodThisType, blackboard->astContext)) {
                            auto this_local = stack.locals["this"];
                            if (auto Ctor = llvm::dyn_cast<clang::CXXConstructorDecl>(S)) {
                                for (auto ctor_init : Ctor->inits()) {
                                    auto init = ctor_init->getInit();
                                    ExprTranslator v(&stack, blackboard, builder, init);
                                    if (v.TraverseStmt(init)) {
                                        const auto fid = ctor_init->getMember()->getFieldIndex();
                                        auto mem = builder->member(lc_type, this_local, fid);
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