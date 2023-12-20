#include "Utils/Defer.hpp"
#include "Utils/AttributeHelper.hpp"
#include "ASTConsumer.h"

#include <clang/AST/Stmt.h>
#include <clang/AST/Expr.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/AST/DeclTemplate.h>

#include <luisa/vstl/common.h>
#include <luisa/core/magic_enum.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/ast/op.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/syntax.h>

namespace luisa::clangcxx {

using namespace clang;
using namespace clang::ast_matchers;
using namespace luisa::compute;

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

inline const luisa::compute::RefExpr *LC_ArgOrRef(clang::QualType qt, luisa::shared_ptr<compute::detail::FunctionBuilder> fb, const luisa::compute::Type *lc_type) {
    if (qt->isPointerType())
        luisa::log_error("pointer type is not supported: [{}]", qt.getAsString());
    else if (qt->isReferenceType())
        return fb->reference(lc_type);
    else
        return fb->argument(lc_type);
    return nullptr;
}

struct ExprTranslator : public clang::RecursiveASTVisitor<ExprTranslator> {
    clang::ForStmt *currentCxxForStmt = nullptr;

    bool TraverseStmt(clang::Stmt *x) {
        if (x == nullptr) return true;

        // CONTROL WALK
        if (auto cxxLambda = llvm::dyn_cast<LambdaExpr>(x)) {
            auto cxxCallee = cxxLambda->getLambdaClass()->getLambdaCallOperator();
            Stack newStack = *stack;
            FunctionBuilderBuilder bdbd(db, newStack);
            bdbd.build(cxxCallee);
        } else if (auto cxxBranch = llvm::dyn_cast<clang::IfStmt>(x)) {
            auto _ = db->CommentStmt(fb, cxxBranch);

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
            auto _ = db->CommentStmt(fb, cxxSwitch);

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
            auto _ = db->CommentStmt(fb, cxxWhile);

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
            auto _ = db->CommentStmt(fb, cxxFor);

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
            auto _ = db->CommentStmt(fb, cxxCompound);

            for (auto sub : cxxCompound->body())
                TraverseStmt(sub);
        } else {
            RecursiveASTVisitor<ExprTranslator>::TraverseStmt(x);
        }

        // TRANSLATE
        auto astContext = db->GetASTContext();
        const luisa::compute::Expression *current = nullptr;
        if (x) {
            if (auto cxxLambda = llvm::dyn_cast<LambdaExpr>(x)) {
                auto cxxCallee = cxxLambda->getLambdaClass()->getLambdaCallOperator();
                auto methodThisType = cxxCallee->getThisType()->getPointeeType();
                current = fb->local(db->FindOrAddType(methodThisType));
            } else if (auto cxxDecl = llvm::dyn_cast<clang::DeclStmt>(x)) {
                auto _ = db->CommentStmt(fb, cxxDecl);

                const DeclGroupRef declGroup = cxxDecl->getDeclGroup();
                for (auto decl : declGroup) {
                    if (!decl) continue;

                    if (auto *varDecl = dyn_cast<clang::VarDecl>(decl)) {
                        auto Ty = varDecl->getType();
                        if (isSwizzle(varDecl))
                            luisa::log_error("can not use auto type to deduct swizzles!");

                        if (auto lc_type = db->FindOrAddType(Ty)) {
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
                    } else if (auto aliasDecl = dyn_cast<clang::TypeAliasDecl>(decl)) {// ignore
                    } else {
                        x->dump();
                        luisa::log_error("unsupported decl stmt: {}", cxxDecl->getStmtClassName());
                    }
                }
            } else if (auto cxxRet = llvm::dyn_cast<clang::ReturnStmt>(x)) {
                auto _ = db->CommentStmt(fb, cxxRet);

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
                        current = fb->literal(Type::of<float>(), (float)APV.getFloat().convertToDouble());
                        break;
                    default:
                        luisa::log_error("unsupportted ConstantExpr APValueKind {}", APK);
                        break;
                }
            } else if (auto substNonType = llvm::dyn_cast<SubstNonTypeTemplateParmExpr>(x)) {
                // auto lcType = db->FindOrAddType(substNonType->getType());
                auto lcExpr = stack->expr_map[substNonType->getReplacement()];
                current = lcExpr;
            } else if (auto il = llvm::dyn_cast<IntegerLiteral>(x)) {
                auto limitedVal = il->getValue().getLimitedValue();
                if (limitedVal <= UINT32_MAX)
                    current = fb->literal(Type::of<uint>(), (uint)il->getValue().getLimitedValue());
                else
                    current = fb->literal(Type::of<uint64>(), il->getValue().getLimitedValue());
            } else if (auto bl = llvm::dyn_cast<CXXBoolLiteralExpr>(x)) {
                current = fb->literal(Type::of<bool>(), (bool)bl->getValue());
            } else if (auto fl = llvm::dyn_cast<FloatingLiteral>(x)) {
                current = fb->literal(Type::of<float>(), (float)fl->getValue().convertToDouble());
            } else if (auto cxxCtorCall = llvm::dyn_cast<CXXConstructExpr>(x)) {
                auto _ = db->CommentStmt(fb, cxxCtorCall);

                // TODO: REFACTOR THIS
                auto cxxCtor = cxxCtorCall->getConstructor();
                if (!db->func_builders.contains(cxxCtor)) {
                    auto funcDecl = cxxCtor->getAsFunction();
                    auto methodDecl = llvm::dyn_cast<clang::CXXMethodDecl>(funcDecl);
                    const auto isTemplateInstant = funcDecl->isTemplateInstantiation();
                    if (isTemplateInstant) {
                        FunctionBuilderBuilder fbfb(db, *stack);
                        fbfb.build(funcDecl);
                    }
                }

                auto local = fb->local(db->FindOrAddType(cxxCtorCall->getType()));
                // args
                luisa::vector<const luisa::compute::Expression *> lc_args;
                lc_args.emplace_back(local);
                for (auto arg : cxxCtorCall->arguments()) {
                    if (auto lc_arg = stack->expr_map[arg])
                        lc_args.emplace_back(lc_arg);
                    else
                        luisa::log_error("unfound arg: {}", arg->getStmtClassName());
                }
                if (auto callable = db->func_builders[cxxCtor]) {
                    fb->call(luisa::compute::Function(callable.get()), lc_args);
                    current = local;
                } else if (cxxCtor->getParent()->isLambda()) {
                    // ...IGNORE LAMBDA CTOR...
                    current = local;
                } else {
                    bool isBuiltin = false;
                    llvm::StringRef builtinName = {};
                    if (auto thisType = GetRecordDeclFromQualType(cxxCtorCall->getType())) {
                        for (auto Anno = thisType->specific_attr_begin<clang::AnnotateAttr>(); Anno != thisType->specific_attr_end<clang::AnnotateAttr>(); ++Anno) {
                            if (isBuiltinType(*Anno)) {
                                isBuiltin = true;
                                builtinName = getBuiltinTypeName(*Anno);
                                break;
                            }
                        }
                    }
                    if (isBuiltin) {
                        auto Ty = cxxCtorCall->getType().getDesugaredType(*astContext);
                        if (builtinName == "vec") {
                            if (auto TSD = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(Ty->getAs<clang::RecordType>()->getDecl())) {
                                auto &Arguments = TSD->getTemplateArgs();
                                if (auto EType = Arguments[0].getAsType()->getAs<clang::BuiltinType>()) {
                                    auto N = Arguments[1].getAsIntegral().getLimitedValue();
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
                            } else
                                luisa::log_error("???");
                        } else if (builtinName == "matrix") {
                            if (auto TSD = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(Ty->getAs<clang::RecordType>()->getDecl())) {
                                auto &Arguments = TSD->getTemplateArgs();
                                clang::Expr::EvalResult Result;
                                auto N = Arguments[0].getAsIntegral().getLimitedValue();
                                auto lc_type = Type::matrix(N);
                                const CallOp MATRIX_LUT[3] = {CallOp::MAKE_FLOAT2X2, CallOp::MAKE_FLOAT3X3, CallOp::MAKE_FLOAT4X4};
                                current = fb->call(lc_type, MATRIX_LUT[N - 2], {lc_args.begin() + 1, lc_args.end()});
                            } else
                                luisa::log_error("???");
                        } else if (builtinName == "array") {
                            if (auto TSD = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(Ty->getAs<clang::RecordType>()->getDecl())) {
                                auto &Arguments = TSD->getTemplateArgs();
                                auto EType = Arguments[0].getAsType();
                                auto N = Arguments[1].getAsIntegral().getLimitedValue();
                                auto lcElemType = db->FindOrAddType(EType);
                                auto lcArrayType = Type::array(lcElemType, N);
                                if (cxxCtor->isDefaultConstructor())
                                    current = fb->local(lcArrayType);
                                else if (cxxCtor->isConvertingConstructor(true))
                                    current = fb->cast(lcArrayType, CastOp::STATIC, lc_args[1]);
                                else if (cxxCtor->isCopyOrMoveConstructor())
                                {
                                    fb->assign(lc_args[0], lc_args[1]);
                                    current = lc_args[0];
                                }
                                else
                                    luisa::log_error("unhandled array constructor: {}", cxxCtor->getNameAsString());
                            }
                        } else {
                            luisa::log_error("unhandled builtin constructor: {}", cxxCtor->getNameAsString());
                        }
                    } else {
                        cxxCtor->dump();
                        luisa::log_error("unfound constructor: {}", cxxCtor->getNameAsString());
                    }
                }
            } else if (auto unary = llvm::dyn_cast<UnaryOperator>(x)) {
                const auto cxx_op = unary->getOpcode();
                const auto lhs = stack->expr_map[unary->getSubExpr()];
                const auto lc_type = db->FindOrAddType(unary->getType());
                if (!IsUnaryAssignOp(cxx_op)) {
                    current = fb->unary(lc_type, TranslateUnaryOp(cxx_op), lhs);
                } else {
                    auto one = fb->literal(Type::of<int>(), 1);
                    auto typed_one = one;
                    switch (cxx_op) {
                        case clang::UO_PreInc:
                        case clang::UO_PreDec: {
                            auto lc_binop = (cxx_op == clang::UO_PreInc) ? LCBinOp::ADD : LCBinOp::SUB;
                            auto ca_expr = fb->binary(lc_type, lc_binop, lhs, typed_one);
                            fb->assign(lhs, ca_expr);
                            current = lhs;
                            break;
                        }
                        case clang::UO_PostInc:
                        case clang::UO_PostDec: {
                            auto lc_binop = (cxx_op == clang::UO_PostInc) ? LCBinOp::ADD : LCBinOp::SUB;
                            auto old = fb->local(lc_type);
                            fb->assign(old, lhs);
                            auto ca_expr = fb->binary(lc_type, lc_binop, lhs, typed_one);
                            fb->assign(lhs, ca_expr);
                            current = old;
                            break;
                        }
                    }
                }
            } else if (auto bin = llvm::dyn_cast<BinaryOperator>(x)) {
                auto _ = db->CommentStmt(fb, bin);

                const auto cxx_op = bin->getOpcode();
                const auto lhs = stack->expr_map[bin->getLHS()];
                const auto rhs = stack->expr_map[bin->getRHS()];
                const auto lc_type = db->FindOrAddType(bin->getType());
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
                                current = fb->literal(Type::of<float>(), (float)eval->getFloat().convertToDouble());
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
            } else if (auto _cxxParen = llvm::dyn_cast<ParenExpr>(x)) {
                current = stack->expr_map[_cxxParen->getSubExpr()];
            } else if (auto implicit_cast = llvm::dyn_cast<ImplicitCastExpr>(x)) {
                if (stack->expr_map[implicit_cast->getSubExpr()] != nullptr) {
                    const auto lc_type = db->FindOrAddType(implicit_cast->getType());
                    if (!stack->expr_map[implicit_cast->getSubExpr()]) {
                        implicit_cast->getSubExpr()->dump();
                        luisa::log_error("!!!");
                    }
                    current = fb->cast(lc_type, CastOp::STATIC, stack->expr_map[implicit_cast->getSubExpr()]);
                }
            } else if (auto _explicit_cast = llvm::dyn_cast<ExplicitCastExpr>(x)) {
                if (stack->expr_map[_explicit_cast->getSubExpr()] != nullptr) {
                    const auto lc_type = db->FindOrAddType(_explicit_cast->getType());
                    if (!stack->expr_map[_explicit_cast->getSubExpr()]) {
                        _explicit_cast->getSubExpr()->dump();
                        luisa::log_error("!!!");
                    }
                    current = fb->cast(lc_type, CastOp::STATIC, stack->expr_map[_explicit_cast->getSubExpr()]);
                }
            } else if (auto cxxDefaultArg = llvm::dyn_cast<clang::CXXDefaultArgExpr>(x)) {
                auto _ = db->CommentStmt(fb, cxxDefaultArg);

                const auto _value = fb->local(db->FindOrAddType(cxxDefaultArg->getType()));
                TraverseStmt(cxxDefaultArg->getExpr());
                fb->assign(_value, stack->expr_map[cxxDefaultArg->getExpr()]);
                current = _value;
            } else if (auto t = llvm::dyn_cast<clang::CXXThisExpr>(x)) {
                current = stack->locals[nullptr];
            } else if (auto cxxMember = llvm::dyn_cast<clang::MemberExpr>(x)) {
                if (auto bypass = isByPass(cxxMember->getMemberDecl())) {
                    current = stack->expr_map[cxxMember->getBase()];
                    if (cxxMember->isBoundMemberFunction(*astContext))
                        stack->callers.emplace_back(current);
                } else if (cxxMember->isBoundMemberFunction(*astContext)) {
                    auto lhs = stack->expr_map[cxxMember->getBase()];
                    stack->callers.emplace_back(lhs);
                } else if (auto cxxField = llvm::dyn_cast<FieldDecl>(cxxMember->getMemberDecl())) {
                    if (isSwizzle(cxxField)) {
                        auto swizzleText = cxxField->getName();
                        const auto swizzleType = cxxField->getType().getDesugaredType(*astContext);
                        if (auto TSD = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(swizzleType->getAs<clang::RecordType>()->getDecl())) {
                            const auto cxxResultType = TSD->getTemplateArgs().get(2).getAsType();
                            if (auto lcResultType = db->FindOrAddType(cxxResultType)) {
                                uint64_t swizzle_code = 0u;
                                uint64_t swizzle_seq[] = {0u, 0u, 0u, 0u}; /*4*/
                                int64_t swizzle_size = 0;
                                for (auto iter = swizzleText.begin(); iter != swizzleText.end(); iter++) {
                                    if (*iter == 'x') swizzle_seq[swizzle_size] = 0u;
                                    if (*iter == 'y') swizzle_seq[swizzle_size] = 1u;
                                    if (*iter == 'z') swizzle_seq[swizzle_size] = 2u;
                                    if (*iter == 'w') swizzle_seq[swizzle_size] = 3u;
                                    swizzle_size += 1;
                                }
                                // encode swizzle code
                                for (int64_t cursor = swizzle_size - 1; cursor >= 0; cursor--) {
                                    swizzle_code <<= 4;
                                    swizzle_code |= swizzle_seq[cursor];
                                }
                                auto lhs = stack->expr_map[cxxMember->getBase()];
                                current = fb->swizzle(lcResultType, lhs, swizzle_size, swizzle_code);
                            }
                        } else
                            luisa::log_error("!!!");
                    } else {
                        auto lhs = stack->expr_map[cxxMember->getBase()];
                        const auto lcMemberType = db->FindOrAddType(cxxField->getType());
                        current = fb->member(lcMemberType, lhs, cxxField->getFieldIndex());
                    }
                } else {
                    luisa::log_error("unsupported member expr: {}", cxxMember->getMemberDecl()->getNameAsString());
                }
            } else if (auto call = llvm::dyn_cast<clang::CallExpr>(x)) {
                if (isByPass(call->getCalleeDecl())) {
                    auto caller = stack->callers.back();
                    stack->callers.pop_back();
                    if (!caller) {
                        call->dump();
                        luisa::log_error("incorrect [[bypass]] call detected!");
                    }
                    current = caller;
                } else {
                    auto _ = db->CommentStmt(fb, call);
                    auto calleeDecl = call->getCalleeDecl();
                    llvm::StringRef callopName = {};
                    llvm::StringRef binopName = {};
                    bool isAccess = false;
                    for (auto attr : calleeDecl->specific_attrs<clang::AnnotateAttr>()){
                        if (callopName.empty())
                            callopName = getCallopName(attr);
                        if (binopName.empty())
                            binopName = getBinopName(attr);
                        isAccess |= luisa::clangcxx::isAccess(attr);
                    }
                    // args
                    luisa::vector<const luisa::compute::Expression *> lc_args;
                    if (auto mcall = llvm::dyn_cast<clang::CXXMemberCallExpr>(x)) {
                        auto caller = stack->callers.back();
                        stack->callers.pop_back();

                        lc_args.emplace_back(caller);// from -MemberExpr::isBoundMemberFunction
                    }
                    for (auto arg : call->arguments()) {
                        if (auto lc_arg = stack->expr_map[arg])
                            lc_args.emplace_back(lc_arg);
                        else {
                            arg->dump();
                            luisa::log_error("unfound arg: {}", arg->getStmtClassName());
                        }
                    }
                    // call
                    if (!binopName.empty()) {
                        auto lc_binop = db->FindBinOp(binopName);
                        if (auto lcReturnType = db->FindOrAddType(call->getCallReturnType(*astContext)))
                            current = fb->binary(lcReturnType, lc_binop, lc_args[0], lc_args[1]);
                    } else if (isAccess) {
                        if (auto lcReturnType = db->FindOrAddType(call->getCallReturnType(*astContext))) {
                            current = fb->access(lcReturnType, lc_args[0], lc_args[1]);
                        }
                    } else if (!callopName.empty()) {
                        auto op_or_builtin = db->FindCallOp(callopName);
                        switch (op_or_builtin.index()) {
                            case 0: {
                                CallOp op = luisa::get<0>(op_or_builtin);

                                const bool isHit = (op == CallOp::RAY_QUERY_COMMITTED_HIT);

                                const bool isQueryAny = (op == CallOp::RAY_TRACING_QUERY_ANY);
                                const bool isQueryAll = (op == CallOp::RAY_TRACING_QUERY_ALL);
                                const bool isQuery = isQueryAny || isQueryAll;

                                if (isHit) {
                                    auto _stmt = stack->queries.back();
                                    stack->queries.pop_back();
                                    auto tvar = def<CommittedHit>(fb->call(Type::of<CommittedHit>(), CallOp::RAY_QUERY_COMMITTED_HIT, {_stmt->query()}));
                                    current = tvar.expression();
                                } else if (isQuery) {
                                    const luisa::compute::Type *rq_type = nullptr;
                                    if (isQueryAny)
                                        rq_type = Type::of<luisa::compute::detail::RayQueryProxy<true>>();
                                    else if (isQueryAll)
                                        rq_type = Type::of<luisa::compute::detail::RayQueryProxy<false>>();

                                    auto local = fb->local(rq_type);
                                    auto rq_stmt = fb->call(rq_type, op, lc_args);
                                    fb->assign(local, rq_stmt);
                                    current = local;

                                    auto ctrl_stmt = fb->ray_query_(local);
                                    stack->queries.emplace_back(ctrl_stmt);
                                } else {
                                    if (call->getCallReturnType(*astContext)->isVoidType())
                                        fb->call(op, lc_args);
                                    else if (auto lcReturnType = db->FindOrAddType(call->getCallReturnType(*astContext))) {
                                        auto ret_value = fb->local(lcReturnType);
                                        fb->assign(ret_value, fb->call(lcReturnType, op, lc_args));
                                        current = ret_value;
                                    } else
                                        luisa::log_error(
                                            "unfound return type: {}",
                                            call->getCallReturnType(*astContext)->getCanonicalTypeInternal().getAsString());
                                }
                            } break;
                            case 1: {
                                auto builtin_func = luisa::get<1>(op_or_builtin);
                                current = builtin_func(fb.get());
                            } break;
                        }
                    } else {
                        auto calleeDecl = call->getCalleeDecl();
                        // TODO: REFACTOR THIS
                        auto funcDecl = calleeDecl->getAsFunction();
                        auto methodDecl = llvm::dyn_cast<clang::CXXMethodDecl>(funcDecl);
                        const auto isTemplateInstant = funcDecl->isTemplateInstantiation();
                        const auto isLambda = methodDecl && methodDecl->getParent()->isLambda();
                        if (!db->lambda_builders.contains(calleeDecl) && !db->func_builders.contains(calleeDecl)) {
                            if (isTemplateInstant || isLambda) {
                                FunctionBuilderBuilder fbfb(db, *stack);
                                fbfb.build(calleeDecl->getAsFunction());
                                calleeDecl = funcDecl;
                            }
                        }

                        auto query = stack->queries.empty() ? nullptr : stack->queries.back();
                        luisa::compute::ScopeStmt *query_scope = nullptr;
                        auto functionName = calleeDecl->getAsFunction()->getName();
                        if (functionName.starts_with("on_surface_candidate"))
                            query_scope = query->on_triangle_candidate();
                        if (functionName.starts_with("on_procedural_candidate"))
                            query_scope = query->on_procedural_candidate();
                        if (query_scope)
                            fb->push_scope(query_scope);

                        if (auto methodDecl = llvm::dyn_cast<clang::CXXMethodDecl>(calleeDecl);
                            methodDecl && (methodDecl->isCopyAssignmentOperator() || methodDecl->isMoveAssignmentOperator())) {
                            fb->assign(lc_args[0], lc_args[1]);
                            current = lc_args[0];
                        } else if (auto func_callable = db->func_builders[calleeDecl]) {
                            if (call->getCallReturnType(*astContext)->isVoidType())
                                fb->call(luisa::compute::Function(func_callable.get()), lc_args);
                            else if (auto lcReturnType = db->FindOrAddType(call->getCallReturnType(*astContext))) {
                                auto ret_value = fb->local(lcReturnType);
                                fb->assign(ret_value, fb->call(lcReturnType, luisa::compute::Function(func_callable.get()), lc_args));
                                current = ret_value;
                            } else
                                luisa::log_error("unfound return type in method/function: {}", call->getCallReturnType(*astContext)->getCanonicalTypeInternal().getAsString());
                        } else if (auto lambda_callable = db->lambda_builders[calleeDecl]) {
                            luisa::span<const luisa::compute::Expression *> lambda_args = {lc_args.begin() + 1, lc_args.end()};
                            if (call->getCallReturnType(*astContext)->isVoidType())
                                fb->call(luisa::compute::Function(lambda_callable.get()), lambda_args);
                            else if (auto lcReturnType = db->FindOrAddType(call->getCallReturnType(*astContext))) {
                                auto ret_value = fb->local(lcReturnType);
                                fb->assign(ret_value, fb->call(lcReturnType, luisa::compute::Function(lambda_callable.get()), lambda_args));
                                current = ret_value;
                            } else
                                luisa::log_error("unfound return type in lambda: {}", call->getCallReturnType(*astContext)->getCanonicalTypeInternal().getAsString());
                        } else {
                            calleeDecl->dump();
                            luisa::log_error("unfound function!");
                        }

                        if (query_scope) {
                            fb->pop_scope(query_scope);
                            fb->clangcxx_rayquery_postprocess(query_scope);
                        }
                    }
                }
            } else if (auto _init_expr = llvm::dyn_cast<clang::CXXDefaultInitExpr>(x)) {
                ExprTranslator v(stack, db, fb, _init_expr->getExpr());
                if (!v.TraverseStmt(_init_expr->getExpr()))
                    luisa::log_error("untranslated member call expr: {}", _init_expr->getExpr()->getStmtClassName());
                current = v.translated;
            } else if (auto _exprWithCleanup = llvm::dyn_cast<clang::ExprWithCleanups>(x)) {// TODO
                // luisa::log_warning("unimplemented ExprWithCleanups!");
                current = stack->expr_map[_exprWithCleanup->getSubExpr()];
            } else if (auto _matTemp = llvm::dyn_cast<clang::MaterializeTemporaryExpr>(x)) {// TODO
                // luisa::log_warning("unimplemented MaterializeTemporaryExpr!");
                current = stack->expr_map[_matTemp->getSubExpr()];
            } else if (auto _init_list = llvm::dyn_cast<clang::InitListExpr>(x)) {// TODO
                luisa::log_error("InitList is banned! Explicit use constructor instead!");
            } else if (auto _control_flow = llvm::dyn_cast<CompoundStmt>(x)) {       // CONTROL FLOW
            } else if (auto _control_flow = llvm::dyn_cast<clang::IfStmt>(x)) {      // CONTROL FLOW
            } else if (auto _control_flow = llvm::dyn_cast<clang::ContinueStmt>(x)) {// CONTROL FLOW
            } else if (auto _control_flow = llvm::dyn_cast<clang::BreakStmt>(x)) {   // CONTROL FLOW
            } else if (auto _control_flow = llvm::dyn_cast<clang::WhileStmt>(x)) {   // CONTROL FLOW
            } else if (auto _control_flow = llvm::dyn_cast<clang::SwitchStmt>(x)) {  // CONTROL FLOW
            } else if (auto _control_flow = llvm::dyn_cast<clang::CaseStmt>(x)) {    // CONTROL FLOW
            } else if (auto _control_flow = llvm::dyn_cast<clang::DefaultStmt>(x)) { // CONTROL FLOW
            } else if (auto _control_flow = llvm::dyn_cast<clang::ForStmt>(x)) {     // CONTROL FLOW
            } else if (auto null = llvm::dyn_cast<NullStmt>(x)) {                    // EMPTY
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

    ExprTranslator(Stack *stack, TypeDatabase *db, luisa::shared_ptr<compute::detail::FunctionBuilder> cur, clang::Stmt *root)
        : stack(stack), db(db), fb(cur), root(root) {
    }
    const luisa::compute::Expression *translated = nullptr;

protected:
    Stack *stack = nullptr;
    TypeDatabase *db = nullptr;
    luisa::shared_ptr<compute::detail::FunctionBuilder> fb = nullptr;
    clang::Stmt *root = nullptr;
};

void FunctionBuilderBuilder::build(const clang::FunctionDecl *S) {
    bool is_ignore = false;
    bool is_kernel = false;
    uint3 kernelSize;
    bool is_template = S->isTemplateDecl() && !S->isTemplateInstantiation();
    bool is_scope = false;
    bool is_method = false;
    bool is_lambda = false;
    QualType methodThisType;

    auto params = S->parameters();
    auto astContext = db->GetASTContext();
    for (auto Anno = S->specific_attr_begin<clang::AnnotateAttr>(); Anno != S->specific_attr_end<clang::AnnotateAttr>(); ++Anno) {
        is_ignore |= isIgnore(*Anno);
        is_scope |= isNoignore(*Anno);
        if (isKernel(*Anno)) {
            is_kernel = true;
            getKernelSize(*Anno, kernelSize.x, kernelSize.y, kernelSize.z);
        }
    }

    if (auto Method = llvm::dyn_cast<clang::CXXMethodDecl>(S)) {
        auto thisQt = Method->getThisType()->getPointeeType();
        if (auto thisType = GetRecordDeclFromQualType(thisQt)) {
            is_ignore |= thisType->isUnion();// ignore union
            for (auto Anno : thisType->specific_attrs<clang::AnnotateAttr>())
                is_ignore |= isBuiltinType(Anno);
            if (thisType->isLambda())// ignore global lambda declares, we deal them on stacks only
                is_lambda = true;

            is_ignore |= (Method->isImplicit() && Method->isCopyAssignmentOperator());
            is_ignore |= (Method->isImplicit() && Method->isMoveAssignmentOperator());
            is_template |= (thisType->getTypeForDecl()->getTypeClass() == clang::Type::InjectedClassName);
        } else {
            luisa::log_error("unfound this type [{}] in method [{}]",
                             Method->getThisType()->getPointeeType().getAsString(), S->getNameAsString());
        }
        is_method = !is_lambda;
        methodThisType = Method->getThisType()->getPointeeType();
    }
    for (auto param : params) {
        auto DesugaredParamType = param->getType().getNonReferenceType().getDesugaredType(*astContext);
        is_template |= DesugaredParamType->isTemplateTypeParmType();
        is_template |= (DesugaredParamType->getTypeClass() == clang::Type::TemplateTypeParm);
        is_template |= (DesugaredParamType->getTypeClass() == clang::Type::PackExpansion);
        is_template |= param->isTemplateParameterPack();
        is_template |= param->isTemplateParameter();
    }

    if (is_scope)
        is_ignore = false;
    if (is_template)
        is_ignore = true;

    if (!is_ignore) {
        // S->dump();
        if (S->getReturnType()->isReferenceType())
        {
            S->dump();
            luisa::log_error("return ref is not supportted now!");
        }

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
                builder = db->kernel_builder;
            else
                builder = luisa::make_shared<luisa::compute::detail::FunctionBuilder>(luisa::compute::Function::Tag::CALLABLE);

            if (is_lambda)
                db->lambda_builders[S] = builder;
            else
                db->func_builders[S] = builder;

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
                    if (auto lc_type = db->FindOrAddType(methodThisType)) {
                        auto this_local = builder->reference(lc_type);
                        stack.locals[nullptr] = this_local;
                    } else {
                        luisa::log_error("???");
                    }
                }

                // collect args
                for (auto param : params) {
                    auto Ty = param->getType();
                    if (auto lc_type = db->FindOrAddType(Ty)) {
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
                    } else {
                        luisa::log_error("unfound arg type: {}", Ty.getAsString());
                    }
                }

                // ctor initializers
                if (is_method) {
                    if (auto lc_type = db->FindOrAddType(methodThisType)) {
                        auto this_local = stack.locals[nullptr];
                        if (auto Ctor = llvm::dyn_cast<clang::CXXConstructorDecl>(S)) {
                            for (auto ctor_init : Ctor->inits()) {
                                auto init = ctor_init->getInit();
                                ExprTranslator v(&stack, db, builder, init);
                                if (v.TraverseStmt(init)) {
                                    const auto cxxMember = ctor_init->getMember();
                                    const auto lcMemberType = db->FindOrAddType(cxxMember->getType());
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

    ExprTranslator v(&stack, db, cur, currStmt);
    if (!v.TraverseStmt(currStmt))
        luisa::log_error("untranslated member call expr: {}", currStmt->getStmtClassName());

    return true;
}

void RecordDeclStmtHandler::run(const MatchFinder::MatchResult &Result) {
    auto &kernel_builder = db->kernel_builder;
    if (const auto *S = Result.Nodes.getNodeAs<clang::RecordDecl>("RecordDecl")) {
        QualType Ty = S->getTypeForDecl()->getCanonicalTypeInternal();
        db->RecordAsStuctureType(Ty);
    }
}

void GlobalVarHandler::run(const MatchFinder::MatchResult &Result) {
    auto &kernel_builder = db->kernel_builder;
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
        // lambdas & template instantiations will build at calls
        if (!isLambda && !S->isTemplateInstantiation()) {
            auto stack = Stack();
            FunctionBuilderBuilder bdbd(db, stack);
            bdbd.build(S);
        }
    }
}

ASTConsumer::ASTConsumer(std::string OutputPath, luisa::compute::Device *device, compute::ShaderOption option)
    : OutputPath(std::move(OutputPath)), device(device), option(option) {

    db.kernel_builder = luisa::make_shared<luisa::compute::detail::FunctionBuilder>(luisa::compute::Function::Tag::KERNEL);

    HandlerForTypeDecl.db = &db;
    Matcher.addMatcher(recordDecl(
                           isDefinition(),
                           unless(isExpansionInSystemHeader()))
                           .bind("RecordDecl"),
                       &HandlerForTypeDecl);

    HandlerForFuncionDecl.db = &db;
    Matcher.addMatcher(functionDecl(
                           isDefinition(),
                           unless(isExpansionInSystemHeader()))
                           .bind("FunctionDecl"),
                       &HandlerForFuncionDecl);

    HandlerForGlobalVar.db = &db;
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
        luisa::compute::Function{db.kernel_builder.get()});
}

void ASTConsumer::HandleTranslationUnit(clang::ASTContext &Context) {
    // 1. collect
    db.SetASTContext(&Context);
    Matcher.matchAST(Context);
}

}// namespace luisa::clangcxx