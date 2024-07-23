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
#include <luisa/backends/ext/raster_ext.hpp>
namespace luisa::clangcxx {
using namespace clang;
using namespace clang::ast_matchers;
using namespace luisa::compute;

inline bool FuncIsEmpty(Function func) {
    for (auto &&i : func.body()->statements()) {
        if (i->tag() != Statement::Tag::COMMENT) return false;
    }
    return true;
}

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
            clangcxx_log_error("unsupportted binary op {}!", luisa::to_string(op));
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
            clangcxx_log_error("unsupportted binary-assign op {}!", luisa::to_string(op));
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
            clangcxx_log_error("unsupportted unary op {}!", luisa::to_string(op));
            return LCUnaryOp::PLUS;
    }
}

inline const luisa::compute::RefExpr *LC_ArgOrRef(clang::QualType qt, compute::detail::FunctionBuilder *fb, const luisa::compute::Type *lcType) {
    if (qt->isPointerType())
        clangcxx_log_error("pointer type is not supported: [{}]", qt.getAsString());
    else if (qt->isReferenceType())
        return fb->reference(lcType);
    else
        return fb->argument(lcType);
    return nullptr;
}

inline const luisa::compute::RefExpr *LC_Local(
    compute::detail::FunctionBuilder *fb,
    const luisa::compute::Type *lcType,
    compute::Usage usage) {
    auto lcLocal = fb->local(lcType);
    lcLocal->mark(usage);
    return lcLocal;
}

const luisa::compute::RefExpr *Stack::GetLocal(const clang::ValueDecl *decl) const {
    if (locals.contains(decl))
        return locals.find(decl)->second;
    return nullptr;
}

void Stack::SetLocal(const clang::ValueDecl *decl, const luisa::compute::RefExpr *expr) {
    if (!decl)
        clangcxx_log_error("unknown error: SetLocal with nullptr!");
    if (locals.contains(decl))
        clangcxx_log_error("unknown error: SetLocal with existed!");
    locals[decl] = expr;
}

const luisa::compute::Expression *Stack::GetExpr(const clang::Stmt *stmt) const {
    if (expr_map.contains(stmt))
        return expr_map.find(stmt)->second;
    return nullptr;
}

void Stack::SetExpr(const clang::Stmt *stmt, const luisa::compute::Expression *expr) {
    if (!stmt)
        clangcxx_log_error("unknown error: SetExpr with nullptr!");
    if (expr_map.contains(stmt)) {
        // TODO: ignore template case.
        // stmt->dump();
        // clangcxx_log_error("unknown error: SetExpr with existed!");
        return;
    }
    expr_map[stmt] = expr;
}

const luisa::compute::Expression *Stack::GetConstant(const clang::ValueDecl *var) const {
    if (constants.contains(var))
        return constants.find(var)->second;
    return nullptr;
}

void Stack::SetConstant(const clang::ValueDecl *var, const luisa::compute::Expression *expr) {
    if (!var)
        clangcxx_log_error("unknown error: SetConstant with nullptr!");
    if (constants.contains(var)) {
        var->dump();
        clangcxx_log_error("unknown error: SetConstant with existed!");
    }
    constants[var] = expr;
}

bool Stack::isCtorExpr(const luisa::compute::Expression *expr) {
    return ctor_exprs.contains(expr);
}

void Stack::SetExprAsCtor(const luisa::compute::Expression *expr) {
    ctor_exprs.emplace(expr);
}

struct ExprTranslator : public clang::RecursiveASTVisitor<ExprTranslator> {
    clang::ForStmt *currentCxxForStmt = nullptr;

    const luisa::compute::Expression *TraverseAPArray(const APValue &APV, const luisa::compute::Type *lcType) {
        const APValue &INNER = APV.getStructField(0);
        auto DIM = INNER.getArraySize();
        // clang-format off
#define TYPE_CASE_FLOAT(TYPE, COND)\
    else if (COND) {\
        auto lcArray = LC_Local(fb, lcType, Usage::READ);\
        for (uint32_t i = 0; i < DIM; i++) \
            fb->assign(\
                fb->access(Type::of<TYPE>(), lcArray, fb->literal(Type::of<uint32_t>(), i)),\
                fb->literal(Type::of<TYPE>(), (TYPE)INNER.getArrayInitializedElt(i).getFloat().convertToDouble()));\
        return lcArray; \
    }

#define TYPE_CASE_INT(TYPE, COND)\
    else if (COND) {\
        auto lcArray = LC_Local(fb, lcType, Usage::READ);\
        for (uint32_t i = 0; i < DIM; i++) \
            fb->assign(\
                fb->access(Type::of<TYPE>(), lcArray, fb->literal(Type::of<uint32_t>(), i)),\
                fb->literal(Type::of<TYPE>(), (TYPE)INNER.getArrayInitializedElt(i).getInt().getLimitedValue()));\
        return lcArray; \
    }

        auto AS_HALF = lcType->element()->is_float16();
        auto AS_FLOAT = lcType->element()->is_float32();
        auto AS_INT16 = lcType->element()->is_int16();
        auto AS_INT32 = lcType->element()->is_int32();
        auto AS_INT64 = lcType->element()->is_int64();
        auto AS_UINT16 = lcType->element()->is_uint16();
        auto AS_UINT32 = lcType->element()->is_uint32();
        auto AS_UINT64 = lcType->element()->is_uint64();
        auto AS_BOOL = lcType->element()->is_bool();
        if (false);
        TYPE_CASE_FLOAT(half, AS_HALF)
        TYPE_CASE_FLOAT(float, AS_FLOAT)

        TYPE_CASE_INT(bool, AS_BOOL)

        TYPE_CASE_INT(short, AS_INT16)
        TYPE_CASE_INT(int, AS_INT32)
        TYPE_CASE_INT(slong, AS_INT64)

        TYPE_CASE_INT(ushort, AS_UINT16)
        TYPE_CASE_INT(uint, AS_UINT32)
        TYPE_CASE_INT(ulong, AS_UINT64)
        else
            clangcxx_log_error("array not supportted as constexpr detected!!!");
        // clang-format on
#undef TYPE_CASE_FLOAT
#undef TYPE_CASE_INT
        return nullptr;
    }

    const luisa::compute::Expression *TraverseAPMatrix(const APValue &APV, const luisa::compute::Type *lcType) {
        const APValue &VEC_ARRAY = APV.getStructField(0).getStructField(0);
        auto Y = VEC_ARRAY.getArraySize();
        auto X = (Y == 3) ? 4 : Y;
        const luisa::compute::Type *VecTypeLUT[] = {nullptr, nullptr, Type::of<float2>(), Type::of<float3>(), Type::of<float4>()};
        auto lcMatrix = LC_Local(fb, lcType, Usage::READ);
        for (uint32_t y = 0; y < Y; y++) {
            fb->assign(
                fb->access(VecTypeLUT[X], lcMatrix, fb->literal(Type::of<uint32_t>(), y)),
                TraverseAPVector(VEC_ARRAY.getArrayInitializedElt(y), VecTypeLUT[X]));
        }
        return lcMatrix;
    }

    const luisa::compute::Expression *TraverseAPVector(const APValue &APV, const luisa::compute::Type *lcType) {
        const APValue *VecInnerField = &APV.getStructField(0).getUnionValue();
        auto INNER = VecInnerField->getStructField(0);
        auto DIM = INNER.getArraySize();
        // clang-format off
#define TYPE_CASE_FLOAT(TYPE, COND)\
    else if (DIM == 2 && COND) return fb->literal(lcType, \
            TYPE##2((TYPE)INNER.getArrayInitializedElt(0).getFloat().convertToDouble(),\
                (TYPE)INNER.getArrayInitializedElt(1).getFloat().convertToDouble()));\
    else if (DIM == 3 && COND) return fb->literal(lcType, \
            TYPE##3((TYPE)INNER.getArrayInitializedElt(0).getFloat().convertToDouble(),\
                (TYPE)INNER.getArrayInitializedElt(1).getFloat().convertToDouble(),\
                (TYPE)INNER.getArrayInitializedElt(2).getFloat().convertToDouble()));\
    else if (DIM == 4 && COND) return fb->literal(lcType, \
            TYPE##4((TYPE)INNER.getArrayInitializedElt(0).getFloat().convertToDouble(),\
                (TYPE)INNER.getArrayInitializedElt(1).getFloat().convertToDouble(),\
                (TYPE)INNER.getArrayInitializedElt(2).getFloat().convertToDouble(),\
                (TYPE)INNER.getArrayInitializedElt(3).getFloat().convertToDouble()));

#define TYPE_CASE_INT(TYPE, COND)\
    else if (DIM == 2 && COND) return fb->literal(lcType, \
            TYPE##2((TYPE)INNER.getArrayInitializedElt(0).getInt().getLimitedValue(),\
                (TYPE)INNER.getArrayInitializedElt(1).getInt().getLimitedValue()));\
    else if (DIM == 3 && COND) return fb->literal(lcType, \
            TYPE##3((TYPE)INNER.getArrayInitializedElt(0).getInt().getLimitedValue(),\
                (TYPE)INNER.getArrayInitializedElt(1).getInt().getLimitedValue(),\
                (TYPE)INNER.getArrayInitializedElt(2).getInt().getLimitedValue()));\
    else if (DIM == 4 && COND) return fb->literal(lcType, \
            TYPE##4((TYPE)INNER.getArrayInitializedElt(0).getInt().getLimitedValue(),\
                (TYPE)INNER.getArrayInitializedElt(1).getInt().getLimitedValue(),\
                (TYPE)INNER.getArrayInitializedElt(2).getInt().getLimitedValue(),\
                (TYPE)INNER.getArrayInitializedElt(3).getInt().getLimitedValue()));

        auto AS_HALF = lcType->is_float16_vector();
        auto AS_FLOAT = lcType->is_float32_vector();
        auto AS_INT16 = lcType->is_int16_vector();
        auto AS_INT32 = lcType->is_int32_vector();
        auto AS_INT64 = lcType->is_int64_vector();
        auto AS_UINT16 = lcType->is_uint16_vector();
        auto AS_UINT32 = lcType->is_uint32_vector();
        auto AS_UINT64 = lcType->is_uint64_vector();
        auto AS_BOOL = lcType->is_bool_vector();
        if (false);
        TYPE_CASE_FLOAT(half, AS_HALF)
        TYPE_CASE_FLOAT(float, AS_FLOAT)

        TYPE_CASE_INT(bool, AS_BOOL)

        TYPE_CASE_INT(short, AS_INT16)
        TYPE_CASE_INT(int, AS_INT32)
        TYPE_CASE_INT(slong, AS_INT64)

        TYPE_CASE_INT(ushort, AS_UINT16)
        TYPE_CASE_INT(uint, AS_UINT32)
        TYPE_CASE_INT(ulong, AS_UINT64)
        else
        {
            APV.dump();
            clangcxx_log_error("vec not supportted as constexpr detected! description: {}", lcType->description());
        }
        // clang-format on
#undef TYPE_CASE_FLOAT
#undef TYPE_CASE_INT
        return nullptr;
    }

    const luisa::compute::Expression *TraverseAPValue(const APValue &APV, clang::RecordDecl *what, clang::Stmt *where) {
        const auto APK = APV.getKind();
        switch (APK) {
            case clang::APValue::ValueKind::Int:
                return fb->literal(Type::of<int>(), (int)APV.getInt().getLimitedValue());
            case clang::APValue::ValueKind::Float:
                return fb->literal(Type::of<float>(), (float)APV.getFloat().convertToDouble());
            case clang::APValue::ValueKind::Struct: {
                auto N = APV.getStructNumFields();
                if (auto lcType = db->FindOrAddType(what->getTypeForDecl()->getCanonicalTypeUnqualified(), what->getBeginLoc())) {
                    if (lcType->is_array()) {
                        return TraverseAPArray(APV, lcType);
                    } else if (lcType->is_vector()) {
                        return TraverseAPVector(APV, lcType);
                    } else if (lcType->is_matrix()) {
                        return TraverseAPMatrix(APV, lcType);
                    } else {
                        auto constant = LC_Local(fb, lcType, Usage::READ);
                        auto fields = what->fields();
                        uint32_t i = 0;
                        for (auto field : fields) {
                            auto FieldWhat = GetRecordDeclFromQualType(field->getType(), false);
                            if (auto lcFieldType = db->FindOrAddType(field->getType(), field->getBeginLoc())) {
                                auto lcField = TraverseAPValue(APV.getStructField(i), FieldWhat, where);
                                fb->assign(fb->member(lcFieldType, constant, i), lcField);
                            } else
                                clangcxx_log_error("bad constexpr field!");
                            i += 1;
                        }
                        return constant;
                    }
                }
            } break;
            case clang::APValue::ValueKind::Union:
            case clang::APValue::ValueKind::ComplexInt:
            case clang::APValue::ValueKind::ComplexFloat:
            default:
                if (where)
                    db->DumpWithLocation(where);
                APV.dump();
                clangcxx_log_error("unsupportted ConstantExpr APValueKind {}", luisa::to_string(APK));
                break;
        }
        return nullptr;
    }

    const luisa::compute::Expression *FindOrTraverseAPValue(const clang::ValueDecl *cxxVar, clang::Stmt *where) {
        if (auto Cached = stack->GetConstant(cxxVar))
            return Cached;
        if (auto Decompressed = cxxVar->getPotentiallyDecomposedVarDecl()) {
            if (auto Evaluated = Decompressed->getEvaluatedValue()) {
                auto VarTypeDecl = GetRecordDeclFromQualType(Decompressed->getType(), false);
                if (auto constant = TraverseAPValue(*Evaluated, VarTypeDecl, where)) {
                    stack->SetConstant(cxxVar, constant);
                    return constant;
                } else// TODO: support assignment by constexpr var
                {
                    db->DumpWithLocation(where);
                    clangcxx_log_error("unsupportted gloal const eval type: {}", luisa::to_string(Decompressed->getKind()));
                }
            }
        }
        return nullptr;
    }

    bool TraverseStmt(clang::Stmt *x) {
        if (x == nullptr) return true;

        // CONTROL WALK
        if (auto cxxLambda = llvm::dyn_cast<LambdaExpr>(x)) {
            auto cxxCallee = cxxLambda->getLambdaClass()->getLambdaCallOperator();
            Stack newStack = *stack;
            FunctionBuilderBuilder bdbd(db, newStack);
            bdbd.build(cxxCallee, false);
        } else if (auto cxxBranch = llvm::dyn_cast<clang::IfStmt>(x)) {
            // cxxBranch->getConditionVariable()->getEvaluatedStmt()->Value
            auto _ = db->CommentStmt(fb, cxxBranch);

            auto cxxCond = cxxBranch->getCond();
            auto ifConstVar = cxxCond->getIntegerConstantExpr(*db->GetASTContext());
            if (ifConstVar) {
                if (ifConstVar->getExtValue() != 0) {
                    if (cxxBranch->getThen())
                        TraverseStmt(cxxBranch->getThen());
                } else {
                    if (cxxBranch->getElse())
                        TraverseStmt(cxxBranch->getElse());
                }
            } else {
                TraverseStmt(cxxCond);
                auto lcIf = fb->if_(stack->GetExpr(cxxCond));
                fb->push_scope(lcIf->true_branch());
                if (cxxBranch->getThen())
                    TraverseStmt(cxxBranch->getThen());
                fb->pop_scope(lcIf->true_branch());

                fb->push_scope(lcIf->false_branch());
                if (cxxBranch->getElse())
                    TraverseStmt(cxxBranch->getElse());
                fb->pop_scope(lcIf->false_branch());
            }
        } else if (auto cxxSwitch = llvm::dyn_cast<clang::SwitchStmt>(x)) {
            auto _ = db->CommentStmt(fb, cxxSwitch);

            auto cxxCond = cxxSwitch->getCond();
            TraverseStmt(cxxCond);

            auto lcSwitch = fb->switch_(stack->GetExpr(cxxSwitch->getCond()));
            fb->push_scope(lcSwitch->body());
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
            fb->pop_scope(lcSwitch->body());
        } else if (auto cxxCase = llvm::dyn_cast<clang::CaseStmt>(x)) {
            auto cxxCond = cxxCase->getLHS();
            TraverseStmt(cxxCond);

            auto lcCase = fb->case_(stack->GetExpr(cxxCond));
            fb->push_scope(lcCase->body());
            if (auto cxxBody = cxxCase->getSubStmt())
                TraverseStmt(cxxBody);
            fb->pop_scope(lcCase->body());
        } else if (auto cxxDefault = llvm::dyn_cast<clang::DefaultStmt>(x)) {
            auto lcDefault = fb->default_();
            fb->push_scope(lcDefault->body());
            if (auto cxxBody = cxxDefault->getSubStmt())
                TraverseStmt(cxxBody);
            fb->pop_scope(lcDefault->body());
        } else if (auto cxxContinue = llvm::dyn_cast<clang::ContinueStmt>(x)) {
            if (currentCxxForStmt)
                TraverseStmt(currentCxxForStmt->getInc());
            fb->continue_();
        } else if (auto cxxBreak = llvm::dyn_cast<clang::BreakStmt>(x)) {
            fb->break_();
        } else if (auto cxxWhile = llvm::dyn_cast<clang::WhileStmt>(x)) {
            auto _ = db->CommentStmt(fb, cxxWhile);
            auto cxxCond = cxxWhile->getCond();
            auto ifConstVar = cxxCond->getIntegerConstantExpr(*db->GetASTContext());
            if (ifConstVar) {
                if (ifConstVar->getExtValue() != 0) {
                    auto lcWhile = fb->loop_();
                    fb->push_scope(lcWhile->body());
                    {
                        // body
                        auto cxxBody = cxxWhile->getBody();
                        TraverseStmt(cxxBody);
                    }
                    fb->pop_scope(lcWhile->body());
                }
            } else {
                auto lcWhile = fb->loop_();
                fb->push_scope(lcWhile->body());
                {
                    TraverseStmt(cxxCond);
                    auto lcCondIf = fb->if_(stack->GetExpr(cxxCond));
                    // break
                    fb->push_scope(lcCondIf->false_branch());
                    fb->break_();
                    fb->pop_scope(lcCondIf->false_branch());
                    // body
                    auto cxxBody = cxxWhile->getBody();
                    TraverseStmt(cxxBody);
                }
                fb->pop_scope(lcWhile->body());
            }
        } else if (auto cxxFor = llvm::dyn_cast<clang::ForStmt>(x)) {
            auto _ = db->CommentStmt(fb, cxxFor);

            currentCxxForStmt = cxxFor;
            // i = 0
            auto cxxInit = cxxFor->getInit();
            TraverseStmt(cxxInit);
            // while (cond)
            auto lcWhile = fb->loop_();
            fb->push_scope(lcWhile->body());
            {
                auto cxxCond = cxxFor->getCond();
                TraverseStmt(cxxCond);
                auto lcCondIf = fb->if_(stack->GetExpr(cxxCond));
                // break
                fb->push_scope(lcCondIf->false_branch());
                fb->break_();
                fb->pop_scope(lcCondIf->false_branch());
                // body
                auto cxxBody = cxxFor->getBody();
                TraverseStmt(cxxBody);
                // i++
                auto cxxInc = cxxFor->getInc();
                TraverseStmt(cxxInc);
            }
            fb->pop_scope(lcWhile->body());
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
                current = LC_Local(fb, db->FindOrAddType(methodThisType, x->getBeginLoc()), Usage::READ);
            } else if (auto cxxDecl = llvm::dyn_cast<clang::DeclStmt>(x)) {
                auto _ = db->CommentStmt(fb, cxxDecl);

                const DeclGroupRef declGroup = cxxDecl->getDeclGroup();
                for (auto decl : declGroup) {
                    if (!decl) continue;

                    if (auto *varDecl = dyn_cast<clang::VarDecl>(decl)) {
                        auto Ty = varDecl->getType();
                        auto cxxInit = varDecl->getInit();
                        auto initStyle = varDecl->getInitStyle();
                        const bool isRef = Ty->isReferenceType();
                        const bool isArray = Ty->getAsArrayTypeUnsafe();
                        if (isRef || isArray) {
                            db->DumpWithLocation(x);
                            if (isRef)
                                clangcxx_log_error("VarDecl as reference type is not supported: [{}]", Ty.getAsString());
                            if (isArray)
                                clangcxx_log_error("VarDecl as C-style array type is not supported: [{}]", Ty.getAsString());
                        }
                        if (auto lcType = db->FindOrAddType(Ty, x->getBeginLoc())) {
                            if (auto lcInit = stack->GetExpr(cxxInit)) {
                                if (auto isCtorExpr = stack->isCtorExpr(lcInit)) {
                                    stack->SetLocal(varDecl, (const compute::RefExpr *)lcInit);
                                    current = lcInit;
                                } else {
                                    auto lcVar = LC_Local(fb, lcType, Usage::WRITE);
                                    fb->assign(lcVar, lcInit);
                                    stack->SetLocal(varDecl, lcVar);
                                    current = lcVar;
                                }
                            } else {
                                auto lcVar = LC_Local(fb, lcType, Usage::WRITE);
                                stack->SetLocal(varDecl, lcVar);
                                current = lcVar;
                            }
                        } else
                            clangcxx_log_error("VarDecl with unfound type: [{}]", Ty.getAsString());
                    } else if (auto aliasDecl = dyn_cast<clang::TypeAliasDecl>(decl)) {          // ignore
                    } else if (auto staticAssertDecl = dyn_cast<clang::StaticAssertDecl>(decl)) {// ignore
                    } else {
                        db->DumpWithLocation(x);
                        clangcxx_log_error("unsupported decl stmt: {}", cxxDecl->getStmtClassName());
                    }
                }
            } else if (auto cxxRet = llvm::dyn_cast<clang::ReturnStmt>(x)) {
                auto _ = db->CommentStmt(fb, cxxRet);

                auto cxxRetVal = cxxRet->getRetValue();
                auto lcRet = stack->GetExpr(cxxRetVal);
                if (fb->tag() != compute::Function::Tag::KERNEL) {
                    fb->return_(lcRet);
                } else {
                    fb->return_();
                }
            } else if (auto ce = llvm::dyn_cast<clang::ConstantExpr>(x)) {
                if (auto constant = TraverseAPValue(ce->getAPValueResult(), nullptr, x)) {
                    current = constant;
                } else
                    clangcxx_log_error("unsupportted ConstantExpr APValueKind {}", luisa::to_string(ce->getResultAPValueKind()));
            } else if (auto substNonType = llvm::dyn_cast<SubstNonTypeTemplateParmExpr>(x)) {
                auto lcExpr = stack->GetExpr(substNonType->getReplacement());
                current = lcExpr;
            } else if (auto il = llvm::dyn_cast<IntegerLiteral>(x)) {
                auto limitedVal = il->getValue().getLimitedValue(UINT64_MAX);
                if (il->getType()->isSignedIntegerType()) {
                    if (limitedVal <= INT32_MAX)
                        current = fb->literal(Type::of<int>(), static_cast<int>(limitedVal));
                    else if (limitedVal < INT64_MAX)
                        current = fb->literal(Type::of<int64>(), static_cast<int64>(limitedVal));
                } else {
                    if (limitedVal <= UINT32_MAX)
                        current = fb->literal(Type::of<uint>(), static_cast<uint>(limitedVal));
                    else
                        current = fb->literal(Type::of<uint64>(), static_cast<uint64>(limitedVal));
                }
            } else if (auto bl = llvm::dyn_cast<CXXBoolLiteralExpr>(x)) {
                current = fb->literal(Type::of<bool>(), (bool)bl->getValue());
            } else if (auto fl = llvm::dyn_cast<FloatingLiteral>(x)) {
                current = fb->literal(Type::of<float>(), (float)fl->getValue().convertToDouble());
            } else if (auto cxxCtorCall = llvm::dyn_cast<CXXConstructExpr>(x)) {
                auto _ = db->CommentStmt(fb, cxxCtorCall);

                auto cxxCtor = cxxCtorCall->getConstructor();
                const bool needCustom = !(cxxCtor->isImplicit() && cxxCtor->isCopyOrMoveConstructor());
                const bool moveCtor = cxxCtor->isMoveConstructor();

                // TODO: REFACTOR THIS
                if (needCustom && !db->func_builders.contains(cxxCtor)) {
                    auto funcDecl = cxxCtor->getAsFunction();
                    auto methodDecl = llvm::dyn_cast<clang::CXXMethodDecl>(funcDecl);
                    const auto isTemplateInstant = funcDecl->isTemplateInstantiation();
                    if (isTemplateInstant) {
                        FunctionBuilderBuilder fbfb(db, *stack);
                        fbfb.build(funcDecl, false);
                    }
                }
                luisa::vector<const luisa::compute::Expression *> lcArgs;
                const compute::RefExpr *constructed = nullptr;
                SKR_DEFER({  stack->SetExprAsCtor(constructed); current = constructed; });
                if (!moveCtor) {
                    constructed = LC_Local(fb, db->FindOrAddType(cxxCtorCall->getType(), x->getBeginLoc()), Usage::WRITE);
                    // args
                    lcArgs.emplace_back(constructed);
                    for (auto arg : cxxCtorCall->arguments()) {
                        if (auto lcArg = stack->GetExpr(arg))
                            lcArgs.emplace_back(lcArg);
                        else
                            clangcxx_log_error("unfound arg: {}", arg->getStmtClassName());
                    }
                }

                if (moveCtor) {
                    auto lcArg = stack->GetExpr(cxxCtorCall->getArg(0));
                    constructed = static_cast<const compute::RefExpr *>(lcArg);
                } else if (cxxCtor->isCopyConstructor()) {
                    fb->assign(constructed, lcArgs[1]);
                } else if (auto callable = db->func_builders[cxxCtor]) {
                    luisa::compute::Function func(callable.get());
                    if (!FuncIsEmpty(func))
                        fb->call(func, lcArgs);
                } else if (cxxCtor->getParent()->isLambda()) {
                    // ...IGNORE LAMBDA CTOR...
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
                                    bool defaultInit = (lcArgs.size() <= 1);
                                    if (!defaultInit) {
                                        auto N = Arguments[1].getAsIntegral().getLimitedValue();
                                        // clang-format off
                                        switch (EType->getKind()) {
                #define CASE_VEC_TYPE(stype, type)                                                                                    \
                    switch (N) {                       \
                        case 2: { auto lcType = Type::of<stype##2>(); fb->assign(constructed, fb->call(lcType, CallOp::MAKE_##type##2, { lcArgs.begin() + 1, lcArgs.end() })); } break; \
                        case 3: { auto lcType = Type::of<stype##3>(); fb->assign(constructed, fb->call(lcType, CallOp::MAKE_##type##3, { lcArgs.begin() + 1, lcArgs.end() })); } break; \
                        case 4: { auto lcType = Type::of<stype##4>(); fb->assign(constructed, fb->call(lcType, CallOp::MAKE_##type##4, { lcArgs.begin() + 1, lcArgs.end() })); } break; \
                        default: {                                                                                             \
                            clangcxx_log_error("unsupported type: {}, kind {}, N {}", Ty.getAsString(), luisa::to_string(EType->getKind()), N);    \
                        } break;                                                                                               \
                    }
                                            case (BuiltinType::Kind::Bool): { CASE_VEC_TYPE(bool, BOOL) } break;
                                            case (BuiltinType::Kind::Float): { CASE_VEC_TYPE(float, FLOAT) } break;
                                            case (BuiltinType::Kind::Long): { CASE_VEC_TYPE(slong, LONG) } break;
                                            case (BuiltinType::Kind::Int): { CASE_VEC_TYPE(int, INT) } break;
                                            case (BuiltinType::Kind::ULong): { CASE_VEC_TYPE(ulong, ULONG) } break;
                                            case (BuiltinType::Kind::UInt): { CASE_VEC_TYPE(uint, UINT) } break;
                                            case (BuiltinType::Kind::Short): { CASE_VEC_TYPE(short, SHORT) } break;
                                            case (BuiltinType::Kind::UShort): { CASE_VEC_TYPE(ushort, USHORT) } break;
                                            case (BuiltinType::Kind::Double): { CASE_VEC_TYPE(double, DOUBLE) } break;
                                            default: {
                                                clangcxx_log_error("unsupported type: {}, kind {}", Ty.getAsString(), luisa::to_string(EType->getKind()));
                                            } break;
                #undef CASE_VEC_TYPE
                                        }
                                        // clang-format on
                                    }
                                }
                            } else
                                clangcxx_log_error("???");
                        } else if (builtinName == "matrix") {
                            if (auto TSD = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(Ty->getAs<clang::RecordType>()->getDecl())) {
                                auto &Arguments = TSD->getTemplateArgs();
                                clang::Expr::EvalResult Result;
                                auto N = Arguments[0].getAsIntegral().getLimitedValue();
                                auto lcType = Type::matrix(N);
                                const CallOp MATRIX_LUT[3] = {CallOp::MAKE_FLOAT2X2, CallOp::MAKE_FLOAT3X3, CallOp::MAKE_FLOAT4X4};
                                if (lcArgs.size() > 1)
                                    fb->assign(constructed, fb->call(lcType, MATRIX_LUT[N - 2], {lcArgs.begin() + 1, lcArgs.end()}));
                            } else
                                clangcxx_log_error("???");
                        } else if (builtinName == "array") {
                            if (auto TSD = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(Ty->getAs<clang::RecordType>()->getDecl())) {
                                auto &Arguments = TSD->getTemplateArgs();
                                auto EType = Arguments[0].getAsType();
                                auto N = Arguments[1].getAsIntegral().getLimitedValue();
                                auto lcElemType = db->FindOrAddType(EType, x->getBeginLoc());
                                const luisa::compute::Type *lcArrayType = Type::array(lcElemType, N);

                                auto Flags = Arguments[2].getAsIntegral().getLimitedValue();
                                if (Flags & 1)
                                    constructed = fb->shared(lcArrayType);

                                if (!cxxCtor->isDefaultConstructor()) {
                                    if (cxxCtor->isConvertingConstructor(true))
                                        fb->assign(constructed, fb->cast(lcArrayType, CastOp::STATIC, lcArgs[1]));
                                    else if (cxxCtor->isCopyConstructor())
                                        fb->assign(constructed, lcArgs[1]);
                                    else if (cxxCtor->isMoveConstructor())
                                        clangcxx_log_error("unexpected move array constructor!");
                                    else
                                        clangcxx_log_error("unhandled array constructor: {}", cxxCtor->getNameAsString());
                                }
                            }
                        } else {
                            clangcxx_log_error("unhandled builtin constructor: {}", cxxCtor->getNameAsString());
                        }
                    } else {
                        db->DumpWithLocation(cxxCtor);
                        clangcxx_log_error("unfound constructor: {}", cxxCtor->getNameAsString());
                    }
                }
            } else if (auto unary_or_trait = llvm::dyn_cast<UnaryExprOrTypeTraitExpr>(x)) {
                if (unary_or_trait->getKind() == clang::UETT_SizeOf)
                    current = fb->literal(Type::of<uint>(), (uint)(db->GetASTContext()->getTypeSize(unary_or_trait->getArgumentType()) / 8ull));
                else if (unary_or_trait->getKind() == clang::UETT_PreferredAlignOf || unary_or_trait->getKind() == clang::UETT_AlignOf)
                    current = fb->literal(Type::of<uint>(), (uint)(db->GetASTContext()->getTypeAlign(unary_or_trait->getArgumentType()) / 8ull));
                else
                    clangcxx_log_error("unsupportted UnaryExprOrTypeTraitExpr: {}", unary_or_trait->getStmtClassName());
            } else if (auto unary = llvm::dyn_cast<UnaryOperator>(x)) {
                const auto cxx_op = unary->getOpcode();
                const auto lhs = stack->GetExpr(unary->getSubExpr());
                const auto lcType = db->FindOrAddType(unary->getType(), x->getBeginLoc());
                if (cxx_op == CXXUnaryOp::UO_Deref) {
                    if (auto _this = llvm::dyn_cast<CXXThisExpr>(unary->getSubExpr()))
                        current = db->GetFunctionThis(fb);
                    else
                        clangcxx_log_error("only support deref 'this'(*this)!");
                } else if (!IsUnaryAssignOp(cxx_op)) {
                    if ((cxx_op == clang::UO_AddrOf) && unary->getType()->isFunctionPointerType()) {
                        auto funcPtr = LC_Local(fb, luisa::compute::Type::of<uint64_t>(), compute::Usage::READ_WRITE);
                        auto funcDecl = unary->getSubExpr()->getReferencedDeclOfCallee()->getAsFunction();
                        if (!db->lambda_builders.contains(funcDecl) && !db->func_builders.contains(funcDecl)) {
                            FunctionBuilderBuilder fbfb(db, *stack);
                            funcDecl->getAsFunction()->dump();
                            fbfb.build(funcDecl, false);
                        }
                        auto func = db->func_builders.contains(funcDecl) ?
                                        db->func_builders[funcDecl] :
                                        db->lambda_builders[funcDecl];
                        fb->assign(
                            funcPtr,
                            fb->func_ref(func->function()));
                        current = funcPtr;
                    } else
                        current = fb->unary(lcType, TranslateUnaryOp(cxx_op), lhs);
                } else {
                    auto one = fb->literal(Type::of<int>(), 1);
                    auto typed_one = one;
                    switch (cxx_op) {
                        case clang::UO_PreInc:
                        case clang::UO_PreDec: {
                            auto lcBinop = (cxx_op == clang::UO_PreInc) ? LCBinOp::ADD : LCBinOp::SUB;
                            auto ca_expr = fb->binary(lcType, lcBinop, lhs, typed_one);
                            fb->assign(lhs, ca_expr);
                            current = lhs;
                            break;
                        }
                        case clang::UO_PostInc:
                        case clang::UO_PostDec: {
                            auto lcBinop = (cxx_op == clang::UO_PostInc) ? LCBinOp::ADD : LCBinOp::SUB;
                            auto old = LC_Local(fb, lcType, Usage::WRITE);
                            fb->assign(old, lhs);
                            auto ca_expr = fb->binary(lcType, lcBinop, lhs, typed_one);
                            fb->assign(lhs, ca_expr);
                            current = old;
                            break;
                        }
                    }
                }
            } else if (auto bin = llvm::dyn_cast<BinaryOperator>(x)) {
                auto _ = db->CommentStmt(fb, bin);

                const auto cxx_op = bin->getOpcode();
                const auto lhs = stack->GetExpr(bin->getLHS());
                const auto rhs = stack->GetExpr(bin->getRHS());
                const auto lcType = db->FindOrAddType(bin->getType(), x->getBeginLoc());
                if (auto ca = llvm::dyn_cast<CompoundAssignOperator>(x)) {
                    auto ca_expr = fb->binary(lcType, TranslateBinaryAssignOp(cxx_op), lhs, rhs);
                    fb->assign(lhs, ca_expr);
                    current = lhs;
                } else if (cxx_op == CXXBinOp::BO_Assign) {
                    fb->assign(lhs, rhs);
                    current = lhs;
                } else {
                    if (!rhs) {
                        db->DumpWithLocation(x);
                        clangcxx_log_error("ICE, unexpected parameter: rhs not found!");
                    }
                    current = fb->binary(lcType, TranslateBinaryOp(cxx_op), lhs, rhs);
                }
            } else if (auto cxxCondOp = llvm::dyn_cast<clang::ConditionalOperator>(x)) {
                auto _cond = stack->GetExpr(cxxCondOp->getCond());
                auto _true = stack->GetExpr(cxxCondOp->getTrueExpr());
                auto _false = stack->GetExpr(cxxCondOp->getFalseExpr());
                const auto lcType = db->FindOrAddType(cxxCondOp->getType(), x->getBeginLoc());
                current = fb->call(lcType, CallOp::SELECT, {_false, _true, _cond});
            } else if (auto dref = llvm::dyn_cast<DeclRefExpr>(x)) {
                auto str = luisa::string(dref->getNameInfo().getName().getAsString());
                if (auto _current = stack->GetLocal(dref->getDecl())) {
                    current = _current;
                } else if (auto Var = dref->getDecl(); Var && llvm::isa<clang::VarDecl>(Var)) {// Value Ref
                    if (dref->isNonOdrUse() != NonOdrUseReason::NOUR_Unevaluated ||
                        dref->isNonOdrUse() != NonOdrUseReason::NOUR_Discarded) {
                        if (auto constant = FindOrTraverseAPValue(Var, x))
                            current = constant;
                        else {
                            db->DumpWithLocation(dref);
                            clangcxx_log_error("unfound & unresolved ref: {}", str);
                        }
                    }
                } else if (auto value = dref->getDecl(); value && llvm::isa<clang::FunctionDecl>(value))// Func Ref
                    ;
                else
                    clangcxx_log_error("unfound var ref: {}", str);
            } else if (auto _cxxParen = llvm::dyn_cast<ParenExpr>(x)) {
                current = stack->GetExpr(_cxxParen->getSubExpr());
            } else if (auto implicit_cast = llvm::dyn_cast<ImplicitCastExpr>(x)) {
                if (stack->GetExpr(implicit_cast->getSubExpr()) != nullptr) {
                    const auto lcCastType = db->FindOrAddType(implicit_cast->getType(), x->getBeginLoc());
                    auto lcExpr = stack->GetExpr(implicit_cast->getSubExpr());
                    if (!lcExpr) {
                        db->DumpWithLocation(implicit_cast->getSubExpr());
                        clangcxx_log_error("unknown error: rhs not found!");
                    }
                    if (lcExpr->type() != lcCastType)
                        current = fb->cast(lcCastType, CastOp::STATIC, lcExpr);
                    else
                        current = lcExpr;
                }
            } else if (auto _explicit_cast = llvm::dyn_cast<ExplicitCastExpr>(x)) {
                if (stack->GetExpr(_explicit_cast->getSubExpr()) != nullptr) {
                    const auto lcCastType = db->FindOrAddType(_explicit_cast->getType(), x->getBeginLoc());
                    auto lcExpr = stack->GetExpr(_explicit_cast->getSubExpr());
                    if (!lcExpr) {
                        db->DumpWithLocation(_explicit_cast->getSubExpr());
                        clangcxx_log_error("unknown error: rhs not found!");
                    }
                    if (lcExpr->type() != lcCastType)
                        current = fb->cast(lcCastType, CastOp::STATIC, lcExpr);
                    else
                        current = lcExpr;
                } else {
                    _explicit_cast->getSubExpr()->dump();
                    clangcxx_log_error("dont cast function type, use function pointer type instead");
                }
            } else if (auto cxxDefaultArg = llvm::dyn_cast<clang::CXXDefaultArgExpr>(x)) {
                auto _ = db->CommentStmt(fb, cxxDefaultArg);

                const auto _value = LC_Local(fb, db->FindOrAddType(cxxDefaultArg->getType(), x->getBeginLoc()), Usage::READ_WRITE);
                TraverseStmt(cxxDefaultArg->getExpr());
                fb->assign(_value, stack->GetExpr(cxxDefaultArg->getExpr()));
                current = _value;
            } else if (auto t = llvm::dyn_cast<clang::CXXThisExpr>(x)) {
                current = db->GetFunctionThis(fb);
            } else if (auto cxxMember = llvm::dyn_cast<clang::MemberExpr>(x)) {
                if (auto bypass = isByPass(cxxMember->getMemberDecl())) {
                    current = stack->GetExpr(cxxMember->getBase());
                    if (cxxMember->isBoundMemberFunction(*astContext))
                        stack->callers.emplace_back(current);
                } else if (cxxMember->isBoundMemberFunction(*astContext)) {
                    auto lhs = stack->GetExpr(cxxMember->getBase());
                    stack->callers.emplace_back(lhs);
                } else if (auto cxxField = llvm::dyn_cast<FieldDecl>(cxxMember->getMemberDecl())) {
                    if (isSwizzle(cxxField)) {
                        auto swizzleText = cxxField->getName();
                        const auto swizzleType = cxxField->getType().getDesugaredType(*astContext).getNonReferenceType();
                        if (auto lcResultType = db->FindOrAddType(swizzleType, x->getBeginLoc())) {
                            uint64_t swizzle_code = 0u;
                            uint64_t swizzle_seq[] = {0u, 0u, 0u, 0u}; /*4*/
                            int64_t swizzle_size = 0;
                            for (auto iter = swizzleText.begin(); iter != swizzleText.end(); iter++) {
                                if (*iter == 'x') swizzle_seq[swizzle_size] = 0u;
                                if (*iter == 'y') swizzle_seq[swizzle_size] = 1u;
                                if (*iter == 'z') swizzle_seq[swizzle_size] = 2u;
                                if (*iter == 'w') swizzle_seq[swizzle_size] = 3u;

                                if (*iter == 'r') swizzle_seq[swizzle_size] = 0u;
                                if (*iter == 'g') swizzle_seq[swizzle_size] = 1u;
                                if (*iter == 'b') swizzle_seq[swizzle_size] = 2u;
                                if (*iter == 'a') swizzle_seq[swizzle_size] = 3u;

                                swizzle_size += 1;
                            }
                            // encode swizzle code
                            for (int64_t cursor = swizzle_size - 1; cursor >= 0; cursor--) {
                                swizzle_code <<= 4;
                                swizzle_code |= swizzle_seq[cursor];
                            }
                            auto lhs = stack->GetExpr(cxxMember->getBase());
                            current = fb->swizzle(lcResultType, lhs, swizzle_size, swizzle_code);
                        }
                    } else {
                        auto lhs = stack->GetExpr(cxxMember->getBase());
                        const auto lcMemberType = db->FindOrAddType(cxxField->getType(), x->getBeginLoc());
                        current = fb->member(lcMemberType, lhs, cxxField->getFieldIndex());
                    }
                } else {
                    clangcxx_log_error("unsupported member expr: {}", cxxMember->getMemberDecl()->getNameAsString());
                }
            } else if (auto call = llvm::dyn_cast<clang::CallExpr>(x)) {
                if (isByPass(call->getCalleeDecl())) {
                    auto caller = stack->callers.back();
                    stack->callers.pop_back();
                    if (!caller) {
                        db->DumpWithLocation(call);
                        clangcxx_log_error("incorrect [[bypass]] call detected!");
                    }
                    current = caller;
                } else {
                    auto _ = db->CommentStmt(fb, call);
                    auto calleeDecl = call->getCalleeDecl();
                    auto funcDecl = calleeDecl->getAsFunction();
                    llvm::StringRef callopName = {};
                    llvm::StringRef extCallName = {};
                    llvm::StringRef binopName = {};
                    llvm::StringRef unaopName = {};
                    llvm::StringRef exprName = {};
                    bool isAccess = false;
                    bool isFuncRef = false;
                    for (auto attr : calleeDecl->specific_attrs<clang::AnnotateAttr>()) {
                        if (callopName.empty())
                            callopName = getCallopName(attr);
                        if (extCallName.empty())
                            extCallName = getExtCallName(attr);
                        if (binopName.empty())
                            binopName = getBinopName(attr);
                        if (unaopName.empty())
                            unaopName = getUnaopName(attr);
                        if (exprName.empty())
                            exprName = getExprName(attr);
                        isAccess |= luisa::clangcxx::isAccess(attr);
                        isFuncRef |= luisa::clangcxx::isFuncRef(attr);
                    }
                    // args
                    luisa::string printer_str;
                    luisa::vector<const luisa::compute::Expression *> lcArgs;
                    if (auto mcall = llvm::dyn_cast<clang::CXXMemberCallExpr>(x)) {
                        auto caller = stack->callers.back();
                        stack->callers.pop_back();

                        lcArgs.emplace_back(caller);// from -MemberExpr::isBoundMemberFunction
                    }
                    for (auto arg : call->arguments()) {
                        if (auto lcArg = stack->GetExpr(arg))
                            lcArgs.emplace_back(lcArg);
                        else if (auto _str_literal = llvm::dyn_cast<clang::StringLiteral>(arg)) {
                            auto &&str = _str_literal->getString();
                            if (callopName != "device_log") {
                                lcArgs.emplace_back(fb->constant(ConstantData::create(Type::array(Type::of<uint8_t>(), str.size()), str.bytes_begin(), str.size())));
                                lcArgs.emplace_back(fb->literal(Type::of<uint64_t>(), uint64_t(str.size())));
                            } else {
                                printer_str = luisa::string{reinterpret_cast<char const *>(str.bytes_begin()), str.size()};
                            }
                        } else {
                            db->DumpWithLocation(arg);
                            clangcxx_log_error("unfound arg: {}", arg->getStmtClassName());
                        }
                    }
                    // call
                    auto cxxReturnType = call->getCallReturnType(*astContext);
                    if (callopName == "device_log") [[unlikely]] {
                        fb->print_(std::move(printer_str), lcArgs);
                    } else if (!binopName.empty()) {
                        auto lcBinop = db->FindBinOp(binopName);
                        if (auto lcReturnType = db->FindOrAddType(cxxReturnType, x->getBeginLoc()))
                            current = fb->binary(lcReturnType, lcBinop, lcArgs[0], lcArgs[1]);
                    } else if (!unaopName.empty()) {
                        UnaryOp lcUnaop = (unaopName == "PLUS")  ? UnaryOp::PLUS :
                                          (unaopName == "MINUS") ? UnaryOp::MINUS :
                                                                   (clangcxx_log_error("unsupportted unary op {}!!", unaopName.data()), UnaryOp::PLUS);
                        if (auto lcReturnType = db->FindOrAddType(cxxReturnType, x->getBeginLoc()))
                            current = fb->unary(lcReturnType, lcUnaop, lcArgs[0]);
                    } else if (isAccess) {
                        if (auto lcReturnType = db->FindOrAddType(cxxReturnType, x->getBeginLoc())) {
                            if (lcArgs.size() >= 2) {
                                current = fb->access(lcReturnType, lcArgs[0], lcArgs[1]);
                            } else {
                                current = fb->access(lcReturnType, lcArgs[0], fb->literal(Type::of<int32_t>(), int32_t(0)));
                            }
                        }
                    } else if (!exprName.empty()) {
                        if (exprName == "dispatch_id")
                            current = fb->dispatch_id();
                        else if (exprName == "block_id")
                            current = fb->block_id();
                        else if (exprName == "thread_id")
                            current = fb->thread_id();
                        else if (exprName == "dispatch_size")
                            current = fb->dispatch_size();
                        else if (exprName == "kernel_id")
                            current = fb->kernel_id();
                        else if (exprName == "object_id")
                            current = fb->object_id();
                        else if (exprName == "warp_lane_count")
                            current = fb->warp_lane_count();
                        else if (exprName == "warp_lane_id")
                            current = fb->warp_lane_id();
                        else if (exprName == "bit_cast") {
                            auto lcReturnType = db->FindOrAddType(funcDecl->getReturnType(), x->getBeginLoc());
                            current = fb->cast(lcReturnType, luisa::compute::CastOp::BITWISE, lcArgs[0]);
                        } else
                            clangcxx_log_error("ICE: unsupportted expr: {}", exprName.str());
                    } else if (!callopName.empty()) {
                        auto op = db->FindCallOp(callopName);
                        if (call->getCallReturnType(*astContext)->isVoidType())
                            fb->call(op, lcArgs);
                        else if (auto lcReturnType = db->FindOrAddType(call->getCallReturnType(*astContext), x->getBeginLoc())) {
                            auto ret_value = LC_Local(fb, lcReturnType, Usage::WRITE);
                            fb->assign(ret_value, fb->call(lcReturnType, op, lcArgs));
                            current = ret_value;
                        } else
                            clangcxx_log_error(
                                "unfound return type: {}",
                                call->getCallReturnType(*astContext)->getCanonicalTypeInternal().getAsString());
                    } else if (!extCallName.empty()) {
                        luisa::vector<const Type *> arg_types;
                        luisa::vector<Usage> argument_usages;
                        arg_types.resize_uninitialized(lcArgs.size());
                        argument_usages.resize_uninitialized(lcArgs.size());
                        for (auto &i : argument_usages) {
                            i = Usage::READ;
                        }
                        for (auto i : vstd::range(lcArgs.size())) {
                            arg_types[i] = lcArgs[i]->type();
                        }
                        auto get_ext_func = [&](ExternalFunction &&ext_func) -> auto & {
                            auto iter = db->ext_funcs.try_emplace(ext_func.hash(), vstd::lazy_eval([&]() {
                                                                      return luisa::make_shared<ExternalFunction>(std::move(ext_func));
                                                                  }));
                            return iter.first->second;
                        };
                        if (call->getCallReturnType(*astContext)->isVoidType()) {
                            auto ext_func = get_ext_func(ExternalFunction(luisa::string(extCallName.data(), extCallName.size()), Type::of<void>(), std::move(arg_types), std::move(argument_usages)));
                            fb->call(std::move(ext_func), lcArgs);
                        } else if (auto lcReturnType = db->FindOrAddType(call->getCallReturnType(*astContext), x->getBeginLoc())) {
                            auto ret_value = LC_Local(fb, lcReturnType, Usage::WRITE);
                            auto ext_func = get_ext_func(ExternalFunction(luisa::string(extCallName.data(), extCallName.size()), lcReturnType, std::move(arg_types), std::move(argument_usages)));
                            fb->assign(ret_value, fb->call(lcReturnType, std::move(ext_func), lcArgs));
                            current = ret_value;
                        } else
                            clangcxx_log_error(
                                "unfound return type: {}",
                                call->getCallReturnType(*astContext)->getCanonicalTypeInternal().getAsString());
                        // TODO: external call
                    } else {
                        // TODO: REFACTOR THIS
                        auto methodDecl = llvm::dyn_cast<clang::CXXMethodDecl>(funcDecl);
                        const auto isTemplateInstant = funcDecl->isTemplateInstantiation();
                        const auto isLambda = methodDecl && methodDecl->getParent()->isLambda();
                        if (!db->lambda_builders.contains(calleeDecl) && !db->func_builders.contains(calleeDecl)) {
                            if (isTemplateInstant || isLambda) {
                                FunctionBuilderBuilder fbfb(db, *stack);
                                fbfb.build(calleeDecl->getAsFunction(), false);
                                calleeDecl = funcDecl;
                            }
                        }

                        if (auto methodDecl = llvm::dyn_cast<clang::CXXMethodDecl>(calleeDecl);
                            methodDecl && (methodDecl->isCopyAssignmentOperator() || methodDecl->isMoveAssignmentOperator())) {//TODO
                            fb->assign(lcArgs[0], lcArgs[1]);
                            current = lcArgs[0];
                        } else if (auto func_callable = db->func_builders[calleeDecl]) {
                            if (call->getCallReturnType(*astContext)->isVoidType()) {
                                luisa::compute::Function func(func_callable.get());
                                if (!FuncIsEmpty(func))
                                    fb->call(func, lcArgs);
                            } else if (auto lcReturnType = db->FindOrAddType(call->getCallReturnType(*astContext), x->getBeginLoc())) {
                                auto ret_value = LC_Local(fb, lcReturnType, Usage::WRITE);
                                fb->assign(ret_value, fb->call(lcReturnType, luisa::compute::Function(func_callable.get()), lcArgs));
                                current = ret_value;
                            } else
                                clangcxx_log_error("unfound return type in method/function: {}", call->getCallReturnType(*astContext)->getCanonicalTypeInternal().getAsString());
                        } else if (auto lambda_callable = db->lambda_builders[calleeDecl]) {
                            luisa::span<const luisa::compute::Expression *> lambda_args = {lcArgs.begin() + 1, lcArgs.end()};
                            if (call->getCallReturnType(*astContext)->isVoidType()) {
                                luisa::compute::Function func(lambda_callable.get());
                                if (!FuncIsEmpty(func))
                                    fb->call(func, lambda_args);
                            } else if (auto lcReturnType = db->FindOrAddType(call->getCallReturnType(*astContext), x->getBeginLoc())) {
                                auto ret_value = LC_Local(fb, lcReturnType, Usage::WRITE);
                                fb->assign(ret_value, fb->call(lcReturnType, luisa::compute::Function(lambda_callable.get()), lambda_args));
                                current = ret_value;
                            } else
                                clangcxx_log_error("unfound return type in lambda: {}", call->getCallReturnType(*astContext)->getCanonicalTypeInternal().getAsString());
                        } else {
                            db->DumpWithLocation(calleeDecl);
                            clangcxx_log_error("unfound function!");
                        }
                    }
                }
            } else if (auto _init_expr = llvm::dyn_cast<clang::CXXDefaultInitExpr>(x)) {
                ExprTranslator v(stack, db, fb, _init_expr->getExpr());
                if (!v.TraverseStmt(_init_expr->getExpr()))
                    clangcxx_log_error("untranslated member call expr: {}", _init_expr->getExpr()->getStmtClassName());
                current = v.translated;
            } else if (auto _exprWithCleanup = llvm::dyn_cast<clang::ExprWithCleanups>(x)) {// TODO
                // luisa::log_warning("unimplemented ExprWithCleanups!");
                current = stack->GetExpr(_exprWithCleanup->getSubExpr());
            } else if (auto _matTemp = llvm::dyn_cast<clang::MaterializeTemporaryExpr>(x)) {// TODO
                // luisa::log_warning("unimplemented MaterializeTemporaryExpr!");
                current = stack->GetExpr(_matTemp->getSubExpr());
            } else if (auto _init_list = llvm::dyn_cast<clang::InitListExpr>(x)) {// TODO
                db->DumpWithLocation(x);
                clangcxx_log_error("InitList is banned! Explicit use constructor instead!");
            } else if (auto _str_literal = llvm::dyn_cast<clang::StringLiteral>(x)) {
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
                db->DumpWithLocation(x);
                clangcxx_log_error("unsupportted expr!");
            }
        }

        if (auto existed = stack->GetExpr(x); !existed)
            stack->SetExpr(x, current);

        if (x == root)
            translated = current;

        return true;
    }

    ExprTranslator(Stack *stack, TypeDatabase *db, compute::detail::FunctionBuilder *cur, clang::Stmt *root)
        : stack(stack), db(db), fb(cur), root(root) {
    }
    const luisa::compute::Expression *translated = nullptr;

protected:
    Stack *stack = nullptr;
    TypeDatabase *db = nullptr;
    compute::detail::FunctionBuilder *fb = nullptr;
    clang::Stmt *root = nullptr;
};// namespace luisa::clangcxx

auto FunctionBuilderBuilder::build(const clang::FunctionDecl *S, bool allowKernel) -> BuildResult {
    BuildResult result{
        .dimension = 0};
    bool is_builtin_type_method = false;
    bool is_ignore = false;
    bool is_kernel = false;
    bool is_vertex = false;
    bool is_pixel = false;
    uint3 kernelSize;
    bool is_template = S->isTemplateDecl() && !S->isTemplateInstantiation();
    bool is_scope = false;
    bool is_method = false;
    bool is_lambda = false;
    QualType methodThisType;

    auto params = S->parameters();
    auto astContext = db->GetASTContext();
    for (auto Anno : S->specific_attrs<clang::AnnotateAttr>()) {
        is_ignore |= isIgnore(Anno);
        is_scope |= isNoignore(Anno);
        if (isKernel(Anno)) {
            is_kernel = true;
            if (isKernel1D(Anno)) {
                result.dimension = 1;
            } else if (isKernel2D(Anno)) {
                result.dimension = 2;
            } else {
                result.dimension = 3;
            }
            getKernelSize(Anno, kernelSize.x, kernelSize.y, kernelSize.z);
        } else if (isVertex(Anno)) {
            is_vertex = true;
        } else if (isPixel(Anno)) {
            is_pixel = true;
        }
        if (isDump(Anno))
            db->DumpWithLocation(S);
    }

    const auto TemplateKind = S->getTemplatedKind();
    is_template |= (TemplateKind == clang::FunctionDecl::TemplatedKind::TK_FunctionTemplate);

    if (auto Method = llvm::dyn_cast<clang::CXXMethodDecl>(S)) {
        if (auto thisType = Method->getParent()) {
            is_ignore |= thisType->isUnion();// ignore union
            for (auto Anno : thisType->specific_attrs<clang::AnnotateAttr>()) {
                is_builtin_type_method |= isBuiltinType(Anno);
                is_ignore |= is_builtin_type_method;
            }
            if (thisType->isLambda())// ignore global lambda declares, we deal them on stacks only
                is_lambda = true;

            is_ignore |= (Method->isImplicit() && Method->isCopyAssignmentOperator());
            is_ignore |= (Method->isImplicit() && Method->isMoveAssignmentOperator());
            if (auto Ctor = llvm::dyn_cast<clang::CXXConstructorDecl>(S))
                is_ignore |= (Ctor->isImplicit() && Ctor->isCopyOrMoveConstructor());
            is_template |= (thisType->getTypeForDecl()->getTypeClass() == clang::Type::InjectedClassName);
        } else {
            clangcxx_log_error("unfound this type [{}] in method [{}]",
                               Method->getParent()->getNameAsString(), S->getNameAsString());
        }
        is_method = !is_lambda;
        methodThisType = Method->getParent()->getTypeForDecl()->getCanonicalTypeUnqualified();
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
        if (S->getReturnType()->isReferenceType()) {
            db->DumpWithLocation(S);
            clangcxx_log_error("return ref is not supportted now!");
        }

        if (auto Method = llvm::dyn_cast<clang::CXXMethodDecl>(S)) {
            if (Method->isMoveAssignmentOperator()) {
                db->DumpWithLocation(Method);
                clangcxx_log_error("error: BL-1.1 custom move assignment operator is not allowed!");
            }
            if (Method->isCopyAssignmentOperator()) {
                db->DumpWithLocation(Method);
                clangcxx_log_error("error: BL-1.2 custom copy assignment operator is not allowed!");
            }
            if (auto Ctor = llvm::dyn_cast<clang::CXXConstructorDecl>(S)) {
                if (Ctor->isMoveConstructor() && !Ctor->isImplicit()) {
                    db->DumpWithLocation(Method);
                    clangcxx_log_error("error: BL-1.3 custom move constructor is not allowed!");
                }
                if (Ctor->isCopyConstructor() && !Ctor->isImplicit()) {
                    db->DumpWithLocation(Method);
                    clangcxx_log_error("error: BL-1.4 custom copy constructor is not allowed!");
                }
            }
        }

        if (auto Dtor = llvm::dyn_cast<clang::CXXDestructorDecl>(S)) {
            if (!Dtor->isDefaulted()) {
                db->DumpWithLocation(S);
                auto dtorName = Dtor->getQualifiedNameAsString();
                clangcxx_log_error("dtor {} is not allowed!", dtorName.c_str());
            }
        }

        luisa::shared_ptr<compute::detail::FunctionBuilder> builder_sharedptr;
        Stmt *body = S->getBody();
        {
            if (is_kernel) {
                if (!allowKernel) [[unlikely]] {
                    clangcxx_log_error("Kernel definition not allowed in callable export.");
                }
                if (db->kernel_builder) [[unlikely]] {
                    clangcxx_log_error("Kernel can not be redefined.");
                }
                db->kernel_builder = luisa::make_shared<luisa::compute::detail::FunctionBuilder>(luisa::compute::Function::Tag::KERNEL);
                builder_sharedptr = db->kernel_builder;
            } else if (is_vertex) {
                if (!allowKernel) [[unlikely]] {
                    clangcxx_log_error("Vertex definition not allowed in callable export.");
                }
                if (db->vertex_builder) [[unlikely]] {
                    clangcxx_log_error("Kernel can not be redefined.");
                }
                db->vertex_builder = luisa::make_shared<luisa::compute::detail::FunctionBuilder>(luisa::compute::Function::Tag::RASTER_STAGE);
                builder_sharedptr = db->vertex_builder;
            } else if (is_pixel) {
                if (!allowKernel) [[unlikely]] {
                    clangcxx_log_error("Pixel definition not allowed in callable export.");
                }
                if (db->pixel_builder) [[unlikely]] {
                    clangcxx_log_error("Kernel can not be redefined.");
                }
                db->pixel_builder = luisa::make_shared<luisa::compute::detail::FunctionBuilder>(luisa::compute::Function::Tag::RASTER_STAGE);
                builder_sharedptr = db->pixel_builder;
            } else
                builder_sharedptr = luisa::make_shared<luisa::compute::detail::FunctionBuilder>(luisa::compute::Function::Tag::CALLABLE);
            auto builder = builder_sharedptr.get();
            result.func = luisa::compute::Function{builder};
            if (is_lambda)
                db->lambda_builders[S] = std::move(builder_sharedptr);
            else
                db->func_builders[S] = std::move(builder_sharedptr);

            luisa::compute::detail::FunctionBuilder::push(builder);
            builder->push_scope(builder->body());
            {
                if (is_kernel) {
                    builder->set_block_size(kernelSize);
                }

                // comment name
                luisa::string name;
#if LC_CLANGCXX_ENABLE_COMMENT
                {
                    if (auto Ctor = llvm::dyn_cast<clang::CXXConstructorDecl>(S))
                        name = "[Ctor] ";
                    else if (auto Method = llvm::dyn_cast<clang::CXXMethodDecl>(S))
                        name = "[Method] ";
                    else if (auto Dtor = llvm::dyn_cast<clang::CXXDestructorDecl>(S))
                        name = "[Dtor] ";
                    else
                        name = "[Function] ";
                    name += luisa::string(S->getQualifiedNameAsString());
                    builder->comment_(std::move(name));
                }
#endif
                // Stack stack;
                // this arg
                if (is_method) {
                    auto Method = llvm::dyn_cast<clang::CXXMethodDecl>(S);
                    if (auto lcType = db->FindOrAddType(methodThisType, Method->getBeginLoc())) {
                        auto this_local = builder->reference(lcType);
                        db->SetFunctionThis(builder, this_local);
                    } else {
                        clangcxx_log_error("???");
                    }
                }

                // collect args
                for (auto param : params) {
                    auto Ty = param->getType();
                    if (auto lcType = db->FindOrAddType(Ty, param->getBeginLoc())) {
                        const luisa::compute::RefExpr *local = nullptr;
                        switch (lcType->tag()) {
                            case compute::Type::Tag::BUFFER:
                                local = builder->buffer(lcType);
                                break;
                            case compute::Type::Tag::TEXTURE:
                                local = builder->texture(lcType);
                                break;
                            case compute::Type::Tag::BINDLESS_ARRAY:
                                local = builder->bindless_array();
                                break;
                            case compute::Type::Tag::ACCEL:
                                local = builder->accel();
                                break;
                            default: {
                                /*
                                const bool isBuiltinType = param->getType()->isBuiltinType();
                                auto cxxDecl = param->getType()->getAsCXXRecordDecl();
                                const bool isLambda = cxxDecl ? cxxDecl->isLambda() : false;
                                if (!Ty->isReferenceType() && !isBuiltinType && !isLambda)
                                {
                                    luisa::log_warning("performance warning: "
                                        "PWL-1.1, use 'const T&' rather than 'T' to pass non-builtin types.");
                                    db->DumpWithLocation(param);
                                }
                                */
                                local = LC_ArgOrRef(Ty, builder, lcType);
                            } break;
                        }
                        stack.SetLocal(param, local);
                    } else {
                        clangcxx_log_error("unfound arg type: {}", Ty.getAsString());
                    }
                }

                // ctor initializers
                if (is_method && !is_builtin_type_method) {
                    if (auto lcType = db->FindOrAddType(methodThisType, S->getBeginLoc())) {
                        auto this_local = db->GetFunctionThis(builder);
                        if (auto Ctor = llvm::dyn_cast<clang::CXXConstructorDecl>(S)) {
                            for (auto ctor_init : Ctor->inits()) {
                                auto init = ctor_init->getInit();
                                ExprTranslator v(&stack, db, builder, init);
                                if (v.TraverseStmt(init)) {
                                    const auto cxxMember = ctor_init->getMember();
                                    const auto lcMemberType = db->FindOrAddType(cxxMember->getType(), cxxMember->getBeginLoc());
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
            luisa::compute::detail::FunctionBuilder::pop(builder);
        }
    }
    return result;
}

bool FunctionBuilderBuilder::recursiveVisit(clang::Stmt *currStmt, compute::detail::FunctionBuilder *cur, Stack &stack) {
    if (!currStmt)
        return true;

    ExprTranslator v(&stack, db, cur, currStmt);
    if (!v.TraverseStmt(currStmt))
        clangcxx_log_error("untranslated member call expr: {}", currStmt->getStmtClassName());

    return true;
}

void RecordDeclStmtHandler::run(const MatchFinder::MatchResult &Result) {
    if (const auto *S = Result.Nodes.getNodeAs<clang::RecordDecl>("RecordDecl")) {
        QualType Ty = S->getTypeForDecl()->getCanonicalTypeInternal();
        bool ignore = S->isUnion();
        for (auto Anno : S->specific_attrs<clang::AnnotateAttr>()) {
            ignore |= isIgnore(Anno);
            ignore |= isBuiltinType(Anno);
            if (isDump(Anno))
                db->DumpWithLocation(S);
        }
        if (!ignore)
            db->RecordType(Ty);
    }
}

void GlobalVarHandler::run(const MatchFinder::MatchResult &Result) {
    if (const auto *S = Result.Nodes.getNodeAs<clang::VarDecl>("VarDecl")) {
        bool ignore = false;
        for (auto Anno : S->specific_attrs<clang::AnnotateAttr>())
            ignore |= isIgnore(Anno);
        const auto isGlobal = S->isStaticLocal() || S->isStaticDataMember() || S->isFileVarDecl();
        const auto isConst = S->isConstexpr() || S->getType().isConstQualified();
        const auto isNonConstGlobal = isGlobal && !isConst;
        if (!ignore && isNonConstGlobal) {
            db->DumpWithLocation(S);
            clangcxx_log_error("global vars are banned!");
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
            auto result = bdbd.build(S, call_lib == nullptr);
            const auto is_export_func = [&]() {
                for (auto attr : S->specific_attrs<clang::AnnotateAttr>()) {
                    if (isExport(attr)) return true;
                }
                return false;
            }();
            if (is_export_func) {
                auto func_name = S->getName();
                if (!call_lib) [[unlikely]] {
                    LUISA_WARNING("This is not a ast export compilation. Function {} export attribute ignored.", func_name.str());
                } else {
                    call_lib->add_callable(
                        luisa::string_view{func_name.data(), func_name.size()},
                        result.func.shared_builder());
                }
            }
            if (result.dimension > 0) dimension = result.dimension;
        }
    }
}

ASTConsumer::ASTConsumer(luisa::compute::Device *device, compute::ShaderOption option)
    : device(device), option(std::move(option)) {
}
ASTCallableConsumer::ASTCallableConsumer(compute::CallableLibrary *lib) {
    HandlerForFuncionDecl.call_lib = lib;
}
ASTCallableConsumer::~ASTCallableConsumer() {
}
ASTConsumerBase::ASTConsumerBase() {
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
}

ASTConsumerBase::~ASTConsumerBase() {
}
ASTConsumer::~ASTConsumer() {
    if (db.kernel_builder == nullptr) [[unlikely]] {
        if (db.vertex_builder && db.pixel_builder) {
            auto raster_ext = device->extension<RasterExt>();
            raster_ext->create_raster_shader(luisa::compute::Function{db.vertex_builder.get()}, luisa::compute::Function{db.pixel_builder.get()}, option);
        } else {
            clangcxx_log_error("Kernel not defined.");
        }
    } else {
        device->impl()->create_shader(option, luisa::compute::Function{db.kernel_builder.get()});
    }
}

void ASTConsumerBase::HandleTranslationUnit(clang::ASTContext &Context) {
    // 1. collect
    db.SetASTContext(&Context);
    Matcher.matchAST(Context);
}

}// namespace luisa::clangcxx
