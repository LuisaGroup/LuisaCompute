#include "ASTConsumer.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "AttributeHelpers.hpp"
#include <iostream>
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

inline static luisa::string GetTypeName(clang::QualType type, clang::ASTContext *ctx) {
    type = type.getCanonicalType();
    auto baseName = luisa::string(type.getAsString(ctx->getLangOpts()));
    Remove(baseName, "struct ");
    Remove(baseName, "class ");
    return baseName;
}

inline static clang::RecordDecl *GetRecordDeclFromQualType(clang::QualType Ty, luisa::string parent = {}) {
    clang::RecordDecl *recordDecl = Ty->getAsRecordDecl();
    if (!recordDecl) {
        if (const auto *TDT = Ty->getAs<clang::TypedefType>()) {
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

const luisa::compute::Type *RecordDeclStmtHandler::RecordAsPrimitiveType(const clang::QualType Ty, const clang::RecordDecl *decl) {
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
                luisa::log_error("unsupported field primitive type: [{}], kind [{}] in type [{}]",
                    Ty.getAsString(), builtin->getKind(), decl->getNameAsString());
            }
            break;
        }
        // clang-format on
    }
    if (_type) {
        blackboard->type_map[GetTypeName(Ty, blackboard->astContext)] = _type;
    }
    return _type;
}

const luisa::compute::Type *RecordDeclStmtHandler::RecordAsBuiltinType(const QualType Ty, const clang::RecordDecl *decl) {
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
                    if (Arguments[1].getAsExpr()->EvaluateAsConstantExpr(Result, *blackboard->astContext)) {
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
            }
        } else if (builtin_type_name == "array") {
            if (auto TST = Ty->getAs<TemplateSpecializationType>()) {
                auto Arguments = TST->template_arguments();
                clang::Expr::EvalResult Result;
                if (Arguments[1].getAsExpr()->EvaluateAsConstantExpr(Result, *blackboard->astContext)) {
                    auto N = Result.Val.getInt().getExtValue();
                    auto Qualified = GetTypeName(Arguments[0].getAsType(), blackboard->astContext);
                    auto lc_type = blackboard->type_map.find(Qualified);
                    if (lc_type != blackboard->type_map.end()) {
                        _type = Type::array(lc_type->second, N);
                    } else {
                        luisa::log_error("unfound array element type: {}", Arguments[0].getAsType().getAsString());
                    }
                }
            }
        } else {
            luisa::log_error("unsupported builtin type: {} as a field", luisa::string(builtin_type_name));
        }
    }
    if (_type) {
        blackboard->type_map[GetTypeName(Ty, blackboard->astContext)] = _type;
    }
    return _type;
}

const luisa::compute::Type *RecordDeclStmtHandler::RecordAsStuctureType(const clang::QualType Ty, const clang::RecordDecl *decl) {
    const luisa::compute::Type *_type = nullptr;
    auto field_qualified_type_name = GetTypeName(Ty, blackboard->astContext);
    auto iter = blackboard->type_map.find(field_qualified_type_name);
    if (iter != blackboard->type_map.end()) {
        _type = iter->second;
    }
    if (_type) {
        blackboard->type_map[GetTypeName(Ty, blackboard->astContext)] = _type;
    }
    return _type;
}

bool RecordDeclStmtHandler::TryEmplaceFieldType(const clang::QualType Qt, const clang::RecordDecl *decl, luisa::vector<const luisa::compute::Type *> &types) {
    const luisa::compute::Type *_type = nullptr;
    clang::QualType Ty = Qt;
    // 1. PRIMITIVE
    if (auto builtin = Ty->getAs<clang::BuiltinType>()) {
        _type = RecordAsPrimitiveType(Ty, decl);
        if (!_type) {
            luisa::log_error("unsupported field primitive type: [{}], kind [{}] in type [{}]",
                             Ty.getAsString(), builtin->getKind(), decl->getNameAsString());
        }
    } else {
        // 2. EMPLACE RECORD
        if (clang::RecordDecl *recordDecl = GetRecordDeclFromQualType(Ty, luisa::string(decl->getNameAsString()))) {
            // 2.1 AS BUILTIN
            if (!_type) {
                _type = RecordAsBuiltinType(Ty, recordDecl);
            }
            // 2.2 AS STRUCTURE
            if (!_type) {
                _type = RecordAsStuctureType(Ty, recordDecl);
            }
        } else {
            decl->dump();
            luisa::log_error("unsupported field type [{}] in type [{}]", Ty.getAsString(), decl->getNameAsString());
        }
    }
    if (_type) {
        types.emplace_back(_type);
    }
    return _type;
}

void RecordDeclStmtHandler::run(const MatchFinder::MatchResult &Result) {
    auto &kernel_builder = blackboard->kernel_builder;
    if (const auto *S = Result.Nodes.getNodeAs<clang::RecordDecl>("RecordDecl")) {
        bool ignore = false;
        for (auto Anno = S->specific_attr_begin<clang::AnnotateAttr>(); Anno != S->specific_attr_end<clang::AnnotateAttr>(); ++Anno) {
            ignore |= isIgnore(*Anno) || isBuiltinType(*Anno);
        }
        if (!ignore) {
            luisa::vector<const luisa::compute::Type *> types;
            for (auto f = S->field_begin(); f != S->field_end(); f++) {
                auto Ty = f->getType();
                if (!TryEmplaceFieldType(Ty, S, types)) {
                    // S->dump();
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
            blackboard->type_map[GetTypeName(Ty, blackboard->astContext)] = lc_type;
        }
    }
}

struct ExprTranslator : public clang::RecursiveASTVisitor<ExprTranslator> {
    bool TraverseStmt(clang::Stmt *x) {
        RecursiveASTVisitor<ExprTranslator>::TraverseStmt(x);

        const luisa::compute::Expression *current = nullptr;
        if (auto il = llvm::dyn_cast<IntegerLiteral>(x)) {
            current = cur->literal(Type::of<int>(), (int)il->getValue().getLimitedValue());
        } else if (auto bin = llvm::dyn_cast<BinaryOperator>(x)) {
            const auto cxx_op = bin->getOpcode();
            const auto lhs = stack->expr_map[bin->getLHS()];
            const auto rhs = stack->expr_map[bin->getRHS()];
            const auto lc_type = blackboard->type_map[GetTypeName(bin->getType(), blackboard->astContext)];
            if (auto ca = llvm::dyn_cast<CompoundAssignOperator>(x)) {
                auto local = cur->binary(lc_type, TranslateBinaryAssignOp(cxx_op), lhs, rhs);
                cur->assign(lhs, local);
                current = local;
            } else {
                current = cur->binary(lc_type, TranslateBinaryOp(cxx_op), lhs, rhs);
            }
        } else if (auto dref = llvm::dyn_cast<DeclRefExpr>(x)) {
            auto str = luisa::string(dref->getNameInfo().getName().getAsString());
            current = stack->locals[str];
        } else if (auto implicit_cast = llvm::dyn_cast<ImplicitCastExpr>(x)) {
            const auto lc_type = blackboard->type_map[GetTypeName(implicit_cast->getType(), blackboard->astContext)];
            current = cur->cast(lc_type, CastOp::STATIC, stack->expr_map[last]);
        }

        last = x;
        stack->expr_map[x] = current;
        if (x == root) {
            translated = current;
            if (!translated) {
                x->dump();
                LUISA_ASSERT(translated, "BAD");
            }
        }
        return true;
    }

    clang::Stmt *last = nullptr;
    Stack *stack = nullptr;
    CXXBlackboard *blackboard = nullptr;
    luisa::shared_ptr<compute::detail::FunctionBuilder> cur = nullptr;
    clang::Stmt *root = nullptr;
    const luisa::compute::Expression *translated = nullptr;
};

bool FunctionDeclStmtHandler::recursiveVisit(clang::Stmt *stmt, luisa::shared_ptr<compute::detail::FunctionBuilder> cur, Stack &stack) {
    if (!stmt)
        return true;

    for (Stmt::child_iterator i = stmt->child_begin(), e = stmt->child_end(); i != e; ++i) {
        Stmt *currStmt = *i;
        if (!currStmt)
            continue;

        if (auto declStmt = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
            const DeclGroupRef declGroup = declStmt->getDeclGroup();
            for (auto decl : declGroup) {
                if (!decl) continue;

                if (auto *varDecl = dyn_cast<clang::VarDecl>(decl)) {
                    auto Ty = varDecl->getType();
                    auto lc_type = blackboard->type_map.find(GetTypeName(Ty, blackboard->astContext));
                    if (lc_type != blackboard->type_map.end()) {
                        /*
                        auto idx = cur->literal(Type::of<uint>(), uint(0));
                        auto buffer = cur->buffer(Type::buffer(lc_type->second));
                        cur->mark_variable_usage(buffer->variable().uid(), Usage::WRITE);
                        */
                        auto local = cur->local(lc_type->second);
                        auto str = luisa::string(varDecl->getName());
                        stack.locals[str] = local;

                        ExprTranslator v;
                        auto init = v.root = varDecl->getInit();
                        v.stack = &stack;
                        v.blackboard = blackboard;
                        v.cur = cur;
                        v.TraverseStmt(init);

                        cur->assign(local, v.translated);

                        // cur->call(CallOp::BUFFER_WRITE, {buffer, idx, local});
                    }
                }
            }
        }
        recursiveVisit(currStmt, cur, stack);
    }
    return true;
}

void FunctionDeclStmtHandler::run(const MatchFinder::MatchResult &Result) {
    // The matched 'if' statement was bound to 'ifStmt'.
    if (const auto *S = Result.Nodes.getNodeAs<clang::FunctionDecl>("FunctionDecl")) {
        bool ignore = false;
        auto params = S->parameters();
        for (auto Anno = S->specific_attr_begin<clang::AnnotateAttr>(); Anno != S->specific_attr_end<clang::AnnotateAttr>(); ++Anno) {
            ignore |= isIgnore(*Anno);
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
        if (!ignore) {
            S->dump();

            luisa::shared_ptr<compute::detail::FunctionBuilder> builder;
            Stmt *body = S->getBody();
            {
                if (S->isMain()) {
                    builder = blackboard->kernel_builder;
                } else {
                    builder = luisa::make_shared<luisa::compute::detail::FunctionBuilder>(luisa::compute::Function::Tag::CALLABLE);
                }
                luisa::compute::detail::FunctionBuilder::push(builder.get());
                builder->push_scope(builder->body());
                {
                    if (S->isMain()) {
                        builder->set_block_size(uint3(256, 1, 1));
                    }
                    Stack stack;
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
    // Matcher.addMatcher(stmt().bind("callExpr"), &HandlerForCallExpr);
}

ASTConsumer::~ASTConsumer() {
    for (auto &&[name, type] : blackboard.type_map) {
        std::cout << name << " - ";
        std::cout << type->description() << std::endl;
    }

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