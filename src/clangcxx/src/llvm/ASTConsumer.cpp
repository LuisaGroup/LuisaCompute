#include "ASTConsumer.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Expr.h"
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

void RecordDeclStmtHandler::run(const MatchFinder::MatchResult &Result) {
    auto &kernel_builder = consumer->kernel_builder;
    if (const auto *S = Result.Nodes.getNodeAs<clang::RecordDecl>("RecordDecl")) {
        bool ignore = false;
        for (auto Anno = S->specific_attr_begin<clang::AnnotateAttr>(); Anno != S->specific_attr_end<clang::AnnotateAttr>(); ++Anno) {
            ignore = isIgnore(*Anno) || isBuiltinType(*Anno);
        }
        if (!ignore) {
            // S->dump();
            luisa::vector<const luisa::compute::Type *> types;
            for (auto f = S->field_begin(); f != S->field_end(); f++) {
                auto fType = f->getType();
                if (auto builtin = fType->getAs<clang::BuiltinType>()) {
                    std::cout << fType.getAsString() << std::endl;
                    switch (builtin->getKind()) {
                            /*
                        case (BuiltinType::Kind::SChar): { types.emplace_back(Type::of<signed char>()); } break; 
                        case (BuiltinType::Kind::Char_S): { types.emplace_back(Type::of<signed char>()); } break; 
                        case (BuiltinType::Kind::Char8): { types.emplace_back(Type::of<signed char>()); } break; 

                        case (BuiltinType::Kind::UChar): { types.emplace_back(Type::of<unsigned char>()); } break; 
                        case (BuiltinType::Kind::Char_U): { types.emplace_back(Type::of<unsigned char>()); } break; 
                        
                        case (BuiltinType::Kind::Char16): { types.emplace_back(Type::of<char16_t>()); } break; 
                        */

                        case (BuiltinType::Kind::Bool): {
                            types.emplace_back(Type::of<bool>());
                        } break;

                        case (BuiltinType::Kind::UShort): {
                            types.emplace_back(Type::of<uint16_t>());
                        } break;
                        case (BuiltinType::Kind::UInt): {
                            types.emplace_back(Type::of<uint32_t>());
                        } break;
                        case (BuiltinType::Kind::ULong): {
                            types.emplace_back(Type::of<uint32_t>());
                        } break;
                        case (BuiltinType::Kind::ULongLong): {
                            types.emplace_back(Type::of<uint64_t>());
                        } break;

                        case (BuiltinType::Kind::Short): {
                            types.emplace_back(Type::of<int16_t>());
                        } break;
                        case (BuiltinType::Kind::Int): {
                            types.emplace_back(Type::of<int32_t>());
                        } break;
                        case (BuiltinType::Kind::Long): {
                            types.emplace_back(Type::of<int32_t>());
                        } break;
                        case (BuiltinType::Kind::LongLong): {
                            types.emplace_back(Type::of<int64_t>());
                        } break;

                        case (BuiltinType::Kind::Float): {
                            types.emplace_back(Type::of<float>());
                        } break;
                        case (BuiltinType::Kind::Double): {
                            types.emplace_back(Type::of<double>());
                        } break;

                        default: {
                            luisa::log_error("unsupported type: {}, kind {}", fType.getAsString(), builtin->getKind());
                            break;
                        }
                    }
                } else {
                    auto Ty = fType;
                    clang::RecordDecl *recordDecl = fType->getAsRecordDecl();
                    if (!recordDecl) {
                        if (const clang::TypedefType *TDT = fType->getAs<clang::TypedefType>()) {
                            Ty = TDT->getDecl()->getUnderlyingType();
                            recordDecl = Ty->getAsRecordDecl();
                        }
                    }
                    if (recordDecl) {
                        bool ext_builtin = false;
                        llvm::StringRef builtin_type_name = {};
                        for (auto Anno = recordDecl->specific_attr_begin<clang::AnnotateAttr>();
                             Anno != recordDecl->specific_attr_end<clang::AnnotateAttr>(); ++Anno) {
                            if (ext_builtin = isBuiltinType(*Anno)) {
                                builtin_type_name = getBuiltinTypeName(*Anno);
                            }
                        }

                        if (builtin_type_name == "vec") {
                            if (auto TST = Ty->getAs<TemplateSpecializationType>()) {
                                auto Arguments = TST->template_arguments();
                                if (auto EType = Arguments[0].getAsType()->getAs<clang::BuiltinType>()) {
                                    clang::Expr::EvalResult Result;
                                    Arguments[1].getAsExpr()->EvaluateAsConstantExpr(Result, *consumer->astContext);
                                    auto N = Result.Val.getInt().getExtValue();
                                    // TST->dump();
                                    switch (EType->getKind()) {
#define CASE_VEC_TYPE(type)                                                                                    \
    switch (N) {                                                                                               \
        case 2: {                                                                                              \
            types.emplace_back(Type::of<type##2>());                                                           \
        } break;                                                                                               \
        case 3: {                                                                                              \
            types.emplace_back(Type::of<type##3>());                                                           \
        } break;                                                                                               \
        case 4: {                                                                                              \
            types.emplace_back(Type::of<type##4>());                                                           \
        } break;                                                                                               \
        default: {                                                                                             \
            luisa::log_error("unsupported type: {}, kind {}, N {}", fType.getAsString(), EType->getKind(), N); \
        } break;                                                                                               \
    }
                                        case (BuiltinType::Kind::Bool): {
                                            CASE_VEC_TYPE(bool)
                                        } break;
                                        case (BuiltinType::Kind::Float): {
                                            CASE_VEC_TYPE(float)
                                        } break;
                                        case (BuiltinType::Kind::Long):
                                        case (BuiltinType::Kind::Int): {
                                            CASE_VEC_TYPE(int)
                                        } break;
                                        case (BuiltinType::Kind::ULong):
                                        case (BuiltinType::Kind::UInt): {
                                            CASE_VEC_TYPE(uint)
                                        } break;
                                        case (BuiltinType::Kind::Double):
                                        default: {
                                            luisa::log_error("unsupported type: {}, kind {}", fType.getAsString(), EType->getKind());
                                        } break;
#undef CASE_VEC_TYPE
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // align
            uint64_t alignment = 4;
            for (auto ft : types) {
                alignment = std::max(alignment, ft->alignment());
            }
            consumer->ttt = Type::structure(alignment, types);
            std::cout << consumer->ttt->description() << std::endl;
            // kernel_builder->
        }
    }
}

struct NVIDIA {
    int n;
};

struct Fuck {
    int a;
    NVIDIA b;
    std::array<float, 4> c;
};

bool FunctionDeclStmtHandler::recursiveVisit(clang::Stmt *stmt, luisa::shared_ptr<compute::detail::FunctionBuilder> cur) {
    if (!stmt)
        return true;

    for (Stmt::child_iterator i = stmt->child_begin(), e = stmt->child_end(); i != e; ++i) {
        Stmt *currStmt = *i;
        if (!currStmt)
            continue;

        // currStmt->dump();
        // std::cout << std::endl;

        if (auto declStmt = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
            const DeclGroupRef declGroup = declStmt->getDeclGroup();
            for (auto decl : declGroup) {
                if (decl && isa<clang::VarDecl>(decl)) {
                    auto *varDecl = (VarDecl *)decl;
                    auto at = varDecl->getType();
                    auto t = at->getContainedAutoType()->getCanonicalTypeInternal();
                    {
                        std::cout << at.getAsString() << std::endl;
                        std::cout << t.getAsString() << std::endl;
                        const clang::Expr *expr = varDecl->getInit();
                    }
                    auto idx = cur->literal(Type::of<uint>(), uint(0));
                    auto buffer = cur->buffer(Type::buffer(consumer->ttt));
                    cur->mark_variable_usage(buffer->variable().uid(), Usage::WRITE);
                    auto local = cur->local(consumer->ttt);
                    cur->call(CallOp::BUFFER_WRITE, {buffer, idx, local});
                }
            }
        }

        recursiveVisit(currStmt, cur);
    }
    return true;
}

void FunctionDeclStmtHandler::run(const MatchFinder::MatchResult &Result) {
    // The matched 'if' statement was bound to 'ifStmt'.
    if (const auto *S = Result.Nodes.getNodeAs<clang::FunctionDecl>("FunctionDecl")) {
        // S->dump();
        bool ignore = false;
        auto params = S->parameters();
        for (auto Anno = S->specific_attr_begin<clang::AnnotateAttr>(); Anno != S->specific_attr_end<clang::AnnotateAttr>(); ++Anno) {
            ignore = isIgnore(*Anno);
        }
        if (!ignore) {
            // std::cout << S->getName().data() << std::endl;
            luisa::shared_ptr<compute::detail::FunctionBuilder> builder;
            Stmt *body = S->getBody();
            {
                if (S->isMain()) {
                    builder = consumer->kernel_builder;
                } else {
                    builder = luisa::make_shared<luisa::compute::detail::FunctionBuilder>(luisa::compute::Function::Tag::CALLABLE);
                }
                luisa::compute::detail::FunctionBuilder::push(builder.get());
                builder->push_scope(builder->body());
                {
                    if (S->isMain()) {
                        builder->set_block_size(uint3(256, 1, 1));
                    }
                    recursiveVisit(body, builder);
                }
                builder->pop_scope(builder->body());
                luisa::compute::detail::FunctionBuilder::pop(builder.get());
            }
        }
    }
}

ASTConsumer::ASTConsumer(std::string OutputPath, luisa::compute::Device *device, compute::ShaderOption option)
    : OutputPath(std::move(OutputPath)), device(device), option(option) {

    kernel_builder = luisa::make_shared<luisa::compute::detail::FunctionBuilder>(luisa::compute::Function::Tag::KERNEL);

    HandlerForTypeDecl.consumer = this;
    Matcher.addMatcher(recordDecl(
                           isDefinition(),
                           unless(isExpansionInSystemHeader()))
                           .bind("RecordDecl"),
                       &HandlerForTypeDecl);

    HandlerForFuncionDecl.consumer = this;
    Matcher.addMatcher(functionDecl(
                           isDefinition(),
                           unless(isExpansionInSystemHeader()))
                           .bind("FunctionDecl"),
                       &HandlerForFuncionDecl);
    // Matcher.addMatcher(stmt().bind("callExpr"), &HandlerForCallExpr);
}

ASTConsumer::~ASTConsumer() {
    LUISA_INFO("{}", Type::of<Fuck>()->description());

    device->impl()->create_shader(
        luisa::compute::ShaderOption{
            .compile_only = true,
            .name = "test.bin"},
        luisa::compute::Function{kernel_builder.get()});
}

void ASTConsumer::HandleTranslationUnit(clang::ASTContext &Context) {
    // 1. collect
    astContext = &Context;
    Matcher.matchAST(Context);
}

void Remove(std::string &str, const std::string &remove_str) {
    for (size_t i; (i = str.find(remove_str)) != std::string::npos;)
        str.replace(i, remove_str.length(), "");
}

std::string GetTypeName(clang::QualType type, clang::ASTContext *ctx) {
    type = type.getCanonicalType();
    auto baseName = type.getAsString(ctx->getLangOpts());
    Remove(baseName, "struct ");
    Remove(baseName, "class ");
    return baseName;
}

}// namespace luisa::clangcxx