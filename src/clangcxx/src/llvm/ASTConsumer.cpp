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
                    auto buffer = cur->buffer(Type::of<Buffer<NVIDIA>>());
                    cur->mark_variable_usage(buffer->variable().uid(), Usage::WRITE);
                    auto local = cur->local(Type::of<NVIDIA>());
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
        if (auto Anno = S->getAttr<clang::AnnotateAttr>()) {
            ignore = isIgnore(Anno);
        }
        if (!ignore) {
            // std::cout << S->getName().data() << std::endl;
            luisa::shared_ptr<compute::detail::FunctionBuilder> builder;
            Stmt *body = S->getBody();
            {
                if (S->isMain()) {
                    builder = luisa::make_shared<luisa::compute::detail::FunctionBuilder>(luisa::compute::Function::Tag::KERNEL);
                    kernel_builder = builder;
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
        luisa::compute::Function{HandlerForFuncionDecl.kernel_builder.get()});
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