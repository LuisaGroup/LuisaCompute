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
        blackboard->type_map[luisa::string(decl->getQualifiedNameAsString())] = _type;
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
                    Arguments[1].getAsExpr()->EvaluateAsConstantExpr(Result, *blackboard->astContext);
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
        } else if (builtin_type_name == "array") {

        } else {
            luisa::log_error("unsupported builtin type: {} as a field", luisa::string(builtin_type_name));
        }
    }
    if (_type) {
        blackboard->type_map[luisa::string(decl->getQualifiedNameAsString())] = _type;
    }
    return _type;
}

const luisa::compute::Type *RecordDeclStmtHandler::RecordAsStuctureType(const clang::QualType Ty, const clang::RecordDecl *decl) {
    const luisa::compute::Type *_type = nullptr;
    auto field_qualified_type_name = luisa::string(decl->getQualifiedNameAsString());
    auto iter = blackboard->type_map.find(field_qualified_type_name);
    if (iter != blackboard->type_map.end()) {
        _type = iter->second;
    }
    if (_type) {
        blackboard->type_map[luisa::string(decl->getQualifiedNameAsString())] = _type;
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
        // 2.0 RESOLVE TO RECORD
        clang::RecordDecl *recordDecl = Ty->getAsRecordDecl();
        if (!recordDecl) {
            if (const auto *TDT = Ty->getAs<clang::TypedefType>()) {
                Ty = TDT->getDecl()->getUnderlyingType();
                recordDecl = Ty->getAsRecordDecl();
            } else if (const auto *TST = Ty->getAs<clang::TemplateSpecializationType>()) {
                recordDecl = TST->getAsRecordDecl();
            } else if (const auto *AT = Ty->getAsArrayTypeUnsafe()) {
                luisa::log_error("array type is not supported: [{}] in type [{}]", Ty.getAsString(), decl->getNameAsString());
            } else {
                Ty.dump();
            }
        }
        // 2.1 EMPLACE RECORD
        if (recordDecl) {
            // 2.1.1 AS BUILTIN
            if (!_type) {
                _type = RecordAsBuiltinType(Ty, recordDecl);
            }
            // 2.1.2 AS STRUCTURE
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
            ignore = isIgnore(*Anno) || isBuiltinType(*Anno);
        }
        if (!ignore) {
            luisa::vector<const luisa::compute::Type *> types;
            for (auto f = S->field_begin(); f != S->field_end(); f++) {
                auto Ty = f->getType();
                if (!TryEmplaceFieldType(Ty, S, types)) {
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
            auto qualified_name = S->getQualifiedNameAsString();
            blackboard->type_map[luisa::string(qualified_name)] = lc_type;
            // std::cout << lc_type->description() << std::endl;
        }
    }
}

bool FunctionDeclStmtHandler::recursiveVisit(clang::Stmt *stmt, luisa::shared_ptr<compute::detail::FunctionBuilder> cur) {
    if (!stmt)
        return true;

    for (Stmt::child_iterator i = stmt->child_begin(), e = stmt->child_end(); i != e; ++i) {
        Stmt *currStmt = *i;
        if (!currStmt)
            continue;

        if (auto declStmt = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
            const DeclGroupRef declGroup = declStmt->getDeclGroup();
            for (auto decl : declGroup) {
                if (decl && isa<clang::VarDecl>(decl)) {
                    auto *varDecl = (VarDecl *)decl;
                    /*
                    auto at = varDecl->getType();
                    auto t = at->getContainedAutoType()->getCanonicalTypeInternal();
                    {
                        std::cout << at.getAsString() << std::endl;
                        std::cout << t.getAsString() << std::endl;
                        const clang::Expr *expr = varDecl->getInit();
                    }
                    */
                    auto idx = cur->literal(Type::of<uint>(), uint(0));
                    auto &type = blackboard->type_map["luisa::shader::NVIDIA"];
                    auto buffer = cur->buffer(Type::buffer(type));
                    cur->mark_variable_usage(buffer->variable().uid(), Usage::WRITE);
                    auto local = cur->local(Type::array(type, 2));
                    cur->access(Type::array(type, 2), local, idx);
                    cur->call(CallOp::BUFFER_WRITE, {buffer, idx, cur->access(Type::array(type, 2), local, idx)});
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