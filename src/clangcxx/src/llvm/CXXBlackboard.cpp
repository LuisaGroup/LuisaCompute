#include "ASTConsumer.h"
#include "defer.hpp"
#include "AttributeHelpers.hpp"
#include <luisa/vstl/common.h>
#include <luisa/dsl/sugar.h>

namespace luisa::clangcxx {

using namespace clang;
using namespace clang::ast_matchers;
using namespace luisa::compute;

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

CXXBlackboard::CXXBlackboard() {
    using namespace luisa;
    using namespace luisa::compute;

    call_ops_map.reserve(call_op_count);
    for (auto i : vstd::range(call_op_count)) {
        const auto op = (CallOp)i;
        call_ops_map.emplace(luisa::to_string(op), op);
    }
#define LC_CLANGCXX_REG_BUILTIN(_func_name)                                              \
    call_ops_map.emplace(                                                                \
        #_func_name,                                                                     \
        [](compute::detail::FunctionBuilder *func_builder) -> const compute::RefExpr * { \
            return func_builder->_func_name();                                           \
        });
    LC_CLANGCXX_REG_BUILTIN(dispatch_id)
    LC_CLANGCXX_REG_BUILTIN(dispatch_size)
    LC_CLANGCXX_REG_BUILTIN(block_id)
    LC_CLANGCXX_REG_BUILTIN(thread_id)
    LC_CLANGCXX_REG_BUILTIN(kernel_id)
    LC_CLANGCXX_REG_BUILTIN(warp_lane_count)
    LC_CLANGCXX_REG_BUILTIN(warp_lane_id)
    LC_CLANGCXX_REG_BUILTIN(object_id)
#undef LC_CLANGCXX_REG_BUILTIN

    constexpr auto bin_op_count = to_underlying(BinaryOp::NOT_EQUAL) + 1u;
    bin_ops_map.reserve(bin_op_count);
    for (auto i : vstd::range(bin_op_count)) {
        const auto op = (BinaryOp)i;
        bin_ops_map.emplace(luisa::to_string(op), op);
    }
}

CXXBlackboard::~CXXBlackboard() {
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

auto CXXBlackboard::FindCallOp(const luisa::string_view &name) -> BuiltinCallCmd {
    auto iter = call_ops_map.find(name);
    if (iter) {
        return iter.value();
    }
    luisa::log_error("unfound call op: {}", name);
    return CallOp::ASIN;
}

auto CXXBlackboard::FindBinOp(const luisa::string_view &name) -> luisa::compute::BinaryOp {
    auto iter = bin_ops_map.find(name);
    if (iter) {
        return iter.value();
    }
    luisa::log_error("unfound call op: {}", name);
    return luisa::compute::BinaryOp::ADD;
}

void CXXBlackboard::commentSourceLoc(luisa::shared_ptr<compute::detail::FunctionBuilder> fb, const luisa::string &prefix, const clang::SourceLocation &loc) {
    const auto &SM = astContext->getSourceManager();
    auto RawLocString = loc.printToString(SM);
    luisa::string fmt = prefix + ", at {}";
    fb->comment_(luisa::format(fmt, RawLocString.data()));
}

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
        } else if (auto cxxFor = llvm::dyn_cast<clang::ForStmt>(x)) {
            return Commenter(
                [=] { commentSourceLoc(fb, "BEGIN FOR", x->getBeginLoc()); },
                [=] { commentSourceLoc(fb, "END FOR", x->getBeginLoc()); });
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
    return {[]() {}, []() {}};
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
    const luisa::compute::Type *_type = nullptr;
    bool ext_builtin = false;
    llvm::StringRef builtin_type_name = {};
    if (auto decl = GetRecordDeclFromQualType(Ty)) {
        for (auto Anno = decl->specific_attr_begin<clang::AnnotateAttr>();
             Anno != decl->specific_attr_end<clang::AnnotateAttr>(); ++Anno) {
            if (ext_builtin = isBuiltinType(*Anno)) {
                builtin_type_name = getBuiltinTypeName(*Anno);
            }
        }
    }
    // TODO: REFACTOR THIS (TSD)
    if (auto TSD = GetClassTemplateSpecializationDecl(Ty)) {
        auto decl = TSD->getSpecializedTemplate()->getTemplatedDecl();
        for (auto Anno = decl->specific_attr_begin<clang::AnnotateAttr>();
             Anno != decl->specific_attr_end<clang::AnnotateAttr>(); ++Anno) {
            if (ext_builtin = isBuiltinType(*Anno)) {
                builtin_type_name = getBuiltinTypeName(*Anno);
            }
        }
    }

    if (ext_builtin) {
        const auto is_image = builtin_type_name.startswith("image");
        const auto is_volume = builtin_type_name.startswith("volume");
        const auto is_buffer = builtin_type_name.startswith("buffer");
        if (builtin_type_name == "vec") {
            if (auto TSD = GetClassTemplateSpecializationDecl(Ty)) {
                auto &Arguments = TSD->getTemplateArgs();
                if (auto EType = Arguments.get(0).getAsType()->getAs<clang::BuiltinType>()) {
                    clang::Expr::EvalResult Result;
                    auto N = Arguments.get(1).getAsIntegral().getLimitedValue();
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
            } else {
                Ty->dump();
            }
        } else if (builtin_type_name == "array") {
            if (auto TSD = GetClassTemplateSpecializationDecl(Ty)) {
                auto &Arguments = TSD->getTemplateArgs();
                clang::Expr::EvalResult Result;
                auto N = Arguments.get(1).getAsIntegral().getLimitedValue();
                if (auto lc_type = FindOrAddType(Arguments[0].getAsType(), astContext)) {
                    _type = Type::array(lc_type, N);
                } else {
                    luisa::log_error("unfound array element type: {}", Arguments[0].getAsType().getAsString());
                }
            }
        } else if (builtin_type_name == "matrix") {
            if (auto TSD = GetClassTemplateSpecializationDecl(Ty)) {
                auto &Arguments = TSD->getTemplateArgs();
                auto N = Arguments.get(0).getAsIntegral().getLimitedValue();
                _type = Type::matrix(N);
            }
        } else if (builtin_type_name == "ray_query_all") {
            _type = Type::custom("LC_RayQueryAll");
        } else if (builtin_type_name == "ray_query_any") {
            _type = Type::custom("LC_RayQueryAny");
        } else if (builtin_type_name == "accel") {
            _type = Type::of<luisa::compute::Accel>();
        } else if (is_image || is_buffer || is_volume) {
            if (auto TSD = GetClassTemplateSpecializationDecl(Ty)) {
                auto &Arguments = TSD->getTemplateArgs();
                if (auto lc_type = FindOrAddType(Arguments[0].getAsType(), astContext)) {
                    if (is_buffer)
                        _type = Type::buffer(lc_type);
                    if (is_image)
                        _type = Type::texture(lc_type, 2);
                    if (is_volume)
                        _type = Type::texture(lc_type, 3);
                } else {
                    luisa::log_error("unfound {} element type: {}",
                                     is_buffer ? "buffer" : is_image ? "image" :
                                                                       "volume",
                                     Arguments[0].getAsType().getAsString());
                }
            }
        } else {
            Ty.dump();
            luisa::log_error("ilegal builtin type: {}", luisa::string(builtin_type_name));
        }

        if (!_type) {
            Ty.dump();
            luisa::log_error("unsupported builtin type: {}", luisa::string(builtin_type_name));
        }
    }
    if (_type) {
        registerType(Ty, astContext, _type);
    }
    return _type;
}

const luisa::compute::Type *CXXBlackboard::RecordAsStuctureType(const clang::QualType Ty) {
    if (Ty->isUnionType())
        return nullptr;
    else if (const luisa::compute::Type *_type = findType(Ty, astContext)) {
        return _type;
    } else {
        auto S = GetRecordDeclFromQualType(Ty);
        bool ignore = (S->getTypeForDecl()->getTypeClass() == clang::Type::InjectedClassName); 
        bool is_builtin = false;
        for (auto Anno = S->specific_attr_begin<clang::AnnotateAttr>();
             Anno != S->specific_attr_end<clang::AnnotateAttr>(); ++Anno) {
            is_builtin |= isBuiltinType(*Anno);
            ignore |= isIgnore(*Anno) || is_builtin;
        }
        if (ignore) return nullptr;

        for (auto f : S->fields()) {
            auto Ty = f->getType();
            if (Ty->isTemplateTypeParmType())
                return nullptr;
            if (auto decl = GetRecordDeclFromQualType(Ty, false);
                decl && decl->isTemplateDecl() && !decl->isTemplated())
                return nullptr;
        }

        luisa::vector<const luisa::compute::Type *> types;
        if (!S->isLambda() && !isSwizzle(S)) {// ignore lambda generated capture fields
            for (auto f = S->field_begin(); f != S->field_end(); f++) {
                auto Ty = f->getType();
                if (!tryEmplaceFieldType(Ty, S, types)) {
                    S->dump();
                    luisa::log_error("unsupported field type [{}] in type [{}]", Ty.getAsString(), S->getNameAsString());
                }
            }
            if (!is_builtin && S->field_empty()) {
                Ty->dump();
                luisa::log_error("empty struct [{}] detected!", Ty.getAsString());
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
    clang::QualType Ty = Qt.getNonReferenceType().getDesugaredType(*astContext);

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
            Qt->dump();
            luisa::log_error("unsupported & unresolved type [{}]", Ty.getAsString());
        }
    }
    if (!_type) {
        if (isSwizzle(GetRecordDeclFromQualType(Qt)))
            luisa::log_error("swizzle helper type instantiation detected! please use explicit vector types!");
        else {
            Ty->dump();
            luisa::log_error("unsupported type [{}]", Ty.getAsString());
        }
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
}// namespace luisa::clangcxx