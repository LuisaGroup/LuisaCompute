#pragma once
//#define USE_SPIRV
#include "vstl/common.h"
#include "vstl/functional.h"
#include "ast/function.h"
#include "ast/expression.h"
#include "ast/statement.h"
#include "vstl/md5.h"
#include "shader_property.h"
#include <raster/raster_state.h>
#include <core/logging.h>
#include <filesystem>
#include <core/binary_io.h>
using namespace luisa;
using namespace luisa::compute;
namespace toolhub::directx {
class StringStateVisitor;
class StructVariableTracker;
class StructGenerator;
struct CodegenStackData;
struct CodegenResult {
    using Properties = vstd::vector<Property>;
    vstd::string result;
    Properties properties;
    uint64 bdlsBufferCount = 0;
    uint64 immutableHeaderSize = 0;
    vstd::MD5 typeMD5;
    CodegenResult() {}
    CodegenResult(
        vstd::string &&result,
        Properties &&properties,
        uint64 bdlsBufferCount,
        uint64 immutableHeaderSize,
        vstd::MD5 typeMD5) : result(std::move(result)), properties(std::move(properties)), bdlsBufferCount(bdlsBufferCount), immutableHeaderSize(immutableHeaderSize), typeMD5(typeMD5) {}
    CodegenResult(CodegenResult const &) = delete;
    CodegenResult(CodegenResult &&) = default;
};
class CodegenUtility {

public:
    static uint IsBool(Type const &type);
    static bool GetConstName(uint64 hash, ConstantData const &data, vstd::string &str);
    static void GetVariableName(Variable const &type, vstd::string &str);
    static void GetVariableName(Variable::Tag type, uint id, vstd::string &str);
    static void GetTypeName(Type const &type, vstd::string &str, Usage usage);
    static void GetBasicTypeName(uint64 typeIndex, vstd::string &str);
    static void GetConstantStruct(ConstantData const &data, vstd::string &str);
    //static void
    static void GetConstantData(ConstantData const &data, vstd::string &str);
    static size_t GetTypeAlign(Type const &t);
    static size_t GetTypeSize(Type const &t);
    static vstd::string GetBasicTypeName(uint64 typeIndex) {
        vstd::string s;
        GetBasicTypeName(typeIndex, s);
        return s;
    }
    static void GetFunctionDecl(Function func, vstd::string &str);
    static void GetFunctionName(Function callable, vstd::string &result);
    static void GetFunctionName(CallExpr const *expr, vstd::string &result, StringStateVisitor &visitor);
    static void RegistStructType(Type const *type);

    static void CodegenFunction(
        Function func,
        vstd::string &result,
        bool cBufferNonEmpty);
    static void CodegenVertex(Function vert, vstd::string &result, bool cBufferNonEmpty, vstd::function<void(string &)> const &bindVertex);
    static void CodegenPixel(Function pixel, vstd::string &result, bool cBufferNonEmpty);
    static StructGenerator const *GetStruct(
        Type const *type);
    static bool IsCBufferNonEmpty(std::initializer_list<vstd::IRange<Variable> *> f);
    static bool IsCBufferNonEmpty(Function func);
    static vstd::MD5 GetTypeMD5(vstd::span<Type const *const> types);
    static vstd::MD5 GetTypeMD5(std::initializer_list<vstd::IRange<Variable> *> f);
    static vstd::MD5 GetTypeMD5(Function func);

    static void GenerateCBuffer(
        std::initializer_list<vstd::IRange<Variable> *> f,
        vstd::string &result);
    static void GenerateBindless(
        CodegenResult::Properties &properties,
        vstd::string &str);
    static void PreprocessCodegenProperties(CodegenResult::Properties &properties, vstd::string &varData, vstd::array<uint, 3> &registerCount, bool cbufferNonEmpty,
                                            bool isRaster);
    static void PostprocessCodegenProperties(CodegenResult::Properties &properties, vstd::string &finalResult);
    static void CodegenProperties(
        CodegenResult::Properties &properties,
        vstd::string &finalResult,
        vstd::string &varData,
        Function kernel,
        uint offset,
        vstd::array<uint, 3> &registerCount);
    static CodegenResult Codegen(Function kernel, luisa::compute::BinaryIO *internalDataPath);
    static CodegenResult RasterCodegen(
        MeshFormat const &meshFormat,
        Function vertFunc,
        Function pixelFunc,
        luisa::compute::BinaryIO *internalDataPath);
    static vstd::string ReadInternalHLSLFile(vstd::string_view name, luisa::compute::BinaryIO *ctx);
    static vstd::vector<char> ReadInternalHLSLFileByte(vstd::string_view name, luisa::compute::BinaryIO *ctx);
    /*
#ifdef USE_SPIRV
    static void GenerateBindlessSpirv(
        vstd::string &str);
    static CodegenStackData *StackData();
    static vstd::optional<vstd::string> CodegenSpirv(Function kernel, luisa::compute::BinaryIO*internalDataPath);
#endif*/
    static vstd::string GetNewTempVarName();
};
class StringStateVisitor final : public StmtVisitor, public ExprVisitor {
    Function f;
    struct Scope {
        StringStateVisitor *self;
        Scope(StringStateVisitor *self);
        ~Scope();
    };

public:
    luisa::unordered_map<uint64, Variable> *sharedVariables = nullptr;
    void visit(const UnaryExpr *expr) override;
    void visit(const BinaryExpr *expr) override;
    void visit(const MemberExpr *expr) override;
    void visit(const AccessExpr *expr) override;
    void visit(const LiteralExpr *expr) override;
    void visit(const RefExpr *expr) override;
    void visit(const CallExpr *expr) override;
    void visit(const CastExpr *expr) override;
    void visit(const ConstantExpr *expr) override;

    void visit(const BreakStmt *) override;
    void visit(const ContinueStmt *) override;
    void visit(const ReturnStmt *) override;
    void visit(const ScopeStmt *) override;
    void visit(const IfStmt *) override;
    void visit(const LoopStmt *) override;
    void visit(const ExprStmt *) override;
    void visit(const SwitchStmt *) override;
    void visit(const SwitchCaseStmt *) override;
    void visit(const SwitchDefaultStmt *) override;
    void visit(const AssignStmt *) override;
    void visit(const ForStmt *) override;
    void VisitFunction(Function func);
    void visit(const CommentStmt *) override;
    StringStateVisitor(
        Function f,
        vstd::string &str);
    ~StringStateVisitor();

protected:
    vstd::string &str;
};
template<typename T>
struct PrintValue;
template<>
struct PrintValue<float> {
    void operator()(float const &v, vstd::string &str) {
        if (std::isnan(v)) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Encountered with NaN.");
        }
        if (std::isinf(v)) [[unlikely]] {
            str.append(v < 0.0f ? "(-INFINITY_f)" : "(INFINITY_f)");
        } else {
            vstd::to_string(v, str);
        }
    }
};
template<>
struct PrintValue<int> {
    void operator()(int const &v, vstd::string &str) {
        vstd::to_string(v, str);
    }
};
template<>
struct PrintValue<uint> {
    void operator()(uint const &v, vstd::string &str) {
        vstd::to_string(v, str);
        str << 'u';
    }
};

template<>
struct PrintValue<bool> {
    void operator()(bool const &v, vstd::string &str) {
        if (v)
            str << "true";
        else
            str << "false";
    }
};
template<typename EleType, uint64 N>
struct PrintValue<Vector<EleType, N>> {
    using T = Vector<EleType, N>;
    void PureRun(T const &v, vstd::string &varName) {
        for (uint64 i = 0; i < N; ++i) {
            vstd::to_string(v[i], varName);
            varName += ',';
        }
        auto &&last = varName.end() - 1;
        if (*last == ',')
            varName.erase(last);
    }
    void operator()(T const &v, vstd::string &varName) {
        if constexpr (N > 1) {
            if constexpr (std::is_same_v<EleType, float>) {
                varName << "float";
            } else if constexpr (std::is_same_v<EleType, uint>) {
                varName << "uint";
            } else if constexpr (std::is_same_v<EleType, int>) {
                varName << "int";
            } else if constexpr (std::is_same_v<EleType, bool>) {
                varName << "bool";
            }
            vstd::to_string(N, varName);
            varName << '(';
            PureRun(v, varName);
            varName << ')';
        } else {
            PureRun(v, varName);
        }
    }
};

template<uint64 N>
struct PrintValue<Matrix<N>> {
    using T = Matrix<N>;
    using EleType = float;
    void operator()(T const &v, vstd::string &varName) {
        varName << "make_float";
        auto ss = vstd::to_string(N);
        varName << ss << 'x' << ss << '(';
        PrintValue<Vector<EleType, N>> vecPrinter;
        for (uint64 i = 0; i < N; ++i) {
            vecPrinter.PureRun(v[i], varName);
            varName += ',';
        }
        auto &&last = varName.end() - 1;
        if (*last == ',')
            varName.erase(last);
        varName << ')';
    }
};

}// namespace toolhub::directx