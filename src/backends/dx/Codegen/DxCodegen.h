#pragma once

#include <vstl/Common.h>
#include <vstl/functional.h>
#include <ast/function.h>
#include <ast/expression.h>
#include <ast/statement.h>
#include <Shader/Shader.h>
#include <vstl/MD5.h>
#include <compile/definition_analysis.h>
using namespace luisa;
using namespace luisa::compute;
namespace toolhub::directx {
class StringStateVisitor;
class StructVariableTracker;
class StructGenerator;
struct CodegenResult {
    using Properties = vstd::vector<std::pair<vstd::string, Shader::Property>>;
    vstd::string result;
    Properties properties;
    uint64 bdlsBufferCount;
    vstd::MD5 md5;
    CodegenResult() {}
    template<typename A, typename B>
    CodegenResult(A &&a, B &&b, uint64 c, vstd::MD5 md5)
        : result(std::forward<A>(a)),
          properties(std::forward<B>(b)), bdlsBufferCount(c),
          md5(md5) {}
};
class CodegenUtility {

public:
    static constexpr vstd::string_view rayTypeDesc = "struct<16,array<float,3>,float,array<float,3>,float>"sv;
    static constexpr vstd::string_view hitTypeDesc = "struct<16,uint,uint,vector<float,2>>"sv;
    static uint IsBool(Type const &type);
    static void GetConstName(ConstantData const &data, vstd::string &str);
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
    static void GetFunctionName(CallExpr const *expr, vstd::string &result, StringStateVisitor &visitor);
    static void RegistStructType(Type const *type);

    static void CodegenFunction(
        Function func,
        vstd::string &result);
    static StructGenerator const *GetStruct(
        Type const *type);
    static void GenerateCBuffer(
        Function f,
        std::span<const Variable> vars,
        vstd::string &result);
    static vstd::optional<CodegenResult> Codegen(Function kernel);
    static vstd::string GetNewTempVarName();
};
class StringStateVisitor final : public StmtVisitor, public ExprVisitor {
    Function f;

public:
    DefinitionAnalysis::VariableSet *sharedVariables = nullptr;
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
    void visit(const Function &f);
    void visit(const CommentStmt *) override;
    StringStateVisitor(
        Function f,
        vstd::string &str);
    ~StringStateVisitor();

protected:
    vstd::string &str;
};
template<typename T>
struct PrintValue {};

template<>
struct PrintValue<float> {
    void operator()(float v, vstd::string &str) {
        if (std::isnan(v)) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Encountered with NaN.");
        }
        if (std::isinf(v)) {
            str.append(v < 0.0f ? "(-1.#INF)" : "(1.#INF)");
        } else {
            auto s = fmt::format("{}", v);
            str.append(s);
            if (s.find('.') == std::string_view::npos &&
                s.find('e') == std::string_view::npos) {
                str.append(".0");
            }
            str.append("f");
        }
    }
};
template<>
struct PrintValue<int> {
    void operator()(int v, vstd::string &str) {
        vstd::to_string(v, str);
    }
};
template<>
struct PrintValue<uint> {
    void operator()(uint v, vstd::string &str) {
        vstd::to_string(v, str);
        str << 'u';
    }
};

template<>
struct PrintValue<bool> {
    void operator()(bool v, vstd::string &str) {
        if (v)
            str << "true";
        else
            str << "false";
    }
};
template<typename EleType, uint64 N>
struct PrintValue<Vector<EleType, N>> {
    using T = Vector<EleType, N>;
    void PureRun(T v, vstd::string &varName) {
        for (uint i = 0u; i < N; ++i) {
            PrintValue<EleType> pv;
            pv(v[i], varName);
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