#pragma once

#include <vstl/Common.h>
#include <vstl/functional.h>
#include <ast/function.h>
#include <ast/expression.h>
#include <ast/statement.h>
#include <Shader/Shader.h>
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
    template<typename A, typename B>
    CodegenResult(A &&a, B &&b)
        : result(std::forward<A>(a)),
          properties(std::forward<B>(b)) {}
};
class CodegenUtility {

public:
    static constexpr uint64 INLINE_STMT_LIMIT = 5;
    static StructVariableTracker *GetTracker();
    static void AddScope(int32 v);
    static int64 GetScope();
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
    void visit(const MetaStmt *stmt) override;
    void visit(const CommentStmt *) override;
    StringStateVisitor(
        Function f,
        vstd::string &str);
    ~StringStateVisitor();

protected:
    vstd::string &str;
    size_t lastIdx = 0;
};
template<typename T>
struct PrintValue;
template<>
struct PrintValue<float> {
    void operator()(float const &v, vstd::string &str) {
        if (std::isnan(v)) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Encountered with NaN.");
        }
        if (std::isinf(v)) {
            str.append(v < 0.0f ? "(-INFINITY_f)" : "(INFINITY_f)");
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
        if constexpr (N == 3) {
            varName << "make_float4x3("sv;
        } else {
            varName << "float";
            auto ss = vstd::to_string(N);
            varName << ss << 'x' << ss << '(';
        }
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

template<>
struct PrintValue<LiteralExpr::MetaValue> {
    void operator()(const LiteralExpr::MetaValue &s, vstd::string &varName) const noexcept {
        // TODO...
    }
};

}// namespace toolhub::directx