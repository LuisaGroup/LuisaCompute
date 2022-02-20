#pragma once

#include <vstl/Common.h>
#include <vstl/functional.h>
#include <ast/function.h>
#include <ast/expression.h>
#include <ast/statement.h>

using namespace luisa;
using namespace luisa::compute;
namespace lc::ispc {
class StringExprVisitor;
class CodegenUtility {
public:
    static constexpr uint64 INLINE_STMT_LIMIT = 5;
    static void GetConstName(ConstantData const &data, luisa::string &str);
    static void PrintFunction(Function func, luisa::string &str, uint3 blockSize);
    static void GetVariableName(Variable const &type, luisa::string &str);
    static void GetVariableName(Variable::Tag type, uint id, luisa::string &str);
    static void GetVariableName(Type::Tag type, uint id, luisa::string &str);
    static void GetTypeName(Type const &type, luisa::string &str);
    static void GetBasicTypeName(uint64 typeIndex, luisa::string &str);
    static void GetConstantStruct(ConstantData const &data, luisa::string &str);
    //static void
    static void GetCustomStruct(Type const &t, std::string_view strName, luisa::string &str);
    static void GetArrayStruct(Type const &t, std::string_view name, luisa::string &str);
    static void GetConstantData(ConstantData const &data, luisa::string &str);
    static size_t GetTypeSize(Type const &t);
    static size_t GetTypeAlign(Type const &t);
    static luisa::string GetBasicTypeName(uint64 typeIndex) {
        luisa::string s;
        GetBasicTypeName(typeIndex, s);
        return s;
    }
    static void GetFunctionDecl(Function func, luisa::string &str);
    static vstd::function<void(StringExprVisitor &)> GetFunctionName(CallExpr const *expr, luisa::string &result);
    static void ClearStructType();
    static void RegistStructType(Type const *type);
};
class CodegenGlobalData;
class VisitorBase {
public:
    CodegenGlobalData *ptr;
    VisitorBase(
        CodegenGlobalData *ptr)
        : ptr(ptr) {}
};
class StringExprVisitor final : public ExprVisitor, public VisitorBase {

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
    StringExprVisitor(
        luisa::string &str,
        CodegenGlobalData *ptr);
    ~StringExprVisitor();

protected:
    luisa::string &str;
};
class StringStateVisitor final : public StmtVisitor, public VisitorBase {
public:
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
        luisa::string &str,
        CodegenGlobalData *ptr);
    ~StringStateVisitor();
    uint64 StmtCount() const { return stmtCount; }

protected:
    luisa::string &str;
    uint64 stmtCount = 0;
};

template<typename T>
struct PrintValue;
template<>
struct PrintValue<float> {
    void operator()(float const &v, luisa::string &str) {
        if (std::isnan(v)) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Encountered with NaN.");
        }
        if (std::isinf(v)) {
            str.append(v < 0.0f ? "(-INFINITY_f)" : "(+INFINITY_f)");
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
    void operator()(int const &v, luisa::string &str) {
        vstd::to_string(v, str);
    }
};
template<>
struct PrintValue<uint> {
    void operator()(uint const &v, luisa::string &str) {
        vstd::to_string(v, str);
    }
};

template<>
struct PrintValue<bool> {
    void operator()(bool const &v, luisa::string &str) {
        if (v)
            str << "true";
        else
            str << "false";
    }
};
template<typename EleType, uint64 N>
struct PrintValue<Vector<EleType, N>> {
    using T = Vector<EleType, N>;
    void PureRun(T const &v, luisa::string &varName) {
        for (uint64 i = 0; i < N; ++i) {
            vstd::to_string(v[i], varName);
            varName += ',';
        }
        auto &&last = varName.end() - 1;
        if (*last == ',')
            varName.erase(last);
    }
    void operator()(T const &v, luisa::string &varName) {
        if constexpr (N > 1) {
            if constexpr (std::is_same_v<EleType, float>) {
                varName << "_float";
            } else if constexpr (std::is_same_v<EleType, uint>) {
                varName << "_uint";
            } else if constexpr (std::is_same_v<EleType, int>) {
                varName << "_int";
            } else if constexpr (std::is_same_v<EleType, bool>) {
                varName << "_bool";
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
    void operator()(T const &v, luisa::string &varName) {
        varName << "_float";
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

template<>
struct PrintValue<LiteralExpr::MetaValue> {
    void operator()(const LiteralExpr::MetaValue &s, luisa::string &varName) const noexcept {
        // TODO...
    }
};

}// namespace lc::ispc