#pragma once
//#define USE_SPIRV
#include <luisa/vstl/common.h>
#include <luisa/vstl/functional.h>
#include <luisa/ast/function.h>
#include <luisa/ast/expression.h>
#include <luisa/ast/statement.h>
#include <luisa/vstl/md5.h>
#include "shader_property.h"
#include <luisa/runtime/raster/raster_state.h>
#include <luisa/core/logging.h>
#include <luisa/core/binary_io.h>
#include "string_builder.h"
namespace lc::hlsl {
using namespace luisa;
using namespace luisa::compute;
class StringStateVisitor;
class StructVariableTracker;
class StructGenerator;
struct CodegenStackData;
struct CodegenResult {
    using Properties = vstd::vector<Property>;
    vstd::StringBuilder result;
    Properties properties;
    vstd::vector<std::pair<vstd::string, Type const *>> printers;
    bool useTex2DBindless;
    bool useTex3DBindless;
    bool useBufferBindless;
    uint64 immutableHeaderSize = 0;
    vstd::MD5 typeMD5;
    CodegenResult() {}
    CodegenResult(
        vstd::StringBuilder &&result,
        vstd::vector<std::pair<vstd::string, Type const *>> &&printers,
        Properties &&properties,
        bool useTex2DBindless,
        bool useTex3DBindless,
        bool useBufferBindless,
        uint64 immutableHeaderSize,
        vstd::MD5 typeMD5) : result(std::move(result)), properties(std::move(properties)), printers(std::move(printers)), useTex2DBindless{useTex2DBindless}, useTex3DBindless{useTex3DBindless}, useBufferBindless{useBufferBindless}, immutableHeaderSize(immutableHeaderSize), typeMD5(typeMD5) {}
    CodegenResult(CodegenResult const &) = delete;
    CodegenResult(CodegenResult &&) = default;
};
struct RegisterIndexer;
class CodegenUtility {
    vstd::unique_ptr<CodegenStackData> opt{};
    vstd::unordered_map<vstd::string, std::pair<vstd::string, Type const*>> attributes;
public:
#ifdef USE_SPIRV
    CodegenStackData *StackData() const;
#endif
    CodegenUtility();
    ~CodegenUtility();
    uint IsBool(Type const &type);
    bool GetConstName(uint64 hash, ConstantData const &data, vstd::StringBuilder &str);
    void GetVariableName(Variable const &type, vstd::StringBuilder &str);
    void GetVariableName(Variable::Tag type, uint id, vstd::StringBuilder &str);
    void GetTypeName(Type const &type, vstd::StringBuilder &str, Usage usage, bool local_var = true);
    void GetFunctionDecl(Function func, vstd::StringBuilder &str);
    void GetFunctionName(Function callable, vstd::StringBuilder &result);
    void GetFunctionName(CallExpr const *expr, vstd::StringBuilder &result, StringStateVisitor &visitor);
    void RegistStructType(Type const *type);

    void CodegenFunction(
        Function func,
        vstd::StringBuilder &result,
        bool cBufferNonEmpty);
    void CodegenVertex(Function vert, vstd::StringBuilder &result, bool cBufferNonEmpty);
    void CodegenPixel(Function pixel, vstd::StringBuilder &result, bool cBufferNonEmpty);
    bool IsCBufferNonEmpty(std::initializer_list<vstd::IRange<Variable> *> f);
    bool IsCBufferNonEmpty(Function func);
    static vstd::MD5 GetTypeMD5(vstd::span<Type const *const> types);
    static vstd::MD5 GetTypeMD5(std::initializer_list<vstd::IRange<Variable> *> f);
    static vstd::MD5 GetTypeMD5(Function func);

    void GenerateCBuffer(
        std::initializer_list<vstd::IRange<Variable> *> f,
        vstd::StringBuilder &result,
        uint &bind_count);
    void GenerateBindless(
        CodegenResult::Properties &properties,
        vstd::StringBuilder &str,
        bool isSpirV,
        uint &bind_count);
    void PreprocessCodegenProperties(
        CodegenResult::Properties &properties,
        vstd::StringBuilder &varData,
        RegisterIndexer &registerCount,
        bool cbufferNonEmpty, bool isRaster, bool isSpirv, uint &bind_count);
    void PostprocessCodegenProperties(vstd::StringBuilder &finalResult, bool use_autodiff);
    void CodegenProperties(
        CodegenResult::Properties &properties,
        vstd::StringBuilder &varData,
        Function kernel,
        uint offset,
        RegisterIndexer &registerCount,
        uint &bind_count);
    CodegenResult Codegen(Function kernel, luisa::string_view native_code, uint custom_mask, bool isSpirV);
    CodegenResult RasterCodegen(
        Function vertFunc,
        Function pixelFunc,
        luisa::string_view native_code,
        uint custom_mask,
        bool isSpirV);
    static vstd::string_view ReadInternalHLSLFile(vstd::string_view name);
    uint AddPrinter(vstd::string_view name, Type const *structType);
    vstd::StringBuilder GetNewTempVarName();
};
class StringStateVisitor final : public StmtVisitor, public ExprVisitor {
    Function f;
    CodegenUtility *util;
    struct Scope {
        StringStateVisitor *self;
        Scope(StringStateVisitor *self);
        ~Scope();
    };
    size_t accessCount = 0;
    size_t printCount = 0;
    // size_t rayQuery = 0;
    bool literalBrace = false;
    struct VarHash {
        size_t operator()(Variable const &v) const {
            return v.hash();
        }
    };
    luisa::unordered_set<Variable, VarHash> lazyDeclVars;

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
    void visit(const TypeIDExpr *expr) override { LUISA_NOT_IMPLEMENTED(); }
    void visit(const StringIDExpr *expr) override;
    void visit(const CpuCustomOpExpr *) override { LUISA_NOT_IMPLEMENTED(); }
    void visit(const GpuCustomOpExpr *) override { LUISA_NOT_IMPLEMENTED(); }

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
    void visit(const CommentStmt *) override;
    void visit(const RayQueryStmt *) override;
    void visit(const AutoDiffStmt *stmt) override;
    void visit(const PrintStmt *stmt) override;
    void VisitFunction(
#ifdef LUISA_ENABLE_IR
        vstd::unordered_set<Variable> const &grad_vars,
#endif
        Function func);
    StringStateVisitor(
        Function f,
        vstd::StringBuilder &str,
        CodegenUtility *util);
    ~StringStateVisitor();

protected:
    vstd::StringBuilder &str;
};

template<typename T>
struct PrintValue;

template<>
struct PrintValue<float> {
    void operator()(float const &v, vstd::StringBuilder &str) {
        if (luisa::isnan(v)) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Encountered with NaN.");
        }
        if (luisa::isinf(v)) [[unlikely]] {
            str.append(v < 0.0f ? "(-_INF_f)" : "(_INF_f)");
        } else {
            vstd::to_string(v, str);
        }
    }
};

template<>
struct PrintValue<double> {
    void operator()(double const &v, vstd::StringBuilder &str) {
        if (luisa::isnan(v)) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Encountered with NaN.");
        }
        if (luisa::isinf(v)) [[unlikely]] {
            str.append(v < 0.0 ? "(-_INF_d)" : "(_INF_d)");
        } else {
            str.append(luisa::format("float64_t({})", v));
        }
    }
};

template<>
struct PrintValue<half> {
    void operator()(half const &v, vstd::StringBuilder &str) {
        if (luisa::isnan(v)) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Encountered with NaN.");
        }
        if (luisa::isinf(v)) [[unlikely]] {
            str.append(v < 0.0f ? "(-_INF_f)" : "(_INF_f)");
        } else {
            str.append(luisa::format("float16_t({})", static_cast<float>(v)));
        }
    }
};

template<>
struct PrintValue<short> {
    void operator()(short const &v, vstd::StringBuilder &str) {
        str.append(luisa::format("int16_t({})", v));
    }
};

template<>
struct PrintValue<ushort> {
    void operator()(ushort const &v, vstd::StringBuilder &str) {
        str.append(luisa::format("uint16_t({}u)", v));
    }
};

template<>
struct PrintValue<int> {
    void operator()(int const &v, vstd::StringBuilder &str) {
        str.append(luisa::format("{}", v));
    }
};

template<>
struct PrintValue<uint> {
    void operator()(uint const &v, vstd::StringBuilder &str) {
        str.append(luisa::format("{}u", v));
    }
};

template<>
struct PrintValue<slong> {
    void operator()(slong const &v, vstd::StringBuilder &str) {
        str.append(luisa::format("int64_t({}ll)", v));
    }
};

template<>
struct PrintValue<ulong> {
    void operator()(ulong const &v, vstd::StringBuilder &str) {
        str.append(luisa::format("uint64_t({}ull)", v));
    }
};

template<>
struct PrintValue<bool> {
    void operator()(bool const &v, vstd::StringBuilder &str) {
        if (v)
            str << "true";
        else
            str << "false";
    }
};
template<>
struct PrintValue<luisa::byte> {
    void operator()(bool const &v, vstd::StringBuilder &str) {
        LUISA_ERROR_WITH_LOCATION("Unsupported type.");
    }
};
template<>
struct PrintValue<luisa::ubyte> {
    void operator()(bool const &v, vstd::StringBuilder &str) {
        LUISA_ERROR_WITH_LOCATION("Unsupported type.");
    }
};
template<typename EleType, uint64 N>
struct PrintValue<Vector<EleType, N>> {
    using T = Vector<EleType, N>;
    void print_elem(T const &v, vstd::StringBuilder &varName) {
        for (uint64 i = 0; i < N; ++i) {
            PrintValue<EleType>{}(v[i], varName);
            varName += ',';
        }
        auto &&last = varName.end() - 1;
        if (*last == ',')
            varName.erase(last);
    }
    void operator()(T const &v, vstd::StringBuilder &varName) {
        if constexpr (N > 1) {
            if constexpr (std::is_same_v<EleType, float>) {
                varName << "float";
            } else if constexpr (std::is_same_v<EleType, uint>) {
                varName << "uint";
            } else if constexpr (std::is_same_v<EleType, int>) {
                varName << "int";
            } else if constexpr (std::is_same_v<EleType, bool>) {
                varName << "bool";
            } else if constexpr (std::is_same_v<EleType, half>) {
                varName << "float16_t";
            } else if constexpr (std::is_same_v<EleType, double>) {
                varName << "float64_t";
            } else if constexpr (std::is_same_v<EleType, short>) {
                varName << "int16_t";
            } else if constexpr (std::is_same_v<EleType, ushort>) {
                varName << "uint16_t";
            } else if constexpr (std::is_same_v<EleType, slong>) {
                varName << "int64_t";
            } else if constexpr (std::is_same_v<EleType, ulong>) {
                varName << "uint64_t";
            } else {
                // static_assert(luisa::always_false_v<T>, "Unsupported type.");
                LUISA_ERROR_WITH_LOCATION("Unsupported type.");
            }
            vstd::to_string(N, varName);
            varName << '(';
            print_elem(v, varName);
            varName << ')';
        } else {
            print_elem(v, varName);
        }
    }
};

template<uint64 N>
struct PrintValue<Matrix<N>> {
    using T = Matrix<N>;
    using EleType = float;
    void operator()(T const &v, vstd::StringBuilder &varName) {
        varName << "_float";
        auto ss = vstd::to_string(N);
        varName << ss << 'x' << ss << '(';
        PrintValue<Vector<EleType, N>> vecPrinter;
        for (uint64 i = 0; i < N; ++i) {
            vecPrinter.print_elem(v[i], varName);
            varName += ',';
        }
        auto &&last = varName.end() - 1;
        if (*last == ',')
            varName.erase(last);
        varName << ')';
    }
};

}// namespace lc::hlsl
