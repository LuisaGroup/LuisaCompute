#pragma once
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
#include <luisa/vstl/string_builder.h>
#include "shared_var.h"
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
    vstd::vector<std::pair<vstd::string, luisa::compute::Type const *>> printers;
    bool useTex2DBindless{false};
    bool useTex3DBindless{false};
    bool useBufferBindless{false};
    bool use_8bit{false};
    uint32_t validation_count{0u};// resource size count when debug mode enabled
    uint64 immutableHeaderSize{0u};
    vstd::MD5 typeMD5;
    CodegenResult() = default;
    CodegenResult(
        vstd::StringBuilder &&result,
        vstd::vector<std::pair<vstd::string, Type const *>> &&printers,
        Properties &&properties,
        bool useTex2DBindless,
        bool useTex3DBindless,
        bool useBufferBindless,
        bool use_8bit,
        uint32_t validation_count,
        uint64 immutableHeaderSize,
        vstd::MD5 typeMD5) : result(std::move(result)), properties(std::move(properties)), printers(std::move(printers)), useTex2DBindless{useTex2DBindless}, useTex3DBindless{useTex3DBindless}, useBufferBindless{useBufferBindless}, use_8bit{use_8bit}, validation_count{validation_count}, immutableHeaderSize(immutableHeaderSize), typeMD5(typeMD5) {}
    CodegenResult(CodegenResult const &) = delete;
    CodegenResult(CodegenResult &&) = default;
};
#undef AddPrinter
// A 32-bit word of a generated HLSL struct that mixes 16-bit scalars with 1-byte
// bools.
//
// The host lays a struct out with C++ rules and copies its bytes verbatim into
// the argument block or the buffer, so the generated HLSL struct has to use the
// same byte offsets.  HLSL cannot express that byte pattern for such a word:
// dxc allocates bitfields (the one-byte `bool` spelling) in whole 32-bit units,
// while 16-bit scalars pack two per unit, so a `bool` can never sit next to a
// 16-bit scalar in one unit.  The word is therefore emitted as a single `uint`
// and its fields are read and written through the `_lc_pack_*` helpers of
// builtin/hlsl_header.bytes.
struct PackedStructMember {
    enum class Kind : uint8_t {
        F16,
        I16,
        U16,
        BF16,// software BF16 (its storage is already a raw 16-bit pattern)
        BOOL
    };
    /// Index of the emitted `uint v<word_index>` that holds the word.
    uint word_index;
    /// Byte offset of the member inside that word.
    uint byte_offset;
    Kind kind;
};

/// The member packing of one struct type (see `PackedStructMember`).
struct StructPacking {
    struct Word {
        /// Byte offset of the word inside the struct.
        size_t byte_offset;
        /// Emitted member index of the `uint` that holds the word.
        uint word_index;
        /// AST member indices that live in this word, in declaration order.
        vstd::vector<uint> members;
    };
    vstd::vector<Word> words;
    vstd::unordered_map<uint, PackedStructMember> members;
    [[nodiscard]] bool empty() const noexcept { return words.empty(); }
};

/// Name suffix of the `_lc_pack_get_*`/`_lc_pack_set_*` helper pair of a packed
/// member kind (see builtin/hlsl_header.bytes).
[[nodiscard]] constexpr vstd::string_view packed_kind_suffix(
    PackedStructMember::Kind kind) noexcept {
    switch (kind) {
        case PackedStructMember::Kind::F16: return "f16";
        case PackedStructMember::Kind::I16: return "i16";
        case PackedStructMember::Kind::U16: return "u16";
        case PackedStructMember::Kind::BF16: return "bf16";
        case PackedStructMember::Kind::BOOL: return "bool";
    }
    return "u16";
}

/// Layout of `structType`: the words that must be packed, and the member mapping
/// used when reading or writing a member of such a word.  `bf16_fallback` selects
/// how a BF16 member is stored (raw 16-bit pattern versus the native type, which
/// cannot be packed).
[[nodiscard]] StructPacking plan_struct_packing(Type const *structType,
                                                bool bf16_fallback);

/// Optional shader-capability extensions of the SPIR-V target, as negotiated by
/// the host (see src/backends/vk/vk_shader_capabilities.h).
///
/// Each flag corresponds to a Vulkan extension that adds *shader* capability to a
/// module: a module that declares OpCapability FMAKHR, AbortKHR or
/// RayTracingClusterAccelerationStructureNV is rejected by every device whose
/// logical device did not enable the matching extension, so the generated source
/// may only contain the corresponding instruction when the flag is set.  The DXIL
/// backend passes nothing (DXIL has none of these instructions) and the Vulkan
/// compatibility HLSL-to-SPIR-V route fills the flags in from the device it
/// compiles for; a default-constructed set keeps the generated source
/// byte-identical to a build without these features.
struct SpirvShaderCapabilities {
    /// VK_KHR_shader_fma: `mad()` on 32-bit float lowers to OpFmaKHR, the only
    /// fused multiply-add SPIR-V has.  Without it dxc lowers `mad()` to an unfused
    /// OpFMul/OpFAdd pair, so an emitted `_fma` would round twice while
    /// `luisa::compute::fma` promises a single rounding.
    bool fma{false};
    /// VK_KHR_shader_abort: `$debug_break` lowers to OpAbortKHR, which reports a
    /// 32-bit message to the host and ends the invocation.  Without it the HLSL
    /// backend has no way to express a debug break and rejects the statement.
    bool abort{false};
    /// VK_NV_cluster_acceleration_structure: `CallOp::RAY_QUERY_CLUSTER_ID` lowers
    /// to OpRayQueryGetIntersectionClusterIdNV, which reads the cluster ID of one
    /// intersection of a ray query.  Without it there is no instruction for the
    /// operation at all (DXIL has no cluster acceleration structures, and the
    /// SPIR-V extension is opt-in), so the codegen refuses the operation instead
    /// of emitting a module that declares an extension the device never enabled.
    bool cluster_acceleration_structure{false};
    /// VK_KHR_maintenance9: the `maintenance9` feature lets the bitwise
    /// instructions (OpBitReverse / OpBitCount) take an integer operand of any
    /// width, not only 32-bit.  Without it dxc has to widen a 16-bit operand to
    /// 32 bits (and fix up the result afterwards), or split a 64-bit one into two
    /// 32-bit halves, so `reverse()` on a 16/64-bit value costs several extra
    /// instructions.  With it the codegen hands `reversebits` the natural operand
    /// and converts the result to the DSL result type (`uint`/`uintN`) itself.
    /// The 16-bit form also shifts by 16, preserving the established 32-bit
    /// reversal semantics rather than changing results on capable devices.
    /// Unlike the flags above this one declares no OpCapability of its own: it
    /// only removes an operand-width restriction, so it is an optimization, never
    /// a correctness requirement, and it stays off unless the device negotiated
    /// the extension.  8-bit operands keep the legacy widening either way (the
    /// codegen has no 8-bit bitwise path), and `countbits` already receives the
    /// natural operand, so it needs no flag at the source level.
    bool maintenance9_bitwise{false};
    /// VK_VALVE_shader_mixed_float_dot_product: `dot2add(half2, half2, float)`
    /// lowers to OpFDot2MixAcc32VALVE, which rounds exactly once. Without it dxc
    /// emits the unfused sequence (half products rounded to 16 bits, widened,
    /// then added in FP32), so the flag only selects the fused form, never a
    /// different source spelling.
    bool mixed_float_dot_product{false};
    /// VK_EXT_shader_split_barrier: `split_barrier_arrive()` /
    /// `split_barrier_wait()` lower to OpControlBarrierArriveEXT /
    /// OpControlBarrierWaitEXT (Workgroup execution and memory scopes,
    /// Release on the arrive, Acquire on the wait), so independent work can
    /// run between the two halves of a workgroup barrier.  Without it there
    /// is no instruction for the operation (DXIL has no split barrier, and
    /// the SPIR-V extension is opt-in), so the codegen refuses the operation
    /// instead of degrading it to a full barrier or a no-op.  Compute only:
    /// the SplitBarrierEXT capability is illegal in raster stages.
    bool split_barrier{false};
};

struct RegisterIndexer;
class CodegenUtility {
public:
    vstd::unique_ptr<CodegenStackData> opt{};
    vstd::unordered_map<vstd::string, std::pair<vstd::string, Type const *>> attributes;
    CodegenUtility();
    ~CodegenUtility();
    uint IsBool(Type const &type);
    bool GetConstName(uint64 hash, ConstantData const &data, vstd::StringBuilder &str);
    void GetVariableName(Function func, Variable const &type, vstd::StringBuilder &str);
    void GetVariableName(Function func, Variable::Tag type, uint id, vstd::StringBuilder &str);
    void GetTypeName(Type const &type, vstd::StringBuilder &str, Usage usage, bool local_var = true);
    void GetFunctionDecl(Function func, vstd::StringBuilder &str);
    void GetFunctionName(Function callable, vstd::StringBuilder &result);
    void GetFunctionName(CallExpr const *expr, vstd::StringBuilder &result, StringStateVisitor &visitor);
    void RegistStructType(Type const *type);

    void CodegenFunction(
        Function func,
        vstd::StringBuilder &result,
        bool cBufferNonEmpty,
        bool codegen_self);
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
        bool generate_debug_info,
        uint &validation_count,
        vstd::span<const uint64_t> func_hashes = {});
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
    // `bf16_fallback` makes the codegen emulate BF16 in software (see
    // builtin/bfloat16_fallback.bytes).  The host decides: it is always on for
    // the DirectX backend, and on for the Vulkan compatibility HLSL-to-SPIR-V
    // route when the physical device lacks shaderBFloat16Type.
    // `spirv_capabilities` names the optional shader-capability extensions the
    // SPIR-V target device negotiated (see `SpirvShaderCapabilities`); it is inert
    // unless `isSpirV` is set and every flag defaults to off.
    CodegenResult Codegen(Function kernel, luisa::string_view native_code, uint custom_mask, bool isSpirV, bool noRegister = false, bool enable_debug_info = false, bool enable_fast_math = false, bool fallback_rtx = false, bool bf16_fallback = false, SpirvShaderCapabilities spirv_capabilities = {});
    CodegenResult RayTracingCodegen(Function kernel, luisa::string_view native_code, uint custom_mask, bool isSpirV, bool noRegister = false, bool enable_debug_info = false, bool enable_fast_math = false, bool fallback_rtx = false, bool bf16_fallback = false, SpirvShaderCapabilities spirv_capabilities = {});
    CodegenResult RasterCodegen(
        Function vertFunc,
        Function pixelFunc,
        luisa::string_view native_code,
        uint custom_mask,
        bool isSpirV,
        bool noRegister = false,
        bool enable_debug_info = false,
        bool enable_fast_math = false,
        bool fallback_rtx = false,
        bool bf16_fallback = false,
        SpirvShaderCapabilities spirv_capabilities = {});
    static vstd::string_view ReadInternalHLSLFile(vstd::string_view name);
    /// Fails closed when an atomic access chain targets a member of a packed
    /// struct word: such a member is a slice of a `uint`, so it has no address.
    void CheckPackedAtomicTarget(luisa::span<Expression const *const> args);
    /// The packing a member of `type` needs, or `nullptr` when the member is
    /// emitted as an ordinary HLSL member (see `PackedStructMember`).
    [[nodiscard]] PackedStructMember const *GetPackedStructMember(
        Type const *type, uint member_index);
    /// Whether `type` holds a BF16 scalar or vector payload.
    [[nodiscard]] static bool IsBf16(Type const *type) noexcept {
        return type != nullptr &&
               (type->is_bfloat16() ||
                (type->is_vector() && type->element()->is_bfloat16()));
    }
    uint AddPrinter(vstd::string_view name, luisa::compute::Type const *structType);
    vstd::StringBuilder GetNewTempVarName();
    bool TypeIsAliased(luisa::compute::Type const *t) const;
    bool VectorShouldBeAliased(luisa::compute::Type const *t) const;
    void OriginToAliased(luisa::compute::Type const *t, vstd::StringBuilder &sb);
    void AliasedToOrigin(luisa::compute::Type const *t, vstd::StringBuilder &sb);
};
class StringStateVisitor final : public StmtVisitor, public ExprVisitor {
public:
    Function f;
private:
    CodegenUtility *util;
    luisa::fixed_vector<uint, 4> switchCount;
    struct Scope {
        StringStateVisitor *self;
        explicit Scope(StringStateVisitor *self);
        ~Scope() { self->str << "}\n"sv; }
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
    SharedVarSet *sharedVariables = nullptr;
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
    void visit(const FuncRefExpr *) override { LUISA_NOT_IMPLEMENTED(); }
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
    void visit(const SuspendStmt *) override;

    void visit(const AutoDiffStmt *stmt) override;
    void visit(const PrintStmt *stmt) override;
    void visit(const DebugBreakStmt *stmt) override;
    void VisitFunction(Function func);
    StringStateVisitor(
        Function f,
        vstd::StringBuilder &str,
        CodegenUtility *util);
    ~StringStateVisitor() = default;

protected:
    vstd::StringBuilder &str;
};

/// Name of the BF16 scalar/vector type in the generated source.
///
/// dxc reserves the identifier `bfloat16_t` for `-spirv` ("bfloat16_t is only
/// supported with -spirv"), so the software emulation of
/// builtin/bfloat16_fallback.bytes carries its own storage type names
/// (`lc_bf16_t`, `lc_bf16_t2`, ...).  A device that compiles the native type
/// keeps `bfloat16_t`.
[[nodiscard]] inline vstd::string_view bf16_type_name(bool bf16_fallback) noexcept {
    return bf16_fallback ? "lc_bf16_t"sv : "bfloat16_t"sv;
}

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
struct PrintValue<bfloat16> {
    void operator()(bfloat16 const &v, vstd::StringBuilder &str,
                    bool bf16_fallback = false) {
        if (bf16_fallback) {
            // Software BF16 stores the raw 16-bit pattern, so a literal is its
            // bits: this keeps every value (including NaN and infinities)
            // exactly representable without a BF16 constructor.
                          str.append(luisa::format("(({}){}u)", bf16_type_name(bf16_fallback), v.bits()));
            return;
        }
        if (luisa::isnan(static_cast<float>(v))) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Encountered with NaN.");
        }
        if (luisa::isinf(static_cast<float>(v))) [[unlikely]] {
            str.append(static_cast<float>(v) < 0.0f ? "(-_INF_f)" : "(_INF_f)");
        } else {
            str.append(luisa::format("bfloat16_t({})", static_cast<float>(v)));
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
struct PrintValue<luisa::ulong> {
    void operator()(luisa::ulong const &v, vstd::StringBuilder &str) {
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
    void operator()(luisa::byte const &v, vstd::StringBuilder &str) {
        str.append(luisa::format("int8_t({})", static_cast<int>(v)));
    }
};
template<>
struct PrintValue<luisa::ubyte> {
    void operator()(luisa::ubyte const &v, vstd::StringBuilder &str) {
        str.append(luisa::format("uint8_t({}u)", static_cast<uint>(v)));
    }
};
template<typename EleType, uint64 N>
struct PrintValue<Vector<EleType, N>> {
    using T = Vector<EleType, N>;
    void print_elem(T const &v, vstd::StringBuilder &varName,
                    bool bf16_fallback = false) {
        for (uint64 i = 0; i < N; ++i) {
            if constexpr (std::is_same_v<EleType, bfloat16>) {
                PrintValue<EleType>{}(v[i], varName, bf16_fallback);
            } else {
                PrintValue<EleType>{}(v[i], varName);
            }
            varName += ',';
        }
        auto &&last = varName.end() - 1;
        if (*last == ',')
            varName.erase(last);
    }
    void operator()(T const &v, vstd::StringBuilder &varName,
                    bool bf16_fallback = false) {
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
              } else if constexpr (std::is_same_v<EleType, bfloat16>) {
                  varName << bf16_type_name(bf16_fallback);
            } else if constexpr (std::is_same_v<EleType, double>) {
                varName << "float64_t";
            } else if constexpr (std::is_same_v<EleType, luisa::byte>) {
                varName << "int8_t";
            } else if constexpr (std::is_same_v<EleType, luisa::ubyte>) {
                varName << "uint8_t";
            } else if constexpr (std::is_same_v<EleType, short>) {
                varName << "int16_t";
            } else if constexpr (std::is_same_v<EleType, ushort>) {
                varName << "uint16_t";
            } else if constexpr (std::is_same_v<EleType, slong>) {
                varName << "int64_t";
            } else if constexpr (std::is_same_v<EleType, luisa::ulong> || std::is_same_v<EleType, uint64_t>) {
                varName << "uint64_t";
            } else {
                // static_assert(luisa::always_false_v<T>, "Unsupported type.");
                LUISA_ERROR_WITH_LOCATION("Unsupported type.");
            }
            vstd::to_string(N, varName);
            varName << '(';
            print_elem(v, varName, bf16_fallback);
            varName << ')';
        } else {
            print_elem(v, varName, bf16_fallback);
        }
    }
};

template<uint64 N>
struct PrintValue<Matrix<float, N>> {
    using T = Matrix<float, N>;
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
