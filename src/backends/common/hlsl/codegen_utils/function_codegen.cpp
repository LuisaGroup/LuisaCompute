// Function Code Generation

#include "../hlsl_codegen.h"
#include <luisa/vstl/string_utility.h>
#include <luisa/ast/constant_data.h>
#include <luisa/ast/type_registry.h>
#include <luisa/ast/function_builder.h>
#include "../struct_generator.h"
#include "../codegen_stack_data.h"
#include <luisa/core/dynamic_module.h>
#include <luisa/core/logging.h>
#include <luisa/ast/external_function.h>

// External declaration for shared variable from hlsl_codegen_util.cpp
extern bool shown_buffer_warning;

namespace lc::hlsl {

#ifdef LUISA_ENABLE_IR
// Defined in entry_points.cpp — collects gradient variables from a function body.
void glob_variables_with_grad(Function f, vstd::unordered_set<Variable> &gradient_variables) noexcept;
#endif

namespace {

[[nodiscard]] bool is_validation_resource(Type const *type) noexcept {
    return type->is_buffer() || type->is_bindless_array();
}

[[nodiscard]] bool usage_reads(Usage usage) noexcept {
    return (luisa::to_underlying(usage) &
            luisa::to_underlying(Usage::READ)) != 0u ||
           usage == Usage::NONE;
}

[[nodiscard]] bool usage_writes(Usage usage) noexcept {
    return (luisa::to_underlying(usage) &
            luisa::to_underlying(Usage::WRITE)) != 0u;
}

void print_validation_bound_name(
    vstd::StringBuilder &str, Variable variable) noexcept {
    str << "_validation_bound_"sv;
    vstd::to_string(variable.uid(), str);
}

}// namespace

// Generate function declaration
void CodegenUtility::GetFunctionDecl(Function func, vstd::StringBuilder &str) {
    vstd::StringBuilder data;
    uint64 tempIdx = 0;
    auto GetTemplateName = [&] {
        data << 'T';
        vstd::to_string(tempIdx, data);
        tempIdx++;
    };
    auto GetTypeName = [&](Type const *t, Usage usage) {
        if (t->is_texture() || t->is_buffer())
            GetTemplateName();
        else
            CodegenUtility::GetTypeName(*t, data, usage);
    };
    if (func.return_type()) {
        CodegenUtility::GetTypeName(*func.return_type(), data, Usage::READ);
    } else {
        data += "void"sv;
    }
    {
        data += " "sv;
        GetFunctionName(func, data);
        if (func.arguments().empty()) {
            data += "()"sv;
        } else {
            data += '(';
            for (auto &&i : func.arguments()) {
                Usage usage = func.variable_usage(i.uid());
                if (i.tag() == Variable::Tag::REFERENCE) {
                    if ((static_cast<uint32_t>(usage) & static_cast<uint32_t>(Usage::WRITE)) != 0) {
                        data += opt->isSpirv ? "[[vk::ext_reference]] inout "sv : "inout "sv;
                    }
                }
                RegistStructType(i.type());

                vstd::StringBuilder varName;
                CodegenUtility::GetVariableName(func, i, varName);
                if (opt->isSpirv && i.type()->is_texture() &&
                    usage_reads(usage) && usage_writes(usage)) {
                    GetTemplateName();
                    data << ' ' << varName << ',';
                    GetTemplateName();
                    data << ' ' << varName << "_rw,"sv;
                } else if (i.type()->is_accel()) {
                    if ((to_underlying(usage) & to_underlying(Usage::WRITE)) == 0) {
                        CodegenUtility::GetTypeName(*i.type(), data, usage);
                        data << ' ' << varName << ',';
                    }
                    GetTemplateName();
                    data << ' ' << varName << "Inst,"sv;
                } else {
                    GetTypeName(i.type(), usage);
                    data << ' ';
                    data << varName;
                    if (opt->enable_debug_info &&
                        is_validation_resource(i.type())) {
                        data << ",uint "sv;
                        print_validation_bound_name(data, i);
                    }
                    data << ',';
                }
            }
            data[data.size() - 1] = ')';
        }
    }
    if (tempIdx > 0) {
        str << "template<"sv;
        for (uint64 i : vstd::range(tempIdx)) {
            str << "typename T"sv;
            vstd::to_string(static_cast<int64_t>(i), str);
            str << ',';
        }
        *(str.end() - 1) = '>';
    }
    str << '\n'
        << data;
}

// Get callable function name
void CodegenUtility::GetFunctionName(Function callable, vstd::StringBuilder &str) {
    auto &&count_and_name = opt->GetFuncCountAndName(callable);
    str << (count_and_name.second.empty() ? "custom_"sv : luisa::string_view{count_and_name.second}) << luisa::format("{}", count_and_name.first);
}

void CodegenUtility::GetFunctionName(CallExpr const *expr, vstd::StringBuilder &str, StringStateVisitor &vis) {

    auto args = expr->arguments();
    auto IsNumVec3 = [&](Type const &t) {
        if (t.tag() != Type::Tag::VECTOR || t.dimension() != 3) return false;
        auto &&ele = *t.element();
        return ele.is_scalar();
    };
    auto PrintArgs = [&](size_t offset = 0) {
        if (args.empty()) return;
        auto last = args.size() - 1;
        for (auto i : vstd::range(static_cast<size_t>(offset), static_cast<size_t>(last))) {
            args[i]->accept(vis);
            str << ',';
        }
        args.back()->accept(vis);
    };
    auto PrintTypedBindlessBufferIndex = [&](Expression const *array,
                                              Expression const *slot) {
        str << "_TYPED_BUFFER_INDEX(";
        array->accept(vis);
        str << ',';
        slot->accept(vis);
        str << ')';
    };
    auto PrintTypedBindlessBufferOffset = [&](Expression const *array,
                                               Expression const *slot,
                                               Expression const *offset) {
        str << "(_TYPED_BUFFER_BIAS(";
        array->accept(vis);
        str << ',';
        slot->accept(vis);
        str << ")+";
        offset->accept(vis);
        str << ')';
    };
    auto PrintValidationBound = [&](Expression const *resource) {
        LUISA_ASSERT(
            resource != nullptr && resource->tag() == Expression::Tag::REF,
            "HLSL debug validation requires a referenced resource expression.");
        auto variable = static_cast<RefExpr const *>(resource)->variable();
        LUISA_ASSERT(
            is_validation_resource(variable.type()),
            "HLSL debug validation bound requested for non-buffer resource {}.",
            variable.type()->description());
        if (opt->funcType == CodegenStackData::FuncType::Callable) {
            print_validation_bound_name(str, variable);
            return;
        }
        auto key = CodegenStackData::ValidateKey{
            vis.f.hash(), variable.uid()};
        auto iter = opt->validate_index_map.find(key);
        LUISA_ASSERT(
            iter != opt->validate_index_map.end(),
            "Missing HLSL debug-validation slot for resource {} in function {}.",
            variable.uid(), vis.f.hash());
        str << "_Global[0]._validate_"sv;
        vstd::to_string(iter->second, str);
    };
    auto PrintValidationBoundArgument = [&](Expression const *resource) {
        if (opt->enable_debug_info) {
            str << ',';
            PrintValidationBound(resource);
        }
    };
    auto IsSplitTextureView = [&](Expression const *resource) {
        if (!opt->isSpirv || !resource->type()->is_texture()) {
            return false;
        }
        LUISA_ASSERT(
            resource->tag() == Expression::Tag::REF,
            "HLSL texture-view selection requires a referenced texture.");
        auto variable = static_cast<RefExpr const *>(resource)->variable();
        auto usage = vis.f.variable_usage(variable.uid());
        return usage_reads(usage) && usage_writes(usage);
    };
    auto PrintTextureView = [&](Expression const *resource, bool writable) {
        resource->accept(vis);
        if (writable && IsSplitTextureView(resource)) {
            str << "_rw"sv;
        }
    };
    auto TypeToCoop = [](CoopRefVecType type, vstd::StringBuilder &sb) {
        switch (type) {
            case CoopRefVecType::UINT8:
                sb << "dx::linalg::DATA_TYPE_UINT8";
                break;
            case CoopRefVecType::INT8:
                sb << "dx::linalg::DATA_TYPE_SINT8";
                break;
            case CoopRefVecType::UINT32:
                sb << "dx::linalg::DATA_TYPE_UINT32";
                break;
            case CoopRefVecType::INT32:
                sb << "dx::linalg::DATA_TYPE_SINT32";
                break;
            case CoopRefVecType::FLOAT16:
                sb << "dx::linalg::DATA_TYPE_FLOAT16";
                break;
            case CoopRefVecType::FLOAT32:
                sb << "dx::linalg::DATA_TYPE_FLOAT32";
                break;
            case CoopRefVecType::FLOAT8_E4M3:
                sb << "dx::linalg::DATA_TYPE_FLOAT8_E4M3";
                break;
            case CoopRefVecType::FLOAT8_E5M2:
                sb << "dx::linalg::DATA_TYPE_FLOAT8_E5M2";
                break;
            default:
                LUISA_ERROR("Illegal coop type.");
        }
    };
    check_builtin_call_valid(expr->op(), expr->type(), args);
    auto mark_coherent = [&](Expression const *expr) {
        LUISA_DEBUG_ASSERT(expr->tag() == Expression::Tag::REF);
        auto buffer_expr = static_cast<RefExpr const *>(expr);
        opt->globallyCoherentBuffers.emplace(vis.f.builder()).value().emplace(buffer_expr->variable().uid());
    };
    switch (expr->op()) {
        case CallOp::CUSTOM:
            GetFunctionName(expr->custom(), str);
            str << '(';
            {
                uint64 sz = 0;
                auto custom_arguments = expr->custom().arguments();
                LUISA_ASSERT(
                    custom_arguments.size() == args.size(),
                    "HLSL custom-call argument count mismatch: expected {}, got {}.",
                    custom_arguments.size(), args.size());
                auto iter = opt->globallyCoherentBuffers.find(expr->custom().builder());
                for (auto &&i : args) {
                    auto formal = custom_arguments[sz];
                    if (i->type()->is_accel()) {
                        if ((static_cast<uint>(expr->custom().variable_usage(formal.uid())) & static_cast<uint>(Usage::WRITE)) == 0) {
                            i->accept(vis);
                            str << ',';
                        }
                        i->accept(vis);
                        str << "Inst"sv;
                    } else {
                        // globallycoherent propagated
                        if (i->type()->is_buffer() && i->tag() == Expression::Tag::REF && iter) {
                            if (iter.value().contains(formal.uid())) {
                                opt->globallyCoherentBuffers.emplace(vis.f.builder()).value().emplace(static_cast<RefExpr const *>(i)->variable().uid());
                            }
                        }
                        if (opt->isSpirv && formal.type()->is_texture()) {
                            auto formal_usage =
                                expr->custom().variable_usage(formal.uid());
                            auto reads = usage_reads(formal_usage);
                            auto writes = usage_writes(formal_usage);
                            LUISA_ASSERT(
                                reads || writes,
                                "HLSL callable texture argument {} has no resource usage.",
                                formal.uid());
                            if (reads) {
                                PrintTextureView(i, false);
                            }
                            if (writes) {
                                if (reads) { str << ','; }
                                PrintTextureView(i, true);
                            }
                        } else {
                            i->accept(vis);
                        }
                        if (opt->enable_debug_info &&
                            is_validation_resource(formal.type())) {
                            str << ',';
                            PrintValidationBound(i);
                        }
                    }
                    ++sz;
                    if (sz != args.size()) {
                        str << ',';
                    }
                }
            }
            str << ')';
            return;
        case CallOp::EXTERNAL:
            str << expr->external()->name();
            break;
        case CallOp::ALL:
            str << "all"sv;
            break;
        case CallOp::ANY:
            str << "any"sv;
            break;
        case CallOp::SELECT:
            str << "select"sv;
            str << '(';
            args[2]->accept(vis);
            str << ',';
            args[1]->accept(vis);
            str << ',';
            args[0]->accept(vis);
            str << ')';
            return;
        case CallOp::CLAMP:
            str << "clamp"sv;
            break;
        case CallOp::SATURATE:
            str << "saturate"sv;
            break;
        case CallOp::LERP:
            str << "lerp"sv;
            break;
        case CallOp::STEP:
            str << "step"sv;
            break;
        case CallOp::SMOOTHSTEP:
            str << "smoothstep"sv;
            break;
        case CallOp::ABS:
            str << "abs"sv;
            break;
        case CallOp::MAX:
            str << "max"sv;
            break;
        case CallOp::MIN:
            str << "min"sv;
            break;
        case CallOp::POW:
            str << "pow"sv;
            break;
        case CallOp::CLZ:
            LUISA_DEBUG_ASSERT(args.size() == 1);
            // CLZ always returns uint (32-bit) per DSL semantics,
            // so cast argument to uint and use 31 as the bit-width.
            str << "_clz("sv;
            if (args[0]->type()->is_vector()) {
                str << "uint"sv << args[0]->type()->dimension() << ",("sv;
                str << "uint"sv << args[0]->type()->dimension() << ")("sv;
                args[0]->accept(vis);
                str << "),31)"sv;
            } else if (args[0]->type()->size() < 4u) {
                str << "uint,("sv;
                str << "uint)("sv;
                args[0]->accept(vis);
                str << "),31)"sv;
            } else {
                str << "uint,"sv;
                args[0]->accept(vis);
                str << ",31)"sv;
            }
            return;
        case CallOp::CTZ:
            str << "_ctz"sv;
            break;
        case CallOp::POPCOUNT:
            str << "countbits"sv;
            break;
        case CallOp::REVERSE:
            // REVERSE always returns uint (32-bit) per DSL semantics,
            // so cast argument to uint before reversing bits.
            LUISA_DEBUG_ASSERT(args.size() == 1);
            str << "reversebits("sv;
            if (args[0]->type()->is_vector()) {
                str << "uint"sv << args[0]->type()->dimension() << '(';
                args[0]->accept(vis);
                str << ")"sv;
            } else if (args[0]->type()->size() < 4u) {
                str << "uint("sv;
                args[0]->accept(vis);
                str << ")"sv;
            } else {
                args[0]->accept(vis);
            }
            str << ')';
            return;
        case CallOp::ISINF:
            str << "isinf"sv;
            break;
        case CallOp::ISNAN:
            str << "isnan"sv;
            break;
        case CallOp::ACOS:
            str << "acos"sv;
            break;
        case CallOp::ACOSH:
            str << "_acosh"sv;
            break;
        case CallOp::ASIN:
            str << "asin"sv;
            break;
        case CallOp::ASINH:
            str << "_asinh"sv;
            break;
        case CallOp::ATAN:
            str << "atan"sv;
            break;
        case CallOp::ATAN2:
            str << "_atan2"sv;
            break;
        case CallOp::ATANH:
            str << "_atanh"sv;
            break;
        case CallOp::COS:
            str << "cos"sv;
            break;
        case CallOp::COSH:
            str << "cosh"sv;
            break;
        case CallOp::SIN:
            str << "sin"sv;
            break;
        case CallOp::SINH:
            str << "sinh"sv;
            break;
        case CallOp::TAN:
            str << "tan"sv;
            break;
        case CallOp::TANH:
            str << "tanh"sv;
            break;
        case CallOp::EXP:
            str << "exp"sv;
            break;
        case CallOp::EXP2:
            str << "exp2"sv;
            break;
        case CallOp::EXP10:
            str << "_exp10"sv;
            break;
        case CallOp::LOG:
            str << "log"sv;
            break;
        case CallOp::LOG2:
            str << "log2"sv;
            break;
        case CallOp::LOG10:
            str << "log10"sv;
            break;
        case CallOp::SQRT:
            str << "sqrt"sv;
            break;
        case CallOp::RSQRT:
            str << "rsqrt"sv;
            break;
        case CallOp::CEIL:
            str << "ceil"sv;
            break;
        case CallOp::FLOOR:
            str << "floor"sv;
            break;
        case CallOp::FRACT:
            str << "_fract"sv;
            break;
        case CallOp::TRUNC:
            str << "trunc"sv;
            break;
        case CallOp::ROUND:
            str << "_round"sv;
            break;
        case CallOp::RINT:
            // HLSL round is the round-to-nearest-even intrinsic. ROUND uses
            // the separate _round helper to implement half-away-from-zero.
            str << "round"sv;
            break;
        case CallOp::FMA:
            str << "_fma"sv;
            break;
        case CallOp::COPYSIGN:
            str << "_copysign"sv;
            break;
        case CallOp::CROSS:
            str << "cross"sv;
            break;
        case CallOp::DOT:
            str << "dot"sv;
            break;
        case CallOp::LENGTH:
            str << "length"sv;
            break;
        case CallOp::LENGTH_SQUARED:
            str << "_length_sqr"sv;
            break;
        case CallOp::NORMALIZE:
            str << "normalize"sv;
            break;
        case CallOp::FACEFORWARD:
            str << "faceforward"sv;
            break;
        case CallOp::REFLECT:
            str << "reflect"sv;
            break;
        case CallOp::DETERMINANT:
            str << "_determinant"sv;
            break;
        case CallOp::TRANSPOSE:
            str << "_transpose"sv;
            break;
        case CallOp::INVERSE:
            str << "_inverse"sv;
            break;
        case CallOp::ATOMIC_EXCHANGE:
        case CallOp::ATOMIC_COMPARE_EXCHANGE:
        case CallOp::ATOMIC_FETCH_ADD:
        case CallOp::ATOMIC_FETCH_SUB:
        case CallOp::ATOMIC_FETCH_AND:
        case CallOp::ATOMIC_FETCH_OR:
        case CallOp::ATOMIC_FETCH_XOR:
        case CallOp::ATOMIC_FETCH_MIN:
        case CallOp::ATOMIC_FETCH_MAX: {
            auto rootVar = static_cast<RefExpr const *>(args[0]);
            if ((expr->type()->is_float() && expr->op() != CallOp::ATOMIC_EXCHANGE) || expr->op() == CallOp::ATOMIC_COMPARE_EXCHANGE) {
                mark_coherent(args[0]);
            }
            auto &chain = opt->GetAtomicFunc(vis.f, expr->op(), rootVar->variable(), expr->type(), args);
            chain.call_this_func(args, str, vis);
            return;
        }
        case CallOp::TEXTURE_READ:
            str << "_Readtx";
            break;
        case CallOp::TEXTURE_WRITE:
            str << "_Writetx";
            break;
        case CallOp::MAKE_LONG2:
        case CallOp::MAKE_LONG3:
        case CallOp::MAKE_LONG4:
        case CallOp::MAKE_ULONG2:
        case CallOp::MAKE_ULONG3:
        case CallOp::MAKE_ULONG4:
        case CallOp::MAKE_BOOL2:
        case CallOp::MAKE_BOOL3:
        case CallOp::MAKE_BOOL4:
        case CallOp::MAKE_UINT2:
        case CallOp::MAKE_UINT3:
        case CallOp::MAKE_UINT4:
        case CallOp::MAKE_INT2:
        case CallOp::MAKE_INT3:
        case CallOp::MAKE_INT4:
        case CallOp::MAKE_FLOAT2:
        case CallOp::MAKE_FLOAT3:
        case CallOp::MAKE_FLOAT4:
        case CallOp::MAKE_SHORT2:
        case CallOp::MAKE_SHORT3:
        case CallOp::MAKE_SHORT4:
        case CallOp::MAKE_USHORT2:
        case CallOp::MAKE_USHORT3:
        case CallOp::MAKE_USHORT4:
        case CallOp::MAKE_BYTE2:
        case CallOp::MAKE_BYTE3:
        case CallOp::MAKE_BYTE4:
        case CallOp::MAKE_UBYTE2:
        case CallOp::MAKE_UBYTE3:
        case CallOp::MAKE_UBYTE4:
        case CallOp::MAKE_HALF2:
        case CallOp::MAKE_HALF3:
        case CallOp::MAKE_HALF4:
        case CallOp::MAKE_DOUBLE2:
        case CallOp::MAKE_DOUBLE3:
        case CallOp::MAKE_DOUBLE4: {
            if (args.size() == 1 && (args[0]->type() == expr->type())) {
                args[0]->accept(vis);
            } else {
                if (args.size() == 1) {//  && args[0]->type()->is_scalar()
                    str << "(("sv;
                    GetTypeName(*expr->type(), str, Usage::READ);
                    str << ")("sv;
                    args[0]->accept(vis);
                    str << "))"sv;
                } else {
                    GetTypeName(*expr->type(), str, Usage::READ);
                    str << '(';
                    for (auto &&i : args) {
                        i->accept(vis);
                        str << ',';
                    }
                    *(str.end() - 1) = ')';
                }
            }
            return;
        }
        case CallOp::MAKE_FLOAT2X2:
        case CallOp::MAKE_FLOAT4X4:
        case CallOp::MAKE_FLOAT3X3: {
            auto dim = expr->type()->dimension();
            if (args.size() == 1 && (args[0]->type() == expr->type())) {
                args[0]->accept(vis);
                return;
            } else {
                auto n = vstd::to_string(dim);
                str << "_float"sv << n << 'x' << n;
            }
        } break;
        case CallOp::BUFFER_READ: {
            bool aliasStruct = TypeIsAliased(expr->type());
            if (aliasStruct) {
                AliasedToOrigin(expr->type(), str);
                str << '(';
            }
            str << "_bfread"sv;
            auto elem = args[0]->type()->element();
            if (IsNumVec3(*elem)) {
                str << "Vec3"sv;
            } else if (elem->is_matrix()) {
                str << "Mat";
            }
            str << '(';
            PrintArgs();
            if (opt->enable_debug_info) {
                str << ',';
                PrintValidationBound(args[0]);
            }
            str << ')';
            if (aliasStruct) {
                str << ')';
            }
            return;
        }
        case CallOp::BUFFER_VOLATILE_READ: {
            mark_coherent(args[0]);
            bool aliasStruct = TypeIsAliased(expr->type());
            if (aliasStruct) {
                AliasedToOrigin(expr->type(), str);
                str << '(';
            }
            str << "_volatile_bfread"sv;
            auto elem = args[0]->type()->element();
            if (IsNumVec3(*elem)) {
                str << "Vec3"sv;
            } else if (elem->is_matrix()) {
                str << "Mat";
            }
            str << '<';
            GetTypeName(*expr->type(), str, Usage::NONE);
            str << ">(";
            PrintArgs();
            // Note: volatile reads use template functions (not macros), so we skip the debug vid
            str << ')';
            if (aliasStruct) {
                str << ')';
            }
            return;
        }
        case CallOp::BUFFER_WRITE:
        case CallOp::BUFFER_VOLATILE_WRITE: {
            bool is_volatile = expr->op() == CallOp::BUFFER_VOLATILE_WRITE;
            if (is_volatile) {
                mark_coherent(args[0]);
                str << "_volatile"sv;
            }
            auto elem = args[0]->type()->element();
            bool aliasStruct = TypeIsAliased(elem);
            str << "_bfwrite"sv;
            if (IsNumVec3(*elem)) {
                str << "Vec3("sv;
                PrintArgs();
                str << ',';
                GetTypeName(*elem->element(), str, Usage::NONE);
                if (opt->enable_debug_info && !is_volatile) {
                    str << ',';
                    PrintValidationBound(args[0]);
                }
                str << ')';
                return;
            } else if (elem->is_matrix()) {
                str << "Mat";
            }
            str << '(';
            auto last = args.size() - 1;
            for (auto i : vstd::range(static_cast<size_t>(0), static_cast<size_t>(last))) {
                args[i]->accept(vis);
                str << ',';
            }
            if (aliasStruct) {
                OriginToAliased(args.back()->type(), str);
                str << '(';
                args.back()->accept(vis);
                str << ')';
            } else {
                args.back()->accept(vis);
            }
            if (opt->enable_debug_info && !is_volatile) {
                str << ',';
                PrintValidationBound(args[0]);
            }
            str << ')';
            return;
        }
        case CallOp::BUFFER_SIZE: {
            if (!shown_buffer_warning) {
                LUISA_WARNING_WITH_LOCATION("CallOp::BUFFER_SIZE is broken on dx!"sv);
                shown_buffer_warning = true;
            }
            str << "_bfsize"sv;
        } break;
        case CallOp::BYTE_BUFFER_VOLATILE_READ: {
            mark_coherent(args[0]);
            bool aliasStruct = TypeIsAliased(expr->type());
            if (aliasStruct) {
                AliasedToOrigin(expr->type(), str);
                str << '(';
            }
            str << "_volatile_bytebfread"sv;
            auto elem = expr->type();
            if (IsNumVec3(*elem)) {
                str << "Vec3"sv;
                str << '<';
                GetTypeName(*elem->element(), str, Usage::NONE);
                str << "4,"sv;
                GetTypeName(*expr->type(), str, Usage::NONE);
                str << ">("sv;
                args[0]->accept(vis);
                str << ',';
                args[1]->accept(vis);
                // Note: volatile byte buffer reads use template functions, skip debug vid
                str << ')';

            } else if (elem->is_matrix()) {
                str << "Mat"sv;
                str << '<';
                switch (elem->dimension()) {
                    case 2:
                        str << "_WrappedFloat2x2"sv;
                        break;
                    case 3:
                        str << "_WrappedFloat3x3"sv;
                        break;
                    case 4:
                        str << "_WrappedFloat4x4"sv;
                        break;
                }
                str << ',';
                GetTypeName(*expr->type(), str, Usage::NONE);
                str << ">("sv;
                args[0]->accept(vis);
                str << ',';
                args[1]->accept(vis);
                str << ')';
            } else {
                str << '<';
                if (aliasStruct) {
                    str << opt->CreateAliasedStruct(elem).first;
                } else {
                    GetTypeName(*elem, str, Usage::NONE);
                }
                str << ">("sv;
                args[0]->accept(vis);
                str << ',';
                args[1]->accept(vis);
                str << ')';
            }
            if (aliasStruct) {
                str << ')';
            }
            return;
        }
        case CallOp::BYTE_BUFFER_READ: {
            bool aliasStruct = TypeIsAliased(expr->type());
            if (aliasStruct) {
                AliasedToOrigin(expr->type(), str);
                str << '(';
            }
            str << "_bytebfread"sv;
            auto elem = expr->type();
            if (IsNumVec3(*elem)) {
                str << "Vec3("sv;
                args[0]->accept(vis);
                str << ',';
                GetTypeName(*elem->element(), str, Usage::NONE);
                str << ',';
                args[1]->accept(vis);
                if (opt->enable_debug_info) {
                    str << ',';
                    PrintValidationBound(args[0]);
                }
                str << ')';

            } else if (elem->is_matrix()) {
                str << "Mat(";
                args[0]->accept(vis);
                str << ',';
                switch (elem->dimension()) {
                    case 2:
                        str << "_WrappedFloat2x2"sv;
                        break;
                    case 3:
                        str << "_WrappedFloat3x3"sv;
                        break;
                    case 4:
                        str << "_WrappedFloat4x4"sv;
                        break;
                }
                str << ',';
                args[1]->accept(vis);
                if (opt->enable_debug_info) {
                    str << ',';
                    PrintValidationBound(args[0]);
                }
                str << ')';
            } else {
                str << '(';
                args[0]->accept(vis);
                str << ',';
                if (aliasStruct) {
                    str << opt->CreateAliasedStruct(elem).first;
                } else {
                    GetTypeName(*elem, str, Usage::NONE);
                }
                str << ',';
                args[1]->accept(vis);
                if (opt->enable_debug_info) {
                    str << ',';
                    PrintValidationBound(args[0]);
                }
                str << ')';
            }
            if (aliasStruct) {
                str << ')';
            }
            return;
        }
        case CallOp::BYTE_BUFFER_WRITE:
        case CallOp::BYTE_BUFFER_VOLATILE_WRITE: {
            bool is_volatile = expr->op() == CallOp::BYTE_BUFFER_VOLATILE_WRITE;
            if (is_volatile) {
                mark_coherent(args[0]);
                str << "_volatile"sv;
            }
            str << "_bytebfwrite"sv;
            auto elem = args[2]->type();
            bool aliasStruct = TypeIsAliased(elem);
            if (elem == Type::of<float3>()) {
                str << "Vec3("sv;
                args[0]->accept(vis);
                str << ',';
                GetTypeName(*elem->element(), str, Usage::NONE);
                str << ',';
                args[1]->accept(vis);
                str << ',';
                args[2]->accept(vis);
                if (opt->enable_debug_info && !is_volatile) {
                    str << ',';
                    PrintValidationBound(args[0]);
                }
                str << ')';
                return;
            } else if (elem->is_matrix()) {
                str << "Mat(";
                args[0]->accept(vis);
                str << ',';
                switch (elem->dimension()) {
                    case 2:
                        str << "_WrappedFloat2x2"sv;
                        break;
                    case 3:
                        str << "_WrappedFloat3x3"sv;
                        break;
                    case 4:
                        str << "_WrappedFloat4x4"sv;
                        break;
                }
                str << ',';
                args[1]->accept(vis);
                str << ',';
                if (aliasStruct) {
                    OriginToAliased(args.back()->type(), str);
                    str << '(';
                    args[2]->accept(vis);
                    str << ')';
                } else {
                    args[2]->accept(vis);
                }
                if (opt->enable_debug_info && !is_volatile) {
                    str << ',';
                    PrintValidationBound(args[0]);
                }
                str << ')';
                return;
            }
        } break;
        case CallOp::BYTE_BUFFER_SIZE: {
            str << "_bytebfsize"sv;
        } break;
        case CallOp::TEXTURE_SIZE: {
            str << "_texsize"sv;
        } break;
        case CallOp::RAY_TRACING_TRACE_CLOSEST:
            str << "_TraceClosest"sv;
            break;
        case CallOp::RAY_TRACING_TRACE_ANY:
            str << "_TraceAny"sv;
            break;
        case CallOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR: {
            // Motion blur trace: args are (accel, ray, time, mask)
            if (opt->isRayTracing) {
                str << "_TraceClosestMotion("sv;
                args[0]->accept(vis);// accel
                str << ',';
                args[1]->accept(vis);// ray
                str << ',';
                args[2]->accept(vis);// time argument (was previously ignored)
                str << ',';
                args[3]->accept(vis);// mask
                str << ')';
            } else {
                // Fallback for devices without motion blur support:
                // ignore time and use standard _TraceClosest
                str << "_TraceClosest("sv;
                args[0]->accept(vis);// accel
                str << ',';
                args[1]->accept(vis);// ray
                str << ',';
                args[3]->accept(vis);// mask
                str << ')';
            }
            return;
        }
        // case CallOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR:
        //     // Motion blur trace any: args are (accel, ray, time, mask)
        //     // Map to non-motion _TraceAny(accel, ray, mask), ignoring time
        //     str << "_TraceAny("sv;
        //     args[0]->accept(vis);// accel
        //     str << ',';
        //     args[1]->accept(vis);// ray
        //     str << ',';
        //     args[3]->accept(vis);// mask (skip time at index 2)
        //     str << ')';
        //     return;
        case CallOp::RAY_TRACING_QUERY_ALL:
            str << "_QueryAll("sv;
            PrintArgs();
            return;
        case CallOp::RAY_TRACING_QUERY_ANY:
            str << "_QueryAny("sv;
            PrintArgs();
            return;
        case CallOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR:
            LUISA_ERROR("RAY_TRACING_QUERY_ALL_MOTION_BLUR not supported.");
            break;
        case CallOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR:
            LUISA_ERROR("RAY_TRACING_QUERY_ANY_MOTION_BLUR not supported.");
            break;
        case CallOp::BINDLESS_BUFFER_SIZE: {
            str << "_bdlsBfSize"sv;
            opt->useBufferBindless = true;
            str << '(';
            for (auto &&i : args) {
                i->accept(vis);
                str << ',';
            }
            str << "bdls)"sv;
            return;
        }
        case CallOp::BINDLESS_BUFFER_READ: {
            bool aliasStruct = TypeIsAliased(expr->type());
            if (aliasStruct) {
                AliasedToOrigin(expr->type(), str);
                str << '(';
            }
            str << "_READ_BUFFER"sv;
            opt->useBufferBindless = true;
            str << '(';
            for (auto &&i : args) {
                i->accept(vis);
                str << ',';
            }
            vstd::to_string(expr->type()->size(), str);
            str << ',';
            if (aliasStruct) {
                str << opt->CreateAliasedStruct(expr->type()).first;
            } else {
                GetTypeName(*expr->type(), str, Usage::READ, true);
            }
            str << ",bdls"sv;
            PrintValidationBoundArgument(args[0]);
            str << ')';
            if (aliasStruct) {
                str << ')';
            }
            return;
        }
        case CallOp::BINDLESS_BYTE_BUFFER_READ: {
            bool aliasStruct = TypeIsAliased(expr->type());
            if (aliasStruct) {
                AliasedToOrigin(expr->type(), str);
                str << '(';
            }
            str << "_READ_BUFFER_BYTES"sv;
            opt->useBufferBindless = true;
            str << '(';
            for (auto &&i : args) {
                i->accept(vis);
                str << ',';
            }
            if (aliasStruct) {
                str << opt->CreateAliasedStruct(expr->type()).first;
            } else {
                GetTypeName(*expr->type(), str, Usage::READ, true);
            }
            str << ",bdls"sv;
            PrintValidationBoundArgument(args[0]);
            str << ')';
            if (aliasStruct) {
                str << ')';
            }
            return;
        }
        case CallOp::TYPED_BINDLESS_BUFFER_SIZE: {
            str << "_typed_bdlsBfSize"sv;
            opt->useBufferBindless = true;
            str << '(';
            for (auto &&i : args) {
                i->accept(vis);
                str << ',';
            }
            str << "bdls)"sv;
            return;
        }
        case CallOp::TYPED_BINDLESS_BUFFER_READ: {
            bool aliasStruct = TypeIsAliased(expr->type());
            if (aliasStruct) {
                AliasedToOrigin(expr->type(), str);
                str << '(';
            }
            str << "_typed_READ_BUFFER"sv;
            opt->useBufferBindless = true;
            str << '(';
            for (auto &&i : args) {
                i->accept(vis);
                str << ',';
            }
            vstd::to_string(expr->type()->size(), str);
            str << ',';
            if (aliasStruct) {
                str << opt->CreateAliasedStruct(expr->type()).first;
            } else {
                GetTypeName(*expr->type(), str, Usage::READ, true);
            }
            str << ",bdls"sv;
            PrintValidationBoundArgument(args[0]);
            str << ')';
            if (aliasStruct) {
                str << ')';
            }
            return;
        }
        case CallOp::TYPED_BINDLESS_BYTE_BUFFER_READ: {
            bool aliasStruct = TypeIsAliased(expr->type());
            if (aliasStruct) {
                AliasedToOrigin(expr->type(), str);
                str << '(';
            }
            str << "_typed_READ_BUFFER_BYTES"sv;
            opt->useBufferBindless = true;
            str << '(';
            for (auto &&i : args) {
                i->accept(vis);
                str << ',';
            }
            if (aliasStruct) {
                str << opt->CreateAliasedStruct(expr->type()).first;
            } else {
                GetTypeName(*expr->type(), str, Usage::READ, true);
            }
            str << ",bdls"sv;
            PrintValidationBoundArgument(args[0]);
            str << ')';
            if (aliasStruct) {
                str << ')';
            }
            return;
        }
        case CallOp::TYPED_UNIFORM_BINDLESS_BUFFER_SIZE: {
            str << "_typed_uniform_bdlsBfSize"sv;
            opt->useBufferBindless = true;
            str << '(';
            for (auto &&i : args) {
                i->accept(vis);
                str << ',';
            }
            str << "bdls)"sv;
            return;
        }
        case CallOp::TYPED_UNIFORM_BINDLESS_BUFFER_READ: {
            bool aliasStruct = TypeIsAliased(expr->type());
            if (aliasStruct) {
                AliasedToOrigin(expr->type(), str);
                str << '(';
            }
            str << "_typed_uniform_READ_BUFFER"sv;
            opt->useBufferBindless = true;
            str << '(';
            for (auto &&i : args) {
                i->accept(vis);
                str << ',';
            }
            vstd::to_string(expr->type()->size(), str);
            str << ',';
            if (aliasStruct) {
                str << opt->CreateAliasedStruct(expr->type()).first;
            } else {
                GetTypeName(*expr->type(), str, Usage::READ, true);
            }
            str << ",bdls"sv;
            PrintValidationBoundArgument(args[0]);
            str << ')';
            if (aliasStruct) {
                str << ')';
            }
            return;
        }
        case CallOp::TYPED_UNIFORM_BINDLESS_BYTE_BUFFER_READ: {
            bool aliasStruct = TypeIsAliased(expr->type());
            if (aliasStruct) {
                AliasedToOrigin(expr->type(), str);
                str << '(';
            }
            str << "_typed_uniform_READ_BUFFER_BYTES"sv;
            opt->useBufferBindless = true;
            str << '(';
            for (auto &&i : args) {
                i->accept(vis);
                str << ',';
            }
            if (aliasStruct) {
                str << opt->CreateAliasedStruct(expr->type()).first;
            } else {
                GetTypeName(*expr->type(), str, Usage::READ, true);
            }
            str << ",bdls"sv;
            PrintValidationBoundArgument(args[0]);
            str << ')';
            if (aliasStruct) {
                str << ')';
            }
            return;
        }
        case CallOp::UNIFORM_BINDLESS_BUFFER_SIZE: {
            str << "_uniform_bdlsBfSize"sv;
            opt->useBufferBindless = true;
            str << '(';
            for (auto &&i : args) {
                i->accept(vis);
                str << ',';
            }
            str << "bdls)"sv;
            return;
        }
        case CallOp::UNIFORM_BINDLESS_BUFFER_READ: {
            bool aliasStruct = TypeIsAliased(expr->type());
            if (aliasStruct) {
                AliasedToOrigin(expr->type(), str);
                str << '(';
            }
            str << "_uniform_READ_BUFFER"sv;
            opt->useBufferBindless = true;
            str << '(';
            for (auto &&i : args) {
                i->accept(vis);
                str << ',';
            }
            vstd::to_string(expr->type()->size(), str);
            str << ',';
            if (aliasStruct) {
                str << opt->CreateAliasedStruct(expr->type()).first;
            } else {
                GetTypeName(*expr->type(), str, Usage::READ, true);
            }
            str << ",bdls"sv;
            PrintValidationBoundArgument(args[0]);
            str << ')';
            if (aliasStruct) {
                str << ')';
            }
            return;
        }
        case CallOp::UNIFORM_BINDLESS_BYTE_BUFFER_READ: {
            bool aliasStruct = TypeIsAliased(expr->type());
            if (aliasStruct) {
                AliasedToOrigin(expr->type(), str);
                str << '(';
            }
            str << "_uniform_READ_BUFFER_BYTES"sv;
            opt->useBufferBindless = true;
            str << '(';
            for (auto &&i : args) {
                i->accept(vis);
                str << ',';
            }
            if (aliasStruct) {
                str << opt->CreateAliasedStruct(expr->type()).first;
            } else {
                GetTypeName(*expr->type(), str, Usage::READ, true);
            }
            str << ",bdls"sv;
            PrintValidationBoundArgument(args[0]);
            str << ')';
            if (aliasStruct) {
                str << ')';
            }
            return;
        }
        case CallOp::ASSERT:
        case CallOp::ASSUME:
            return;
        case CallOp::UNREACHABLE: {
            if (auto t = expr->type()) {
                str << "("sv;
                GetTypeName(*t, str, Usage::READ, true);
                str << ")0"sv;
            }
            return;
        }
        case CallOp::FLATTEN:
            opt->cond_opt_value = (CodegenStackData::CondOptValue)(luisa::to_underlying(opt->cond_opt_value) | luisa::to_underlying(CodegenStackData::CondOptValue::Flatten));
            return;
        case CallOp::BRANCH:
            opt->cond_opt_value = (CodegenStackData::CondOptValue)(luisa::to_underlying(opt->cond_opt_value) | luisa::to_underlying(CodegenStackData::CondOptValue::Branch));
            return;
        case CallOp::FORCE_CASE:
            opt->cond_opt_value = (CodegenStackData::CondOptValue)(luisa::to_underlying(opt->cond_opt_value) | luisa::to_underlying(CodegenStackData::CondOptValue::ForceCase));
            return;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE:
            opt->useTex2DBindless = true;
            if (opt->isPixelShader) {
                str << "_SampleTex2DPixel"sv;
            } else {
                str << "_SampleTex2D"sv;
            }
            break;
        case CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE:
            opt->useTex2DBindless = true;
            if (opt->isPixelShader) {
                str << "_uniform_SampleTex2DPixel"sv;
            } else {
                str << "_uniform_SampleTex2D"sv;
            }
            break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER:
            opt->useTex2DBindless = true;
            if (opt->isPixelShader) {
                str << "_SampleTex2DPixelSmp"sv;
            } else {
                str << "_SampleTex2DSmp"sv;
            }
            break;
        case CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_SAMPLER:
            opt->useTex2DBindless = true;
            if (opt->isPixelShader) {
                str << "_uniform_SampleTex2DPixelSmp"sv;
            } else {
                str << "_uniform_SampleTex2DSmp"sv;
            }
            break;

        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL:
            opt->useTex2DBindless = true;
            str << "_SampleTex2DLevel"sv;
            break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER:
            opt->useTex2DBindless = true;
            str << "_SampleTex2DLevelSmp"sv;
            break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD:
            opt->useTex2DBindless = true;
            str << "_SampleTex2DGrad"sv;
            break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER:
            opt->useTex2DBindless = true;
            str << "_SampleTex2DGradSmp"sv;
            break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL:
            opt->useTex2DBindless = true;
            str << "_SampleTex2DGradLevel"sv;
            break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER:
            opt->useTex2DBindless = true;
            str << "_SampleTex2DGradLevelSmp"sv;
            break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE:
            opt->useTex3DBindless = true;
            str << "_SampleTex3D"sv;
            break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER:
            opt->useTex3DBindless = true;
            str << "_SampleTex3DSmp"sv;
            break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL:
            opt->useTex3DBindless = true;
            str << "_SampleTex3DLevel"sv;
            break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER:
            opt->useTex3DBindless = true;
            str << "_SampleTex3DLevelSmp"sv;
            break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD:
            opt->useTex3DBindless = true;
            str << "_SampleTex3DGrad"sv;
            break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER:
            opt->useTex3DBindless = true;
            str << "_SampleTex3DGradSmp"sv;
            break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL:
            opt->useTex3DBindless = true;
            str << "_SampleTex3DGradLevel"sv;
            break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER:
            opt->useTex3DBindless = true;
            str << "_SampleTex3DGradLevelSmp"sv;
            break;
        case CallOp::BINDLESS_TEXTURE2D_READ:
            opt->useTex2DBindless = true;
            str << "_ReadTex2D"sv;
            break;
        case CallOp::BINDLESS_TEXTURE2D_READ_LEVEL:
            opt->useTex2DBindless = true;
            str << "_ReadTex2DLevel"sv;
            break;
        case CallOp::BINDLESS_TEXTURE3D_READ:
            opt->useTex3DBindless = true;
            str << "_ReadTex3D"sv;
            break;
        case CallOp::BINDLESS_TEXTURE3D_READ_LEVEL:
            opt->useTex3DBindless = true;
            str << "_ReadTex3DLevel"sv;
            break;
        case CallOp::BINDLESS_TEXTURE2D_SIZE:
            opt->useTex2DBindless = true;
            str << "_Tex2DSize"sv;
            break;
        case CallOp::BINDLESS_TEXTURE2D_SIZE_LEVEL:
            opt->useTex2DBindless = true;
            str << "_Tex2DSizeLevel"sv;
            break;
        case CallOp::BINDLESS_TEXTURE3D_SIZE:
            opt->useTex3DBindless = true;
            str << "_Tex3DSize"sv;
            break;
        case CallOp::BINDLESS_TEXTURE3D_SIZE_LEVEL:
            opt->useTex3DBindless = true;
            str << "_Tex3DSizeLevel"sv;
            break;
        case CallOp::TYPED_BINDLESS_TEXTURE2D_SAMPLE_SAMPLER:
            opt->useTex2DBindless = true;
            if (opt->isPixelShader) {
                str << "_typed_SampleTex2DPixelSmp"sv;
            } else {
                str << "_typed_SampleTex2DSmp"sv;
            }
            break;
        case CallOp::TYPED_BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER:
            opt->useTex2DBindless = true;
            str << "_typed_SampleTex2DLevelSmp"sv;
            break;
        case CallOp::TYPED_BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER:
            opt->useTex2DBindless = true;
            str << "_typed_SampleTex2DGradSmp"sv;
            break;
        case CallOp::TYPED_BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER:
            opt->useTex2DBindless = true;
            str << "_typed_SampleTex2DGradLevelSmp"sv;
            break;
        case CallOp::TYPED_BINDLESS_TEXTURE3D_SAMPLE_SAMPLER:
            opt->useTex3DBindless = true;
            str << "_typed_SampleTex3DSmp"sv;
            break;
        case CallOp::TYPED_BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER:
            opt->useTex3DBindless = true;
            str << "_typed_SampleTex3DLevelSmp"sv;
            break;
        case CallOp::TYPED_BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER:
            opt->useTex3DBindless = true;
            str << "_typed_SampleTex3DGradSmp"sv;
            break;
        case CallOp::TYPED_BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER:
            opt->useTex3DBindless = true;
            str << "_typed_SampleTex3DGradLevelSmp"sv;
            break;
        case CallOp::TYPED_BINDLESS_TEXTURE2D_READ:
            opt->useTex2DBindless = true;
            str << "_typed_ReadTex2D"sv;
            break;
        case CallOp::TYPED_BINDLESS_TEXTURE2D_READ_LEVEL:
            opt->useTex2DBindless = true;
            str << "_typed_ReadTex2DLevel"sv;
            break;
        case CallOp::TYPED_BINDLESS_TEXTURE3D_READ:
            opt->useTex3DBindless = true;
            str << "_typed_ReadTex3D"sv;
            break;
        case CallOp::TYPED_BINDLESS_TEXTURE3D_READ_LEVEL:
            opt->useTex3DBindless = true;
            str << "_typed_ReadTex3DLevel"sv;
            break;
        case CallOp::TYPED_BINDLESS_TEXTURE2D_SIZE:
            opt->useTex2DBindless = true;
            str << "_typed_Tex2DSize"sv;
            break;
        case CallOp::TYPED_BINDLESS_TEXTURE2D_SIZE_LEVEL:
            opt->useTex2DBindless = true;
            str << "_typed_Tex2DSizeLevel"sv;
            break;
        case CallOp::TYPED_BINDLESS_TEXTURE3D_SIZE:
            opt->useTex3DBindless = true;
            str << "_typed_Tex3DSize"sv;
            break;
        case CallOp::TYPED_BINDLESS_TEXTURE3D_SIZE_LEVEL:
            opt->useTex3DBindless = true;
            str << "_typed_Tex3DSizeLevel"sv;
            break;
        case CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_SAMPLER:
            opt->useTex2DBindless = true;
            if (opt->isPixelShader) {
                str << "_typed_uniform_SampleTex2DPixelSmp"sv;
            } else {
                str << "_typed_uniform_SampleTex2DSmp"sv;
            }
            break;
        case CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER:
            opt->useTex2DBindless = true;
            str << "_typed_uniform_SampleTex2DLevelSmp"sv;
            break;
        case CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER:
            opt->useTex2DBindless = true;
            str << "_typed_uniform_SampleTex2DGradSmp"sv;
            break;
        case CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER:
            opt->useTex2DBindless = true;
            str << "_typed_uniform_SampleTex2DGradLevelSmp"sv;
            break;
        case CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_SAMPLER:
            opt->useTex3DBindless = true;
            str << "_typed_uniform_SampleTex3DSmp"sv;
            break;
        case CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER:
            opt->useTex3DBindless = true;
            str << "_typed_uniform_SampleTex3DLevelSmp"sv;
            break;
        case CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER:
            opt->useTex3DBindless = true;
            str << "_typed_uniform_SampleTex3DGradSmp"sv;
            break;
        case CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER:
            opt->useTex3DBindless = true;
            str << "_typed_uniform_SampleTex3DGradLevelSmp"sv;
            break;
        case CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE2D_READ:
            opt->useTex2DBindless = true;
            str << "_typed_uniform_ReadTex2D"sv;
            break;
        case CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE2D_READ_LEVEL:
            opt->useTex2DBindless = true;
            str << "_typed_uniform_ReadTex2DLevel"sv;
            break;
        case CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE3D_READ:
            opt->useTex3DBindless = true;
            str << "_typed_uniform_ReadTex3D"sv;
            break;
        case CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE3D_READ_LEVEL:
            opt->useTex3DBindless = true;
            str << "_typed_uniform_ReadTex3DLevel"sv;
            break;
        case CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE2D_SIZE:
            opt->useTex2DBindless = true;
            str << "_typed_uniform_Tex2DSize"sv;
            break;
        case CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE2D_SIZE_LEVEL:
            opt->useTex2DBindless = true;
            str << "_typed_uniform_Tex2DSizeLevel"sv;
            break;
        case CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE3D_SIZE:
            opt->useTex3DBindless = true;
            str << "_typed_uniform_Tex3DSize"sv;
            break;
        case CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE3D_SIZE_LEVEL:
            opt->useTex3DBindless = true;
            str << "_typed_uniform_Tex3DSizeLevel"sv;
            break;
        case CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER:
            opt->useTex2DBindless = true;
            str << "_uniform_SampleTex2DLevelSmp"sv;
            break;
        case CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER:
            opt->useTex2DBindless = true;
            str << "_uniform_SampleTex2DGradSmp"sv;
            break;
        case CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER:
            opt->useTex2DBindless = true;
            str << "_uniform_SampleTex2DGradLevelSmp"sv;
            break;
        case CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_SAMPLER:
            opt->useTex3DBindless = true;
            str << "_uniform_SampleTex3DSmp"sv;
            break;
        case CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER:
            opt->useTex3DBindless = true;
            str << "_uniform_SampleTex3DLevelSmp"sv;
            break;
        case CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER:
            opt->useTex3DBindless = true;
            str << "_uniform_SampleTex3DGradSmp"sv;
            break;
        case CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER:
            opt->useTex3DBindless = true;
            str << "_uniform_SampleTex3DGradLevelSmp"sv;
            break;
        case CallOp::UNIFORM_BINDLESS_TEXTURE2D_READ:
            opt->useTex2DBindless = true;
            str << "_uniform_ReadTex2D"sv;
            break;
        case CallOp::UNIFORM_BINDLESS_TEXTURE2D_READ_LEVEL:
            opt->useTex2DBindless = true;
            str << "_uniform_ReadTex2DLevel"sv;
            break;
        case CallOp::UNIFORM_BINDLESS_TEXTURE3D_READ:
            opt->useTex3DBindless = true;
            str << "_uniform_ReadTex3D"sv;
            break;
        case CallOp::UNIFORM_BINDLESS_TEXTURE3D_READ_LEVEL:
            opt->useTex3DBindless = true;
            str << "_uniform_ReadTex3DLevel"sv;
            break;
        case CallOp::UNIFORM_BINDLESS_TEXTURE2D_SIZE:
            opt->useTex2DBindless = true;
            str << "_uniform_Tex2DSize"sv;
            break;
        case CallOp::UNIFORM_BINDLESS_TEXTURE2D_SIZE_LEVEL:
            opt->useTex2DBindless = true;
            str << "_uniform_Tex2DSizeLevel"sv;
            break;
        case CallOp::UNIFORM_BINDLESS_TEXTURE3D_SIZE:
            opt->useTex3DBindless = true;
            str << "_uniform_Tex3DSize"sv;
            break;
        case CallOp::UNIFORM_BINDLESS_TEXTURE3D_SIZE_LEVEL:
            opt->useTex3DBindless = true;
            str << "_uniform_Tex3DSizeLevel"sv;
            break;
        case CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_LEVEL:
            opt->useTex2DBindless = true;
            str << "_uniform_SampleTex2DLevel"sv;
            break;
        case CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD:
            opt->useTex2DBindless = true;
            str << "_uniform_SampleTex2DGrad"sv;
            break;
        case CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL:
            opt->useTex2DBindless = true;
            str << "_uniform_SampleTex2DGradLevel"sv;
            break;
        case CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE:
            opt->useTex3DBindless = true;
            str << "_uniform_SampleTex3D"sv;
            break;
        case CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_LEVEL:
            opt->useTex3DBindless = true;
            str << "_uniform_SampleTex3DLevel"sv;
            break;
        case CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD:
            opt->useTex3DBindless = true;
            str << "_uniform_SampleTex3DGrad"sv;
            break;
        case CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL:
            opt->useTex3DBindless = true;
            str << "_uniform_SampleTex3DGradLevel"sv;
            break;
        case CallOp::SYNCHRONIZE_BLOCK:
            str << "GroupMemoryBarrierWithGroupSync()"sv;
            return;
        case CallOp::RASTER_DISCARD:
            LUISA_ASSERT(opt->isPixelShader, "Raster-Discard can only be used in pixel shader");
            str << "discard";
            return;
        case CallOp::RASTER_SET_Z_DEPTH:
            LUISA_ASSERT(opt->isPixelShader, "Raster-Discard can only be used in pixel shader");
            str << "_z_depth=";
            args[0]->accept(vis);
            return;
        case CallOp::RASTER_SET_Z_DEPTH_GREATER_EQUAL:
            LUISA_ASSERT(opt->isPixelShader, "Raster-Discard can only be used in pixel shader");
            str << "_z_depth_gequal=";
            args[0]->accept(vis);
            return;
        case CallOp::RASTER_SET_Z_DEPTH_LESS_EQUAL:
            LUISA_ASSERT(opt->isPixelShader, "Raster-Discard can only be used in pixel shader");
            str << "_z_depth_lequal=";
            args[0]->accept(vis);
            return;
        case CallOp::DDX: {
            if (opt->isRaster) {
                LUISA_ASSERT(opt->isPixelShader, "ddx can only be used in pixel shader");
                str << "ddx"sv;
            } else {
                str << "_ddx"sv;
            }
        } break;
        case CallOp::DDY: {
            if (opt->isRaster) {
                LUISA_ASSERT(opt->isPixelShader, "ddy can only be used in pixel shader");
                str << "ddy"sv;
            } else {
                str << "_ddy"sv;
            }
        } break;
        case CallOp::RAY_TRACING_INSTANCE_TRANSFORM: {
            str << "_InstMatrix("sv;
            args[0]->accept(vis);
            str << "Inst,"sv;
            args[1]->accept(vis);
            str << ')';
            return;
        }
        case CallOp::RAY_TRACING_INSTANCE_USER_ID: {
            str << "_InstId("sv;
            args[0]->accept(vis);
            str << "Inst,"sv;
            args[1]->accept(vis);
            str << ')';
            return;
        }
        case CallOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK: {
            str << "_InstVis("sv;
            args[0]->accept(vis);
            str << "Inst,"sv;
            args[1]->accept(vis);
            str << ')';
            return;
        }
        case CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM: {
            str << "_SetAccelTransform("sv;
            args[0]->accept(vis);
            str << "Inst,"sv;
            PrintArgs(1);
            str << ')';
            return;
        }
        case CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY: {
            str << "_SetAccelVis("sv;
            args[0]->accept(vis);
            str << "Inst,"sv;
            PrintArgs(1);
            str << ')';
            return;
        }
        case CallOp::RAY_TRACING_SET_INSTANCE_OPACITY: {
            str << "_SetAccelOpaque("sv;
            args[0]->accept(vis);
            str << "Inst,"sv;
            PrintArgs(1);
            str << ')';
            return;
        }
        case CallOp::RAY_TRACING_SET_INSTANCE_USER_ID: {
            str << "_SetUserId("sv;
            args[0]->accept(vis);
            str << "Inst,"sv;
            PrintArgs(1);
            str << ')';
            return;
        }
        case CallOp::INDIRECT_SET_DISPATCH_COUNT: {
            str << "_SetDispCount"sv;
        } break;
        case CallOp::INDIRECT_SET_DISPATCH_KERNEL: {
            str << "_SetDispInd"sv;
        } break;
        case CallOp::RAY_QUERY_WORLD_SPACE_RAY:
            str << "_RayQueryGetWorldRay<"sv;
            GetTypeName(*expr->type(), str, Usage::NONE, false);
            str << ',';
            GetTypeName(*args[0]->type(), str, Usage::NONE, false);
            str << '>';
            break;
        case CallOp::RAY_QUERY_TRIANGLE_CANDIDATE_HIT:
            str << "_GetTriangleCandidateHit"sv;
            break;
        case CallOp::RAY_QUERY_PROCEDURAL_CANDIDATE_HIT:
            str << "_GetProceduralCandidateHit"sv;
            break;
        case CallOp::RAY_QUERY_COMMITTED_HIT:
            str << "_GetCommitedHit"sv;
            break;
        case CallOp::RAY_QUERY_COMMIT_TRIANGLE:
            args[0]->accept(vis);
            str << ".CommitNonOpaqueTriangleHit()"sv;
            return;
        case CallOp::RAY_QUERY_COMMIT_PROCEDURAL:
            str << "_CommitProcedural"sv;
            break;
        case CallOp::RAY_QUERY_TERMINATE:
            args[0]->accept(vis);
            str << ".Abort()"sv;
            return;
        case CallOp::RAY_QUERY_PROCEED:
            args[0]->accept(vis);
            str << ".Proceed()"sv;
            return;
        case CallOp::RAY_QUERY_IS_TRIANGLE_CANDIDATE:
            str << '(';
            args[0]->accept(vis);
            str << ".CandidateType()==CANDIDATE_NON_OPAQUE_TRIANGLE)"sv;
            return;
        case CallOp::RAY_QUERY_IS_PROCEDURAL_CANDIDATE:
            str << '(';
            args[0]->accept(vis);
            str << ".CandidateType()!=CANDIDATE_NON_OPAQUE_TRIANGLE)"sv;
            return;
        case CallOp::ZERO: {
            str << "_zero("sv;
            GetTypeName(*expr->type(), str, Usage::READ, true);
            str << ')';
            return;
        }
        case CallOp::ONE: {
            str << "_one("sv;
            GetTypeName(*expr->type(), str, Usage::READ, true);
            str << ')';
            return;
        }
        case CallOp::REQUIRES_GRADIENT: {
            str << "_REQUIRES_GRAD("sv;
            for (auto &&i : args) {
                i->accept(vis);
                str << ',';
            }
            GetTypeName(*args[0]->type(), str, Usage::READ, true);
            str << ')';
            return;
        }
        case CallOp::GRADIENT:
            str << "_GRAD";
            break;
        case CallOp::GRADIENT_MARKER:
            str << "_MARK_GRAD";
            break;
        case CallOp::ACCUMULATE_GRADIENT:
            LUISA_ASSERT(args.size() == 2, "accumulate_gradient must have 2 arguments");
            str << "_accum_grad";
            if (args[0]->type()->is_structure() || args[0]->type()->is_array()) {
                str << luisa::format("_{:016X}", args[0]->type()->hash());
            }
            break;
        case CallOp::DETACH:
            str << "_detach";
            break;
        case CallOp::REDUCE_SUM: str << "_reduce_sum"; break;
        case CallOp::REDUCE_PRODUCT: str << "_reduce_prod"; break;
        case CallOp::REDUCE_MIN: str << "_reduce_min"; break;
        case CallOp::REDUCE_MAX: str << "_reduce_max"; break;
        case CallOp::OUTER_PRODUCT: str << "_outer_product"; break;
        case CallOp::MATRIX_COMPONENT_WISE_MULTIPLICATION: str << "_mat_comp_mul"; break;
        case CallOp::BINDLESS_BUFFER_TYPE: LUISA_NOT_IMPLEMENTED(); break;
        case CallOp::TYPED_BINDLESS_BUFFER_TYPE: LUISA_NOT_IMPLEMENTED(); break;
        case CallOp::WARP_IS_FIRST_ACTIVE_LANE:
            str << "WaveIsFirstLane"sv;
            break;
        case CallOp::WARP_ACTIVE_ALL_EQUAL:
            str << "WaveActiveAllEqual"sv;
            break;
        case CallOp::WARP_ACTIVE_BIT_AND:
            str << "WaveActiveBitAnd"sv;
            break;
        case CallOp::WARP_ACTIVE_BIT_OR:
            str << "WaveActiveBitOr"sv;
            break;
        case CallOp::WARP_ACTIVE_BIT_XOR:
            str << "WaveActiveBitXor"sv;
            break;
        case CallOp::WARP_ACTIVE_COUNT_BITS:
            str << "WaveActiveCountBits"sv;
            break;
        case CallOp::WARP_PREFIX_COUNT_BITS:
            str << "WavePrefixCountBits"sv;
            break;
        case CallOp::WARP_ACTIVE_MAX:
            str << "WaveActiveMax"sv;
            break;
        case CallOp::WARP_ACTIVE_MIN:
            str << "WaveActiveMin"sv;
            break;
        case CallOp::WARP_PREFIX_PRODUCT:
            str << "WavePrefixProduct"sv;
            break;
        case CallOp::WARP_ACTIVE_PRODUCT:
            str << "WaveActiveProduct"sv;
            break;
        case CallOp::WARP_PREFIX_SUM:
            str << "WavePrefixSum"sv;
            break;
        case CallOp::WARP_ACTIVE_SUM:
            str << "WaveActiveSum"sv;
            break;
        case CallOp::WARP_ACTIVE_ALL:
            str << "WaveActiveAllTrue"sv;
            break;
        case CallOp::WARP_ACTIVE_ANY:
            str << "WaveActiveAnyTrue"sv;
            break;
        case CallOp::WARP_ACTIVE_BIT_MASK:
            str << "WaveActiveBallot"sv;
            break;
        case CallOp::WARP_READ_LANE:
            str << "WaveReadLaneAt"sv;
            break;
        case CallOp::WARP_READ_FIRST_ACTIVE_LANE:
            str << "WaveReadLaneFirst"sv;
            break;
        case CallOp::BACKWARD:
            LUISA_ERROR_WITH_LOCATION("`backward()` should not be called directly.");
            break;
            // TODO: save save hlsl
        case CallOp::PACK: LUISA_NOT_IMPLEMENTED();
        case CallOp::UNPACK: LUISA_NOT_IMPLEMENTED();
        case CallOp::BINDLESS_BUFFER_WRITE: LUISA_NOT_IMPLEMENTED();
        case CallOp::TYPED_BINDLESS_BUFFER_WRITE: LUISA_NOT_IMPLEMENTED();
        case CallOp::WARP_FIRST_ACTIVE_LANE: LUISA_NOT_IMPLEMENTED();
        case CallOp::TEXTURE2D_SAMPLE:
        case CallOp::TEXTURE3D_SAMPLE:
            if (opt->isPixelShader) {
                str << "_SmptxPixel"sv;
            } else {
                str << "_Smptx"sv;
            }
            break;
        case CallOp::TEXTURE2D_SAMPLE_LEVEL:
        case CallOp::TEXTURE3D_SAMPLE_LEVEL:
            str << "_SmptxLevel"sv;
            break;
        case CallOp::TEXTURE3D_SAMPLE_GRAD:
        case CallOp::TEXTURE2D_SAMPLE_GRAD:
            str << "_SmptxGrad"sv;
            break;
        case CallOp::TEXTURE2D_SAMPLE_GRAD_LEVEL:
            str << "_SmptxGrad2DLevel"sv;
            break;
        case CallOp::TEXTURE3D_SAMPLE_GRAD_LEVEL:
            str << "_SmptxGrad3DLevel"sv;
            break;
        case CallOp::SHADER_EXECUTION_REORDER:
            str << "(void)";
            break;
        case CallOp::COOPERATIVE_OUTER_PRODUCT_ACCUMULATE: {
            auto matrix_dimension = args[1]->type()->coop_matrix_dimension();// weight is KxN
            str << "dx::linalg::CoopOuterProductAccum<";
            GetTypeName(*args[0]->type(), str, args[0]->usage());
            str << ',';
            GetTypeName(*args[2]->type()->element(), str, args[2]->usage());
            str << luisa::format(",{},{},", matrix_dimension.x, matrix_dimension.y);
            TypeToCoop(args[1]->type()->coop_vec_ref_type(), str);
            str << '>';
        } break;
        case CallOp::COOPERATIVE_VECTOR_ACCUMULATE: {
            str << "dx::linalg::CoopVectorAccumulate<";
            GetTypeName(*args[2]->type()->element(), str, args[2]->usage());
            str << luisa::format(",{}>", args[2]->type()->dimension());
        } break;
        case CallOp::COOPERATIVE_VECTOR_LOAD: {
            str << "dx::linalg::CoopVecLoad<";
            GetTypeName(*expr->type()->element(), str, Usage::NONE);
            str << luisa::format(",{}>", expr->type()->dimension());
        } break;
        case CallOp::COOPERATIVE_VECTOR_STORE: {
            str << "dx::linalg::CoopVecStore<";
            GetTypeName(*args[2]->type()->element(), str, args[2]->usage());
            str << luisa::format(",{}>", args[2]->type()->dimension());
        } break;
        case CallOp::COOPERATIVE_VECTOR_SPLAT: {
            str << "dx::linalg::CoopVecSplat<";
            GetTypeName(*expr->type()->element(), str, Usage::NONE);
            str << luisa::format(",{}>", expr->type()->dimension());
        } break;
        case CallOp::COOPERATIVE_VECTOR_CAST: {
            str << "dx::linalg::CoopVecCast<";
            GetTypeName(*expr->type()->element(), str, Usage::NONE);
            str << ',';
            GetTypeName(*args[0]->type()->element(), str, Usage::NONE);
            str << luisa::format(",{}>", expr->type()->dimension());
        } break;
        case CallOp::TYPED_BINDLESS_COOPERATIVE_VECTOR_LOAD:
        case CallOp::BINDLESS_COOPERATIVE_VECTOR_LOAD: {
            opt->useBufferBindless = true;
            str << "dx::linalg::CoopVecLoad<";
            GetTypeName(*expr->type()->element(), str, Usage::NONE);
            str << luisa::format(",{}>(", expr->type()->dimension());
            str << "bdls[NonUniformResourceIndex(";
            if (expr->op() == CallOp::TYPED_BINDLESS_COOPERATIVE_VECTOR_LOAD) {
                PrintTypedBindlessBufferIndex(args[0], args[1]);
            } else {
                str << "_ReadBdlsBuffer(";
                args[0]->accept(vis);
                str << ',';
                args[1]->accept(vis);
                str << ')';
            }
            str << ")],";
            if (expr->op() == CallOp::TYPED_BINDLESS_COOPERATIVE_VECTOR_LOAD) {
                PrintTypedBindlessBufferOffset(args[0], args[1], args[2]);
            } else {
                args[2]->accept(vis);
            }
            str << ')';
        }
            return;
        case CallOp::TYPED_BINDLESS_COOPERATIVE_VECTOR_STORE:
        case CallOp::BINDLESS_COOPERATIVE_VECTOR_STORE: {
            opt->useBufferBindless = true;
            str << "dx::linalg::CoopVecStore<";
            GetTypeName(*args[3]->type()->element(), str, args[3]->usage());
            str << luisa::format(",{}>(", args[3]->type()->dimension());
            str << "bdls[NonUniformResourceIndex(";
            if (expr->op() == CallOp::TYPED_BINDLESS_COOPERATIVE_VECTOR_STORE) {
                PrintTypedBindlessBufferIndex(args[0], args[1]);
            } else {
                str << "_ReadBdlsBuffer(";
                args[0]->accept(vis);
                str << ',';
                args[1]->accept(vis);
                str << ')';
            }
            str << ")],";
            if (expr->op() == CallOp::TYPED_BINDLESS_COOPERATIVE_VECTOR_STORE) {
                PrintTypedBindlessBufferOffset(args[0], args[1], args[2]);
            } else {
                args[2]->accept(vis);
            }
            str << ',';
            args[3]->accept(vis);
            str << ')';
        }
            return;
        case CallOp::COOPERATIVE_VECTOR_WORKGROUP_LOAD: {
            str << "dx::linalg::CoopVecWorkgroupLoad<";
            GetTypeName(*expr->type()->element(), str, Usage::NONE);
            str << ',';
            str << luisa::format("{}", expr->type()->dimension());
            str << ',';
            GetTypeName(*args[0]->type(), str, args[0]->usage());
            str << '>';
        } break;
        case CallOp::COOPERATIVE_VECTOR_WORKGROUP_STORE: {
            str << "dx::linalg::CoopVecWorkgroupStore<";
            GetTypeName(*args[2]->type()->element(), str, args[2]->usage());
            str << ',';
            str << luisa::format("{}", args[2]->type()->dimension());
            str << ',';
            GetTypeName(*args[0]->type(), str, args[0]->usage());
            str << '>';
        } break;
        case CallOp::COOPERATIVE_MUL_ADD: {
            auto matrix_dimension = args[1]->type()->coop_matrix_dimension();// weight is KxN
            str << "dx::linalg::CoopMulAdd<";
            GetTypeName(*args[0]->type(), str, args[0]->usage());
            str << ',';
            GetTypeName(*args[2]->type(), str, args[2]->usage());
            str << ',';
            GetTypeName(*args[4]->type()->element(), str, Usage::NONE);
            str << ',';
            GetTypeName(*expr->type()->element(), str, Usage::NONE);
            str << ',';
            TypeToCoop(args[1]->type()->coop_vec_ref_type(), str);
            str << ',';
            TypeToCoop(args[3]->type()->coop_vec_ref_type(), str);
            str << luisa::format(",{},{}>", matrix_dimension.x, matrix_dimension.y);
        } break;
        case CallOp::TYPED_BINDLESS_COOPERATIVE_MUL_ADD:
        case CallOp::BINDLESS_COOPERATIVE_MUL_ADD: {
            opt->useBufferBindless = true;
            auto matrix_dimension = args[2]->type()->coop_matrix_dimension();// weight is KxN
            str << "dx::linalg::CoopMulAdd<ByteAddressBuffer,ByteAddressBuffer,";
            GetTypeName(*args[5]->type()->element(), str, Usage::NONE);
            str << ',';
            GetTypeName(*expr->type()->element(), str, Usage::NONE);
            str << ',';
            TypeToCoop(args[2]->type()->coop_vec_ref_type(), str);
            str << ',';
            TypeToCoop(args[4]->type()->coop_vec_ref_type(), str);
            str << luisa::format(",{},{}>(", matrix_dimension.x, matrix_dimension.y);
            str << "bdls[NonUniformResourceIndex(";
            if (expr->op() == CallOp::TYPED_BINDLESS_COOPERATIVE_MUL_ADD) {
                PrintTypedBindlessBufferIndex(args[0], args[1]);
            } else {
                str << "_ReadBdlsBuffer(";
                args[0]->accept(vis);
                str << ',';
                args[1]->accept(vis);
                str << ')';
            }
            str << ")],";
            if (expr->op() == CallOp::TYPED_BINDLESS_COOPERATIVE_MUL_ADD) {
                PrintTypedBindlessBufferOffset(args[0], args[1], args[2]);
            } else {
                args[2]->accept(vis);
            }
            str << ",bdls[NonUniformResourceIndex(";
            if (expr->op() == CallOp::TYPED_BINDLESS_COOPERATIVE_MUL_ADD) {
                PrintTypedBindlessBufferIndex(args[0], args[3]);
            } else {
                str << "_ReadBdlsBuffer(";
                args[0]->accept(vis);
                str << ',';
                args[3]->accept(vis);
                str << ')';
            }
            str << ")],";
            if (expr->op() == CallOp::TYPED_BINDLESS_COOPERATIVE_MUL_ADD) {
                PrintTypedBindlessBufferOffset(args[0], args[3], args[4]);
                str << ',';
                args[5]->accept(vis);
            } else {
                args[4]->accept(vis);
                str << ',';
                args[5]->accept(vis);
            }
            str << ')';
        }
            return;
        case CallOp::COOPERATIVE_MUL: {
            auto matrix_dimension = args[1]->type()->coop_matrix_dimension();// weight is KxN
            str << "dx::linalg::CoopMul<";
            GetTypeName(*args[0]->type(), str, args[0]->usage());
            str << ',';
            GetTypeName(*args[2]->type()->element(), str, Usage::NONE);
            str << ',';
            GetTypeName(*expr->type()->element(), str, Usage::NONE);
            str << ',';
            TypeToCoop(args[1]->type()->coop_vec_ref_type(), str);
            str << luisa::format(",{},{}>", matrix_dimension.x, matrix_dimension.y);
        } break;
        case CallOp::ASYNC_COPY: {
            if (!opt->isSpirv) {
                LUISA_NOT_IMPLEMENTED();
            }
            // Emit per-thread copy from source buffer to workgroup scratch.
            // AST args: [scope, dst, src, elem_bytes, num, stride, event]
            str << "/* async_copy */ for(uint _i=0;_i<";
            args[4]->accept(vis);
            str << ";_i++)_vk_wg_copy_buf[(";
            args[1]->accept(vis);
            str << "+_i*";
            args[3]->accept(vis);
            str << ")>>2]=";
            if (opt->kernel) {
                auto kernel_args = opt->kernel.arguments();
                bool found = false;
                for (auto &&v : kernel_args) {
                    if (v.type()->is_buffer()) {
                        auto name = opt->kernel.get_variable_name(v.uid());
                        str << "_bfread(" << name << "_b";
                        vstd::to_string(v.uid(), str);
                        str << ",(";
                        args[2]->accept(vis);
                        str << "+_i*";
                        args[5]->accept(vis);
                        str << ")/";
                        args[3]->accept(vis);
                        str << ");";
                        found = true;
                        break;
                    }
                }
                if (!found) { str << "0/*no buf*/;"; }
            } else { str << "0/*null*/;"; }
            str << "}";
            return;
        }
        case CallOp::PIPELINE_COMMIT:
            // SPIR-V: each OpGroupAsyncCopy is implicitly committed.
            // Pipeline tracking counter increment is handled in _vk_async_copy_impl.
            str << "/* pipeline_commit (no-op in SPIR-V) */";
            return;
        case CallOp::PIPELINE_WAIT_PRIOR:
            if (!opt->isSpirv) {
                LUISA_NOT_IMPLEMENTED();
            }
            // Emit workgroup barrier for synchronization.
            // In a full implementation this would use OpGroupWaitEvents with
            // event tracking. For now, a full memory barrier is correct for
            // pipeline_wait_prior(0) and conservative for N > 0.
            str << "GroupMemoryBarrierWithGroupSync()";
            return;
        case CallOp::TYPED_BINDLESS_COOPERATIVE_MUL:
        case CallOp::BINDLESS_COOPERATIVE_MUL: {
            opt->useBufferBindless = true;
            auto matrix_dimension = args[2]->type()->coop_matrix_dimension();// weight is KxN
            str << "dx::linalg::CoopMul<ByteAddressBuffer,";
            GetTypeName(*args[3]->type()->element(), str, Usage::NONE);
            str << ',';
            GetTypeName(*expr->type()->element(), str, Usage::NONE);
            str << ',';
            TypeToCoop(args[2]->type()->coop_vec_ref_type(), str);
            str << luisa::format(",{},{}>(", matrix_dimension.x, matrix_dimension.y);

            str << "bdls[NonUniformResourceIndex(";
            if (expr->op() == CallOp::TYPED_BINDLESS_COOPERATIVE_MUL) {
                PrintTypedBindlessBufferIndex(args[0], args[1]);
            } else {
                str << "_ReadBdlsBuffer(";
                args[0]->accept(vis);
                str << ',';
                args[1]->accept(vis);
                str << ')';
            }
            str << ")],";
            if (expr->op() == CallOp::TYPED_BINDLESS_COOPERATIVE_MUL) {
                PrintTypedBindlessBufferOffset(args[0], args[1], args[2]);
                str << ',';
                args[3]->accept(vis);
            } else {
                args[2]->accept(vis);
                str << ',';
                args[3]->accept(vis);
            }
            str << ')';
        }
            return;
        default:
            LUISA_ERROR("Bad op. {}", luisa::to_string(expr->op()));
            break;
    }
    str << '(';
    if (expr->is_builtin() && !args.empty() &&
        args[0]->type()->is_texture()) {
        PrintTextureView(args[0], expr->op() == CallOp::TEXTURE_WRITE);
        for (auto argument : args.subspan(1u)) {
            str << ',';
            argument->accept(vis);
        }
    } else {
        PrintArgs();
    }
    if (opt->enable_debug_info) {
        // Append the byte-buffer validation bound for the generic path.
        auto op = expr->op();
        if (op == CallOp::BYTE_BUFFER_WRITE || op == CallOp::BYTE_BUFFER_READ) {
            str << ',';
            PrintValidationBound(args[0]);
        }
    }
    str << ')';
}

void CodegenUtility::CodegenVertex(Function vert, vstd::StringBuilder &result, bool cBufferNonEmpty) {
    CodegenFunction(vert, result, cBufferNonEmpty, false);
    auto args = vert.arguments();
    vstd::StringBuilder retName;
    auto retType = vert.return_type();
    GetTypeName(*retType, retName, Usage::READ);
    result << retName << " main("sv;
    GetTypeName(*args[0].type(), result, Usage::NONE);
    result << " vv){\n"sv;
    if (cBufferNonEmpty) {
        result << "_Args a = _Global[0];\n"sv;
    }
    opt->funcType = CodegenStackData::FuncType::Vert;
    opt->arguments.clear();
    opt->arguments.reserve(args.size() - 1);
    size_t idx = 0;
    for (auto &&i : vstd::make_ite_range(args.subspan(1))) {
        opt->arguments.try_emplace(i.uid(), idx);
        ++idx;
    }
#ifdef LUISA_ENABLE_IR
    vstd::unordered_set<Variable> grad_vars;
    glob_variables_with_grad(vert, grad_vars);
#endif
    {
        StringStateVisitor vis(vert, result, this);
        vis.sharedVariables = &opt->sharedVariable;
        vis.VisitFunction(
#ifdef LUISA_ENABLE_IR
            grad_vars,
#endif
            vert);
    }
    result << "}\n"sv;
}

void CodegenUtility::CodegenPixel(Function pixel, vstd::StringBuilder &result, bool cBufferNonEmpty) {
    opt->isPixelShader = true;
    opt->pixelUseBarycentric = false;
    auto resetPixelShaderKey = vstd::scope_exit([&] { opt->isPixelShader = false; });
    CodegenFunction(pixel, result, cBufferNonEmpty, false);
    vstd::StringBuilder retName;
    auto retType = pixel.return_type();
    GetTypeName(*retType, retName, Usage::READ);
    auto set_depth = pixel.propagated_builtin_callables().test(CallOp::RASTER_SET_Z_DEPTH);
    auto set_depth_lequal = pixel.propagated_builtin_callables().test(CallOp::RASTER_SET_Z_DEPTH_LESS_EQUAL);
    auto set_depth_gequal = pixel.propagated_builtin_callables().test(CallOp::RASTER_SET_Z_DEPTH_GREATER_EQUAL);
    result << retName << " pixel(v2p p,uint primId,float3 bary"sv;
    if (set_depth) {
        result << ",out float _z_depth"sv;
    }
    if (set_depth_lequal) {
        result << ",out float _z_depth_lequal";
    }
    if (set_depth_gequal) {
        result << ",out float _z_depth_gequal";
    }
    result << "){\n"sv;
    if (cBufferNonEmpty) {
        result << "_Args a = _Global[0];\n"sv;
    }
    opt->funcType = CodegenStackData::FuncType::Pixel;
    opt->pixelFirstArgIsStruct = pixel.arguments()[0].type()->is_structure();
    opt->arguments.clear();
    opt->arguments.reserve(pixel.arguments().size());
    size_t idx = 0;
    for (auto &&i : pixel.arguments()) {
        opt->arguments.try_emplace(i.uid(), idx);
        ++idx;
    }
#ifdef LUISA_ENABLE_IR
    vstd::unordered_set<Variable> grad_vars;
    glob_variables_with_grad(pixel, grad_vars);
#endif
    {
        StringStateVisitor vis(pixel, result, this);
        vis.sharedVariables = &opt->sharedVariable;
        vis.VisitFunction(
#ifdef LUISA_ENABLE_IR
            grad_vars,
#endif
            pixel);
    }
    result << "\n}\nvoid main(v2p p"sv;
    result << ",uint primId:SV_PrimitiveID"sv;
    if (opt->pixelUseBarycentric) {
        result << ",float3 bary:SV_Barycentrics"sv;
    }
    if (set_depth) {
        result << ",out float _z_depth:SV_Depth"sv;
    }
    if (set_depth_lequal) {
        result << ",out float _z_depth_lequal:SV_DepthLessEqual"sv;
    }
    if (set_depth_gequal) {
        result << ",out float _z_depth_gequal:SV_DepthGreaterEqual"sv;
    }
    auto write_arg = [&]() {
        if (opt->pixelUseBarycentric) {
            result << ",bary"sv;
        } else {
            result << ",float3(0,0,0)"sv;
        }
        if (set_depth) {
            result << ",_z_depth"sv;
        }
        if (set_depth_lequal) {
            result << ",_z_depth_lequal"sv;
        }
        if (set_depth_gequal) {
            result << ",_z_depth_gequal"sv;
        }
    };
    if (retType->is_scalar() || retType->is_vector()) {
        result << ",out "sv;
        GetTypeName(*retType, result, Usage::READ);
        result << R"( o0:SV_TARGET0){
o0=pixel(p,primId)"sv;
        write_arg();
        result << ");\n}\n"sv;
    } else if (retType->is_structure()) {
        size_t idx = 0;
        for (auto &&i : retType->members()) {
            result << ",out "sv;
            GetTypeName(*i, result, Usage::READ);
            auto num = vstd::to_string(idx);
            result << " o"sv << num << ":SV_TARGET"sv << num;
            ++idx;
        }
        result << "){\n"sv;
        GetTypeName(*retType, result, Usage::READ);
        result << " o=pixel(p,primId"sv;
        write_arg();
        result << ");\n"sv;
        for (auto i : vstd::range(retType->members().size())) {
            auto num = vstd::to_string(static_cast<int64_t>(i));
            result << 'o' << num << "=o.v"sv << num << ";\n"sv;
        }
        result << "}\n"sv;
    } else {
        LUISA_ERROR("Illegal pixel shader return type!");
    }

    // TODO
    // pixel return value
    // value assignment
}
}// namespace lc::hlsl
