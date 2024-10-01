#include "hlsl_codegen.h"
#include <luisa/vstl/string_utility.h>
#include <luisa/ast/constant_data.h>
#include <luisa/ast/type_registry.h>
#include <luisa/ast/function_builder.h>
#include "struct_generator.h"
#include "codegen_stack_data.h"
#include <luisa/core/dynamic_module.h>
#include <luisa/core/logging.h>
#include <luisa/ast/external_function.h>
#include "builtin/hlsl_builtin.hpp"
#include <zlib/zlib.h>
static bool shown_buffer_warning = false;
namespace lc::hlsl {
static std::atomic_bool rootsig_exceed_warned = false;
#ifdef LUISA_ENABLE_IR
static void glob_variables_with_grad(Function f, vstd::unordered_set<Variable> &gradient_variables) noexcept {
    if (f.requires_autodiff())
        traverse_expressions<true>(
            f.body(),
            [&](auto expr) noexcept {
                if (expr->tag() == Expression::Tag::CALL) {
                    if (auto call = static_cast<const CallExpr *>(expr);
                        call->op() == CallOp::GRADIENT ||
                        call->op() == CallOp::GRADIENT_MARKER ||
                        call->op() == CallOp::REQUIRES_GRADIENT) {
                        LUISA_ASSERT(!call->arguments().empty() &&
                                         call->arguments().front()->tag() == Expression::Tag::REF,
                                     "Invalid gradient function call.");
                        auto v = static_cast<const RefExpr *>(call->arguments().front())->variable();
                        gradient_variables.emplace(v);
                    }
                }
            },
            [](auto) noexcept {},
            [](auto) noexcept {});
}
#endif
struct RegisterIndexer {
    virtual void init() = 0;
    virtual uint &get(uint idx) = 0;
};
struct DXILRegisterIndexer : public RegisterIndexer {
    std::array<uint, 3> values;
    void init() override {
        values = {1, 0, 0};
    }
    uint &get(uint idx) override {
        return values[idx];
    }
};
struct SpirVRegisterIndexer : public RegisterIndexer {
    uint count;
    void init() override {
        count = 2;
    }
    uint &get(uint idx) override {
        return count;
    }
};

vstd::string_view CodegenUtility::ReadInternalHLSLFile(vstd::string_view name) {
    struct CachedHeader {
        std::mutex mtx;
        vstd::vector<char> result;
    };
    static vstd::HashMap<vstd::string, CachedHeader> headers;
    static std::mutex header_mtx;
    auto iter = [&]() {
        std::lock_guard lck{header_mtx};
        return headers.emplace(name);
    }();
    auto &v = iter.value();
    {
        std::lock_guard lck{v.mtx};
        if (v.result.empty()) {
            auto compressed = lc_hlsl::get_hlsl_builtin(name);
            v.result.push_back_uninitialized(compressed.uncompressed_size);
            uLong dest_len = compressed.uncompressed_size;
            auto r = uncompress((Bytef *)v.result.data(), &dest_len, (Bytef const *)compressed.ptr, compressed.compressed_size);
            if (r != Z_OK) [[unlikely]] {
                LUISA_ERROR("Uncompress header failed. {}", r);
            }
        }
    }
    return {v.result.data(), v.result.size()};
}
namespace detail {
static size_t AddHeader(CallOpSet const &ops, vstd::StringBuilder &builder, bool isRaster) {
    builder << CodegenUtility::ReadInternalHLSLFile("hlsl_header");
    size_t immutable_size = builder.size();
    if (ops.uses_raytracing()) {
        builder << CodegenUtility::ReadInternalHLSLFile("raytracing_header");
    }
    if (ops.test(CallOp::DETERMINANT)) {
        builder << CodegenUtility::ReadInternalHLSLFile("determinant");
    }
    if (ops.test(CallOp::INVERSE)) {
        builder << CodegenUtility::ReadInternalHLSLFile("inverse");
    }
    if (ops.test(CallOp::INDIRECT_SET_DISPATCH_KERNEL) || ops.test(CallOp::INDIRECT_SET_DISPATCH_COUNT)) {
        builder << CodegenUtility::ReadInternalHLSLFile("indirect");
    }
    if (ops.test(CallOp::BUFFER_SIZE) || ops.test(CallOp::TEXTURE_SIZE) || ops.test(CallOp::BYTE_BUFFER_SIZE)) {
        builder << CodegenUtility::ReadInternalHLSLFile("resource_size");
    }
    bool useBindless = false;
    for (auto i : vstd::range(
             luisa::to_underlying(CallOp::BINDLESS_TEXTURE2D_SAMPLE),
             luisa::to_underlying(CallOp::BINDLESS_BUFFER_TYPE) + 1)) {
        if (ops.test(static_cast<CallOp>(i))) {
            useBindless = true;
            break;
        }
    }
    if (useBindless) {
        builder << CodegenUtility::ReadInternalHLSLFile("bindless_common");
    }
    if (ops.test(CallOp::RAY_TRACING_INSTANCE_TRANSFORM) ||
        ops.test(CallOp::RAY_TRACING_INSTANCE_USER_ID) ||
        ops.test(CallOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK) ||
        ops.test(CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM) ||
        ops.test(CallOp::RAY_TRACING_SET_INSTANCE_OPACITY) ||
        ops.test(CallOp::RAY_TRACING_SET_INSTANCE_USER_ID) ||
        ops.test(CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY)) {
        builder << CodegenUtility::ReadInternalHLSLFile("accel_header");
    }
    if (ops.test(CallOp::COPYSIGN)) {
        builder << CodegenUtility::ReadInternalHLSLFile("copy_sign");
    }
    if (!isRaster && (ops.test(CallOp::DDX) || ops.test(CallOp::DDY))) {
        builder << CodegenUtility::ReadInternalHLSLFile("compute_quad");
    }
    if (ops.uses_autodiff()) {
        builder << CodegenUtility::ReadInternalHLSLFile("auto_diff");
    }
    if (ops.test(CallOp::REDUCE_MAX) ||
        ops.test(CallOp::REDUCE_MIN) ||
        ops.test(CallOp::REDUCE_PRODUCT) ||
        ops.test(CallOp::REDUCE_SUM) ||
        ops.test(CallOp::OUTER_PRODUCT) ||
        ops.test(CallOp::MATRIX_COMPONENT_WISE_MULTIPLICATION)) {
        builder << CodegenUtility::ReadInternalHLSLFile("reduce");
    }
    return immutable_size;
}
}// namespace detail
// static thread_local vstd::unique_ptr<CodegenStackData> opt;
#ifdef USE_SPIRV
CodegenStackData *CodegenUtility::StackData() const { return opt.get(); }
#endif
uint CodegenUtility::IsBool(Type const &type) {
    if (type.tag() == Type::Tag::BOOL) {
        return 1;
    } else if (type.tag() == Type::Tag::VECTOR && type.element()->tag() == Type::Tag::BOOL) {
        return type.dimension();
    }
    return 0;
};
vstd::StringBuilder CodegenUtility::GetNewTempVarName() {
    vstd::StringBuilder name;
    name << "tmp"sv;
    vstd::to_string(opt->tempCount, name);
    opt->tempCount++;
    return name;
}
void CodegenUtility::RegistStructType(Type const *type) {
    if (type->is_structure() || type->is_array())
        opt->structTypes.try_emplace(type, opt->count++);
    else if (type->is_buffer()) {
        if (type->element())
            RegistStructType(type->element());
    }
}

void CodegenUtility::GetVariableName(Variable::Tag type, uint id, vstd::StringBuilder &str) {
    switch (type) {
        case Variable::Tag::BLOCK_ID:
            str << "grpId"sv;
            break;
        case Variable::Tag::DISPATCH_ID:
            str << "dspId"sv;
            break;
        case Variable::Tag::THREAD_ID:
            str << "thdId"sv;
            break;
        case Variable::Tag::DISPATCH_SIZE:
            if (opt->funcType == CodegenStackData::FuncType::Kernel) {
                str << "dsp_c.xyz"sv;
            } else {
                str << "dsp_c"sv;
            }
            break;
        case Variable::Tag::KERNEL_ID:
            if (opt->funcType == CodegenStackData::FuncType::Kernel) {
                str << "dsp_c.w"sv;
            } else {
                str << "ker"sv;
            }
            break;
        case Variable::Tag::OBJECT_ID:
            LUISA_ASSERT(opt->isRaster, "object id only allowed in raster shader");
            str << "obj_id"sv;
            break;
        case Variable::Tag::WARP_LANE_COUNT:
            if (opt->funcType == CodegenStackData::FuncType::Callable) {
                str << "_wrpct"sv;
            } else {
                str << "WaveGetLaneCount()"sv;
            }
            break;
        case Variable::Tag::WARP_LANE_ID:
            if (opt->funcType == CodegenStackData::FuncType::Callable) {
                str << "_wrpid"sv;
            } else {
                str << "WaveGetLaneIndex()"sv;
            }
            break;
        case Variable::Tag::LOCAL:
            switch (opt->funcType) {
                case CodegenStackData::FuncType::Kernel:
                case CodegenStackData::FuncType::Vert:
                    if (id == opt->appdataId) {
                        str << "vv"sv;
                    } else {
                        if (opt->arguments.find(id) != opt->arguments.end()) {
                            id += opt->argOffset;
                            str << "a.l"sv;
                        } else {
                            str << 'l';
                        }
                        vstd::to_string(id, str);
                    }
                    break;
                case CodegenStackData::FuncType::Pixel: {
                    auto ite = opt->arguments.find(id);
                    if (ite == opt->arguments.end()) {
                        str << 'l';
                        vstd::to_string(id, str);
                    } else {
                        if (ite->second == 0) {
                            if (opt->pixelFirstArgIsStruct) {
                                str << 'p';
                            } else {
                                str << "p.v0"sv;
                            }
                        } else {
                            id += opt->argOffset;
                            str << "a.l"sv;
                            vstd::to_string(id, str);
                        }
                    }
                } break;
                default: {
                    str << 'l';
                    vstd::to_string(id, str);
                }
            }
            break;
        case Variable::Tag::SHARED:
            str << "_s"sv;
            vstd::to_string(id, str);
            break;
        case Variable::Tag::REFERENCE:
            str << 'r';
            vstd::to_string(id, str);
            break;
        case Variable::Tag::BUFFER:
            str << "_b"sv;
            vstd::to_string(id, str);
            break;
        case Variable::Tag::TEXTURE:
            str << "_t"sv;
            vstd::to_string(id, str);
            break;
        case Variable::Tag::BINDLESS_ARRAY:
            str << "_ba"sv;
            vstd::to_string(id, str);
            break;
        case Variable::Tag::ACCEL:
            str << "_ac"sv;
            vstd::to_string(id, str);
            break;
        default:
            str << 'v';
            vstd::to_string(id, str);
            break;
    }
}

void CodegenUtility::GetVariableName(Variable const &type, vstd::StringBuilder &str) {
    GetVariableName(type.tag(), type.uid(), str);
}

bool CodegenUtility::GetConstName(uint64 hash, ConstantData const &data, vstd::StringBuilder &str) {
    auto constCount = opt->GetConstCount(hash);
    str << "c";
    vstd::to_string((constCount.first), str);
    return constCount.second;
}

void CodegenUtility::GetTypeName(Type const &type, vstd::StringBuilder &str, Usage usage, bool local_var) {
    switch (type.tag()) {
        case Type::Tag::BOOL:
            str << "bool"sv;
            return;
        case Type::Tag::FLOAT32:
            str << "float"sv;
            return;
        case Type::Tag::INT32:
            str << "int"sv;
            return;
        case Type::Tag::UINT32:
            str << "uint"sv;
            return;
        case Type::Tag::FLOAT16:
            str << "float16_t"sv;
            return;
        case Type::Tag::FLOAT64:
            str << "float64_t"sv;
            return;
        case Type::Tag::INT16:
            str << "int16_t"sv;
            return;
        case Type::Tag::UINT16:
            str << "uint16_t"sv;
            return;
        case Type::Tag::INT64:
            str << "int64_t"sv;
            return;
        case Type::Tag::UINT64:
            str << "uint64_t"sv;
            return;
        case Type::Tag::MATRIX: {
            GetTypeName(*type.element(), str, usage);
            vstd::to_string(type.dimension(), str);
            str << 'x';
            vstd::to_string((type.dimension() == 3) ? 4 : type.dimension(), str);
        }
            return;
        case Type::Tag::VECTOR: {
            if (type.dimension() != 3 || local_var) {
                GetTypeName(*type.element(), str, usage);
                vstd::to_string((type.dimension()), str);
            } else {
                str << "_w"sv;
                GetTypeName(*type.element(), str, usage);
                vstd::to_string(3, str);
            }
        }
            return;
        case Type::Tag::STRUCTURE:
        case Type::Tag::ARRAY: {
            auto customType = opt->CreateStruct(&type);
            str << customType;
        }
            return;
        case Type::Tag::BUFFER: {
            if ((static_cast<uint>(usage) & static_cast<uint>(Usage::WRITE)) != 0)
                str << "RW"sv;
            auto ele = type.element();
            // StructuredBuffer
            if (ele != nullptr) {
                str << "StructuredBuffer<"sv;
                if (ele->is_matrix()) {
                    auto n = vstd::to_string(ele->dimension());
                    str << "_WrappedFloat"sv << n << 'x' << n;
                } else {
                    if (ele->is_vector() && ele->dimension() == 3) {
                        GetTypeName(*ele->element(), str, usage);
                        str << '4';
                    } else {
                        GetTypeName(*ele, str, usage);
                    }
                }
                str << '>';
            }
            // ByteAddressBuffer
            else {
                str << "ByteAddressBuffer"sv;
            }
        } break;
        case Type::Tag::TEXTURE: {
            if ((static_cast<uint>(usage) & static_cast<uint>(Usage::WRITE)) != 0)
                str << "RW"sv;
            str << "Texture"sv;
            vstd::to_string((type.dimension()), str);
            str << "D<"sv;
            GetTypeName(*type.element(), str, usage);
            if (type.tag() != Type::Tag::VECTOR) {
                str << '4';
            }
            str << '>';
            break;
        }
        case Type::Tag::BINDLESS_ARRAY: {
            str << "StructuredBuffer<uint>"sv;
        } break;
        case Type::Tag::ACCEL: {
            str << "RaytracingAccelerationStructure"sv;
        } break;
        case Type::Tag::CUSTOM: {
            str << '_' << type.description();
        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("Bad.");
            break;
    }
}

void CodegenUtility::GetFunctionDecl(Function func, vstd::StringBuilder &funcDecl) {
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
        data += " custom_"sv;
        vstd::to_string((opt->GetFuncCount(func)), data);
        if (func.arguments().empty()) {
            data += "()"sv;
        } else {
            data += '(';
            for (auto &&i : func.arguments()) {
                Usage usage = func.variable_usage(i.uid());
                if (i.tag() == Variable::Tag::REFERENCE) {
                    if ((static_cast<uint32_t>(usage) & static_cast<uint32_t>(Usage::WRITE)) != 0) {
                        data += "inout "sv;
                    }
                }
                RegistStructType(i.type());

                vstd::StringBuilder varName;
                CodegenUtility::GetVariableName(i, varName);
                if (i.type()->is_accel()) {
                    if ((to_underlying(usage) & to_underlying(Usage::WRITE)) == 0) {
                        CodegenUtility::GetTypeName(*i.type(), data, usage);
                        data << ' ' << varName << ',';
                    }
                    GetTemplateName();
                    data << ' ' << varName << "Inst,"sv;
                } else {
                    GetTypeName(i.type(), usage);
                    data << ' ';
                    data << varName << ',';
                }
            }
            data[data.size() - 1] = ')';
        }
    }
    if (tempIdx > 0) {
        funcDecl << "template<"sv;
        for (auto i : vstd::range(tempIdx)) {
            funcDecl << "typename T"sv;
            vstd::to_string(i, funcDecl);
            funcDecl << ',';
        }
        *(funcDecl.end() - 1) = '>';
    }
    funcDecl << '\n'
             << data;
}
void CodegenUtility::GetFunctionName(Function callable, vstd::StringBuilder &result) {
    result << "custom_"sv << vstd::to_string((opt->GetFuncCount(callable)));
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
        for (auto i : vstd::range(offset, last)) {
            args[i]->accept(vis);
            str << ',';
        }
        args.back()->accept(vis);
    };
    switch (expr->op()) {
        case CallOp::CUSTOM:
            str << "custom_"sv << vstd::to_string((opt->GetFuncCount(expr->custom())));
            str << '(';
            {
                uint64 sz = 0;
                for (auto &&i : args) {
                    if (i->type()->is_accel()) {
                        if ((static_cast<uint>(expr->custom().variable_usage(expr->custom().arguments()[sz].uid())) & static_cast<uint>(Usage::WRITE)) == 0) {
                            i->accept(vis);
                            str << ',';
                        }
                        i->accept(vis);
                        str << "Inst"sv;
                    } else {
                        i->accept(vis);
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
            str << "firstbithigh"sv;
            break;
        case CallOp::CTZ:
            str << "firstbitlow"sv;
            break;
        case CallOp::POPCOUNT:
            str << "countbits"sv;
            break;
        case CallOp::REVERSE:
            str << "reversebits"sv;
            break;
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
            str << "atan2"sv;
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
            auto &chain = opt->GetAtomicFunc(expr->op(), rootVar->variable(), expr->type(), args);
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
            str << "_bfread"sv;
            auto elem = args[0]->type()->element();
            if (IsNumVec3(*elem)) {
                str << "Vec3"sv;
            } else if (elem->is_matrix()) {
                str << "Mat";
            }
        } break;
        case CallOp::BUFFER_WRITE: {
            str << "_bfwrite"sv;
            auto elem = args[0]->type()->element();
            if (IsNumVec3(*elem)) {
                str << "Vec3("sv;
                PrintArgs();
                str << ',';
                GetTypeName(*elem->element(), str, Usage::NONE);
                str << ')';
                return;
            } else if (elem->is_matrix()) {
                str << "Mat";
            }
        } break;
        case CallOp::BUFFER_SIZE: {
            if (!shown_buffer_warning) {
                LUISA_WARNING_WITH_LOCATION("CallOp::BUFFER_SIZE is broken on dx!"sv);
                shown_buffer_warning = true;
            }
            str << "_bfsize"sv;
        } break;
        case CallOp::BYTE_BUFFER_READ: {
            str << "_bytebfread"sv;
            auto elem = expr->type();
            if (IsNumVec3(*elem)) {
                str << "Vec3("sv;
                args[0]->accept(vis);
                str << ',';
                GetTypeName(*elem->element(), str, Usage::NONE);
                str << ',';
                args[1]->accept(vis);
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
                str << ')';
            } else {
                str << '(';
                args[0]->accept(vis);
                str << ',';
                GetTypeName(*elem, str, Usage::NONE);
                str << ',';
                args[1]->accept(vis);
                str << ')';
            }
            return;
        }
        case CallOp::BYTE_BUFFER_WRITE: {
            str << "_bytebfwrite"sv;
            auto elem = args[2]->type();
            if (elem == Type::of<float3>()) {
                str << "Vec3("sv;
                args[0]->accept(vis);
                str << ',';
                GetTypeName(*elem->element(), str, Usage::NONE);
                str << ',';
                args[1]->accept(vis);
                str << ',';
                args[2]->accept(vis);
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
                args[2]->accept(vis);
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
        case CallOp::RAY_TRACING_QUERY_ALL:
            str << "_QueryAll"sv;
            break;
        case CallOp::RAY_TRACING_QUERY_ANY:
            str << "_QueryAny"sv;
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
            str << "_READ_BUFFER"sv;
            opt->useBufferBindless = true;
            str << '(';
            for (auto &&i : args) {
                i->accept(vis);
                str << ',';
            }
            vstd::to_string(expr->type()->size(), str);
            str << ',';
            GetTypeName(*expr->type(), str, Usage::READ, true);
            str << ",bdls)"sv;
            return;
        }
        case CallOp::BINDLESS_BYTE_BUFFER_READ: {
            str << "_READ_BUFFER_BYTES"sv;
            opt->useBufferBindless = true;
            str << '(';
            for (auto &&i : args) {
                i->accept(vis);
                str << ',';
            }
            GetTypeName(*expr->type(), str, Usage::READ, true);
            str << ",bdls)"sv;
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
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE:
            opt->useTex2DBindless = true;
            if (opt->isPixelShader) {
                str << "_SampleTex2DPixel"sv;
            } else {
                str << "_SampleTex2D"sv;
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
        case CallOp::SYNCHRONIZE_BLOCK:
            str << "GroupMemoryBarrierWithGroupSync()"sv;
            return;
        case CallOp::RASTER_DISCARD:
            LUISA_ASSERT(opt->funcType == CodegenStackData::FuncType::Pixel, "Raster-Discard can only be used in pixel shader");
            str << "discard";
            return;
        case CallOp::DDX: {
            if (opt->isRaster) {
                LUISA_ASSERT(opt->funcType == CodegenStackData::FuncType::Pixel, "ddx can only be used in pixel shader");
                str << "ddx"sv;
            } else {
                str << "_ddx"sv;
            }
        } break;
        case CallOp::DDY: {
            if (opt->isRaster) {
                LUISA_ASSERT(opt->funcType == CodegenStackData::FuncType::Pixel, "ddy can only be used in pixel shader");
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
        default:
            LUISA_ERROR("Bad op.");
            break;
    }
    str << '(';
    PrintArgs();
    str << ')';
}

template<typename T>
struct TypeNameStruct {
    void operator()(vstd::StringBuilder &str) {
        if constexpr (std::is_same_v<bool, T>) {
            str << "bool";
        } else if constexpr (std::is_same_v<int, T>) {
            str << "int";
        } else if constexpr (std::is_same_v<uint, T>) {
            str << "uint";
        } else if constexpr (std::is_same_v<float, T>) {
            str << "float";
        } else {
            static_assert(vstd::AlwaysFalse<T>, "illegal type");
        }
    }
};
template<typename T, size_t t>
struct TypeNameStruct<luisa::Vector<T, t>> {
    void operator()(vstd::StringBuilder &str) {
        TypeNameStruct<T>()(str);
        size_t n = (t == 3) ? 4 : t;
        str += ('0' + n);
    }
};
template<size_t t>
struct TypeNameStruct<luisa::Matrix<t>> {
    void operator()(vstd::StringBuilder &str) {
        TypeNameStruct<float>()(str);
        if constexpr (t == 2) {
            str << "2x2";
        } else if constexpr (t == 3) {
            str << "4x3";
        } else if constexpr (t == 4) {
            str << "4x4";
        } else {
            static_assert(vstd::AlwaysFalse<luisa::Matrix<t>>, "illegal type");
        }
    }
};

class CodegenConstantPrinter final : public ConstantDecoder {

private:
    vstd::StringBuilder &_str;

public:
    CodegenConstantPrinter(CodegenUtility &codegen,
                           vstd::StringBuilder &str) noexcept
        : _str{str} {}

protected:
    void _decode_bool(bool x) noexcept override {
        PrintValue<bool>{}(x, _str);
    }
    void _decode_char(char x) noexcept override {
        LUISA_NOT_IMPLEMENTED();
    }
    void _decode_uchar(uchar x) noexcept override {
        LUISA_NOT_IMPLEMENTED();
    }
    void _decode_short(short x) noexcept override {
        LUISA_NOT_IMPLEMENTED();
    }
    void _decode_ushort(ushort x) noexcept override {
        LUISA_NOT_IMPLEMENTED();
    }
    void _decode_int(int x) noexcept override {
        PrintValue<int>{}(x, _str);
    }
    void _decode_uint(uint x) noexcept override {
        PrintValue<uint>{}(x, _str);
    }
    void _decode_long(slong x) noexcept override {
        LUISA_NOT_IMPLEMENTED();
    }
    void _decode_ulong(ulong x) noexcept override {
        LUISA_NOT_IMPLEMENTED();
    }
    void _decode_half(half x) noexcept override {
        LUISA_NOT_IMPLEMENTED();
    }
    void _decode_float(float x) noexcept override {
        PrintValue<float>{}(x, _str);
    }
    void _decode_double(double x) noexcept override {
        LUISA_NOT_IMPLEMENTED();
    }
    void _vector_separator(const Type *type, uint index) noexcept override {
        LUISA_ERROR_WITH_LOCATION("Should not be called.");
    }
    void _matrix_separator(const Type *type, uint index) noexcept override {
        LUISA_ERROR_WITH_LOCATION("Should not be called.");
    }
    void _decode_vector(const Type *type, const std::byte *data) noexcept override {
#define LUISA_HLSL_DECODE_CONST_VEC(T, N)                   \
    do {                                                    \
        if (type == Type::of<T##N>()) {                     \
            auto x = *reinterpret_cast<const T##N *>(data); \
            if constexpr (N == 3) { _str << "{"sv; }        \
            PrintValue<T##N>{}(x, _str);                    \
            if constexpr (N == 3) { _str << ",0}"sv; }      \
            return;                                         \
        }                                                   \
    } while (false)
#define LUISA_HLSL_DECODE_CONST(T)     \
    LUISA_HLSL_DECODE_CONST_VEC(T, 2); \
    LUISA_HLSL_DECODE_CONST_VEC(T, 3); \
    LUISA_HLSL_DECODE_CONST_VEC(T, 4)
        LUISA_HLSL_DECODE_CONST(bool);
        LUISA_HLSL_DECODE_CONST(int);
        LUISA_HLSL_DECODE_CONST(uint);
        LUISA_HLSL_DECODE_CONST(float);
        LUISA_ERROR_WITH_LOCATION(
            "Constant type '{}' is not supported yet.",
            type->description());
#undef LUISA_HLSL_DECODE_CONST_VEC
#undef LUISA_HLSL_DECODE_CONST
    }
    void _decode_matrix(const Type *type, const std::byte *data) noexcept override {
#define LUISA_HLSL_DECODE_CONST_MAT(N)                               \
    do {                                                             \
        using M = float##N##x##N;                                    \
        if (type == Type::of<M>()) {                                 \
            auto x = *reinterpret_cast<const M *>(data);             \
            _str << "float" << #N "x" << (N == 3 ? "4" : #N) << "("; \
            for (auto i = 0; i < N; i++) {                           \
                _str << "float" << (N == 3 ? "4" : #N) << "(";       \
                for (auto j = 0; j < 3; j++) {                       \
                    PrintValue<float>{}(x[i][j], _str);              \
                    if (j != N - 1) { _str << ","; }                 \
                }                                                    \
                if (N == 3) { _str << ",0"; }                        \
                _str << ")";                                         \
                if (i != N - 1) { _str << ","; }                     \
            }                                                        \
            _str << ")";                                             \
            return;                                                  \
        }                                                            \
    } while (false)
        LUISA_HLSL_DECODE_CONST_MAT(2);
        LUISA_HLSL_DECODE_CONST_MAT(3);
        LUISA_HLSL_DECODE_CONST_MAT(4);
        LUISA_ERROR_WITH_LOCATION(
            "Constant type '{}' is not supported yet.",
            type->description());
#undef LUISA_HLSL_DECODE_CONST_MAT
    }
    void _struct_separator(const Type *type, uint index) noexcept override {
        auto n = type->members().size();
        if (index == 0u) {
            _str << "{"sv;
        } else if (index == n) {
            _str << "}"sv;
        } else {
            _str << ',';
        }
    }
    void _array_separator(const Type *type, uint index) noexcept override {
        auto n = type->dimension();
        if (index == 0u) {
            _str << "{{"sv;
        } else if (index == n) {
            _str << "}}"sv;
        } else {
            _str << ',';
        }
    }
};

void CodegenUtility::CodegenFunction(Function func, vstd::StringBuilder &result, bool cbufferNonEmpty) {
    auto codegenOneFunc = [&](Function func) {
        auto constants = func.constants();
        for (auto &&i : constants) {
            vstd::StringBuilder constValueName;
            if (!GetConstName(i.hash(), i, constValueName)) continue;
            result << "static const "sv;
            GetTypeName(*i.type(), result, Usage::READ);
            result << ' ' << constValueName << " = "sv;
            CodegenConstantPrinter printer{*this, result};
            i.decode(printer);
            result << ";\n"sv;
        }
#ifdef LUISA_ENABLE_IR
        vstd::unordered_set<Variable> grad_vars;
        glob_variables_with_grad(func, grad_vars);
#endif
        if (func.tag() == Function::Tag::KERNEL) {
            opt->funcType = CodegenStackData::FuncType::Kernel;
            result << "[numthreads("
                   << vstd::to_string(func.block_size().x)
                   << ','
                   << vstd::to_string(func.block_size().y)
                   << ','
                   << vstd::to_string(func.block_size().z)
                   << R"()]
void main(uint3 thdId:SV_GroupThreadId,uint3 dspId:SV_DispatchThreadID,uint3 grpId:SV_GroupId){
)"sv;
            auto blockSize = func.block_size();
            vstd::fixed_vector<char, 3> swizzle;
            //  result << "if(any(dspId >= dsp_c.xyz)) return;\n"sv;
            if (blockSize.x > 1) {
                swizzle.emplace_back('x');
            }
            if (blockSize.y > 1) {
                swizzle.emplace_back('y');
            }
            if (blockSize.z > 1) {
                swizzle.emplace_back('z');
            }
            if (!swizzle.empty()) {
                if (swizzle.size() == 1) {
                    result << "if(dspId."sv << swizzle[0] << ">=dsp_c."sv << swizzle[0] << ") return;\n"sv;
                } else {
                    vstd::string_view strv(swizzle.data(), swizzle.size());
                    result << "if(any(dspId."sv << strv << ">=dsp_c."sv << strv << ")) return;\n"sv;
                }
            }
            if (cbufferNonEmpty) {
                result << "_Args a = _Global[0];\n"sv;
            }
            opt->arguments.clear();
            opt->arguments.reserve(func.arguments().size());
            size_t idx = 0;
            for (auto &&i : func.arguments()) {
                opt->arguments.try_emplace(i.uid(), idx);
                ++idx;
            }
        } else {
            opt->funcType = CodegenStackData::FuncType::Callable;
            GetFunctionDecl(func, result);
            result << "{\n"sv;
        }
        {

            StringStateVisitor vis(func, result, this);
            vis.sharedVariables = &opt->sharedVariable;
            vis.VisitFunction(
#ifdef LUISA_ENABLE_IR
                grad_vars,
#endif
                func);
        }
        result << "}\n"sv;
    };
    vstd::unordered_set<uint64_t> callableMap;
    auto callable = [&](auto &&callable, Function func) -> void {
        for (auto &&i : func.custom_callables()) {
            if (callableMap.emplace(i->hash()).second) {
                callable(callable, i->function());
            }
        }
        codegenOneFunc(func);
    };
    callable(callable, func);
}
void CodegenUtility::CodegenVertex(Function vert, vstd::StringBuilder &result, bool cBufferNonEmpty) {
    vstd::unordered_set<void const *> callableMap;
    auto gen = [&](auto &callable, Function func) -> void {
        for (auto &&i : func.custom_callables()) {
            if (callableMap.emplace(i.get()).second) {
                Function f(i.get());
                callable(callable, f);
            }
        }
    };
    auto callable = [&](auto &callable, Function func) -> void {
        gen(callable, func);
        CodegenFunction(func, result, cBufferNonEmpty);
    };
    gen(callable, vert);
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
    auto resetPixelShaderKey = vstd::scope_exit([&] { opt->isPixelShader = false; });
    vstd::unordered_set<void const *> callableMap;
    auto gen = [&](auto &callable, Function func) -> void {
        for (auto &&i : func.custom_callables()) {
            if (callableMap.emplace(i.get()).second) {
                Function f(i.get());
                callable(callable, f);
            }
        }
    };
    auto callable = [&](auto &callable, Function func) -> void {
        gen(callable, func);
        CodegenFunction(func, result, cBufferNonEmpty);
    };
    gen(callable, pixel);
    vstd::StringBuilder retName;
    auto retType = pixel.return_type();
    GetTypeName(*retType, retName, Usage::READ);
    result << retName << " pixel(v2p p){\n"sv;
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
    result << R"(
}
void main(v2p p)"sv;
    if (retType->is_scalar() || retType->is_vector()) {
        result << ",out "sv;
        GetTypeName(*retType, result, Usage::READ);
        result << R"( o0:SV_TARGET0){
o0=pixel(p);
}
)"sv;
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
        result << " o=pixel(p);\n"sv;
        for (auto i : vstd::range(retType->members().size())) {
            auto num = vstd::to_string(i);
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
namespace detail {
static bool IsCBuffer(Variable::Tag t) {
    switch (t) {
        case Variable::Tag::BUFFER:
        case Variable::Tag::TEXTURE:
        case Variable::Tag::BINDLESS_ARRAY:
        case Variable::Tag::ACCEL:
        case Variable::Tag::THREAD_ID:
        case Variable::Tag::BLOCK_ID:
        case Variable::Tag::DISPATCH_ID:
        case Variable::Tag::DISPATCH_SIZE:
        case Variable::Tag::KERNEL_ID:
        case Variable::Tag::OBJECT_ID:
            return false;
        default:
            return true;
    }
}
}// namespace detail
bool CodegenUtility::IsCBufferNonEmpty(std::initializer_list<vstd::IRange<Variable> *> fs) {
    for (auto &&f : fs) {
        for (auto &&i : *f) {
            if (detail::IsCBuffer(i.tag())) {
                return true;
            }
        }
    }
    return false;
}
bool CodegenUtility::IsCBufferNonEmpty(Function f) {
    for (auto &&i : f.arguments()) {
        if (detail::IsCBuffer(i.tag())) {
            return true;
        }
    }
    return false;
}
void CodegenUtility::GenerateCBuffer(
    std::initializer_list<vstd::IRange<Variable> *> fs,
    vstd::StringBuilder &result,
    uint &bind_count) {
    result << "struct _Args{\n"sv;
    size_t align = 0;
    size_t size = 0;
    for (auto &&f : fs) {
        size_t size_cache = 0;
        for (auto &&i : *f) {
            if (!detail::IsCBuffer(i.tag())) continue;
            size_cache++;
            GetTypeName(*i.type(), result, Usage::READ, true);
            // vec3 need extra alignment
            result << " l" << vstd::to_string(i.uid() + size) << ";\n"sv;
            if (i.type()->is_vector() && i.type()->dimension() == 3) {
                GetTypeName(*i.type()->element(), result, Usage::READ, true);
                result << " _a"sv;
                vstd::to_string(align, result);
                result << ";\n"sv;
                ++align;
            }
        }
        size += size_cache;
    }
    result << R"(};
StructuredBuffer<_Args> _Global:register(t0);
)"sv;
    bind_count += 2;
}
void CodegenUtility::GenerateBindless(
    CodegenResult::Properties &properties,
    vstd::StringBuilder &str,
    bool isSpirV,
    uint &bind_count) {
    uint table_idx = isSpirV ? 2 : 1;
    auto add_prop = [&](ShaderVariableType svt) {
        properties.emplace_back(
            Property{
                svt,
                table_idx,
                0u, std::numeric_limits<uint>::max()});
    };

    if (opt->useBufferBindless) {
        str << "ByteAddressBuffer bdls[]:register(t0,space"sv << vstd::to_string(table_idx) << ");\n"sv;
        add_prop(ShaderVariableType::SRVBufferHeap);
        table_idx++;
        bind_count += 1;
    }
    if (opt->useTex2DBindless) {
        str << "Texture2D<float4> _BindlessTex[]:register(t0,space"sv << vstd::to_string(table_idx) << ");"sv;
        add_prop(ShaderVariableType::SRVTextureHeap);
        table_idx++;
        str << CodegenUtility::ReadInternalHLSLFile("tex2d_bindless");
        bind_count += 1;
    }
    if (opt->useTex3DBindless) {
        str << "Texture3D<float4> _BindlessTex3D[]:register(t0,space"sv << vstd::to_string(table_idx) << ");"sv;
        add_prop(ShaderVariableType::SRVTextureHeap);
        table_idx++;
        str << CodegenUtility::ReadInternalHLSLFile("tex3d_bindless");
        bind_count += 1;
    }
}

void CodegenUtility::PreprocessCodegenProperties(
    CodegenResult::Properties &properties,
    vstd::StringBuilder &varData,
    RegisterIndexer &registerCount,
    bool cbufferNonEmpty,
    bool isRaster, bool isSpirv, uint &bind_count) {
    // 1,0,0
    registerCount.init();
    if (isSpirv) {
        properties.emplace_back(
            Property{
                ShaderVariableType::ConstantValue,
                0,
                1,
                1});
    } else {
        if (!isRaster) {
            properties.emplace_back(
                Property{
                    ShaderVariableType::ConstantValue,
                    4,
                    0,
                    1});
        }
    }
    properties.emplace_back(
        Property{
            ShaderVariableType::SamplerHeap,
            1u,
            0u,
            16u});
    if (cbufferNonEmpty) {
        registerCount.get(2)++;
        properties.emplace_back(
            Property{
                ShaderVariableType::StructuredBuffer,
                0,
                0,
                1});
    }
    GenerateBindless(properties, varData, isSpirv, bind_count);
}

namespace detail {
[[nodiscard]] static auto can_accum_grad(const Type *t) noexcept {
    auto tt = t->tag();
    if (tt == Type::Tag::FLOAT16 ||
        tt == Type::Tag::FLOAT32 ||
        tt == Type::Tag::FLOAT64 ||
        tt == Type::Tag::STRUCTURE) {
        return true;
    }
    if (tt == Type::Tag::ARRAY ||
        tt == Type::Tag::VECTOR ||
        tt == Type::Tag::MATRIX) {
        return can_accum_grad(t->element());
    }
    return false;
}
}// namespace detail

void CodegenUtility::PostprocessCodegenProperties(vstd::StringBuilder &finalResult, bool use_autodiff) {
    if (!opt->customStruct.empty()) {
        for (auto v : opt->customStructVector) {
            finalResult << "struct " << v->GetStructName() << "{\n"
                        << v->GetStructDesc() << "};\n";
            // accum grad while using autodiff
            if (use_autodiff) {
                auto accum_grad = [&s = finalResult](luisa::string_view access, const Type *t) noexcept {
                    if (t->is_structure() || t->is_array()) {
                        s << luisa::format("_accum_grad_{:016X}(x_grad{}, dx{});\n", t->hash(), access, access);
                    } else {
                        s << luisa::format("_accum_grad(x_grad{}, dx{});\n", access, access);
                    }
                };
                if (auto t = v->GetType(); t->is_structure() || t->is_array()) {
                    finalResult << luisa::format("void _accum_grad_{:016X}(inout {} x_grad, {} dx){{\n",
                                                 t->hash(), v->GetStructName(), v->GetStructName());
                    if (t->is_structure()) {
                        for (auto i = 0u; i < t->members().size(); i++) {
                            if (auto m = t->members()[i]; detail::can_accum_grad(m)) {
                                accum_grad(luisa::format(".v{}", i), m);
                            }
                        }
                    } else if (detail::can_accum_grad(t->element())) {
                        finalResult << luisa::format("for(uint i=0;i<{};++i){{", t->dimension());
                        accum_grad(luisa::format(".v[i]"), t->element());
                        finalResult << "}\n";
                    }
                    finalResult << "}\n";
                }
            }
        }
    }
    for (auto &&kv : opt->sharedVariable) {
        auto &&i = kv.second;
        finalResult << "groupshared "sv;
        GetTypeName(*i.type()->element(), finalResult, Usage::READ, false);
        finalResult << ' ';
        GetVariableName(i, finalResult);
        finalResult << '[';
        vstd::to_string(i.type()->dimension(), finalResult);
        finalResult << "];\n"sv;
    }
}
uint CodegenUtility::AddPrinter(vstd::string_view name, Type const *structType) {
    auto z = opt->printer.size();
    opt->printer.emplace_back(name, structType);
    return z;
}
void CodegenUtility::CodegenProperties(
    CodegenResult::Properties &properties,
    vstd::StringBuilder &varData,
    Function kernel,
    uint offset,
    RegisterIndexer &registerCount,
    uint &bind_count) {
    enum class RegisterType : uint8_t {
        CBV,
        UAV,
        SRV
    };
    auto Writable = [&](Variable const &v) {
        return (static_cast<uint>(kernel.variable_usage(v.uid())) & static_cast<uint>(Usage::WRITE)) != 0;
    };
    auto args = kernel.arguments();
    for (auto &&i : vstd::ptr_range(args.data() + offset, args.size() - offset)) {
        auto print = [&] {
            auto usage = kernel.variable_usage(i.uid());
            if (i.type()->is_buffer() || i.type()->is_texture()) {
                auto attris = i.type()->member_attributes();
                if (!attris.empty()) {
                    for (auto &a : attris) {
                        if ((to_underlying(usage) & to_underlying(Usage::WRITE)) != 0) {
                            if (a.key == "cache"sv) {
                                if (a.value == "coherent"sv) {
                                    varData << "globallycoherent "sv;
                                }
                            }
                        }
                    }
                }
            }
            GetTypeName(*i.type(), varData, usage);
            varData << ' ';
            GetVariableName(i, varData);
        };
        auto printInstBuffer = [&]<bool writable>() {
            if constexpr (writable)
                varData << "RWStructuredBuffer<_MeshInst> "sv;
            else
                varData << "StructuredBuffer<_MeshInst> "sv;
            GetVariableName(i, varData);
            varData << "Inst"sv;
        };
        auto genArg = [&]<RegisterType regisT, bool rtBuffer = false, bool writable = false>(ShaderVariableType sT, char v) {
            auto &&r = registerCount.get((uint8_t)regisT);
            Property prop = {
                .type = sT,
                .space_index = 0,
                .register_index = r,
                .array_size = 1};
            if constexpr (rtBuffer) {
                printInstBuffer.operator()<writable>();
                properties.emplace_back(prop);
            } else {
                print();
                properties.emplace_back(prop);
            }
            varData << ":register("sv << v;
            vstd::to_string(r, varData);
            varData << ");\n"sv;
            r++;
            switch (sT) {
                case ShaderVariableType::ConstantBuffer:
                case ShaderVariableType::StructuredBuffer:
                case ShaderVariableType::RWStructuredBuffer:
                case ShaderVariableType::ConstantValue:
                case ShaderVariableType::SamplerHeap:
                    bind_count += 2;
                    break;
                case ShaderVariableType::SRVTextureHeap:
                case ShaderVariableType::UAVTextureHeap:
                case ShaderVariableType::SRVBufferHeap:
                case ShaderVariableType::UAVBufferHeap:
                case ShaderVariableType::CBVBufferHeap:
                    bind_count += 1;
                    break;
            }
        };
        switch (i.type()->tag()) {
            case Type::Tag::TEXTURE:
                if (Writable(i)) {
                    genArg.operator()<RegisterType::UAV>(ShaderVariableType::UAVTextureHeap, 'u');
                } else {
                    genArg.operator()<RegisterType::SRV>(ShaderVariableType::SRVTextureHeap, 't');
                }
                break;
            case Type::Tag::BUFFER: {
                if (Writable(i)) {
                    genArg.operator()<RegisterType::UAV>(ShaderVariableType::RWStructuredBuffer, 'u');
                } else {
                    genArg.operator()<RegisterType::SRV>(ShaderVariableType::StructuredBuffer, 't');
                }
            } break;
            case Type::Tag::BINDLESS_ARRAY:
                genArg.operator()<RegisterType::SRV>(ShaderVariableType::StructuredBuffer, 't');
                break;
            case Type::Tag::ACCEL:
                if (Writable(i)) {
                    genArg.operator()<RegisterType::UAV, true, true>(ShaderVariableType::RWStructuredBuffer, 'u');
                } else {
                    genArg.operator()<RegisterType::SRV>(ShaderVariableType::StructuredBuffer, 't');
                    genArg.operator()<RegisterType::SRV, true>(ShaderVariableType::StructuredBuffer, 't');
                }
                break;
            case Type::Tag::CUSTOM: {
                if (i.type()->description() == "LC_IndirectDispatchBuffer"sv) {
                    genArg.operator()<RegisterType::UAV>(ShaderVariableType::RWStructuredBuffer, 'u');
                }
            } break;
            default: break;
        }
    }
    if (kernel.requires_printing()) {
        auto &&r = registerCount.get((uint8_t)RegisterType::UAV);
        {
            Property prop = {
                .type = ShaderVariableType::RWStructuredBuffer,
                .space_index = 0,
                .register_index = r,
                .array_size = 1};
            properties.emplace_back(prop);
            varData << "RWStructuredBuffer<uint> _printCounter:register(u"sv;
            vstd::to_string(r, varData);
            varData << ");\n"sv;
            r += 1;
        }
        {
            Property prop = {
                .type = ShaderVariableType::RWStructuredBuffer,
                .space_index = 0,
                .register_index = r,
                .array_size = 1};
            properties.emplace_back(prop);
            varData << "RWByteAddressBuffer _printBuffer:register(u"sv;
            vstd::to_string(r, varData);
            varData << ");\n"sv;
            r += 1;
        }
        bind_count += 4;
    }
}
vstd::MD5 CodegenUtility::GetTypeMD5(vstd::span<Type const *const> types) {
    vstd::vector<uint64_t> typeDescs;
    typeDescs.reserve(types.size());
    for (auto &&i : types) {
        if ((i->is_buffer() || i->is_texture()) && !i->member_attributes().empty())
            if (i->is_buffer())
                typeDescs.emplace_back(Type::buffer(i->element())->hash());
            else
                typeDescs.emplace_back(Type::texture(i->element(), i->dimension())->hash());
        else
            typeDescs.emplace_back(i->hash());
    }
    return {vstd::span<uint8_t const>(reinterpret_cast<uint8_t const *>(typeDescs.data()), typeDescs.size_bytes())};
}
vstd::MD5 CodegenUtility::GetTypeMD5(std::initializer_list<vstd::IRange<Variable> *> f) {
    vstd::vector<uint64_t> typeDescs;
    for (auto &&rg : f) {
        for (auto &&i : *rg) {
            auto type = i.type();
            if ((type->is_buffer() || type->is_texture()) && !type->member_attributes().empty())
                if (type->is_buffer())
                    typeDescs.emplace_back(Type::buffer(type->element())->hash());
                else
                    typeDescs.emplace_back(Type::texture(type->element(), type->dimension())->hash());
            else
                typeDescs.emplace_back(type->hash());
        }
    }
    return {vstd::span<uint8_t const>(reinterpret_cast<uint8_t const *>(typeDescs.data()), typeDescs.size_bytes())};
}
vstd::MD5 CodegenUtility::GetTypeMD5(Function func) {
    vstd::vector<uint64_t> typeDescs;
    auto args = func.arguments();
    typeDescs.reserve(args.size());
    for (auto &&i : args) {
        auto type = i.type();
        if ((type->is_buffer() || type->is_texture()) && !type->member_attributes().empty())
            if (type->is_buffer())
                typeDescs.emplace_back(Type::buffer(type->element())->hash());
            else
                typeDescs.emplace_back(Type::texture(type->element(), type->dimension())->hash());
        else
            typeDescs.emplace_back(type->hash());
    }
    return {vstd::span<uint8_t const>(reinterpret_cast<uint8_t const *>(typeDescs.data()), typeDescs.size_bytes())};
}
CodegenUtility::CodegenUtility() {
    attributes.try_emplace("position", "POSITION", nullptr);
    attributes.try_emplace("normal", "NORMAL", nullptr);
    attributes.try_emplace("tangent", "TANGENT", nullptr);
    attributes.try_emplace("color", "COLOR", nullptr);
    attributes.try_emplace("uv0", "TEXCOORD0", nullptr);
    attributes.try_emplace("uv1", "TEXCOORD1", nullptr);
    attributes.try_emplace("uv2", "TEXCOORD2", nullptr);
    attributes.try_emplace("uv3", "TEXCOORD3", nullptr);
    attributes.try_emplace("vertex_id", "SV_VertexID", Type::of<uint>());
    attributes.try_emplace("instance_id", "SV_InstanceID", Type::of<uint>());
    attributes.try_emplace("is_front_face", "SV_IsFrontFace", Type::of<bool>());
}
CodegenUtility::~CodegenUtility() {}

CodegenResult CodegenUtility::Codegen(
    Function kernel, luisa::string_view native_code, uint custom_mask, bool isSpirV) {
    opt = CodegenStackData::Allocate(this);
    auto disposeOpt = vstd::scope_exit([&] {
        CodegenStackData::DeAllocate(std::move(opt));
    });
    // CodegenStackData::ThreadLocalSpirv() = false;
    opt->kernel = kernel;
    bool nonEmptyCbuffer = IsCBufferNonEmpty(kernel);

    vstd::StringBuilder codegenData;
    vstd::StringBuilder varData;
    vstd::StringBuilder incrementalFunc;
    vstd::StringBuilder finalResult;
    opt->incrementalFunc = &incrementalFunc;
    finalResult.reserve(65500);
    uint64 immutableHeaderSize = detail::AddHeader(kernel.propagated_builtin_callables(), finalResult, false);
    finalResult << native_code << "\n//"sv;
    static_cast<void>(vstd::to_string(custom_mask));
    finalResult << '\n';
    CodegenFunction(kernel, codegenData, nonEmptyCbuffer);

    opt->funcType = CodegenStackData::FuncType::Callable;
    auto argRange = vstd::make_ite_range(kernel.arguments()).i_range();
    uint bind_count = 2;
    if (nonEmptyCbuffer) {
        GenerateCBuffer({&argRange}, varData, bind_count);
    }
    if (isSpirV) {
        varData << R"(cbuffer CB:register(b1){
uint4 dsp_c;
}
)"sv;
        bind_count += 2;
    } else {
        varData << "uint4 dsp_c:register(b0);\n"sv;
        bind_count += 2;
    }
    CodegenResult::Properties properties;
    DXILRegisterIndexer dxilRegisters;
    SpirVRegisterIndexer spvRegisters;
    RegisterIndexer &indexer = isSpirV ? static_cast<RegisterIndexer &>(spvRegisters) : static_cast<RegisterIndexer &>(dxilRegisters);
    PreprocessCodegenProperties(properties, varData, indexer, nonEmptyCbuffer, false, isSpirV, bind_count);
    CodegenProperties(properties, varData, kernel, 0, indexer, bind_count);
    PostprocessCodegenProperties(finalResult, kernel.requires_autodiff());
    finalResult << varData << incrementalFunc << codegenData;
    if (bind_count >= 64) [[unlikely]] {
        LUISA_ERROR("Arguments binding size: {} exceeds 64 32-bit units not supported by hardware device. Try to use bindless instead.", bind_count);
    } else if (bind_count > 16) [[unlikely]] {
        if (!rootsig_exceed_warned.exchange(true)) {
            LUISA_WARNING("Arguments binding size exceeds 16 32-bit unit (max 64 allowed). This may cause extra performance cost, try to use bindless instead.");
        }
    }
    return {
        std::move(finalResult),
        std::move(opt->printer),
        std::move(properties),
        opt->useTex2DBindless,
        opt->useTex3DBindless,
        opt->useBufferBindless,
        immutableHeaderSize,
        GetTypeMD5(kernel)};
}
CodegenResult CodegenUtility::RasterCodegen(
    Function vertFunc,
    Function pixelFunc,

    luisa::string_view native_code,
    uint custom_mask,
    bool isSpirV) {
    opt = CodegenStackData::Allocate(this);
    // CodegenStackData::ThreadLocalSpirv() = false;
    opt->kernel = vertFunc;
    opt->isRaster = true;
    auto disposeOpt = vstd::scope_exit([&] {
        opt->isRaster = false;
        CodegenStackData::DeAllocate(std::move(opt));
    });
    vstd::StringBuilder codegenData;
    vstd::StringBuilder varData;
    vstd::StringBuilder finalResult;
    vstd::StringBuilder incrementalFunc;
    opt->incrementalFunc = &incrementalFunc;
    finalResult.reserve(65500);
    auto opSet = vertFunc.propagated_builtin_callables();
    opSet.propagate(pixelFunc.propagated_builtin_callables());
    uint64 immutableHeaderSize = detail::AddHeader(opSet, finalResult, true);
    finalResult << native_code << "\n//"sv;
    static_cast<void>(vstd::to_string(custom_mask));
    finalResult << '\n';
    // Vertex
    codegenData << "struct v2p{\n"sv;
    auto v2pType = vertFunc.return_type();
    if (v2pType->is_structure()) {
        opt->internalStruct.emplace(v2pType, "v2p");
        if (v2pType->members().size() != v2pType->member_attributes().size()) [[unlikely]] {
            LUISA_ERROR("Vertex-to-pixel structure's attribute size is illegal.");
        }
        size_t memberIdx = 0;
        bool pos = false;
        for (auto &&i : v2pType->members()) {
            GetTypeName(*i, codegenData, Usage::READ);
            codegenData << " v"sv << vstd::to_string(memberIdx);
            if (v2pType->member_attributes()[memberIdx].key == "position"sv) {
                if (pos) [[unlikely]] {
                    LUISA_ERROR("Vertex-to-pixel structure can only have one position.");
                }
                codegenData << ":SV_POSITION;\n"sv;
                pos = true;
                if (!i->is_vector() || i->dimension() != 4) [[unlikely]] {
                    LUISA_ERROR("Position must be float4.");
                }
            } else {
                codegenData << ":TEXCOORD"sv << vstd::to_string(memberIdx - 1) << ";\n"sv;
            }
            ++memberIdx;
        }
        if (!pos) [[unlikely]] {
            LUISA_ERROR("Vertex-to-pixel structure should contained position.");
        }
    } else {
        LUISA_ERROR("Illegal vertex return type!");
    }
    uint bind_count = 2;
    if (isSpirV) {
        codegenData << R"(};
cbuffer CB:register(b1){
uint obj_id;
)"sv;
        bind_count += 2;
    } else {
        codegenData << R"(};
uint obj_id:register(b0);
)"sv;
        bind_count += 2;
    }
    codegenData << "#ifdef VS\n";
    auto vert_args = vertFunc.arguments();
    if (vert_args.empty()) [[unlikely]] {
        LUISA_ERROR("Vertex arguments illegal.");
    }
    auto appdataType = vert_args[0].type();
    if (appdataType->is_structure()) {
        auto appdataAttris = appdataType->member_attributes();
        auto appdataMems = appdataType->members();
        if (appdataAttris.size() != appdataMems.size()) [[unlikely]] {
            LUISA_ERROR("Mesh-to-vertex structure must have attributes.");
        }
        opt->internalStruct.try_emplace(appdataType, "_mesh");
        codegenData << "struct _mesh{\n"sv;
        for (auto i : vstd::range(appdataAttris.size())) {
            auto member = appdataMems[i];
            auto &attr = appdataAttris[i];
            if (attr.key.empty()) [[unlikely]] {
                LUISA_ERROR("Mesh-to-vertex structure member {} miss attributes.", i);
            }
            if (!(member->is_scalar() || member->is_vector())) [[unlikely]] {
                LUISA_ERROR("Mesh-to-vertex structure do not support type {}", member->description());
            }

            auto iter = attributes.find(attr.key);
            if (iter == attributes.end()) [[unlikely]] {
                LUISA_ERROR("Invalid attribute: {}", attr.key);
            }

            if (iter->second.second && iter->second.second != member) [[unlikely]] {
                LUISA_ERROR("Attribute {} type {} mismatch with {}", attr.key, iter->second.second->description(), member->description());
            }
            GetTypeName(*member, codegenData, Usage::READ);
            codegenData
                << " v"sv << vstd::to_string(i) << ':'
                << iter->second.first
                << ";\n"sv;
        }
        codegenData << "};\n";
    } else {
        LUISA_ERROR("Mesh-to-vertex must be a structure");
    }

    auto vertRange = vstd::make_ite_range(vert_args.subspan(1)).i_range();
    auto pixelRange = vstd::make_ite_range(pixelFunc.arguments().subspan(1)).i_range();
    std::initializer_list<vstd::IRange<Variable> *> funcs = {&vertRange, &pixelRange};

    bool nonEmptyCbuffer = IsCBufferNonEmpty(funcs);
    opt->appdataId = vert_args[0].uid();
    CodegenVertex(vertFunc, codegenData, nonEmptyCbuffer);
    opt->appdataId = -1;
    // TODO: gen vertex data
    codegenData << "#elif defined(PS)\n"sv;
    opt->argOffset = vert_args.size() - 1;
    // TODO: gen pixel data
    CodegenPixel(pixelFunc, codegenData, nonEmptyCbuffer);
    codegenData << "#endif\n"sv;

    opt->funcType = CodegenStackData::FuncType::Callable;
    if (nonEmptyCbuffer) {
        GenerateCBuffer(funcs, varData, bind_count);
    }
    CodegenResult::Properties properties;
    DXILRegisterIndexer dxilRegisters;
    SpirVRegisterIndexer spvRegisters;
    RegisterIndexer &indexer = isSpirV ? static_cast<RegisterIndexer &>(spvRegisters) : static_cast<RegisterIndexer &>(dxilRegisters);
    PreprocessCodegenProperties(properties, varData, indexer, nonEmptyCbuffer, true, isSpirV, bind_count);
    CodegenProperties(properties, varData, vertFunc, 1, indexer, bind_count);
    CodegenProperties(properties, varData, pixelFunc, 1, indexer, bind_count);
    PostprocessCodegenProperties(finalResult, false);
    finalResult << varData << incrementalFunc << codegenData;
    if (bind_count >= 64) [[unlikely]] {
        LUISA_ERROR("Arguments binding size: {} exceeds 64 32-bit units not supported by hardware device. Try to use bindless instead.", bind_count);
    } else if (bind_count > 16) [[unlikely]] {
        if (!rootsig_exceed_warned.exchange(true)) {
            LUISA_WARNING("Arguments binding size exceeds 16 32-bit unit (max 64 allowed). This may cause extra performance cost, try to use bindless instead.");
        }
    }
    return {
        std::move(finalResult),
        std::move(opt->printer),
        std::move(properties),
        opt->useTex2DBindless,
        opt->useTex3DBindless,
        opt->useBufferBindless,
        immutableHeaderSize,
        GetTypeMD5(funcs)};
}
}// namespace lc::hlsl
