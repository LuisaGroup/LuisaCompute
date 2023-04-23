#include "hlsl_codegen.h"
#include <vstl/string_utility.h>
#include "variant_util.h"
#include <ast/constant_data.h>
#include "struct_generator.h"
#include "codegen_stack_data.h"
#include <vstl/pdqsort.h>
#include <core/dynamic_module.h>
namespace lc::hlsl {
struct RegisterIndexer {
    virtual void init() = 0;
    virtual uint &get(uint idx) = 0;
};
struct DXILRegisterIndexer : public RegisterIndexer {
    vstd::array<uint, 3> values;
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
vstd::string_view CodegenUtility::ReadInternalHLSLFile(vstd::string_view name, luisa::BinaryIO const *ctx) {
    static DynamicModule dyna_module = DynamicModule::load("lc-hlsl-builtin");
    auto get_name = vstd::string{"get_"}.append(name);
    auto get_func = dyna_module.function<char *()>(get_name);
    get_name += "_size"sv;
    auto get_size_func = dyna_module.function<int()>(get_name);
    return {get_func(), static_cast<size_t>(get_size_func())};
}
namespace detail {
static inline uint64 CalcAlign(uint64 value, uint64 align) {
    return (value + (align - 1)) & ~(align - 1);
}
static vstd::string_view HLSLHeader(CodegenUtility *util, luisa::BinaryIO const *internalDataPath) {
    return CodegenUtility::ReadInternalHLSLFile("hlsl_header", internalDataPath);
}
static vstd::string_view RayTracingHeader(CodegenUtility *util, luisa::BinaryIO const *internalDataPath) {
    return CodegenUtility::ReadInternalHLSLFile("raytracing_header", internalDataPath);
}
}// namespace detail
// static thread_local vstd::unique_ptr<CodegenStackData> opt;
#ifdef USE_SPIRV
CodegenStackData *CodegenUtility::StackData() { return opt.get(); }
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
            str << "dsp_c.xyz"sv;
            break;
        case Variable::Tag::KERNEL_ID:
            str << "dsp_c.w"sv;
            break;
        case Variable::Tag::OBJECT_ID:
            // assert(opt->funcType == CodegenStackData::FuncType::Vert);
            str << "obj_id"sv;
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
            str << 's';
            vstd::to_string(id, str);
            break;
        case Variable::Tag::REFERENCE:
            str << 'r';
            vstd::to_string(id, str);
            break;
        case Variable::Tag::BUFFER:
            str << 'b';
            vstd::to_string(id, str);
            break;
        case Variable::Tag::TEXTURE:
            str << 't';
            vstd::to_string(id, str);
            break;
        case Variable::Tag::BINDLESS_ARRAY:
            str << "ba"sv;
            vstd::to_string(id, str);
            break;
        case Variable::Tag::ACCEL:
            str << "ac"sv;
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
void CodegenUtility::GetConstantStruct(ConstantData const &data, vstd::StringBuilder &str) {
    uint64 constCount = opt->GetConstCount(data.hash()).first;
    //auto typeName = CodegenUtility::GetBasicTypeName(view.index());
    str << "struct tc";
    vstd::to_string((constCount), str);
    uint64 varCount = 1;
    luisa::visit(
        [&](auto &&arr) {
            varCount = arr.size();
        },
        data.view());
    str << "{\n";
    str << CodegenUtility::GetBasicTypeName(data.view().index()) << " v[";
    vstd::to_string((varCount), str);
    str << "];\n";
    str << "};\n";
}
void CodegenUtility::GetConstantData(ConstantData const &data, vstd::StringBuilder &str) {
    auto &&view = data.view();
    uint64 constCount = opt->GetConstCount(data.hash()).first;

    vstd::string name = vstd::to_string((constCount));
    str << "uniform const tc" << name << " c" << name;
    str << "={{";
    luisa::visit(
        [&](auto &&arr) {
            for (auto const &ele : arr) {
                PrintValue<std::remove_cvref_t<typename std::remove_cvref_t<decltype(arr)>::element_type>> prt;
                prt(ele, str);
                str << ',';
            }
        },
        view);
    auto last = str.end() - 1;
    if (*last == ',')
        *last = '}';
    else
        str << '}';
    str << "};\n";
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
        case Type::Tag::INT16:
            str << "int16_t"sv;
            return;
        case Type::Tag::UINT16:
            str << "uint16_t"sv;
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
                str << 'w';
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
            str << "StructuredBuffer<"sv;
            auto ele = type.element();
            if (ele->is_matrix()) {
                auto n = vstd::to_string(ele->dimension());
                str << "WrappedFloat"sv << n << 'x' << n;
            } else {
                vstd::StringBuilder typeName;
                if (ele->is_vector() && ele->dimension() == 3) {
                    GetTypeName(*ele->element(), typeName, usage);
                    typeName << '4';
                } else {
                    GetTypeName(*ele, typeName, usage);
                }
                auto ite = opt->structReplaceName.find(typeName);
                if (ite != opt->structReplaceName.end()) {
                    str << ite->second;
                } else {
                    str << typeName;
                }
            }
            str << '>';
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
            str << "ByteAddressBuffer"sv;
        } break;
        case Type::Tag::ACCEL: {
            str << "RaytracingAccelerationStructure"sv;
        } break;
        case Type::Tag::CUSTOM: {
            str << type.description();
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
        vstd::to_string((opt->GetFuncCount(func.builder())), data);
        if (func.arguments().empty()) {
            data += "()"sv;
        } else {
            data += '(';
            for (auto &&i : func.arguments()) {
                Usage usage = func.variable_usage(i.uid());
                if (i.tag() == Variable::Tag::REFERENCE) {
                    if ((static_cast<uint32_t>(usage) & static_cast<uint32_t>(Usage::WRITE)) != 0) {
                        data += "inout "sv;
                    } else {
                        data += "const "sv;
                    }
                } else {
                    if ((static_cast<uint32_t>(usage) & static_cast<uint32_t>(Usage::WRITE)) == 0) {
                        data += "const "sv;
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
    result << "custom_"sv << vstd::to_string((opt->GetFuncCount(callable.builder())));
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
            str << "custom_"sv << vstd::to_string((opt->GetFuncCount(expr->custom().builder())));
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

        case CallOp::ALL:
            str << "all"sv;
            break;
        case CallOp::ANY:
            str << "any"sv;
            break;
        case CallOp::SELECT:
            str << "select"sv;
            assert(args.size() == 3);
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
            str << "fract"sv;
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
            str << "copysign"sv;
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
            str << "determinant"sv;
            break;
        case CallOp::TRANSPOSE:
            str << "_transpose"sv;
            break;
        case CallOp::INVERSE:
            str << "inverse"sv;
            break;
        // case CallOp::ATOMIC_EXCHANGE: {
        //     if ((expr->type()->tag() == Type::Tag::FLOAT32)) {
        //         str << "_atomic_exchange_float"sv;
        //     } else {
        //         str << "_atomic_exchange"sv;
        //     }
        // } break;
        // case CallOp::ATOMIC_COMPARE_EXCHANGE: {
        //     if ((expr->type()->tag() == Type::Tag::FLOAT32)) {
        //         str << "_atomic_compare_exchange_float"sv;
        //     } else {
        //         str << "_atomic_compare_exchange"sv;
        //     }
        // } break;
        // case CallOp::ATOMIC_FETCH_ADD: {
        //     if ((expr->type()->tag() == Type::Tag::FLOAT32))
        //         str << "_atomic_add_float"sv;
        //     else
        //         str << "_atomic_add"sv;
        // } break;
        // case CallOp::ATOMIC_FETCH_SUB: {
        //     if ((expr->type()->tag() == Type::Tag::FLOAT32))
        //         str << "_atomic_sub_float"sv;
        //     else
        //         str << "_atomic_sub"sv;
        // } break;
        // case CallOp::ATOMIC_FETCH_AND: {
        //     str << "_atomic_and"sv;
        // } break;
        // case CallOp::ATOMIC_FETCH_OR: {
        //     str << "_atomic_or"sv;
        // } break;
        // case CallOp::ATOMIC_FETCH_XOR: {
        //     str << "_atomic_xor"sv;
        // } break;
        // case CallOp::ATOMIC_FETCH_MIN: {
        //     if ((expr->type()->tag() == Type::Tag::FLOAT32))
        //         str << "_atomic_min_float"sv;
        //     else
        //         str << "_atomic_min"sv;
        // } break;
        // case CallOp::ATOMIC_FETCH_MAX: {
        //     if ((expr->type()->tag() == Type::Tag::FLOAT32))
        //         str << "_atomic_max_float"sv;
        //     else
        //         str << "_atomic_max"sv;
        // } break;
        case CallOp::ATOMIC_EXCHANGE:
        case CallOp::ATOMIC_COMPARE_EXCHANGE:
        case CallOp::ATOMIC_FETCH_ADD:
        case CallOp::ATOMIC_FETCH_SUB:
        case CallOp::ATOMIC_FETCH_AND:
        case CallOp::ATOMIC_FETCH_OR:
        case CallOp::ATOMIC_FETCH_XOR:
        case CallOp::ATOMIC_FETCH_MIN:
        case CallOp::ATOMIC_FETCH_MAX: {
            auto &chain = opt->GetAtomicFunc(expr->op(), args[0]->type(), expr->type(), args);
            chain.call_this_func(args, str, vis);
            return;
        }
        case CallOp::TEXTURE_READ:
            str << "Smptx";
            break;
        case CallOp::TEXTURE_WRITE:
            assert(!opt->isRaster);
            str << "Writetx";
            break;
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
        case CallOp::MAKE_INT16_2:
        case CallOp::MAKE_INT16_3:
        case CallOp::MAKE_INT16_4:
        case CallOp::MAKE_UINT16_2:
        case CallOp::MAKE_UINT16_3:
        case CallOp::MAKE_UINT16_4:
        case CallOp::MAKE_FLOAT16_2:
        case CallOp::MAKE_FLOAT16_3:
        case CallOp::MAKE_FLOAT16_4: {
            if (args.size() == 1 && (args[0]->type() == expr->type())) {
                args[0]->accept(vis);
            } else {
                uint tarDim = [&]() -> uint {
                    switch (expr->type()->tag()) {
                        case Type::Tag::VECTOR:
                            return expr->type()->dimension();
                        case Type::Tag::MATRIX:
                            return expr->type()->dimension() * expr->type()->dimension();
                        default:
                            return 1;
                    }
                }();
                if (args.size() == 1) {//  && args[0]->type()->is_scalar()
                    str << "(("sv;
                    GetTypeName(*expr->type(), str, Usage::READ);
                    str << ")("sv;
                    args[0]->accept(vis);
                    str << "))"sv;
                } else {
                    GetTypeName(*expr->type(), str, Usage::READ);
                    str << '(';
                    uint count = 0;
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
            str << "bfread"sv;
            auto elem = args[0]->type()->element();
            if (IsNumVec3(*elem)) {
                str << "Vec3"sv;
            } else if (elem->is_matrix()) {
                str << "Mat";
            }
        } break;
        case CallOp::BUFFER_WRITE: {
            assert(!opt->isRaster);
            str << "bfwrite"sv;
            auto elem = args[0]->type()->element();
            if (IsNumVec3(*elem)) {
                str << "Vec3"sv;
            } else if (elem->is_matrix()) {
                str << "Mat";
            }
        } break;
        case CallOp::RAY_TRACING_TRACE_CLOSEST:
            str << "TraceClosest"sv;
            break;
        case CallOp::RAY_TRACING_TRACE_ANY:
            str << "TraceAny"sv;
            break;
        case CallOp::RAY_TRACING_QUERY_ALL:
            str << "QueryAll"sv;
            break;
        case CallOp::RAY_TRACING_QUERY_ANY:
            str << "QueryAny"sv;
            break;
        case CallOp::BINDLESS_BUFFER_READ: {
            str << "READ_BUFFER"sv;
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
        case CallOp::ASSUME:
        case CallOp::UNREACHABLE: {
            return;
        }
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE:
            opt->useTex2DBindless = true;
            if (opt->isPixelShader) {
                str << "SampleTex2DPixel"sv;
            } else {
                str << "SampleTex2D"sv;
            }
            break;

        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL:
            opt->useTex2DBindless = true;
            str << "SampleTex2DLevel"sv;
            break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD:
            opt->useTex2DBindless = true;
            str << "SampleTex2DGrad"sv;
            break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE:
            opt->useTex3DBindless = true;
            str << "SampleTex3D"sv;
            break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL:
            opt->useTex3DBindless = true;
            str << "SampleTex3DLevel"sv;
            break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD:
            opt->useTex3DBindless = true;
            str << "SampleTex3DGrad"sv;
            break;
        case CallOp::BINDLESS_TEXTURE2D_READ:
            opt->useTex2DBindless = true;
            str << "ReadTex2D"sv;
            break;
        case CallOp::BINDLESS_TEXTURE2D_READ_LEVEL:
            opt->useTex2DBindless = true;
            str << "ReadTex2DLevel"sv;
            break;
        case CallOp::BINDLESS_TEXTURE3D_READ:
            opt->useTex3DBindless = true;
            str << "ReadTex3D"sv;
            break;
        case CallOp::BINDLESS_TEXTURE3D_READ_LEVEL:
            opt->useTex3DBindless = true;
            str << "ReadTex3DLevel"sv;
            break;
        case CallOp::BINDLESS_TEXTURE2D_SIZE:
            opt->useTex2DBindless = true;
            str << "Tex2DSize"sv;
            break;
        case CallOp::BINDLESS_TEXTURE2D_SIZE_LEVEL:
            opt->useTex2DBindless = true;
            str << "Tex2DSizeLevel"sv;
            break;
        case CallOp::BINDLESS_TEXTURE3D_SIZE:
            opt->useTex3DBindless = true;
            str << "Tex3DSize"sv;
            break;
        case CallOp::BINDLESS_TEXTURE3D_SIZE_LEVEL:
            opt->useTex3DBindless = true;
            str << "Tex3DSizeLevel"sv;
            break;
        case CallOp::SYNCHRONIZE_BLOCK:
            str << "GroupMemoryBarrierWithGroupSync()"sv;
            return;
        case CallOp::RASTER_DISCARD:
            assert(opt->funcType == CodegenStackData::FuncType::Pixel);
            str << "discard";
            return;
        case CallOp::RAY_TRACING_INSTANCE_TRANSFORM: {
            str << "InstMatrix("sv;
            args[0]->accept(vis);
            str << "Inst,"sv;
            args[1]->accept(vis);
            str << ')';
            return;
        }
        case CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM: {
            str << "SetAccelTransform("sv;
            args[0]->accept(vis);
            str << "Inst,"sv;
            PrintArgs(1);
            str << ')';
            return;
        }
        case CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY: {
            str << "SetAccelVis("sv;
            args[0]->accept(vis);
            str << "Inst,"sv;
            PrintArgs(1);
            str << ')';
            return;
        }
        case CallOp::RAY_TRACING_SET_INSTANCE_OPACITY: {
            str << "SetAccelOpaque("sv;
            args[0]->accept(vis);
            str << "Inst,"sv;
            PrintArgs(1);
            str << ')';
            return;
        }
        case CallOp::INDIRECT_CLEAR_DISPATCH_BUFFER:
            str << "ClearDispInd"sv;
            break;
        case CallOp::INDIRECT_EMPLACE_DISPATCH_KERNEL: {
            assert(!opt->isRaster);
            auto tp = args[1]->type();
            if (tp->is_scalar()) {
                str << "EmplaceDispInd1D"sv;
            } else if (tp->dimension() == 2) {
                str << "EmplaceDispInd2D"sv;
            } else {
                str << "EmplaceDispInd3D"sv;
            }
        } break;
        case CallOp::RAY_QUERY_WORLD_SPACE_RAY:
            str << "RayQueryGetWorldRay<"sv;
            GetTypeName(*expr->type(), str, Usage::NONE, false);
            str << ',';
            GetTypeName(*args[0]->type(), str, Usage::NONE, false);
            str << '>';
            break;
        case CallOp::RAY_QUERY_TRIANGLE_CANDIDATE_HIT:
            str << "GetTriangleCandidateHit"sv;
            break;
        case CallOp::RAY_QUERY_PROCEDURAL_CANDIDATE_HIT:
            str << "GetProceduralCandidateHit"sv;
            break;
        case CallOp::RAY_QUERY_COMMITTED_HIT:
            str << "GetCommitedHit"sv;
            break;
        case CallOp::RAY_QUERY_COMMIT_TRIANGLE:
            args[0]->accept(vis);
            str << ".CommitNonOpaqueTriangleHit()"sv;
            return;
        case CallOp::RAY_QUERY_COMMIT_PROCEDURAL:
            args[0]->accept(vis);
            str << ".CommitProceduralPrimitiveHit("sv;
            args[1]->accept(vis);
            str << ')';
            return;
        case CallOp::RAY_QUERY_TERMINATE:
            args[0]->accept(vis);
            str << ".Abort()"sv;
            return;
        default: {
            LUISA_ERROR("Function Not Implemented");
        } break;
    }
    str << '(';
    PrintArgs();
    str << ')';
}
size_t CodegenUtility::GetTypeSize(Type const &t) {
    switch (t.tag()) {
        case Type::Tag::BOOL:
            return 1;
        case Type::Tag::FLOAT16:
        case Type::Tag::INT16:
        case Type::Tag::UINT16:
            return 2;
        case Type::Tag::FLOAT32:
        case Type::Tag::INT32:
        case Type::Tag::UINT32:
            return 4;
        case Type::Tag::VECTOR:
            switch (t.dimension()) {
                case 1:
                    return 4;
                case 2:
                    return 8;
                default:
                    return 16;
            }
        case Type::Tag::MATRIX: {
            return 4 * t.dimension() * sizeof(float);
        }
        case Type::Tag::STRUCTURE: {
            size_t v = 0;
            size_t maxAlign = 0;
            for (auto &&i : t.members()) {
                auto align = GetTypeAlign(*i);
                v = detail::CalcAlign(v, align);
                maxAlign = std::max(align, align);
                v += GetTypeSize(*i);
            }
            v = detail::CalcAlign(v, maxAlign);
            return v;
        }
        case Type::Tag::ARRAY: {
            return GetTypeSize(*t.element()) * t.dimension();
        }
        default:
            return 0;
    }
}

size_t CodegenUtility::GetTypeAlign(Type const &t) {// TODO: use t.alignment()
    switch (t.tag()) {
        case Type::Tag::BOOL:
            return 1;
        case Type::Tag::FLOAT16:
        case Type::Tag::INT16:
        case Type::Tag::UINT16:
            return 2;
        case Type::Tag::FLOAT32:
        case Type::Tag::INT32:
        case Type::Tag::UINT32:
            return 4;
            // TODO: incorrect
        case Type::Tag::VECTOR:
            switch (t.dimension()) {
                case 1:
                    return 4;
                case 2:
                    return 8;
                default:
                    return 16;
            }
        case Type::Tag::MATRIX: {
            return 16;
        }
        case Type::Tag::ARRAY: {
            return GetTypeAlign(*t.element());
        }
        case Type::Tag::STRUCTURE: {
            return 16;
        }
        case Type::Tag::BUFFER:
        case Type::Tag::TEXTURE:
        case Type::Tag::ACCEL:
        case Type::Tag::BINDLESS_ARRAY:
            return 8;
        default:
            LUISA_ERROR_WITH_LOCATION(
                "Invalid type: {}.", t.description());
    }
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
void CodegenUtility::GetBasicTypeName(uint64 typeIndex, vstd::StringBuilder &str) {
    vstd::VariantVisitor_t<basic_types>()(
        [&]<typename T>() {
            TypeNameStruct<T>()(str);
        },
        typeIndex);
}
void CodegenUtility::CodegenFunction(Function func, vstd::StringBuilder &result, bool cbufferNonEmpty) {

    auto codegenOneFunc = [&](Function func) {
        auto constants = func.constants();
        for (auto &&i : constants) {
            vstd::StringBuilder constValueName;
            if (!GetConstName(i.data.hash(), i.data, constValueName)) continue;
            result << "static const "sv;
            GetTypeName(*i.type->element(), result, Usage::READ);
            result << ' ' << constValueName << '[';
            vstd::to_string(i.type->dimension(), result);
            result << "]={"sv;
            auto &&dataView = i.data.view();
            luisa::visit(
                [&]<typename T>(vstd::span<T> const &sp) {
                    for (auto i : vstd::range(sp.size())) {
                        auto &&value = sp[i];
                        PrintValue<std::remove_cvref_t<T>>()(value, result);
                        if (i != (sp.size() - 1)) {
                            result << ',';
                        }
                    }
                },
                dataView);
            result << "};\n"sv;
        }

        if (func.tag() == Function::Tag::KERNEL) {
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
                result << "Args a = _Global[0];\n"sv;
            }
            opt->funcType = CodegenStackData::FuncType::Kernel;
            opt->arguments.clear();
            opt->arguments.reserve(func.arguments().size());
            size_t idx = 0;
            for (auto &&i : func.arguments()) {
                opt->arguments.try_emplace(i.uid(), idx);
                ++idx;
            }
        } else {
            GetFunctionDecl(func, result);
            result << "{\n"sv;
            opt->funcType = CodegenStackData::FuncType::Callable;
        }
        {

            StringStateVisitor vis(func, result, this);
            vis.sharedVariables = &opt->sharedVariable;
            vis.VisitFunction(func);
        }
        result << "}\n"sv;
    };
    vstd::unordered_set<void const *> callableMap;
    auto callable = [&](auto &&callable, Function func) -> void {
        for (auto &&i : func.custom_callables()) {
            if (callableMap.emplace(i.get()).second) {
                Function f(i.get());
                callable(callable, f);
            }
        }
        codegenOneFunc(func);
    };
    callable(callable, func);
}
void CodegenUtility::CodegenVertex(Function vert, vstd::StringBuilder &result, bool cBufferNonEmpty, bool isDX, vstd::function<void(vstd::StringBuilder &)> const &bindVertex) {
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
    auto args = vert.arguments();
    gen(callable, vert);
    vstd::StringBuilder retName;
    auto retType = vert.return_type();
    GetTypeName(*retType, retName, Usage::READ);
    result << "template<typename T>\n"sv << retName << " vert(T vt){\n"sv;
    if (cBufferNonEmpty) {
        result << "Args a = _Global[0];\n"sv;
    }
    GetTypeName(*args[0].type(), result, Usage::NONE);
    result << " vv;\n"sv;
    bindVertex(result);
    opt->funcType = CodegenStackData::FuncType::Vert;
    opt->arguments.clear();
    opt->arguments.reserve(args.size() - 1);
    size_t idx = 0;
    for (auto &&i : vstd::ite_range(args.begin() + 1, args.end())) {
        opt->arguments.try_emplace(i.uid(), idx);
        ++idx;
    }
    {
        StringStateVisitor vis(vert, result, this);
        vis.sharedVariables = &opt->sharedVariable;
        vis.VisitFunction(vert);
    }
    result << R"(
}
v2p main(vertex vt){
v2p o;
)"sv;

    if (retType->is_vector()) {
        result << "o.v0=vert(vt);\n"sv;
    } else {
        result << retName
               << " r=vert(vt);\n"sv;
        for (auto i : vstd::range(retType->members().size())) {
            auto num = vstd::to_string(i);
            result << "o.v"sv << num << "=r.v"sv << num << ";\n"sv;
        }
    }
    if (isDX) {
        result << "o.v0.y*=-1.0;\n"sv;
    }
    result << R"(return o;
}
)"sv;
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
        result << "Args a = _Global[0];\n"sv;
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
    {
        StringStateVisitor vis(pixel, result, this);
        vis.sharedVariables = &opt->sharedVariable;
        vis.VisitFunction(pixel);
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
    vstd::StringBuilder &result) {
    result << "struct Args{\n"sv;
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
StructuredBuffer<Args> _Global:register(t0);
)"sv;
}
void CodegenUtility::GenerateBindless(
    CodegenResult::Properties &properties,
    vstd::StringBuilder &str,
    bool isSpirV) {
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
    }
    if (opt->useTex2DBindless) {
        str << "Texture2D<float4> _BindlessTex[]:register(t0,space"sv << vstd::to_string(table_idx) << ");"sv;
        add_prop(ShaderVariableType::SRVTextureHeap);
        table_idx++;
        str << R"(
float4 SampleTex2DLevel(const ByteAddressBuffer arr,const uint index,const float2 uv,const float level){SamplerState samp=samplers[NonUniformResourceIndex(ReadBdlsSmp2D(arr,index))];return _BindlessTex[NonUniformResourceIndex(ReadBdlsTex2D(arr,index))].SampleLevel(samp,uv,level);}
float4 SampleTex2D(const ByteAddressBuffer arr,const uint index,const float2 uv){return SampleTex2DLevel(arr,index,uv,0);}
float4 SampleTex2DGrad(const ByteAddressBuffer arr,const uint index,const float2 uv,const float2 ddx,const float2 ddy){SamplerState samp=samplers[NonUniformResourceIndex(ReadBdlsSmp2D(arr,index))];return _BindlessTex[NonUniformResourceIndex(ReadBdlsTex2D(arr,index))].SampleGrad(samp,uv,ddx,ddy);}
#ifdef PS
float4 SampleTex2DPixel(const ByteAddressBuffer arr,const uint index,const float2 uv){SamplerState samp=samplers[NonUniformResourceIndex(ReadBdlsSmp2D(arr,index))];return _BindlessTex[NonUniformResourceIndex(ReadBdlsTex2D(arr,index))].Sample(samp,uv);}
#endif
float4 ReadTex2DLevel(const ByteAddressBuffer arr,const uint index,const uint2 coord,const uint level){return _BindlessTex[NonUniformResourceIndex(ReadBdlsTex2D(arr,index))].Load(uint3(coord,level));}
float4 ReadTex2D(const ByteAddressBuffer arr,const uint index,const uint2 coord){return ReadTex2DLevel(arr,index,coord,0);}
uint2 Tex2DSize(const ByteAddressBuffer arr,const uint index){return ReadBdlsTex2DXY(arr,index);}
uint2 Tex2DSizeLevel(const ByteAddressBuffer arr,const uint index,const uint level){return max(Tex2DSize(arr,index)>>level,1u);}
)"sv;
    }
    if (opt->useTex3DBindless) {
        str << "Texture3D<float4> _BindlessTex3D[]:register(t0,space"sv << vstd::to_string(table_idx) << ");"sv;
        add_prop(ShaderVariableType::SRVTextureHeap);
        table_idx++;
        str << R"(
float4 SampleTex3DLevel(const ByteAddressBuffer arr,const uint index,const float3 uv,const float level){SamplerState samp=samplers[NonUniformResourceIndex(ReadBdlsSmp3D(arr,index))];return _BindlessTex3D[NonUniformResourceIndex(ReadBdlsTex3D(arr,index))].SampleLevel(samp,uv,level);}
float4 SampleTex3D(const ByteAddressBuffer arr,const uint index,const float3 uv){return SampleTex3DLevel(arr,index,uv,0);}
float4 SampleTex3DGrad(const ByteAddressBuffer arr,const uint index,const float3 uv,const float3 ddx,const float3 ddy){SamplerState samp=samplers[NonUniformResourceIndex(ReadBdlsSmp3D(arr,index))];return _BindlessTex3D[NonUniformResourceIndex(ReadBdlsTex3D(arr,index))].SampleGrad(samp,uv,ddx,ddy);}
float4 ReadTex3DLevel(const ByteAddressBuffer arr,const uint index,const uint3 coord,const Lvl level){return _BindlessTex3D[NonUniformResourceIndex(ReadBdlsTex3D(arr,index))].Load(uint4(coord,level));}
float4 ReadTex3D(const ByteAddressBuffer arr,const uint index,const uint3 coord){return ReadTex3DLevel(arr,index,coord,0);}
uint3 Tex3DSize(const ByteAddressBuffer arr,const uint index){return ReadBdlsTex3DXYZ(arr,index);}
uint3 Tex3DSizeLevel(const ByteAddressBuffer arr,const uint index,const uint level){return max(Tex3DSize(arr,index)>>level,1u);}
)"sv;
    }
}

void CodegenUtility::PreprocessCodegenProperties(
    CodegenResult::Properties &properties, vstd::StringBuilder &varData, RegisterIndexer &registerCount, bool cbufferNonEmpty,
    bool isRaster, bool isSpirv) {
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
            ShaderVariableType::SampHeap,
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
    GenerateBindless(properties, varData, isSpirv);
}
void CodegenUtility::PostprocessCodegenProperties(vstd::StringBuilder &finalResult) {
    if (!opt->customStruct.empty()) {
        vstd::fixed_vector<StructGenerator *, 16> customStructVector;
        customStructVector.reserve(opt->customStruct.size());
        for (auto &&i : opt->customStruct) {
            customStructVector.emplace_back(i.second.get());
        }
        pdqsort(customStructVector.begin(), customStructVector.end(), [](auto lhs, auto rhs) noexcept {
            return lhs->GetType()->index() < rhs->GetType()->index();
        });
        for (auto v : customStructVector) {
            finalResult << "struct " << v->GetStructName() << "{\n"
                        << v->GetStructDesc() << "};\n";
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
void CodegenUtility::CodegenProperties(
    CodegenResult::Properties &properties,
    vstd::StringBuilder &varData,
    Function kernel,
    uint offset,
    RegisterIndexer &registerCount) {
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
            GetTypeName(*i.type(), varData, kernel.variable_usage(i.uid()));
            varData << ' ';
            GetVariableName(i, varData);
        };
        auto printInstBuffer = [&]<bool writable>() {
            if constexpr (writable)
                varData << "RWStructuredBuffer<MeshInst> "sv;
            else
                varData << "StructuredBuffer<MeshInst> "sv;
            GetVariableName(i, varData);
            varData << "Inst"sv;
        };
        auto genArg = [&]<bool rtBuffer = false, bool writable = false>(RegisterType regisT, ShaderVariableType sT, char v) {
            auto &&r = registerCount.get((uint8_t)regisT);
            Property prop = {
                .type = sT,
                .spaceIndex = 0,
                .registerIndex = r,
                .arrSize = 1};
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
        };

        switch (i.type()->tag()) {
            case Type::Tag::TEXTURE:
                if (Writable(i)) {
                    genArg(RegisterType::UAV, ShaderVariableType::UAVTextureHeap, 'u');
                } else {
                    genArg(RegisterType::SRV, ShaderVariableType::SRVTextureHeap, 't');
                }
                break;
            case Type::Tag::BUFFER: {
                if (Writable(i)) {
                    genArg(RegisterType::UAV, ShaderVariableType::RWStructuredBuffer, 'u');
                } else {
                    genArg(RegisterType::SRV, ShaderVariableType::StructuredBuffer, 't');
                }
            } break;
            case Type::Tag::BINDLESS_ARRAY:
                genArg(RegisterType::SRV, ShaderVariableType::StructuredBuffer, 't');
                break;
            case Type::Tag::ACCEL:
                if (Writable(i)) {
                    genArg.operator()<true, true>(RegisterType::UAV, ShaderVariableType::RWStructuredBuffer, 'u');
                } else {
                    genArg(RegisterType::SRV, ShaderVariableType::StructuredBuffer, 't');
                    genArg.operator()<true>(RegisterType::SRV, ShaderVariableType::StructuredBuffer, 't');
                }
                break;
            default: break;
        }
    }
}
vstd::MD5 CodegenUtility::GetTypeMD5(vstd::span<Type const *const> types) {
    vstd::vector<uint64_t> typeDescs;
    typeDescs.reserve(types.size());
    for (auto &&i : types) {
        typeDescs.emplace_back(i->hash());
    }
    return {vstd::span<uint8_t const>(reinterpret_cast<uint8_t const *>(typeDescs.data()), typeDescs.size_bytes())};
}
vstd::MD5 CodegenUtility::GetTypeMD5(std::initializer_list<vstd::IRange<Variable> *> f) {
    vstd::vector<uint64_t> typeDescs;
    for (auto &&rg : f) {
        for (auto &&i : *rg) {
            typeDescs.emplace_back(i.type()->hash());
        }
    }
    return {vstd::span<uint8_t const>(reinterpret_cast<uint8_t const *>(typeDescs.data()), typeDescs.size_bytes())};
}
vstd::MD5 CodegenUtility::GetTypeMD5(Function func) {
    vstd::vector<uint64_t> typeDescs;
    auto args = func.arguments();
    typeDescs.reserve(args.size());
    for (auto &&i : args) {
        typeDescs.emplace_back(i.type()->hash());
    }
    return {vstd::span<uint8_t const>(reinterpret_cast<uint8_t const *>(typeDescs.data()), typeDescs.size_bytes())};
}
CodegenUtility::CodegenUtility() {}
CodegenUtility::~CodegenUtility() {}

CodegenResult CodegenUtility::Codegen(
    Function kernel, luisa::BinaryIO const *internalDataPath, bool isSpirV) {
    assert(kernel.tag() == Function::Tag::KERNEL);
    opt = CodegenStackData::Allocate(this);
    auto disposeOpt = vstd::scope_exit([&] {
        CodegenStackData::DeAllocate(std::move(opt));
    });
    // CodegenStackData::ThreadLocalSpirv() = false;
    opt->kernel = kernel;
    bool nonEmptyCbuffer = IsCBufferNonEmpty(kernel);

    vstd::StringBuilder codegenData;
    vstd::StringBuilder varData;
    vstd::StringBuilder finalResult;
    opt->finalResult = &finalResult;
    finalResult.reserve(65500);
    finalResult << detail::HLSLHeader(this, internalDataPath);
    if (kernel.requires_raytracing()) {
        finalResult << detail::RayTracingHeader(this, internalDataPath);
    }

    CodegenFunction(kernel, codegenData, nonEmptyCbuffer);

    opt->funcType = CodegenStackData::FuncType::Callable;
    auto argRange = vstd::RangeImpl(vstd::CacheEndRange(kernel.arguments()) | vstd::ValueRange{});
    if (nonEmptyCbuffer) {
        GenerateCBuffer({static_cast<vstd::IRange<Variable> *>(&argRange)}, varData);
    }
    if (isSpirV) {
        varData << R"(cbuffer CB:register(b1){
uint4 dsp_c;
}
)"sv;
    } else {
        varData << "uint4 dsp_c:register(b0);\n"sv;
    }
    CodegenResult::Properties properties;
    uint64 immutableHeaderSize = finalResult.size();
    DXILRegisterIndexer dxilRegisters;
    SpirVRegisterIndexer spvRegisters;
    RegisterIndexer &indexer = isSpirV ? static_cast<RegisterIndexer &>(spvRegisters) : static_cast<RegisterIndexer &>(dxilRegisters);
    PreprocessCodegenProperties(properties, varData, indexer, nonEmptyCbuffer, false, isSpirV);
    CodegenProperties(properties, varData, kernel, 0, indexer);
    PostprocessCodegenProperties(finalResult);
    finalResult << varData << codegenData;
    return {
        std::move(finalResult),
        std::move(properties),
        opt->useTex2DBindless,
        opt->useTex3DBindless,
        opt->useBufferBindless,
        immutableHeaderSize,
        GetTypeMD5(kernel)};
}
CodegenResult CodegenUtility::RasterCodegen(
    MeshFormat const &meshFormat,
    Function vertFunc,
    Function pixelFunc,
    luisa::BinaryIO const *internalDataPath,
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
    opt->finalResult = &finalResult;
    finalResult.reserve(65500);
    finalResult << detail::HLSLHeader(this, internalDataPath);
    if (vertFunc.requires_raytracing() || pixelFunc.requires_raytracing()) {
        finalResult << detail::RayTracingHeader(this, internalDataPath);
    }
    // Vertex
    codegenData << "struct v2p{\n"sv;
    auto v2pType = vertFunc.return_type();
    if (v2pType->is_structure()) {
        size_t memberIdx = 0;
        for (auto &&i : v2pType->members()) {
            GetTypeName(*i, codegenData, Usage::READ);
            codegenData << " v"sv << vstd::to_string(memberIdx);
            if (memberIdx == 0) {
                codegenData << ":SV_POSITION;\n"sv;
            } else {
                codegenData << ":TEXCOORD"sv << vstd::to_string(memberIdx - 1) << ";\n"sv;
            }
            ++memberIdx;
        }
    } else if (v2pType->is_vector() && v2pType->dimension() == 4) {
        codegenData << "float4 v0:SV_POSITION;\n"sv;
    } else {
        LUISA_ERROR("Illegal vertex return type!");
    }
    if (isSpirV) {
        codegenData << R"(};
cbuffer CB:register(b1){
uint obj_id;
)"sv;
    } else {
        codegenData << R"(};
uint obj_id:register(b0);
)"sv;
    }
    codegenData << "#ifdef VS\n";
    std::bitset<kVertexAttributeCount> bits;
    bits.reset();
    auto vertexAttriName = {
        "position"sv,
        "normal"sv,
        "tangent"sv,
        "color"sv,
        "uv0"sv,
        "uv1"sv,
        "uv2"sv,
        "uv3"sv};
    auto semanticName = {
        "POSITION"sv,
        "NORMAL"sv,
        "TANGENT"sv,
        "COLOR"sv,
        "UV0"sv,
        "UV1"sv,
        "UV2"sv,
        "UV3"sv};
    auto semanticType = {
        "float3"sv,
        "float3"sv,
        "float4"sv,
        "float4"sv,
        "float2"sv,
        "float2"sv,
        "float2"sv,
        "float2"sv};
    auto PrintSetValue = [&](vstd::StringBuilder &d) {
        for (auto i : vstd::range(meshFormat.vertex_stream_count())) {
            for (auto &&j : meshFormat.attributes(i)) {
                auto type = j.type;
                auto idx = static_cast<size_t>(type);
                assert(!bits[idx]);
                bits[idx] = true;
                auto name = vertexAttriName.begin()[idx];
                if (idx >= 4) {
                    d << "vv.v4.v["sv << vstd::to_string(idx - 4) << "]=vt."sv << name << ";\n"sv;
                } else {
                    d << "vv.v"sv << vstd::to_string(i);
                    if (i < 2) {
                        d << ".v"sv;
                    }
                    d << "=vt."sv << name << ";\n"sv;
                }
            }
        }
        for (auto i : vstd::range(bits.size())) {
            if (bits[i]) continue;
            if (i >= 4) {
                d << "vv.v4.v["sv << vstd::to_string(i - 4) << "]=0;\n"sv;
            } else {
                d << "vv.v"sv << vstd::to_string(i);
                if (i < 2) {
                    d << ".v"sv;
                }
                d << "=0;\n"sv;
            }
        }
        d << "vv.v5=vt.vid;\nvv.v6=vt.iid;\n"sv;
    };

    codegenData << "struct vertex{\n"sv;
    for (auto i : vstd::range(meshFormat.vertex_stream_count())) {
        for (auto &&j : meshFormat.attributes(i)) {
            auto type = j.type;
            auto idx = static_cast<size_t>(type);
            codegenData << semanticType.begin()[idx] << ' ' << vertexAttriName.begin()[idx] << ':' << semanticName.begin()[idx] << ";\n"sv;
        }
    }
    codegenData << R"(uint vid:SV_VERTEXID;
uint iid:SV_INSTANCEID;
};
)"sv;
    auto vertRange = vstd::RangeImpl(vstd::CacheEndRange(vstd::ite_range(vertFunc.arguments().begin() + 1, vertFunc.arguments().end())) | vstd::ValueRange{});
    auto pixelRange = vstd::RangeImpl(vstd::ite_range(pixelFunc.arguments().begin() + 1, pixelFunc.arguments().end()) | vstd::ValueRange{});
    std::initializer_list<vstd::IRange<Variable> *> funcs = {&vertRange, &pixelRange};

    bool nonEmptyCbuffer = IsCBufferNonEmpty(funcs);
    opt->appdataId = vertFunc.arguments()[0].uid();
    CodegenVertex(vertFunc, codegenData, nonEmptyCbuffer, !isSpirV, PrintSetValue);
    opt->appdataId = -1;
    // TODO: gen vertex data
    codegenData << "#elif defined(PS)\n"sv;
    opt->argOffset = vertFunc.arguments().size() - 1;
    // TODO: gen pixel data
    CodegenPixel(pixelFunc, codegenData, nonEmptyCbuffer);
    codegenData << "#endif\n"sv;

    opt->funcType = CodegenStackData::FuncType::Callable;
    if (nonEmptyCbuffer) {
        GenerateCBuffer(funcs, varData);
    }
    CodegenResult::Properties properties;
    uint64 immutableHeaderSize = finalResult.size();
    DXILRegisterIndexer dxilRegisters;
    SpirVRegisterIndexer spvRegisters;
    RegisterIndexer &indexer = isSpirV ? static_cast<RegisterIndexer &>(spvRegisters) : static_cast<RegisterIndexer &>(dxilRegisters);
    PreprocessCodegenProperties(properties, varData, indexer, nonEmptyCbuffer, true, isSpirV);
    CodegenProperties(properties, varData, vertFunc, 1, indexer);
    CodegenProperties(properties, varData, pixelFunc, 1, indexer);
    PostprocessCodegenProperties(finalResult);
    finalResult << varData << codegenData;
    return {
        std::move(finalResult),
        std::move(properties),
        opt->useTex2DBindless,
        opt->useTex3DBindless,
        opt->useBufferBindless,
        immutableHeaderSize,
        GetTypeMD5(funcs)};
}
}// namespace lc::hlsl
