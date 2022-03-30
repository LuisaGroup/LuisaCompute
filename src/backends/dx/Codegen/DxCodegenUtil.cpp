

#include <Codegen/DxCodegen.h>
#include <vstl/StringUtility.h>
#include <d3dx12.h>
#include <vstl/variant_util.h>
#include <ast/constant_data.h>
#include <Codegen/StructGenerator.h>
#include <Codegen/ShaderHeader.h>
#include <Codegen/CodegenStackData.h>
namespace toolhub::directx {
static thread_local vstd::unique_ptr<CodegenStackData> opt;
uint CodegenUtility::IsBool(Type const &type) {
    if (type.tag() == Type::Tag::BOOL) {
        return 1;
    } else if (type.tag() == Type::Tag::VECTOR && type.element()->tag() == Type::Tag::BOOL) {
        return type.dimension();
    }
    return 0;
};
vstd::string CodegenUtility::GetNewTempVarName() {
    vstd::string name = "tmp";
    vstd::to_string(opt->tempCount, name);
    opt->tempCount++;
    return name;
}
StructGenerator const *CodegenUtility::GetStruct(
    Type const *type) {
    return opt->CreateStruct(type);
}
void CodegenUtility::RegistStructType(Type const *type) {
    if (type->is_structure() || type->is_array())
        opt->structTypes.Emplace(type, opt->count++);
    else if (type->is_buffer()) {
        RegistStructType(type->element());
    }
}

static bool IsVarWritable(Function func, Variable i) {
    return ((uint)func.variable_usage(i.uid()) & (uint)Usage::WRITE) != 0;
}
void CodegenUtility::GetVariableName(Variable::Tag type, uint id, vstd::string &str) {
    switch (type) {
        case Variable::Tag::BLOCK_ID:
            str << "thdId"sv;
            break;
        case Variable::Tag::DISPATCH_ID:
            str << "dspId"sv;
            break;
        case Variable::Tag::THREAD_ID:
            str << "grpId"sv;
            break;
        case Variable::Tag::DISPATCH_SIZE:
            str << "a.dsp_c"sv;
            break;
        case Variable::Tag::LOCAL:
            if (opt->isKernel && opt->arguments.Find(id)) {
                str << "a."sv;
            }
            str << 'l';
            vstd::to_string(id, str);
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

void CodegenUtility::GetVariableName(Variable const &type, vstd::string &str) {
    GetVariableName(type.tag(), type.uid(), str);
}
void CodegenUtility::GetConstName(ConstantData const &data, vstd::string &str) {
    uint64 constCount = opt->GetConstCount(data.hash());
    str << "c";
    vstd::to_string((constCount), str);
}
void CodegenUtility::GetConstantStruct(ConstantData const &data, vstd::string &str) {
    uint64 constCount = opt->GetConstCount(data.hash());
    //auto typeName = CodegenUtility::GetBasicTypeName(view.index());
    str << "struct tc";
    vstd::to_string((constCount), str);
    uint64 varCount = 1;
    eastl::visit(
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
void CodegenUtility::GetConstantData(ConstantData const &data, vstd::string &str) {
    auto &&view = data.view();
    uint64 constCount = opt->GetConstCount(data.hash());

    vstd::string name = vstd::to_string((constCount));
    str << "uniform const tc" << name << " c" << name;
    str << "={{";
    eastl::visit(
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

void CodegenUtility::GetTypeName(Type const &type, vstd::string &str, Usage usage) {
    switch (type.tag()) {
        case Type::Tag::BOOL:
            str << "bool"sv;
            return;
        case Type::Tag::FLOAT:
            str << "float"sv;
            return;
        case Type::Tag::INT:
            str << "int"sv;
            return;
        case Type::Tag::UINT:
            str << "uint"sv;
            return;
        case Type::Tag::MATRIX: {
            CodegenUtility::GetTypeName(*type.element(), str, usage);
            vstd::to_string(type.dimension(), str);
            str << 'x';
            vstd::to_string((type.dimension() == 3) ? 4 : type.dimension(), str);
        }
            return;
        case Type::Tag::VECTOR: {
            CodegenUtility::GetTypeName(*type.element(), str, usage);
            vstd::to_string((type.dimension()), str);
        }
            return;
        case Type::Tag::ARRAY:
        case Type::Tag::STRUCTURE: {
            auto customType = opt->CreateStruct(&type);
            str << customType->GetStructName();
        }
            return;
        case Type::Tag::BUFFER: {

            if ((static_cast<uint>(usage) & static_cast<uint>(Usage::WRITE)) != 0)
                str << "RW"sv;
            str << "StructuredBuffer<"sv;
            auto ele = type.element();
            if (ele->is_matrix()) {
                auto n = ele->dimension();
                str << luisa::format("WrappedFloat{}x{}", n, n);
            } else {
                vstd::string typeName;
                if (ele->is_vector() && ele->dimension() == 3) {
                    typeName << "float4"sv;
                } else {
                    GetTypeName(*ele, typeName, usage);
                }
                auto ite = opt->structReplaceName.Find(typeName);
                if (ite) {
                    str << ite.Value();
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
            str << "BINDLESS_ARRAY"sv;
        } break;
        case Type::Tag::ACCEL: {
            str << "RaytracingAccelerationStructure"sv;
        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("Bad.");
            break;
    }
}

void CodegenUtility::GetFunctionDecl(Function func, vstd::string &data) {
    if (func.return_type()) {
        //TODO: return type
        CodegenUtility::GetTypeName(*func.return_type(), data, Usage::READ);
    } else {
        data += "void"sv;
    }
    switch (func.tag()) {
        case Function::Tag::CALLABLE: {
            data += " custom_"sv;
            vstd::to_string((opt->GetFuncCount(func.hash())), data);
            if (func.arguments().empty()) {
                data += "()"sv;
            } else {
                data += '(';
                for (auto &&i : func.arguments()) {
                    if (i.tag() == Variable::Tag::REFERENCE) {
                        data += "inout ";
                    }
                    RegistStructType(i.type());
                    Usage usage = func.variable_usage(i.uid());

                    vstd::string varName;
                    CodegenUtility::GetVariableName(i, varName);
                    if (i.type()->is_accel()) {
                        if (usage == Usage::READ) {
                            CodegenUtility::GetTypeName(*i.type(), data, usage);
                            data << ' '
                                 << varName << ",StructuredBuffer<MeshInst> " << varName << "Inst,"sv;
                        } else {
                            data << "RWStructuredBuffer<MeshInst> "sv << varName << "Inst,"sv;
                        }
                    } else {
                        CodegenUtility::GetTypeName(*i.type(), data, usage);
                        data << ' ';
                        data << varName << ',';
                    }
                }
                data[data.size() - 1] = ')';
            }
        } break;
    }
}
void CodegenUtility::GetFunctionName(CallExpr const *expr, vstd::string &str, StringStateVisitor &vis) {
    auto args = expr->arguments();
    auto GenMakeFunc = [&]() {
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
        auto is_make_matrix = expr->type()->is_matrix();
        auto n = luisa::format("{}", expr->type()->dimension());
        if (args.size() == 1 && args[0]->type()->is_scalar()) {
            str << "(("sv;
            if (is_make_matrix) {
                str << "make_float" << n << "x" << n;
            } else {
                GetTypeName(*expr->type(), str, Usage::READ);
            }
            str << ")("sv;
            for (auto &&i : args) {
                i->accept(vis);
                str << ',';
            }
            *(str.end() - 1) = ')';
            str << ')';
        } else {
            if (is_make_matrix) {
                str << "make_float" << n << "x" << n;
            } else {
                GetTypeName(*expr->type(), str, Usage::READ);
            }
            str << '(';
            uint count = 0;
            for (auto &&i : args) {
                i->accept(vis);
                if (i->type()->is_vector()) {
                    auto dim = i->type()->dimension();
                    auto ele = i->type()->element();
                    auto leftEle = tarDim - count;
                    //More lefted
                    if (dim <= leftEle) {
                    } else {
                        auto swizzle = "xyzw";
                        str << '.' << vstd::string_view(swizzle, leftEle);
                    }
                    count += dim;
                } else if (i->type()->is_scalar()) {
                    count++;
                }
                str << ',';
                if (count >= tarDim) break;
            }
            if (count < tarDim) {
                for (auto i : vstd::range(tarDim - count)) {
                    str << "0,"sv;
                }
            }
            *(str.end() - 1) = ')';
        }
    };
    auto getPointer = [&]() {
        str << '(';
        uint64 sz = 1;
        if (args.size() >= 1) {
            str << "&(";
            args[0]->accept(vis);
            str << "),";
        }
        for (auto i : vstd::range(1, args.size())) {
            ++sz;
            args[i]->accept(vis);
            if (sz != args.size()) {
                str << ',';
            }
        }
        str << ')';
    };
    auto IsNumVec3 = [&](Type const &t) {
        if (t.tag() != Type::Tag::VECTOR || t.dimension() != 3) return false;
        auto &&ele = *t.element();
        switch (ele.tag()) {
            case Type::Tag::INT:
            case Type::Tag::UINT:
            case Type::Tag::FLOAT:
                return true;
            default:
                return false;
        }
    };
    auto PrintArgs = [&] {
        uint64 sz = 0;
        for (auto &&i : args) {
            ++sz;
            i->accept(vis);
            if (sz != args.size()) {
                str << ',';
            }
        }
    };
    auto PrintArgsAccelPossible = [&] {
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
    };
    switch (expr->op()) {
        case CallOp::CUSTOM:
            str << "custom_"sv << vstd::to_string((opt->GetFuncCount(expr->custom().hash())));
            str << '(';
            PrintArgsAccelPossible();
            str << ')';
            return;

        case CallOp::ALL:
            str << "all"sv;
            break;
        case CallOp::ANY:
            str << "any"sv;
            break;
        case CallOp::SELECT: {
            auto type = args[2]->type();
            str << "selectVec"sv;
            if (type->tag() == Type::Tag::VECTOR) {
                vstd::to_string(type->dimension(), str);
            }
        } break;
        case CallOp::CLAMP:
            str << "clamp"sv;
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
            str << "clz"sv;
            break;
        case CallOp::CTZ:
            str << "ctz"sv;
            break;
        case CallOp::POPCOUNT:
            str << "popcount"sv;
            break;
        case CallOp::REVERSE:
            str << "reverse"sv;
            break;
        case CallOp::ISINF:
            str << "_isinf"sv;
            break;
        case CallOp::ISNAN:
            str << "_isnan"sv;
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
            str << "fma"sv;
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
        case CallOp::DETERMINANT:
            str << "determinant"sv;
            break;
        case CallOp::TRANSPOSE:
            str << "my_transpose"sv;
            break;
        case CallOp::INVERSE:
            str << "inverse"sv;
            break;
        case CallOp::ATOMIC_EXCHANGE: {
            str << "_atomic_exchange"sv;
            getPointer();
            return;
        }
        case CallOp::ATOMIC_COMPARE_EXCHANGE: {
            str << "_atomic_compare_exchange"sv;
            getPointer();
            return;
        }
        case CallOp::ATOMIC_FETCH_ADD: {
            str << "_atomic_add"sv;
            getPointer();
            return;
        }
        case CallOp::ATOMIC_FETCH_SUB: {
            str << "_atomic_sub"sv;
            getPointer();
            return;
        }
        case CallOp::ATOMIC_FETCH_AND: {
            str << "_atomic_and"sv;
            getPointer();
            return;
        }
        case CallOp::ATOMIC_FETCH_OR: {
            str << "_atomic_or"sv;
            getPointer();
            return;
        }
        case CallOp::ATOMIC_FETCH_XOR: {
            str << "_atomic_xor"sv;
            getPointer();
            return;
        }
        case CallOp::ATOMIC_FETCH_MIN: {
            str << "_atomic_min"sv;
            getPointer();
            return;
        }
        case CallOp::ATOMIC_FETCH_MAX: {

            str << "_atomic_max"sv;
            getPointer();
            return;
        }
        case CallOp::TEXTURE_READ:
            str << "Smptx";
            break;
        case CallOp::TEXTURE_WRITE:
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
        case CallOp::MAKE_FLOAT2X2:
        case CallOp::MAKE_FLOAT4X4: {
            if (args.size() == 1 && (args[0]->type() == expr->type())) {
                args[0]->accept(vis);
            } else {
                GenMakeFunc();
            }
            return;
        }
        case CallOp::MAKE_FLOAT3X3: {
            if (args.size() == 1 && (args[0]->type() == expr->type())) {
                args[0]->accept(vis);
                return;
            } else {
                str << "make_float3x3";
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
            str << "bfwrite"sv;
            auto elem = args[0]->type()->element();
            if (IsNumVec3(*elem)) {
                str << "Vec3"sv;
            } else if (elem->is_matrix()) {
                str << "Mat";
            }
        } break;
        case CallOp::TRACE_CLOSEST:
            str << "TraceClosest"sv;
            break;
        case CallOp::TRACE_ANY:
            str << "TraceAny"sv;
            break;
        case CallOp::BINDLESS_BUFFER_READ: {
            str << "READ_BUFFER"sv;
            if (IsNumVec3(*expr->type())) {
                str << "Vec3"sv;
            }
            auto index = opt->AddBindlessType(expr->type());
            str << '(';
            auto args = expr->arguments();
            for (auto &&i : args) {
                i->accept(vis);
                str << ',';
            }
            str << "bdls"sv
                << vstd::to_string(index)
                << ')';
            return;
        }
        case CallOp::ASSUME:
        case CallOp::UNREACHABLE: {
            return;
        }
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE:
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL:
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD:
            str << "SampleTex2D"sv;
            break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE:
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL:
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD:
            str << "SampleTex3D"sv;
            break;
        case CallOp::BINDLESS_TEXTURE2D_READ:
        case CallOp::BINDLESS_TEXTURE2D_READ_LEVEL:
            str << "ReadTex2D"sv;
            break;
        case CallOp::BINDLESS_TEXTURE3D_READ:
        case CallOp::BINDLESS_TEXTURE3D_READ_LEVEL:
            str << "ReadTex3D"sv;
            break;
        case CallOp::BINDLESS_TEXTURE2D_SIZE:
        case CallOp::BINDLESS_TEXTURE2D_SIZE_LEVEL:
            str << "Tex2DSize"sv;
            break;
        case CallOp::BINDLESS_TEXTURE3D_SIZE:
        case CallOp::BINDLESS_TEXTURE3D_SIZE_LEVEL:
            str << "Tex3DSize"sv;
            break;
        case CallOp::SYNCHRONIZE_BLOCK:
            str << "GroupMemoryBarrierWithGroupSync()"sv;
            return;
        case CallOp::INSTANCE_TO_WORLD_MATRIX: {
            str << "InstMatrix("sv;
            args[0]->accept(vis);
            str << "Inst,"sv;
            args[1]->accept(vis);
            str << ')';
            return;
        }
        case CallOp::SET_ACCEL_TRANFORM: {
            str << "SetAccelTransform("sv;
            args[0]->accept(vis);
            str << "Inst,"sv;
            for (auto i : vstd::range(1, args.size())) {
                args[i]->accept(vis);
                if (i != (args.size() - 1)) {
                    str << ',';
                }
            }
            str << ')';
            return;
        }
        case CallOp::SET_ACCEL_TRANSFORM_VISIBILITY: {
            str << "SetAccelTransformVis("sv;
            args[0]->accept(vis);
            str << "Inst,"sv;
            for (auto i : vstd::range(1, args.size())) {
                args[i]->accept(vis);
                if (i != (args.size() - 1)) {
                    str << ',';
                }
            }
            str << ')';
            return;
        }
        case CallOp::SET_ACCEL_VISIBILITY: {
            str << "SetAccelVis("sv;
            args[0]->accept(vis);
            str << "Inst,"sv;
            for (auto i : vstd::range(1, args.size())) {
                args[i]->accept(vis);
                if (i != (args.size() - 1)) {
                    str << ',';
                }
            }
            str << ')';
            return;
        }
        default: {
            auto errorType = expr->op();
            VEngine_Log("Function Not Implemented"sv);
            VSTL_ABORT();
        }
    }
    str << '(';
    PrintArgs();
    str << ')';
}
size_t CodegenUtility::GetTypeSize(Type const &t) {
    switch (t.tag()) {
        case Type::Tag::BOOL:
            return 1;
        case Type::Tag::FLOAT:
        case Type::Tag::INT:
        case Type::Tag::UINT:
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
                v = CalcAlign(v, align);
                maxAlign = std::max(align, align);
                v += GetTypeSize(*i);
            }
            v = CalcAlign(v, maxAlign);
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
        case Type::Tag::FLOAT:
        case Type::Tag::INT:
        case Type::Tag::UINT:
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
    void operator()(vstd::string &str) {
        using BasicTypeUtil = vstd::VariantVisitor_t<basic_types>;
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
    void operator()(vstd::string &str) {
        TypeNameStruct<T>()(str);
        size_t n = (t == 3) ? 4 : t;
        str += ('0' + n);
    }
};
template<size_t t>
struct TypeNameStruct<luisa::Matrix<t>> {
    void operator()(vstd::string &str) {
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
void CodegenUtility::GetBasicTypeName(uint64 typeIndex, vstd::string &str) {
    vstd::VariantVisitor_t<basic_types>()(
        [&]<typename T>() {
            TypeNameStruct<T>()(str);
        },
        typeIndex);
}
void CodegenUtility::CodegenFunction(Function func, vstd::string &result) {

    
    if (func.tag() == Function::Tag::KERNEL) {
        result << "[numthreads("
               << vstd::to_string(func.block_size().x)
               << ','
               << vstd::to_string(func.block_size().y)
               << ','
               << vstd::to_string(func.block_size().z)
               << R"()]
void main(uint3 thdId : SV_GroupThreadId, uint3 dspId : SV_DispatchThreadID, uint3 grpId : SV_GroupId){
Args a = _Global[0];
if(any(dspId >= a.dsp_c)) return;
)"sv;

    } else {
        GetFunctionDecl(func, result);
        result << "{\n"sv;
    }
    auto constants = func.constants();
    for (auto &&i : constants) {
        if (!opt->generatedConstants.TryEmplace(i.hash()).second) {
            continue;
        }
        GetTypeName(*i.type, result, Usage::READ);
        result << ' ';
        vstd::string constName;
        GetConstName(
            i.data,
            constName);

        result << constName << ";\nconst "sv;
        vstd::string constValueName(constName + "_v");
        GetTypeName(*i.type->element(), result, Usage::READ);
        result << ' ' << constValueName << '[';
        vstd::to_string(i.type->dimension(), result);
        result << "]={"sv;
        auto &&dataView = i.data.view();
        eastl::visit(
            [&]<typename T>(eastl::span<T> const &sp) {
                for (auto i : vstd::range(sp.size())) {
                    auto &&value = sp[i];
                    PrintValue<std::remove_cvref_t<T>>()(value, result);
                    if (i != (sp.size() - 1)) {
                        result << ',';
                    }
                }
            },
            dataView);
        //TODO: constants
        result << "};\n"sv
               << constName
               << ".v="sv
               << constValueName
               << ";\n"sv;
    }
    if (func.tag() == Function::Tag::KERNEL) {
        opt->isKernel = true;
        opt->arguments.Clear();
        opt->arguments.reserve(func.arguments().size());
        for (auto &&i : func.arguments()) {
            opt->arguments.Emplace(i.uid());
        }
    } else {
        opt->isKernel = false;
    }
    StringStateVisitor vis(func, result);
    vis.sharedVariables = (func.tag() == Function::Tag::KERNEL) ? &opt->sharedVariable : nullptr;
    func.body()->accept(vis);
    result << "}\n"sv;
}
void CodegenUtility::GenerateCBuffer(
    Function f,
    std::span<const Variable> vars,
    vstd::string &result) {
    result << R"(struct Args{
uint3 dsp_c;
)"sv;
    size_t alignCount = 0;
    auto isCBuffer = [&](Variable::Tag t) {
        switch (t) {
            case Variable::Tag::BUFFER:
            case Variable::Tag::TEXTURE:
            case Variable::Tag::BINDLESS_ARRAY:
            case Variable::Tag::ACCEL:
            case Variable::Tag::THREAD_ID:
            case Variable::Tag::BLOCK_ID:
            case Variable::Tag::DISPATCH_ID:
            case Variable::Tag::DISPATCH_SIZE:
                return false;
        }
        return true;
    };
    for (auto &&i : vars) {
        if (!isCBuffer(i.tag())) continue;
        GetTypeName(*i.type(), result, f.variable_usage(i.uid()));
        result << ' ';
        GetVariableName(i, result);
        result << ";\n"sv;
    }
    result << R"(};
StructuredBuffer<Args> _Global:register(t0);
)"sv;
}
vstd::optional<CodegenResult> CodegenUtility::Codegen(
    Function kernel) {
    if (kernel.tag() != Function::Tag::KERNEL) return {};
    opt = CodegenStackData::Allocate();
    auto disposeOpt = vstd::create_disposer([&] {
        CodegenStackData::DeAllocate(std::move(opt));
    });
    vstd::string codegenData;
    // Custom callable
    {
        vstd::HashMap<void const *> callableMap;
        auto callable = [&](auto &&callable, Function func) -> void {
            for (auto &&i : func.custom_callables()) {
                if (callableMap.TryEmplace(i.get()).second) {
                    Function f(i.get());
                    callable(callable, f);
                }
            }
            CodegenFunction(func, codegenData);
        };
        callable(callable, kernel);
    }
    vstd::string finalResult;
    finalResult.reserve(65500);
    //TODO: include
    finalResult << GetHeader();
    if (kernel.raytracing()) {
        finalResult << GetRayTracingHeader();
    }
    size_t unChangedOffset = finalResult.size();

    opt->isKernel = false;
    GenerateCBuffer(kernel, kernel.arguments(), finalResult);
    CodegenResult::Properties properties;
    properties.reserve(kernel.arguments().size() + opt->bindlessBufferCount + 4);
    // Bindless Buffers;
    for (auto &&i : opt->bindlessBufferTypes) {
        finalResult << "StructuredBuffer<"sv;
        if (i.first->is_matrix()) {
            auto n = i.first->dimension();
            finalResult << luisa::format("WrappedFloat{}x{}", n, n);
        } else if (i.first->is_vector() && i.first->dimension() == 3) {
            finalResult << "float4"sv;
        } else {
            GetTypeName(*i.first, finalResult, Usage::READ);
        }
        vstd::string instName("bdls"sv);
        vstd::to_string(i.second, instName);
        finalResult << "> " << instName << "[]:register(t0,space"sv;
        vstd::to_string(i.second + 3, finalResult);
        finalResult << ");\n"sv;

        properties.emplace_back(
            std::move(instName),
            Shader::Property{
                ShaderVariableType::SRVDescriptorHeap,
                static_cast<uint>(i.second + 3u),
                0u, 0u});
    }
    properties.emplace_back(
        "_Global"sv,
        Shader::Property{
            ShaderVariableType::StructuredBuffer,
            0,
            0,
            0});
    properties.emplace_back(
        "_BindlessTex"sv,
        Shader::Property{
            ShaderVariableType::SRVDescriptorHeap,
            1,
            0,
            0});
    properties.emplace_back(
        "_BindlessTex3D"sv,
        Shader::Property{
            ShaderVariableType::SRVDescriptorHeap,
            2,
            0,
            0});
    properties.emplace_back(
        "samplers"sv,
        Shader::Property{
            ShaderVariableType::SampDescriptorHeap,
            1u,
            0u,
            16u});
    enum class RegisterType : vbyte {
        CBV,
        UAV,
        SRV
    };
    uint registerCount[3] = {0, 0, 1};
    auto Writable = [&](Variable const &v) {
        return (static_cast<uint>(kernel.variable_usage(v.uid())) & static_cast<uint>(Usage::WRITE)) != 0;
    };
    for (auto &&i : kernel.arguments()) {
        auto print = [&] {
            GetTypeName(*i.type(), finalResult, kernel.variable_usage(i.uid()));
            finalResult << ' ';
            vstd::string varName;
            GetVariableName(i, varName);
            finalResult << varName;
            return varName;
        };
        auto printInstBuffer = [&]<bool writable>() {
            if constexpr (writable)
                finalResult << "RWStructuredBuffer<MeshInst> "sv;
            else
                finalResult << "StructuredBuffer<MeshInst> "sv;
            vstd::string varName;
            GetVariableName(i, varName);
            varName << "Inst"sv;
            finalResult << varName;
            return varName;
        };
        auto genArg = [&]<bool rtBuffer = false, bool writable = false>(RegisterType regisT, ShaderVariableType sT, char v) {
            auto &&r = registerCount[(vbyte)regisT];
            Shader::Property prop = {
                .type = sT,
                .spaceIndex = 0,
                .registerIndex = r,
                .arrSize = 0};
            if constexpr (rtBuffer) {
                properties.emplace_back(printInstBuffer.operator()<writable>(), prop);

            } else {
                properties.emplace_back(print(), prop);
            }
            finalResult << ":register("sv << v;
            vstd::to_string(r, finalResult);
            finalResult << ");\n"sv;
            r++;
        };

        switch (i.type()->tag()) {
            case Type::Tag::TEXTURE:
                if (Writable(i)) {
                    genArg(RegisterType::UAV, ShaderVariableType::UAVDescriptorHeap, 'u');
                } else {
                    genArg(RegisterType::SRV, ShaderVariableType::SRVDescriptorHeap, 't');
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
        }
    }
    if (!opt->customStructVector.empty()) {
        luisa::vector<const StructGenerator *> structures(
            opt->customStructVector.begin(),
            opt->customStructVector.end());
        std::sort(structures.begin(), structures.end(), [](auto lhs, auto rhs) noexcept {
            return lhs->GetType()->index() < rhs->GetType()->index();
        });
        structures.erase(
            std::unique(structures.begin(), structures.end(), [](auto lhs, auto rhs) noexcept {
                return lhs->GetType()->hash() == rhs->GetType()->hash();
            }),
            structures.end());
        for (auto v : structures) {
            finalResult << "struct " << v->GetStructName() << "{\n"
                        << v->GetStructDesc() << "};\n";
        }
    }
    for (auto&& i : opt->sharedVariable) {
        finalResult << "groupshared "sv;
        GetTypeName(*i.type()->element(), finalResult, Usage::READ);
        finalResult << ' ';
        GetVariableName(i, finalResult);
        finalResult << '[';
        vstd::to_string(i.type()->dimension(), finalResult);
        finalResult << "];\n"sv;
    }
    
    finalResult << codegenData;
    return {
        std::move(finalResult),
        std::move(properties),
        opt->bindlessBufferCount, 
        vstd::MD5(vstd::string_view(finalResult.c_str() + unChangedOffset, finalResult.size() - unChangedOffset))};
}
}// namespace toolhub::directx