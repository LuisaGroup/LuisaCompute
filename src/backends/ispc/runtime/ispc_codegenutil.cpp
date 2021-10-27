#include <backends/ispc/runtime/ispc_codegen.h>

namespace lc::ispc {
#include "ispc.inl"
static thread_local vstd::optional<vstd::HashMap<Type const *, void>> codegenStructType;
void CodegenUtility::ClearStructType() {
    codegenStructType.New();
    codegenStructType->Clear();
}
void CodegenUtility::RegistStructType(Type const *type) {
    if (type->is_structure())
        codegenStructType->Emplace(type);
    else if (type->is_buffer()) {
        RegistStructType(type->element());
    }
}

static bool IsVarWritable(Function func, Variable i) {
    return ((uint)func.variable_usage(i.uid()) & (uint)Usage::WRITE) != 0;
}
void CodegenUtility::GetVariableName(Variable::Tag type, uint id, std::string &str) {
    switch (type) {
        case Variable::Tag::BLOCK_ID:
            str += "blk_id"sv;
            break;
        case Variable::Tag::DISPATCH_ID:
            str += "dsp_id"sv;
            break;
        case Variable::Tag::THREAD_ID:
            str += "thd_id"sv;
            break;
        case Variable::Tag::LOCAL:
            str += "_v"sv;
            vstd::to_string(id, str);
            break;
        case Variable::Tag::BUFFER:
            str += "_b"sv;
            vstd::to_string(id, str);
            break;
        case Variable::Tag::TEXTURE:
            str += "_t"sv;
            vstd::to_string(id, str);
            break;
        default:
            str += 'v';
            vstd::to_string(id, str);
            break;
    }
}

void CodegenUtility::GetVariableName(Type::Tag type, uint id, std::string &str) {
    switch (type) {
        case Type::Tag::BUFFER:
            str += "_b"sv;
            vstd::to_string(id, str);
            break;
        case Type::Tag::TEXTURE:
            str += "_t"sv;
            vstd::to_string(id, str);
            break;
        default:
            str += 'v';
            vstd::to_string(id, str);
            break;
    }
}

void CodegenUtility::GetVariableName(Variable const &type, std::string &str) {
    GetVariableName(type.tag(), type.uid(), str);
}

void CodegenUtility::GetTypeName(Type const &type, std::string &str, bool isWritable) {
    switch (type.tag()) {
        case Type::Tag::ARRAY:
            CodegenUtility::GetTypeName(*type.element(), str, isWritable);
            return;
            //		case Type::Tag::ATOMIC:
            //			CodegenUtility::GetTypeName(*type.element(), str, isWritable);
            //			return;
        case Type::Tag::BOOL:
            str += "bool"sv;
            return;
        case Type::Tag::FLOAT:
            str += "float"sv;
            return;
        case Type::Tag::INT:
            str += "int"sv;
            return;
        case Type::Tag::UINT:
            str += "uint"sv;
            return;
        case Type::Tag::MATRIX: {
            auto dim = std::to_string(type.dimension());
            CodegenUtility::GetTypeName(*type.element(), str, isWritable);
            str += dim;
            str += 'x';
            str += dim;
        }
            return;
        case Type::Tag::VECTOR: {
            CodegenUtility::GetTypeName(*type.element(), str, isWritable);
            vstd::to_string(static_cast<uint64_t>(type.dimension()), str);
        }
            return;
        case Type::Tag::STRUCTURE:
            str += 'T';
            vstd::to_string(type.hash(), str);
            return;
        case Type::Tag::BUFFER:
            if (isWritable) {
                str += "RWStructuredBuffer<"sv;
            } else {
                str += "StructuredBuffer<"sv;
            }
            GetTypeName(*type.element(), str, isWritable);
            str += '>';
            break;
        case Type::Tag::TEXTURE: {
            if (isWritable) {
                str += "RWTexture"sv;
            } else {
                str += "Texture"sv;
            }
            vstd::to_string(static_cast<uint64_t>(type.dimension()), str);
            str += "D<"sv;
            GetTypeName(*type.element(), str, isWritable);
            if (type.tag() != Type::Tag::VECTOR) {
                str += '4';
            }
            str += '>';
            break;
        }
        default:
            LUISA_ERROR_WITH_LOCATION("Bad.");
            break;
    }
}

void CodegenUtility::GetFunctionDecl(Function func, std::string &data) {

    if (func.return_type()) {
        CodegenUtility::GetTypeName(*func.return_type(), data);
    } else {
        data += "void"sv;
    }
    switch (func.tag()) {
        case Function::Tag::CALLABLE: {
            data += " custom_"sv;
            vstd::to_string(func.hash(), data);
            if (func.arguments().empty()) {
                data += "()"sv;
            } else {
                data += '(';
                for (auto &&i : func.arguments()) {
                    RegistStructType(i.type());
                    CodegenUtility::GetTypeName(*i.type(), data, IsVarWritable(func, i));
                    data += ' ';
                    CodegenUtility::GetVariableName(i, data);
                    data += ',';
                }
                data[data.size() - 1] = ')';
            }
        } break;
        default:
            //TODO
            break;
    }
}
void CodegenUtility::GetFunctionName(CallExpr const *expr, std::string &result) {
    auto IsType = [](Type const *const type, Type::Tag const tag, uint const vecEle) {
        if (type->tag() == Type::Tag::VECTOR) {
            if (vecEle > 1) {
                return type->element()->tag() == tag && type->dimension() == vecEle;
            } else {
                return type->tag() == tag;
            }
        } else {
            return vecEle == 1;
        }
    };
    switch (expr->op()) {
        case CallOp::CUSTOM:
            result << "custom_"sv << vstd::to_string(expr->custom().hash());
            break;
        case CallOp::ALL:
            result << "all"sv;
            break;
        case CallOp::ANY:
            result << "any"sv;
            break;
        case CallOp::NONE:
            result << "!any"sv;
            break;
        case CallOp::SELECT: {
            result << "select"sv;
        } break;
        case CallOp::CLAMP:
            result << "clamp"sv;
            break;
        case CallOp::LERP:
            result << "lerp"sv;
            break;
        case CallOp::SATURATE:
            result << "saturate"sv;
            break;
        case CallOp::SIGN:
            result << "sign"sv;
            break;
        case CallOp::STEP:
            result << "step"sv;
            break;
        case CallOp::SMOOTHSTEP:
            result << "smoothstep"sv;
            break;
        case CallOp::ABS:
            result << "abs"sv;
            break;
        case CallOp::MIN:
            result << "min"sv;
            break;
        case CallOp::POW:
            result << "pow"sv;
            break;
        case CallOp::CLZ:
            result << "clz"sv;
            break;
        case CallOp::CTZ:
            result << "ctz"sv;
            break;
        case CallOp::POPCOUNT:
            result << "popcount"sv;
            break;
        case CallOp::REVERSE:
            result << "reverse"sv;
            break;
        case CallOp::ISINF:
            result << "isinf"sv;
            break;
        case CallOp::ISNAN:
            result << "isnan"sv;
            break;
        case CallOp::ACOS:
            result << "acos"sv;
            break;
        case CallOp::ACOSH:
            result << "acosh"sv;
            break;
        case CallOp::ASIN:
            result << "asin"sv;
            break;
        case CallOp::ASINH:
            result << "asinh"sv;
            break;
        case CallOp::ATAN:
            result << "atan"sv;
            break;
        case CallOp::ATAN2:
            result << "atan2"sv;
            break;
        case CallOp::ATANH:
            result << "atanh"sv;
            break;
        case CallOp::COS:
            result << "cos"sv;
            break;
        case CallOp::COSH:
            result << "cosh"sv;
            break;
        case CallOp::SIN:
            result << "sin"sv;
            break;
        case CallOp::SINH:
            result << "sinh"sv;
            break;
        case CallOp::TAN:
            result << "tan"sv;
            break;
        case CallOp::TANH:
            result << "tanh"sv;
            break;
        case CallOp::EXP:
            result << "exp"sv;
            break;
        case CallOp::EXP2:
            result << "exp2"sv;
            break;
        case CallOp::EXP10:
            result << "exp10"sv;
            break;
        case CallOp::LOG:
            result << "log"sv;
            break;
        case CallOp::LOG2:
            result << "log2"sv;
            break;
        case CallOp::LOG10:
            result << "log10"sv;
            break;
        case CallOp::SQRT:
            result << "sqrt"sv;
            break;
        case CallOp::RSQRT:
            result << "rsqrt"sv;
            break;
        case CallOp::CEIL:
            result << "ceil"sv;
            break;
        case CallOp::FLOOR:
            result << "floor"sv;
            break;
        case CallOp::FRACT:
            result << "fract"sv;
            break;
        case CallOp::TRUNC:
            result << "trunc"sv;
            break;
        case CallOp::ROUND:
            result << "round"sv;
            break;
        case CallOp::DEGREES:
            result << "degrees"sv;
            break;
        case CallOp::RADIANS:
            result << "radians"sv;
            break;
        case CallOp::FMA:
            result << "fma"sv;
            break;
        case CallOp::COPYSIGN:
            result << "copysign"sv;
            break;
        case CallOp::CROSS:
            result << "cross"sv;
            break;
        case CallOp::DOT:
            result << "dot"sv;
            break;
        case CallOp::DISTANCE:
            result << "distance"sv;
            break;
        case CallOp::DISTANCE_SQUARED:
            result << "distance_sqr"sv;
            break;
        case CallOp::LENGTH:
            result << "length"sv;
            break;
        case CallOp::LENGTH_SQUARED:
            result << "length_sqr"sv;
            break;
        case CallOp::NORMALIZE:
            result << "normalize"sv;
            break;
        case CallOp::FACEFORWARD:
            result << "faceforward"sv;
            break;
        case CallOp::DETERMINANT:
            result << "determinant"sv;
            break;
        case CallOp::TRANSPOSE:
            result << "transpose"sv;
            break;
        case CallOp::INVERSE:
            result << "inverse"sv;
            break;
        //TODO
        case CallOp::BLOCK_BARRIER:
            result << "memory_barrier"sv;
            break;
        case CallOp::DEVICE_BARRIER:
            result << "memory_barrier"sv;
            break;
        case CallOp::ALL_BARRIER:
            result << "memory_barrier"sv;
            break;
        case CallOp::ATOMIC_LOAD:
            break;
        case CallOp::ATOMIC_STORE:
            break;
        case CallOp::ATOMIC_EXCHANGE:
            break;
        case CallOp::ATOMIC_COMPARE_EXCHANGE:
            break;
        case CallOp::ATOMIC_FETCH_ADD:
            break;
        case CallOp::ATOMIC_FETCH_SUB:
            break;
        case CallOp::ATOMIC_FETCH_AND:
            break;
        case CallOp::ATOMIC_FETCH_OR:
            break;
        case CallOp::ATOMIC_FETCH_XOR:
            break;
        case CallOp::ATOMIC_FETCH_MIN:
            break;
        case CallOp::ATOMIC_FETCH_MAX:
            break;
        case CallOp::TEXTURE_READ:
            break;
        case CallOp::TEXTURE_WRITE:
            break;
        case CallOp::MAKE_BOOL2:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::BOOL, 2))
                result << "make_bool2"sv;

            break;
        case CallOp::MAKE_BOOL3:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::BOOL, 3))
                result << "make_bool3"sv;

            break;
        case CallOp::MAKE_BOOL4:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::BOOL, 4))
                result << "make_bool4"sv;

            break;
        case CallOp::MAKE_UINT2:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::UINT, 2))
                result << "make_uint2"sv;

            break;
        case CallOp::MAKE_UINT3:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::UINT, 3))
                result << "make_uint3"sv;
            break;
        case CallOp::MAKE_UINT4:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::UINT, 4))
                result << "make_uint4"sv;

            break;
        case CallOp::MAKE_INT2:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::INT, 2))
                result << "make_int2"sv;

            break;
        case CallOp::MAKE_INT3:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::INT, 3))
                result << "make_int3"sv;

            break;
        case CallOp::MAKE_INT4:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::INT, 4))
                result << "make_int4"sv;

            break;
        case CallOp::MAKE_FLOAT2:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::FLOAT, 2))
                result << "make_float2"sv;

            break;
        case CallOp::MAKE_FLOAT3:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::FLOAT, 3))
                result << "make_float3"sv;

            break;
        case CallOp::MAKE_FLOAT4:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::FLOAT, 4))
                result << "make_float4"sv;

            break;
        default:
            VEngine_Log("Function Not Implemented"sv);
            VSTL_ABORT();
    }
}
void CodegenUtility::PrintFunction(Function func, std::string &str) {
    str << headerName;
    //arguments
    size_t ofst = 0;
    for (auto &&i : func.arguments()) {
        std::string argName;
        std::string argType;
        GetVariableName(i, argName);
        GetTypeName(*i.type(), argType);
        str << argType << ' ' << argName << '=' << "*((" << argType << "*)(arg";
        if (ofst > 0) {
            str << '+';
            vstd::to_string(static_cast<uint64_t>(ofst), str);
            str << "ull";
        }
        str << "));\n";
        ofst += 8;
    }
    //foreach
    str << foreachName << "{\n"
        << "uint3 dsp_id={x,y,z};"
        << "uint3 thd_id={x,y,z};"
        << "uint3 blk_id={0,0,0};";
    StringStateVisitor vis(str);
    func.body()->accept(vis);
    //end
    str << "}}";
}
void CodegenUtility::GetBasicTypeName(size_t typeIndex, std::string &str) {
    // Matrix
    if (typeIndex > 15) {

    } else {
        auto vec = typeIndex / 4 + 1;
        auto typeCount = typeIndex & 3;
        switch (typeCount) {
            case 0:
                str << "bool";
                break;
            case 1:
                str << "float";
                break;
            case 2:
                str << "int";
                break;
            case 3:
                str << "uint";
                break;
            default:
                break;
        }
        // vector
        if (vec > 1) {
            vstd::to_string(static_cast<uint64_t>(vec), str);
        }
    }
}

}// namespace lc::ispc