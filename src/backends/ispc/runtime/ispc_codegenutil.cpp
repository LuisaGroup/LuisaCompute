#pragma vengine_package ispc_vsproject

#include <backends/ispc/runtime/ispc_codegen.h>
#include <vstl/StringUtility.h>
namespace lc::ispc {
struct CodegenGlobal {
    vstd::HashMap<Type const *, size_t> structTypes;
    vstd::HashMap<uint64, size_t> constTypes;
    vstd::HashMap<uint64, size_t> funcTypes;
    size_t count = 0;
    size_t constCount = 0;
    size_t funcCount = 0;
    void Clear() {
        structTypes.Clear();
        constTypes.Clear();
        funcTypes.Clear();
        constCount = 0;
        count = 0;
        funcCount = 0;
    }
    size_t GetConstCount(uint64 data) {
        auto ite = constTypes.Emplace(
            data,
            vstd::MakeLazyEval(
                [&] {
                    return constCount++;
                }));
        return ite.Value();
    }
    size_t GetFuncCount(uint64 data) {
        auto ite = funcTypes.Emplace(
            data,
            vstd::MakeLazyEval(
                [&] {
                    return funcCount++;
                }));
        return ite.Value();
    }
    size_t GetTypeCount(Type const *t) {
        auto ite = structTypes.Emplace(
            t,
            vstd::MakeLazyEval(
                [&] {
                    return count++;
                }));
        return ite.Value();
    }
};
static thread_local vstd::optional<CodegenGlobal> opt;
#include "ispc.inl"
void CodegenUtility::ClearStructType() {
    opt.New();
    opt->Clear();
}
void CodegenUtility::RegistStructType(Type const *type) {
    if (type->is_structure())
        opt->structTypes.Emplace(type, opt->count++);
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
void CodegenUtility::GetConstName(ConstantData const &data, std::string &str) {
    size_t constCount = opt->GetConstCount(data.hash());
    str << "c";
    vstd::to_string(static_cast<uint64_t>(constCount), str);
}
void CodegenUtility::GetConstantStruct(ConstantData const &data, std::string &str) {
    size_t constCount = opt->GetConstCount(data.hash());
    //auto typeName = CodegenUtility::GetBasicTypeName(view.index());
    str << "struct tc";
    vstd::to_string(static_cast<uint64_t>(constCount), str);
    size_t varCount = 1;
    std::visit(
        [&](auto &&arr) {
            varCount = arr.size();
        },
        data.view());
    str << "{\n";
    str << CodegenUtility::GetBasicTypeName(data.view().index()) << " v[";
    vstd::to_string(static_cast<uint64_t>(varCount), str);
    str << "];\n";
    str << "};\n";
}
void CodegenUtility::GetConstantData(ConstantData const &data, std::string &str) {
    auto &&view = data.view();
    size_t constCount = opt->GetConstCount(data.hash());

    std::string name = vstd::to_string(static_cast<uint64_t>(constCount));
    str << "uniform const tc" << name << " c" << name;
    str << "={{";
    std::visit(
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

void CodegenUtility::GetTypeName(Type const &type, std::string &str) {
    switch (type.tag()) {
        case Type::Tag::ARRAY:
            CodegenUtility::GetTypeName(*type.element(), str);
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
            CodegenUtility::GetTypeName(*type.element(), str);
            str += dim;
            str += 'x';
            str += dim;
        }
            return;
        case Type::Tag::VECTOR: {
            CodegenUtility::GetTypeName(*type.element(), str);
            vstd::to_string(static_cast<uint64_t>(type.dimension()), str);
        }
            return;
        case Type::Tag::STRUCTURE:
            str += 'T';
            vstd::to_string(static_cast<uint64_t>(opt->GetTypeCount(&type)), str);
            return;
        case Type::Tag::BUFFER:

            GetTypeName(*type.element(), str);
            str << '*';
            break;
        case Type::Tag::TEXTURE: {
            str += "Texture"sv;

            vstd::to_string(static_cast<uint64_t>(type.dimension()), str);
            str += "D<"sv;
            GetTypeName(*type.element(), str);
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
            vstd::to_string(static_cast<uint64_t>(opt->GetFuncCount(func.hash())), data);
            if (func.arguments().empty()) {
                data += "()"sv;
            } else {
                data += '(';
                for (auto &&i : func.arguments()) {
                    RegistStructType(i.type());
                    CodegenUtility::GetTypeName(*i.type(), data);
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
vstd::function<void(StringExprVisitor &)> CodegenUtility::GetFunctionName(CallExpr const *expr, std::string &str) {
    auto defaultArgs = [&str, expr](StringExprVisitor &vis) {
        str << '(';
        size_t sz = 0;
        auto args = expr->arguments();
        for (auto &&i : args) {
            ++sz;
            i->accept(vis);
            if (sz != args.size()) {
                str << ',';
            }
        }
    };
    auto getPointer = [&str, expr](StringExprVisitor &vis) {
        str << '(';
        size_t sz = 1;
        auto args = expr->arguments();
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
            str << "custom_"sv << vstd::to_string(static_cast<uint64_t>(opt->GetFuncCount(expr->custom().hash())));
            break;
        case CallOp::ALL:
            str << "all"sv;
            break;
        case CallOp::ANY:
            str << "any"sv;
            break;
        case CallOp::SELECT: {
            str << "select"sv;
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
            str << "isinf"sv;
            break;
        case CallOp::ISNAN:
            str << "isnan"sv;
            break;
        case CallOp::ACOS:
            str << "acos"sv;
            break;
        case CallOp::ACOSH:
            str << "acosh"sv;
            break;
        case CallOp::ASIN:
            str << "asin"sv;
            break;
        case CallOp::ASINH:
            str << "asinh"sv;
            break;
        case CallOp::ATAN:
            str << "atan"sv;
            break;
        case CallOp::ATAN2:
            str << "atan2"sv;
            break;
        case CallOp::ATANH:
            str << "atanh"sv;
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
            str << "exp10"sv;
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
            str << "length_sqr"sv;
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
            str << "transpose"sv;
            break;
        case CallOp::INVERSE:
            str << "inverse"sv;
            break;
        case CallOp::ATOMIC_EXCHANGE:
            str << "_atomic_exchange"sv;
            return getPointer;
        case CallOp::ATOMIC_COMPARE_EXCHANGE:
            str << "_atomic_compare_exchange"sv;
            return getPointer;
        case CallOp::ATOMIC_FETCH_ADD:
            str << "_atomic_add"sv;
            return getPointer;
        case CallOp::ATOMIC_FETCH_SUB:
            str << "_atomic_sub"sv;
            return getPointer;
        case CallOp::ATOMIC_FETCH_AND:
            str << "_atomic_and"sv;
            return getPointer;
        case CallOp::ATOMIC_FETCH_OR:
            str << "_atomic_or"sv;
            return getPointer;
        case CallOp::ATOMIC_FETCH_XOR:
            str << "_atomic_xor"sv;
            return getPointer;
        case CallOp::ATOMIC_FETCH_MIN:
            str << "_atomic_min"sv;
            return getPointer;
        case CallOp::ATOMIC_FETCH_MAX:
            str << "_atomic_max"sv;
            return getPointer;
        case CallOp::TEXTURE_READ:
            str << "Smptx";
            break;
        case CallOp::TEXTURE_WRITE:
            str << "Writetx";
            break;
        case CallOp::MAKE_BOOL2:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::BOOL, 2))
                str << "_bool2"sv;

            break;
        case CallOp::MAKE_BOOL3:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::BOOL, 3))
                str << "_bool3"sv;

            break;
        case CallOp::MAKE_BOOL4:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::BOOL, 4))
                str << "_bool4"sv;

            break;
        case CallOp::MAKE_UINT2:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::UINT, 2))
                str << "_uint2"sv;

            break;
        case CallOp::MAKE_UINT3:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::UINT, 3))
                str << "_uint3"sv;
            break;
        case CallOp::MAKE_UINT4:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::UINT, 4))
                str << "_uint4"sv;

            break;
        case CallOp::MAKE_INT2:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::INT, 2))
                str << "_int2"sv;

            break;
        case CallOp::MAKE_INT3:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::INT, 3))
                str << "_int3"sv;

            break;
        case CallOp::MAKE_INT4:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::INT, 4))
                str << "_int4"sv;

            break;
        case CallOp::MAKE_FLOAT2:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::FLOAT, 2))
                str << "_float2"sv;

            break;
        case CallOp::MAKE_FLOAT3:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::FLOAT, 3))
                str << "_float3"sv;

            break;
        case CallOp::MAKE_FLOAT4:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::FLOAT, 4))
                str << "_float4"sv;

            break;
        default:
            VEngine_Log("Function Not Implemented"sv);
            VSTL_ABORT();
    }
    return defaultArgs;
}
void CodegenUtility::PrintFunction(Function func, std::string &str) {
    auto CurryArr = [&](auto&& f) {
        return [&] {
            auto consts = func.constants();
            for (auto &&c : consts) {
                f(c.data, str);
            }
        };
    };
    auto ConstStruct = CurryArr(GetConstantStruct);
    auto ConstData = CurryArr(GetConstantData);
    if (func.tag() == Function::Tag::KERNEL) {
        ClearStructType();
        str << "#include \"lib.h\"\n";
        ConstStruct();
        ConstData();
        auto callables = func.custom_callables();
        for (auto &&i : callables) {
            PrintFunction(Function(i.get()), str);
        }
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
        str << foreachName << R"(
uint3 dsp_id={x,y,z};
uint3 thd_id={x,y,z};
uint3 blk_id={0,0,0};
)"sv;
        StringStateVisitor vis(str);
        func.body()->accept(vis);
        //end
        str << "}";
    } else {
        GetTypeName(*func.return_type(), str);
        str << " f";
        vstd::to_string(func.hash(), str);
        auto args = func.arguments();
        if (args.empty()) {
            str << "()";
        } else {
            str << '(';
            for (auto &&i : args) {
                GetTypeName(*i.type(), str);
                str << ' ';
                GetVariableName(i, str);
                str << ',';
            }
            *(str.end() - 1) = ')';
        }
        StringStateVisitor vis(str);
        func.body()->accept(vis);

    }
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