#pragma vengine_package vengine_directx
#include <Codegen/StructGenerator.h>
#include <Codegen/DxCodegen.h>
#include <vstl/CommonIterator.h>
namespace toolhub::directx {
size_t StructureType::size() const {
    switch (mTag) {
        case Tag::Scalar:
            return 4;
        case Tag::Vector:
            return 4 * mDimension;
        case Tag::Matrix:
            return 4 * (mDimension == 3 ? 4 : mDimension) * mDimension;
    }
    return 0;
}
size_t StructureType::align() const {
    switch (mTag) {
        case Tag::Scalar:
            return 4;
        case Tag::Matrix:
        case Tag::Vector: {
            auto v = {4,
                      8,
                      16,
                      16};
            return v.begin()[0];
        }
    }
}
StructureType StructureType::GetScalar() {
    return {Tag::Scalar, vbyte(0)};
}
StructureType StructureType::GetVector(vbyte dim) {
    return {Tag::Vector, dim};
}
StructureType StructureType::GetMatrix(vbyte dim) {
    return {Tag::Matrix, dim};
}
void StructGenerator::InitAsStruct(
    vstd::Iterator<Type const *const> const &vars,
    size_t structIdx,
    vstd::function<StructGenerator *(Type const *)> const &visitor) {
    structName = "S";
    vstd::to_string(structIdx, structName);
    vbyte boolCount = 0;
    uint alignCount = 0;
    structDesc.reserve(1024);
    auto szOpt = vars.Get()->Length();
    if (szOpt)
        structVars.reserve(*szOpt);
    vstd::string varName;
    auto clearBool = [&] {
        if (boolCount != 0) {
            boolCount = 0;
            structDesc << "uint "sv << varName << ";\n"sv;
            structSize += 4;
            structTypes.emplace_back(StructureType::GetScalar());
        }
    };
    auto updateVarName = [&] {
        varName.clear();
        varName << 'v';
        vstd::to_string(structTypes.size(), varName);
    };
    auto Align = [&](size_t tarAlign) {
        auto leftedValue = tarAlign - (structSize % tarAlign);
        switch (leftedValue) {
            case 4:
                structDesc << "float _aa";
                vstd::to_string(alignCount, structDesc);
                alignCount++;
                break;
            case 8:
                structDesc << "float2 _aa";
                vstd::to_string(alignCount, structDesc);
                alignCount++;
                break;
            case 12:
                structDesc << "float3_aa";
                vstd::to_string(alignCount, structDesc);
                alignCount++;
                break;
            case 16:
                structDesc << "float4_aa";
                vstd::to_string(alignCount, structDesc);
                alignCount++;
                break;
        }
    };
    for (; vars; vars++) {
        auto &&i = *vars;
        if (auto vecDim = CodegenUtility::IsBool(*i)) {
            if (boolCount + vecDim >= 4) {
                updateVarName();
                clearBool();
            }
            updateVarName();
            structVars.emplace_back(
                std::move(varName),
                vecDim,
                boolCount);
            boolCount += vecDim;
        } else {
            updateVarName();
            clearBool();
            switch (i->tag()) {
                case Type::Tag::VECTOR:
                case Type::Tag::MATRIX: {
                    switch (i->dimension()) {
                        case 2:
                            Align(8);
                            break;
                        case 3:
                        case 4:
                            Align(16);
                            break;
                    }
                } break;
            }
            vstd::variant<StructureType, StructGenerator *> ele;
            switch (i->tag()) {
                case Type::Tag::FLOAT:
                    structSize += 4;
                    ele = StructureType::GetScalar();
                    break;
                case Type::Tag::INT:
                    structSize += 4;
                    ele = StructureType::GetScalar();
                    break;
                case Type::Tag::UINT:
                    structSize += 4;
                    ele = StructureType::GetScalar();
                    break;
                case Type::Tag::VECTOR:
                    structSize += 4 * i->dimension();
                    ele = StructureType::GetVector(i->dimension());
                    break;
                case Type::Tag::MATRIX: {
                    auto alignDim = i->dimension();
                    alignDim = (alignDim == 3) ? 4 : alignDim;
                    structSize += 4 * alignDim * i->dimension();
                    ele = StructureType::GetMatrix(i->dimension());
                } break;
                case Type::Tag::ARRAY:
                case Type::Tag::STRUCTURE: {
                    auto subStruct = visitor(i);
                    structSize += subStruct->GetStructSize();
                    ele = subStruct;
                } break;
            }
            CodegenUtility::GetTypeName(*i, structDesc);
            structDesc << ' ' << varName << ";\n"sv;
            structTypes.emplace_back(ele);
            structVars.emplace_back(
                std::move(varName));
        }
    }
    updateVarName();
    clearBool();
    structDesc << "};\n"sv;
}
void StructGenerator::InitAsArray(
    Type const *t,
    size_t structIdx,
    vstd::function<StructGenerator *(Type const *)> const &visitor) {
    structName = "A";
    vstd::to_string(structIdx, structName);
    auto &&ele = t->element();
    auto GetSize = [](auto &&GetSize, Type const *type) -> size_t {
        switch (type->tag()) {
            case Type::Tag::FLOAT:
            case Type::Tag::UINT:
            case Type::Tag::INT:
                return 4;
            case Type::Tag::VECTOR:
                return type->dimension() * GetSize(GetSize, type);
            case Type::Tag::MATRIX:
                return 4 * ((type->dimension() == 3) ? 4 : type->dimension()) * type->dimension();
        }
        return 0;
    };
    CodegenUtility::GetTypeName(*ele, structDesc);
    structDesc << " c["sv << vstd::to_string(t->dimension()) << "];\n";
}
StructGenerator::StructGenerator(
    Type const *structureType,
    size_t structIdx,
    vstd::function<StructGenerator *(Type const *)> const &visitor)
    : idx(structIdx) {
    if (structureType->tag() == Type::Tag::STRUCTURE) {
        InitAsStruct(vstd::GetIterator(structureType->members()), structIdx, visitor);
    } else {
        InitAsArray(structureType, structIdx, visitor);
    }
}
StructGenerator::StructGenerator(
    vstd::Iterator<Type const *const> const &vars,
    size_t structIdx,
    vstd::function<StructGenerator *(Type const *)> const &visitor)
    : idx(structIdx) {
    InitAsStruct(vars, structIdx, visitor);
}
void StructGenerator::GetMemberExpr(
    uint memberIndex,
    Type const *type,
    vstd::function<void()> const &printMember,
    vstd::string &str) {
    auto &&vars = structVars[memberIndex];
    if (vars.boolOffset == StructVariable::OFFSET_NPOS) {
        printMember();
        str << '.' << vars.name;
    } else {
        char const *ptr = "xyzw";
        str << "GetBool("sv;
        printMember();
        str << '.' << vars.name
            << ')' << vstd::string_view(ptr, vars.boolVecDim);
    }
}
void StructGenerator::SetMemberExpr(
    uint memberIndex,
    Type const *type,
    vstd::function<void()> const &printMember,
    vstd::function<void()> const &printExpr,
    vstd::string &str) {
    auto &&vars = structVars[memberIndex];
    if (vars.boolOffset == StructVariable::OFFSET_NPOS) {
        printMember();
        str << '.' << vars.name;
    } else {
        printMember();
        str << '.' << vars.name
            << "=SetBool("sv;
        printExpr();
        str << ')';
    }
}
StructGenerator::~StructGenerator() {
}
}// namespace toolhub::directx