
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
void StructGenerator::ProvideAlignVariable(size_t tarAlign, size_t &structSize, size_t &alignCount, vstd::string &structDesc) {
    auto leftedValue = tarAlign - (structSize % tarAlign);
    if (leftedValue == tarAlign) {
        leftedValue = 0;
    }
    if (leftedValue == 0) return;
    structSize += leftedValue;
    switch (leftedValue) {
        case 1:
            structDesc << "uint _aa";
            vstd::to_string(alignCount, structDesc);
            alignCount++;
            structDesc << ":8;\n";
            break;
        case 2:
            structDesc << "uint _aa";
            vstd::to_string(alignCount, structDesc);
            alignCount++;
            structDesc << ":16;\n";
            break;
        case 3:
            structDesc << "uint _aa";
            vstd::to_string(alignCount, structDesc);
            alignCount++;
            structDesc << ":8;\nuint _aa";
            vstd::to_string(alignCount, structDesc);
            alignCount++;
            structDesc << ":16;\n";
            break;
        case 4:
            structDesc << "uint _aa";
            vstd::to_string(alignCount, structDesc);
            alignCount++;
            structDesc << ";\n";
            break;
        case 8:
            structDesc << "uint2 _aa";
            vstd::to_string(alignCount, structDesc);
            alignCount++;
            structDesc << ";\n";
            break;
        case 12:
            structDesc << "uint3 _aa";
            vstd::to_string(alignCount, structDesc);
            alignCount++;
            structDesc << ";\n";
            break;
        case 16:
            structDesc << "uint4 _aa";
            vstd::to_string(alignCount, structDesc);
            alignCount++;
            structDesc << ";\n";
            break;
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
    size_t structSize = 0;
    structName = "S";
    vstd::to_string(structIdx, structName);
    structDesc.reserve(1024);
    auto szOpt = vars.Get()->Length();
    vstd::string varName;

    auto updateVarName = [&] {
        varName.clear();
        varName << 'v';
        vstd::to_string(structTypes.size(), varName);
    };
    size_t maxAlign = 4;
    auto Align = [&](size_t tarAlign) {
        maxAlign = std::max(maxAlign, tarAlign);
        ProvideAlignVariable(tarAlign, structSize, alignCount, structDesc);
    };

    for (; vars; vars++) {
        auto &&i = *vars;
        updateVarName();
        vstd::variant<StructureType, StructGenerator *> ele;
        switch (i->tag()) {
            case Type::Tag::BOOL:
                structSize += 1;
                ele = StructureType::GetScalar();
                break;
            case Type::Tag::FLOAT:
                Align(4);
                structSize += 4;
                ele = StructureType::GetScalar();
                break;
            case Type::Tag::INT:
                Align(4);
                structSize += 4;
                ele = StructureType::GetScalar();
                break;
            case Type::Tag::UINT:
                Align(4);
                structSize += 4;
                ele = StructureType::GetScalar();
                break;
            case Type::Tag::VECTOR:
                Align(i->alignment());
                structSize += 4 * i->dimension();
                ele = StructureType::GetVector(i->dimension());
                break;
            case Type::Tag::MATRIX: {
                Align(i->alignment());
                auto alignDim = i->dimension();
                alignDim = (alignDim == 3) ? 4 : alignDim;
                structSize += 4 * alignDim * i->dimension();
                ele = StructureType::GetMatrix(i->dimension());
            } break;
            case Type::Tag::STRUCTURE: {
                Align(i->alignment());
                auto subStruct = visitor(i);
                structSize += i->size();
                ele = subStruct;
            } break;
            case Type::Tag::ARRAY: {
                auto subStruct = visitor(i);
                Align(i->element()->alignment());
                structSize += i->size();
                ele = subStruct;
            } break;
        }
        CodegenUtility::GetTypeName(*i, structDesc, Usage::READ);
        structDesc << ' ' << varName;
        if (i->tag() == Type::Tag::BOOL) {
            structDesc << ":8"sv;
        }
        structDesc << ";\n"sv;
        Align(i->alignment());
        structTypes.emplace_back(
            std::move(varName),
            ele);
    }

    updateVarName();
    Align(maxAlign);
}
void StructGenerator::InitAsArray(
    Type const *t,
    size_t structIdx,
    vstd::function<StructGenerator *(Type const *)> const &visitor) {
    structName = "A";
    vstd::to_string(structIdx, structName);
    auto &&ele = t->element();
    CodegenUtility::GetTypeName(*ele, structDesc, Usage::READ);
    structDesc << " v["sv << vstd::to_string(t->dimension()) << "];\n";
}
StructGenerator::StructGenerator(
    Type const *structureType,
    size_t structIdx,
    vstd::function<StructGenerator *(Type const *)> const &visitor)
    : idx(structIdx),
      structureType{structureType} {
    if (structureType->tag() == Type::Tag::STRUCTURE) {
        InitAsStruct(vstd::GetIterator(structureType->members()), structIdx, visitor);
    } else {
        InitAsArray(structureType, structIdx, visitor);
    }
}
StructGenerator::~StructGenerator() = default;
}// namespace toolhub::directx