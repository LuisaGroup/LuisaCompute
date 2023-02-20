
#include "codegen_stack_data.h"
#include "struct_generator.h"
#include "dx_codegen.h"
#include "vstl/common_iterator.h"
namespace toolhub::directx {
/*
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
}*/
void StructGenerator::ProvideAlignVariable(size_t tarAlign, size_t &structSize,
                                           size_t &alignCount, vstd::StringBuilder &structDesc) {
    auto alignedSize = (structSize + tarAlign - 1u) / tarAlign * tarAlign;
    auto padding = alignedSize - structSize;
    if (padding == 0) return;
    // use bitfields to fill small gaps (< 4B)
    for (; (padding & 3) > 1; padding -= 2) {
        structDesc.append(luisa::format(
            "int pad{}:16;\n", alignCount++));
    }
    for (; (padding & 3) > 0; padding--) {
        structDesc.append(luisa::format(
            "int pad{}:8;\n", alignCount++));
    }
    // handle remaining gaps (4 to 12B)
    if (padding != 0) {
        structDesc.append(luisa::format(
            "int pad{}[{}];\n",
            alignCount++, padding / 4));
    }
    structSize = alignedSize;
}

StructureType StructureType::GetScalar() {
    return {Tag::Scalar, uint8_t(0)};
}
StructureType StructureType::GetVector(uint8_t dim) {
    return {Tag::Vector, dim};
}
StructureType StructureType::GetMatrix(uint8_t dim) {
    return {Tag::Matrix, dim};
}
void StructGenerator::InitAsStruct(
    vstd::Iterator<Type const *const> const &vars,
    size_t structIdx,
    vstd::function<StructGenerator *(Type const *)> const &visitor) {
    size_t structSize = 0;
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
        Align(i->alignment());
        switch (i->tag()) {
            case Type::Tag::BOOL:
            case Type::Tag::FLOAT32:
            case Type::Tag::INT32:
            case Type::Tag::UINT32:
                structTypes.emplace_back(varName, StructureType::GetScalar());
                break;
            case Type::Tag::VECTOR:
                structTypes.emplace_back(varName, StructureType::GetVector(i->dimension()));
                break;
            case Type::Tag::MATRIX:
                structTypes.emplace_back(varName, StructureType::GetMatrix(i->dimension()));
                break;
            case Type::Tag::STRUCTURE:
            case Type::Tag::ARRAY:
                structTypes.emplace_back(varName, visitor(i));
                break;
            default:
                LUISA_ERROR_WITH_LOCATION("Invalid struct member '{}'.",
                                          i->description());
        }
        structSize += i->size();
        CodegenUtility::GetTypeName(*i, structDesc, Usage::READ, false);
        structDesc << ' ' << varName;
        if (i->tag() == Type::Tag::BOOL) {
            structDesc << ":8"sv;
        }
        structDesc << ";\n"sv;
        Align(i->alignment());
    }

    updateVarName();
    Align(maxAlign);
}
void StructGenerator::InitAsArray(
    Type const *t,
    size_t structIdx,
    vstd::function<StructGenerator *(Type const *)> const &visitor) {
    auto &&ele = t->element();
    CodegenUtility::GetTypeName(*ele, structDesc, Usage::READ, false);
    structDesc << " v["sv << vstd::to_string(t->dimension()) << "];\n";
}
void StructGenerator::Init(vstd::function<StructGenerator *(Type const *)> const &visitor) {
    if (structureType->tag() == Type::Tag::STRUCTURE) {
        InitAsStruct(vstd::GetIterator(structureType->members()), idx, visitor);
    } else {
        InitAsArray(structureType, idx, visitor);
    }
}
StructGenerator::StructGenerator(
    Type const *structureType,
    size_t structIdx)
    : idx(structIdx),
      structureType{structureType} {
    if (structureType->tag() == Type::Tag::STRUCTURE) {
        structName = "S";
        vstd::to_string(structIdx, structName);
    } else {
        structName = "A";
        vstd::to_string(structIdx, structName);
    }
}
StructGenerator::~StructGenerator() = default;
}// namespace toolhub::directx