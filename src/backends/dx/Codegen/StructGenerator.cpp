
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
    LUISA_ERROR_WITH_LOCATION("Invalid.");
}
void StructGenerator::ProvideAlignVariable(size_t tarAlign, size_t &structSize,
                                           size_t &alignCount, vstd::string &structDesc) {
    auto alignedSize = (structSize + tarAlign - 1u) / tarAlign * tarAlign;
    auto padding = alignedSize - structSize;
    if (padding == 0) return;
    // use bitfields to fill small gaps (< 4B)
    for (; padding % 4 != 0; padding--) {
        structDesc.append(luisa::format(
            "bool pad{}:8;\n", alignCount++));
    }
    // handle remaining gaps (4 to 12B)
    if (padding != 0) {
        structDesc.append(luisa::format(
            "uint pad{}[{}];\n",
            alignCount++, padding / 4));
    }
    structSize = alignedSize;
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
    auto Align = [&](size_t tarAlign) {
        ProvideAlignVariable(tarAlign, structSize, alignCount, structDesc);
    };

    for (; vars; vars++) {
        auto &&i = *vars;
        updateVarName();
        Align(i->alignment());
        switch (i->tag()) {
            case Type::Tag::BOOL:
            case Type::Tag::FLOAT:
            case Type::Tag::INT:
            case Type::Tag::UINT:
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
        CodegenUtility::GetTypeName(*i, structDesc, Usage::READ);
        structDesc << ' ' << varName;
        if (i->tag() == Type::Tag::BOOL) {
            // HLSL bool are 4-byte aligned, so
            // use bitfields here to workaround
            structDesc << ":8"sv;
        }
        structDesc << ";\n"sv;
    }
    // final padding for structure-wise alignment
    Align(structureType->alignment());
}

void StructGenerator::InitAsArray(
    Type const *t,
    size_t structIdx,
    vstd::function<StructGenerator *(Type const *)> const &visitor) {
    structName = "A";
    vstd::to_string(structIdx, structName);
    auto ele = t->element();
    if (ele->is_vector() && ele->dimension() == 3u) {// work around vector3 alignment
        ele = Type::from(luisa::format("vector<{},4>", ele->element()->description()));
    }
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
