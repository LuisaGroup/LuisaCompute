
#include "codegen_stack_data.h"
#include "struct_generator.h"
#include "dx_codegen.h"
#include <HLSL/common_iterator.h>
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
void StructGenerator::ProvideAlignVariable(size_t tarAlign, size_t &alignCount, size_t &structSize, vstd::StringBuilder &structDesc) {
    auto alignedSize = (structSize + tarAlign - 1u) / tarAlign * tarAlign;
    auto padding = alignedSize - structSize;
    if (padding == 0) return;
    // use bitfields to fill small gaps (< 4B)
    for (; (padding & 3) > 1; padding -= 2) {
        structDesc << "int _a"sv << vstd::to_string(alignCount++) << ":16;\n"sv;
    }
    for (; (padding & 3) > 0; padding--) {
        structDesc << "int _a"sv << vstd::to_string(alignCount++) << ":8;\n"sv;
    }
    // handle remaining gaps (4 to 12B)
    if (padding != 0) {
        auto varCount = padding / 4;
        if (varCount > 1) {
            structDesc << "int _a"sv << vstd::to_string(alignCount++) << '[' << vstd::to_string(varCount) << ']' << ";\n"sv;
        } else {
            structDesc << "int _a"sv << vstd::to_string(alignCount++) << ";\n"sv;
        }
    }
    structSize = alignedSize;
}

void StructGenerator::InitAsStruct(
    vstd::Iterator<Type const *const> const &vars,
    size_t structIdx,
    Callback const &visitor) {
    size_t alignCount = 0;
    size_t structSize = 0;
    structDesc.reserve(256);
    auto szOpt = vars.Get()->Length();

    size_t maxAlign = 4;
    auto Align = [&](size_t tarAlign) {
        maxAlign = std::max(maxAlign, tarAlign);
        ProvideAlignVariable(tarAlign, alignCount, structSize, structDesc);
    };
    size_t varIdx = 0;
    for (; vars; vars++) {
        auto &&i = *vars;
        Align(i->alignment());
        switch (i->tag()) {
            case Type::Tag::STRUCTURE:
            case Type::Tag::ARRAY:
                visitor(i);
                break;
        }
        structSize += i->size();
        CodegenUtility::GetTypeName(*i, structDesc, Usage::READ, false);
        structDesc << " v"sv << vstd::to_string(varIdx);
        varIdx++;
        if (i->tag() == Type::Tag::BOOL) {
            structDesc << ":8"sv;
        }
        structDesc << ";\n"sv;
        Align(i->alignment());
    }
    Align(maxAlign);
}
void StructGenerator::InitAsArray(
    Type const *t,
    size_t structIdx,
    Callback const &visitor) {
    auto &&ele = t->element();
    CodegenUtility::GetTypeName(*ele, structDesc, Usage::READ, false);
    structDesc << " v["sv << vstd::to_string(t->dimension()) << "];\n";
}
void StructGenerator::Init(Callback const &visitor) {
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