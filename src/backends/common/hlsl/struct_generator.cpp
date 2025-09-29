
#include "codegen_stack_data.h"
#include "struct_generator.h"
#include "hlsl_codegen.h"
namespace lc::hlsl {
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
    auto bit_padding = (padding & 3);
    if (bit_padding > 0)
        structDesc << "int _a"sv << vstd::to_string(alignCount++) << ":" << vstd::to_string(bit_padding * 8) << ";\n"sv;
    padding -= bit_padding;
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

void StructGenerator::InitAsStructAlised(
    Type const *originType,
    vstd::span<Type const *const> const &vars,
    size_t structIdx,
    Callback const &visitor,
    bool isSpirv) {
    size_t alignCount = 0;
    size_t structSize = 0;
    structDesc.reserve(256);

    auto Align = [&](size_t tarAlign) {
        ProvideAlignVariable(tarAlign, alignCount, structSize, structDesc);
    };
    size_t varIdx = 0;
    for (auto &&i : vars) {
        Align(i->alignment());
        switch (i->tag()) {
            case Type::Tag::STRUCTURE:
            case Type::Tag::ARRAY:
                visitor(i);
                break;
            default:
                break;
        }
        structSize += i->size();
        if (i->is_structure() || i->is_array()) {
            auto name = util->opt->CreateAliasedStruct(i);
            structDesc << name.first;
        } else if (i->is_vector() && i->dimension() >= 3 && i->element()->size() > 4) {
            structDesc << "_Als";
            util->GetTypeName(*i->element(), structDesc, Usage::READ);
            structDesc << luisa::format("{}", i->dimension());
        } else if (i->is_bool_vector()) {
            structDesc << "int"sv;
        } else {
            util->GetTypeName(*i, structDesc, Usage::READ, false);
        }
        structDesc << " v"sv << vstd::to_string(varIdx);
        varIdx++;
        if (i->tag() == Type::Tag::BOOL) {
            structDesc << ":8"sv;
        } else if (i->is_bool_vector()) {
            if (i->dimension() < 4)
                structDesc << luisa::format(":{}", 8 * i->dimension());
        }
        structDesc << ";\n"sv;
        Align(i->alignment());
    }
    Align(originType->alignment());
}

void StructGenerator::InitAsArrayAliased(
    Type const *t,
    size_t structIdx,
    Callback const &visitor,
    bool isSpirv) {
    if (t->alignment() < 4) [[unlikely]] {
        LUISA_ERROR("HLSL do not support member's type {} which alignment less than 4-bytes.", t->description());
    }
    auto i = t->element();
    if (i->is_structure() || i->is_array()) {
        auto name = util->opt->CreateAliasedStruct(i);
        structDesc << name.first;
    } else if (i->is_vector() && i->dimension() >= 3 && i->element()->size() > 4) {
        structDesc << "_Als";
        util->GetTypeName(*i->element(), structDesc, Usage::READ);
        structDesc << luisa::format("{}", i->dimension());
    } else {
        util->GetTypeName(*i, structDesc, Usage::READ, false);
    }
    structDesc << " v["sv << vstd::to_string(t->dimension()) << "];\n";
}

void StructGenerator::InitAsStruct(
    Type const *originType,
    vstd::span<Type const *const> const &vars,
    size_t structIdx,
    Callback const &visitor,
    bool isSpirv) {
    size_t alignCount = 0;
    size_t structSize = 0;
    structDesc.reserve(256);

    auto Align = [&](size_t tarAlign) {
        ProvideAlignVariable(tarAlign, alignCount, structSize, structDesc);
    };
    size_t varIdx = 0;
    for (auto &&i : vars) {
        Align(i->alignment());
        switch (i->tag()) {
            case Type::Tag::STRUCTURE:
            case Type::Tag::ARRAY:
                visitor(i);
                break;
            default:
                break;
        }
        structSize += i->size();
        util->GetTypeName(*i, structDesc, Usage::READ, false);
        structDesc << " v"sv << vstd::to_string(varIdx);
        varIdx++;
        if (i->tag() == Type::Tag::BOOL) {
            structDesc << ":8"sv;
        }
        structDesc << ";\n"sv;
        Align(i->alignment());
    }
    Align(originType->alignment());
}
void StructGenerator::InitAsArray(
    Type const *t,
    size_t structIdx,
    Callback const &visitor,
    bool isSpirv) {
    auto &&ele = t->element();
    if (t->alignment() < 4) [[unlikely]] {
        LUISA_ERROR("HLSL do not support member's type {} which alignment less than 4-bytes.", t->description());
    }
    util->GetTypeName(*ele, structDesc, Usage::READ, false);
    structDesc << " v["sv << vstd::to_string(t->dimension()) << "];\n";
}
void StructGenerator::InitAliased(Callback const &visitor, bool isSpirv) {
    if (structureType->tag() == Type::Tag::STRUCTURE) {
        InitAsStructAlised(structureType, structureType->members(), idx, visitor, isSpirv);
    } else {
        InitAsArrayAliased(structureType, idx, visitor, isSpirv);
    }
}
void StructGenerator::Init(Callback const &visitor, bool isSpirv) {
    if (structureType->tag() == Type::Tag::STRUCTURE) {
        InitAsStruct(structureType, structureType->members(), idx, visitor, isSpirv);
    } else {
        InitAsArray(structureType, idx, visitor, isSpirv);
    }
}

StructGenerator::StructGenerator(
    Type const *structureType,
    size_t structIdx,
    CodegenUtility *util)
    : structureType{structureType},
      util(util),
      idx(structIdx) {
    if (structureType->tag() == Type::Tag::STRUCTURE) {
        structName = "_S"sv;
        vstd::to_string(structIdx, structName);
    } else {
        structName = "_A"sv;
        vstd::to_string(structIdx, structName);
    }
}
StructGenerator::~StructGenerator() = default;
}// namespace lc::hlsl
