
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
void StructGenerator::ProvideAlignVariable(Type const *type, size_t tarAlign, size_t &alignCount, size_t &structSize, vstd::StringBuilder &structDesc) {
    auto alignedSize = (structSize + tarAlign - 1u) / tarAlign * tarAlign;
    auto padding = alignedSize - structSize;
    if (padding == 0) return;
    // use bitfields to fill small gaps (< 4B)
    auto bit_padding = (padding & 3);
    if (bit_padding > 0) {
        if (type && (type->is_float16() || type->is_float16_vector())) {
            LUISA_ASSERT(bit_padding == 2, "Invalid struct alignment.");
            structDesc << "float16_t _a"sv << vstd::to_string(alignCount++) << ";\n"sv;
        } else if (type && (type->is_int16() || type->is_int16_vector())) {
            LUISA_ASSERT(bit_padding == 2, "Invalid struct alignment.");
            structDesc << "int16_t _a"sv << vstd::to_string(alignCount++) << ";\n"sv;
        } else if (type && (type->is_uint16() || type->is_uint16_vector())) {
            LUISA_ASSERT(bit_padding == 2, "Invalid struct alignment.");
            structDesc << "uint16_t _a"sv << vstd::to_string(alignCount++) << ";\n"sv;
        } else {
            structDesc << "int _a"sv << vstd::to_string(alignCount++) << ":" << vstd::to_string(bit_padding * 8) << ";\n"sv;
        }
    }
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

bool StructGenerator::half_type_adjacent_with_bool(Type const *a, Type const *b) {
    switch (a->tag() == Type::Tag::VECTOR ? a->element()->tag() : a->tag()) {
        case Type::Tag::FLOAT16:
        case Type::Tag::INT16:
        case Type::Tag::UINT16:
            break;
        default:
            return false;
    }
    switch (b->tag() == Type::Tag::VECTOR ? b->element()->tag() : b->tag()) {
        case Type::Tag::BOOL:
            return true;
        default:
            return false;
    }
}

void StructGenerator::InitAsStructAliased(
    Type const *originType,
    vstd::span<Type const *const> const &vars,
    size_t /*structIdx*/,
    Callback const &visitor,
    bool isSpirv) {
    size_t alignCount = 0;
    size_t structSize = 0;
    structDesc.reserve(256);
    Type const *last_type = nullptr;

    auto Align = [&](size_t tarAlign) {
        ProvideAlignVariable(last_type, tarAlign, alignCount, structSize, structDesc);
    };
    size_t varIdx = 0;
    for (auto &&i : vars) {
        Align(i->alignment());
        if (last_type && (half_type_adjacent_with_bool(last_type, i) || half_type_adjacent_with_bool(i, last_type))) [[unlikely]] {
            LUISA_ERROR("HLSL do not support 16-bit variables adjacent with bool");
        }
        last_type = i;
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
        } else if (isSpirv && (i->is_vector() && i->dimension() >= 3 && i->element()->size() > 4)) {
            structDesc << "_Als";
            util->GetTypeName(*i->element(), structDesc, Usage::READ);
            structDesc << luisa::format("{}", i->dimension());
        } else if (i->tag() == Type::Tag::BOOL || i->is_bool_vector()) { 
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
    Type const *structureType,
    size_t /*structIdx*/,
    Callback const & /*visitor*/,
    bool isSpirv) {
    auto i = structureType->element();
    if (i->is_structure() || i->is_array()) {
        auto name = util->opt->CreateAliasedStruct(i);
        structDesc << name.first;
    } else if (isSpirv && i->is_vector() && i->dimension() >= 3 && i->element()->size() > 4) {
        structDesc << "_Als";
        util->GetTypeName(*i->element(), structDesc, Usage::READ);
        structDesc << luisa::format("{}", i->dimension());
    } else {
        util->GetTypeName(*i, structDesc, Usage::READ, false);
    }
    structDesc << " v["sv << vstd::to_string(structureType->dimension()) << "];\n";
}

void StructGenerator::InitAsStruct(
    Type const *originType,
    vstd::span<Type const *const> const &vars,
    size_t /*structIdx*/,
    Callback const &visitor,
    bool isSpirv) {
    size_t alignCount = 0;
    size_t structSize = 0;
    structDesc.reserve(256);
    Type const *last_type = nullptr;
    auto Align = [&](size_t tarAlign) {
        ProvideAlignVariable(last_type, tarAlign, alignCount, structSize, structDesc);
    };
    size_t varIdx = 0;
    for (auto &&i : vars) {
        if (util->opt->dispatch_grid_records.contains(originType) && varIdx == 0) {
            structDesc << "uint3 v0 : SV_DispatchGrid;\n"sv;
            structDesc << "int _v0_pad;\n"sv;
            structSize += 16;
            last_type = i;
            varIdx += 1;
            continue;
        }

        Align(i->alignment());
        if (last_type && (half_type_adjacent_with_bool(last_type, i) || half_type_adjacent_with_bool(i, last_type))) [[unlikely]] {
            LUISA_ERROR("HLSL do not support 16-bit variables adjacent with bool");
        }
        last_type = i;
        switch (i->tag()) {
            case Type::Tag::STRUCTURE:
            case Type::Tag::ARRAY:
                visitor(i);
                break;
            default:
                break;
        }
        structSize += i->size();
        // Note: This non-aliased path does NOT need bool-vector bitfield support
        // (like line 119 in InitAsStructAliased) because:
        // 1. VectorShouldBeAliased() always returns true for bool vectors
        //    (element()->is_bool()), so any struct with bool vectors goes through
        //    CreateAliasedStruct → InitAsStructAliased, never here.
        // 2. Even if a bool vector reached here, GetTypeName emits native HLSL
        //    bool2/bool3/bool4 types which are valid for non-aliased local structs.
        // Verified by test BoolVecTest in src/tests/unit/dsl/test_dsl.cpp
        if (i->tag() == Type::Tag::BOOL) {
            structDesc << "int"sv;
        } else {
            util->GetTypeName(*i, structDesc, Usage::READ, false);
        }
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
    Type const *structureType,
    size_t /*structIdx*/,
    Callback const & /*visitor*/,
    bool isSpirv) {
    const auto ele = structureType->element();
    util->GetTypeName(*ele, structDesc, Usage::READ, false);
    structDesc << " v["sv << vstd::to_string(structureType->dimension()) << "];\n";
}
void StructGenerator::InitAliased(Callback const &visitor, bool isSpirv) {
    if (structureType->tag() == Type::Tag::STRUCTURE) {
        InitAsStructAliased(structureType, structureType->members(), idx, visitor, isSpirv);
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
