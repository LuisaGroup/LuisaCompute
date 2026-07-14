
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

size_t StructGenerator::EmitBoolScalarType(size_t offset, vstd::StringBuilder &str) {
    // Choose a bit-field container whose alignment does not push the bool past its
    // C++ byte offset. C++ bool has size/alignment 1, so it can sit at any offset.
    // HLSL only provides 4-byte (int) and 2-byte (uint16_t) integer containers in
    // DXIL; a 1-byte container is unavailable. For offsets that are multiples of 4
    // we use int. For offsets that are 2 mod 4 (e.g. after a half/int16) we use
    // uint16_t so the bool occupies the same byte as in C++ instead of jumping to
    // the next 4-byte boundary.
    if ((offset & 3) == 0) {
        str << "int"sv;
        return 4;
    }
    str << "uint16_t"sv;
    return 2;
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
        last_type = i;
        switch (i->tag()) {
            case Type::Tag::STRUCTURE:
            case Type::Tag::ARRAY:
                visitor(i);
                break;
            default:
                break;
        }
        auto member_offset = structSize;
        structSize += i->size();
        if (i->is_structure() || i->is_array()) {
            auto name = util->opt->CreateAliasedStruct(i);
            structDesc << name.first;
            structDesc << " v"sv << vstd::to_string(varIdx);
            varIdx++;
            structDesc << ";\n"sv;
        } else if (isSpirv && (i->is_vector() && i->dimension() >= 3 && i->element()->size() > 4)) {
            structDesc << "_Als";
            util->GetTypeName(*i->element(), structDesc, Usage::READ);
            structDesc << luisa::format("{}", i->dimension());
            structDesc << " v"sv << vstd::to_string(varIdx);
            varIdx++;
            structDesc << ";\n"sv;
        } else if (i->is_bool_vector()) {
            structDesc << "int"sv;
            structDesc << " v"sv << vstd::to_string(varIdx);
            varIdx++;
            if (i->dimension() < 4)
                structDesc << luisa::format(":{}", 8 * i->dimension());
            structDesc << ";\n"sv;
        } else if (i->tag() == Type::Tag::BOOL) {
            auto container_size = EmitBoolScalarType(member_offset, structDesc);
            structDesc << " v"sv << vstd::to_string(varIdx);
            varIdx++;
            structDesc << ":8"sv;
            structDesc << ";\n"sv;
            structSize += container_size - i->size();
        } else {
            util->GetTypeName(*i, structDesc, Usage::READ, false);
            structDesc << " v"sv << vstd::to_string(varIdx);
            varIdx++;
            structDesc << ";\n"sv;
        }
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
        last_type = i;
        switch (i->tag()) {
            case Type::Tag::STRUCTURE:
            case Type::Tag::ARRAY:
                visitor(i);
                break;
            default:
                break;
        }
        auto member_offset = structSize;
        structSize += i->size();
        if (i->tag() == Type::Tag::BOOL) {
            auto container_size = EmitBoolScalarType(member_offset, structDesc);
            structDesc << " v"sv << vstd::to_string(varIdx);
            varIdx++;
            structDesc << ":8"sv;
            structDesc << ";\n"sv;
            structSize += container_size - i->size();
        } else {
            util->GetTypeName(*i, structDesc, Usage::READ, false);
            structDesc << " v"sv << vstd::to_string(varIdx);
            varIdx++;
            structDesc << ";\n"sv;
        }
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
