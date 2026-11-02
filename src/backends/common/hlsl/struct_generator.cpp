
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
void StructGenerator::ProvideAlignVariable(Type const *type, size_t tarAlign, size_t &alignCount, size_t &structSize, vstd::StringBuilder &structDesc, bool bf16_fallback) {
    auto alignedSize = (structSize + tarAlign - 1u) / tarAlign * tarAlign;
    auto padding = alignedSize - structSize;
    if (padding == 0) return;
    // use bitfields to fill small gaps (< 4B)
    auto bit_padding = (padding & 3);
    if (bit_padding > 0) {
        if (type && (type->is_float16() || type->is_float16_vector())) {
            LUISA_ASSERT(bit_padding == 2, "Invalid struct alignment.");
            structDesc << "float16_t _a"sv << vstd::to_string(alignCount++) << ";\n"sv;
        } else if (type && (type->is_bfloat16() || type->is_bfloat16_vector())) {
            LUISA_ASSERT(bit_padding == 2, "Invalid struct alignment.");
            structDesc << bf16_type_name(bf16_fallback) << " _a"sv << vstd::to_string(alignCount++) << ";\n"sv;
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

namespace {

// The kind of a member that can share a packed 32-bit word, if any.
[[nodiscard]] bool packed_member_kind(Type const *type, bool bf16_fallback,
                                      PackedStructMember::Kind &kind) noexcept {
    if (type == nullptr || !type->is_scalar()) { return false; }
    switch (type->tag()) {
        case Type::Tag::FLOAT16: kind = PackedStructMember::Kind::F16; return true;
        case Type::Tag::INT16: kind = PackedStructMember::Kind::I16; return true;
        case Type::Tag::UINT16: kind = PackedStructMember::Kind::U16; return true;
        case Type::Tag::BOOL: kind = PackedStructMember::Kind::BOOL; return true;
        case Type::Tag::BFLOAT16:
            // The native HLSL BF16 type cannot be reinterpreted as a 16-bit
            // pattern, so only the software emulation (raw 16-bit storage) can
            // take part in a packed word.
            if (!bf16_fallback) { return false; }
            kind = PackedStructMember::Kind::BF16;
            return true;
        default: return false;
    }
}

[[nodiscard]] constexpr size_t align_up(size_t value, size_t alignment) noexcept {
    return (value + alignment - 1u) / alignment * alignment;
}

}// namespace

StructPacking plan_struct_packing(Type const *structType,
                                 bool bf16_fallback) {
    StructPacking packing;
    if (structType == nullptr || !structType->is_structure()) { return packing; }
    auto members = structType->members();
    if (members.size() < 2u) { return packing; }
    // Byte offsets of the members in the host (C++) layout, which is what the
    // generated HLSL struct has to reproduce: values are copied verbatim into the
    // argument block and buffer elements are the host structs themselves.
    vstd::vector<size_t> offsets;
    offsets.reserve(members.size());
    auto offset = size_t{0u};
    for (auto member : members) {
        offset = align_up(offset, member->alignment());
        offsets.emplace_back(offset);
        offset += member->size();
    }
    // Group the members into 32-bit words.  A word needs packing exactly when it
    // mixes a one-byte bool with a 16-bit scalar: a word of only bools (bitfields
    // pack bit-wise from the word start) or of only 16-bit scalars (two per word)
    // is laid out by dxc exactly like the host does.
    size_t emitted = 0u;// number of members the struct emitter has produced so far
    for (auto i = size_t{0u}; i < members.size();) {
        auto kind = PackedStructMember::Kind::BOOL;
        if (!packed_member_kind(members[i], bf16_fallback, kind)) {
            ++i;
            ++emitted;
            continue;
        }
        auto word_offset = offsets[i] & ~size_t{3u};
        auto end = i;
        auto has_bool = false;
        auto has_narrow = false;
        while (end < members.size()) {
            auto other_kind = PackedStructMember::Kind::BOOL;
            if (!packed_member_kind(members[end], bf16_fallback, other_kind)) { break; }
            if ((offsets[end] & ~size_t{3u}) != word_offset) { break; }
            has_bool |= other_kind == PackedStructMember::Kind::BOOL;
            has_narrow |= other_kind != PackedStructMember::Kind::BOOL;
            ++end;
        }
        if (!has_bool || !has_narrow) {
            // dxc reproduces the host layout of this word; emit it normally.
            emitted += end - i;
            i = end;
            continue;
        }
        if (offsets[i] != word_offset) {
            // The word starts inside the previous member, so it cannot become one
            // `uint` of its own.
            LUISA_ERROR(
                "HLSL cannot express the layout of this struct: a 16-bit member "
                "shares a 32-bit word with a one-byte bool, but the word also holds "
                "the tail of '{}'. Reorder the members.",
                members[i - 1u]->description());
        }
        if (end < members.size() && offsets[end] < word_offset + 4u) {
            // A member that is not a 16-bit scalar or a bool (a bool vector, for
            // example) starts inside the word; emitting the word as one `uint`
            // would move that member.
            LUISA_ERROR(
                "HLSL cannot express the layout of this struct: '{}' starts inside a "
                "32-bit word that mixes 16-bit scalars with a one-byte bool. Move "
                "that member out of the word (for example give it 4-byte "
                "alignment).",
                members[end]->description());
        }
        StructPacking::Word word{.byte_offset = word_offset,
                                 .word_index = static_cast<uint>(emitted),
                                 .members = {}};
        for (auto j = i; j < end; ++j) {
            auto member_kind = PackedStructMember::Kind::BOOL;
            packed_member_kind(members[j], bf16_fallback, member_kind);
            word.members.emplace_back(static_cast<uint>(j));
            packing.members.try_emplace(
                static_cast<uint>(j),
                PackedStructMember{.word_index = word.word_index,
                                   .byte_offset = static_cast<uint>(offsets[j] - word_offset),
                                   .kind = member_kind});
        }
        packing.words.emplace_back(std::move(word));
        ++emitted;// the whole word is one emitted member
        i = end;
    }
    return packing;
}

bool StructGenerator::half_type_adjacent_with_bool(Type const *a, Type const *b) {
    switch (a->tag() == Type::Tag::VECTOR ? a->element()->tag() : a->tag()) {
        case Type::Tag::FLOAT16:
        case Type::Tag::BFLOAT16:
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
        ProvideAlignVariable(last_type, tarAlign, alignCount, structSize, structDesc,
                             util->opt->bf16_fallback);
    };
    // Words that mix 16-bit scalars with one-byte bools are emitted as one `uint`
    // and accessed through the `_lc_pack_*` helpers (see `PackedStructMember`).
    auto packing = util->opt->GetStructPacking(originType);
    auto packed_end = size_t{0u};
    auto next_word = size_t{0u};
    size_t varIdx = 0;
    for (auto i = size_t{0u}; i < vars.size(); ++i) {
        if (i < packed_end) { continue; }
        if (next_word < packing.words.size() &&
            packing.words[next_word].members.front() == static_cast<uint>(i)) {
            auto &word = packing.words[next_word];
            LUISA_ASSERT(align_up(structSize, 4u) == word.byte_offset,
                         "A packed struct word must start at its host byte offset.");
            Align(4u);
            structDesc << "uint v"sv << vstd::to_string(word.word_index) << ";\n"sv;
            structSize = word.byte_offset + 4u;
            varIdx = word.word_index + 1u;
            packed_end = word.members.back() + 1u;
            last_type = vars[word.members.back()];
            ++next_word;
            continue;
        }
        auto member = vars[i];
        Align(member->alignment());
        if (last_type && (half_type_adjacent_with_bool(last_type, member) || half_type_adjacent_with_bool(member, last_type))) [[unlikely]] {
            LUISA_ERROR(
                "HLSL cannot express this struct layout: '{}' is adjacent to '{}'. "
                "dxc allocates bitfields in whole 32-bit units, so a one-byte bool "
                "member cannot share a word with 16-bit data.",
                member->description(), last_type->description());
        }
        last_type = member;
        switch (member->tag()) {
            case Type::Tag::STRUCTURE:
            case Type::Tag::ARRAY:
                visitor(member);
                break;
            default:
                break;
        }
        structSize += member->size();
        if (member->is_structure() || member->is_array()) {
            auto name = util->opt->CreateAliasedStruct(member);
            structDesc << name.first;
        } else if (isSpirv && (member->is_vector() && member->dimension() >= 3 && member->element()->size() > 4)) {
            structDesc << "_Als";
            util->GetTypeName(*member->element(), structDesc, Usage::READ);
            structDesc << luisa::format("{}", member->dimension());
        } else if (member->tag() == Type::Tag::BOOL || member->is_bool_vector()) {
            structDesc << "int"sv;
        } else {
            util->GetTypeName(*member, structDesc, Usage::READ, false);
        }
        structDesc << " v"sv << vstd::to_string(varIdx);
        varIdx++;
        if (member->tag() == Type::Tag::BOOL) {
            structDesc << ":8"sv;
        } else if (member->is_bool_vector()) {
            if (member->dimension() < 4)
                structDesc << luisa::format(":{}", 8 * member->dimension());
        }
        structDesc << ";\n"sv;
        Align(member->alignment());
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
        ProvideAlignVariable(last_type, tarAlign, alignCount, structSize, structDesc,
                             util->opt->bf16_fallback);
    };
    // Words that mix 16-bit scalars with one-byte bools are emitted as one `uint`
    // and accessed through the `_lc_pack_*` helpers (see `PackedStructMember`).
    auto packing = util->opt->GetStructPacking(originType);
    auto packed_end = size_t{0u};
    auto next_word = size_t{0u};
    size_t varIdx = 0;
    for (auto i = size_t{0u}; i < vars.size(); ++i) {
        if (i < packed_end) { continue; }
        if (next_word < packing.words.size() &&
            packing.words[next_word].members.front() == static_cast<uint>(i)) {
            auto &word = packing.words[next_word];
            LUISA_ASSERT(align_up(structSize, 4u) == word.byte_offset,
                         "A packed struct word must start at its host byte offset.");
            Align(4u);
            structDesc << "uint v"sv << vstd::to_string(word.word_index) << ";\n"sv;
            structSize = word.byte_offset + 4u;
            varIdx = word.word_index + 1u;
            packed_end = word.members.back() + 1u;
            last_type = vars[word.members.back()];
            ++next_word;
            continue;
        }
        auto member = vars[i];
        Align(member->alignment());
        if (last_type && (half_type_adjacent_with_bool(last_type, member) || half_type_adjacent_with_bool(member, last_type))) [[unlikely]] {
            LUISA_ERROR(
                "HLSL cannot express this struct layout: '{}' is adjacent to '{}'. "
                "dxc allocates bitfields in whole 32-bit units, so a one-byte bool "
                "member cannot share a word with 16-bit data.",
                member->description(), last_type->description());
        }
        last_type = member;
        switch (member->tag()) {
            case Type::Tag::STRUCTURE:
            case Type::Tag::ARRAY:
                visitor(member);
                break;
            default:
                break;
        }
        structSize += member->size();
        // Note: This non-aliased path does NOT need bool-vector bitfield support
        // (like InitAsStructAliased) because:
        // 1. VectorShouldBeAliased() always returns true for bool vectors
        //    (element()->is_bool()), so any struct with bool vectors goes through
        //    CreateAliasedStruct → InitAsStructAliased, never here.
        // 2. Even if a bool vector reached here, GetTypeName emits native HLSL
        //    bool2/bool3/bool4 types which are valid for non-aliased local structs.
        // Verified by test BoolVecTest in src/tests/unit/dsl/test_dsl.cpp
        if (member->tag() == Type::Tag::BOOL) {
            structDesc << "int"sv;
        } else {
            util->GetTypeName(*member, structDesc, Usage::READ, false);
        }
        structDesc << " v"sv << vstd::to_string(varIdx);
        varIdx++;
        if (member->tag() == Type::Tag::BOOL) {
            structDesc << ":8"sv;
        }
        structDesc << ";\n"sv;
        Align(member->alignment());
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

PackedStructMember const *CodegenUtility::GetPackedStructMember(
    Type const *type, uint member_index) {
    if (opt == nullptr || type == nullptr || !type->is_structure()) {
        return nullptr;
    }
    auto const &packing = opt->GetStructPacking(type);
    if (packing.empty()) { return nullptr; }
    auto iter = packing.members.find(member_index);
    return iter == packing.members.end() ? nullptr : &iter->second;
}
}// namespace lc::hlsl
