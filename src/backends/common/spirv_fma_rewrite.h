#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace luisa::compute::detail {

// Workaround for SPIRV-Tools (both vendored and inside prebuilt DXC) omitting
// "SPV_KHR_fma" from the extension allowlists of AggressiveDCEPass,
// LocalAccessChainConvertPass, LocalSingleBlockLoadStoreElimPass, and
// LocalSingleStoreElimPass. When a module declares OpExtension "SPV_KHR_fma",
// all four passes silently return Status::SuccessWithoutChange, which both
// blocks performance optimizations and leaves unlegalized HLSL storage-buffer
// function variables alive (causing SPIR-V validation failures).
//
// By emitting/optimizing with core GLSL.std.450 Fma during spirv-opt and
// promoting float32 GLSL.std.450 Fma instructions to OpFmaKHR afterwards, all
// SPIRV-Tools legalization and optimization passes run cleanly while preserving
// exact SPV_KHR_fma single-rounding semantics in the final SPIR-V binary.

namespace spirv_fma_detail {

inline constexpr uint32_t kSpirvMagic = 0x07230203u;
inline constexpr size_t kHeaderWordCount = 5u;

inline constexpr uint16_t kOpExtension = 10u;
inline constexpr uint16_t kOpExtInstImport = 11u;
inline constexpr uint16_t kOpExtInst = 12u;
inline constexpr uint16_t kOpMemoryModel = 14u;
inline constexpr uint16_t kOpCapability = 17u;
inline constexpr uint16_t kOpTypeFloat = 22u;
inline constexpr uint16_t kOpTypeVector = 23u;
inline constexpr uint16_t kOpDecorate = 71u;
inline constexpr uint16_t kOpFmaKHR = 4427u;

inline constexpr uint32_t kCapabilityFMAKHR = 6030u;
inline constexpr uint32_t kDecorationNoContraction = 42u;
inline constexpr uint32_t kGlslStd450Fma = 50u;

inline constexpr std::string_view kExtSpvKhrFma = "SPV_KHR_fma";
inline constexpr std::string_view kExtGlslStd450 = "GLSL.std.450";

[[nodiscard]] inline bool literal_string_equals(
    const uint32_t *words, size_t word_count, std::string_view target) noexcept {
    if (word_count == 0u) { return false; }
    const auto *bytes = reinterpret_cast<const char *>(words);
    const size_t max_bytes = word_count * sizeof(uint32_t);
    size_t len = 0u;
    while (len < max_bytes && bytes[len] != '\0') { ++len; }
    return len < max_bytes && std::string_view{bytes, len} == target;
}

inline void append_literal_string(std::vector<uint32_t> &out,
                                  std::string_view text) {
    const size_t byte_len = text.size() + 1u;
    const size_t word_len = (byte_len + 3u) / 4u;
    const size_t start = out.size();
    out.resize(start + word_len, 0u);
    std::memcpy(out.data() + start, text.data(), text.size());
}

[[nodiscard]] inline bool validate_word_stream(const uint32_t *words,
                                               size_t word_count) noexcept {
    if (words == nullptr || word_count < kHeaderWordCount ||
        words[0] != kSpirvMagic) {
        return false;
    }
    for (size_t i = kHeaderWordCount; i < word_count;) {
        const size_t wc = words[i] >> 16u;
        if (wc == 0u || wc > word_count - i) { return false; }
        i += wc;
    }
    return true;
}

}// namespace spirv_fma_detail

// Demotes all OpFmaKHR instructions to GLSL.std.450 Fma and removes
// OpCapability FMAKHR and OpExtension "SPV_KHR_fma" before running spirv-opt.
// Returns true if the module declared SPV_KHR_fma and was rewritten.
[[nodiscard]] inline bool demote_op_fma_khr_for_spirv_opt(
    std::vector<uint32_t> &words) {
    using namespace spirv_fma_detail;
    if (!validate_word_stream(words.data(), words.size())) { return false; }

    bool has_fma_ext = false;
    bool has_fma_cap = false;
    bool has_fma_inst = false;
    uint32_t glsl_import_id = 0u;

    for (size_t i = kHeaderWordCount; i < words.size();) {
        const uint32_t wc = words[i] >> 16u;
        const uint16_t op = static_cast<uint16_t>(words[i] & 0xffffu);
        if (op == kOpExtension && wc >= 2u &&
            literal_string_equals(words.data() + i + 1u, wc - 1u, kExtSpvKhrFma)) {
            has_fma_ext = true;
        } else if (op == kOpCapability && wc == 2u &&
                   words[i + 1u] == kCapabilityFMAKHR) {
            has_fma_cap = true;
        } else if (op == kOpExtInstImport && wc >= 3u &&
                   literal_string_equals(words.data() + i + 2u, wc - 2u, kExtGlslStd450)) {
            glsl_import_id = words[i + 1u];
        } else if (op == kOpFmaKHR && wc == 6u) {
            has_fma_inst = true;
        }
        i += wc;
    }

    if (!has_fma_ext && !has_fma_cap && !has_fma_inst) { return false; }

    bool insert_glsl_import = (has_fma_inst && glsl_import_id == 0u);
    uint32_t new_bound = words[3];
    if (insert_glsl_import) {
        glsl_import_id = new_bound++;
    }

    std::vector<uint32_t> out;
    out.reserve(words.size() + 16u);
    out.insert(out.end(), words.begin(), words.begin() + kHeaderWordCount);
    out[3] = new_bound;

    bool glsl_inserted = false;
    for (size_t i = kHeaderWordCount; i < words.size();) {
        const uint32_t wc = words[i] >> 16u;
        const uint16_t op = static_cast<uint16_t>(words[i] & 0xffffu);

        if (op == kOpExtension && wc >= 2u &&
            literal_string_equals(words.data() + i + 1u, wc - 1u, kExtSpvKhrFma)) {
            i += wc;
            continue;
        }
        if (op == kOpCapability && wc == 2u &&
            words[i + 1u] == kCapabilityFMAKHR) {
            i += wc;
            continue;
        }
        if (insert_glsl_import && !glsl_inserted && op == kOpMemoryModel) {
            const size_t str_words = (kExtGlslStd450.size() + 1u + 3u) / 4u;
            const uint32_t imp_wc = static_cast<uint32_t>(2u + str_words);
            out.push_back((imp_wc << 16u) | kOpExtInstImport);
            out.push_back(glsl_import_id);
            append_literal_string(out, kExtGlslStd450);
            glsl_inserted = true;
        }
        if (op == kOpFmaKHR && wc == 6u) {
            out.push_back((8u << 16u) | kOpExtInst);
            out.push_back(words[i + 1u]);// result type
            out.push_back(words[i + 2u]);// result id
            out.push_back(glsl_import_id);
            out.push_back(kGlslStd450Fma);
            out.push_back(words[i + 3u]);// a
            out.push_back(words[i + 4u]);// b
            out.push_back(words[i + 5u]);// c
            i += wc;
            continue;
        }
        out.insert(out.end(), words.begin() + i, words.begin() + i + wc);
        i += wc;
    }

    words.swap(out);
    return true;
}

// Promotes float32 scalar/vector GLSL.std.450 Fma instructions to OpFmaKHR and
// ensures OpCapability FMAKHR and OpExtension "SPV_KHR_fma" are declared.
// Returns the number of instructions promoted.
[[nodiscard]] inline size_t promote_glsl_fma_to_op_fma_khr(
    std::vector<uint32_t> &words) {
    using namespace spirv_fma_detail;
    if (!validate_word_stream(words.data(), words.size())) { return 0u; }

    uint32_t glsl_import_id = 0u;
    bool has_fma_ext = false;
    bool has_fma_cap = false;
    std::unordered_set<uint32_t> f32_scalar_types;
    std::unordered_set<uint32_t> f32_types;
    std::unordered_set<uint32_t> promoted_ids;

    for (size_t i = kHeaderWordCount; i < words.size();) {
        const uint32_t wc = words[i] >> 16u;
        const uint16_t op = static_cast<uint16_t>(words[i] & 0xffffu);
        if (op == kOpExtension && wc >= 2u &&
            literal_string_equals(words.data() + i + 1u, wc - 1u, kExtSpvKhrFma)) {
            has_fma_ext = true;
        } else if (op == kOpCapability && wc == 2u &&
                   words[i + 1u] == kCapabilityFMAKHR) {
            has_fma_cap = true;
        } else if (op == kOpExtInstImport && wc >= 3u &&
                   literal_string_equals(words.data() + i + 2u, wc - 2u, kExtGlslStd450)) {
            glsl_import_id = words[i + 1u];
        } else if (op == kOpTypeFloat && wc == 3u && words[i + 2u] == 32u) {
            f32_scalar_types.insert(words[i + 1u]);
            f32_types.insert(words[i + 1u]);
        } else if (op == kOpTypeVector && wc == 4u &&
                   f32_scalar_types.contains(words[i + 2u])) {
            f32_types.insert(words[i + 1u]);
        } else if (op == kOpExtInst && wc == 8u &&
                   glsl_import_id != 0u &&
                   words[i + 3u] == glsl_import_id &&
                   words[i + 4u] == kGlslStd450Fma &&
                   f32_types.contains(words[i + 1u])) {
            promoted_ids.insert(words[i + 2u]);
        }
        i += wc;
    }

    if (promoted_ids.empty()) { return 0u; }

    std::vector<uint32_t> out;
    out.reserve(words.size() + 8u);
    out.insert(out.end(), words.begin(), words.begin() + kHeaderWordCount);

    bool cap_inserted = has_fma_cap;
    bool ext_inserted = has_fma_ext;

    auto emit_cap = [&]() {
        if (!cap_inserted) {
            out.push_back((2u << 16u) | kOpCapability);
            out.push_back(kCapabilityFMAKHR);
            cap_inserted = true;
        }
    };
    auto emit_ext = [&]() {
        if (!ext_inserted) {
            const size_t str_words = (kExtSpvKhrFma.size() + 1u + 3u) / 4u;
            const uint32_t ext_wc = static_cast<uint32_t>(1u + str_words);
            out.push_back((ext_wc << 16u) | kOpExtension);
            append_literal_string(out, kExtSpvKhrFma);
            ext_inserted = true;
        }
    };

    for (size_t i = kHeaderWordCount; i < words.size();) {
        const uint32_t wc = words[i] >> 16u;
        const uint16_t op = static_cast<uint16_t>(words[i] & 0xffffu);

        if (op != kOpCapability) {
            emit_cap();
        }
        if (op != kOpCapability && op != kOpExtension) {
            emit_ext();
        }

        // OpFmaKHR already specifies exact single-rounding FMA semantics;
        // drop redundant NoContraction decorations that DXC attaches to GLSL Fma.
        if (op == kOpDecorate && wc == 3u &&
            words[i + 2u] == kDecorationNoContraction &&
            promoted_ids.contains(words[i + 1u])) {
            i += wc;
            continue;
        }

        if (op == kOpExtInst && wc == 8u &&
            words[i + 3u] == glsl_import_id &&
            words[i + 4u] == kGlslStd450Fma &&
            promoted_ids.contains(words[i + 2u])) {
            out.push_back((6u << 16u) | kOpFmaKHR);
            out.push_back(words[i + 1u]);// result type
            out.push_back(words[i + 2u]);// result id
            out.push_back(words[i + 5u]);// a
            out.push_back(words[i + 6u]);// b
            out.push_back(words[i + 7u]);// c
            i += wc;
            continue;
        }

        out.insert(out.end(), words.begin() + i, words.begin() + i + wc);
        i += wc;
    }

    words.swap(out);
    return promoted_ids.size();
}

}// namespace luisa::compute::detail
