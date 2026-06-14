#include "spirv_motion_patch.h"
#include <luisa/core/logging.h>
#include <cstring>

namespace lc::vk {

// SPIR-V constants
static constexpr uint32_t SpvMagicNumber = 0x07230203u;
static constexpr uint32_t SpvOpCapability = 17u;
static constexpr uint32_t SpvOpExtension = 10u;
static constexpr uint32_t SpvOpTypeStruct = 30u;
static constexpr uint32_t SpvOpCompositeConstruct = 80u;
static constexpr uint32_t SpvOpStore = 62u;
static constexpr uint32_t SpvOpTraceRayKHR = 4445u;
static constexpr uint32_t SpvOpTraceRayMotionNV = 5339u;  // NOT 5338 (OpTraceMotionNV, reserved)
static constexpr uint32_t SpvCapabilityRayTracingMotionBlurNV = 5341u;

static uint32_t spv_opcode(uint32_t word) { return word & 0xFFFFu; }
static uint32_t spv_word_count(uint32_t word) { return word >> 16u; }
static uint32_t spv_make_word(uint32_t opcode, uint32_t word_count) {
    return (word_count << 16u) | (opcode & 0xFFFFu);
}

static vstd::vector<uint32_t> spv_encode_string(const char *str) {
    auto len = strlen(str);
    auto word_count = (len + 4) / 4;
    vstd::vector<uint32_t> words(word_count, 0u);
    memcpy(words.data(), str, len);
    return words;
}

vstd::vector<uint32_t> patch_spirv_for_motion_blur(vstd::span<uint32_t const> spirv) {
    if (spirv.size() < 5 || spirv[0] != SpvMagicNumber) {
        LUISA_WARNING("Invalid SPIR-V binary, skipping motion blur patch.");
        return {spirv.begin(), spirv.end()};
    }

    // Strategy: The HLSL stores the time value as the last field (index 4)
    // of the _MotionPayload struct in the payload variable before calling
    // TraceRay. In SPIR-V this becomes:
    //
    //   %payload_val = OpCompositeConstruct %PayloadType ... %time_value
    //   OpStore %payload_var %payload_val
    //   OpTraceRayKHR ... %payload_var
    //
    // We find the OpStore to the payload variable preceding OpTraceRayKHR,
    // then find the OpCompositeConstruct that produced the stored value,
    // and extract the time value (last field of the struct).

    // Phase 1: Find all OpTraceRayKHR instructions and count them
    uint32_t trace_ray_count = 0;
    {
        size_t i = 5;
        while (i < spirv.size()) {
            auto wc = spv_word_count(spirv[i]);
            if (wc == 0) break;
            if (spv_opcode(spirv[i]) == SpvOpTraceRayKHR) trace_ray_count++;
            i += wc;
        }
    }

    if (trace_ray_count == 0) {
        return {spirv.begin(), spirv.end()};
    }

    // Phase 2: Build maps for analysis
    // Map from result ID -> instruction offset (for OpCompositeConstruct lookups)
    vstd::unordered_map<uint32_t, size_t> result_id_to_offset;

    {
        size_t i = 5;
        while (i < spirv.size()) {
            auto wc = spv_word_count(spirv[i]);
            auto op = spv_opcode(spirv[i]);
            if (wc == 0) break;

            if (op == SpvOpCompositeConstruct && wc >= 3) {
                result_id_to_offset[spirv[i + 2]] = i;
            }

            i += wc;
        }
    }

    // Phase 3: For each OpTraceRayKHR, find the time value
    struct TraceRayPatch {
        size_t offset;
        uint32_t time_value_id;
    };
    vstd::vector<TraceRayPatch> patches;

    {
        vstd::vector<size_t> instruction_offsets;
        {
            size_t i = 5;
            while (i < spirv.size()) {
                auto wc = spv_word_count(spirv[i]);
                if (wc == 0) break;
                instruction_offsets.emplace_back(i);
                i += wc;
            }
        }

        for (size_t idx = 0; idx < instruction_offsets.size(); idx++) {
            auto off = instruction_offsets[idx];
            auto op = spv_opcode(spirv[off]);
            if (op != SpvOpTraceRayKHR) continue;

            // OpTraceRayKHR: 12 words (1 header + 11 operands)
            // Last operand (word[11]) is the payload variable
            uint32_t payload_var_id = spirv[off + 11];

            // Scan backwards to find OpStore to payload_var_id
            uint32_t time_value_id = 0;
            for (size_t j = idx; j > 0; j--) {
                auto prev_off = instruction_offsets[j - 1];
                auto prev_op = spv_opcode(spirv[prev_off]);
                auto prev_wc = spv_word_count(spirv[prev_off]);

                if (prev_op == SpvOpStore && prev_wc >= 3) {
                    if (spirv[prev_off + 1] == payload_var_id) {
                        uint32_t stored_value_id = spirv[prev_off + 2];
                        auto cc_it = result_id_to_offset.find(stored_value_id);
                        if (cc_it != result_id_to_offset.end()) {
                            auto cc_off = cc_it->second;
                            auto cc_wc = spv_word_count(spirv[cc_off]);
                            // Last element of OpCompositeConstruct is the time field
                            if (cc_wc >= 4) {
                                time_value_id = spirv[cc_off + cc_wc - 1];
                            }
                        }
                        break;
                    }
                }
            }

            if (time_value_id == 0) {
                LUISA_WARNING("Could not find time value for OpTraceRayKHR at offset {}.", off);
            }
            patches.push_back({off, time_value_id});
        }
    }

    if (patches.empty()) {
        return {spirv.begin(), spirv.end()};
    }

    // Check if all patches have valid time values
    for (auto &p : patches) {
        if (p.time_value_id == 0) {
            LUISA_WARNING("Some OpTraceRayKHR instructions could not be patched. "
                          "Returning unpatched SPIR-V.");
            return {spirv.begin(), spirv.end()};
        }
    }

    // Phase 4: Build the patched SPIR-V
    vstd::vector<uint32_t> result;
    result.reserve(spirv.size() + 16 + trace_ray_count);

    // Copy header (no new IDs needed)
    for (size_t i = 0; i < 5; i++) {
        result.emplace_back(spirv[i]);
    }

    vstd::unordered_map<size_t, uint32_t> trace_ray_time_map;
    for (auto &p : patches) {
        trace_ray_time_map[p.offset] = p.time_value_id;
    }

    enum class Section { CAPABILITY, EXTENSION, REST };
    auto section = Section::CAPABILITY;
    bool capability_inserted = false;
    bool extension_inserted = false;

    size_t i = 5;
    while (i < spirv.size()) {
        auto wc = spv_word_count(spirv[i]);
        auto op = spv_opcode(spirv[i]);
        if (wc == 0) break;

        if (section == Section::CAPABILITY) {
            if (op != SpvOpCapability) {
                if (!capability_inserted) {
                    result.emplace_back(spv_make_word(SpvOpCapability, 2));
                    result.emplace_back(SpvCapabilityRayTracingMotionBlurNV);
                    capability_inserted = true;
                }
                section = Section::EXTENSION;
            }
        }
        if (section == Section::EXTENSION) {
            if (op != SpvOpExtension) {
                if (!extension_inserted) {
                    auto ext_str = spv_encode_string("SPV_NV_ray_tracing_motion_blur");
                    result.emplace_back(spv_make_word(SpvOpExtension,
                                                      1 + static_cast<uint32_t>(ext_str.size())));
                    for (auto w : ext_str) result.emplace_back(w);
                    extension_inserted = true;
                }
                section = Section::REST;
            }
        }

        auto time_it = trace_ray_time_map.find(i);
        if (time_it != trace_ray_time_map.end()) {
            // Replace OpTraceRayKHR (12 words) with OpTraceRayMotionNV (13 words)
            result.emplace_back(spv_make_word(SpvOpTraceRayMotionNV, 13));
            for (uint32_t j = 1; j <= 10; j++) {
                result.emplace_back(spirv[i + j]);
            }
            result.emplace_back(time_it->second);
            result.emplace_back(spirv[i + 11]);
        } else {
            for (uint32_t j = 0; j < wc; j++) {
                result.emplace_back(spirv[i + j]);
            }
        }

        i += wc;
    }

    LUISA_INFO("SPIR-V motion blur patch: replaced {} OpTraceRayKHR -> OpTraceRayMotionNV",
               trace_ray_count);
    return result;
}

}// namespace lc::vk
