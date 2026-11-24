#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <spirv-tools/libspirv.hpp>
#include <spirv/unified1/spirv.hpp11>

namespace luisa::compute::detail {

// Workarounds for vendored SPIRV-Tools omitting `"SPV_EXT_replicated_composites"`
// and `"SPV_KHR_expect_assume"` from the extension allowlists of
// AggressiveDCEPass, LocalAccessChainConvertPass,
// LocalSingleBlockLoadStoreElimPass, and LocalSingleStoreElimPass (and lacking
// optimization support for `OpCompositeConstructReplicateEXT`).
//
// Following the same pattern as `spirv_fma_rewrite.h`, we temporarily lower
// those two extensions before `spirv-opt` so all legalization/optimization passes
// run at full effectiveness, and re-promote them immediately before the final
// `validate_and_commit_spirv_transform` validation pass.

namespace spirv_opt_ext_detail {

inline constexpr uint32_t kSpirvMagic = 0x07230203u;
inline constexpr size_t kHeaderWordCount = 5u;
inline constexpr uint32_t kMaxReplicatedConstituents = 4096u;

inline constexpr std::string_view kExtReplicatedComposites = "SPV_EXT_replicated_composites";
inline constexpr std::string_view kExtExpectAssume = "SPV_KHR_expect_assume";

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

inline void append_capability(std::vector<uint32_t> &out,
                              spv::Capability cap) {
    out.push_back((2u << 16u) | static_cast<uint32_t>(spv::Op::OpCapability));
    out.push_back(static_cast<uint32_t>(cap));
}

inline void append_extension(std::vector<uint32_t> &out,
                             std::string_view ext) {
    const size_t str_words = (ext.size() + 1u + 3u) / 4u;
    const uint32_t ext_wc = static_cast<uint32_t>(1u + str_words);
    out.push_back((ext_wc << 16u) | static_cast<uint32_t>(spv::Op::OpExtension));
    append_literal_string(out, ext);
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

struct CompositeTypeShape {
    uint32_t count{0u};
    bool uniform{false};
};

struct ReplicatedTypeTable {
    struct IntTypeInfo {
        uint32_t width{0u};
        bool is_signed{false};
    };
    std::unordered_map<uint32_t, IntTypeInfo> int_types;
    std::unordered_map<uint32_t, uint64_t> int_constants;
    std::unordered_map<uint32_t, CompositeTypeShape> composites;
    std::unordered_set<uint32_t> coop_vector_types;
    bool has_ext{false};
    bool has_cap{false};
    bool has_const_replicate{false};
    bool has_spec_replicate{false};
    bool has_construct_replicate{false};
    bool saw_invalid_or_unsupported_replicate{false};

    void scan(const std::vector<uint32_t> &words) {
        for (size_t i = kHeaderWordCount; i < words.size();) {
            const uint32_t wc = words[i] >> 16u;
            const auto op = static_cast<spv::Op>(words[i] & 0xffffu);
            if (op == spv::Op::OpExtension && wc >= 2u &&
                literal_string_equals(words.data() + i + 1u, wc - 1u,
                                      kExtReplicatedComposites)) {
                has_ext = true;
            } else if (op == spv::Op::OpCapability && wc == 2u &&
                       words[i + 1u] ==
                           static_cast<uint32_t>(spv::Capability::ReplicatedCompositesEXT)) {
                has_cap = true;
            } else if (op == spv::Op::OpTypeInt && wc == 4u) {
                int_types[words[i + 1u]] = {words[i + 2u], words[i + 3u] != 0u};
            } else if (op == spv::Op::OpConstant && (wc == 4u || wc == 5u)) {
                if (auto it = int_types.find(words[i + 1u]); it != int_types.end()) {
                    const uint32_t width = it->second.width;
                    const bool is_signed = it->second.is_signed;
                    if (wc == 4u && width > 0u && width <= 32u) {
                        uint64_t val = words[i + 3u];
                        if (is_signed) {
                            const uint64_t sign_bit = uint64_t{1u} << (width - 1u);
                            if ((val & sign_bit) == 0u) {
                                int_constants[words[i + 2u]] = val & ((uint64_t{1u} << width) - 1u);
                            }
                        } else {
                            const uint64_t mask =
                                width == 32u ? 0xffffffffull : ((uint64_t{1u} << width) - 1u);
                            int_constants[words[i + 2u]] = val & mask;
                        }
                    } else if (wc == 5u && width == 64u) {
                        uint64_t val = static_cast<uint64_t>(words[i + 3u]) |
                                       (static_cast<uint64_t>(words[i + 4u]) << 32u);
                        if (!is_signed || static_cast<int64_t>(val) >= 0) {
                            int_constants[words[i + 2u]] = val;
                        }
                    }
                }
            } else if (op == spv::Op::OpTypeVector && wc == 4u) {
                composites[words[i + 1u]] = {.count = words[i + 3u], .uniform = true};
            } else if (op == spv::Op::OpTypeMatrix && wc == 4u) {
                composites[words[i + 1u]] = {.count = words[i + 3u], .uniform = true};
            } else if (op == spv::Op::OpTypeArray && wc == 4u) {
                if (auto it = int_constants.find(words[i + 3u]);
                    it != int_constants.end() && it->second >= 1u &&
                    it->second <= kMaxReplicatedConstituents) {
                    composites[words[i + 1u]] = {
                        .count = static_cast<uint32_t>(it->second),
                        .uniform = true};
                }
            } else if (op == spv::Op::OpTypeStruct && wc >= 3u) {
                const uint32_t member_count = wc - 2u;
                const uint32_t first_member = words[i + 2u];
                bool all_same = member_count <= kMaxReplicatedConstituents;
                for (uint32_t m = 1u; m < member_count && all_same; ++m) {
                    if (words[i + 2u + m] != first_member) { all_same = false; }
                }
                if (all_same) {
                    composites[words[i + 1u]] = {.count = member_count, .uniform = true};
                }
            } else if (op == spv::Op::OpTypeCooperativeVectorNV && wc >= 2u) {
                coop_vector_types.insert(words[i + 1u]);
            } else if (op == spv::Op::OpSpecConstantCompositeReplicateEXT) {
                has_spec_replicate = true;
                saw_invalid_or_unsupported_replicate = true;
            } else if (op == spv::Op::OpConstantCompositeReplicateEXT ||
                       op == spv::Op::OpCompositeConstructReplicateEXT) {
                if (op == spv::Op::OpConstantCompositeReplicateEXT) {
                    has_const_replicate = true;
                } else {
                    has_construct_replicate = true;
                }
                if (wc != 4u) {
                    saw_invalid_or_unsupported_replicate = true;
                } else {
                    const uint32_t res_type = words[i + 1u];
                    auto it = composites.find(res_type);
                    if (coop_vector_types.contains(res_type) ||
                        it == composites.end() || !it->second.uniform ||
                        it->second.count < 1u ||
                        it->second.count > kMaxReplicatedConstituents) {
                        saw_invalid_or_unsupported_replicate = true;
                    }
                }
            }
            i += wc;
        }
    }
};

}// namespace spirv_opt_ext_detail

// Demotes `OpConstantCompositeReplicateEXT` -> `OpConstantComposite` and
// `OpCompositeConstructReplicateEXT` -> `OpCompositeConstruct`, and removes
// `OpCapability ReplicatedCompositesEXT` and
// `OpExtension "SPV_EXT_replicated_composites"` before `spirv-opt`.
// Returns true if the module declared/used replicated composites and was
// safely rewritten. Leaves cooperative-vector or spec-constant uses untouched
// (conservatively returning false).
[[nodiscard]] inline bool demote_replicated_composites_for_spirv_opt(
    std::vector<uint32_t> &words) {
    using namespace spirv_opt_ext_detail;
    if (!validate_word_stream(words.data(), words.size())) { return false; }

    ReplicatedTypeTable table;
    table.scan(words);

    if (!table.has_ext && !table.has_cap &&
        !table.has_const_replicate && !table.has_spec_replicate &&
        !table.has_construct_replicate) {
        return false;
    }
    if (table.saw_invalid_or_unsupported_replicate || !table.coop_vector_types.empty()) {
        return false;
    }

    std::vector<uint32_t> out;
    out.reserve(words.size() + 32u);
    out.insert(out.end(), words.begin(), words.begin() + kHeaderWordCount);

    for (size_t i = kHeaderWordCount; i < words.size();) {
        const uint32_t wc = words[i] >> 16u;
        const auto op = static_cast<spv::Op>(words[i] & 0xffffu);

        if (op == spv::Op::OpExtension && wc >= 2u &&
            literal_string_equals(words.data() + i + 1u, wc - 1u,
                                  kExtReplicatedComposites)) {
            i += wc;
            continue;
        }
        if (op == spv::Op::OpCapability && wc == 2u &&
            words[i + 1u] == static_cast<uint32_t>(spv::Capability::ReplicatedCompositesEXT)) {
            i += wc;
            continue;
        }
        if (op == spv::Op::OpConstantCompositeReplicateEXT ||
            op == spv::Op::OpCompositeConstructReplicateEXT) {
            const uint32_t res_type = words[i + 1u];
            const uint32_t res_id = words[i + 2u];
            const uint32_t val_id = words[i + 3u];
            const uint32_t count = table.composites.at(res_type).count;
            const auto new_op =
                op == spv::Op::OpConstantCompositeReplicateEXT ?
                    spv::Op::OpConstantComposite :
                    spv::Op::OpCompositeConstruct;
            const uint32_t new_wc = 3u + count;
            out.push_back((new_wc << 16u) | static_cast<uint32_t>(new_op));
            out.push_back(res_type);
            out.push_back(res_id);
            for (uint32_t k = 0u; k < count; ++k) {
                out.push_back(val_id);
            }
            i += wc;
            continue;
        }

        out.insert(out.end(), words.begin() + i, words.begin() + i + wc);
        i += wc;
    }

    words.swap(out);
    return true;
}

// Re-promotes splatted `OpConstantComposite` and `OpCompositeConstruct`
// instructions (with >= 2 identical constituents matching the full composite
// element/member count) to `OpConstantCompositeReplicateEXT` /
// `OpCompositeConstructReplicateEXT`, and ensures `OpCapability
// ReplicatedCompositesEXT` and `OpExtension "SPV_EXT_replicated_composites"`
// are declared only when at least one instruction is promoted (or already uses
// the extension). Returns the number of newly promoted instructions.
inline size_t promote_replicated_composites_after_spirv_opt(
    std::vector<uint32_t> &words) {
    using namespace spirv_opt_ext_detail;
    if (!validate_word_stream(words.data(), words.size())) { return 0u; }

    ReplicatedTypeTable table;
    table.scan(words);

    auto can_replicate = [&](size_t i, uint32_t wc, spv::Op op) noexcept -> bool {
        if (op != spv::Op::OpConstantComposite &&
            op != spv::Op::OpCompositeConstruct) {
            return false;
        }
        if (wc < 5u) { return false; }// Need at least 2 constituents (3 + 2).
        const uint32_t res_type = words[i + 1u];
        if (table.coop_vector_types.contains(res_type)) { return false; }
        auto it = table.composites.find(res_type);
        if (it == table.composites.end() || !it->second.uniform) { return false; }
        const uint32_t num_constituents = wc - 3u;
        // Exact count check also prevents mis-promoting sub-vector constructs
        // such as `OpCompositeConstruct %v4float %v2 %v2`.
        if (num_constituents != it->second.count || num_constituents < 2u) {
            return false;
        }
        const uint32_t first = words[i + 3u];
        for (uint32_t k = 1u; k < num_constituents; ++k) {
            if (words[i + 3u + k] != first) { return false; }
        }
        return true;
    };

    size_t promotable_count = 0u;
    for (size_t i = kHeaderWordCount; i < words.size();) {
        const uint32_t wc = words[i] >> 16u;
        const auto op = static_cast<spv::Op>(words[i] & 0xffffu);
        if (can_replicate(i, wc, op)) {
            ++promotable_count;
        }
        i += wc;
    }

    if (promotable_count == 0u) { return 0u; }

    std::vector<uint32_t> out;
    out.reserve(words.size() + 16u);
    out.insert(out.end(), words.begin(), words.begin() + kHeaderWordCount);

    bool cap_inserted = table.has_cap;
    bool ext_inserted = table.has_ext;

    for (size_t i = kHeaderWordCount; i < words.size();) {
        const uint32_t wc = words[i] >> 16u;
        const auto op = static_cast<spv::Op>(words[i] & 0xffffu);

        if (!cap_inserted && op != spv::Op::OpCapability) {
            append_capability(out, spv::Capability::ReplicatedCompositesEXT);
            cap_inserted = true;
        }
        if (!ext_inserted && op != spv::Op::OpCapability &&
            op != spv::Op::OpExtension) {
            append_extension(out, kExtReplicatedComposites);
            ext_inserted = true;
        }

        if (can_replicate(i, wc, op)) {
            const auto new_op =
                op == spv::Op::OpConstantComposite ?
                    spv::Op::OpConstantCompositeReplicateEXT :
                    spv::Op::OpCompositeConstructReplicateEXT;
            out.push_back((4u << 16u) | static_cast<uint32_t>(new_op));
            out.push_back(words[i + 1u]);
            out.push_back(words[i + 2u]);
            out.push_back(words[i + 3u]);
            i += wc;
            continue;
        }

        out.insert(out.end(), words.begin() + i, words.begin() + i + wc);
        i += wc;
    }

    words.swap(out);
    return promotable_count;
}

struct DemotedExpectAssumeGuard {
    uint32_t entry_func_ordinal{0u};
    bool expected_value{false};
};

// Demotes `OpExpectKHR` on each entry point's dispatch-bounds early-return
// branch (`OpBranchConditional %exp %pure_return_block %body_block`) and
// strips `OpCapability ExpectAssumeKHR` / `OpExtension "SPV_KHR_expect_assume"`
// before `spirv-opt`. Only rewrites if every `OpExpectKHR` in the module matches
// this exact single-use entry-return guard shape and no `OpAssumeTrueKHR` is
// present. Otherwise leaves `words` unchanged and returns false.
[[nodiscard]] inline bool demote_expect_assume_for_spirv_opt(
    std::vector<uint32_t> &words,
    std::vector<DemotedExpectAssumeGuard> &out_guards) {
    using namespace spirv_opt_ext_detail;
    out_guards.clear();
    if (!validate_word_stream(words.data(), words.size())) { return false; }

    struct ExpectInfo {
        size_t offset{0u};
        uint32_t func_id{0u};
        uint32_t type_id{0u};
        uint32_t result_id{0u};
        uint32_t value_id{0u};
        uint32_t expected_id{0u};
    };
    struct BranchInfo {
        size_t offset{0u};
        uint32_t func_id{0u};
        uint32_t cond_id{0u};
        uint32_t true_lbl{0u};
    };
    struct ParseState {
        bool has_ext{false};
        bool has_cap{false};
        bool has_assume_true{false};
        bool duplicate_entry_func{false};
        uint32_t bool_type_id{0u};
        bool multiple_bool_types{false};
        uint32_t const_false_id{0u};
        uint32_t const_true_id{0u};
        std::vector<uint32_t> entry_funcs;
        std::unordered_set<uint32_t> entry_func_set;
        std::unordered_set<uint32_t> decorated_ids;
        std::unordered_map<uint32_t, uint32_t> use_counts;
        std::unordered_set<uint32_t> pure_return_blocks;
        std::vector<ExpectInfo> expects;
        std::vector<BranchInfo> cond_branches;

        const uint32_t *base{nullptr};
        uint32_t current_func{0u};
        uint32_t current_block{0u};
        bool block_pure_ret{false};
        bool block_saw_return{false};

        void finish_block() {
            if (current_block != 0u && block_pure_ret && block_saw_return) {
                pure_return_blocks.insert(current_block);
            }
            current_block = 0u;
            block_pure_ret = false;
            block_saw_return = false;
        }
    } state;
    state.base = words.data();

    auto callback = [](void *user_data,
                       const spv_parsed_instruction_t *inst) -> spv_result_t {
        auto &s = *static_cast<ParseState *>(user_data);
        const auto op = static_cast<spv::Op>(inst->opcode);
        const size_t offset = static_cast<size_t>(inst->words - s.base);

        // Count semantic ID uses across all instructions (excluding OpName /
        // OpMemberName).
        if (op != spv::Op::OpName && op != spv::Op::OpMemberName) {
            for (uint16_t k = 0u; k < inst->num_operands; ++k) {
                const auto &oper = inst->operands[k];
                if (oper.type == SPV_OPERAND_TYPE_ID ||
                    oper.type == SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID ||
                    oper.type == SPV_OPERAND_TYPE_SCOPE_ID) {
                    const uint32_t used_id = inst->words[oper.offset];
                    s.use_counts[used_id] += 1u;
                }
            }
        }

        if (op == spv::Op::OpDecorate || op == spv::Op::OpMemberDecorate ||
            op == spv::Op::OpDecorateId || op == spv::Op::OpDecorateString ||
            op == spv::Op::OpMemberDecorateString) {
            if (inst->num_words >= 2u) {
                s.decorated_ids.insert(inst->words[1u]);
            }
        } else if (op == spv::Op::OpGroupDecorate) {
            for (uint16_t w = 2u; w < inst->num_words; ++w) {
                s.decorated_ids.insert(inst->words[w]);
            }
        } else if (op == spv::Op::OpExtension && inst->num_words >= 2u) {
            if (literal_string_equals(inst->words + 1u, inst->num_words - 1u,
                                      kExtExpectAssume)) {
                s.has_ext = true;
            }
        } else if (op == spv::Op::OpCapability && inst->num_words == 2u) {
            if (inst->words[1u] ==
                static_cast<uint32_t>(spv::Capability::ExpectAssumeKHR)) {
                s.has_cap = true;
            }
        } else if (op == spv::Op::OpAssumeTrueKHR) {
            s.has_assume_true = true;
        } else if (op == spv::Op::OpEntryPoint && inst->num_words >= 3u) {
            const uint32_t fn = inst->words[2u];
            if (!s.entry_func_set.insert(fn).second) {
                s.duplicate_entry_func = true;
            }
            s.entry_funcs.push_back(fn);
        } else if (op == spv::Op::OpTypeBool && inst->num_words == 2u) {
            if (s.bool_type_id != 0u && s.bool_type_id != inst->result_id) {
                s.multiple_bool_types = true;
            } else {
                s.bool_type_id = inst->result_id;
            }
        } else if (op == spv::Op::OpConstantFalse && inst->num_words == 3u) {
            if (inst->type_id == s.bool_type_id && s.const_false_id == 0u) {
                s.const_false_id = inst->result_id;
            }
        } else if (op == spv::Op::OpConstantTrue && inst->num_words == 3u) {
            if (inst->type_id == s.bool_type_id && s.const_true_id == 0u) {
                s.const_true_id = inst->result_id;
            }
        } else if (op == spv::Op::OpFunction) {
            s.finish_block();
            s.current_func = inst->result_id;
        } else if (op == spv::Op::OpFunctionEnd) {
            s.finish_block();
            s.current_func = 0u;
        } else if (op == spv::Op::OpLabel) {
            s.finish_block();
            s.current_block = inst->result_id;
            s.block_pure_ret = true;
            s.block_saw_return = false;
        } else if (s.current_block != 0u) {
            if (op == spv::Op::OpReturn && inst->num_words == 1u && !s.block_saw_return) {
                s.block_saw_return = true;
            } else if (op != spv::Op::OpLine && op != spv::Op::OpNoLine) {
                s.block_pure_ret = false;
            }
        }

        if (op == spv::Op::OpExpectKHR) {
            if (inst->num_words == 5u && s.current_func != 0u) {
                s.expects.push_back({
                    .offset = offset,
                    .func_id = s.current_func,
                    .type_id = inst->type_id,
                    .result_id = inst->result_id,
                    .value_id = inst->words[3u],
                    .expected_id = inst->words[4u]});
            } else {
                s.expects.push_back({.offset = offset, .func_id = 0u});
            }
        } else if (op == spv::Op::OpBranchConditional && inst->num_words >= 4u &&
                   s.current_func != 0u) {
            s.cond_branches.push_back({
                .offset = offset,
                .func_id = s.current_func,
                .cond_id = inst->words[1u],
                .true_lbl = inst->words[2u]});
        }
        return SPV_SUCCESS;
    };

    spvtools::Context ctx{SPV_ENV_VULKAN_1_2};
    if (spvBinaryParse(ctx.CContext(), &state, words.data(), words.size(),
                       nullptr, callback, nullptr) != SPV_SUCCESS) {
        return false;
    }

    if (state.has_assume_true || state.duplicate_entry_func || state.multiple_bool_types) {
        return false;
    }
    if (!state.has_ext && !state.has_cap && state.expects.empty()) {
        return false;
    }
    if (state.expects.empty() || state.bool_type_id == 0u) {
        return false;
    }

    std::unordered_map<uint32_t, uint32_t> func_to_ordinal;
    for (uint32_t ord = 0u; ord < state.entry_funcs.size(); ++ord) {
        func_to_ordinal[state.entry_funcs[ord]] = ord;
    }

    std::unordered_map<uint32_t, const BranchInfo *> first_ret_branch;
    for (const auto &br : state.cond_branches) {
        if (func_to_ordinal.contains(br.func_id) &&
            state.pure_return_blocks.contains(br.true_lbl) &&
            !first_ret_branch.contains(br.func_id)) {
            first_ret_branch[br.func_id] = &br;
        }
    }

    struct MatchedExpect {
        size_t expect_offset;
        size_t branch_offset;
        uint32_t result_id;
        uint32_t value_id;
        uint32_t entry_ordinal;
        bool expected_value;
    };
    std::vector<MatchedExpect> matched;
    std::unordered_set<uint32_t> seen_funcs;
    std::unordered_map<size_t, uint32_t> branch_cond_replacement;
    std::unordered_set<size_t> dropped_expect_offsets;
    std::unordered_set<uint32_t> dropped_result_ids;

    for (const auto &exp : state.expects) {
        if (exp.func_id == 0u || exp.type_id != state.bool_type_id) {
            return false;
        }
        auto ord_it = func_to_ordinal.find(exp.func_id);
        if (ord_it == func_to_ordinal.end()) { return false; }
        if (!seen_funcs.insert(exp.func_id).second) { return false; }
        bool expected_val = false;
        if (exp.expected_id != 0u && exp.expected_id == state.const_false_id) {
            expected_val = false;
        } else if (exp.expected_id != 0u && exp.expected_id == state.const_true_id) {
            expected_val = true;
        } else {
            return false;
        }
        if (state.decorated_ids.contains(exp.result_id)) { return false; }
        auto use_it = state.use_counts.find(exp.result_id);
        if (use_it == state.use_counts.end() || use_it->second != 1u) {
            return false;
        }
        auto br_it = first_ret_branch.find(exp.func_id);
        if (br_it == first_ret_branch.end() ||
            br_it->second->cond_id != exp.result_id) {
            return false;
        }
        matched.push_back({
            .expect_offset = exp.offset,
            .branch_offset = br_it->second->offset,
            .result_id = exp.result_id,
            .value_id = exp.value_id,
            .entry_ordinal = ord_it->second,
            .expected_value = expected_val});
        branch_cond_replacement[br_it->second->offset] = exp.value_id;
        dropped_expect_offsets.insert(exp.offset);
        dropped_result_ids.insert(exp.result_id);
    }

    std::vector<uint32_t> out;
    out.reserve(words.size());
    out.insert(out.end(), words.begin(), words.begin() + kHeaderWordCount);

    for (size_t i = kHeaderWordCount; i < words.size();) {
        const uint32_t wc = words[i] >> 16u;
        const auto op = static_cast<spv::Op>(words[i] & 0xffffu);

        if (op == spv::Op::OpExtension && wc >= 2u &&
            literal_string_equals(words.data() + i + 1u, wc - 1u,
                                  kExtExpectAssume)) {
            i += wc;
            continue;
        }
        if (op == spv::Op::OpCapability && wc == 2u &&
            words[i + 1u] == static_cast<uint32_t>(spv::Capability::ExpectAssumeKHR)) {
            i += wc;
            continue;
        }
        if (op == spv::Op::OpName && wc >= 2u &&
            dropped_result_ids.contains(words[i + 1u])) {
            i += wc;
            continue;
        }
        if (dropped_expect_offsets.contains(i)) {
            i += wc;
            continue;
        }
        if (auto rep_it = branch_cond_replacement.find(i);
            rep_it != branch_cond_replacement.end()) {
            out.push_back(words[i]);
            out.push_back(rep_it->second);
            out.insert(out.end(), words.begin() + i + 2u, words.begin() + i + wc);
            i += wc;
            continue;
        }

        out.insert(out.end(), words.begin() + i, words.begin() + i + wc);
        i += wc;
    }

    words.swap(out);
    out_guards.reserve(matched.size());
    for (const auto &m : matched) {
        out_guards.push_back({
            .entry_func_ordinal = m.entry_ordinal,
            .expected_value = m.expected_value});
    }
    return true;
}

// Re-inserts `OpExpectKHR` on the matching entry-point early-return branches
// after `spirv-opt`, along with `OpCapability ExpectAssumeKHR` and
// `OpExtension "SPV_KHR_expect_assume"`. Always places `OpExpectKHR` BEFORE
// any `OpSelectionMerge` immediately preceding the conditional branch so
// SPIR-V structured control-flow validation holds.
inline size_t promote_expect_assume_after_spirv_opt(
    std::vector<uint32_t> &words,
    const std::vector<DemotedExpectAssumeGuard> &guards) {
    using namespace spirv_opt_ext_detail;
    if (guards.empty() || !validate_word_stream(words.data(), words.size())) {
        return 0u;
    }

    bool has_ext = false;
    bool has_cap = false;
    uint32_t bool_type_id = 0u;
    bool multiple_bool_types = false;
    size_t bool_type_offset = 0u;
    uint32_t const_false_id = 0u;
    uint32_t const_true_id = 0u;
    std::unordered_set<uint32_t> const_bool_ids;
    std::vector<uint32_t> entry_funcs;
    std::unordered_set<uint32_t> entry_func_set;
    bool duplicate_entry_func = false;
    std::unordered_set<uint32_t> pure_return_blocks;

    struct CandidateBranch {
        uint32_t func_id{0u};
        uint32_t cond_id{0u};
        size_t insert_offset{0u};
        size_t branch_offset{0u};
    };
    std::vector<CandidateBranch> first_ret_branches;
    std::unordered_set<uint32_t> seen_branch_funcs;

    uint32_t cur_func = 0u;
    uint32_t cur_block = 0u;
    bool block_pure_ret = false;
    bool block_saw_ret = false;
    spv::Op last_non_line_op = spv::Op::OpNop;
    size_t last_non_line_offset = 0u;

    auto finish_block = [&]() {
        if (cur_block != 0u && block_pure_ret && block_saw_ret) {
            pure_return_blocks.insert(cur_block);
        }
        cur_block = 0u;
        block_pure_ret = false;
        block_saw_ret = false;
        last_non_line_op = spv::Op::OpNop;
        last_non_line_offset = 0u;
    };

    // Pass 1: discover OpTypeBool, constants, pure OpReturn blocks, and the
    // first OpBranchConditional targeting a pure OpReturn block in each entry
    // point.
    for (size_t i = kHeaderWordCount; i < words.size();) {
        const uint32_t wc = words[i] >> 16u;
        const auto op = static_cast<spv::Op>(words[i] & 0xffffu);

        if (op == spv::Op::OpExtension && wc >= 2u &&
            literal_string_equals(words.data() + i + 1u, wc - 1u,
                                  kExtExpectAssume)) {
            has_ext = true;
        } else if (op == spv::Op::OpCapability && wc == 2u &&
                   words[i + 1u] ==
                       static_cast<uint32_t>(spv::Capability::ExpectAssumeKHR)) {
            has_cap = true;
        } else if (op == spv::Op::OpEntryPoint && wc >= 3u) {
            const uint32_t fn = words[i + 2u];
            if (!entry_func_set.insert(fn).second) { duplicate_entry_func = true; }
            entry_funcs.push_back(fn);
        } else if (op == spv::Op::OpTypeBool && wc == 2u) {
            if (bool_type_id != 0u && bool_type_id != words[i + 1u]) {
                multiple_bool_types = true;
            } else {
                bool_type_id = words[i + 1u];
                bool_type_offset = i;
            }
        } else if (op == spv::Op::OpConstantFalse && wc == 3u) {
            const_bool_ids.insert(words[i + 2u]);
            if (words[i + 1u] == bool_type_id && const_false_id == 0u) {
                const_false_id = words[i + 2u];
            }
        } else if (op == spv::Op::OpConstantTrue && wc == 3u) {
            const_bool_ids.insert(words[i + 2u]);
            if (words[i + 1u] == bool_type_id && const_true_id == 0u) {
                const_true_id = words[i + 2u];
            }
        } else if ((op == spv::Op::OpSpecConstantFalse ||
                    op == spv::Op::OpSpecConstantTrue) &&
                   wc == 3u) {
            const_bool_ids.insert(words[i + 2u]);
        } else if (op == spv::Op::OpFunction && wc >= 3u) {
            finish_block();
            cur_func = words[i + 2u];
        } else if (op == spv::Op::OpFunctionEnd) {
            finish_block();
            cur_func = 0u;
        } else if (op == spv::Op::OpLabel && wc >= 2u) {
            finish_block();
            cur_block = words[i + 1u];
            block_pure_ret = true;
            block_saw_ret = false;
            last_non_line_op = spv::Op::OpLabel;
            last_non_line_offset = i;
        } else if (cur_block != 0u) {
            if (op == spv::Op::OpReturn && wc == 1u && !block_saw_ret) {
                block_saw_ret = true;
            } else if (op != spv::Op::OpLine && op != spv::Op::OpNoLine) {
                block_pure_ret = false;
            }
            if (op == spv::Op::OpBranchConditional && wc >= 4u &&
                cur_func != 0u) {
                const size_t insert_off =
                    last_non_line_op == spv::Op::OpSelectionMerge ?
                        last_non_line_offset :
                        i;
                first_ret_branches.push_back({
                    .func_id = cur_func,
                    .cond_id = words[i + 1u],
                    .insert_offset = insert_off,
                    .branch_offset = i});
                // Store true label transiently using seen_branch_funcs below?
                // Wait until Pass 1 completes to know all pure_return_blocks!
            }
            if (op != spv::Op::OpLine && op != spv::Op::OpNoLine) {
                last_non_line_op = op;
                last_non_line_offset = i;
            }
        }
        i += wc;
    }
    finish_block();

    if (bool_type_id == 0u || multiple_bool_types || duplicate_entry_func) {
        return 0u;
    }

    // Filter `first_ret_branches` now that `pure_return_blocks` is complete.
    std::unordered_map<uint32_t, CandidateBranch> first_ret_by_func;
    for (const auto &br : first_ret_branches) {
        const uint32_t true_lbl = words[br.branch_offset + 2u];
        if (!entry_func_set.contains(br.func_id) ||
            !pure_return_blocks.contains(true_lbl) ||
            !seen_branch_funcs.insert(br.func_id).second) {
            continue;
        }
        first_ret_by_func[br.func_id] = br;
    }

    uint32_t new_bound = words[3u];
    bool need_false = false;
    bool need_true = false;

    struct Insertion {
        uint32_t cond_id{0u};
        uint32_t new_exp_id{0u};
        uint32_t expected_const_id{0u};
        size_t branch_offset{0u};
        bool expected_value{false};
    };
    std::unordered_map<size_t, Insertion> insert_before;
    std::unordered_map<size_t, uint32_t> rewrite_branch;
    size_t promoted_count = 0u;

    for (const auto &guard : guards) {
        if (guard.entry_func_ordinal >= entry_funcs.size()) { continue; }
        const uint32_t fn = entry_funcs[guard.entry_func_ordinal];
        auto br_it = first_ret_by_func.find(fn);
        if (br_it == first_ret_by_func.end()) { continue; }
        const auto &br = br_it->second;
        if (br.cond_id == 0u || const_bool_ids.contains(br.cond_id)) { continue; }
        if (insert_before.contains(br.insert_offset)) { continue; }
        if (guard.expected_value) {
            if (const_true_id == 0u) {
                const_true_id = new_bound++;
                need_true = true;
            }
        } else {
            if (const_false_id == 0u) {
                const_false_id = new_bound++;
                need_false = true;
            }
        }
        const uint32_t exp_id = new_bound++;
        insert_before[br.insert_offset] = Insertion{
            .cond_id = br.cond_id,
            .new_exp_id = exp_id,
            .expected_const_id =
                guard.expected_value ? const_true_id : const_false_id,
            .branch_offset = br.branch_offset,
            .expected_value = guard.expected_value};
        rewrite_branch[br.branch_offset] = exp_id;
        ++promoted_count;
    }

    if (promoted_count == 0u) { return 0u; }

    std::vector<uint32_t> out;
    out.reserve(words.size() + 16u + promoted_count * 5u);
    out.insert(out.end(), words.begin(), words.begin() + kHeaderWordCount);
    out[3u] = new_bound;

    bool cap_inserted = has_cap;
    bool ext_inserted = has_ext;

    for (size_t i = kHeaderWordCount; i < words.size();) {
        const uint32_t wc = words[i] >> 16u;
        const auto op = static_cast<spv::Op>(words[i] & 0xffffu);

        if (!cap_inserted && op != spv::Op::OpCapability) {
            append_capability(out, spv::Capability::ExpectAssumeKHR);
            cap_inserted = true;
        }
        if (!ext_inserted && op != spv::Op::OpCapability &&
            op != spv::Op::OpExtension) {
            append_extension(out, kExtExpectAssume);
            ext_inserted = true;
        }

        if (auto ins_it = insert_before.find(i); ins_it != insert_before.end()) {
            out.push_back((5u << 16u) | static_cast<uint32_t>(spv::Op::OpExpectKHR));
            out.push_back(bool_type_id);
            out.push_back(ins_it->second.new_exp_id);
            out.push_back(ins_it->second.cond_id);
            out.push_back(ins_it->second.expected_const_id);
        }

        if (auto br_it = rewrite_branch.find(i); br_it != rewrite_branch.end()) {
            out.push_back(words[i]);
            out.push_back(br_it->second);
            out.insert(out.end(), words.begin() + i + 2u, words.begin() + i + wc);
        } else {
            out.insert(out.end(), words.begin() + i, words.begin() + i + wc);
        }

        if (i == bool_type_offset) {
            if (need_false) {
                out.push_back((3u << 16u) |
                              static_cast<uint32_t>(spv::Op::OpConstantFalse));
                out.push_back(bool_type_id);
                out.push_back(const_false_id);
            }
            if (need_true) {
                out.push_back((3u << 16u) |
                              static_cast<uint32_t>(spv::Op::OpConstantTrue));
                out.push_back(bool_type_id);
                out.push_back(const_true_id);
            }
        }

        i += wc;
    }

    words.swap(out);
    return promoted_count;
}

}// namespace luisa::compute::detail
