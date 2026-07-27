#include "instruction_layout.h"

#include <limits>

#include <luisa/core/stl/format.h>

namespace lc::spirv {

namespace {

constexpr auto max_instruction_word_count =
    static_cast<size_t>(std::numeric_limits<uint16_t>::max());

}// namespace

SpirvSwitchInstructionLayout plan_spirv_switch_instruction(
    const luisa::compute::Type *selector_type,
    size_t case_count) noexcept {
    SpirvSwitchInstructionLayout plan{.case_count = case_count};
    if (selector_type == nullptr) {
        plan.diagnostic = "SPIR-V switch selector type is null.";
        return plan;
    }
    if (selector_type->is_bool()) {
        // Logical booleans are converted to uint32 before OpSwitch emission.
        plan.selector_bit_width = 32u;
    } else if (selector_type->is_int() || selector_type->is_uint()) {
        auto size = selector_type->size();
        if (size != 1u && size != 2u && size != 4u && size != 8u) {
            plan.diagnostic = luisa::format(
                "SPIR-V switch selector '{}' has unsupported byte width {}.",
                selector_type->description(), size);
            return plan;
        }
        plan.selector_bit_width = static_cast<uint32_t>(size * 8u);
    } else {
        plan.diagnostic = luisa::format(
            "SPIR-V switch selector must be a scalar integer or bool, got '{}'.",
            selector_type->description());
        return plan;
    }

    plan.literal_word_count =
        plan.selector_bit_width == 64u ? 2u : 1u;
    auto words_per_case = static_cast<size_t>(
        plan.literal_word_count + 1u);// literal plus target ID
    constexpr auto fixed_instruction_words = size_t{3u};
    plan.max_case_count =
        (max_instruction_word_count - fixed_instruction_words) /
        words_per_case;
    if (case_count > plan.max_case_count) {
        plan.diagnostic = luisa::format(
            "SPIR-V OpSwitch with a {}-bit selector has {} cases, but its "
            "16-bit instruction word count permits at most {}.",
            plan.selector_bit_width, case_count, plan.max_case_count);
        return plan;
    }

    auto operand_words = size_t{2u} + case_count * words_per_case;
    auto instruction_words = size_t{1u} + operand_words;
    plan.operand_word_count = static_cast<uint32_t>(operand_words);
    plan.instruction_word_count = static_cast<uint32_t>(instruction_words);
    return plan;
}

SpirvPhiInstructionLayout plan_spirv_phi_instruction(
    size_t incoming_count) noexcept {
    SpirvPhiInstructionLayout plan{.incoming_count = incoming_count};
    constexpr auto fixed_instruction_words = size_t{3u};
    constexpr auto words_per_incoming = size_t{2u};
    plan.max_incoming_count =
        (max_instruction_word_count - fixed_instruction_words) /
        words_per_incoming;
    if (incoming_count > plan.max_incoming_count) {
        plan.diagnostic = luisa::format(
            "SPIR-V OpPhi has {} incoming edges, but its 16-bit instruction "
            "word count permits at most {}.",
            incoming_count, plan.max_incoming_count);
        return plan;
    }
    auto operand_words = incoming_count * words_per_incoming;
    auto instruction_words = fixed_instruction_words + operand_words;
    plan.operand_word_count = static_cast<uint32_t>(operand_words);
    plan.instruction_word_count = static_cast<uint32_t>(instruction_words);
    return plan;
}

SpirvVariadicInstructionLayout plan_spirv_variadic_instruction(
    luisa::string_view instruction_name,
    size_t fixed_word_count,
    size_t item_count,
    size_t words_per_item) noexcept {
    SpirvVariadicInstructionLayout plan{.item_count = item_count};
    if (fixed_word_count > max_instruction_word_count ||
        words_per_item == 0u) {
        plan.diagnostic = luisa::format(
            "SPIR-V {} has an invalid physical layout ({} fixed words, "
            "{} words per item).",
            instruction_name, fixed_word_count, words_per_item);
        return plan;
    }
    plan.max_item_count =
        (max_instruction_word_count - fixed_word_count) /
        words_per_item;
    if (item_count > plan.max_item_count) {
        plan.diagnostic = luisa::format(
            "SPIR-V {} has {} variable items, but its 16-bit instruction "
            "word count permits at most {}.",
            instruction_name, item_count, plan.max_item_count);
        return plan;
    }
    plan.instruction_word_count = static_cast<uint32_t>(
        fixed_word_count + item_count * words_per_item);
    return plan;
}

}// namespace lc::spirv
