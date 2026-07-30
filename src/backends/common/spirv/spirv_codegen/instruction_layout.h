#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/ast/type.h>
#include <luisa/core/stl/string.h>

namespace lc::spirv {

struct SpirvSwitchInstructionLayout {
    uint32_t selector_bit_width{0u};
    uint32_t literal_word_count{0u};
    size_t case_count{0u};
    size_t max_case_count{0u};
    uint32_t operand_word_count{0u};
    uint32_t instruction_word_count{0u};
    luisa::string diagnostic;

    [[nodiscard]] bool succeeded() const noexcept {
        return diagnostic.empty();
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return succeeded();
    }
};

struct SpirvPhiInstructionLayout {
    size_t incoming_count{0u};
    size_t max_incoming_count{0u};
    uint32_t operand_word_count{0u};
    uint32_t instruction_word_count{0u};
    luisa::string diagnostic;

    [[nodiscard]] bool succeeded() const noexcept {
        return diagnostic.empty();
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return succeeded();
    }
};

struct SpirvVariadicInstructionLayout {
    size_t item_count{0u};
    size_t max_item_count{0u};
    uint32_t instruction_word_count{0u};
    luisa::string diagnostic;

    [[nodiscard]] bool succeeded() const noexcept {
        return diagnostic.empty();
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return succeeded();
    }
};

// Plans physical instruction encodings before emission. SPIR-V stores every
// instruction's total word count in 16 bits. OpSwitch also uses two literal
// words per case for a 64-bit selector, while OpPhi uses a value/block pair for
// each incoming edge. These contracts reject unencodable IR before any
// intermediate size is narrowed.
[[nodiscard]] SpirvSwitchInstructionLayout
plan_spirv_switch_instruction(
    const luisa::compute::Type *selector_type,
    size_t case_count) noexcept;

[[nodiscard]] SpirvPhiInstructionLayout
plan_spirv_phi_instruction(size_t incoming_count) noexcept;

// Plans an instruction with a fixed prefix and a repeated, fixed-width item
// sequence (for example, members of OpTypeStruct or constituents of
// OpCompositeConstruct). `fixed_word_count` includes the instruction header.
[[nodiscard]] SpirvVariadicInstructionLayout
plan_spirv_variadic_instruction(
    luisa::string_view instruction_name,
    size_t fixed_word_count,
    size_t item_count,
    size_t words_per_item = 1u) noexcept;

}// namespace lc::spirv
