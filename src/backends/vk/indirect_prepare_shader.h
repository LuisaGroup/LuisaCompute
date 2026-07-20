#pragma once

#include <luisa/vstl/string_builder.h>

#include "../common/indirect_dispatch_layout.h"

namespace lc::vk {

// StringBuilder treats integral operands as characters through append(char),
// so generated numeric source tokens must use its explicit integer formatter.
// Keeping the complete layout preamble here makes that source-level ABI
// directly testable without invoking DXC or creating a Vulkan device.
[[nodiscard]] inline vstd::StringBuilder
indirect_prepare_hlsl_layout_definitions() noexcept {
    vstd::StringBuilder result;
    result << "#define LC_INDIRECT_HEADER_WORDS ";
    vstd::to_string(IndirectDispatchLayout::header_word_count, result);
    result << "u\n#define LC_INDIRECT_RECORD_WORDS ";
    vstd::to_string(IndirectDispatchLayout::record_word_count, result);
    result << "u\n#define LC_INDIRECT_LOGICAL_WORD ";
    vstd::to_string(IndirectDispatchLayout::logical_size_word, result);
    result << "u\n#define LC_INDIRECT_GROUP_WORD ";
    vstd::to_string(IndirectDispatchLayout::group_count_word, result);
    result << "u\n#define LC_INDIRECT_COMMAND_WORDS ";
    vstd::to_string(
        IndirectDispatchLayout::vulkan_command_size /
            IndirectDispatchLayout::word_size,
        result);
    result << "u\n#define LC_INDIRECT_PREPARE_BLOCK_SIZE ";
    vstd::to_string(IndirectDispatchLayout::prepare_block_size, result);
    result << "u\n";
    return result;
}

}// namespace lc::vk
