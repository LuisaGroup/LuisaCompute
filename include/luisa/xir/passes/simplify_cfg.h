#pragma once

#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class Function;

struct SimplifyCFGInfo {
    size_t folded_constant_cond_br_count = 0u;
    size_t folded_switch_count = 0u;
    size_t threaded_empty_block_count = 0u;
    size_t merged_straight_line_count = 0u;
    size_t removed_unreachable_block_count = 0u;
};

[[nodiscard]] LUISA_XIR_API SimplifyCFGInfo simplify_cfg_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API SimplifyCFGInfo simplify_cfg_pass_run_on_module(Module *module) noexcept;

}// namespace luisa::compute::xir
