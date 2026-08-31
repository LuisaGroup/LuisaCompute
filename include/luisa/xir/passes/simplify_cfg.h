#pragma once

#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class PassReport;

class Function;

struct SimplifyCFGInfo {
    size_t folded_constant_cond_br_count = 0u;
    size_t folded_switch_count = 0u;
    size_t threaded_empty_block_count = 0u;
    size_t merged_straight_line_count = 0u;
    size_t removed_unreachable_block_count = 0u;
    // A straight-line scan consumes every currently eligible maximal chain.
    // The visit counter includes each live source recheck after a merge.
    size_t straight_line_scan_count = 0u;
    size_t straight_line_block_visit_count = 0u;
    [[nodiscard]] bool changed() const noexcept {
        return folded_constant_cond_br_count != 0u ||
               folded_switch_count != 0u ||
               threaded_empty_block_count != 0u ||
               merged_straight_line_count != 0u ||
               removed_unreachable_block_count != 0u;
    }
};

[[nodiscard]] LUISA_XIR_API SimplifyCFGInfo simplify_cfg_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API SimplifyCFGInfo simplify_cfg_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
