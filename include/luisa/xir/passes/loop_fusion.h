#pragma once

#include <luisa/xir/function.h>

namespace luisa::compute::xir {

class PassReport;

struct LoopFusionInfo {
    size_t fused_loop_count{0u};
};

[[nodiscard]] LUISA_XIR_API LoopFusionInfo loop_fusion_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API LoopFusionInfo loop_fusion_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
