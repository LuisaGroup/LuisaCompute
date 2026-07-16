#pragma once

#include <luisa/xir/function.h>

namespace luisa::compute::xir {

class PassReport;

struct LoopVectorizationInfo {
    size_t vectorized_loop_count{0u};
    size_t created_vector_inst_count{0u};
    size_t structured_cfg_error_count{0u};
    [[nodiscard]] bool succeeded() const noexcept { return structured_cfg_error_count == 0u; }
};

// Unstructured-CFG-only: structured functions are rejected without mutation.
// Plain CFG is currently accepted unchanged pending canonical-loop support.
[[nodiscard]] LUISA_XIR_API LoopVectorizationInfo loop_vectorization_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API LoopVectorizationInfo loop_vectorization_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
