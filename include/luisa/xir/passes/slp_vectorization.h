#pragma once
#include <luisa/xir/function.h>
namespace luisa::compute::xir {
class PassReport;
struct SLPVectorizationInfo {
    size_t vectorized_tree_count{0u};
    size_t vectorized_inst_count{0u};
    size_t rejected_candidate_count{0u};
    [[nodiscard]] bool changed() const noexcept {
        return vectorized_tree_count != 0u ||
               vectorized_inst_count != 0u;
    }
};
[[nodiscard]] LUISA_XIR_API SLPVectorizationInfo slp_vectorization_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API SLPVectorizationInfo slp_vectorization_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;
}// namespace luisa::compute::xir
