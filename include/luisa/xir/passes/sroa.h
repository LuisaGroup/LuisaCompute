#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class PassReport;

struct SROAOptions {
    bool decompose_vectors{false};
    bool decompose_matrices{false};
    // Reserved for future profitability heuristics. Dynamic indices in the
    // decomposed (first) dimension are never rewritten: selecting among the
    // replacement allocas would require an explicit control-flow lowering.
    bool aggressive{false};
};

struct SROAInfo {
    size_t decomposed_alloca_count{0u};
    size_t inserted_alloca_count{0u};
};

[[nodiscard]] LUISA_XIR_API SROAInfo sroa_pass_run_on_function(Function *function, SROAOptions options = {}) noexcept;
[[nodiscard]] LUISA_XIR_API SROAInfo sroa_pass_run_on_module(Module *module, SROAOptions options = {}, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
