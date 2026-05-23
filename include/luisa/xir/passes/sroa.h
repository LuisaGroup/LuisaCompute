#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

struct SROAOptions {
    bool decompose_vectors{false};
    bool decompose_matrices{false};
    bool aggressive{false};// If true, decompose structs even without all-constant indices
};

struct SROAInfo {
    size_t decomposed_alloca_count{0u};
    size_t inserted_alloca_count{0u};
};

[[nodiscard]] LUISA_XIR_API SROAInfo sroa_pass_run_on_function(Function *function, SROAOptions options = {}) noexcept;
[[nodiscard]] LUISA_XIR_API SROAInfo sroa_pass_run_on_module(Module *module, SROAOptions options = {}) noexcept;

}// namespace luisa::compute::xir
