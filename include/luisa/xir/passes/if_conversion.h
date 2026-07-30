#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class PassReport;
class Function;

struct IfConversionInfo {
    size_t converted_diamond_count{0u};
    size_t hoisted_inst_count{0u};
    size_t replaced_phi_count{0u};
    size_t structured_cfg_error_count{0u};
    [[nodiscard]] bool changed() const noexcept {
        return converted_diamond_count != 0u ||
               hoisted_inst_count != 0u ||
               replaced_phi_count != 0u;
    }
    [[nodiscard]] bool succeeded() const noexcept { return structured_cfg_error_count == 0u; }
};

// Unstructured-CFG-only: structured functions are rejected without mutation.
// Metadata on the replaced parent terminator is cloned to the new branch.
// Annotated side blocks or arm-exit terminators are retained because deleting
// either arm provides no unique, verifier-valid metadata owner.
[[nodiscard]] LUISA_XIR_API IfConversionInfo if_conversion_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API IfConversionInfo if_conversion_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
