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
};

[[nodiscard]] LUISA_XIR_API IfConversionInfo if_conversion_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API IfConversionInfo if_conversion_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
