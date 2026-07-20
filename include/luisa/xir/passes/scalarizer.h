#pragma once

#include <luisa/xir/function.h>

namespace luisa::compute::xir {

class PassReport;

struct ScalarizerInfo {
    size_t scalarized_inst_count{0u};
};

[[nodiscard]] LUISA_XIR_API ScalarizerInfo scalarizer_pass_run_on_function(FunctionDefinition *def) noexcept;
[[nodiscard]] LUISA_XIR_API ScalarizerInfo scalarizer_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
