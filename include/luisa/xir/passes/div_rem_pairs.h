#pragma once

#include <luisa/xir/function.h>

namespace luisa::compute::xir {

class PassReport;

struct DivRemPairsInfo {
    size_t merged_pair_count{0u};
};

[[nodiscard]] LUISA_XIR_API DivRemPairsInfo div_rem_pairs_pass_run_on_function(FunctionDefinition *def) noexcept;
[[nodiscard]] LUISA_XIR_API DivRemPairsInfo div_rem_pairs_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
