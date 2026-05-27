#pragma once

#include <luisa/xir/function.h>

namespace luisa::compute::xir {

class PassReport;

struct IndVarSimplifyInfo {
    size_t simplified_iv_count{0u};
    size_t removed_dead_iv_count{0u};
};

[[nodiscard]] LUISA_XIR_API IndVarSimplifyInfo indvar_simplify_pass_run_on_function(FunctionDefinition *def) noexcept;
[[nodiscard]] LUISA_XIR_API IndVarSimplifyInfo indvar_simplify_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
