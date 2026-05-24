#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class PassReport;

class Module;
class Function;

struct DestructureCFGInfo {
    size_t destructured_if_count{0u};
    size_t destructured_loop_count{0u};
    size_t destructured_simple_loop_count{0u};
    size_t destructured_break_count{0u};
    size_t destructured_continue_count{0u};
    size_t destructured_early_return_count{0u};
    size_t leaked_block_count{0u};
};

[[nodiscard]] LUISA_XIR_API DestructureCFGInfo destructure_cfg_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API DestructureCFGInfo destructure_cfg_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
