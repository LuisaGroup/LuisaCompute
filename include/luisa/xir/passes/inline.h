#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class PassReport;

struct InlineInfo {
    size_t inlined_call_count{0u};
    size_t removed_callable_count{0u};
};

[[nodiscard]] LUISA_XIR_API InlineInfo inline_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;
[[nodiscard]] LUISA_XIR_API InlineInfo inline_all_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
