#pragma once

#include <luisa/core/dll_export.h>

namespace luisa::compute::xir {

class PassReport;
class Function;
class Module;

struct FixSelfReferentialInfo {
    size_t fixed_count = 0u;
    size_t unresolved_count = 0u;
    [[nodiscard]] bool succeeded() const noexcept { return unresolved_count == 0u; }
};

[[nodiscard]] LUISA_XIR_API FixSelfReferentialInfo fix_self_referential_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API FixSelfReferentialInfo fix_self_referential_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
