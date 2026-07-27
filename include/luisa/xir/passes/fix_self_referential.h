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

// The function and module entry points are transactional: every candidate is
// validated before any repair load is inserted. If unresolved_count is
// non-zero, fixed_count is zero and the input is unchanged.
[[nodiscard]] LUISA_XIR_API FixSelfReferentialInfo fix_self_referential_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API FixSelfReferentialInfo fix_self_referential_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
