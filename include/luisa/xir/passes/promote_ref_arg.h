#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/unordered_map.h>

namespace luisa::compute::xir {

class PassReport;

class ReferenceArgument;
class ValueArgument;
class Module;

struct PromoteRefArgInfo {
    size_t promoted_ref_arg_count{0u};
};

[[nodiscard]] LUISA_XIR_API PromoteRefArgInfo promote_ref_arg_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
