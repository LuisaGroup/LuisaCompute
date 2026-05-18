#pragma once

#include "luisa/xir/module.h"

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/unordered_map.h>

namespace luisa::compute::xir {

class Module;
class CallableFunction;

struct UnusedCallableRemovalInfo {
    size_t removed_callable_count{0u};
};

[[nodiscard]] LUISA_XIR_API UnusedCallableRemovalInfo unused_callable_removal_pass_run_on_module(Module *module) noexcept;

}// namespace luisa::compute::xir
