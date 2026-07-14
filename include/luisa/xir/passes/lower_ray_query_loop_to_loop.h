#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class PassReport;

class Module;
class Function;

struct LowerRayQueryLoopToLoopInfo {
    size_t lowered_ray_query_loop_count{0u};
};

[[nodiscard]] LUISA_XIR_API LowerRayQueryLoopToLoopInfo lower_ray_query_loop_to_loop_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API LowerRayQueryLoopToLoopInfo lower_ray_query_loop_to_loop_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
