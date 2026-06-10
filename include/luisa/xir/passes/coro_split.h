#pragma once

#include <luisa/core/dll_export.h>

namespace luisa::compute {
class Type;
}

namespace luisa::compute::xir {

class Module;
struct CoroCfgDistillResult;

[[nodiscard]] LUISA_XIR_API size_t coro_split_pass_run_on_module(Module *m) noexcept;
[[nodiscard]] LUISA_XIR_API size_t coro_split_pass_run_on_module_with_cfg(
    Module *m, const CoroCfgDistillResult &cfg) noexcept;
[[nodiscard]] LUISA_XIR_API size_t coro_split_pass_run_on_module_with_cfg_and_frame(
    Module *m, const CoroCfgDistillResult &cfg, const Type *frame_type) noexcept;

}// namespace luisa::compute::xir
