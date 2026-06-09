#pragma once

#include <luisa/core/dll_export.h>

namespace luisa::compute::xir {

class Module;

[[nodiscard]] LUISA_XIR_API size_t coro_split_pass_run_on_module(Module *m) noexcept;

}// namespace luisa::compute::xir
