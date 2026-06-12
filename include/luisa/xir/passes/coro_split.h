#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute {
class Type;
}

namespace luisa::compute::xir {

class CallableFunction;
class Module;
class Value;
struct CoroCfgDistillResult;

struct CoroSplitInfo {
    struct Subroutine {
        size_t scope_index{0u};
        uint32_t trigger_token{0u};
        luisa::optional<luisa::string> trigger_name;
        CallableFunction *callable{nullptr};
        Value *frame_argument{nullptr};
    };
    luisa::vector<Subroutine> subroutines;
};

[[nodiscard]] LUISA_XIR_API size_t coro_split_pass_run_on_module(Module *m) noexcept;
[[nodiscard]] LUISA_XIR_API size_t coro_split_pass_run_on_module_with_cfg(
    Module *m, const CoroCfgDistillResult &cfg) noexcept;
[[nodiscard]] LUISA_XIR_API size_t coro_split_pass_run_on_module_with_cfg_and_frame(
    Module *m, const CoroCfgDistillResult &cfg, const Type *frame_type) noexcept;
[[nodiscard]] LUISA_XIR_API CoroSplitInfo coro_split_pass_run_on_module_with_cfg_and_frame_info(
    Module *m, const CoroCfgDistillResult &cfg, const Type *frame_type) noexcept;

}// namespace luisa::compute::xir
