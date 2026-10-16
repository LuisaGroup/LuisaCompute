#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute {
class Type;
}// namespace luisa::compute

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
        // The callable signature is [frame, projected source arguments...].
        // Splitting initially records the identity projection. Coroutine
        // compilation may subsequently remove continuation-local dead
        // arguments while preserving this ordered map back to the source
        // coroutine ABI.
        luisa::vector<size_t> source_argument_indices;
    };
    luisa::vector<Subroutine> subroutines;
    // Coro splitting is an unstructured-CFG-only transform. A non-zero value
    // means the complete request was rejected before any IR was mutated.
    size_t structured_cfg_error_count{0u};
    size_t invalid_cfg_error_count{0u};

    [[nodiscard]] bool changed() const noexcept {
        return !subroutines.empty();
    }
    [[nodiscard]] bool succeeded() const noexcept {
        return structured_cfg_error_count == 0u && invalid_cfg_error_count == 0u;
    }
};

// Coro split does not lower structured control flow or edge-sensitive Phis.
// Call destructure_cfg and reg2mem before invoking this pass; destructure_cfg
// converts SwitchInst to IndexedBranchInst. Module entry points are
// atomic: if any coroutine definition is unsupported, no definition is split.
// Explicit distilled CFG input must be non-empty and wholly owned by the module
// passed to the entry point.
[[nodiscard]] LUISA_XIR_API CoroSplitInfo coro_split_pass_run_on_module_info(Module *m) noexcept;
// Compatibility count-only entry points return zero on rejection. Use an
// Info-returning entry point when the caller must inspect error counts.
[[nodiscard]] LUISA_XIR_API size_t coro_split_pass_run_on_module(Module *m) noexcept;
[[nodiscard]] LUISA_XIR_API size_t coro_split_pass_run_on_module_with_cfg(
    Module *m, const CoroCfgDistillResult &cfg) noexcept;
[[nodiscard]] LUISA_XIR_API size_t coro_split_pass_run_on_module_with_cfg_and_frame(
    Module *m, const CoroCfgDistillResult &cfg, const Type *frame_type) noexcept;
[[nodiscard]] LUISA_XIR_API CoroSplitInfo coro_split_pass_run_on_module_with_cfg_and_frame_info(
    Module *m, const CoroCfgDistillResult &cfg, const Type *frame_type) noexcept;

}// namespace luisa::compute::xir
