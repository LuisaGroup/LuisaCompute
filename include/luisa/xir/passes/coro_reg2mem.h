#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class Module;
struct CoroSplitInfo;

struct CoroReg2MemInfo {
    size_t callable_count{0u};
    size_t lowered_phi_count{0u};
    size_t lowered_cross_block_value_count{0u};
    [[nodiscard]] bool changed() const noexcept {
        return lowered_phi_count != 0u ||
               lowered_cross_block_value_count != 0u;
    }
};

[[nodiscard]] LUISA_XIR_API CoroReg2MemInfo coro_reg2mem_pass_run_on_module(Module *m) noexcept;
[[nodiscard]] LUISA_XIR_API CoroReg2MemInfo coro_reg2mem_pass_run_on_split(const CoroSplitInfo &split) noexcept;

}// namespace luisa::compute::xir
