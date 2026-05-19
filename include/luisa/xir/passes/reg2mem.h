#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class PhiInst;
class Function;
class Module;

struct Reg2MemInfo {
    size_t lowered_phi_count{0u};
    size_t lowered_cross_block_value_count{0u};
};

[[nodiscard]] LUISA_XIR_API Reg2MemInfo reg2mem_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API Reg2MemInfo reg2mem_pass_run_on_module(Module *module) noexcept;

}// namespace luisa::compute::xir
