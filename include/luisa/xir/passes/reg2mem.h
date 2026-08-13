#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class PassReport;

class PhiInst;
class Function;
class Module;

struct Reg2MemInfo {
    size_t lowered_phi_count{0u};
    size_t lowered_cross_block_value_count{0u};
    size_t hoisted_alloca_count{0u};
    [[nodiscard]] bool changed() const noexcept {
        return lowered_phi_count != 0u ||
               lowered_cross_block_value_count != 0u ||
               hoisted_alloca_count != 0u;
    }
};

struct Reg2MemSpillAuditInfo {
    size_t remaining_phi_spill_count{0u};
    size_t remaining_cross_block_spill_count{0u};
    size_t remaining_invalid_spill_count{0u};

    [[nodiscard]] auto remaining_spill_count() const noexcept {
        return remaining_phi_spill_count +
               remaining_cross_block_spill_count +
               remaining_invalid_spill_count;
    }
    [[nodiscard]] auto succeeded() const noexcept {
        return remaining_spill_count() == 0u;
    }
};

[[nodiscard]] LUISA_XIR_API Reg2MemInfo reg2mem_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API Reg2MemInfo reg2mem_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;
[[nodiscard]] LUISA_XIR_API Reg2MemInfo reg2mem_pass_repair_cross_block_rvalue_uses_on_function(Function *function) noexcept;
/// Audit helpers are total: null inputs report an empty, successful audit.
[[nodiscard]] LUISA_XIR_API Reg2MemSpillAuditInfo audit_reg2mem_spills_on_function(const Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API Reg2MemSpillAuditInfo audit_reg2mem_spills_on_module(const Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
