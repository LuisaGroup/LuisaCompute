#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class PassReport;

class Module;
class Function;
class FunctionDefinition;

class AllocaInst;
class LoadInst;
class StoreInst;
class PhiInst;

// This pass is similar to LLVM's mem2reg pass. It tries to rewrite
// alloca/load/store instructions into SSA values.
//
// Note: this pass does not guarantee that all alloca's are eliminated.
// Typically, the following cases are not handled:
// - Aggregates that are used by GEPs;
// - Shared memory alloca's; and
// - Alloca's that are used as reference arguments.
// It's recommended to run this pass after load elimination and dead
// code elimination passes.
//
// Ordinary allocas with metadata other than a debug NameMD, and annotated
// loads/stores, are retained because their SSA replacements have no unique
// metadata owner. NameMD is copied to every Phi created for the logical
// variable; this also recovers a NameMD cloned onto a synthetic reg2mem Phi
// reload. If no Phi is needed, the storage-only debug name is deliberately
// dropped. Other synthetic reg2mem spill metadata is explicitly consumed
// during a successful round trip.

struct Mem2RegInfo {
    size_t promoted_alloca_count{0u};
    size_t removed_store_count{0u};
    size_t removed_load_count{0u};
    size_t inserted_phi_count{0u};
    [[nodiscard]] bool changed() const noexcept {
        return promoted_alloca_count != 0u ||
               removed_store_count != 0u ||
               removed_load_count != 0u ||
               inserted_phi_count != 0u;
    }
};

[[nodiscard]] LUISA_XIR_API Mem2RegInfo mem2reg_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API Mem2RegInfo mem2reg_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

// Controls which local allocas CUDA/LLVM codegen isolates from LLVM SROA/mem2reg
// because promoting or scalar-replacing them would blow up PHI operands or stack slots.
struct AllocaSpillHazardOptions {
    size_t max_sroa_scalar_leaves{256u};
    size_t min_phi_operands_per_alloca{256u};
    size_t max_phi_operands_per_alloca{4096u};
    size_t phi_operand_access_multiplier{16u};
    size_t max_total_phi_operands_per_function{65536u};
    size_t max_gep_slice_multiplier{16u};
};

[[nodiscard]] LUISA_XIR_API luisa::vector<const AllocaInst *>
collect_llvm_alloca_spill_hazards(const FunctionDefinition *definition,
                                  AllocaSpillHazardOptions options = {}) noexcept;

}// namespace luisa::compute::xir
