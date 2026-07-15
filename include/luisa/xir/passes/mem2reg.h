#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/unordered_map.h>

namespace luisa::compute::xir {

class PassReport;

class Module;
class Function;

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

struct Mem2RegInfo {
    size_t promoted_alloca_count{0u};
    size_t removed_store_count{0u};
    size_t removed_load_count{0u};
    size_t inserted_phi_count{0u};
};

[[nodiscard]] LUISA_XIR_API Mem2RegInfo mem2reg_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API Mem2RegInfo mem2reg_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
