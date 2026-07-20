#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/unordered_map.h>

namespace luisa::compute::xir {

class PassReport;

class LoadInst;
class Function;
class Module;

// This pass implements a simple local load elimination optimization.
// For each load instruction, if a recent load instruction with the same
// variable has happened without possible intervening stores, the load
// instruction can be replaced with the recent load instruction.

struct LocalLoadEliminationInfo {
    size_t removed_load_count{0u};
};

[[nodiscard]] LUISA_XIR_API LocalLoadEliminationInfo local_load_elimination_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API LocalLoadEliminationInfo local_load_elimination_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
