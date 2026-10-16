#pragma once

#include <cstddef>

#include <luisa/core/dll_export.h>
#include <luisa/xir/passes/pass_verification.h>

namespace luisa::compute::xir {

class Function;
class Module;
class PassReport;

struct LowerIrreducibleCFGInfo {
    size_t lowered_region_count{0u};
    size_t created_dispatch_block_count{0u};
    size_t created_edge_block_count{0u};
    size_t remaining_irreducible_region_count{0u};
    size_t error_count{0u};
    size_t boundary_verifier_count{0u};

    [[nodiscard]] bool changed() const noexcept {
        return lowered_region_count != 0u;
    }
    [[nodiscard]] bool succeeded() const noexcept {
        return error_count == 0u &&
               remaining_irreducible_region_count == 0u;
    }
};

struct LowerIrreducibleCFGOptions {
    const XIRPassVerificationTransaction *verification_transaction{nullptr};
};

// Makes every reachable multi-entry cyclic region reducible without cloning
// its body. Regions are discovered by recursively decomposing SCCs through
// unique headers, so an irreducible inner cycle cannot hide inside a natural
// outer loop. For a region with entry nodes e_i, all incoming edges to every
// e_i are redirected through an edge-local selector store and one dispatcher:
//
//   predecessor -> store(i) -> dispatcher -> e_i
//
// Redirecting internal as well as external entry edges puts the dispatcher in
// the region and makes it the unique entry. Each lowering step is O(V + E),
// adds one local uint selector, and does not duplicate the original shader
// body. If lowering is needed, the reachable CFG must use raw Branch,
// ConditionalBranch, or IndexedBranch successor edges. Unsupported inputs are
// rejected before the first mutation; the module overload preflights every
// definition before mutating any definition.
[[nodiscard]] LUISA_XIR_API LowerIrreducibleCFGInfo
lower_irreducible_cfg_pass_run_on_function(
    Function *function,
    const LowerIrreducibleCFGOptions &options = {}) noexcept;
[[nodiscard]] LUISA_XIR_API LowerIrreducibleCFGInfo
lower_irreducible_cfg_pass_run_on_module(
    Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
