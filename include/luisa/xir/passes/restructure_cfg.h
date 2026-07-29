#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class PassReport;

class Module;
class Function;

struct RestructureCFGInfo {
    size_t restructured_loop_count{0u};
    size_t restructured_if_count{0u};
    size_t restructured_switch_count{0u};
    // Number of successful auxiliary CFG-canonicalization phases. These
    // rewrites do not create a new structured construct, but they still mutate
    // the IR (for example by splitting duplicate Switch case targets).
    size_t canonicalized_cfg_count{0u};
    // Diagnostic operation count for dominance trees built while enforcing
    // unique structured-construct entries. This is not a change count.
    size_t construct_entry_dom_tree_count{0u};
    size_t irreducible_region_count{0u};
    size_t unstructured_branch_count{0u};
    size_t invalid_construct_count{0u};
    size_t iteration_limit_count{0u};
    [[nodiscard]] bool changed() const noexcept {
        return restructured_loop_count != 0u ||
               restructured_if_count != 0u ||
               restructured_switch_count != 0u ||
               canonicalized_cfg_count != 0u;
    }
    [[nodiscard]] bool succeeded() const noexcept {
        return irreducible_region_count == 0u &&
               unstructured_branch_count == 0u &&
               invalid_construct_count == 0u &&
               iteration_limit_count == 0u;
    }
};

struct RestructureCFGOptions {
    // These are safety bounds, not semantic tuning knobs. Exhaustion rejects
    // the transaction and leaves the input function/module unchanged.
    size_t main_iteration_limit{10000u};
    size_t post_iteration_limit{64u};
};

// Converts reducible plain CFG regions into structured control flow. A function
// containing an irreducible (multi-entry) cyclic SCC is rejected before any IR
// mutation and reported through irreducible_region_count. The complete pass is
// transactional: any late failure, including an exhausted safety bound or
// output-verifier rejection, discards the shadow CFG. Declaration-like
// callables with no body own no CFG and are successful no-ops.
[[nodiscard]] LUISA_XIR_API RestructureCFGInfo restructure_cfg_pass_run_on_function(
    Function *function,
    const RestructureCFGOptions &options = {}) noexcept;
[[nodiscard]] LUISA_XIR_API RestructureCFGInfo restructure_cfg_pass_run_on_module(
    Module *module, PassReport *report = nullptr,
    const RestructureCFGOptions &options = {}) noexcept;

}// namespace luisa::compute::xir
