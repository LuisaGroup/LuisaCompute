#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class Module;
class Function;
class BasicBlock;
class Instruction;

struct XIRVerificationOptions {
    bool require_terminated_blocks{true};
    bool require_reachable_blocks{false};
    bool require_no_phi{false};
    bool require_no_unstructured_control_flow{false};
    bool require_unique_merge_blocks{false};
    bool require_canonical_break_continue_targets{false};
};

struct XIRVerificationError {
    const Function *function{nullptr};
    const BasicBlock *block{nullptr};
    const Instruction *instruction{nullptr};
    luisa::string message;
};

struct XIRVerificationResult {
    luisa::vector<XIRVerificationError> errors;
    struct Statistics {
        // Exact O(1) physical-owner predicates evaluated while checking the
        // logical Value of each instruction operand.
        size_t use_list_owner_checks{0u};
        // Membership never traverses a use-list. Keeping the work observable
        // makes an accidental return to fanout-dependent scans testable.
        size_t use_list_membership_traversal_steps{0u};
        // Reachable CFG blocks represented by the verifier's sparse
        // immediate-dominator trees.
        size_t dominance_tree_nodes{0u};
        // A non-empty immediate-dominator tree has exactly V - 1 parent
        // edges. Keeping this observable guards against accidentally
        // materializing the O(V^2) dominance relation again.
        size_t dominance_tree_edges{0u};
        // Locally-owned reachable CFG edges encoded in predecessor CSR form
        // while constructing the trees.
        size_t dominance_cfg_edges{0u};
        // Cooper-Harvey-Kennedy fixed-point sweeps, including the final
        // unchanged sweep. This makes unexpectedly poor convergence visible
        // without relying on a wall-clock threshold in unit tests.
        size_t dominance_fixed_point_iterations{0u};
        // Exact dominance predicates requested by SSA and structured-control
        // validation. Each query is answered by two ancestry-interval tests.
        size_t dominance_queries{0u};
    } statistics;
    [[nodiscard]] bool succeeded() const noexcept { return errors.empty(); }
};

[[nodiscard]] LUISA_XIR_API XIRVerificationResult
xir_verify_function(const Function *function,
                    const XIRVerificationOptions &options = {}) noexcept;

// Verifies a bounded set of functions with one verifier invocation. This is
// useful for transactional passes whose candidate output lives in shadow
// definitions: the complete output boundary can be checked once without
// re-verifying unrelated input definitions one by one.
[[nodiscard]] LUISA_XIR_API XIRVerificationResult
xir_verify_functions(luisa::span<const Function *const> functions,
                     const XIRVerificationOptions &options = {}) noexcept;

[[nodiscard]] LUISA_XIR_API XIRVerificationResult
xir_verify_module(const Module *module,
                  const XIRVerificationOptions &options = {}) noexcept;

}// namespace luisa::compute::xir
