#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/ast/coro_suspend.h>
#include <luisa/xir/passes/pass_verification.h>

namespace luisa::compute::xir {

class BasicBlock;
class Function;
class FunctionDefinition;
class Module;
class Value;

}// namespace luisa::compute::xir

namespace luisa::compute {
class Type;
}// namespace luisa::compute

namespace luisa::compute::xir {

// Complete suspend-extension ownership at one coroutine boundary. Binding
// indices in every extension resolve into binding_values. The wrapper is
// deeply copyable so a distilled CFG remains a regular analysis certificate;
// copying never aliases plugin objects or loses unknown extension payloads.
struct CoroSuspendExtensionOwner {
    luisa::vector<CoroSuspendExtensionPtr> extensions;
    luisa::vector<Value *> binding_values;

    CoroSuspendExtensionOwner() noexcept = default;
    CoroSuspendExtensionOwner(
        const CoroSuspendExtensionOwner &other) noexcept
        : binding_values{other.binding_values} {
        extensions.reserve(other.extensions.size());
        for (auto &&extension : other.extensions) {
            extensions.emplace_back(
                extension == nullptr ? nullptr : extension->clone());
        }
    }
    CoroSuspendExtensionOwner(
        CoroSuspendExtensionOwner &&) noexcept = default;
    CoroSuspendExtensionOwner &operator=(
        const CoroSuspendExtensionOwner &other) noexcept {
        if (this != &other) {
            CoroSuspendExtensionOwner copy{other};
            *this = std::move(copy);
        }
        return *this;
    }
    CoroSuspendExtensionOwner &operator=(
        CoroSuspendExtensionOwner &&) noexcept = default;

    [[nodiscard]] bool empty() const noexcept {
        return extensions.empty() && binding_values.empty();
    }
};

struct CoroCfgDistillResult {

    struct FrameValue {
        // Root SSA value or local allocation. A non-empty access chain
        // identifies a disjoint statically indexed subobject. Paths of an SSA
        // root form a complete partition and are reconstructed at continuation
        // entry; local-allocation paths may be only the live subset. Frame
        // analysis operates on this (root, path) identity; source XIR is
        // deliberately left aggregate so frame minimization does not inflate
        // the computation IR.
        Value *value{nullptr};
        luisa::vector<uint32_t> access_chain;
        luisa::string name;
        // Explicit scheduler ABI aliases. Diagnostic SSA/local names are not
        // aliases unless a suspension boundary exports them semantically.
        // One logical value may have several aliases; all resolve to its one
        // physical frame slot after coloring.
        luisa::vector<luisa::string> aliases;
        const Type *type{nullptr};
        // Physical payload slot assigned by interference coloring. Several
        // exact-typed logical values may share a slot when their
        // continuation/transition lifetimes do not overlap; Boolean values
        // additionally share a uint slot through distinct bit offsets. Their
        // names remain physical-field aliases in CoroMaterializeInfo.
        size_t slot{0u};
        // Boolean values are colored at bit granularity. A present offset
        // selects one bit in a physical uint slot; interfering values receive
        // distinct (slot, bit) identities, while disjoint lifetimes may reuse
        // the same bit. Non-Boolean values keep this empty.
        luisa::optional<uint32_t> bit_offset;
    };

    struct FrameSlot {
        luisa::string name;
        const Type *type{nullptr};
    };

    struct Scope {
        struct SuspendPoint {
            BasicBlock *block{nullptr};
            uint32_t token{0u};
            luisa::string name;
            CoroSuspendExtensionOwner extension_owner;
        };
        luisa::vector<BasicBlock *> blocks;
        luisa::vector<SuspendPoint> suspend_points;
        int scope_id{0};
        luisa::optional<uint32_t> suspend_token;
        luisa::optional<luisa::string> suspend_name;
        uint32_t trigger_token{0u};
        luisa::optional<luisa::string> trigger_name;
        luisa::vector<Value *> external_values;
        luisa::vector<Value *> touched_values;
        luisa::vector<Value *> live_in_values;
        luisa::vector<Value *> live_out_values;
        // Primary lossless frame-state relations. The legacy Value* vectors
        // above contain unique roots for source-level diagnostics; one root
        // can correspond to several access-path frame values.
        luisa::vector<size_t> external_frame_value_indices;
        luisa::vector<size_t> touched_frame_value_indices;
        luisa::vector<size_t> live_in_frame_value_indices;
        luisa::vector<size_t> live_out_frame_value_indices;
        luisa::vector<luisa::string> external_variables;
        luisa::vector<luisa::string> touched_variables;
        luisa::vector<luisa::string> live_in_variables;
        luisa::vector<luisa::string> live_out_variables;
        bool is_terminal{false};
    };

    struct Edge {
        size_t from_scope{0u};
        size_t to_scope{0u};
        uint32_t token{0u};
        BasicBlock *exit_block{nullptr};
        bool is_suspend{false};
        CoroSuspendExtensionOwner extension_owner;
        // For each owner binding, the logical frame-value pieces that encode
        // it after ABI decomposition. CoroGraph projects these through slot
        // coloring into CoroSlotAccess; the full Extension above remains the
        // semantic source of schema, access, lifetime, and attributes.
        luisa::vector<luisa::vector<size_t>>
            extension_binding_frame_value_indices;
        // Static local-lvalue path relative to the owning alloca. Empty for
        // rvalue bindings and whole-allocation lvalues. CoroSlotAccess removes
        // this prefix from FrameValue::access_chain when reconstructing the
        // binding's logical type; it does not allocate additional storage.
        luisa::vector<luisa::vector<uint32_t>>
            extension_binding_access_chains;
        luisa::vector<Value *> killed_values;
        luisa::vector<Value *> touched_values;
        luisa::vector<Value *> live_values;
        luisa::vector<Value *> store_values;
        luisa::vector<size_t> killed_frame_value_indices;
        luisa::vector<size_t> touched_frame_value_indices;
        luisa::vector<size_t> live_frame_value_indices;
        luisa::vector<size_t> store_frame_value_indices;
        luisa::vector<luisa::string> killed_variables;
        luisa::vector<luisa::string> touched_variables;
        luisa::vector<luisa::string> live_variables;
        luisa::vector<luisa::string> store_variables;
    };

    luisa::vector<Scope> scopes;
    luisa::vector<luisa::vector<size_t>> edges;
    luisa::vector<Edge> transition_edges;
    luisa::vector<FrameValue> frame_values;
    luisa::vector<FrameSlot> frame_slots;
    size_t structured_cfg_error_count{0u};
    size_t invalid_input_error_count{0u};
    size_t invalid_cfg_error_count{0u};
    size_t boundary_verifier_count{0u};

private:
    // A distilled result is an analysis certificate for one immutable source
    // CFG version. Split validates this seal in linear time instead of
    // rerunning the complete liveness fixed point. The seal covers the source
    // instruction graph and every semantic result field, so copying is valid
    // while mutation or reuse after CFG edits is rejected.
    FunctionDefinition *_source_definition{nullptr};
    uint64_t _validation_hash{0u};

    void _seal(FunctionDefinition *definition) noexcept;
    friend LUISA_XIR_API CoroCfgDistillResult
    coro_cfg_distill_pass_run_on_function(
        Function *f, const struct CoroCfgDistillOptions &options) noexcept;

public:
    [[nodiscard]] bool validation_certificate_matches(
        const FunctionDefinition *definition) const noexcept;

    [[nodiscard]] bool succeeded() const noexcept {
        return structured_cfg_error_count == 0u &&
               invalid_input_error_count == 0u &&
               invalid_cfg_error_count == 0u;
    }
};

// Optional host-analysis diagnostics. Counts are reset on every invocation,
// including rejected input, and do not participate in the immutable semantic
// certificate carried by CoroCfgDistillResult.
struct CoroCfgDistillStats {
    size_t value_atom_count{0u};
    size_t scope_count{0u};
    // Sum of each scope's compact active atom domain. The corresponding
    // unprojected product size is value_atom_count * scope_count.
    size_t projected_scope_atom_count{0u};
    size_t max_projected_scope_atom_count{0u};
    size_t block_membership_count{0u};
    size_t must_block_evaluation_count{0u};
    size_t may_block_evaluation_count{0u};
};

struct CoroCfgDistillOptions {
    const XIRPassVerificationTransaction *verification_transaction{nullptr};
    CoroCfgDistillStats *stats{nullptr};
};

// These analysis entry points do not mutate the input. Their scope/liveness
// model requires verifier-valid, void, Phi-free raw CFG: call destructure_cfg
// for structured control and reg2mem for SSA Phis before distilling. Coroutine
// functions carrying a fixed SignatureConstraint cannot be split because the
// continuation ABI prepends a frame argument. Unsupported or invalid input
// returns an empty, explicitly failed result. The module overload returns the
// number of definitions successfully distilled.
[[nodiscard]] LUISA_XIR_API CoroCfgDistillResult coro_cfg_distill_pass_run_on_function(
    Function *f,
    const CoroCfgDistillOptions &options = {}) noexcept;
[[nodiscard]] LUISA_XIR_API size_t coro_cfg_distill_pass_run_on_module(Module *m) noexcept;

}// namespace luisa::compute::xir
