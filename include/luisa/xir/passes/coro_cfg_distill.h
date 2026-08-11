#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
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

struct CoroCfgDistillResult {

    struct FrameValue {
        Value *value{nullptr};
        luisa::string name;
        const Type *type{nullptr};
    };

    struct Scope {
        struct SuspendPoint {
            BasicBlock *block{nullptr};
            uint32_t token{0u};
            luisa::string name;
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
        luisa::vector<Value *> killed_values;
        luisa::vector<Value *> touched_values;
        luisa::vector<Value *> live_values;
        luisa::vector<Value *> store_values;
        luisa::vector<luisa::string> killed_variables;
        luisa::vector<luisa::string> touched_variables;
        luisa::vector<luisa::string> live_variables;
        luisa::vector<luisa::string> store_variables;
    };

    luisa::vector<Scope> scopes;
    luisa::vector<luisa::vector<size_t>> edges;
    luisa::vector<Edge> transition_edges;
    luisa::vector<FrameValue> frame_values;
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
    friend CoroCfgDistillResult
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

struct CoroCfgDistillOptions {
    const XIRPassVerificationTransaction *verification_transaction{nullptr};
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
