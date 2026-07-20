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
    [[nodiscard]] bool succeeded() const noexcept { return errors.empty(); }
};

[[nodiscard]] LUISA_XIR_API XIRVerificationResult
xir_verify_function(const Function *function,
                    const XIRVerificationOptions &options = {}) noexcept;

[[nodiscard]] LUISA_XIR_API XIRVerificationResult
xir_verify_module(const Module *module,
                  const XIRVerificationOptions &options = {}) noexcept;

}// namespace luisa::compute::xir
