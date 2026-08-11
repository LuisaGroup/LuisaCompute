#pragma once

#include <luisa/xir/verifier.h>

namespace luisa::compute::xir {

class Function;
class Module;

// A non-forgeable witness that a complete module was verified before an
// enclosing mutation transaction. Composed passes accept the witness only for
// functions owned by that exact module and keep their transform-specific
// checks active. The owner must close the witness with one complete output
// verification before publishing or translating the result.
class LUISA_XIR_API XIRPassVerificationTransaction {

private:
    Module *_module{nullptr};
    bool _output_boundary_checked{false};

private:
    explicit XIRPassVerificationTransaction(Module *module) noexcept;
    friend XIRPassVerificationTransaction
    begin_xir_pass_verification_transaction(
        Module *module,
        const XIRVerificationOptions &options) noexcept;

public:
    XIRPassVerificationTransaction(
        XIRPassVerificationTransaction &&other) noexcept;
    XIRPassVerificationTransaction(
        const XIRPassVerificationTransaction &) = delete;
    XIRPassVerificationTransaction &operator=(
        XIRPassVerificationTransaction &&) = delete;
    XIRPassVerificationTransaction &operator=(
        const XIRPassVerificationTransaction &) = delete;
    ~XIRPassVerificationTransaction() noexcept;

    [[nodiscard]] Module *module() const noexcept { return _module; }
    [[nodiscard]] bool contains(const Module *module) const noexcept {
        return _module != nullptr && module == _module;
    }
    [[nodiscard]] bool contains(const Function *function) const noexcept;
    [[nodiscard]] bool output_boundary_checked() const noexcept {
        return _output_boundary_checked;
    }
    [[nodiscard]] XIRVerificationResult verify_output(
        const XIRVerificationOptions &options = {}) noexcept;
};

[[nodiscard]] LUISA_XIR_API XIRPassVerificationTransaction
begin_xir_pass_verification_transaction(
    Module *module,
    const XIRVerificationOptions &options = {}) noexcept;

// Returns true for a standalone call. A transaction witness suppresses the
// nested generic boundary only after exact function/module identity is
// asserted.
[[nodiscard]] LUISA_XIR_API bool
xir_pass_has_standalone_verification(
    const XIRPassVerificationTransaction *transaction,
    const Function *function) noexcept;

[[nodiscard]] LUISA_XIR_API bool
xir_pass_has_standalone_verification(
    const XIRPassVerificationTransaction *transaction,
    const Module *module) noexcept;

}// namespace luisa::compute::xir
