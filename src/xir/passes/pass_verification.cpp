#include <luisa/core/logging.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/pass_verification.h>

namespace luisa::compute::xir {

XIRPassVerificationTransaction::XIRPassVerificationTransaction(
    Module *module) noexcept
    : _module{module} {}

XIRPassVerificationTransaction::XIRPassVerificationTransaction(
    XIRPassVerificationTransaction &&other) noexcept
    : _module{other._module},
      _output_boundary_checked{
          other._output_boundary_checked} {
    other._module = nullptr;
    other._output_boundary_checked = true;
}

XIRPassVerificationTransaction::~XIRPassVerificationTransaction() noexcept {
    LUISA_ASSERT(
        _module == nullptr || _output_boundary_checked,
        "An enclosing XIR pass verification transaction ended without its "
        "complete output boundary being checked.");
}

bool XIRPassVerificationTransaction::contains(
    const Function *function) const noexcept {
    return _module != nullptr && function != nullptr &&
           function->parent_module() == _module;
}

XIRVerificationResult
XIRPassVerificationTransaction::verify_output(
    const XIRVerificationOptions &options) noexcept {
    LUISA_ASSERT(
        _module != nullptr && !_output_boundary_checked,
        "An XIR pass verification transaction output boundary may be "
        "checked exactly once.");
    _output_boundary_checked = true;
    return xir_verify_module(_module, options);
}

XIRPassVerificationTransaction
begin_xir_pass_verification_transaction(
    Module *module,
    const XIRVerificationOptions &options) noexcept {
    auto verification = xir_verify_module(module, options);
    if (!verification.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid XIR at enclosing pass-transaction input: {} "
            "({} error(s) total).",
            verification.errors.front().message,
            verification.errors.size());
    }
    return XIRPassVerificationTransaction{module};
}

bool xir_pass_has_standalone_verification(
    const XIRPassVerificationTransaction *transaction,
    const Function *function) noexcept {
    if (transaction == nullptr) { return true; }
    LUISA_ASSERT(
        transaction->contains(function),
        "A composed XIR pass received a verification transaction for a "
        "different module.");
    LUISA_ASSERT(
        !transaction->output_boundary_checked(),
        "A composed XIR pass cannot mutate a transaction after its output "
        "boundary was checked.");
    return false;
}

bool xir_pass_has_standalone_verification(
    const XIRPassVerificationTransaction *transaction,
    const Module *module) noexcept {
    if (transaction == nullptr) { return true; }
    LUISA_ASSERT(
        transaction->contains(module),
        "A composed XIR pass received a verification transaction for a "
        "different module.");
    LUISA_ASSERT(
        !transaction->output_boundary_checked(),
        "A composed XIR pass cannot mutate a transaction after its output "
        "boundary was checked.");
    return false;
}

}// namespace luisa::compute::xir
