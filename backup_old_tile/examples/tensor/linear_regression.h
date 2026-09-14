// =============================================================================
// linear_regression.h — linear & logistic regression tile-language entry point
// =============================================================================
// The regression driver (host data build + device training loop + host
// reference verification) is folded into the single `example_tensor_stub`
// target: `example_tensor_stub <backend> --linear-regression [--steps N]`.
// The implementation lives in examples/tensor/linear_regression.cpp.
// =============================================================================

#pragma once

namespace lreg {

/// Run the Luisa tile-language linear & logistic regression training (the C++
/// twin of examples/tensor/linear_regression_train.py).  Parses the backend
/// name from the positional command-line arguments and the optional --steps
/// flag (the --linear-regression dispatch flag itself is ignored here).
/// Returns the process exit code (0 on success).
int run_linear_regression(int argc, char *argv[]);

}// namespace lreg
