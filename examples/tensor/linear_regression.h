// =============================================================================
// linear_regression.h — linear & logistic regression entry point (new Tile DSL)
// =============================================================================
// The regression driver (host data build + device training loop + host
// reference verification) is invoked through example_tensor's main() with the
// `--linear-regression` flag:
//   example_tensor <backend> --linear-regression [--steps N]
// The implementation lives in examples/tensor/linear_regression.cpp; the
// kernels live in examples/tensor/linear_regression_kernels.{h,cpp}.
// =============================================================================

#pragma once

namespace tensor_example::lreg {

/// Run the linear & logistic regression training (the C++ twin of
/// examples/tensor/linear_regression_train.py) entirely with Tile DSL kernels
/// on the device, then verify against an independent host CPU reference.
/// Parses the backend name from the positional command-line arguments and the
/// optional --steps flag (the --linear-regression dispatch flag itself is
/// ignored here).  Returns the process exit code (0 on success).
int run_linear_regression(int argc, char *argv[]);

}// namespace tensor_example::lreg
