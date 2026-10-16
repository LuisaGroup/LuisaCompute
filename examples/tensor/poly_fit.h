// =============================================================================
// poly_fit.h — polynomial-fit training entry point (new Tile DSL)
// =============================================================================
// The polynomial-fit driver (host data build + device training loop + host
// reference verification) is invoked through example_tensor's main() with the
// `--poly-fit` flag:
//   example_tensor <backend> --poly-fit [--steps N]
// The implementation lives in examples/tensor/poly_fit.cpp; the kernels live
// in examples/tensor/poly_fit_kernels.{h,cpp}.
// =============================================================================

#pragma once

namespace tensor_example::polyfit {

/// Run the polynomial-fit training (the C++ twin of examples/tensor/
/// poly_fit_train.py): fit y = sin(x) on [-pi, pi] with a degree-3 polynomial
/// trained by manually applying the gradients, every step of the loop (forward
/// GEMM, MSE residual, gradient GEMM, SGD update) running as a Tile DSL kernel
/// on the device.  Parses the backend name from the positional command-line
/// arguments and the optional --steps flag (the --poly-fit dispatch flag
/// itself is ignored here).  Returns the process exit code (0 on success).
int run_poly_fit(int argc, char *argv[]);

}// namespace tensor_example::polyfit
