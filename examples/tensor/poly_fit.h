// =============================================================================
// poly_fit.h — polynomial-fit tile-language training entry point
// =============================================================================
// The polynomial-fit driver (host data build + device training loop + host
// reference verification) is folded into the single `example_tensor_stub`
// target: `example_tensor_stub <backend> --poly-fit [--steps N]`.
// The implementation lives in examples/tensor/poly_fit.cpp.
// =============================================================================

#pragma once

namespace polyfit {

/// Run the Luisa tile-language polynomial-fit training (the C++ twin of
/// examples/tensor/poly_fit_train.py).  Parses the backend name from the
/// positional command-line arguments and the optional --steps flag (the
/// --poly-fit dispatch flag itself is ignored here).  Returns the process
/// exit code (0 on success).
int run_poly_fit(int argc, char *argv[]);

}// namespace polyfit
