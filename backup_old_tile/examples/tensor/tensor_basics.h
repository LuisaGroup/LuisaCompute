// =============================================================================
// tensor_basics.h — tensor-basics exercises tile-language entry point
// =============================================================================
// The basics driver (host data build + device kernels + host checks) is folded
// into the single `example_tensor_stub` target:
// `example_tensor_stub <backend> --basics`.
// The implementation lives in examples/tensor/tensor_basics.cpp.
// =============================================================================

#pragma once

namespace basics {

/// Run the Luisa tile-language tensor-basics exercises (the C++ twin of
/// examples/tensor/tensor_basics.py): tensors, elementwise operations,
/// autograd-style derivative checks and a tiny 1 -> 1 neural network trained
/// with SGD.  Parses the backend name from the positional command-line
/// arguments (the --basics dispatch flag itself is ignored here).  Returns the
/// process exit code (0 on success).
int run_basics(int argc, char *argv[]);

}// namespace basics
