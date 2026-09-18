// =============================================================================
// tensor_basics.h — tensor-basics exercises entry point (new Tile DSL)
// =============================================================================
// The basics driver (host data build + device kernels + host checks) is invoked
// through example_tensor's main() with the `--basics` flag:
//   example_tensor <backend> --basics
// The implementation lives in examples/tensor/tensor_basics.cpp; the kernels
// live in examples/tensor/tensor_basics_kernels.h.
// =============================================================================

#pragma once

namespace tensor_example::basics {

/// Run the tensor-basics exercises (the C++ twin of examples/tensor/
/// tensor_basics.py): tensors, elementwise operations, an autograd-style
/// derivative check and a tiny 1 -> 1 neural network trained with SGD.
/// Parses the backend name from the positional command-line arguments (the
/// --basics dispatch flag itself is ignored here).  Returns the process exit
/// code (0 on success, non-zero on verification failure).
int run_basics(int argc, char *argv[]);

}// namespace tensor_example::basics
