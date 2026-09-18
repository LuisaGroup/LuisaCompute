// =============================================================================
// mnist.h — synthetic-MNIST MLP training tile-language entry point
// =============================================================================
// The MNIST driver (host data build + device training loop + host reference
// verification) is folded into the single `example_tensor` target:
// `example_tensor <backend> --mnist [--epochs N]`.
// The implementation lives in examples/tensor/mnist.cpp.
// =============================================================================

#pragma once

namespace tensor_example::mnisttrain {

/// Run the new-DSL Tile MLP training on the synthetic MNIST stand-in
/// (the C++ twin of examples/tensor/mnist_train.py --dataset synthetic).
/// Parses the backend name from the positional command-line arguments and the
/// optional --epochs flag (the --mnist dispatch flag itself is ignored here).
/// Returns the process exit code (0 on success).
int run_mnist(int argc, char *argv[]);

}// namespace tensor_example::mnisttrain
