// =============================================================================
// rnn.h — New Tile DSL RNN sequence classification training entry point
// =============================================================================
// The RNN training driver (synthetic counting dataset + host reference
// training loop + Tile kernel compilation + device forward evaluation) is
// invoked through the example_tensor main() with the `--rnn` flag:
//   example_tensor <backend> --rnn [--epochs N]
// The implementation lives in examples/tensor/rnn.cpp.
// =============================================================================

#pragma once

namespace tensor_example::rnntrain {

/// Runs the tanh-RNN training demo ported to the new execution-structure-first
/// Tile DSL. Parses the backend name and --epochs N from the command line
/// (the --rnn dispatch flag itself is ignored here). Returns the process exit
/// code (0 on success, non-zero when any compile / verification step failed).
int run_rnn(int argc, char *argv[]);

}// namespace tensor_example::rnntrain
