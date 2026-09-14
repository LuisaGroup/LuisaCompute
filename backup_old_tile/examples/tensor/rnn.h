// =============================================================================
// rnn.h — RNN sequence-classification training tile-language entry point
// =============================================================================
// The RNN driver (host data build + device training loop + host reference
// verification) is folded into the single `example_tensor_stub` target:
// `example_tensor_stub <backend> --rnn [--epochs N]`.
// The implementation lives in examples/tensor/rnn.cpp.
// =============================================================================

#pragma once

namespace rnntrain {

/// Run the Luisa tile-language RNN sequence-classification training (the C++
/// twin of examples/tensor/rnn_train.py).  Parses the backend name from the
/// positional command-line arguments and the optional --epochs flag (the
/// --rnn dispatch flag itself is ignored here).  Returns the process exit
/// code (0 on success).
int run_rnn(int argc, char *argv[]);

}// namespace rnntrain
