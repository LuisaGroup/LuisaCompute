// =============================================================================
// mlp.h — MLP training tile-language entry point
// =============================================================================
// The MLP driver (host data build + device training loop + host reference
// verification) is folded into the single `example_tensor_stub` target:
// `example_tensor_stub <backend> --mlp [--epochs N]`.
// The implementation lives in examples/tensor/mlp.cpp.
// =============================================================================

#pragma once

namespace mlptrain {

/// Run the Luisa tile-language 3-layer MLP training (the C++ twin of
/// examples/tensor/mlp_train.py).  Parses the backend name from the positional
/// command-line arguments and the optional --epochs flag (the --mlp dispatch
/// flag itself is ignored here).  Returns the process exit code (0 on success).
int run_mlp(int argc, char *argv[]);

}// namespace mlptrain
