// =============================================================================
// cnn_inference.h — TinyCNN tile-language inference entry point
// =============================================================================
// The CNN inference driver (weights/im2col build + device dispatch + PyTorch
// verification + benchmark) is folded into the single `example_tensor_stub`
// target: `example_tensor_stub <backend> --cnn [cnn_input.bin] [--bench]`.
// The implementation lives in examples/tensor/cnn_inference.cpp.
// =============================================================================

#pragma once

namespace cnn {

/// Run the Luisa tile-language TinyCNN inference.  Parses the backend name and
/// the optional .bin path from the non-flag command-line arguments and the
/// --bench flag (the --cnn dispatch flag itself is ignored here).  Returns the
/// process exit code (0 on success).
int run_cnn_inference(int argc, char *argv[]);

}// namespace cnn
