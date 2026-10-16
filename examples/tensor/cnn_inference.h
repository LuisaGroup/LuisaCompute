// =============================================================================
// cnn_inference.h — TinyCNN new Tile DSL inference entry point
// =============================================================================
// The CNN inference driver (weights/im2col build + device dispatch + PyTorch
// verification) is invoked through the example_tensor main() with the `--cnn`
// flag:
//   example_tensor <backend> --cnn [cnn_input.bin] [--bench]
// The implementation lives in examples/tensor/cnn_inference.cpp.
// =============================================================================

#pragma once

namespace tensor_example::cnn {

/// Runs the TinyCNN inference ported to the new execution-structure-first
/// Tile DSL. Parses the backend name and the optional .bin path from the
/// non-flag command-line arguments (the --cnn dispatch flag itself is ignored
/// here) plus the --bench flag. Returns the process exit code (0 on success,
/// non-zero when any compile / verification step failed).
int run_cnn_inference(int argc, char *argv[]);

}// namespace tensor_example::cnn
