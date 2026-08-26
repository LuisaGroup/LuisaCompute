// torch2_import.h — torch2 graph artifact importer (--rnn-pt2 driver)
// =============================================================================
// C++ side of the torch.export -> LuisaCompute pipeline: parses the portable
// JSON artifact written by examples/tensor/torch2_export.py and executes the
// graph on a Luisa backend with tile-language kernels (see torch2_kernels.h),
// cross-checked against a host CPU executor and the embedded PyTorch reference.
// =============================================================================
#pragma once
namespace torch2 {
/// example_tensor_stub <backend> --rnn-pt2 [path.json] [--tol F]
///   backend : dx | vk | cuda
///   path.json : torch2 export artifact (default rnn_exported.pt2.json)
///   --tol F  : max absolute error vs the PyTorch reference (default 1e-3)
int run_rnn_import(int argc, char *argv[]);
}// namespace torch2
