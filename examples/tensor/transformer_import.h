// transformer_import.h — torch2 transformer graph artifact importer
// =============================================================================
// C++ side of the torch.export -> LuisaCompute pipeline for the tiny transformer:
// parses the portable JSON artifact written by examples/tensor/transformer_train.py
// and executes the graph on a Luisa backend with tile-language kernels.
// =============================================================================
#pragma once
namespace transformer2 {
/// example_tensor_stub <backend> --transformer-pt2 [path.json] [--tol F]
///   backend : dx | vk | cuda
///   path.json : torch2 export artifact (default transformer_exported.pt2.json)
///   --tol F  : max absolute error vs the PyTorch reference (default 1e-3)
int run_transformer_import(int argc, char *argv[]);
}// namespace transformer2
