//
// LLVM adapter for generated-callable inlining decisions.
//

#pragma once

#include "hip_llvm_pipeline.h"

#include <vector>

namespace llvm {
class Function;
class Module;
}// namespace llvm

namespace luisa::compute::hip {

struct GeneratedCallableInlineGraph {
    std::vector<llvm::Function *> functions;
    std::vector<GeneratedCallableInlineGraphNode> nodes;
};

// Extracts generated-callable direct call sites and mutually exclusive switch
// frontiers from LLVM CFGs. The `generated_attribute` parameter is explicit so
// the adapter is independently testable without the rest of HIP codegen.
[[nodiscard]] GeneratedCallableInlineGraph
build_generated_callable_inline_graph(
    llvm::Module &module,
    const char *generated_attribute) noexcept;

}// namespace luisa::compute::hip
