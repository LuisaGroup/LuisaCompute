#pragma once

#include <luisa/xir/module.h>

namespace luisa::compute::xir {

// Opaque to AST. Created before an AST builder escapes; neither the snapshot
// nor its values are exposed to backend passes. Each consumer imports its own
// mutable, selectively cloned dependency closure.
struct ASTFunctionCache final {
    luisa::shared_ptr<const Module> owner;
    const FunctionDefinition *root;
};

}// namespace luisa::compute::xir
