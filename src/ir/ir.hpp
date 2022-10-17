#pragma once

#include <bindings.hpp>

#include <core/dll_export.h>
#include <ast/function_builder.h>

namespace luisa::compute {
LC_IR_API void convert_to_ast(const ir::Module *module, detail::FunctionBuilder *builder) noexcept;
LC_IR_API ir::Module convert_to_ir(const ScopeStmt *stmt) noexcept;
inline ir::NodeRef build_call(ir::IrBuilder *builder, ir::Func func, luisa::vector<ir::NodeRef> args, const ir::Type *ret_type) noexcept {
    return ir::luisa_compute_ir_build_call(builder, func, {args.data(), args.size()}, ret_type);
}
struct NodeRefHash {
    [[nodiscard]] size_t operator()(ir::NodeRef ref) const noexcept {
        return ref._0;
    }
};

// For future
// struct AutodiffResult {
//     const ir::BasicBlock *forward = nullptr;
//     const ir::BasicBlock *backward = nullptr;
//     luisa::unordered_map<ir::NodeRef, ir::NodeRef, NodeRefHash> grads;
// };
struct AutodiffResult {
    luisa::unordered_map<Variable, Variable> grads;
};
LC_IR_API void begin_autodiff() noexcept;
LC_IR_API AutodiffResult end_autodiff() noexcept;
}// namespace luisa::compute
