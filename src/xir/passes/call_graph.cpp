#include <luisa/core/logging.h>
#include <luisa/xir/module.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/passes/call_graph.h>

namespace luisa::compute::xir {

inline void CallGraph::_add_function(Function *f) noexcept {
    if (f == nullptr) { return; }
    auto any_caller = false;
    for (auto &&use : f->use_list()) {
        if (auto user = use->user(); user != nullptr && user->isa<CallInst>()) {
            auto call = static_cast<CallInst *>(user);
            // Function values may also appear as ordinary operands in a
            // partially constructed module. Only operand zero is a call-graph
            // edge; treating an argument use as the callee invents an edge to
            // the unrelated call target and can hide a real root.
            if (call->callee() != f ||
                use != call->operand_use(CallInst::operand_index_callee)) {
                continue;
            }
            auto *caller_function = call->parent_function();
            auto *caller = caller_function == nullptr ?
                               nullptr :
                               caller_function->definition();
            if (caller == nullptr) { continue; }
            _call_edges[caller].emplace_back(call);
            any_caller = true;
        }
    }
    if (!any_caller) { _root_functions.emplace_back(f); }
}

luisa::span<Function *const> CallGraph::root_functions() const noexcept {
    return luisa::span{_root_functions};
}

luisa::span<CallInst *const> CallGraph::call_edges(FunctionDefinition *f) const noexcept {
    auto iter = _call_edges.find(f);
    return iter == _call_edges.cend() ? luisa::span<CallInst *const>{} : luisa::span{iter->second};
}

CallGraph compute_call_graph(Module *module) noexcept {
    CallGraph graph;
    if (module != nullptr) {
        for (auto f : module->function_list()) { graph._add_function(f); }
    }
    return graph;
}

}// namespace luisa::compute::xir
