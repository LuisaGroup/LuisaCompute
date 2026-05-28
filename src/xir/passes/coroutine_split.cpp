#include <luisa/core/logging.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/ast/type.h>
#include <luisa/ast/type_registry.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/coroutine.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/unreachable.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coroutine.h>
#include <luisa/xir/passes/coroutine_split.h>
#include <luisa/xir/passes/coro_materialize.h>
#include <luisa/xir/undefined.h>

namespace luisa::compute::xir {

CoroutineSplitInfo coroutine_split_run_on_function(Function *function) noexcept {
    CoroutineSplitInfo info;
    if (function == nullptr || !function->is_definition()) {
        info.diagnostics.emplace_back("coroutine_split: function is null or not a definition");
        return info;
    }
    auto analysis = coroutine_analysis_run_on_function(function);
    info.diagnostics = analysis.diagnostics;
    if (!analysis.is_coroutine) { return info; }

    auto mat_result = coro::coro_materialize_run_on_function(function);
    if (mat_result.ok) {
        return mat_result.split_info;
    }
    info.diagnostics.insert(info.diagnostics.end(),
                            mat_result.diagnostics.begin(),
                            mat_result.diagnostics.end());
    info.is_supported = false;
    return info;
}

luisa::vector<CoroutineSplitInfo> coroutine_split_run_on_module(Module *module) noexcept {
    luisa::vector<CoroutineSplitInfo> results;
    if (module == nullptr) { return results; }
    luisa::vector<Function *> targets;
    for (auto function : module->function_list()) { targets.emplace_back(function); }
    for (auto function : targets) {
        auto sub = coroutine_split_run_on_function(function);
        if (!sub.continuations.empty() || !sub.diagnostics.empty()) {
            results.emplace_back(std::move(sub));
        }
    }
    return results;
}

}// namespace luisa::compute::xir
