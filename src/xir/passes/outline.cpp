#include <luisa/xir/passes/outline.h>
#include <luisa/core/logging.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/outline.h>

namespace luisa::compute::xir {

OutlineInfo outline_pass_run_on_function(Module *module, Function *function) noexcept {
    static_cast<void>(module);
    OutlineInfo info;
    if (function == nullptr || function->definition() == nullptr) { return info; }
    function->definition()->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<OutlineInst>()) { ++info.unsupported_outline_count; }
    });
    if (info.unsupported_outline_count != 0u) {
        LUISA_WARNING_WITH_LOCATION(
            "Outline pass encountered {} OutlineInst node(s), but region "
            "outlining is not implemented. IR was left unchanged.",
            info.unsupported_outline_count);
    }
    return info;
}

OutlineInfo outline_pass_run_on_module(Module *module) noexcept {
    OutlineInfo info;
    luisa::vector<Function *> functions;
    for (auto f : module->function_list()) { functions.emplace_back(f); }
    for (auto f : functions) {
        auto function_info = outline_pass_run_on_function(module, f);
        info.outlined_func_count += function_info.outlined_func_count;
        info.unsupported_outline_count += function_info.unsupported_outline_count;
    }
    return info;
}

}// namespace luisa::compute::xir
