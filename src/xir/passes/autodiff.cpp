#include <luisa/core/logging.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/undefined.h>
#include <luisa/xir/instructions/autodiff.h>
#include <luisa/xir/passes/autodiff.h>
#include <luisa/xir/passes/dom_tree.h>
#include "helpers.h"

namespace luisa::compute::xir {

struct TransformAdScope {
    Function *function{};
    AutodiffScopeInst *ad_scope{};
    void run() {
    }
};

struct AutodiffPass {
    Function *function{};
    AutodiffOptions options{};
    [[nodiscard]] auto locate_autodiff_scopes() const {
        auto def = function->definition();
        auto dom = compute_dom_tree(def);
        luisa::vector<AutodiffScopeInst *> ad_scopes;

        def->traverse_instructions([&](Instruction *inst) {
            if (inst->isa<AutodiffScopeInst>()) {
                auto ad_scope = static_cast<AutodiffScopeInst *>(inst);
                ad_scopes.emplace_back(ad_scope);
                LUISA_INFO("Found autodiff scope: {}", ad_scope->name().value_or("unnamed"));
            }
        });

        return ad_scopes;
    }
    void run() {
        if (!function->definition()) {
            return;
        }
        auto scopes = locate_autodiff_scopes();
        for (auto scope : scopes) {
            TransformAdScope transform{function, scope};
            transform.run();
        }
    }
};

LUISA_XIR_API void autodiff_pass_run_on_function(Function *function, const AutodiffOptions &options) noexcept {
    AutodiffPass pass{function, options};
    pass.run();
}

LUISA_XIR_API void autodiff_pass_run_on_module(Module *module, const AutodiffOptions &options) noexcept {
    for (auto func : module->function_list()) {
        autodiff_pass_run_on_function(func, options);
    }
}

}// namespace luisa::compute::xir
