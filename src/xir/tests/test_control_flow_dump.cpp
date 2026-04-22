#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/luisa-compute.h>
#include <luisa/xir/passes/Canonicalize_Control_Flow.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/xir2text.h>

using namespace luisa;
using namespace luisa::compute;

int main() {
    Callable<int()> callable = []() noexcept {
        Int acc = 0;
        $for (i, 0, 8) {
            $if (i == 2) {
                $continue;
            };
            acc += i;
            $if (i == 5) {
                $return(acc + 10);
            };
            $if (i == 6) {
                $break;
            };
        };
        return acc + 1;
    };

    auto module = xir::ast_to_xir_translate(callable.function(), {});
    auto before = xir::xir_to_text_translate(module.get(), true);
    LUISA_INFO("Before canonicalize_control_flow:\n{}", before);

    auto info = xir::Canoinicalize_Control_Flow_pass_run_on_Module(module.get());
    auto after = xir::xir_to_text_translate(module.get(), true);
    LUISA_INFO("canonicalize_control_flow info: lowered_loop_count = {}, skipped_loop_count = {}",
               info.lowered_loop_count, info.skipped_loop_count);
    LUISA_INFO("After canonicalize_control_flow:\n{}", after);
    return 0;
}
