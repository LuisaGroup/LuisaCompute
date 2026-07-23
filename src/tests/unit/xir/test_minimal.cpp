
#include "ut/ut.hpp"
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/indvar_simplify.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;

int main() {
    "empty"_test = [] {
        Module m;
        auto info = indvar_simplify_pass_run_on_module(&m);
    };
    return 0;
}
