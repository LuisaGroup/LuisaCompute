#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>
using namespace luisa::compute;

int main() {
    // Test 1: with param — should work
    {
        Coroutine c = [](Var<int> x) { $suspend("s"); };
        volatile auto sc = c.subroutine_count(); (void)sc;
    }
    // Test 2: no param — check if this crashes
    {
        Coroutine c = []() { $suspend("s"); };
        volatile auto sc = c.subroutine_count(); (void)sc;
    }
    return 0;
}
