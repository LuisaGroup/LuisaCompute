#include "luisa/std.hpp"
#include "luisa/printer.hpp"
using namespace luisa::shader;

namespace luisa::shader {
[[kernel_1d(64)]] int kernel() {
    if (dispatch_id().x % 10 == 0)
        device_log("test printer: {}", dispatch_id().x);
    return 0;
}
}// namespace luisa::shader