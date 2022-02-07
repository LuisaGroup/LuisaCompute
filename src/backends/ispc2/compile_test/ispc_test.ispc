//
// Created by Mike Smith on 2022/2/7.
//

#include "../ispc_device_math.isph"

export void kernel_main() {
    int3 a = make_int3(1u);
    print("% % %", a[0], a[1], a[2]);
}
