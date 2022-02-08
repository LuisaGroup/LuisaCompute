//
// Created by Mike Smith on 2022/2/7.
//

#include "../ispc_device_math.isph"

export void kernel_main() {
    int3 a = make_int3(1, 2, 3);
    vector_access(a, 0) = programIndex;
    print("% % % %", vector_access(a, 0), a._y, a._z, clamp(3, -2, 0));
}
