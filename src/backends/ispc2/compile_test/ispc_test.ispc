//
// Created by Mike Smith on 2022/2/7.
//

#include "../ispc_device_math.isph"

struct S_AE65E4A61E2BB371 {
    int m0;
    int m1;
};

make_array_type(A_4F70686130F49386, int, 128)

make_array_type(A_CCE8E08E78A4AB61, float, 2)

make_array_type(A_9E1BA40BDAF42D5E, float4, 16)

struct S_1CEF5BFA539C20E8 {
    int3 m0;
    float m1;
    uint8 pad_0[12];
};

inline S_AE65E4A61E2BB371 custom_A20A5F021ABD200D(
    const int v0,
    const int v1,
    const float *uniform b3) {
    int v2;
    float v4;
    int v5;
    int v6;
    S_AE65E4A61E2BB371 v7;
    v2 = binary_add(v0, v1);
    v4 = buffer_read(b3, v2);
    v5 = ((int)(v4));
    v6 = binary_mul(v0, v1);
    v7.m0 = v5;
    v7.m1 = v6;
    return v7;
}

inline int custom_5EF75AB570F42F30(
    const int v0,
    const int v1) {
    int v2;
    v2 = binary_add(v0, v1);
    return v2;
}

static const uniform A_4F70686130F49386 c82B7BAEDC8C4A038 = {{
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
    96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127}};

inline float custom_BC665AD2B8D69471(
    const int v0,
    const int v1,
    const float v2) {
    float v3;
    int v4;
    float v5;
    float v6;
    float v7;
    v3 = ((float)(v0));
    v4 = array_access(c82B7BAEDC8C4A038, v1);
    v5 = ((float)(v4));
    v6 = binary_mul(v5, v2);
    v7 = binary_add(v3, v6);
    return v7;
}

static const uniform A_CCE8E08E78A4AB61 cA541F95793F903F0 = {{1.0f, 2.0f}};

export void kernel_main() {
    int3 a = make_int3(1, 2, 3);
    vector_access(a, 0) = programIndex;
    print("% % % %\n", vector_access(a, 0), a._y, a._z, clamp(3, -2, 0));
    foreach (i = 0 ... 8) {
        print("program: %\n", programIndex);
    }
}
