#include "std.hpp"

using namespace luisa::shader;

namespace luisa::shader
{
    struct F
    {
        int f;
    };
    struct NVIDIA
    {
        int i;
        long l;
        long long ll;
        uint64 u64;
        float f;
        short ss;
        double d;
        float3 f3;
        int3 i3;
        uint3 u3;
        F fuck;
        Array<int, 3> a3;
        // ! not supportted ! 
        // int ds[5];
        // Buffer<int> b;
    };
}

// Buffer<NVIDIA> buffer;

[[kernel_2d(16, 16)]]
int kernel(Buffer<int> buffer)
{
    int n = 0 + 2 - 56;
    int m = n += 65;
    int x = n -= 65;
    int xx = n *= 65;
    int yy = n /= 65;
    int ww = n %= 65;
    // NVIDIA n = {};
    buffer.store(0, n);

    return 0;
}