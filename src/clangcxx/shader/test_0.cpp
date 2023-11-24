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
        int ix;
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
int kernel(Buffer<NVIDIA> buffer)
{
    // binary op
    int n = 0 + 2 - 56;

    // binary assign ops
    int m = n += 65;
    int x = n -= 65;
    int xx = n *= 65;
    int yy = n /= 65;
    int ww = n %= 65;

    // member assign
    NVIDIA nvidia = {};
    int i = nvidia.i = n;
    int ii = nvidia.ix = n;

    // member call
    // n = buffer.load(0);
    buffer.store(0, nvidia);

    /*
    // lambda
    auto l = [=](int i){
        int x = n + 1;
        return x;
    };
    auto f = l(2);
    */

    return 0;
}