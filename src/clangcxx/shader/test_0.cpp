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

Buffer<NVIDIA> buffer;

[[kernel_2d(16, 16)]]
int main()
{
    NVIDIA n = {};
    buffer.store(0, n);
    int a = 13;
    int b {13};
    int c (13);
    int d = int{13};
    int e = int(13);
    int f = {13};
    int g = (13);
    return 0;
}