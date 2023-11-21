#include "std.hpp"

using namespace luisa::shader;

namespace luisa::shader
{
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
    };
}

Buffer<NVIDIA> buffer;

[[kernel_2d(16, 16)]]
int main()
{
    auto n = NVIDIA{0};
    buffer.store(0, n);
    return 0;
}