#include "std.hpp"

using namespace luisa::shader;

namespace luisa::shader
{
    struct NVIDIA
    {
        int b;
    };
}

Buffer<uint3> buffer;

[[kernel_2d(16, 16)]]
int main()
{
    auto ff = ( int* )nullptr;
    auto f3 = uint3(uint2(1u, 1u), 1u);
    buffer.store(0, dispatch_id());
    buffer.store(0, f3);
    return 0;
}