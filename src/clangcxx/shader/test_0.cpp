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
    buffer.store(0, dispatch_id());
    return 0;
}