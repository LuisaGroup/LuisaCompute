#include <luisa/std.hpp>
using namespace luisa::shader;

[[kernel_1d(1)]] int kernel(BufferView<uint32> buf)
{
    auto v = atomic_add(buf[0], 1u);
    v = atomic_compare_exchange<uint32>(buf[0], 0, 1);
    v = atomic_exchange<uint32>(buf[0], 1);
    v = atomic_sub<uint32>(buf[0], 1);
    v = atomic_and<uint32>(buf[0], 1);
    v = atomic_or<uint32>(buf[0], 1);
    v = atomic_xor<uint32>(buf[0], 1);
}