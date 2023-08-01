#include <luisa/luisa-compute.h>

int main() {

    using namespace luisa;
    using namespace luisa::compute;

    Callable callable = [](Var<SOA<float3>> v, Var<SOA<uint>> u) noexcept {
        return v.read(u.read(0));
    };

    Kernel2D kernel = [&](Var<SOA<float3x3>> m, Var<SOA<std::array<uint3, 10>>> a) noexcept {
        auto x = callable(m[0], a[1].x);
    };

    Device *device;
    SOA<float3x3> m;
    SOA<std::array<uint3, 10>> a;
    auto shader = device->compile(kernel);
    auto cmd = shader(m, a).dispatch(1u, 1u);
}
