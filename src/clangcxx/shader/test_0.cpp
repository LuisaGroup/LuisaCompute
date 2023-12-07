#include "std.hpp"

using namespace luisa::shader;

namespace luisa::shader {
struct F {
    int f;
};
struct NVIDIA {
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
    float4 f4;
    float4x4 f44;
    // ! not supportted as field!
    // int ds[5];
    // Buffer<int> b;
};
}// namespace luisa::shader

// Buffer<NVIDIA> buffer;
template<typename T>
struct Holder {
    Holder(T v)
        : value(v), value2(v) {}
    void call() {
        if constexpr (is_floatN<T>::value) {
            value = 2.f;
            value2 = 2.f;
        } else if constexpr (is_intN<T>::value) {
            value = 0;
            value2 = 0;
        }
    }
    T value;
    T value2;
};

auto TestHolder()
{
    int v = 5;
    Holder h(v);
    h.call();
    return h;
}

[[kernel_2d(16, 16)]] 
int kernel(Buffer<NVIDIA> &buffer) 
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

    // template
    Holder h = TestHolder();
    int xxxx = nvidia.l += h.value;

    // call
    float fff = nvidia.f = sin(nvidia.f);

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