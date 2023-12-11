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
    // ! C-Style array is not supportted !
    // int ds[5];
    // ! Buffer<> is not supportted as field !
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

struct TestCtor {
    TestCtor(int v)
        : x(-1/*unary -*/), xx(v) {
    }
    int x;
    int xx;
    int xxx = +2; /*unary +*/
};

/* dtors are not allowed
struct TestDtor {
    ~TestDtor() {
        f = 1.f;
    }
    float f = 22.f;
};
*/

auto TestBinary() {
    // binary op
    int n = 0 + 2 - 56;

    // binary assign ops
    int m = n += 65;
    int x = n -= 65;
    int xx = n *= 65;
    int yy = n /= 65;
    int ww = n %= 65;

    return m + x + xx + yy + ww;
}

auto TestUnary()
{
    int n = 0;
    int m = n++;
    int x = n--;
    int xx = ++n;
    int yy = --n;
    return m + x + xx + yy;
}

auto TestHolder() {
    int v = 5;
    Holder h(v);
    h.call();
    return h;
}

auto TestBranch() {
    if (sin(5.f) > 0.f)
        return 1.f;
    else if (cos(2.f) > 2.f)
        return 2.f;
    else
        return 3.f;

    if constexpr (is_floatN<float4>::value)
        return 4.f;
    else if constexpr (is_floatN<int4>::value)
        return 5.f;
    else
        return 6.f;
}

auto TestForLoop() {
    float f = 1.f;
    for (int i = 0; i < 10; i++) {
        if (i == 5)
            continue;
        f += static_cast<float>(i);
        f += (float)i;
    }
    return f;
}

auto TestWhileLoop() {
    float f = 1.f;
    int i = 0;
    while (i < 10) {
        f += static_cast<float>(i);
        f += (float)i;
        ++i;

        if (f > 10.f)
            break;
    }
    return f;
}

auto TestSwitch() {
    int i = 0;
    switch (i) {
    case 0:
        return 0.f;
    case 1:
        return 1.f;
    default:
        return 2.f;
    }
    return 3.f;
}

[[kernel_2d(16, 16)]] int kernel(Buffer<NVIDIA> &buffer) {
    // member assign
    NVIDIA nvidia = {};
    int i = nvidia.ix = is_floatN<int4>::value;

    // binary ops
    int ii = nvidia.i = TestBinary();
    
    // unary ops
    int iii = nvidia.i = TestUnary();

    // template
    Holder h = TestHolder();
    int xxxx = nvidia.l += h.value;

    // ctor
    TestCtor ctor(nvidia.ix);
    nvidia.i += ctor.x;

    // control_flows
    float f = nvidia.f += TestBranch();
    float ff = nvidia.f += TestWhileLoop();
    float fff = nvidia.f += TestForLoop();
    float ffff = nvidia.f += TestSwitch();

    // built-in call
    float _f = nvidia.f += sin(nvidia.f);

    // member call
    auto n = buffer.load(0);
    buffer.store(n.i, nvidia);

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