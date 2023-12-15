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
    float4 f4;

    float4x4 f44;
    // TODO: CTOR, COPY
    // Array<int, 3> a3;

    // ! C-Style array is not supportted !
    // int ds[5];
    // ! Buffer<> is not supportted as field !
    // Buffer<int> b;
};
}// namespace luisa::shader

// Buffer<NVIDIA> buffer;
template<typename T>
struct Template {
    Template(T v)
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

template<>
struct Template<NVIDIA> {
    Template(NVIDIA v)
        : nv(v) {}
    void call() {
        nv.f += 1.f;
    }
    NVIDIA nv;
};

struct TestCtor {
    TestCtor(int v)
        : x(-1 /*unary -*/), xx(v) {
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

    int v = 5;
    int complex = v += 5; 

    return m + x + xx + yy + ww + complex;
}

auto TestUnary() {
    int n = 0;
    int m = n++;
    int x = n--;
    int xx = ++n;
    int yy = --n;
    return m + x + xx + yy;
}

auto TestTemplate() {
    int v = 5;
    Template h(v);
    h.call();
    return h;
}

auto TestTemplateSpec() {
    NVIDIA nvidia = NVIDIA();
    Template h(nvidia);
    h.call();

    auto vec = make_vector(1.f, 2.f);
    auto vec2 = make_vector(1.f, vec);
    auto vec3 = make_vector(vec, 2.f);
    auto vec4 = make_vector(1.f, vec, 2.f);
    auto fsum = vec.y + vec2.z + vec3.z + vec4.w;

    auto ivec = make_vector(1, 2);
    auto ivec2 = make_vector(1, ivec);
    auto ivec3 = make_vector(ivec, 2);
    auto ivec4 = make_vector(1, ivec, 2);
    auto isum = ivec.y + ivec2.z + ivec3.z + ivec4.w;

    return nvidia.f + fsum + (float)isum;
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

auto TestVecOp()
{
    float2 d0 = float2(1.f, 2.f);
    float2 d1 = float2(2.f, 1.f);
    d0 = d0 = d0 + d1;
    d0 = d0 - d1;
    d0 = d0 * d1;
    d0 = d0 / d1;
    d0 = d0 % d1;
    float N = 5.f;
    d0 = N + d0;
    d0 = d0 + N;
    d0 = d0 - N;
    d0 = N * d0;
    d0 = d0 * N;
    d0 = d0 / N;
    d0 = d0 % N;
    return (float)d0.x;
}

auto TestSwizzle()
{
    float2 d = float2(1.f, 1.f);
    float dx = d.x;
    float dx2 = d.y;
    float2 dd = d.xy;
    float2 dd2 = d.yx;
    float dx3 = dd.y + dd2.x;
    float4 FFFF = float4(1.f, 2.f, 3.f, 4.f);
    float3 FFF = FFFF.wxz;
    float2 FF = FFF.zy;
    FF.x += 2.f;
    float dx4 = FF.x;
    return dx + dx2 + dx3 + dx4;
}

template <typename T>
auto TestArgsPack_Sum(T v)
{
    return v;
}

template <typename T, typename... Args>
auto TestArgsPack_Sum(T v, Args... args)
{
    return v + TestArgsPack_Sum(args...);
}

template <typename T>
void TestArgsPack_Div(T Div, T &v)
{
    v /= Div;
}

template <typename T, typename... Args>
void TestArgsPack_Div(T Div, T& v, Args&... args)
{
    TestArgsPack_Div(Div, v);
    TestArgsPack_Div(Div, args...);
}

template <typename... Args>
void TestArgsPack_Percentage(Args&... args)
{
    auto Sum = TestArgsPack_Sum(args...);
    TestArgsPack_Div(Sum, args...);
}

template <typename F, typename... Args>
auto TestInvoke(F func, Args&... args)
{
    return func(args...);
}

template <typename F, typename... Args>
auto TestInvokeInvoke(F func, Args&... args)
{
    return TestInvoke(func, args...);
}

#define WIDTH 3200u
#define HEIGHT 2400u
#define PI 3.141592653589793238462643383279502f
struct Pixel
{
    float4 value;
};

void mandelbrot(Buffer<float4> &mandelbrot_out, uint3 tid)
{
    if(tid.x >= WIDTH || tid.y >= HEIGHT)
        return;
    float x = float(tid.x) / WIDTH;
    float y = float(tid.y) / HEIGHT;
    float2 uv = float2(x, y);
    float n = 0.0f;
    float2 c = float2(-0.444999992847442626953125f, 0.0f);
    c = c + (uv - float2(0.5f, 0.5f)) * 2.3399999141693115234375f;
    float2 z = float2(0.f, 0.f);
    const int M =128;
    for (int i = 0; i < M; i++)
    {
        z = float2((z.x * z.x) - (z.y * z.y), (2.0f * z.x) * z.y) + c;
        if (dot(z, z) > 2.0f)
        {
            break;
        }
        n += 1.0f;
    }
    // we use a simple cosine palette to determine color:
    // http://iquilezles.org/www/articles/palettes/palettes.htm
    float t = float(n) / float(M);
    float3 d = float3(0.3f, 0.3f ,0.5f);
    float3 e = float3(-0.2f, -0.3f ,-0.5f);
    float3 f = float3(2.1f, 2.0f, 3.0f);
    float3 g = float3(0.0f, 0.1f, 0.0f);
    float4 color = float4(d + (e * cos(((f * t) + g) * 2.f * PI)), 1.0f);
    
    mandelbrot_out.store(WIDTH * tid.y + tid.x, color);
}

float TestOrder()
{
    float N = 1.f;
    float x = (N + 2.f) * 3.f;
    float xx = N + 2.f * 3.f;
    return x + xx;
}

[[kernel_2d(32, 32)]] 
int kernel(Buffer<NVIDIA> &buffer, Buffer<float4> &buffer2, Buffer<float4> &mandelbrot_out, Accel& accel) {
    // member assign
    NVIDIA nvidia = NVIDIA();
    int i = nvidia.ix = is_floatN<int4>::value;

    // binary ops
    int ii = nvidia.i = TestBinary();

    // unary ops
    int iii = nvidia.i = TestUnary();

    // order
    nvidia.f += TestOrder();

    // template
    Template h = TestTemplate();
    int xxxx = nvidia.l += h.value;
    nvidia.f += TestTemplateSpec();

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
    nvidia.u64 += accel.instance_user_id(0);

    // vec
    float4 a(1.f, 1.f, 1.f, 1.f);
    buffer2.store(0, a);

    // vec bin ops
    nvidia.f += TestVecOp();

    // swizzle
    nvidia.f += TestSwizzle();

    // lambda
    auto TestLambda = [&, _f](float v, float& vv){
        ff *= 1.f;
        vv /= 1.f;
        auto x = _f + ff - v;
        return x;
    };
    nvidia.f += TestLambda(nvidia.f, nvidia.f);

    // invoke
    auto TestLambda2 = [&]() { return _f + 1.f; };
    nvidia.f += TestInvoke(TestLambda2);
    nvidia.f += TestInvoke(TestLambda, nvidia.f, nvidia.f);
    nvidia.f += TestInvokeInvoke(TestLambda2);

    // args pack
    nvidia.f += TestArgsPack_Sum(1.f, 2.f, 3.f);
    float p0 = 1.f, p1 = 2.f, p2 = 3.f, p3 = 4.f;
    TestArgsPack_Percentage(p0, p1, p2);
    TestArgsPack_Percentage(p0, p1, p2, p3);

    // member call
    auto n = buffer.load(0);
    buffer.store(n.i, nvidia);

    // draw mandelbrot
    mandelbrot(mandelbrot_out, thread_id());

    // query
    const auto Origin = float3(0.f, 0.f, 0.f);
    const auto Direction = float3(1.f, 0.f, 0.f);
    accel.query_all(Ray(Origin, Direction));
    /*
    accel.query_all(Ray(Origin, Direction))
        .on_procedural_candidate(1)
        .on_surface_candidate(1)
        .trace();
    */

    return 0;
}