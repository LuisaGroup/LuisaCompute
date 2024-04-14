#include "luisa/std.hpp"
#include "luisa/type_traits/is_callable.hpp"

using namespace luisa::shader;

namespace luisa::shader {
struct Structure {
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
    Structure s;
    float4 f4;

    float2x2 f22;
    float3x3 f33;
    float4x4 f44;
    Array<int, 3> a3;

    // ! C-Style array is not supportted !
    // int ds[5];
    // ! Buffer<> is not supportted as field !
    // Buffer<int> b;
};
}// namespace luisa::shader

// Buffer<NVIDIA> buffer;
template<typename T>
struct Template {
    explicit Template(T v)
        : value(v), value2(v) {}
    void call() {
        if constexpr (is_sint_family_v<T>) {
            value = 2.f;
            value2 = 2.f;
        } else if constexpr (is_sint_family_v<T>) {
            value = 0;
            value2 = 0;
        }
    }
    T value;
    T value2;
};

template<>
struct Template<NVIDIA> {
    explicit Template(const NVIDIA &v)
        : nv(v) {}
    void call() {
        nv.f += 1.f;
    }
    NVIDIA nv;
};

struct TestCtorStructure {
    explicit TestCtorStructure(int v)
        : x(-1 /*unary -*/), xx(v) {
    }
    int x;
    int xx;
    float xxx = +2.f; /*unary +*/
};

export auto TestCtor() {
    // CallInit
    TestCtorStructure ctor0(5);
    auto ctor1(TestCtorStructure(TestCtorStructure(TestCtorStructure(3))));
    TestCtorStructure ctor2(ctor1);
    // CInit
    auto cinit0 = TestCtorStructure(5);
    auto cinit1 = TestCtorStructure(TestCtorStructure(TestCtorStructure(3)));
    return ctor0.xxx + ctor1.xxx + ctor2.xxx + cinit0.xxx + cinit1.xxx;
}

/* dtors are not allowed
struct TestDtor {
    ~TestDtor() {
        f = 1.f;
    }
    float f = 22.f;
};
*/

export auto TestBuiltinExprs() {
    float3 did = dispatch_id();
    float3 bid = block_id();
    float3 tid = thread_id();
    float3 ds = dispatch_size();
    auto kid = (float)kernel_id();
    auto wlc = (float)warp_lane_count();
    auto wli = (float)warp_lane_id();
    auto _d = dot(float3(1.f, 1.f, 1.f), float3(2.f, 2.f, 2.f));
    static_assert(is_same_v<decltype(_d), float>);
    return did.x + bid.x + tid.x + ds.x + kid + wlc + wli + _d;
}

export auto TestCast() {
    int i = 5;
    const auto fx = bit_cast<float>(i);
    const auto f0 = (float)i;
    const auto f1 = float(i);
    const auto f2 = static_cast<float>(i);
    return f0 + f1 + f2;
}

export auto TestBinary() {
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

export auto TestUnary() {
    int n = 0;
    int x = n++;
    int y = n--;
    int xx = ++n;
    int yy = --n;
    return x + y + xx + yy;
}

template<concepts::array A, typename Predicate>
void bubble_sort(A &arr, Predicate pred) {
    auto n = A::N;
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (pred(arr[j], arr[j + 1])) {
                const auto temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
}

export auto TestArray() {
    // default ctor
    auto a = Array<float, 4>();
    // assign ctor
    auto b = Array<float, 4>(1.f, 2.f, 3.f, 4.f);
    b[0] = 1.f;
    b.set(1, 2.f);
    // copy
    auto z = a;
    z = b;
    float sum = 0.f;
    for (int i = 0; i < 4; i++) {
        sum += z[i];
        sum += z.get(i);
    }

    int i = -1;
    auto arr = Array<float, 4>(1.f, 2.f, 3.f, 4.f);
    bubble_sort(arr, [&](float lhs, float rhs) {
        i++;
        if (i > 2)
            return abs(lhs) > abs(rhs);
        else
            return abs(lhs) < abs(rhs);
    });

    return 2.f;
}

export auto TestTemplate() {
    int v = 5;
    Template h(v);
    h.call();
    return h;
}

export auto TestTemplateSpec() {
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
    if constexpr (is_float_family_v<float4>)
        return 4.f;
    else if constexpr (is_float_family_v<int4>)
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

auto TestVecOp() {
    float2 d0 = float2(1.f, 2.f);
    float2 d1 = float2(2.f, 1.f);
    d0 = d0 + d1;
    d0 = d0 - d1;
    d0 = d0 * d1;
    d0 = d0 / d1;
    float N = 5.f;
    d0 = N + d0;
    d0 = d0 + N;
    d0 = d0 - N;
    d0 = N * d0;
    d0 = d0 * N;
    d0 = d0 / N;
    uint2 u0 = uint2(1, 2);
    uint2 u1 = uint2(2, 1);
    uint32 M = 5;
    u0 = u0 % u1;
    u0 = u0 % M;
    return (float)u0.x + d0.x;
}

auto TestSwizzle() {
    float4 FFFF = float4(1.f, 2.f, 3.f, 4.f);

    static_assert(is_float_family_v<float4 &>);
    static_assert(sizeof(FFFF.x) == sizeof(float));
    static_assert(sizeof(FFFF.xx) == sizeof(float2));
    static_assert(sizeof(FFFF.xxx) == sizeof(float3));
    static_assert(sizeof(FFFF.xxxx) == sizeof(float4));
    static_assert(sizeof(float2) == 2 * sizeof(float));
    static_assert(sizeof(float3) == 4 * sizeof(float));
    static_assert(sizeof(float4) == 4 * sizeof(float));

    float2 d = float2(1.f, 1.f);
    float dx = d.x;
    float dx2 = d.y;
    float2 dd = d.xy;
    float2 dd2 = d.yx;
    float dx3 = dd.y + dd2.x;
    float3 FFF = FFFF.wxz;
    float2 FF = FFF.zy;
    FF.x += 2.f;
    auto dx4 = FF.x;
    return dx + dx2 + dx3 + dx4;
}

template<typename T>
auto TestArgsPack_Sum(T v) {
    return v;
}

template<typename T, typename... Args>
auto TestArgsPack_Sum(T v, Args... args) {
    return v + TestArgsPack_Sum(args...);
}

template<typename T>
void TestArgsPack_Div(T Div, T &v) {
    v /= Div;
}

template<typename T, typename... Args>
void TestArgsPack_Div(T Div, T &v, Args &...args) {
    TestArgsPack_Div(Div, v);
    TestArgsPack_Div(Div, args...);
}

template<typename... Args>
void TestArgsPack_Percentage(Args &...args) {
    auto Sum = TestArgsPack_Sum(args...);
    TestArgsPack_Div(Sum, args...);
}

template<typename F, typename... Args>
auto TestInvoke(F func, Args &...args) {
    return func(args...);
}

template<typename F, typename... Args>
auto TestInvokeInvoke(F func, Args &...args) {
    return TestInvoke(func, args...);
}

#define WIDTH 3200u
#define HEIGHT 2400u
struct Pixel {
    float4 value;
};

float TestOrder() {
    float N = 1.f;
    float x = (N + 2.f) * 3.f;
    float xx = N + 2.f * 3.f;
    return x + xx;
}

auto TestIgnoreReturn(float &f) {
    f += 2.f;
    return 2.f;
}

static constexpr auto c_f3 = identity<float3>;

constexpr auto c_arr = Array<float, 2>(1.f, 2.f);
constexpr auto c_arr2 = Array<float, 2>(3.f, 4.f);
static_assert(c_arr[0] == 1.f);
static_assert(c_arr[1] == 2.f);

constexpr matrix<2> c_f22 = matrix<2>(1.f, 2.f, 3.f, 4.f);
static_assert(c_f22.get(0, 1) == 2.f);
static_assert(c_f22.get(1, 0) == 3.f);
static_assert(c_f22.get(1, 1) == 4.f);

constexpr auto c_f33 = matrix<3>(c_f22);
static_assert(c_f33.get(0, 0) == 1.f);
static_assert(c_f33[0, 0] == 1.f);
static_assert(c_f33.get(0, 2) == 0.f);

struct Complex {
    constexpr Complex(int i)
        : i(i) {}
    int i = 5;
    int ix = 25;
    Array<float, 2> a = Array<float, 2>(1.f, 3.f);
    float2 f2 = float2(2.f, 4.f);
};

struct ComplexComplex {
    constexpr ComplexComplex(int i)
        : c(i) {}
    Complex c;
    float f = 3.f;
    matrix<3> m = identity<matrix<3>>;
};
constexpr auto c_complex = Complex(33);
constexpr auto c_complexcomplex = ComplexComplex(666);

auto TestConstexprAssign() {
    decay_t<decltype(c_arr)> arr = c_arr;
    arr = c_arr2;

    float3 f3 = c_f3;
    f3 = c_f3;
    f3.x = c_f33[0, 2];

    matrix<3> f33 = identity<matrix<3>>;
    f33 = c_f33;

    auto f22 = matrix<2>(c_f33);

    Complex complex(22);
    complex = c_complex;

    ComplexComplex complexcomplex(44);
    complexcomplex = c_complexcomplex;

    return (float)(complexcomplex.c.i + complex.i) + arr[0];
}

export auto TestMatrix() {
    auto f22 = matrix<2>(1.f, 2.f, 3.f, 4.f);
    f22 = f22 * f22;
    auto f33 = matrix<3>(f22);
    auto f3 = f33[0];
    f3 = identity<float3>;
    auto i3 = identity<uint3>;
    return f33.get(1, 2);
}

template<typename T, uint32 StackSize>
struct FixedVector {
    [[nodiscard]] uint32 capacity() const { return StackSize; }
    [[nodiscard]] uint32 size() const { return size_; }
    void emplace_back(T v) {
        a.set(size_, v);
        size_ += 1;
    }
    [[nodiscard]] T get(uint32 i) const { return a.get(i); }
private:
    uint32 size_ = 0;
    Array<T, StackSize> a;
};

export auto TestVector() {
    FixedVector<float, 32> fs;
    fs.emplace_back(1.f);
    fs.emplace_back(12.f);
    fs.emplace_back(3.f);
    float sum = 0.f;
    for (int i = 0; i < fs.size(); i++)
        sum += fs.get(i);
    return sum;
}

[[kernel_2d(32, 32)]] int kernel(Buffer<NVIDIA> &buffer, Buffer<float4> &buffer2, Buffer<float4> &mandelbrot_out, Accel &accel) {
    // member assign
    NVIDIA nvidia = NVIDIA();
    uint32 i0 = 4294967294;
    uint32 i01 = 4294967295u;
    int32 i03 = -94967;
    int i = nvidia.ix = is_float_family_v<int4> + i01 + i0 + i03;

    // copy
    NVIDIA nvidia2 = nvidia;
    nvidia2 = nvidia;

    // binary & unary ops
    int ii = nvidia.i = TestBinary();
    int iii = nvidia.i = TestUnary();

    nvidia.f += TestBuiltinExprs();
    nvidia.f += TestCast();
    nvidia.f += TestOrder();
    nvidia.f += TestConstexprAssign();

    // array & vector & matrix
    nvidia.f += TestArray();
    nvidia.f += TestVector();
    nvidia.f += TestMatrix();

    // template
    Template h = TestTemplate();
    int xxxx = nvidia.l += h.value;
    nvidia.f += TestTemplateSpec();

    // ctor
    nvidia.f += TestCtor();

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
    auto TestLambda = [&, _f](float v, float &vv) {
        ff *= 1.f;
        vv /= 1.f;
        auto x = _f + ff - v;
        return x;
    };
    nvidia.f += TestLambda(nvidia.f, nvidia.f);

    // invoke
    auto TestLambda2 = [&]() { return _f + 1.f; };
    nvidia.f += TestInvoke([&]() { return _f + 1.f; });
    nvidia.f += TestInvoke(TestLambda2);
    nvidia.f += TestInvoke(TestLambda, nvidia.f, nvidia.f);
    nvidia.f += TestInvokeInvoke(TestLambda2);

    // args pack
    nvidia.f += TestArgsPack_Sum(1.f, 2.f, 3.f);
    float p0 = 1.f, p1 = 2.f, p2 = 3.f, p3 = 4.f;
    TestArgsPack_Percentage(p0, p1, p2);
    TestArgsPack_Percentage(p0, p1, p2, p3);

    // ignore return value
    TestIgnoreReturn(nvidia.f);

    // query
    const auto Origin = float3(0.f, 0.f, 0.f);
    const auto Direction = float3(1.f, 0.f, 0.f);
    auto query = accel.query_all(Ray(Origin, Direction));
    while (query.proceed()) {
        if (query.is_triangle_candidate()) {
            auto hit = query.triangle_candidate();
            nvidia.f += Origin.x;
            nvidia.i -= static_cast<int>(hit.inst);
            query.commit_procedural(1.f);
            query.terminate();
        }
    }

    // member call
    auto n = buffer.load(0);
    buffer.store(n.i, nvidia);

    return 0;
}