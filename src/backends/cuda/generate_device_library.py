from os.path import realpath, dirname
from sys import argv

HALF_IMPL = '''
struct lc_half {
private:
    union U { __fp16 h; lc_ushort bits; };
public:
    lc_ushort bits;
    inline constexpr lc_half() noexcept : bits{0} {}
    [[nodiscard]] static inline constexpr auto from_bits(lc_ushort bits) noexcept {
        lc_half h;
        h.bits = bits;
        return h;
    }
    inline constexpr lc_half(float x) noexcept {
        U u;
        u.h = x;
        bits = u.bits;
    }
    template<typename T>
    inline constexpr operator T() const noexcept {
        U u;
        u.bits = bits;
        return static_cast<T>(static_cast<float>(u.h));
    }
    inline constexpr auto operator-() const noexcept { return from_bits(bits ^ 0x8000u); }
    inline constexpr auto operator+() const noexcept { return *this; }
    inline constexpr auto operator!() const noexcept { return bits == 0u || bits == 0x8000u; }
#define IMPL_HALF_BINOP(op)                                         \
    inline constexpr auto operator op(lc_half rhs) const noexcept { \
        U u_lhs; u_lhs.bits = bits;                                 \
        U u_rhs; u_rhs.bits = rhs.bits;                             \
        return lc_half{lc_float(u_lhs.h op u_rhs.h)};               \
    }
    IMPL_HALF_BINOP(+)
    IMPL_HALF_BINOP(-)
    IMPL_HALF_BINOP(*)
    IMPL_HALF_BINOP(/)
#undef IMPL_HALF_BINOP
#define IMPL_HALF_CMP(op) inline constexpr auto operator op(lc_half rhs) const noexcept { return float(*this) op float(rhs); }
    IMPL_HALF_CMP(==)
    IMPL_HALF_CMP(!=)
    IMPL_HALF_CMP(<)
    IMPL_HALF_CMP(<=)
    IMPL_HALF_CMP(>)
    IMPL_HALF_CMP(>=)

};
static_assert(sizeof(lc_half) == 2);
[[nodiscard]] inline lc_short __half_as_short(lc_half x) noexcept { 
    return x.bits;
}
[[nodiscard]] inline lc_half __short_as_half(lc_short x) noexcept {
    return lc_half::from_bits(x);
}
[[nodiscard]] inline lc_half __hmax(lc_half x, lc_half y) noexcept { return lc_half{lc_float(x) > lc_float(y) ? x : y}; }
[[nodiscard]] inline lc_half __hmin(lc_half x, lc_half y) noexcept { return lc_half{lc_float(x) < lc_float(y) ? x : y}; }
[[nodiscard]] inline lc_half __habs(lc_half x) noexcept { return lc_half{lc_float(x) < 0.0f ? -x : x}; }
[[nodiscard]] inline lc_half hexp2(lc_half x) noexcept { return lc_half{exp2f(lc_float(x))}; }
[[nodiscard]] inline lc_half hceil(lc_half x) noexcept { return lc_half{ceilf(lc_float(x))}; }
[[nodiscard]] inline lc_half hfloor(lc_half x) noexcept { return lc_half{floorf(lc_float(x))}; }
[[nodiscard]] inline lc_half htrunc(lc_half x) noexcept { return lc_half{truncf(lc_float(x))}; }
[[nodiscard]] inline lc_half hround(lc_half x) noexcept { return lc_half{roundf(lc_float(x))}; }
[[nodiscard]] inline lc_half hsqrt(lc_half x) noexcept { return lc_half{sqrtf(lc_float(x))}; }
[[nodiscard]] inline lc_half hrsqrt(lc_half x) noexcept { return lc_half{rsqrtf(lc_float(x))}; }
[[nodiscard]] inline lc_half __hfma(lc_half x, lc_half y, lc_half z) noexcept { return lc_half{fmaf(lc_float(x), lc_float(y), lc_float(z))}; }
[[nodiscard]] inline bool __hisnan(lc_half x) noexcept { return isnan_impl(lc_float(x)); }
[[nodiscard]] inline bool __hisinf(lc_half x) noexcept { return isinf_impl(lc_float(x)); }
'''

if __name__ == "__main__":
    if len(argv) < 2:
        print("usage: python generate_device_library.py <output_file>")
        exit(1)
    output_file_name = argv[1]
    is_cpu = 'cpu' in output_file_name
    with open(f"{output_file_name}", "w") as file:
        # scalar types
        scalar_types = ["byte", "ubyte", "short", "ushort", "int",
                        "uint", "half", "float", "bool", "long", "ulong"]
        native_types = ["char", "unsigned char", "short", "unsigned short", "int", "unsigned int",
                        "half", "float", "bool", "long long", "unsigned long long"]
        scalar_alignments = {
            "byte": 1,
            "ubyte": 1,
            "short": 2,
            "ushort": 2,
            "int": 4,
            "uint": 4,
            "half": 2,
            "float": 4,
            "bool": 1,
            "long": 8,
            "ulong": 8,
        }
        for t, native_t in zip(scalar_types, native_types):
            if t == 'half' and is_cpu:
                continue
            print(f"using lc_{t} = {native_t};", file=file)
        print(file=file)
        print('''[[nodiscard]] __device__ inline bool isinf_impl(lc_float x) noexcept {
    auto u = __float_as_int(x);
    return u == 0x7f800000u | u == 0xff800000u;
}
[[nodiscard]] __device__ inline bool isnan_impl(lc_float x) noexcept {
    auto u = __float_as_int(x);
    return ((u & 0x7F800000u) == 0x7F800000u) & ((u & 0x7FFFFFu) != 0u);
}
[[nodiscard]] __device__ inline lc_float powi_impl(lc_float x, lc_int y) noexcept {
    lc_float r = 1.0f;
    auto is_y_neg = y < 0;
    auto y_abs = is_y_neg ? -y : y;

    while (y_abs) {
        if (y_abs & 1) r *= x;
        x *= x;
        y_abs >>= 1;
    }
    return is_y_neg ? 1.0f / r : r;
}
[[nodiscard]] __device__ inline lc_float powf_impl(lc_float x, lc_float y) noexcept {
    auto y_int = static_cast<lc_int>(y);
    return y_int == y ? powi_impl(x, y_int) : powf(x, y);
}
''', file=file)
        if is_cpu:
            print(HALF_IMPL, file=file)
        # vector types
        for type in scalar_types:
            for i in range(2, 5):
                align = min(16, scalar_alignments[type] * (i if i != 3 else 4))
                elements = ["x", "y", "z", "w"][:i]
                print(
                    f"""struct alignas({align}) lc_{type}{i} {{
    lc_{type} {', '.join(elements[:i + 1])};
    __device__ inline constexpr lc_{type}{i}() noexcept
        : {', '.join(f"{m}{{}}" for m in elements)} {{}}
    __device__ inline constexpr static auto zero() noexcept {{ return lc_{type}{i}{{}}; }}
    __device__ inline constexpr static auto one() noexcept {{ return lc_{type}{i}{{{', '.join('1' for _ in elements)}}}; }}
    __device__ inline explicit constexpr lc_{type}{i}(lc_{type} s) noexcept
        : {', '.join(f"{m}{{s}}" for m in elements)} {{}}
    __device__ inline constexpr lc_{type}{i}({', '.join(f"lc_{type} {m}" for m in elements)}) noexcept
        : {', '.join(f"{m}{{{m}}}" for m in elements)} {{}}
    __device__ inline constexpr auto &operator[](lc_uint i) noexcept {{ return (&x)[i]; }}
    __device__ inline constexpr auto operator[](lc_uint i) const noexcept {{ return (&x)[i]; }}
}};
""", file=file)

        # make type[n]
        for type in scalar_types:
            # make type2
            print(
                f"""[[nodiscard]] __device__ inline constexpr auto lc_make_{type}2(lc_{type} s = 0) noexcept {{ return lc_{type}2{{s, s}}; }}
[[nodiscard]] __device__ inline constexpr auto lc_make_{type}2(lc_{type} x, lc_{type} y) noexcept {{ return lc_{type}2{{x, y}}; }}""",
                file=file)
            for t in scalar_types:
                for l in range(2, 5):
                    print(
                        f"[[nodiscard]] __device__ inline constexpr auto lc_make_{type}2(lc_{t}{l} v) noexcept {{ return lc_{type}2{{static_cast<lc_{type}>(v.x), static_cast<lc_{type}>(v.y)}}; }}",
                        file=file)
            # make type3
            print(
                f"""[[nodiscard]] inline __device__ constexpr auto lc_make_{type}3(lc_{type} s = 0) noexcept {{ return lc_{type}3{{s, s, s}}; }}
[[nodiscard]] __device__ inline constexpr auto lc_make_{type}3(lc_{type} x, lc_{type} y, lc_{type} z) noexcept {{ return lc_{type}3{{x, y, z}}; }}
[[nodiscard]] __device__ inline constexpr auto lc_make_{type}3(lc_{type} x, lc_{type}2 yz) noexcept {{ return lc_{type}3{{x, yz.x, yz.y}}; }}
[[nodiscard]] __device__ inline constexpr auto lc_make_{type}3(lc_{type}2 xy, lc_{type} z) noexcept {{ return lc_{type}3{{xy.x, xy.y, z}}; }}""",
                file=file)
            for t in scalar_types:
                for l in range(3, 5):
                    print(
                        f"[[nodiscard]] __device__ constexpr auto lc_make_{type}3(lc_{t}{l} v) noexcept {{ return lc_{type}3{{static_cast<lc_{type}>(v.x), static_cast<lc_{type}>(v.y), static_cast<lc_{type}>(v.z)}}; }}",
                        file=file)
            # make type4
            print(
                f"""[[nodiscard]] inline __device__ constexpr auto lc_make_{type}4(lc_{type} s = 0) noexcept {{ return lc_{type}4{{s, s, s, s}}; }}
[[nodiscard]] __device__ inline constexpr auto lc_make_{type}4(lc_{type} x, lc_{type} y, lc_{type} z, lc_{type} w) noexcept {{ return lc_{type}4{{x, y, z, w}}; }}
[[nodiscard]] __device__ inline constexpr auto lc_make_{type}4(lc_{type} x, lc_{type} y, lc_{type}2 zw) noexcept {{ return lc_{type}4{{x, y, zw.x, zw.y}}; }}
[[nodiscard]] __device__ inline constexpr auto lc_make_{type}4(lc_{type} x, lc_{type}2 yz, lc_{type} w) noexcept {{ return lc_{type}4{{x, yz.x, yz.y, w}}; }}
[[nodiscard]] __device__ inline constexpr auto lc_make_{type}4(lc_{type}2 xy, lc_{type} z, lc_{type} w) noexcept {{ return lc_{type}4{{xy.x, xy.y, z, w}}; }}
[[nodiscard]] __device__ inline constexpr auto lc_make_{type}4(lc_{type}2 xy, lc_{type}2 zw) noexcept {{ return lc_{type}4{{xy.x, xy.y, zw.x, zw.y}}; }}
[[nodiscard]] __device__ inline constexpr auto lc_make_{type}4(lc_{type} x, lc_{type}3 yzw) noexcept {{ return lc_{type}4{{x, yzw.x, yzw.y, yzw.z}}; }}
[[nodiscard]] __device__ inline constexpr auto lc_make_{type}4(lc_{type}3 xyz, lc_{type} w) noexcept {{ return lc_{type}4{{xyz.x, xyz.y, xyz.z, w}}; }}""",
                file=file)
            for t in scalar_types:
                print(
                    f"[[nodiscard]] inline __device__ constexpr auto lc_make_{type}4(lc_{t}4 v) noexcept {{ return lc_{type}4{{static_cast<lc_{type}>(v.x), static_cast<lc_{type}>(v.y), static_cast<lc_{type}>(v.z), static_cast<lc_{type}>(v.w)}}; }}",
                    file=file)
            print(file=file)

        # unary operators
        for type in scalar_types:
            for i in range(2, 5):
                elements = ["x", "y", "z", "w"][:i]
                print(
                    f"[[nodiscard]] inline __device__ constexpr auto operator!(lc_{type}{i} v) noexcept {{ return lc_make_bool{i}({', '.join(f'!v.{m}' for m in elements)}); }}",
                    file=file)
                if type != "bool":
                    print(
                        f"[[nodiscard]] inline __device__ constexpr auto operator+(lc_{type}{i} v) noexcept {{ return lc_make_{type}{i}({', '.join(f'+v.{m}' for m in elements)}); }}",
                        file=file)
                    print(
                        f"[[nodiscard]] inline __device__ constexpr auto operator-(lc_{type}{i} v) noexcept {{ return lc_make_{type}{i}({', '.join(f'-v.{m}' for m in elements)}); }}",
                        file=file)
                    if type != "float" and type != "half":
                        print(
                            f"[[nodiscard]] inline  __device__ constexpr auto operator~(lc_{type}{i} v) noexcept {{ return lc_make_{type}{i}({', '.join(f'~v.{m}' for m in elements)}); }}",
                            file=file)
            print(file=file)


        def gen_binary_op(arg_t, ret_t, op):
            for i in range(2, 5):
                elements = ["x", "y", "z", "w"][:i]
                # vector-vector
                print(
                    f"[[nodiscard]] inline __device__ constexpr auto operator{op}(lc_{arg_t}{i} lhs, lc_{arg_t}{i} rhs) noexcept {{ return lc_make_{ret_t}{i}({', '.join(f'lhs.{m} {op} rhs.{m}' for m in elements)}); }}",
                    file=file)
                # vector-scalar
                operation = ", ".join(f"lhs.{e} {op} rhs" for e in "xyzw"[:i])
                print(
                    f"[[nodiscard]] inline __device__ constexpr auto operator{op}(lc_{arg_t}{i} lhs, lc_{arg_t} rhs) noexcept {{ return lc_make_{ret_t}{i}({operation}); }}",
                    file=file)
                # scalar-vector
                operation = ", ".join(f"lhs {op} rhs.{e}" for e in "xyzw"[:i])
                print(
                    f"[[nodiscard]] inline __device__ constexpr auto operator{op}(lc_{arg_t} lhs, lc_{arg_t}{i} rhs) noexcept {{ return lc_make_{ret_t}{i}({operation}); }}",
                    file=file)


        # binary operators
        for op in ["==", "!="]:
            for type in scalar_types:
                gen_binary_op(type, "bool", op)
            print(file=file)
        for op in ["<", ">", "<=", ">="]:
            for type in ["short", "ushort", "int", "uint", "half", "float", "long", "ulong"]:
                gen_binary_op(type, "bool", op)
            print(file=file)
        for op in ["+", "-", "*", "/"]:
            for type in ["short", "ushort", "int", "uint", "half", "float", "long", "ulong"]:
                gen_binary_op(type, type, op)
            print(file=file)
        for op in ["%", "<<", ">>"]:
            for type in ["short", "ushort", "int", "uint", "long", "ulong"]:
                gen_binary_op(type, type, op)
            print(file=file)
        for op in ["|", "&", "^"]:
            for type in ["short", "ushort", "int", "uint", "bool", "long", "ulong"]:
                gen_binary_op(type, type, op)
            print(file=file)
        for op in ["||", "&&"]:
            gen_binary_op("bool", "bool", op)
            print(file=file)

        # any, all, none
        for f, uop, bop in [("any", "", "||"), ("all", "", "&&"), ("none", "!", "&&")]:
            for i in range(2, 5):
                elements = ["x", "y", "z", "w"][:i]
                print(
                    f"[[nodiscard]] __device__ inline constexpr auto lc_{f}(lc_bool{i} v) noexcept {{ return {f' {bop} '.join(f'{uop}v.{m}' for m in elements)}; }}",
                    file=file)

        # matrix types
        for i in range(2, 5):
            def init(j):
                return ', '.join(["0.0f", "0.0f", "0.0f", "s", "0.0f", "0.0f", "0.0f"][3 - j:3 + i - j])


            print(f"""
struct lc_float{i}x{i} {{
    lc_float{i} cols[{i}];
    __device__ inline constexpr lc_float{i}x{i}() noexcept : cols{{}} {{}}
    __device__ inline explicit constexpr lc_float{i}x{i}(lc_float s) noexcept
        : cols{{{", ".join(f"lc_make_float{i}({init(j)})" for j in range(i))}}} {{}}
    __device__ inline constexpr static auto full(lc_float s) noexcept {{ return lc_float{i}x{i}{{{", ".join(f"lc_float{i}(s)" for j in range(i))}}}; }}
    __device__ inline constexpr static auto zero() noexcept {{ return lc_float{i}x{i}{{{", ".join(f"lc_float{i}::zero()" for j in range(i))}}}; }}
    __device__ inline constexpr static auto one() noexcept {{ return lc_float{i}x{i}{{{", ".join(f"lc_float{i}::one()" for j in range(i))}}}; }}
    __device__ inline constexpr lc_float{i}x{i}({", ".join(f"lc_float{i} c{j}" for j in range(i))}) noexcept
        : cols{{{", ".join(f"c{j}" for j in range(i))}}} {{}}
    [[nodiscard]] __device__ inline constexpr auto &operator[](lc_uint i) noexcept {{ return cols[i]; }}
    [[nodiscard]] __device__ inline constexpr auto operator[](lc_uint i) const noexcept {{ return cols[i]; }}
    [[nodiscard]] __device__ inline constexpr auto comp_mul(const lc_float{i}x{i} &rhs) const noexcept {{ return lc_float{i}x{i}{{{", ".join(f"cols[{j}] * rhs[{j}]" for j in range(i))}}}; }}
}};""", file=file)

        for i in range(2, 5):
            elements = ["x", "y", "z", "w"][:i]
            print(f"""
[[nodiscard]] __device__ inline constexpr auto operator*(const lc_float{i}x{i} m, lc_float s) noexcept {{ return lc_float{i}x{i}{{{", ".join(f"m[{j}] * s" for j in range(i))}}}; }}
[[nodiscard]] __device__ inline constexpr auto operator*(lc_float s, const lc_float{i}x{i} m) noexcept {{ return m * s; }}
[[nodiscard]] __device__ inline constexpr auto operator/(const lc_float{i}x{i} m, lc_float s) noexcept {{ return m * (1.0f / s); }}
[[nodiscard]] __device__ inline constexpr auto operator*(const lc_float{i}x{i} m, const lc_float{i} v) noexcept {{ return {' + '.join(f"v.{e} * m[{j}]" for j, e in enumerate(elements))}; }}
[[nodiscard]] __device__ inline constexpr auto operator*(const lc_float{i}x{i} lhs, const lc_float{i}x{i} rhs) noexcept {{ return lc_float{i}x{i}{{{', '.join(f"lhs * rhs[{j}]" for j in range(i))}}}; }}
[[nodiscard]] __device__ inline constexpr auto operator+(const lc_float{i}x{i} lhs, const lc_float{i}x{i} rhs) noexcept {{ return lc_float{i}x{i}{{{", ".join(f"lhs[{j}] + rhs[{j}]" for j in range(i))}}}; }}
[[nodiscard]] __device__ inline constexpr auto operator-(const lc_float{i}x{i} lhs, const lc_float{i}x{i} rhs) noexcept {{ return lc_float{i}x{i}{{{", ".join(f"lhs[{j}] - rhs[{j}]" for j in range(i))}}}; }}""",
                  file=file)

        for i in range(2, 5):
            def init(j):
                return ', '.join(["0.0f", "0.0f", "0.0f", "s", "0.0f", "0.0f", "0.0f"][3 - j:3 + i - j])


            print(f"""
[[nodiscard]] __device__ inline constexpr auto lc_make_float{i}x{i}(lc_float s = 1.0f) noexcept {{ return lc_float{i}x{i}{{{", ".join(f"lc_make_float{i}({init(j)})" for j in range(i))}}}; }}
[[nodiscard]] __device__ inline constexpr auto lc_make_float{i}x{i}({', '.join(', '.join(f"lc_float m{j}{k}" for k in range(i)) for j in range(i))}) noexcept {{ return lc_float{i}x{i}{{{", ".join(f"lc_make_float{i}({', '.join(f'm{j}{k}' for k in range(i))})" for j in range(i))}}}; }}
[[nodiscard]] __device__ inline constexpr auto lc_make_float{i}x{i}({", ".join(f"lc_float{i} c{j}" for j in range(i))}) noexcept {{ return lc_float{i}x{i}{{{", ".join(f"c{j}" for j in range(i))}}}; }}""",
                  file=file)
            if i == 3:
                print(
                    f"[[nodiscard]] __device__ inline constexpr auto lc_make_float{i}x{i}(lc_float2x2 m) noexcept {{ return lc_float3x3{{lc_make_float3(m[0], 0.0f), lc_make_float3(m[1], 0.0f), lc_make_float3(0.0f, 0.0f, 1.0f)}}; }}",
                    file=file)
            if i == 4:
                print(
                    f"[[nodiscard]] __device__ inline constexpr auto lc_make_float{i}x{i}(lc_float2x2 m) noexcept {{ return lc_float4x4{{lc_make_float4(m[0], 0.0f, 0.0f), lc_make_float4(m[1], 0.0f, 0.0f), lc_make_float4(0.0f, 0.0f, 0.0f, 0.0f), lc_make_float4(0.0f, 0.0f, 0.0f, 1.0f)}}; }}",
                    file=file)
                print(
                    f"[[nodiscard]] __device__ inline constexpr auto lc_make_float{i}x{i}(lc_float3x3 m) noexcept {{ return lc_float4x4{{lc_make_float4(m[0], 0.0f), lc_make_float4(m[1], 0.0f), lc_make_float4(m[2], 0.0f), lc_make_float4(0.0f, 0.0f, 0.0f, 1.0f)}}; }}",
                    file=file)
            print(
                f"[[nodiscard]] __device__ inline constexpr auto lc_make_float{i}x{i}(lc_float{i}x{i} m) noexcept {{ return m; }}",
                file=file)
            for t in range(i + 1, 5):
                print(
                    f"[[nodiscard]] __device__ inline constexpr auto lc_make_float{i}x{i}(lc_float{t}x{t} m) noexcept {{ return lc_float{i}x{i}{{{', '.join(f'lc_make_float{i}(m[{j}])' for j in range(i))}}}; }}",
                    file=file)
        print(file=file)


        def generate_vector_call(name, c, types, args):
            types = [{
                         "s": "short",
                         "r": "ushort",
                         "i": "int",
                         "u": "uint",
                         "h": "half",
                         "f": "float",
                         "b": "bool",
                         "l": "long",
                         "z": "ulong"}[t] for t in types]

            def call(i):
                e = "xyzw"[i]
                return f"{c}(" + ", ".join(f"{a}.{e}" for a in args) + ")"

            for t in types:
                ret = t if name not in ['isnan', 'isinf'] else 'bool'
                print(
                    f"[[nodiscard]] __device__ inline lc_{ret} lc_{name}({', '.join(f'lc_{t} {a}' for a in args)}) noexcept {{ return {c}({', '.join(args)}); }}",
                    file=file)
                for n in range(2, 5):
                    print(
                        f"[[nodiscard]] __device__ inline lc_{ret}{n} lc_{name}({', '.join(f'lc_{t}{n} {a}' for a in args)}) noexcept {{ return lc_make_{ret}{n}({', '.join(call(i) for i in range(n))}); }}",
                        file=file)
            print(file=file)


        # select
        print(
            "template<typename T>\n[[nodiscard]] __device__ inline auto lc_select(T f, T t, bool p) noexcept { return p ? t : f; }",
            file=file)
        for t in ["short", "ushort", "int", "uint", "half", "float", "bool", "long", "ulong"]:
            for n in range(2, 5):
                print(
                    f"[[nodiscard]] __device__ inline auto lc_select(lc_{t}{n} f, lc_{t}{n} t, lc_bool{n} p) noexcept {{ return lc_make_{t}{n}({', '.join(f'lc_select<lc_{t}>(f.{e}, t.{e}, p.{e})' for e in 'xyzw'[:n])}); }}",
                    file=file)
        print(file=file)

        # outer product
        for t in ["float"]:
            for n in range(2, 5):
                print(
                    f"[[nodiscard]] __device__ inline auto lc_outer_product(lc_{t}{n} a, lc_{t}{n} b) noexcept {{ return lc_{t}{n}x{n}({', '.join(f'a * b.{f}' for f in 'xyzw'[:n])}); }}",
                    file=file)
        # min/max/abs/acos/asin/asinh/acosh/atan/atanh/atan2/
        # cos/cosh/sin/sinh/tan/tanh/exp/exp2/exp10/log/log2/
        # log10/sqrt/rsqrt/ceil/floor/trunc/round/fma/copysignf/
        # isinf/isnan
        generate_vector_call("min", "fminf", "f", ["a", "b"])
        generate_vector_call("min", "__hmin", "h", ["a", "b"])
        generate_vector_call("max", "fmaxf", "f", ["a", "b"])
        generate_vector_call("max", "__hmax", "h", ["a", "b"])
        generate_vector_call("abs", "fabsf", "f", ["x"])
        generate_vector_call("abs", "__habs", "h", ["x"])

        generate_vector_call("acos", "acosf", "hf", ["x"])
        generate_vector_call("asin", "asinf", "hf", ["x"])
        generate_vector_call("atan", "atanf", "hf", ["x"])
        generate_vector_call("acosh", "acoshf", "hf", ["x"])
        generate_vector_call("asinh", "asinhf", "hf", ["x"])
        generate_vector_call("atanh", "atanhf", "hf", ["x"])
        generate_vector_call("atan2", "atan2f", "hf", ["y", "x"])

        generate_vector_call("cosh", "coshf", "hf", ["x"])
        generate_vector_call("sinh", "sinhf", "hf", ["x"])
        generate_vector_call("tanh", "tanhf", "hf", ["x"])

        generate_vector_call("cos", "cosf", "hf", ["x"])
        generate_vector_call("sin", "sinf", "hf", ["x"])
        generate_vector_call("tan", "tanf", "hf", ["x"])
        generate_vector_call("exp", "expf", "hf", ["x"])
        generate_vector_call("exp2", "exp2f", "f", ["x"])
        generate_vector_call("exp2", "hexp2", "h", ["x"])
        generate_vector_call("exp10", "exp10f", "hf", ["x"])
        generate_vector_call("log", "logf", "hf", ["x"])
        generate_vector_call("log2", "log2f", "hf", ["x"])
        generate_vector_call("log10", "log10f", "hf", ["x"])
        generate_vector_call("pow", "powf_impl", "hf", ["x", "a"])
        generate_vector_call("powi", "powi_impl", "hf", ["x", "a"])

        generate_vector_call("sqrt", "sqrtf", "f", ["x"])
        generate_vector_call("sqrt", "hsqrt", "h", ["x"])

        generate_vector_call("rsqrt", "rsqrtf", "f", ["x"])
        generate_vector_call("rsqrt", "hrsqrt", "h", ["x"])

        generate_vector_call("ceil", "ceilf", "f", ["x"])
        generate_vector_call("ceil", "hceil", "h", ["x"])

        generate_vector_call("floor", "floorf", "f", ["x"])
        generate_vector_call("floor", "hfloor", "h", ["x"])

        generate_vector_call("trunc", "truncf", "f", ["x"])
        generate_vector_call("trunc", "htrunc", "h", ["x"])

        generate_vector_call("round", "roundf", "hf", ["x"])

        generate_vector_call("fma", "fmaf", "f", ["x", "y", "z"])
        generate_vector_call("fma", "__hfma", "h", ["x", "y", "z"])

        print("""
[[nodiscard]] __device__ inline auto lc_copysign_impl(lc_half x, lc_half y) noexcept {
    auto ux = __half_as_short(x);
    auto uy = __half_as_short(y);
    return __short_as_half((ux & 0x7fffu) | (uy & 0x8000u));
}""", file=file)

        generate_vector_call("copysign", "copysignf", "f", ["x", "y"])
        generate_vector_call("copysign", "lc_copysign_impl", "h", ["x", "y"])

        generate_vector_call("isinf", "isinf_impl", "f", ["x"])
        generate_vector_call("isnan", "isnan_impl", "f", ["x"])
        generate_vector_call("isinf", "__hisinf", "h", ["x"])
        generate_vector_call("isnan", "__hisnan", "h", ["x"])
        # TODO: half

        # reduce operations
        for t in ["short", "ushort", "int", "uint", "half", "float", "long", "ulong"]:
            for n in range(2, 5):
                print(
                    f"[[nodiscard]] __device__ inline auto lc_reduce_sum(lc_{t}{n} v) noexcept {{ return lc_{t}({'+'.join(f'v.{e}' for e in 'xyzw'[:n])}); }}",
                    file=file)
                print(
                    f"[[nodiscard]] __device__ inline auto lc_reduce_prod(lc_{t}{n} v) noexcept {{ return lc_{t}({'*'.join(f'v.{e}' for e in 'xyzw'[:n])}); }}",
                    file=file)
                print(
                    f"[[nodiscard]] __device__ inline auto lc_reduce_min(lc_{t}{n} v) noexcept {{ return lc_{t}({', '.join(f'lc_min(v.{e}' for e in 'xyzw'[:n - 1])}, v.{'xyzw'[n - 1]}{')' * (n)}; }}",
                    file=file)
                print(
                    f"[[nodiscard]] __device__ inline auto lc_reduce_max(lc_{t}{n} v) noexcept {{ return lc_{t}({', '.join(f'lc_max(v.{e}' for e in 'xyzw'[:n - 1])}, v.{'xyzw'[n - 1]}{')' * (n)}; }}",
                    file=file)

        # min/max for int
        for t in ["short", "ushort", "int", "uint", "long", "ulong"]:
            # lc_min_impl/lc_max_impl
            print(
                f"[[nodiscard]] __device__ inline auto lc_min_impl(lc_{t} a, lc_{t} b) noexcept {{ return a < b ? a : b; }}",
                file=file)
            print(
                f"[[nodiscard]] __device__ inline auto lc_max_impl(lc_{t} a, lc_{t} b) noexcept {{ return a > b ? a : b; }}",
                file=file)
        generate_vector_call("min", "lc_min_impl", "sriulz", ["a", "b"])
        generate_vector_call("max", "lc_max_impl", "sriulz", ["a", "b"])

        # clamp
        for t in ["short", "ushort", "int", "uint", "half", "float", "long", "ulong"]:
            print(
                f"[[nodiscard]] __device__ inline auto lc_clamp_impl(lc_{t} v, lc_{t} lo, lc_{t} hi) noexcept {{ return lc_min(lc_max(v, lo), hi); }}",
                file=file)
        generate_vector_call("clamp", "lc_clamp_impl",
                             "sriulzhf", ["v", "lo", "hi"])

        for t in ["half", "float"]:
            # lerp
            print(
                f"[[nodiscard]] __device__ inline auto lc_lerp_impl(lc_{t} a, lc_{t} b, lc_{t} t) noexcept {{ return t * (b - a) + a; }}",
                file=file)

            # saturate
            print(
                f"[[nodiscard]] __device__ inline auto lc_saturate(lc_{t} x) noexcept {{ return lc_clamp(x, lc_{t}(0.0f), lc_{t}(1.0f)); }}",
                file=file)
            for n in range(2, 5):
                print(
                    f"[[nodiscard]] __device__ inline auto lc_saturate(lc_{t}{n} x) noexcept {{ return lc_clamp(x, lc_make_{t}{n}(0.0f), lc_make_{t}{n}(1.0f)); }}",
                    file=file)
            print(file=file)

            # degrees/radians
            print(
                f"[[nodiscard]] __device__ inline auto lc_degrees_impl(lc_{t} rad) noexcept {{ return rad * (lc_{t})(180.0f * 0.318309886183790671537767526745028724f); }}",
                file=file)
            print(
                f"[[nodiscard]] __device__ inline auto lc_radians_impl(lc_{t} deg) noexcept {{ return deg * (lc_{t})(3.14159265358979323846264338327950288f / 180.0f); }}",
                file=file)

            # step
            print(
                f"[[nodiscard]] __device__ inline auto lc_step_impl(lc_{t} edge, lc_{t} x) noexcept {{ return lc_select(lc_{t}(1.f), lc_{t}(0.f), x < edge); }}",
                file=file)

            # smoothstep
            print(
                f"""[[nodiscard]] __device__ inline auto lc_smoothstep_impl(lc_{t} edge0, lc_{t} edge1, lc_{t} x) noexcept {{
    auto t = lc_clamp((x - edge0) / (edge1 - edge0), lc_{t}(0.0f), lc_{t}(1.0f));
    return t * t * (lc_{t}(3.f) - lc_{t}(2.f) * t);
}}""",
                file=file)

            # mod
            print(
                f"[[nodiscard]] __device__ inline auto lc_mod_impl(lc_{t} x, lc_{t} y) noexcept {{ return x - y * lc_floor(x / y); }}",
                file=file)

            # fmod
            if t == "half":
                print(
                    f"[[nodiscard]] __device__ inline auto lc_fmod_impl(lc_{t} x, lc_{t} y) noexcept {{ return x - y * lc_trunc(x / y); }}",
                    file=file)
            else:
                print(
                    f"[[nodiscard]] __device__ inline auto lc_fmod_impl(lc_{t} x, lc_{t} y) noexcept {{ return fmodf(x, y); }}",
                    file=file)

            # fract
            print(
                f"[[nodiscard]] __device__ inline auto lc_fract_impl(lc_{t} x) noexcept {{ return x - lc_floor(x); }}",
                file=file)

        generate_vector_call("lerp", "lc_lerp_impl", "hf", ["a", "b", "t"])
        generate_vector_call("degrees", "lc_degrees_impl", "hf", ["rad"])
        generate_vector_call("radians", "lc_radians_impl", "hf", ["deg"])
        generate_vector_call("step", "lc_step_impl", "hf", ["edge", "x"])
        generate_vector_call("smoothstep", "lc_smoothstep_impl", "hf", ["e0", "e1", "x"])
        generate_vector_call("mod", "lc_mod_impl", "hf", ["x", "y"])
        generate_vector_call("fmod", "lc_fmod_impl", "hf", ["x", "y"])
        generate_vector_call("fract", "lc_fract_impl", "hf", ["x"])

        # clz/popcount/reverse
        generate_vector_call("clz", "__clz", "uz", ["x"])
        generate_vector_call("popcount", "__popc", "uz", ["x"])
        generate_vector_call("reverse", "__brev", "uz", ["x"])

        # ctz
        print(
            f"[[nodiscard]] __device__ inline auto lc_ctz_impl(lc_uint x) noexcept {{ return (__ffs(x) - 1u) % 32u; }}",
            file=file)
        generate_vector_call("ctz", "lc_ctz_impl", "u", ["x"])

        for t in ["float", "half"]:
            # cross
            print(f"""[[nodiscard]] __device__ inline constexpr auto lc_cross(lc_{t}3 u, lc_{t}3 v) noexcept {{
    return lc_make_{t}3(u.y * v.z - v.y * u.z,
                          u.z * v.x - v.z * u.x,
                          u.x * v.y - v.x * u.y);
}}""", file=file)
            print(file=file)

            # dot
            print(f"""[[nodiscard]] __device__ inline auto lc_dot(lc_{t}2 a, lc_{t}2 b) noexcept {{
    return a.x * b.x + a.y * b.y;
}}""", file=file)
            print(f"""[[nodiscard]] __device__ inline auto lc_dot(lc_{t}3 a, lc_{t}3 b) noexcept {{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}}""", file=file)
            print(f"""[[nodiscard]] __device__ inline auto lc_dot(lc_{t}4 a, lc_{t}4 b) noexcept {{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}}""", file=file)
            print(file=file)

            # length
            print(
                f"[[nodiscard]] __device__ inline auto lc_length(lc_{t}2 v) noexcept {{ return lc_sqrt(lc_dot(v, v)); }}",
                file=file)
            print(
                f"[[nodiscard]] __device__ inline auto lc_length(lc_{t}3 v) noexcept {{ return lc_sqrt(lc_dot(v, v)); }}",
                file=file)
            print(
                f"[[nodiscard]] __device__ inline auto lc_length(lc_{t}4 v) noexcept {{ return lc_sqrt(lc_dot(v, v)); }}",
                file=file)
            print(file=file)

            # length_squared
            for n in range(2, 5):
                print(
                    f"[[nodiscard]] __device__ inline auto lc_length_squared(lc_{t}{n} v) noexcept {{ return lc_dot(v, v); }}",
                    file=file)
            print(file=file)

            # distance
            for n in range(2, 5):
                print(
                    f"[[nodiscard]] __device__ inline auto lc_distance(lc_{t}{n} a, lc_{t}{n} b) noexcept {{ return lc_length(a - b); }}",
                    file=file)
            print(file=file)

            # distance_squared
            for n in range(2, 5):
                print(
                    f"[[nodiscard]] __device__ inline auto lc_distance_squared(lc_{t}{n} a, lc_{t}{n} b) noexcept {{ return lc_length_squared(a - b); }}",
                    file=file)
            print(file=file)

            # faceforward
            print(
                f"[[nodiscard]] __device__ inline auto lc_faceforward(lc_{t}3 n, lc_{t}3 i, lc_{t}3 n_ref) noexcept {{ return lc_select(-n, n, lc_dot(n_ref, i) < lc_{t}(0.f)); }}",
                file=file)
            print(file=file)

            # normalize
            for n in range(2, 5):
                print(
                    f"[[nodiscard]] __device__ inline auto lc_normalize(lc_{t}{n} v) noexcept {{ return v * lc_rsqrt(lc_dot(v, v)); }}",
                    file=file)
            print(file=file)

        # transpose
        print("""[[nodiscard]] __device__ inline constexpr auto lc_transpose(const lc_float2x2 m) noexcept { return lc_make_float2x2(m[0].x, m[1].x, m[0].y, m[1].y); }
[[nodiscard]] __device__ inline constexpr auto lc_transpose(const lc_float3x3 m) noexcept { return lc_make_float3x3(m[0].x, m[1].x, m[2].x, m[0].y, m[1].y, m[2].y, m[0].z, m[1].z, m[2].z); }
[[nodiscard]] __device__ inline constexpr auto lc_transpose(const lc_float4x4 m) noexcept { return lc_make_float4x4(m[0].x, m[1].x, m[2].x, m[3].x, m[0].y, m[1].y, m[2].y, m[3].y, m[0].z, m[1].z, m[2].z, m[3].z, m[0].w, m[1].w, m[2].w, m[3].w); }
""", file=file)

        # determinant/inverse
        print("""[[nodiscard]] __device__ inline constexpr auto lc_determinant(const lc_float2x2 m) noexcept {
    return m[0][0] * m[1][1] - m[1][0] * m[0][1];
}

[[nodiscard]] __device__ constexpr auto lc_determinant(const lc_float3x3 m) noexcept {// from GLM
    return m[0].x * (m[1].y * m[2].z - m[2].y * m[1].z)
         - m[1].x * (m[0].y * m[2].z - m[2].y * m[0].z)
         + m[2].x * (m[0].y * m[1].z - m[1].y * m[0].z);
}

[[nodiscard]] __device__ inline constexpr auto lc_determinant(const lc_float4x4 m) noexcept {// from GLM
    const auto coef00 = m[2].z * m[3].w - m[3].z * m[2].w;
    const auto coef02 = m[1].z * m[3].w - m[3].z * m[1].w;
    const auto coef03 = m[1].z * m[2].w - m[2].z * m[1].w;
    const auto coef04 = m[2].y * m[3].w - m[3].y * m[2].w;
    const auto coef06 = m[1].y * m[3].w - m[3].y * m[1].w;
    const auto coef07 = m[1].y * m[2].w - m[2].y * m[1].w;
    const auto coef08 = m[2].y * m[3].z - m[3].y * m[2].z;
    const auto coef10 = m[1].y * m[3].z - m[3].y * m[1].z;
    const auto coef11 = m[1].y * m[2].z - m[2].y * m[1].z;
    const auto coef12 = m[2].x * m[3].w - m[3].x * m[2].w;
    const auto coef14 = m[1].x * m[3].w - m[3].x * m[1].w;
    const auto coef15 = m[1].x * m[2].w - m[2].x * m[1].w;
    const auto coef16 = m[2].x * m[3].z - m[3].x * m[2].z;
    const auto coef18 = m[1].x * m[3].z - m[3].x * m[1].z;
    const auto coef19 = m[1].x * m[2].z - m[2].x * m[1].z;
    const auto coef20 = m[2].x * m[3].y - m[3].x * m[2].y;
    const auto coef22 = m[1].x * m[3].y - m[3].x * m[1].y;
    const auto coef23 = m[1].x * m[2].y - m[2].x * m[1].y;
    const auto fac0 = lc_make_float4(coef00, coef00, coef02, coef03);
    const auto fac1 = lc_make_float4(coef04, coef04, coef06, coef07);
    const auto fac2 = lc_make_float4(coef08, coef08, coef10, coef11);
    const auto fac3 = lc_make_float4(coef12, coef12, coef14, coef15);
    const auto fac4 = lc_make_float4(coef16, coef16, coef18, coef19);
    const auto fac5 = lc_make_float4(coef20, coef20, coef22, coef23);
    const auto Vec0 = lc_make_float4(m[1].x, m[0].x, m[0].x, m[0].x);
    const auto Vec1 = lc_make_float4(m[1].y, m[0].y, m[0].y, m[0].y);
    const auto Vec2 = lc_make_float4(m[1].z, m[0].z, m[0].z, m[0].z);
    const auto Vec3 = lc_make_float4(m[1].w, m[0].w, m[0].w, m[0].w);
    const auto inv0 = Vec1 * fac0 - Vec2 * fac1 + Vec3 * fac2;
    const auto inv1 = Vec0 * fac0 - Vec2 * fac3 + Vec3 * fac4;
    const auto inv2 = Vec0 * fac1 - Vec1 * fac3 + Vec3 * fac5;
    const auto inv3 = Vec0 * fac2 - Vec1 * fac4 + Vec2 * fac5;
    constexpr auto sign_a = lc_make_float4(+1.0f, -1.0f, +1.0f, -1.0f);
    constexpr auto sign_b = lc_make_float4(-1.0f, +1.0f, -1.0f, +1.0f);
    const auto inv_0 = inv0 * sign_a;
    const auto inv_1 = inv1 * sign_b;
    const auto inv_2 = inv2 * sign_a;
    const auto inv_3 = inv3 * sign_b;
    const auto dot0 = m[0] * lc_make_float4(inv_0.x, inv_1.x, inv_2.x, inv_3.x);
    return dot0.x + dot0.y + dot0.z + dot0.w;
}

[[nodiscard]] __device__ inline constexpr auto lc_inverse(const lc_float2x2 m) noexcept {
    const auto one_over_determinant = 1.0f / (m[0][0] * m[1][1] - m[1][0] * m[0][1]);
    return lc_make_float2x2(m[1][1] * one_over_determinant,
                          - m[0][1] * one_over_determinant,
                          - m[1][0] * one_over_determinant,
                          + m[0][0] * one_over_determinant);
}

[[nodiscard]] __device__ inline constexpr auto lc_inverse(const lc_float3x3 m) noexcept {// from GLM
    const auto one_over_determinant = 1.0f
                                      / (m[0].x * (m[1].y * m[2].z - m[2].y * m[1].z)
                                       - m[1].x * (m[0].y * m[2].z - m[2].y * m[0].z)
                                       + m[2].x * (m[0].y * m[1].z - m[1].y * m[0].z));
    return lc_make_float3x3(
        (m[1].y * m[2].z - m[2].y * m[1].z) * one_over_determinant,
        (m[2].y * m[0].z - m[0].y * m[2].z) * one_over_determinant,
        (m[0].y * m[1].z - m[1].y * m[0].z) * one_over_determinant,
        (m[2].x * m[1].z - m[1].x * m[2].z) * one_over_determinant,
        (m[0].x * m[2].z - m[2].x * m[0].z) * one_over_determinant,
        (m[1].x * m[0].z - m[0].x * m[1].z) * one_over_determinant,
        (m[1].x * m[2].y - m[2].x * m[1].y) * one_over_determinant,
        (m[2].x * m[0].y - m[0].x * m[2].y) * one_over_determinant,
        (m[0].x * m[1].y - m[1].x * m[0].y) * one_over_determinant);
}

[[nodiscard]] __device__ inline constexpr auto lc_inverse(const lc_float4x4 m) noexcept {// from GLM
    const auto coef00 = m[2].z * m[3].w - m[3].z * m[2].w;
    const auto coef02 = m[1].z * m[3].w - m[3].z * m[1].w;
    const auto coef03 = m[1].z * m[2].w - m[2].z * m[1].w;
    const auto coef04 = m[2].y * m[3].w - m[3].y * m[2].w;
    const auto coef06 = m[1].y * m[3].w - m[3].y * m[1].w;
    const auto coef07 = m[1].y * m[2].w - m[2].y * m[1].w;
    const auto coef08 = m[2].y * m[3].z - m[3].y * m[2].z;
    const auto coef10 = m[1].y * m[3].z - m[3].y * m[1].z;
    const auto coef11 = m[1].y * m[2].z - m[2].y * m[1].z;
    const auto coef12 = m[2].x * m[3].w - m[3].x * m[2].w;
    const auto coef14 = m[1].x * m[3].w - m[3].x * m[1].w;
    const auto coef15 = m[1].x * m[2].w - m[2].x * m[1].w;
    const auto coef16 = m[2].x * m[3].z - m[3].x * m[2].z;
    const auto coef18 = m[1].x * m[3].z - m[3].x * m[1].z;
    const auto coef19 = m[1].x * m[2].z - m[2].x * m[1].z;
    const auto coef20 = m[2].x * m[3].y - m[3].x * m[2].y;
    const auto coef22 = m[1].x * m[3].y - m[3].x * m[1].y;
    const auto coef23 = m[1].x * m[2].y - m[2].x * m[1].y;
    const auto fac0 = lc_make_float4(coef00, coef00, coef02, coef03);
    const auto fac1 = lc_make_float4(coef04, coef04, coef06, coef07);
    const auto fac2 = lc_make_float4(coef08, coef08, coef10, coef11);
    const auto fac3 = lc_make_float4(coef12, coef12, coef14, coef15);
    const auto fac4 = lc_make_float4(coef16, coef16, coef18, coef19);
    const auto fac5 = lc_make_float4(coef20, coef20, coef22, coef23);
    const auto Vec0 = lc_make_float4(m[1].x, m[0].x, m[0].x, m[0].x);
    const auto Vec1 = lc_make_float4(m[1].y, m[0].y, m[0].y, m[0].y);
    const auto Vec2 = lc_make_float4(m[1].z, m[0].z, m[0].z, m[0].z);
    const auto Vec3 = lc_make_float4(m[1].w, m[0].w, m[0].w, m[0].w);
    const auto inv0 = Vec1 * fac0 - Vec2 * fac1 + Vec3 * fac2;
    const auto inv1 = Vec0 * fac0 - Vec2 * fac3 + Vec3 * fac4;
    const auto inv2 = Vec0 * fac1 - Vec1 * fac3 + Vec3 * fac5;
    const auto inv3 = Vec0 * fac2 - Vec1 * fac4 + Vec2 * fac5;
    constexpr auto sign_a = lc_make_float4(+1.0f, -1.0f, +1.0f, -1.0f);
    constexpr auto sign_b = lc_make_float4(-1.0f, +1.0f, -1.0f, +1.0f);
    const auto inv_0 = inv0 * sign_a;
    const auto inv_1 = inv1 * sign_b;
    const auto inv_2 = inv2 * sign_a;
    const auto inv_3 = inv3 * sign_b;
    const auto dot0 = m[0] * lc_make_float4(inv_0.x, inv_1.x, inv_2.x, inv_3.x);
    const auto dot1 = dot0.x + dot0.y + dot0.z + dot0.w;
    const auto one_over_determinant = 1.0f / dot1;
    return lc_make_float4x4(inv_0 * one_over_determinant,
                            inv_1 * one_over_determinant,
                            inv_2 * one_over_determinant,
                            inv_3 * one_over_determinant);
}

[[nodiscard]] __device__ inline auto lc_reflect(const lc_float3 v, const lc_float3 n) noexcept {
    return v - 2.0f * lc_dot(v, n) * n;
}

template<typename D, typename S>
[[nodiscard]] __device__ inline auto lc_bit_cast(S s) noexcept {
    static_assert(sizeof(D) == sizeof(S));
    return reinterpret_cast<const D &>(s);
}
template<class T>
[[nodiscard]] __device__ inline constexpr auto lc_zero() noexcept {
    return T{};
}
template<class T>
[[nodiscard]] __device__ inline constexpr auto lc_one() noexcept {
    return T::one();
}
template<>
[[nodiscard]] __device__ inline constexpr auto lc_one<lc_int>() noexcept {
    return lc_int(1);
}
template<>
[[nodiscard]] __device__ inline constexpr auto lc_one<lc_float>() noexcept {
    return lc_float(1.0f);
}
template<>
[[nodiscard]] __device__ inline auto lc_one<lc_half>() noexcept {
    return lc_half(1.0f);
}
template<>
[[nodiscard]] __device__ inline constexpr auto lc_one<lc_uint>() noexcept {
    return lc_uint(1u);
}
template<>
[[nodiscard]] __device__ inline constexpr auto lc_one<lc_long>() noexcept {
    return lc_long(1);
}
template<>
[[nodiscard]] __device__ inline constexpr auto lc_one<lc_ulong>() noexcept {
    return lc_ulong(1);
}
template<>
[[nodiscard]] __device__ inline constexpr auto lc_one<lc_short>() noexcept {
    return lc_short(1);
}
template<>
[[nodiscard]] __device__ inline constexpr auto lc_one<lc_ushort>() noexcept {
    return lc_ushort(1);
}
template<>
[[nodiscard]] __device__ inline constexpr auto lc_one<lc_bool>() noexcept {
    return true;
}
template<typename T, size_t N>
class lc_array {

private:
    T _data[N];

public:
    template<typename... Elem>
    __device__ constexpr lc_array(Elem... elem) noexcept : _data{elem...} {}
    __device__ constexpr lc_array(lc_array &&) noexcept = default;
    __device__ constexpr lc_array(const lc_array &) noexcept = default;
    __device__ constexpr lc_array &operator=(lc_array &&) noexcept = default;
    __device__ constexpr lc_array &operator=(const lc_array &) noexcept = default;
    [[nodiscard]] __device__ T &operator[](size_t i) noexcept { return _data[i]; }
    [[nodiscard]] __device__ T operator[](size_t i) const noexcept { return _data[i]; }

public:
    [[nodiscard]] __device__ static auto one() noexcept {
        lc_array<T, N> ret;
        #pragma unroll
        for (auto i = 0u; i < N; i++) { ret[i] = lc_one<T>(); }
        return ret;
    }
};

[[nodiscard]] __device__ inline auto lc_mat_comp_mul(lc_float2x2 lhs, lc_float2x2 rhs) noexcept {
    return lc_make_float2x2(lhs[0] * rhs[0],
                            lhs[1] * rhs[1]);
}

[[nodiscard]] __device__ inline auto lc_mat_comp_mul(lc_float3x3 lhs, lc_float3x3 rhs) noexcept {
    return lc_make_float3x3(lhs[0] * rhs[0],
                            lhs[1] * rhs[1],
                            lhs[2] * rhs[2]);
}

[[nodiscard]] __device__ inline auto lc_mat_comp_mul(lc_float4x4 lhs, lc_float4x4 rhs) noexcept {
    return lc_make_float4x4(lhs[0] * rhs[0],
                            lhs[1] * rhs[1],
                            lhs[2] * rhs[2],
                            lhs[3] * rhs[3]);
}
[[nodiscard]] __device__ inline constexpr auto lc_remove_nan(lc_float v) noexcept {
    return lc_select(v, lc_zero<lc_float>(), lc_isnan(v) | lc_isinf(v));
}
[[nodiscard]] __device__ inline constexpr auto lc_remove_nan(lc_float2 v) noexcept {
    return lc_select(v, lc_zero<lc_float2>(), lc_isnan(v) | lc_isinf(v));
}
[[nodiscard]] __device__ inline constexpr auto lc_remove_nan(lc_float3 v) noexcept {
    return lc_select(v, lc_zero<lc_float3>(), lc_isnan(v) | lc_isinf(v));
}
[[nodiscard]] __device__ inline constexpr auto lc_remove_nan(lc_float4 v) noexcept {
    return lc_select(v, lc_zero<lc_float4>(), lc_isnan(v) | lc_isinf(v));
}
[[nodiscard]] __device__ inline constexpr auto lc_remove_nan(lc_half v) noexcept {
    return lc_select(v, lc_zero<lc_half>(), lc_isnan(v) | lc_isinf(v));
}
[[nodiscard]] __device__ inline constexpr auto lc_remove_nan(lc_half2 v) noexcept {
    return lc_select(v, lc_zero<lc_half2>(), lc_isnan(v) | lc_isinf(v));
}
[[nodiscard]] __device__ inline constexpr auto lc_remove_nan(lc_half3 v) noexcept {
    return lc_select(v, lc_zero<lc_half3>(), lc_isnan(v) | lc_isinf(v));
}
[[nodiscard]] __device__ inline constexpr auto lc_remove_nan(lc_half4 v) noexcept {
    return lc_select(v, lc_zero<lc_half4>(), lc_isnan(v) | lc_isinf(v));
}
[[nodiscard]] __device__ inline constexpr auto lc_remove_nan(lc_float2x2 v) noexcept {
    v.cols[0] = lc_remove_nan(v.cols[0]);
    v.cols[1] = lc_remove_nan(v.cols[1]);
    return v;
}
[[nodiscard]] __device__ inline constexpr auto lc_remove_nan(lc_float3x3 v) noexcept {
    v.cols[0] = lc_remove_nan(v.cols[0]);
    v.cols[1] = lc_remove_nan(v.cols[1]);
    v.cols[2] = lc_remove_nan(v.cols[2]);
    return v;
}
[[nodiscard]] __device__ inline constexpr auto lc_remove_nan(lc_float4x4 v) noexcept {
    v.cols[0] = lc_remove_nan(v.cols[0]);
    v.cols[1] = lc_remove_nan(v.cols[1]);
    v.cols[2] = lc_remove_nan(v.cols[2]);
    v.cols[3] = lc_remove_nan(v.cols[3]);
    return v;
}
""", file=file)
        # accumlate_grad(T*, const T) for all types
        float_types = [
            "lc_float",
            "lc_float2x2", "lc_float3x3", "lc_float4x4",
            "lc_float2", "lc_float3", "lc_float4",
            "lc_half", "lc_half2", "lc_half3", "lc_half4",
        ]
        for t in float_types:
            print(
                f"__device__ inline void lc_accumulate_grad({t} *dst, {t} grad) noexcept {{ *dst = *dst + lc_remove_nan(grad); }}",
                file=file)
        non_differentiable_types = [
            'lc_byte2', 'lc_byte3', 'lc_byte4',
            'lc_ubyte2', 'lc_ubyte3', 'lc_ubyte4',
            "lc_short", "lc_ushort", "lc_int", "lc_uint", "lc_long", "lc_ulong", "lc_bool",
            "lc_short2", "lc_short3", "lc_short4",
            "lc_ushort2", "lc_ushort3", "lc_ushort4",
            "lc_int2", "lc_int3", "lc_int4",
            "lc_uint2", "lc_uint3", "lc_uint4",
            "lc_long2", "lc_long3", "lc_long4",
            "lc_ulong2", "lc_ulong3", "lc_ulong4",
            "lc_bool2", "lc_bool3", "lc_bool4",
        ]
        for t in non_differentiable_types:
            print(
                f"__device__ inline void lc_accumulate_grad({t} *dst, {t} grad) noexcept {{}}", file=file)
        print(
            "struct lc_user_data_t{}; constexpr lc_user_data_t _lc_user_data{};", file=file)
        print('''template<class T> struct element_type_{using type = void;};
template<class T> using element_type = typename element_type_<T>::type;
''', file=file)


        def gen_element_type(vt, et):
            print(f'''template<> struct element_type_<{vt}> {{ using type = {et}; }};''', file=file)


        for vt in ['lc_float2', 'lc_float3', 'lc_float4']:
            gen_element_type(vt, "lc_float")
        for vt in ['lc_half2', 'lc_half3', 'lc_half4']:
            gen_element_type(vt, "lc_half")
        for vt in ['lc_short2', 'lc_short3', 'lc_short4']:
            gen_element_type(vt, 'lc_short')
        for vt in ['lc_ushort2', 'lc_ushort3', 'lc_ushort4']:
            gen_element_type(vt, 'lc_ushort')
        for vt in ['lc_byte2', 'lc_byte3', 'lc_byte4']:
            gen_element_type(vt, 'lc_byte')
        for vt in ['lc_ubyte2', 'lc_ubyte3', 'lc_ubyte4']:
            gen_element_type(vt, 'lc_ubyte')
        for vt in ['lc_int2', 'lc_int3', 'lc_int4']:
            gen_element_type(vt, 'lc_int')
        for vt in ['lc_uint2', 'lc_uint3', 'lc_uint4']:
            gen_element_type(vt, 'lc_uint')
        for vt in ['lc_long2', 'lc_long3', 'lc_long4']:
            gen_element_type(vt, 'lc_long')
        for vt in ['lc_ulong2', 'lc_ulong3', 'lc_ulong4']:
            gen_element_type(vt, 'lc_ulong')

        print('''
template<typename T, size_t N>
__device__ inline void lc_accumulate_grad(lc_array<T, N> *dst, lc_array<T, N> grad) noexcept {
    #pragma unroll
    for (auto i = 0u; i < N; i++) { lc_accumulate_grad(&(*dst)[i], grad[i]); }
}''', file=file)
