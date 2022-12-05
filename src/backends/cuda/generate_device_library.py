from os.path import realpath, dirname

if __name__ == "__main__":
    curr_dir = dirname(realpath(__file__))
    math_library_name = "cuda_device_math"
    surf_library_name = "cuda_device_resource"
    with open(f"{curr_dir}/{math_library_name}.h", "w") as file:
        # scalar types
        print("#pragma once\n", file=file)
        scalar_types = ["int", "uint", "float", "bool"]
        native_types = ["int", "unsigned int", "float", "bool"]
        for t, native_t in zip(scalar_types, native_types):
            print(f"using lc_{t} = {native_t};", file=file)
        print(file=file)

        # vector types
        vector_alignments = {2: 8, 3: 16, 4: 16}
        for type in scalar_types:
            for i in range(2, 5):
                elements = ["x", "y", "z", "w"][:i]
                print(
                    f"""struct alignas({vector_alignments[i] if type != 'bool' else vector_alignments[i] // 4}) lc_{type}{i} {{
    lc_{type} {', '.join(elements[:i + 1])};
    __device__ constexpr lc_{type}{i}() noexcept
        : {', '.join(f"{m}{{}}" for m in elements)} {{}}
    __device__ explicit constexpr lc_{type}{i}(lc_{type} s) noexcept
        : {', '.join(f"{m}{{s}}" for m in elements)} {{}}
    __device__ constexpr lc_{type}{i}({', '.join(f"lc_{type} {m}" for m in elements)}) noexcept
        : {', '.join(f"{m}{{{m}}}" for m in elements)} {{}}
    __device__ constexpr auto &operator[](lc_uint i) noexcept {{ return (&x)[i]; }}
    __device__ constexpr auto operator[](lc_uint i) const noexcept {{ return (&x)[i]; }}
}};
""", file=file)

        # make type[n]
        for type in scalar_types:
            # make type2
            print(f"""[[nodiscard]] __device__ constexpr auto lc_make_{type}2(lc_{type} s = 0) noexcept {{ return lc_{type}2{{s, s}}; }}
[[nodiscard]] __device__ constexpr auto lc_make_{type}2(lc_{type} x, lc_{type} y) noexcept {{ return lc_{type}2{{x, y}}; }}""",
                  file=file)
            for t in scalar_types:
                for l in range(2, 5):
                    print(
                        f"[[nodiscard]] __device__ constexpr auto lc_make_{type}2(lc_{t}{l} v) noexcept {{ return lc_{type}2{{static_cast<lc_{type}>(v.x), static_cast<lc_{type}>(v.y)}}; }}",
                        file=file)
            # make type3
            print(f"""[[nodiscard]] __device__ constexpr auto lc_make_{type}3(lc_{type} s = 0) noexcept {{ return lc_{type}3{{s, s, s}}; }}
[[nodiscard]] __device__ constexpr auto lc_make_{type}3(lc_{type} x, lc_{type} y, lc_{type} z) noexcept {{ return lc_{type}3{{x, y, z}}; }}
[[nodiscard]] __device__ constexpr auto lc_make_{type}3(lc_{type} x, lc_{type}2 yz) noexcept {{ return lc_{type}3{{x, yz.x, yz.y}}; }}
[[nodiscard]] __device__ constexpr auto lc_make_{type}3(lc_{type}2 xy, lc_{type} z) noexcept {{ return lc_{type}3{{xy.x, xy.y, z}}; }}""",
                  file=file)
            for t in scalar_types:
                for l in range(3, 5):
                    print(
                        f"[[nodiscard]] __device__ constexpr auto lc_make_{type}3(lc_{t}{l} v) noexcept {{ return lc_{type}3{{static_cast<lc_{type}>(v.x), static_cast<lc_{type}>(v.y), static_cast<lc_{type}>(v.z)}}; }}",
                        file=file)
            # make type4
            print(f"""[[nodiscard]] __device__ constexpr auto lc_make_{type}4(lc_{type} s = 0) noexcept {{ return lc_{type}4{{s, s, s, s}}; }}
[[nodiscard]] __device__ constexpr auto lc_make_{type}4(lc_{type} x, lc_{type} y, lc_{type} z, lc_{type} w) noexcept {{ return lc_{type}4{{x, y, z, w}}; }}
[[nodiscard]] __device__ constexpr auto lc_make_{type}4(lc_{type} x, lc_{type} y, lc_{type}2 zw) noexcept {{ return lc_{type}4{{x, y, zw.x, zw.y}}; }}
[[nodiscard]] __device__ constexpr auto lc_make_{type}4(lc_{type} x, lc_{type}2 yz, lc_{type} w) noexcept {{ return lc_{type}4{{x, yz.x, yz.y, w}}; }}
[[nodiscard]] __device__ constexpr auto lc_make_{type}4(lc_{type}2 xy, lc_{type} z, lc_{type} w) noexcept {{ return lc_{type}4{{xy.x, xy.y, z, w}}; }}
[[nodiscard]] __device__ constexpr auto lc_make_{type}4(lc_{type}2 xy, lc_{type}2 zw) noexcept {{ return lc_{type}4{{xy.x, xy.y, zw.x, zw.y}}; }}
[[nodiscard]] __device__ constexpr auto lc_make_{type}4(lc_{type} x, lc_{type}3 yzw) noexcept {{ return lc_{type}4{{x, yzw.x, yzw.y, yzw.z}}; }}
[[nodiscard]] __device__ constexpr auto lc_make_{type}4(lc_{type}3 xyz, lc_{type} w) noexcept {{ return lc_{type}4{{xyz.x, xyz.y, xyz.z, w}}; }}""",
                  file=file)
            for t in scalar_types:
                print(
                    f"[[nodiscard]] __device__ constexpr auto lc_make_{type}4(lc_{t}4 v) noexcept {{ return lc_{type}4{{static_cast<lc_{type}>(v.x), static_cast<lc_{type}>(v.y), static_cast<lc_{type}>(v.z), static_cast<lc_{type}>(v.w)}}; }}",
                    file=file)
            print(file=file)

        # unary operators
        for type in scalar_types:
            for i in range(2, 5):
                elements = ["x", "y", "z", "w"][:i]
                print(
                    f"[[nodiscard]] __device__ constexpr auto operator!(lc_{type}{i} v) noexcept {{ return lc_make_bool{i}({', '.join(f'!v.{m}' for m in elements)}); }}",
                    file=file)
                if type != "bool":
                    print(
                        f"[[nodiscard]] __device__ constexpr auto operator+(lc_{type}{i} v) noexcept {{ return lc_make_{type}{i}({', '.join(f'+v.{m}' for m in elements)}); }}",
                        file=file)
                    print(
                        f"[[nodiscard]] __device__ constexpr auto operator-(lc_{type}{i} v) noexcept {{ return lc_make_{type}{i}({', '.join(f'-v.{m}' for m in elements)}); }}",
                        file=file)
                    if type != "float":
                        print(
                            f"[[nodiscard]] __device__ constexpr auto operator~(lc_{type}{i} v) noexcept {{ return lc_make_{type}{i}({', '.join(f'~v.{m}' for m in elements)}); }}",
                            file=file)
            print(file=file)


        def gen_binary_op(arg_t, ret_t, op):
            for i in range(2, 5):
                elements = ["x", "y", "z", "w"][:i]
                # vector-vector
                print(
                    f"[[nodiscard]] __device__ constexpr auto operator{op}(lc_{arg_t}{i} lhs, lc_{arg_t}{i} rhs) noexcept {{ return lc_make_{ret_t}{i}({', '.join(f'lhs.{m} {op} rhs.{m}' for m in elements)}); }}",
                    file=file)
                # vector-scalar
                operation = ", ".join(f"lhs.{e} {op} rhs" for e in "xyzw"[:i])
                print(
                    f"[[nodiscard]] __device__ constexpr auto operator{op}(lc_{arg_t}{i} lhs, lc_{arg_t} rhs) noexcept {{ return lc_make_{ret_t}{i}({operation}); }}",
                    file=file)
                # scalar-vector
                operation = ", ".join(f"lhs {op} rhs.{e}" for e in "xyzw"[:i])
                print(
                    f"[[nodiscard]] __device__ constexpr auto operator{op}(lc_{arg_t} lhs, lc_{arg_t}{i} rhs) noexcept {{ return lc_make_{ret_t}{i}({operation}); }}",
                    file=file)


        # binary operators
        for op in ["==", "!="]:
            for type in scalar_types:
                gen_binary_op(type, "bool", op)
            print(file=file)
        for op in ["<", ">", "<=", ">="]:
            for type in ["int", "uint", "float"]:
                gen_binary_op(type, "bool", op)
            print(file=file)
        for op in ["+", "-", "*", "/"]:
            for type in ["int", "uint", "float"]:
                gen_binary_op(type, type, op)
            print(file=file)
        for op in ["%", "<<", ">>"]:
            for type in ["int", "uint"]:
                gen_binary_op(type, type, op)
            print(file=file)
        for op in ["|", "&", "^"]:
            for type in ["int", "uint", "bool"]:
                gen_binary_op(type, type, op)
            print(file=file)
        for op in ["||", "&&"]:
            gen_binary_op("bool", "bool", op)
            print(file=file)


        def gen_assign_op(arg_t, op):
            newline = "\n"
            for i in range(2, 5):
                elements = ["x", "y", "z", "w"][:i]
                fast_op = "*= 1.0f /" if op == "/=" else op
                print(
                    f"""__device__ void operator{op}(lc_{arg_t}{i} &lhs, lc_{arg_t}{i} rhs) noexcept {{
{newline.join(f'    lhs.{m} {fast_op} rhs.{m};' for m in elements)}
}}""", file=file)
                elems = "\n".join(f"    lhs.{e} {op} rhs;" for e in "xyzw"[:i])
                print(
                    f"__device__ void operator{op}(lc_{arg_t}{i} &lhs, lc_{arg_t} rhs) noexcept {{{newline}{elems} }}",
                    file=file)


        # assign operators
        for op in ["+=", "-=", "*=", "/="]:
            for type in ["int", "uint", "float"]:
                gen_assign_op(type, op)
            print(file=file)
        for op in ["%=", "<<=", ">>="]:
            for type in ["int", "uint"]:
                gen_assign_op(type, op)
            print(file=file)
        for op in ["|=", "&=", "^="]:
            for type in ["int", "uint", "bool"]:
                gen_assign_op(type, op)
            print(file=file)

        # any, all, none
        for f, uop, bop in [("any", "", "||"), ("all", "", "&&"), ("none", "!", "&&")]:
            for i in range(2, 5):
                elements = ["x", "y", "z", "w"][:i]
                print(
                    f"[[nodiscard]] __device__ constexpr auto lc_{f}(lc_bool{i} v) noexcept {{ return {f' {bop} '.join(f'{uop}v.{m}' for m in elements)}; }}",
                    file=file)

        # matrix types
        for i in range(2, 5):
            def init(j):
                return ', '.join(["0.0f", "0.0f", "0.0f", "s", "0.0f", "0.0f", "0.0f"][3 - j:3 + i - j])


            print(f"""
struct lc_float{i}x{i} {{
    lc_float{i} cols[{i}];
    __device__ constexpr lc_float{i}x{i}() noexcept : cols{{}} {{}}
    __device__ explicit constexpr lc_float{i}x{i}(lc_float s) noexcept
        : cols{{{", ".join(f"lc_make_float{i}({init(j)})" for j in range(i))}}} {{}}
    __device__ constexpr lc_float{i}x{i}({", ".join(f"lc_float{i} c{j}" for j in range(i))}) noexcept
        : cols{{{", ".join(f"c{j}" for j in range(i))}}} {{}}
    [[nodiscard]] __device__ constexpr auto &operator[](lc_uint i) noexcept {{ return cols[i]; }}
    [[nodiscard]] __device__ constexpr auto operator[](lc_uint i) const noexcept {{ return cols[i]; }}
}};""", file=file)

        for i in range(2, 5):
            elements = ["x", "y", "z", "w"][:i]
            print(f"""
[[nodiscard]] __device__ constexpr auto operator*(const lc_float{i}x{i} m, lc_float s) noexcept {{ return lc_float{i}x{i}{{{", ".join(f"m[{j}] * s" for j in range(i))}}}; }}
[[nodiscard]] __device__ constexpr auto operator*(lc_float s, const lc_float{i}x{i} m) noexcept {{ return m * s; }}
[[nodiscard]] __device__ constexpr auto operator/(const lc_float{i}x{i} m, lc_float s) noexcept {{ return m * (1.0f / s); }}
[[nodiscard]] __device__ constexpr auto operator*(const lc_float{i}x{i} m, const lc_float{i} v) noexcept {{ return {' + '.join(f"v.{e} * m[{j}]" for j, e in enumerate(elements))}; }}
[[nodiscard]] __device__ constexpr auto operator*(const lc_float{i}x{i} lhs, const lc_float{i}x{i} rhs) noexcept {{ return lc_float{i}x{i}{{{', '.join(f"lhs * rhs[{j}]" for j in range(i))}}}; }}
[[nodiscard]] __device__ constexpr auto operator+(const lc_float{i}x{i} lhs, const lc_float{i}x{i} rhs) noexcept {{ return lc_float{i}x{i}{{{", ".join(f"lhs[{j}] + rhs[{j}]" for j in range(i))}}}; }}
[[nodiscard]] __device__ constexpr auto operator-(const lc_float{i}x{i} lhs, const lc_float{i}x{i} rhs) noexcept {{ return lc_float{i}x{i}{{{", ".join(f"lhs[{j}] - rhs[{j}]" for j in range(i))}}}; }}""",
                  file=file)

        for i in range(2, 5):
            def init(j):
                return ', '.join(["0.0f", "0.0f", "0.0f", "s", "0.0f", "0.0f", "0.0f"][3 - j:3 + i - j])


            print(f"""
[[nodiscard]] __device__ constexpr auto lc_make_float{i}x{i}(lc_float s = 1.0f) noexcept {{ return lc_float{i}x{i}{{{", ".join(f"lc_make_float{i}({init(j)})" for j in range(i))}}}; }}
[[nodiscard]] __device__ constexpr auto lc_make_float{i}x{i}({', '.join(', '.join(f"lc_float m{j}{k}" for k in range(i)) for j in range(i))}) noexcept {{ return lc_float{i}x{i}{{{", ".join(f"lc_make_float{i}({', '.join(f'm{j}{k}' for k in range(i))})" for j in range(i))}}}; }}
[[nodiscard]] __device__ constexpr auto lc_make_float{i}x{i}({", ".join(f"lc_float{i} c{j}" for j in range(i))}) noexcept {{ return lc_float{i}x{i}{{{", ".join(f"c{j}" for j in range(i))}}}; }}""",
                  file=file)
            if i == 3:
                print(
                    f"[[nodiscard]] __device__ constexpr auto lc_make_float{i}x{i}(lc_float2x2 m) noexcept {{ return lc_float3x3{{lc_make_float3(m[0], 0.0f), lc_make_float3(m[1], 0.0f), lc_make_float3(0.0f, 0.0f, 1.0f)}}; }}",
                    file=file)
            if i == 4:
                print(
                    f"[[nodiscard]] __device__ constexpr auto lc_make_float{i}x{i}(lc_float2x2 m) noexcept {{ return lc_float4x4{{lc_make_float4(m[0], 0.0f, 0.0f), lc_make_float4(m[1], 0.0f, 0.0f), lc_make_float4(0.0f, 0.0f, 0.0f, 0.0f), lc_make_float4(0.0f, 0.0f, 0.0f, 1.0f)}}; }}",
                    file=file)
                print(
                    f"[[nodiscard]] __device__ constexpr auto lc_make_float{i}x{i}(lc_float3x3 m) noexcept {{ return lc_float4x4{{lc_make_float4(m[0], 0.0f), lc_make_float4(m[1], 0.0f), lc_make_float4(m[2], 0.0f), lc_make_float4(0.0f, 0.0f, 0.0f, 1.0f)}}; }}",
                    file=file)
            print(f"[[nodiscard]] __device__ constexpr auto lc_make_float{i}x{i}(lc_float{i}x{i} m) noexcept {{ return m; }}",
                  file=file)
            for t in range(i + 1, 5):
                print(
                    f"[[nodiscard]] __device__ constexpr auto lc_make_float{i}x{i}(lc_float{t}x{t} m) noexcept {{ return lc_float{i}x{i}{{{', '.join(f'lc_make_float{i}(m[{j}])' for j in range(i))}}}; }}",
                    file=file)
        print(file=file)

        print('''[[nodiscard]] inline bool isinf_impl(lc_float x) noexcept {
    auto u = __float_as_int(x);
    return u == 0x7f800000u | u == 0xff800000u;
}
[[nodiscard]] inline bool isnan_impl(lc_float x) noexcept {
    auto u = __float_as_int(x);
    return ((u & 0x7F800000u) == 0x7F800000u) & ((u & 0x7FFFFFu) != 0u);
}
''', file=file)


        def generate_vector_call(name, c, types, args):
            types = [{"i": "int",
                      "u": "uint",
                      "f": "float",
                      "b": "bool"}[t] for t in types]

            def call(i):
                e = "xyzw"[i]
                return f"{c}(" + ", ".join(f"{a}.{e}" for a in args) + ")"

            for t in types:
                print(
                    f"[[nodiscard]] __device__ inline auto lc_{name}({', '.join(f'lc_{t} {a}' for a in args)}) noexcept {{ return {c}({', '.join(args)}); }}",
                    file=file)
                for n in range(2, 5):
                    print(
                        f"[[nodiscard]] __device__ inline auto lc_{name}({', '.join(f'lc_{t}{n} {a}' for a in args)}) noexcept {{ return lc_make_{t if name not in ['isnan', 'isinf'] else 'bool'}{n}({', '.join(call(i) for i in range(n))}); }}",
                        file=file)
            print(file=file)


        # select
        print(
            "template<typename T>\n[[nodiscard]] __device__ inline auto lc_select(T f, T t, bool p) noexcept { return p ? t : f; }",
            file=file)
        for t in ["int", "uint", "float"]:
            for n in range(2, 5):
                print(
                    f"[[nodiscard]] __device__ inline auto lc_select(lc_{t}{n} f, lc_{t}{n} t, lc_bool{n} p) noexcept {{ return lc_make_{t}{n}({', '.join(f'lc_select<lc_{t}>(f.{e}, t.{e}, p.{e})' for e in 'xyzw'[:n])}); }}",
                    file=file)
        print(file=file)

        # min/max/abs/acos/asin/asinh/acosh/atan/atanh/atan2/
        # cos/cosh/sin/sinh/tan/tanh/exp/exp2/exp10/log/log2/
        # log10/sqrt/rsqrt/ceil/floor/trunc/round/fma/copysignf/
        # isinf/isnan
        generate_vector_call("min", "min", "iu", ["a", "b"])
        generate_vector_call("max", "max", "iu", ["a", "b"])
        generate_vector_call("abs", "abs", "i", ["x"])
        generate_vector_call("min", "fminf", "f", ["a", "b"])
        generate_vector_call("max", "fmaxf", "f", ["a", "b"])
        generate_vector_call("abs", "fabsf", "f", ["x"])
        generate_vector_call("acos", "acosf", "f", ["x"])
        generate_vector_call("asin", "asinf", "f", ["x"])
        generate_vector_call("atan", "atanf", "f", ["x"])
        generate_vector_call("acosh", "acoshf", "f", ["x"])
        generate_vector_call("asinh", "asinhf", "f", ["x"])
        generate_vector_call("atanh", "atanhf", "f", ["x"])
        generate_vector_call("atan2", "atan2f", "f", ["y", "x"])
        generate_vector_call("cos", "cosf", "f", ["x"])
        generate_vector_call("cosh", "coshf", "f", ["x"])
        generate_vector_call("sin", "sinf", "f", ["x"])
        generate_vector_call("sinh", "sinhf", "f", ["x"])
        generate_vector_call("tan", "tanf", "f", ["x"])
        generate_vector_call("tanh", "tanhf", "f", ["x"])
        generate_vector_call("exp", "expf", "f", ["x"])
        generate_vector_call("exp2", "exp2f", "f", ["x"])
        generate_vector_call("exp10", "exp10f", "f", ["x"])
        generate_vector_call("log", "logf", "f", ["x"])
        generate_vector_call("log2", "log2f", "f", ["x"])
        generate_vector_call("log10", "log10f", "f", ["x"])
        generate_vector_call("pow", "powf", "f", ["x", "a"])
        generate_vector_call("sqrt", "sqrtf", "f", ["x"])
        generate_vector_call("rsqrt", "rsqrtf", "f", ["x"])
        generate_vector_call("ceil", "ceilf", "f", ["x"])
        generate_vector_call("floor", "floorf", "f", ["x"])
        generate_vector_call("trunc", "truncf", "f", ["x"])
        generate_vector_call("round", "roundf", "f", ["x"])
        generate_vector_call("fma", "fmaf", "f", ["x", "y", "z"])
        generate_vector_call("copysign", "copysignf", "f", ["x", "y"])
        generate_vector_call("isinf", "isinf_impl", "f", ["x"])
        generate_vector_call("isnan", "isnan_impl", "f", ["x"])

        # clamp
        for t in ["int", "uint", "float"]:
            print(
                f"[[nodiscard]] __device__ inline auto lc_clamp_impl(lc_{t} v, lc_{t} lo, lc_{t} hi) noexcept {{ return lc_min(lc_max(v, lo), hi); }}",
                file=file)
        generate_vector_call("clamp", "lc_clamp_impl", "iuf", ["v", "lo", "hi"])

        # lerp
        print(
            f"[[nodiscard]] __device__ inline auto lc_lerp_impl(lc_float a, lc_float b, lc_float t) noexcept {{ return lc_fma(t, b - a, a); }}",
            file=file)
        generate_vector_call("lerp", "lc_lerp_impl", "f", ["a", "b", "t"])

        # saturate
        print(
            "[[nodiscard]] __device__ inline auto lc_saturate(lc_float x) noexcept { return lc_clamp(x, 0.0f, 1.0f); }",
            file=file)
        for n in range(2, 5):
            print(
                f"[[nodiscard]] __device__ inline auto lc_saturate(lc_float{n} x) noexcept {{ return lc_clamp(x, lc_make_float{n}(0.0f), lc_make_float{n}(1.0f)); }}",
                file=file)
        print(file=file)

        # degrees/radians
        print(
            f"[[nodiscard]] __device__ inline auto lc_degrees_impl(lc_float rad) noexcept {{ return rad * (180.0f * 0.318309886183790671537767526745028724f); }}",
            file=file)
        generate_vector_call("degrees", "lc_degrees_impl", "f", ["rad"])
        print(
            f"[[nodiscard]] __device__ inline auto lc_radians_impl(lc_float deg) noexcept {{ return deg * (3.14159265358979323846264338327950288f / 180.0f); }}",
            file=file)
        generate_vector_call("radians", "lc_radians_impl", "f", ["deg"])

        # step
        print(
            f"[[nodiscard]] __device__ inline auto lc_step_impl(lc_float edge, lc_float x) noexcept {{ return x < edge ? 0.0f : 1.0f; }}",
            file=file)
        generate_vector_call("step", "lc_step_impl", "f", ["edge", "x"])

        # smoothstep
        print(
            f"""[[nodiscard]] __device__ inline auto lc_smoothstep_impl(lc_float edge0, lc_float edge1, lc_float x) noexcept {{
    auto t = lc_clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * lc_fma(-2.f, t, 3.f);
}}""",
            file=file)
        generate_vector_call("smoothstep", "lc_smoothstep_impl", "f", ["edge0", "edge1", "x"])

        # mod
        print(
            f"[[nodiscard]] __device__ inline auto lc_mod_impl(lc_float x, lc_float y) noexcept {{ return lc_fma(-y, lc_floor(x / y), x); }}",
            file=file)
        generate_vector_call("mod", "lc_mod_impl", "f", ["x", "y"])

        # fmod
        generate_vector_call("fmod", "fmodf", "f", ["x", "y"])

        # fract
        print(
            f"[[nodiscard]] __device__ inline auto lc_fract_impl(lc_float x) noexcept {{ return x - lc_floor(x); }}",
            file=file)
        generate_vector_call("fract", "lc_fract_impl", "f", ["x"])

        # clz/popcount/reverse
        generate_vector_call("clz", "__clz", "u", ["x"])
        generate_vector_call("popcount", "__popc", "u", ["x"])
        generate_vector_call("reverse", "__brev", "u", ["x"])

        # ctz
        print(
            f"[[nodiscard]] __device__ inline auto lc_ctz_impl(lc_uint x) noexcept {{ return 32u - __clz(x); }}",
            file=file)
        generate_vector_call("ctz", "lc_ctz_impl", "u", ["x"])

        # cross
        print("""[[nodiscard]] __device__ constexpr auto lc_cross(lc_float3 u, lc_float3 v) noexcept {
    return lc_make_float3(u.y * v.z - v.y * u.z,
                          u.z * v.x - v.z * u.x,
                          u.x * v.y - v.x * u.y);
}""", file=file)
        print(file=file)

        # dot
        print("""[[nodiscard]] __device__ inline auto lc_dot(lc_float2 a, lc_float2 b) noexcept {
    return a.x * b.x + a.y * b.y;
}""", file=file)
        print("""[[nodiscard]] __device__ inline auto lc_dot(lc_float3 a, lc_float3 b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}""", file=file)
        print("""[[nodiscard]] __device__ inline auto lc_dot(lc_float4 a, lc_float4 b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}""", file=file)
        print(file=file)

        # length
        print(f"[[nodiscard]] __device__ inline auto lc_length(lc_float2 v) noexcept {{ return sqrtf(lc_dot(v, v)); }}", file=file)
        print(f"[[nodiscard]] __device__ inline auto lc_length(lc_float3 v) noexcept {{ return sqrtf(lc_dot(v, v)); }}",
              file=file)
        print(f"[[nodiscard]] __device__ inline auto lc_length(lc_float4 v) noexcept {{ return sqrtf(lc_dot(v, v)); }}",
              file=file)
        print(file=file)

        # length_squared
        for n in range(2, 5):
            print(f"[[nodiscard]] __device__ inline auto lc_length_squared(lc_float{n} v) noexcept {{ return lc_dot(v, v); }}",
                  file=file)
        print(file=file)

        # distance
        for n in range(2, 5):
            print(
                f"[[nodiscard]] __device__ inline auto lc_distance(lc_float{n} a, lc_float{n} b) noexcept {{ return lc_length(a - b); }}",
                file=file)
        print(file=file)

        # distance_squared
        for n in range(2, 5):
            print(
                f"[[nodiscard]] __device__ inline auto lc_distance_squared(lc_float{n} a, lc_float{n} b) noexcept {{ return lc_length_squared(a - b); }}",
                file=file)
        print(file=file)

        # normalize
        for n in range(2, 5):
            inv_norm = {
                2: "rhypotf",
                3: "rnorm3df",
                4: "rnorm4df"
            }[n]
            print(
                f"[[nodiscard]] __device__ inline auto lc_normalize(lc_float{n} v) noexcept {{ return v * rsqrtf(lc_dot(v, v)); }}",
                file=file)
        print(file=file)

        # faceforward
        print(
            "[[nodiscard]] __device__ inline auto lc_faceforward(lc_float3 n, lc_float3 i, lc_float3 n_ref) noexcept { return lc_dot(n_ref, i) < 0.0f ? n : -n; }",
            file=file)
        print(file=file)

        # transpose
        print("""[[nodiscard]] __device__ constexpr auto lc_transpose(const lc_float2x2 m) noexcept { return lc_make_float2x2(m[0].x, m[1].x, m[0].y, m[1].y); }
[[nodiscard]] __device__ constexpr auto lc_transpose(const lc_float3x3 m) noexcept { return lc_make_float3x3(m[0].x, m[1].x, m[2].x, m[0].y, m[1].y, m[2].y, m[0].z, m[1].z, m[2].z); }
[[nodiscard]] __device__ constexpr auto lc_transpose(const lc_float4x4 m) noexcept { return lc_make_float4x4(m[0].x, m[1].x, m[2].x, m[3].x, m[0].y, m[1].y, m[2].y, m[3].y, m[0].z, m[1].z, m[2].z, m[3].z, m[0].w, m[1].w, m[2].w, m[3].w); }
""", file=file)

        # determinant/inverse
        print("""[[nodiscard]] __device__ constexpr auto lc_determinant(const lc_float2x2 m) noexcept {
    return m[0][0] * m[1][1] - m[1][0] * m[0][1];
}

[[nodiscard]] __device__ constexpr auto lc_determinant(const lc_float3x3 m) noexcept {// from GLM
    return m[0].x * (m[1].y * m[2].z - m[2].y * m[1].z)
         - m[1].x * (m[0].y * m[2].z - m[2].y * m[0].z)
         + m[2].x * (m[0].y * m[1].z - m[1].y * m[0].z);
}

[[nodiscard]] __device__ constexpr auto lc_determinant(const lc_float4x4 m) noexcept {// from GLM
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

[[nodiscard]] __device__ constexpr auto lc_inverse(const lc_float2x2 m) noexcept {
    const auto one_over_determinant = 1.0f / (m[0][0] * m[1][1] - m[1][0] * m[0][1]);
    return lc_make_float2x2(m[1][1] * one_over_determinant,
                           -m[0][1] * one_over_determinant,
                           -m[1][0] * one_over_determinant,
                           +m[0][0] * one_over_determinant);
}

[[nodiscard]] __device__ constexpr auto lc_inverse(const lc_float3x3 m) noexcept {// from GLM
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

[[nodiscard]] __device__ constexpr auto lc_inverse(const lc_float4x4 m) noexcept {// from GLM
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

[[nodiscard]] __device__ inline auto lc_half_to_float(unsigned short x) noexcept {
    lc_float val;
    asm("{  cvt.f32.f16 %0, %1;}\\n" : "=f"(val) : "h"(x));
    return val;
}

[[nodiscard]] __device__ inline auto lc_float_to_half(lc_float x) noexcept {
    unsigned short val;
    asm("{  cvt.rn.f16.f32 %0, %1;}\\n" : "=h"(val) : "f"(x));
    return val;
}

template<typename D, typename S>
[[nodiscard]] inline auto lc_bit_cast(S s) noexcept {
    static_assert(sizeof(D) == sizeof(S));
    return reinterpret_cast<const D &>(s);
}
""", file=file)

    def src2c(lib, postfix):
        with open(f"{curr_dir}/{lib}.{postfix}", "r") as fin:
            content = "".join(fin.readlines()).replace(".version 7.6", ".version 6.3")
            chars = [c for c in content] + ['\0']
        with open(f"{curr_dir}/{lib}_embedded.inl.h", "w") as fout:
            print(f"static const char {lib}_source[{len(chars) + 1}] = {{", file=fout)
            chars_per_row = 32
            rows = (len(chars) + chars_per_row) // chars_per_row
            for row in range(rows):
                begin = row * chars_per_row
                end = begin + chars_per_row
                line = ", ".join(f"0x{ord(c):02x}" for c in chars[begin:end])
                print(f"    {line}{'' if row + 1 == rows else ','}", file=fout)
            print("};", file=fout)

    src2c(math_library_name, "h")
    src2c(surf_library_name, "h")
    src2c("cuda_accel_update", "ptx")
