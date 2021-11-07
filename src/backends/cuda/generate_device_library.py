from os.path import realpath, dirname

if __name__ == "__main__":
    curr_dir = dirname(realpath(__file__))

    with open(f"{curr_dir}/cuda_device_lib.h", "w") as file:
        # scalar types
        print("""#pragma once

#include <cmath>
""", file=file)
        ns = "lc_"
        scalar_types = ["int", "uint", "float", "bool"]
        native_types = ["int", "unsigned int", "float", "bool"]
        for t, native_t in zip(scalar_types, native_types):
            print(f"using {ns}{t} = {native_t};", file=file)
        print(file=file)

        # vector types
        vector_alignments = {2: 8, 3: 16, 4: 16}
        for type in scalar_types:
            for i in range(2, 5):
                elements = ["x", "y", "z", "w"][:i]
                print(f"""struct alignas({vector_alignments[i]}) {ns}{type}{i} {{
    {ns}{type} {', '.join(elements[:i + 1])};
    explicit constexpr {ns}{type}{i}({type} s) noexcept
        : {', '.join(f"{m}{{s}}" for m in elements)} {{}}
    constexpr {ns}{type}{i}({', '.join(f"{ns}{type} {m}" for m in elements)}) noexcept
        : {', '.join(f"{m}{{{m}}}" for m in elements)} {{}}
    constexpr auto &operator[]({ns}uint i) noexcept {{ return (&x)[i]; }}
    constexpr auto operator[]({ns}uint i) const noexcept {{ return (&x)[i]; }}
}};
""", file=file)

        # make type[n]
        for type in scalar_types:
            # make type2
            print(f"""[[nodiscard]] constexpr auto {ns}make_{type}2({ns}{type} s) noexcept {{ return {ns}{type}2{{s, s}}; }}
[[nodiscard]] constexpr auto {ns}make_{type}2({ns}{type} x, {ns}{type} y) noexcept {{ return {ns}{type}2{{x, y}}; }}""",
                  file=file)
            for t in scalar_types:
                for l in range(2, 5):
                    print(
                        f"[[nodiscard]] constexpr auto {ns}make_{type}2({ns}{t}{l} v) noexcept {{ return {ns}{type}2{{static_cast<{ns}{type}>(v.x), static_cast<{ns}{type}>(v.y)}}; }}",
                        file=file)
            # make type3
            print(f"""[[nodiscard]] constexpr auto {ns}make_{type}3({ns}{type} s) noexcept {{ return {ns}{type}3{{s, s, s}}; }}
[[nodiscard]] constexpr auto {ns}make_{type}3({ns}{type} x, {ns}{type} y, {ns}{type} z) noexcept {{ return {ns}{type}3{{x, y, z}}; }}
[[nodiscard]] constexpr auto {ns}make_{type}3({ns}{type} x, {ns}{type}2 yz) noexcept {{ return {ns}{type}3{{x, yz.x, yz.y}}; }}
[[nodiscard]] constexpr auto {ns}make_{type}3({ns}{type}2 xy, {ns}{type} z) noexcept {{ return {ns}{type}3{{xy.x, xy.y, z}}; }}""",
                  file=file)
            for t in scalar_types:
                for l in range(3, 5):
                    print(
                        f"[[nodiscard]] constexpr auto {ns}make_{type}3({ns}{t}{l} v) noexcept {{ return {ns}{type}3{{static_cast<{ns}{type}>(v.x), static_cast<{ns}{type}>(v.y), static_cast<{ns}{type}>(v.z)}}; }}",
                        file=file)
            # make type4
            print(f"""[[nodiscard]] constexpr auto {ns}make_{type}4({ns}{type} s) noexcept {{ return {ns}{type}4{{s, s, s, s}}; }}
[[nodiscard]] constexpr auto {ns}make_{type}4({ns}{type} x, {ns}{type} y, {ns}{type} z, {ns}{type} w) noexcept {{ return {ns}{type}4{{x, y, z, w}}; }}
[[nodiscard]] constexpr auto {ns}make_{type}4({ns}{type} x, {ns}{type} y, {ns}{type}2 zw) noexcept {{ return {ns}{type}4{{x, y, zw.x, zw.y}}; }}
[[nodiscard]] constexpr auto {ns}make_{type}4({ns}{type} x, {ns}{type}2 yz, {ns}{type} w) noexcept {{ return {ns}{type}4{{x, yz.x, yz.y, w}}; }}
[[nodiscard]] constexpr auto {ns}make_{type}4({ns}{type}2 xy, {ns}{type} z, {ns}{type} w) noexcept {{ return {ns}{type}4{{xy.x, xy.y, z, w}}; }}
[[nodiscard]] constexpr auto {ns}make_{type}4({ns}{type}2 xy, {ns}{type}2 zw) noexcept {{ return {ns}{type}4{{xy.x, xy.y, zw.x, zw.y}}; }}
[[nodiscard]] constexpr auto {ns}make_{type}4({ns}{type} x, {ns}{type}3 yzw) noexcept {{ return {ns}{type}4{{x, yzw.x, yzw.y, yzw.z}}; }}
[[nodiscard]] constexpr auto {ns}make_{type}4({ns}{type}3 xyz, {ns}{type} w) noexcept {{ return {ns}{type}4{{xyz.x, xyz.y, xyz.z, w}}; }}""",
                  file=file)
            for t in scalar_types:
                print(
                    f"[[nodiscard]] constexpr auto {ns}make_{type}4({ns}{t}4 v) noexcept {{ return {ns}{type}4{{static_cast<{ns}{type}>(v.x), static_cast<{ns}{type}>(v.y), static_cast<{ns}{type}>(v.z), static_cast<{ns}{type}>(v.w)}}; }}",
                    file=file)
            print(file=file)

        # unary operators
        for type in scalar_types:
            for i in range(2, 5):
                elements = ["x", "y", "z", "w"][:i]
                print(
                    f"[[nodiscard]] constexpr auto operator!({ns}{type}{i} v) noexcept {{ return {ns}make_bool{i}({', '.join(f'!v.{m}' for m in elements)}); }}",
                    file=file)
                if type != "bool":
                    print(
                        f"[[nodiscard]] constexpr auto operator+({ns}{type}{i} v) noexcept {{ return {ns}make_{type}{i}({', '.join(f'+v.{m}' for m in elements)}); }}",
                        file=file)
                    print(
                        f"[[nodiscard]] constexpr auto operator-({ns}{type}{i} v) noexcept {{ return {ns}make_{type}{i}({', '.join(f'-v.{m}' for m in elements)}); }}",
                        file=file)
                    if type != "float":
                        print(
                            f"[[nodiscard]] constexpr auto operator~({ns}{type}{i} v) noexcept {{ return {ns}make_{type}{i}({', '.join(f'~v.{m}' for m in elements)}); }}",
                            file=file)
            print(file=file)


        def gen_binary_op(arg_t, ret_t, op):
            for i in range(2, 5):
                elements = ["x", "y", "z", "w"][:i]
                if op == "/" and arg_t == "float":
                    print(
                        f"[[nodiscard]] constexpr auto operator{op}({ns}{arg_t}{i} lhs, {ns}{arg_t}{i} rhs) noexcept {{ return {ns}make_{ret_t}{i}({', '.join(f'lhs.{m} * (1.0f / rhs.{m})' for m in elements)}); }}",
                        file=file)
                    print(
                        f"[[nodiscard]] constexpr auto operator{op}({ns}{arg_t}{i} lhs, {ns}{arg_t} rhs) noexcept {{ return lhs * {ns}make_{arg_t}{i}(1.0f / rhs); }}",
                        file=file)
                else:
                    print(
                        f"[[nodiscard]] constexpr auto operator{op}({ns}{arg_t}{i} lhs, {ns}{arg_t}{i} rhs) noexcept {{ return {ns}make_{ret_t}{i}({', '.join(f'lhs.{m} {op} rhs.{m}' for m in elements)}); }}",
                        file=file)
                    print(
                        f"[[nodiscard]] constexpr auto operator{op}({ns}{arg_t}{i} lhs, {ns}{arg_t} rhs) noexcept {{ return lhs {op} {ns}make_{arg_t}{i}(rhs); }}",
                        file=file)
                print(
                    f"[[nodiscard]] constexpr auto operator{op}({ns}{arg_t} lhs, {ns}{arg_t}{i} rhs) noexcept {{ return {ns}make_{arg_t}{i}(lhs) {op} rhs; }}",
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
                    f"""[[nodiscard]] constexpr auto &operator{op}({ns}{arg_t}{i} &lhs, {ns}{arg_t}{i} rhs) noexcept {{
{newline.join(f'    lhs.{m} {fast_op} rhs.{m};' for m in elements)}
    return lhs;
}}""", file=file)
                if op == "/=" and arg_t == "float":
                    print(
                        f"[[nodiscard]] constexpr auto &operator{op}({ns}{arg_t}{i} &lhs, {ns}{arg_t} rhs) noexcept {{ return lhs *= {ns}make_{arg_t}{i}(1.0f / rhs); }}",
                        file=file)
                else:
                    print(
                        f"[[nodiscard]] constexpr auto &operator{op}({ns}{arg_t}{i} &lhs, {ns}{arg_t} rhs) noexcept {{ return lhs {op} {ns}make_{arg_t}{i}(rhs); }}",
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
                    f"[[nodiscard]] constexpr auto {ns}{f}({ns}bool{i} v) noexcept {{ return {f' {bop} '.join(f'{uop}v.{m}' for m in elements)}; }}",
                    file=file)

        # matrix types
        for i in range(2, 5):
            def init(j):
                return ', '.join(["0.0f", "0.0f", "0.0f", "s", "0.0f", "0.0f", "0.0f"][3 - j:3 + i - j])


            print(f"""
struct {ns}float{i}x{i} {{
    {ns}float{i} cols[{i}];
    constexpr {ns}float{i}x{i}({ns}float s = 1.0f) noexcept
        : cols{{{", ".join(f"{ns}make_float{i}({init(j)})" for j in range(i))}}} {{}}
    constexpr {ns}float{i}x{i}({", ".join(f"{ns}float{i} c{j}" for j in range(i))}) noexcept
        : cols{{{", ".join(f"c{j}" for j in range(i))}}} {{}}
    [[nodiscard]] constexpr auto &operator[]({ns}uint i) noexcept {{ return cols[i]; }}
    [[nodiscard]] constexpr auto operator[]({ns}uint i) const noexcept {{ return cols[i]; }}
}};""", file=file)

        for i in range(2, 5):
            elements = ["x", "y", "z", "w"][:i]
            print(f"""
[[nodiscard]] constexpr auto operator*(const {ns}float{i}x{i} m, {ns}float s) noexcept {{ return {ns}float{i}x{i}{{{", ".join(f"m[{j}] * s" for j in range(i))}}}; }}
[[nodiscard]] constexpr auto operator*({ns}float s, const {ns}float{i}x{i} m) noexcept {{ return m * s; }}
[[nodiscard]] constexpr auto operator/(const {ns}float{i}x{i} m, {ns}float s) noexcept {{ return m * (1.0f / s); }}
[[nodiscard]] constexpr auto operator*(const {ns}float{i}x{i} m, const {ns}float{i} v) noexcept {{ return {' + '.join(f"v.{e} * m[{j}]" for j, e in enumerate(elements))}; }}
[[nodiscard]] constexpr auto operator*(const {ns}float{i}x{i} lhs, const {ns}float{i}x{i} rhs) noexcept {{ return {ns}float{i}x{i}{{{', '.join(f"lhs * rhs[{j}]" for j in range(i))}}}; }}
[[nodiscard]] constexpr auto operator+(const {ns}float{i}x{i} lhs, const {ns}float{i}x{i} rhs) noexcept {{ return {ns}float{i}x{i}{{{", ".join(f"lhs[{j}] + rhs[{j}]" for j in range(i))}}}; }}
[[nodiscard]] constexpr auto operator-(const {ns}float{i}x{i} lhs, const {ns}float{i}x{i} rhs) noexcept {{ return {ns}float{i}x{i}{{{", ".join(f"lhs[{j}] - rhs[{j}]" for j in range(i))}}}; }}""",
                  file=file)

        for i in range(2, 5):
            def init(j):
                return ', '.join(["0.0f", "0.0f", "0.0f", "s", "0.0f", "0.0f", "0.0f"][3 - j:3 + i - j])


            print(f"""
[[nodiscard]] constexpr auto {ns}make_float{i}x{i}({ns}float s = 1.0f) noexcept {{ return {ns}float{i}x{i}{{{", ".join(f"{ns}make_float{i}({init(j)})" for j in range(i))}}}; }}
[[nodiscard]] constexpr auto {ns}make_float{i}x{i}({', '.join(', '.join(f"{ns}float m{j}{k}" for k in range(i)) for j in range(i))}) noexcept {{ return {ns}float{i}x{i}{{{", ".join(f"{ns}make_float{i}({', '.join(f'm{j}{k}' for k in range(i))})" for j in range(i))}}}; }}
[[nodiscard]] constexpr auto {ns}make_float{i}x{i}({", ".join(f"{ns}float{i} c{j}" for j in range(i))}) noexcept {{ return {ns}float{i}x{i}{{{", ".join(f"c{j}" for j in range(i))}}}; }}""",
                  file=file)
            if i == 3:
                print(f"[[nodiscard]] constexpr auto {ns}make_float{i}x{i}({ns}float2x2 m) noexcept {{ return {ns}float3x3{{{ns}make_float3(m[0], 0.0f), {ns}make_float3(m[1], 0.0f), {ns}make_float3(0.0f, 0.0f, 1.0f)}}; }}", file=file)
            if i == 4:
                print(f"[[nodiscard]] constexpr auto {ns}make_float{i}x{i}({ns}float2x2 m) noexcept {{ return {ns}float4x4{{{ns}make_float4(m[0], 0.0f, 0.0f), {ns}make_float4(m[1], 0.0f, 0.0f), {ns}make_float4(0.0f, 0.0f, 0.0f, 0.0f), {ns}make_float4(0.0f, 0.0f, 0.0f, 1.0f)}}; }}", file=file)
                print(f"[[nodiscard]] constexpr auto {ns}make_float{i}x{i}({ns}float3x3 m) noexcept {{ return {ns}float4x4{{{ns}make_float4(m[0], 0.0f), {ns}make_float4(m[1], 0.0f), {ns}make_float4(m[2], 0.0f), {ns}make_float4(0.0f, 0.0f, 0.0f, 1.0f)}}; }}", file=file)
            print(f"[[nodiscard]] constexpr auto {ns}make_float{i}x{i}({ns}float{i}x{i} m) noexcept {{ return m; }}", file=file)
            for t in range(i + 1, 5):
                print(f"[[nodiscard]] constexpr auto {ns}make_float{i}x{i}({ns}float{t}x{t} m) noexcept {{ return {ns}float{i}x{i}{{{', '.join(f'{ns}make_float{i}(m[{j}])' for j in range(i))}}}; }}", file=file)
