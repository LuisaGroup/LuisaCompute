from os.path import realpath, dirname

if __name__ == "__main__":
    curr_dir = dirname(realpath(__file__))
    math_library_name = "ispc_device_math"
    surf_library_name = "ispc_device_resource"
    with open(f"{curr_dir}/{math_library_name}.isph", "w") as file:

        # constants
        print('''// constants
#define M_E         2.71828182845904523536028747135266250f   /* e              */
#define M_LOG2E     1.44269504088896340735992468100189214f   /* log2(e)        */
#define M_LOG10E    0.434294481903251827651128918916605082f  /* log10(e)       */
#define M_LN2       0.693147180559945309417232121458176568f  /* loge(2)        */
#define M_LN10      2.30258509299404568401799145468436421f   /* loge(10)       */
#define M_PI        3.14159265358979323846264338327950288f   /* pi             */
#define M_PI_2      1.57079632679489661923132169163975144f   /* pi/2           */
#define M_PI_4      0.785398163397448309615660845819875721f  /* pi/4           */
#define M_1_PI      0.318309886183790671537767526745028724f  /* 1/pi           */
#define M_2_PI      0.636619772367581343075535053490057448f  /* 2/pi           */
#define M_2_SQRTPI  1.12837916709551257389615890312154517f   /* 2/sqrt(pi)     */
#define M_SQRT2     1.41421356237309504880168872420969808f   /* sqrt(2)        */
#define M_SQRT1_2   0.707106781186547524400844362104849039f  /* 1/sqrt(2)      */
        ''', file=file)
        # vector types
        scalar_types = ["int", "uint", "float", "bool"]
        vector_elements = "xyzw"
        for t in scalar_types:
            for n in range(2, 5):
                print(f"typedef {t}<{n}> {t}{n};", file=file)
        template = '''
inline uniform {T}2 make_{T}2() {
    uniform {T}2 v;
    v.x = v.y = 0;
    return v;
}
inline uniform {T}3 make_{T}3() {
    uniform {T}3 v;
    v.x = v.y = v.z = 0;
    return v;
}
inline uniform {T}4 make_{T}4() {
    uniform {T}4 v;
    v.x = v.y = v.z = v.w = 0;
    return v;
}'''
        for t in scalar_types:
            print(template.replace("{T}", t), file=file)
        # make_typeN
        template = '''
// make_{T}2 functions
inline {U}2 make_{T}2({U} s) {
    {U}2 v;
    v.x = v.y = s;
    return v;
}
inline {U}2 make_{T}2({U} x, {U} y) {
    {U}2 v;
    v.x = x;
    v.y = y;
    return v;
}
inline {U}2 make_{T}2({U}2 v) {
    return v;
}
inline {U}2 make_{T}2({U}3 v) {
    {U}2 u;
    u.x = v.x;
    u.y = v.y;
    return u;
}
inline {U}2 make_{T}2({U}4 v) {
    {U}2 u;
    u.x = v.x;
    u.y = v.y;
    return u;
}

// make_{T}3 functions
inline {U}3 make_{T}3({U} s) {
    {U}3 v;
    v.x = v.y = v.z = s;
    return v;
}
inline {U}3 make_{T}3({U} x, {U} y, {U} z) {
    {U}3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}
inline {U}3 make_{T}3({U} x, {U}2 yz) {
    {U}3 v;
    v.x = x;
    v.y = yz.x;
    v.z = yz.y;
    return v;
}
inline {U}3 make_{T}3({U}2 xy, {U} z) {
    {U}3 v;
    v.x = xy.x;
    v.y = xy.y;
    v.z = z;
    return v;
}
inline {U}3 make_{T}3({U}3 v) {
    return v;
}
inline {U}3 make_{T}3({U}4 v) {
    {U}3 u;
    u.x = v.x;
    u.y = v.y;
    u.z = v.z;
    return u;
}

// make_{T}4 functions
inline {U}4 make_{T}4({U} s) {
    {U}4 v;
    v.x = v.y = v.z = v.w = s;
    return v;
}
inline {U}4 make_{T}4({U} x, {U} y, {U} z, {U} w) {
    {U}4 v;
    v.x = x;
    v.y = y;
    v.z = z;
    v.w = w;
    return v;
}
inline {U}4 make_{T}4({U} x, {U} y, {U}2 zw) {
    {U}4 v;
    v.x = x;
    v.y = y;
    v.z = zw.x;
    v.w = zw.y;
    return v;
}
inline {U}4 make_{T}4({U} x, {U}2 yz, {U} w) {
    {U}4 v;
    v.x = x;
    v.y = yz.x;
    v.z = yz.y;
    v.w = w;
    return v;
}
inline {U}4 make_{T}4({U}2 xy, {U} z, {U} w) {
    {U}4 v;
    v.x = xy.x;
    v.y = xy.y;
    v.z = z;
    v.w = w;
    return v;
}
inline {U}4 make_{T}4({U}2 xy, {U}2 zw) {
    {U}4 v;
    v.x = xy.x;
    v.y = xy.y;
    v.z = zw.x;
    v.w = zw.y;
    return v;
}
inline {U}4 make_{T}4({U} x, {U}3 yzw) {
    {U}4 v;
    v.x = x;
    v.y = yzw.x;
    v.z = yzw.y;
    v.w = yzw.z;
    return v;
}
inline {U}4 make_{T}4({U}3 xyz, {U} w) {
    {U}4 v;
    v.x = xyz.x;
    v.y = xyz.y;
    v.z = xyz.z;
    v.w = w;
    return v;
}
inline {U}4 make_{T}4({U}4 v) {
    return v;
}'''
        for t in scalar_types:
            print(template.replace("{U}", t).replace("{T}", t), file=file)
            print(template.replace("{U}", f"uniform {t}").replace("{T}", t), file=file)
        template = '''
// conversions
inline {U}2 make_{T}2({other}2 v) {
    {U}2 u;
    u.x = ({U})v.x;
    u.y = ({U})v.y;
    return u;
}
inline {U}2 make_{T}2({other}3 v) {
    {U}2 u;
    u.x = ({U})v.x;
    u.y = ({U})v.y;
    return u;
}
inline {U}2 make_{T}2({other}4 v) {
    {U}2 u;
    u.x = ({U})v.x;
    u.y = ({U})v.y;
    return u;
}
inline {U}3 make_{T}3({other}3 v) {
    {U}3 u;
    u.x = ({U})v.x;
    u.y = ({U})v.y;
    u.z = ({U})v.z;
    return u;
}
inline {U}3 make_{T}3({other}4 v) {
    {U}3 u;
    u.x = ({U})v.x;
    u.y = ({U})v.y;
    u.z = ({U})v.z;
    return u;
}
inline {U}4 make_{T}4({other}4 v) {
    {U}4 u;
    u.x = ({U})v.x;
    u.y = ({U})v.y;
    u.z = ({U})v.z;
    u.w = ({U})v.w;
    return u;
}'''
        for t in scalar_types:
            for other in [o for o in scalar_types if o != t]:
                print(template.replace("{U}", t).replace("{T}", t).replace("{other}", other), file=file)
                print(template.replace("{U}", f"uniform {t}").replace("{T}", t).replace("{other}", f"uniform {other}"),
                      file=file)
        print(file=file)

        # unary operators
        for type in scalar_types:
            print(
                f"inline bool unary_not({type} s) {{ return !s; }}",
                file=file)
            print(
                f"inline uniform bool unary_not(uniform {type} s) {{ return !s; }}",
                file=file)
            if type != "bool":
                print(
                    f"inline {type} unary_plus({type} s) {{ return +s; }}",
                    file=file)
                print(
                    f"inline {type} unary_minus({type} s) {{ return -s; }}",
                    file=file)
                print(
                    f"inline uniform {type} unary_plus(uniform {type} s) {{ return +s; }}",
                    file=file)
                print(
                    f"inline uniform {type} unary_minus(uniform {type} s) {{ return -s; }}",
                    file=file)
                if type != "float":
                    print(
                        f"inline {type} unary_bit_not({type} s) {{ return ~s; }}",
                        file=file)
                    print(
                        f"inline uniform {type} unary_bit_not(uniform {type} s) {{ return ~s; }}",
                        file=file)
            for i in range(2, 5):
                elements = ["x", "y", "z", "w"][:i]
                print(
                    f"inline bool{i} unary_not({type}{i} v) {{ return make_bool{i}({', '.join(f'!v.{m}' for m in elements)}); }}",
                    file=file)
                print(
                    f"inline uniform bool{i} unary_not(uniform {type}{i} v) {{ return make_bool{i}({', '.join(f'!v.{m}' for m in elements)}); }}",
                    file=file)
                if type != "bool":
                    print(
                        f"inline {type}{i} unary_plus({type}{i} v) {{ return make_{type}{i}({', '.join(f'+v.{m}' for m in elements)}); }}",
                        file=file)
                    print(
                        f"inline {type}{i} unary_minus({type}{i} v) {{ return make_{type}{i}({', '.join(f'-v.{m}' for m in elements)}); }}",
                        file=file)
                    print(
                        f"inline uniform {type}{i} unary_plus(uniform {type}{i} v) {{ return make_{type}{i}({', '.join(f'+v.{m}' for m in elements)}); }}",
                        file=file)
                    print(
                        f"inline uniform {type}{i} unary_minus(uniform {type}{i} v) {{ return make_{type}{i}({', '.join(f'-v.{m}' for m in elements)}); }}",
                        file=file)
                    if type != "float":
                        print(
                            f"inline {type}{i} unary_bit_not({type}{i} v) {{ return make_{type}{i}({', '.join(f'~v.{m}' for m in elements)}); }}",
                            file=file)
                        print(
                            f"inline {type}{i} unary_bit_not(uniform {type}{i} v) {{ return make_{type}{i}({', '.join(f'~v.{m}' for m in elements)}); }}",
                            file=file)
            print(file=file)


        def gen_binary_op(arg_t, ret_t, op):
            op2name = {
                "==": "binary_eq",
                "!=": "binary_ne",
                "<": "binary_lt",
                ">": "binary_gt",
                "<=": "binary_le",
                ">=": "binary_ge",
                "+": "binary_add",
                "-": "binary_sub",
                "*": "binary_mul",
                "/": "binary_div",
                "%": "binary_mod",
                "<<": "binary_shl",
                ">>": "binary_shr",
                "|": "binary_bit_or",
                "&": "binary_bit_and",
                "^": "binary_bit_xor",
                "||": "binary_or",
                "&&": "binary_and"}
            # scalar-scalar
            print(
                f"inline uniform {ret_t} {op2name[op]}(uniform {arg_t} lhs, uniform {arg_t} rhs) {{ return lhs {op} rhs; }}",
                file=file)
            print(
                f"inline {ret_t} {op2name[op]}({arg_t} lhs, uniform {arg_t} rhs) {{ return lhs {op} rhs; }}",
                file=file)
            print(
                f"inline {ret_t} {op2name[op]}(uniform {arg_t} lhs, {arg_t} rhs) {{ return lhs {op} rhs; }}",
                file=file)
            print(
                f"inline {ret_t} {op2name[op]}({arg_t} lhs, {arg_t} rhs) {{ return lhs {op} rhs; }}", file=file)
            for i in range(2, 5):
                elements = ["x", "y", "z", "w"][:i]
                # vector-vector
                print(
                    f"inline uniform {ret_t}{i} {op2name[op]}(uniform {arg_t}{i} lhs, uniform {arg_t}{i} rhs) {{ return make_{ret_t}{i}({', '.join(f'lhs.{m} {op} rhs.{m}' for m in elements)}); }}",
                    file=file)
                print(
                    f"inline {ret_t}{i} {op2name[op]}({arg_t}{i} lhs, uniform {arg_t}{i} rhs) {{ return make_{ret_t}{i}({', '.join(f'lhs.{m} {op} rhs.{m}' for m in elements)}); }}",
                    file=file)
                print(
                    f"inline {ret_t}{i} {op2name[op]}(uniform {arg_t}{i} lhs, {arg_t}{i} rhs) {{ return make_{ret_t}{i}({', '.join(f'lhs.{m} {op} rhs.{m}' for m in elements)}); }}",
                    file=file)
                print(
                    f"inline {ret_t}{i} {op2name[op]}({arg_t}{i} lhs, {arg_t}{i} rhs) {{ return make_{ret_t}{i}({', '.join(f'lhs.{m} {op} rhs.{m}' for m in elements)}); }}",
                    file=file)
                # vector-scalar
                operation = ", ".join(f"lhs.{e} {op} rhs" for e in "xyzw"[:i])
                print(
                    f"inline uniform {ret_t}{i} {op2name[op]}(uniform {arg_t}{i} lhs, uniform {arg_t} rhs) {{ return make_{ret_t}{i}({operation}); }}",
                    file=file)
                print(
                    f"inline {ret_t}{i} {op2name[op]}({arg_t}{i} lhs, uniform {arg_t} rhs) {{ return make_{ret_t}{i}({operation}); }}",
                    file=file)
                print(
                    f"inline {ret_t}{i} {op2name[op]}(uniform {arg_t}{i} lhs, {arg_t} rhs) {{ return make_{ret_t}{i}({operation}); }}",
                    file=file)
                print(
                    f"inline {ret_t}{i} {op2name[op]}({arg_t}{i} lhs, {arg_t} rhs) {{ return make_{ret_t}{i}({operation}); }}",
                    file=file)
                # scalar-vector
                operation = ", ".join(f"lhs {op} rhs.{e}" for e in "xyzw"[:i])
                print(
                    f"inline uniform {ret_t}{i} {op2name[op]}(uniform {arg_t} lhs, uniform {arg_t}{i} rhs) {{ return make_{ret_t}{i}({operation}); }}",
                    file=file)
                print(
                    f"inline {ret_t}{i} {op2name[op]}({arg_t} lhs, uniform {arg_t}{i} rhs) {{ return make_{ret_t}{i}({operation}); }}",
                    file=file)
                print(
                    f"inline {ret_t}{i} {op2name[op]}(uniform {arg_t} lhs, {arg_t}{i} rhs) {{ return make_{ret_t}{i}({operation}); }}",
                    file=file)
                print(
                    f"inline {ret_t}{i} {op2name[op]}({arg_t} lhs, {arg_t}{i} rhs) {{ return make_{ret_t}{i}({operation}); }}",
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
        # any, all, none
        for f, uop, bop in [("any", "", "||"), ("all", "", "&&"), ("none", "!", "&&")]:
            for i in range(2, 5):
                elements = ["x", "y", "z", "w"][:i]
                print(
                    f"inline uniform bool {f}(uniform bool{i} v) {{ return {f' {bop} '.join(f'{uop}v.{m}' for m in elements)}; }}",
                    file=file)
                print(
                    f"inline bool {f}(bool{i} v) {{ return {f' {bop} '.join(f'{uop}v.{m}' for m in elements)}; }}",
                    file=file)
        print(file=file)


        def generate_vector_call(name, c, types, args):
            types = [{"i": "int",
                      "u": "uint",
                      "f": "float",
                      "b": "bool"}[t] for t in types]

            def call(i):
                e = "xyzw"[i]
                return f"{c}(" + ", ".join(f"{a}.{e}" for a in args) + ")"

            for t in types:
                for n in range(2, 5):
                    ret_t = f"{t if name not in ['isnan', 'isinf'] else 'bool'}{n}"
                    print(
                        f"inline {ret_t} {name}({', '.join(f'{t}{n} {a}' for a in args)}) {{ return make_{ret_t}({', '.join(call(i) for i in range(n))}); }}",
                        file=file)
                    print(
                        f"inline uniform {ret_t} {name}({', '.join(f'uniform {t}{n} {a}' for a in args)}) {{ return make_{ret_t}({', '.join(call(i) for i in range(n))}); }}",
                        file=file)
            print(file=file)


        # select
        print(
            "#define select_scalar(f, t, p) ((p) ? (t) : (f))",
            file=file)
        for t in ["int", "uint", "float"]:
            for n in range(2, 5):
                print(
                    f"inline {t}{n} select({t}{n} f, {t}{n} t, bool{n} p) {{ return make_{t}{n}({', '.join(f'select_scalar(f.{e}, t.{e}, p.{e})' for e in 'xyzw'[:n])}); }}",
                    file=file)
                print(
                    f"inline {t}{n} select({t}{n} f, {t}{n} t, uniform bool{n} p) {{ return make_{t}{n}({', '.join(f'select_scalar(f.{e}, t.{e}, p.{e})' for e in 'xyzw'[:n])}); }}",
                    file=file)
                print(
                    f"inline uniform {t}{n} select(uniform {t}{n} f, uniform {t}{n} t, uniform bool{n} p) {{ return make_{t}{n}({', '.join(f'select_scalar(f.{e}, t.{e}, p.{e})' for e in 'xyzw'[:n])}); }}",
                    file=file)

        print('''
inline float fma(float a, float b, float c) { return a * b + c; }
inline uniform float fma(uniform float a, uniform float b, uniform float c) { return a * b + c; }
inline float copysign(float a, float b) { return floatbits((intbits(a) & 0x7fffffffu) | signbits(b)); }
inline uniform float copysign(uniform float a, uniform float b) { return floatbits((intbits(a) & 0x7fffffffu) | signbits(b)); }
inline float log2(float x) { return log(x) / log(2.f); }
inline uniform float log2(uniform float x) { return log(x) / log(2.f); }
inline float log10(float x) { return log(x) / log(10.f); }
inline uniform float log10(uniform float x) { return log(x) / log(10.f); }
inline float exp2(float x) { return pow(2.f, x); }
inline uniform float exp2(uniform float x) { return pow(2.f, x); }
inline float exp10(float x) { return pow(10.f, x); }
inline uniform float exp10(uniform float x) { return pow(10.f, x); }
inline bool is_nan(float x) {
  uint u = intbits(x);
  return (u & 0x7F800000u) == 0x7F800000u && (u & 0x7FFFFFu);
}
inline bool is_inf(float x) {
  uint u = intbits(x);
  return (u & 0x7F800000u) == 0x7F800000u && !(u & 0x7FFFFFu);
}
inline uniform bool is_nan(uniform float x) {
  uniform uint u = intbits(x);
  return (u & 0x7F800000u) == 0x7F800000u && (u & 0x7FFFFFu);
}
inline uniform bool is_inf(uniform float x) {
  uniform uint u = intbits(x);
  return (u & 0x7F800000u) == 0x7F800000u && !(u & 0x7FFFFFu);
}
inline float sinh(float x) { return .5f * (exp(x) - exp(-x)); }
inline uniform float sinh(uniform float x) { return .5f * (exp(x) - exp(-x)); }
inline float cosh(float x) { return .5f * (exp(x) + exp(-x)); }
inline uniform float cosh(uniform float x) { return .5f * (exp(x) + exp(-x)); }
inline float tanh(float x) { return sinh(x) / cosh(x); }
inline uniform float tanh(uniform float x) { return sinh(x) / cosh(x); }
inline float asinh(float x) { return log(x + sqrt(x * x + 1.f)); }
inline uniform float asinh(uniform float x) { return log(x + sqrt(x * x + 1.f)); }
inline float acosh(float x) { return log(x + sqrt(x * x - 1.f)); }
inline uniform float acosh(uniform float x) { return log(x + sqrt(x * x - 1.f)); }
inline float atanh(float x) { return .5f * log((1.f + x) / (1.f - x)); }
inline uniform float atanh(uniform float x) { return .5f * log((1.f + x) / (1.f - x)); }
inline float saturate(float x) { return clamp(x, 0.f, 1.f); }
inline uniform float saturate(uniform float x) { return clamp(x, 0.f, 1.f); }
inline float lerp(float a, float b, float t) { return fma(t, b - a, a); }
inline uniform float lerp(uniform float a, uniform float b, uniform float t) { return fma(t, b - a, a); }
inline float degrees(float x) { return x * (180.f * M_1_PI); }
inline uniform float degrees(uniform float x) { return x * (180.f * M_1_PI); }
inline float radians(float x) { return x * (M_PI / 180.f); }
inline uniform float radians(uniform float x) { return x * (M_PI / 180.f); }
inline float step(float edge, float x) { return x < edge ? 0.0f : 1.0f; }
inline uniform float step(uniform float edge, uniform float x) { return x < edge ? 0.0f : 1.0f; }
inline float smoothstep(float edge0, float edge1, float x) {
    float t = saturate((x - edge0) / (edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}
inline uniform float smoothstep(uniform float edge0, uniform float edge1, uniform float x) {
    uniform float t = saturate((x - edge0) / (edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}
inline uint ctz(uint x) { return count_trailing_zeros((int)x); }
inline uniform uint ctz(uniform uint x) { return count_trailing_zeros((int)x); }
inline uint clz(uint x) { return count_leading_zeros((int)x); }
inline uniform uint clz(uniform uint x) { return count_leading_zeros((int)x); }
inline uint popcount(uint x) { return popcnt((int)x); }
inline uniform uint popcount(uniform uint x) { return popcnt((int)x); }

inline float3 cross(float3 u, float3 v) {
    return make_float3(
        u.y * v.z - v.y * u.z,
        u.z * v.x - v.z * u.x,
        u.x * v.y - v.x * u.y);
}
inline uniform float3 cross(uniform float3 u, uniform float3 v) {
    return make_float3(
        u.y * v.z - v.y * u.z,
        u.z * v.x - v.z * u.x,
        u.x * v.y - v.x * u.y);
}
inline float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float dot(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
inline uniform float dot(uniform float2 a, uniform float2 b) { return a.x * b.x + a.y * b.y; }
inline uniform float dot(uniform float3 a, uniform float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline uniform float dot(uniform float4 a, uniform float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
inline float length(float2 v) { return sqrt(dot(v, v)); }
inline float length(float3 v) { return sqrt(dot(v, v)); }
inline float length(float4 v) { return sqrt(dot(v, v)); }
inline uniform float length(uniform float2 v) { return sqrt(dot(v, v)); }
inline uniform float length(uniform float3 v) { return sqrt(dot(v, v)); }
inline uniform float length(uniform float4 v) { return sqrt(dot(v, v)); }
inline float length_squared(float2 v) { return dot(v, v); }
inline float length_squared(float3 v) { return dot(v, v); }
inline float length_squared(float4 v) { return dot(v, v); }
inline uniform float length_squared(uniform float2 v) { return dot(v, v); }
inline uniform float length_squared(uniform float3 v) { return dot(v, v); }
inline uniform float length_squared(uniform float4 v) { return dot(v, v); }
inline float2 normalize(float2 v) { return v * rsqrt(length_squared(v)); }
inline float3 normalize(float3 v) { return v * rsqrt(length_squared(v)); }
inline float4 normalize(float4 v) { return v * rsqrt(length_squared(v)); }
inline uniform float2 normalize(uniform float2 v) { return v * rsqrt(length_squared(v)); }
inline uniform float3 normalize(uniform float3 v) { return v * rsqrt(length_squared(v)); }
inline uniform float4 normalize(uniform float4 v) { return v * rsqrt(length_squared(v)); }
inline float3 faceforward(float3 n, float3 i, float3 n_ref) { return dot(n_ref, i) < 0.f ? n : -n; }
inline uniform float3 faceforward(uniform float3 n, uniform float3 i, uniform float3 n_ref) { return dot(n_ref, i) < 0.f ? n : -n; }
''', file=file)

        # min/max/abs/acos/asin/asinh/acosh/atan/atanh/atan2/
        # cos/cosh/sin/sinh/tan/tanh/exp/exp2/exp10/log/log2/
        # log10/sqrt/rsqrt/ceil/floor/trunc/round/fma/copysignf/
        # isinf/isnan/clamp/saturate/lerp/radians/degrees/step
        # smoothstep/clz/ctz/popcount
        generate_vector_call("min", "min", "iuf", ["a", "b"])
        generate_vector_call("max", "max", "iuf", ["a", "b"])
        generate_vector_call("abs", "abs", "if", ["x"])
        generate_vector_call("acos", "acos", "f", ["x"])
        generate_vector_call("asin", "asin", "f", ["x"])
        generate_vector_call("atan", "atan", "f", ["x"])
        generate_vector_call("acosh", "acosh", "f", ["x"])
        generate_vector_call("asinh", "asinh", "f", ["x"])
        generate_vector_call("atanh", "atanh", "f", ["x"])
        generate_vector_call("atan2", "atan2", "f", ["y", "x"])
        generate_vector_call("cos", "cos", "f", ["x"])
        generate_vector_call("cosh", "cosh", "f", ["x"])
        generate_vector_call("sin", "sin", "f", ["x"])
        generate_vector_call("sinh", "sinh", "f", ["x"])
        generate_vector_call("tan", "tan", "f", ["x"])
        generate_vector_call("tanh", "tanh", "f", ["x"])
        generate_vector_call("exp", "exp", "f", ["x"])
        generate_vector_call("exp2", "exp2", "f", ["x"])
        generate_vector_call("exp10", "exp10", "f", ["x"])
        generate_vector_call("log", "log", "f", ["x"])
        generate_vector_call("log2", "log2", "f", ["x"])
        generate_vector_call("log10", "log10", "f", ["x"])
        generate_vector_call("pow", "pow", "f", ["x", "a"])
        generate_vector_call("sqrt", "sqrt", "f", ["x"])
        generate_vector_call("rsqrt", "rsqrt", "f", ["x"])
        generate_vector_call("ceil", "ceil", "f", ["x"])
        generate_vector_call("floor", "floor", "f", ["x"])
        generate_vector_call("trunc", "trunc", "f", ["x"])
        generate_vector_call("round", "round", "f", ["x"])
        generate_vector_call("fma", "fma", "f", ["x", "y", "z"])
        generate_vector_call("copysign", "copysign", "f", ["x", "y"])
        generate_vector_call("is_inf", "is_inf", "f", ["x"])
        generate_vector_call("is_nan", "is_nan", "f", ["x"])
        generate_vector_call("clamp", "clamp", "iuf", ["x", "l", "h"])
        generate_vector_call("saturate", "saturate", "f", ["x"])
        generate_vector_call("lerp", "lerp", "f", ["a", "b", "t"])
        generate_vector_call("radians", "radians", "f", ["x"])
        generate_vector_call("degrees", "degrees", "f", ["x"])
        generate_vector_call("step", "step", "f", ["e", "x"])
        generate_vector_call("smoothstep", "smoothstep", "f", ["e0", "e1", "x"])
        generate_vector_call("clz", "clz", "u", ["x"])
        generate_vector_call("ctz", "ctz", "u", ["x"])
        generate_vector_call("popcount", "popcount", "u", ["x"])

        # matrix types
        print('''
struct float2x2 {
    float2 cols[2];
};
struct float3x3 {
    float3 cols[3];
};
struct float4x4 {
    float4 cols[4];
};

inline float2 matrix_access(float2x2 m, uint i) { return m.cols[i]; }
inline float2 matrix_access(float2x2 m, int i) { return m.cols[i]; }
inline float2 matrix_access(uniform float2x2 m, uint i) { return m.cols[i]; }
inline float2 matrix_access(uniform float2x2 m, int i) { return m.cols[i]; }
inline float2 matrix_access(float2x2 m, uniform uint i) { return m.cols[i]; }
inline float2 matrix_access(float2x2 m, uniform int i) { return m.cols[i]; }
inline uniform float2 matrix_access(uniform float2x2 m, uniform uint i) { return m.cols[i]; }
inline uniform float2 matrix_access(uniform float2x2 m, uniform int i) { return m.cols[i]; }

inline float3 matrix_access(float3x3 m, uint i) { return m.cols[i]; }
inline float3 matrix_access(float3x3 m, int i) { return m.cols[i]; }
inline float3 matrix_access(uniform float3x3 m, uint i) { return m.cols[i]; }
inline float3 matrix_access(uniform float3x3 m, int i) { return m.cols[i]; }
inline float3 matrix_access(float3x3 m, uniform uint i) { return m.cols[i]; }
inline float3 matrix_access(float3x3 m, uniform int i) { return m.cols[i]; }
inline uniform float3 matrix_access(uniform float3x3 m, uniform uint i) { return m.cols[i]; }
inline uniform float3 matrix_access(uniform float3x3 m, uniform int i) { return m.cols[i]; }

inline float4 matrix_access(float4x4 m, uint i) { return m.cols[i]; }
inline float4 matrix_access(float4x4 m, int i) { return m.cols[i]; }
inline float4 matrix_access(uniform float4x4 m, uint i) { return m.cols[i]; }
inline float4 matrix_access(uniform float4x4 m, int i) { return m.cols[i]; }
inline float4 matrix_access(float4x4 m, uniform uint i) { return m.cols[i]; }
inline float4 matrix_access(float4x4 m, uniform int i) { return m.cols[i]; }
inline uniform float4 matrix_access(uniform float4x4 m, uniform uint i) { return m.cols[i]; }
inline uniform float4 matrix_access(uniform float4x4 m, uniform int i) { return m.cols[i]; }

inline uniform float2x2 make_float2x2() {
    uniform float2x2 m;
    m.cols[0] = make_float2();
    m.cols[1] = make_float2();
    return m;
}
inline uniform float3x3 make_float3x3() {
    uniform float3x3 m;
    m.cols[0] = make_float3();
    m.cols[1] = make_float3();
    m.cols[2] = make_float3();
    return m;
}
inline uniform float4x4 make_float4x4() {
    uniform float4x4 m;
    m.cols[0] = make_float4();
    m.cols[1] = make_float4();
    m.cols[2] = make_float4();
    m.cols[3] = make_float4();
    return m;
}''', file=file)
        template = '''
// make_float2x2 functions
inline uniform float2x2 make_float2x2(uniform float s) {
    uniform float2x2 m;
    m.cols[0] = make_float2(s);
    m.cols[1] = make_float2(s);
    return m;
}
inline uniform float2x2 make_float2x2(uniform float2 c0, uniform float2 c1) {
    uniform float2x2 m;
    m.cols[0] = c0;
    m.cols[1] = c1;
    return m;
}
inline uniform float2x2 make_float2x2(
        uniform float m00, uniform float m01,
        uniform float m10, uniform float m11) {
    uniform float2x2 m;
    m.cols[0] = make_float2(m00, m01);
    m.cols[1] = make_float2(m10, m11);
    return m;
}
inline uniform float2x2 make_float2x2(uniform float2x2 m) { return m; }

// make_float3x3 functions
inline uniform float3x3 make_float3x3(uniform float s) {
    uniform float3x3 m;
    m.cols[0] = make_float3(s);
    m.cols[1] = make_float3(s);
    m.cols[2] = make_float3(s);
    return m;
}
inline uniform float3x3 make_float3x3(uniform float3 c0, uniform float3 c1, uniform float3 c2) {
    uniform float3x3 m;
    m.cols[0] = c0;
    m.cols[1] = c1;
    m.cols[2] = c2;
    return m;
}
inline uniform float3x3 make_float3x3(
        uniform float m00, uniform float m01, uniform float m02,
        uniform float m10, uniform float m11, uniform float m12,
        uniform float m20, uniform float m21, uniform float m22) {
    uniform float3x3 m;
    m.cols[0] = make_float3(m00, m01, m02);
    m.cols[1] = make_float3(m10, m11, m12);
    m.cols[2] = make_float3(m20, m21, m22);
    return m;
}
inline uniform float3x3 make_float3x3(uniform float3x3 m) { return m; }

// make_float4x4 functions
inline uniform float4x4 make_float4x4(uniform float s) {
    uniform float4x4 m;
    m.cols[0] = make_float4(s);
    m.cols[1] = make_float4(s);
    m.cols[2] = make_float4(s);
    m.cols[3] = make_float4(s);
    return m;
}
inline uniform float4x4 make_float4x4(uniform float4 c0, uniform float4 c1, uniform float4 c2, uniform float4 c3) {
    uniform float4x4 m;
    m.cols[0] = c0;
    m.cols[1] = c1;
    m.cols[2] = c2;
    m.cols[3] = c3;
    return m;
}
inline uniform float4x4 make_float4x4(
        uniform float m00, uniform float m01, uniform float m02, uniform float m03,
        uniform float m10, uniform float m11, uniform float m12, uniform float m13,
        uniform float m20, uniform float m21, uniform float m22, uniform float m23,
        uniform float m30, uniform float m31, uniform float m32, uniform float m33) {
    uniform float4x4 m;
    m.cols[0] = make_float4(m00, m01, m02, m03);
    m.cols[1] = make_float4(m10, m11, m12, m13);
    m.cols[2] = make_float4(m20, m21, m22, m23);
    m.cols[3] = make_float4(m30, m31, m32, m33);
    return m;
}
inline uniform float4x4 make_float4x4(uniform float4x4 m) { return m; }

// conversions
inline uniform float2x2 make_float2x2(uniform float3x3 m) {
    uniform float2x2 n;
    n.cols[0] = make_float2(m.cols[0]);
    n.cols[1] = make_float2(m.cols[1]);
    return n;
}
inline uniform float2x2 make_float2x2(uniform float4x4 m) {
    uniform float2x2 n;
    n.cols[0] = make_float2(m.cols[0]);
    n.cols[1] = make_float2(m.cols[1]);
    return n;
}
inline uniform float3x3 make_float3x3(uniform float2x2 m) {
    uniform float3x3 n;
    n.cols[0] = make_float3(m.cols[0], 0.f);
    n.cols[1] = make_float3(m.cols[1], 0.f);
    n.cols[2] = make_float3(0.f, 0.f, 1.f);
    return n;
}
inline uniform float3x3 make_float3x3(uniform float4x4 m) {
    uniform float3x3 n;
    n.cols[0] = make_float3(m.cols[0]);
    n.cols[1] = make_float3(m.cols[1]);
    n.cols[2] = make_float3(m.cols[2]);
    return n;
}
inline uniform float4x4 make_float4x4(uniform float2x2 m) {
    uniform float4x4 n;
    n.cols[0] = make_float4(m.cols[0], 0.f, 0.f);
    n.cols[1] = make_float4(m.cols[1], 0.f, 0.f);
    n.cols[2] = make_float4(0.f, 0.f, 1.f, 0.f);
    n.cols[3] = make_float4(0.f, 0.f, 0.f, 1.f);
    return n;
}
inline uniform float4x4 make_float4x4(uniform float3x3 m) {
    uniform float4x4 n;
    n.cols[0] = make_float4(m.cols[0], 0.f);
    n.cols[1] = make_float4(m.cols[1], 0.f);
    n.cols[2] = make_float4(m.cols[2], 0.f);
    n.cols[3] = make_float4(0.f, 0.f, 0.f, 1.f);
    return n;
}

// unary operators
inline uniform float2x2 unary_plus(uniform float2x2 m) { return m; }
inline uniform float3x3 unary_plus(uniform float3x3 m) { return m; }
inline uniform float4x4 unary_plus(uniform float4x4 m) { return m; }
inline uniform float2x2 unary_minus(uniform float2x2 m) {
    return make_float2x2(
        unary_minus(m.cols[0]),
        unary_minus(m.cols[1]));
}
inline uniform float3x3 unary_minus(uniform float3x3 m) {
    return make_float3x3(
        unary_minus(m.cols[0]),
        unary_minus(m.cols[1]),
        unary_minus(m.cols[2]));
}
inline uniform float4x4 unary_minus(uniform float4x4 m) {
    return make_float4x4(
        unary_minus(m.cols[0]),
        unary_minus(m.cols[1]),
        unary_minus(m.cols[2]),
        unary_minus(m.cols[3]));
}'''
        print(template, file=file)
        print(template.replace("uniform ", ""), file=file)

        template = '''
// matrix-scalar binary operators
inline {ret}float2x2 binary_add({lhs}float2x2 m, {rhs}float s) {
    return make_float2x2(
        binary_add(m.cols[0], s),
        binary_add(m.cols[1], s));
}
inline {ret}float2x2 binary_add({lhs}float s, {rhs}float2x2 m) {
    return make_float2x2(
        binary_add(s, m.cols[0]),
        binary_add(s, m.cols[1]));
}
inline {ret}float2x2 binary_sub({lhs}float2x2 m, {rhs}float s) {
    return make_float2x2(
        binary_sub(m.cols[0], s),
        binary_sub(m.cols[1], s));
}
inline {ret}float2x2 binary_sub({lhs}float s, {rhs}float2x2 m) {
    return make_float2x2(
        binary_sub(s, m.cols[0]),
        binary_sub(s, m.cols[1]));
}
inline {ret}float2x2 binary_mul({lhs}float2x2 m, {rhs}float s) {
    return make_float2x2(
        binary_mul(m.cols[0], s),
        binary_mul(m.cols[1], s));
}
inline {ret}float2x2 binary_mul({lhs}float s, {rhs}float2x2 m) {
    return make_float2x2(
        binary_mul(s, m.cols[0]),
        binary_mul(s, m.cols[1]));
}
inline {ret}float2x2 binary_div({lhs}float2x2 m, {rhs}float s) {
    return make_float2x2(
        binary_div(m.cols[0], s),
        binary_div(m.cols[1], s));
}
inline {ret}float2x2 binary_div({lhs}float s, {rhs}float2x2 m) {
    return make_float2x2(
        binary_div(s, m.cols[0]),
        binary_div(s, m.cols[1]));
}
inline {ret}float3x3 binary_add({lhs}float3x3 m, {rhs}float s) {
    return make_float3x3(
        binary_add(m.cols[0], s),
        binary_add(m.cols[1], s),
        binary_add(m.cols[2], s));
}
inline {ret}float3x3 binary_add({lhs}float s, {rhs}float3x3 m) {
    return make_float3x3(
        binary_add(s, m.cols[0]),
        binary_add(s, m.cols[1]),
        binary_add(s, m.cols[2]));
}
inline {ret}float3x3 binary_sub({lhs}float3x3 m, {rhs}float s) {
    return make_float3x3(
        binary_sub(m.cols[0], s),
        binary_sub(m.cols[1], s),
        binary_sub(m.cols[2], s));
}
inline {ret}float3x3 binary_sub({lhs}float s, {rhs}float3x3 m) {
    return make_float3x3(
        binary_sub(s, m.cols[0]),
        binary_sub(s, m.cols[1]),
        binary_sub(s, m.cols[2]));
}
inline {ret}float3x3 binary_mul({lhs}float3x3 m, {rhs}float s) {
    return make_float3x3(
        binary_mul(m.cols[0], s),
        binary_mul(m.cols[1], s),
        binary_mul(m.cols[2], s));
}
inline {ret}float3x3 binary_mul({lhs}float s, {rhs}float3x3 m) {
    return make_float3x3(
        binary_mul(s, m.cols[0]),
        binary_mul(s, m.cols[1]),
        binary_mul(s, m.cols[2]));
}
inline {ret}float3x3 binary_div({lhs}float3x3 m, {rhs}float s) {
    return make_float3x3(
        binary_div(m.cols[0], s),
        binary_div(m.cols[1], s),
        binary_div(m.cols[2], s));
}
inline {ret}float3x3 binary_div({lhs}float s, {rhs}float3x3 m) {
    return make_float3x3(
        binary_div(s, m.cols[0]),
        binary_div(s, m.cols[1]),
        binary_div(s, m.cols[2]));
}
inline {ret}float4x4 binary_add({lhs}float4x4 m, {rhs}float s) {
    return make_float4x4(
        binary_add(m.cols[0], s),
        binary_add(m.cols[1], s),
        binary_add(m.cols[2], s),
        binary_add(m.cols[3], s));
}
inline {ret}float4x4 binary_add({lhs}float s, {rhs}float4x4 m) {
    return make_float4x4(
        binary_add(s, m.cols[0]),
        binary_add(s, m.cols[1]),
        binary_add(s, m.cols[2]),
        binary_add(s, m.cols[3]));
}
inline {ret}float4x4 binary_sub({lhs}float4x4 m, {rhs}float s) {
    return make_float4x4(
        binary_sub(m.cols[0], s),
        binary_sub(m.cols[1], s),
        binary_sub(m.cols[2], s),
        binary_sub(m.cols[3], s));
}
inline {ret}float4x4 binary_sub({lhs}float s, {rhs}float4x4 m) {
    return make_float4x4(
        binary_sub(s, m.cols[0]),
        binary_sub(s, m.cols[1]),
        binary_sub(s, m.cols[2]),
        binary_sub(s, m.cols[3]));
}
inline {ret}float4x4 binary_mul({lhs}float4x4 m, {rhs}float s) {
    return make_float4x4(
        binary_mul(m.cols[0], s),
        binary_mul(m.cols[1], s),
        binary_mul(m.cols[2], s),
        binary_mul(m.cols[3], s));
}
inline {ret}float4x4 binary_mul({lhs}float s, {rhs}float4x4 m) {
    return make_float4x4(
        binary_mul(s, m.cols[0]),
        binary_mul(s, m.cols[1]),
        binary_mul(s, m.cols[2]),
        binary_mul(s, m.cols[3]));
}
inline {ret}float4x4 binary_div({lhs}float4x4 m, {rhs}float s) {
    return make_float4x4(
        binary_div(m.cols[0], s),
        binary_div(m.cols[1], s),
        binary_div(m.cols[2], s),
        binary_div(m.cols[3], s));
}
inline {ret}float4x4 binary_div({lhs}float s, {rhs}float4x4 m) {
    return make_float4x4(
        binary_div(s, m.cols[0]),
        binary_div(s, m.cols[1]),
        binary_div(s, m.cols[2]),
        binary_div(s, m.cols[3]));
}

// matrix-vector binary operators
inline {ret}float2 binary_mul({lhs}float2x2 m, {rhs}float2 v) {
    return binary_mul(m.cols[0], v.x) +
           binary_mul(m.cols[1], v.y);
}
inline {ret}float3 binary_mul({lhs}float3x3 m, {rhs}float3 v) {
    return binary_mul(m.cols[0], v.x) +
           binary_mul(m.cols[1], v.y) +
           binary_mul(m.cols[2], v.z);
}
inline {ret}float4 binary_mul({lhs}float4x4 m, {rhs}float4 v) {
    return binary_mul(m.cols[0], v.x) +
           binary_mul(m.cols[1], v.y) +
           binary_mul(m.cols[2], v.z) +
           binary_mul(m.cols[3], v.w);
}

// matrix-matrix binary operators
inline {ret}float2x2 binary_add({lhs}float2x2 lhs, {rhs}float2x2 rhs) {
    return make_float2x2(
        binary_add(lhs.cols[0], rhs.cols[0]),
        binary_add(lhs.cols[1], rhs.cols[1]));
}
inline {ret}float2x2 binary_sub({lhs}float2x2 lhs, {rhs}float2x2 rhs) {
    return make_float2x2(
        binary_sub(lhs.cols[0], rhs.cols[0]),
        binary_sub(lhs.cols[1], rhs.cols[1]));
}
inline {ret}float3x3 binary_add({lhs}float3x3 lhs, {rhs}float3x3 rhs) {
    return make_float3x3(
        binary_add(lhs.cols[0], rhs.cols[0]),
        binary_add(lhs.cols[1], rhs.cols[1]),
        binary_add(lhs.cols[2], rhs.cols[2]));
}
inline {ret}float3x3 binary_sub({lhs}float3x3 lhs, {rhs}float3x3 rhs) {
    return make_float3x3(
        binary_sub(lhs.cols[0], rhs.cols[0]),
        binary_sub(lhs.cols[1], rhs.cols[1]),
        binary_sub(lhs.cols[2], rhs.cols[2]));
}
inline {ret}float4x4 binary_add({lhs}float4x4 lhs, {rhs}float4x4 rhs) {
    return make_float4x4(
        binary_add(lhs.cols[0], rhs.cols[0]),
        binary_add(lhs.cols[1], rhs.cols[1]),
        binary_add(lhs.cols[2], rhs.cols[2]),
        binary_add(lhs.cols[3], rhs.cols[3]));
}
inline {ret}float4x4 binary_sub({lhs}float4x4 lhs, {rhs}float4x4 rhs) {
    return make_float4x4(
        binary_sub(lhs.cols[0], rhs.cols[0]),
        binary_sub(lhs.cols[1], rhs.cols[1]),
        binary_sub(lhs.cols[2], rhs.cols[2]),
        binary_sub(lhs.cols[3], rhs.cols[3]));
}
inline {ret}float2x2 binary_mul({lhs}float2x2 lhs, {rhs}float2x2 rhs) {
    return make_float2x2(
        binary_mul(lhs, rhs.cols[0]),
        binary_mul(lhs, rhs.cols[1]));
}
inline {ret}float3x3 binary_mul({lhs}float3x3 lhs, {rhs}float3x3 rhs) {
    return make_float3x3(
        binary_mul(lhs, rhs.cols[0]),
        binary_mul(lhs, rhs.cols[1]),
        binary_mul(lhs, rhs.cols[2]));
}
inline {ret}float4x4 binary_mul({lhs}float4x4 lhs, {rhs}float4x4 rhs) {
    return make_float4x4(
        binary_mul(lhs, rhs.cols[0]),
        binary_mul(lhs, rhs.cols[1]),
        binary_mul(lhs, rhs.cols[2]),
        binary_mul(lhs, rhs.cols[3]));
}'''
        print(template.replace("{ret}", "").replace("{lhs}", "").replace("{rhs}", ""), file=file)
        print(template.replace("{ret}", "").replace("{lhs}", "uniform ").replace("{rhs}", ""), file=file)
        print(template.replace("{ret}", "").replace("{lhs}", "").replace("{rhs}", "uniform "), file=file)
        print(template.replace("{ret}", "uniform ").replace("{lhs}", "uniform ").replace("{rhs}", "uniform "), file=file)

        template = '''
// transpose
inline uniform float2x2 transpose(uniform float2x2 m) {
    return make_float2x2(
        make_float2(m.cols[0].x, m.cols[1].x),
        make_float2(m.cols[0].y, m.cols[1].y));
}
inline uniform float3x3 transpose(uniform float3x3 m) {
    return make_float3x3(
        make_float3(m.cols[0].x, m.cols[1].x, m.cols[2].x),
        make_float3(m.cols[0].y, m.cols[1].y, m.cols[2].y),
        make_float3(m.cols[0].z, m.cols[1].z, m.cols[2].z));
}
inline uniform float4x4 transpose(uniform float4x4 m) {
    return make_float4x4(
        make_float4(m.cols[0].x, m.cols[1].x, m.cols[2].x, m.cols[3].x),
        make_float4(m.cols[0].y, m.cols[1].y, m.cols[2].y, m.cols[3].y),
        make_float4(m.cols[0].z, m.cols[1].z, m.cols[2].z, m.cols[3].z),
        make_float4(m.cols[0].w, m.cols[1].w, m.cols[2].w, m.cols[3].w));
}

// determinant
inline uniform float determinant(uniform float2x2 m) {
    return m.cols[0].x * m.cols[1].y - m.cols[1].x * m.cols[0].y;
}
inline uniform float determinant(uniform float3x3 m) {
    return m.cols[0].x * (m.cols[1].y * m.cols[2].z - m.cols[2].y * m.cols[1].z)
         - m.cols[1].x * (m.cols[0].y * m.cols[2].z - m.cols[2].y * m.cols[0].z)
         + m.cols[2].x * (m.cols[0].y * m.cols[1].z - m.cols[1].y * m.cols[0].z);
}
inline uniform float determinant(uniform float4x4 m) {
    const uniform float coef00 = m.cols[2].z * m.cols[3].w - m.cols[3].z * m.cols[2].w;
    const uniform float coef02 = m.cols[1].z * m.cols[3].w - m.cols[3].z * m.cols[1].w;
    const uniform float coef03 = m.cols[1].z * m.cols[2].w - m.cols[2].z * m.cols[1].w;
    const uniform float coef04 = m.cols[2].y * m.cols[3].w - m.cols[3].y * m.cols[2].w;
    const uniform float coef06 = m.cols[1].y * m.cols[3].w - m.cols[3].y * m.cols[1].w;
    const uniform float coef07 = m.cols[1].y * m.cols[2].w - m.cols[2].y * m.cols[1].w;
    const uniform float coef08 = m.cols[2].y * m.cols[3].z - m.cols[3].y * m.cols[2].z;
    const uniform float coef10 = m.cols[1].y * m.cols[3].z - m.cols[3].y * m.cols[1].z;
    const uniform float coef11 = m.cols[1].y * m.cols[2].z - m.cols[2].y * m.cols[1].z;
    const uniform float coef12 = m.cols[2].x * m.cols[3].w - m.cols[3].x * m.cols[2].w;
    const uniform float coef14 = m.cols[1].x * m.cols[3].w - m.cols[3].x * m.cols[1].w;
    const uniform float coef15 = m.cols[1].x * m.cols[2].w - m.cols[2].x * m.cols[1].w;
    const uniform float coef16 = m.cols[2].x * m.cols[3].z - m.cols[3].x * m.cols[2].z;
    const uniform float coef18 = m.cols[1].x * m.cols[3].z - m.cols[3].x * m.cols[1].z;
    const uniform float coef19 = m.cols[1].x * m.cols[2].z - m.cols[2].x * m.cols[1].z;
    const uniform float coef20 = m.cols[2].x * m.cols[3].y - m.cols[3].x * m.cols[2].y;
    const uniform float coef22 = m.cols[1].x * m.cols[3].y - m.cols[3].x * m.cols[1].y;
    const uniform float coef23 = m.cols[1].x * m.cols[2].y - m.cols[2].x * m.cols[1].y;
    const uniform float4 fac0 = make_float4(coef00, coef00, coef02, coef03);
    const uniform float4 fac1 = make_float4(coef04, coef04, coef06, coef07);
    const uniform float4 fac2 = make_float4(coef08, coef08, coef10, coef11);
    const uniform float4 fac3 = make_float4(coef12, coef12, coef14, coef15);
    const uniform float4 fac4 = make_float4(coef16, coef16, coef18, coef19);
    const uniform float4 fac5 = make_float4(coef20, coef20, coef22, coef23);
    const uniform float4 Vec0 = make_float4(m.cols[1].x, m.cols[0].x, m.cols[0].x, m.cols[0].x);
    const uniform float4 Vec1 = make_float4(m.cols[1].y, m.cols[0].y, m.cols[0].y, m.cols[0].y);
    const uniform float4 Vec2 = make_float4(m.cols[1].z, m.cols[0].z, m.cols[0].z, m.cols[0].z);
    const uniform float4 Vec3 = make_float4(m.cols[1].w, m.cols[0].w, m.cols[0].w, m.cols[0].w);
    const uniform float4 inv0 = binary_add(binary_sub(binary_mul(Vec1, fac0), binary_mul(Vec2, fac1)), binary_mul(Vec3, fac2));
    const uniform float4 inv1 = binary_add(binary_sub(binary_mul(Vec0, fac0), binary_mul(Vec2, fac3)), binary_mul(Vec3, fac4));
    const uniform float4 inv2 = binary_add(binary_sub(binary_mul(Vec0, fac1), binary_mul(Vec1, fac3)), binary_mul(Vec3, fac5));
    const uniform float4 inv3 = binary_add(binary_sub(binary_mul(Vec0, fac2), binary_mul(Vec1, fac4)), binary_mul(Vec2, fac5));
    const uniform float4 sign_a = make_float4(+1.0f, -1.0f, +1.0f, -1.0f);
    const uniform float4 sign_b = make_float4(-1.0f, +1.0f, -1.0f, +1.0f);
    const uniform float4 inv_0 = binary_mul(inv0, sign_a);
    const uniform float4 inv_1 = binary_mul(inv1, sign_b);
    const uniform float4 inv_2 = binary_mul(inv2, sign_a);
    const uniform float4 inv_3 = binary_mul(inv3, sign_b);
    const uniform float4 dot0 = binary_mul(m.cols[0], make_float4(inv_0.x, inv_1.x, inv_2.x, inv_3.x));
    return dot0.x + dot0.y + dot0.z + dot0.w;
}

// inverse
inline uniform float2x2 inverse(uniform float2x2 m) {
    const uniform float one_over_determinant = 1.f / determinant(m);
    return make_float2x2(m.cols[1].y * one_over_determinant,
                        -m.cols[0].y * one_over_determinant,
                        -m.cols[1].x * one_over_determinant,
                        +m.cols[0].x * one_over_determinant);
}
inline uniform float3x3 inverse(uniform float3x3 m) {
    const uniform float one_over_determinant = 1.f / determinant(m);
    return make_float3x3(
        (m.cols[1].y * m.cols[2].z - m.cols[2].y * m.cols[1].z) * one_over_determinant,
        (m.cols[2].y * m.cols[0].z - m.cols[0].y * m.cols[2].z) * one_over_determinant,
        (m.cols[0].y * m.cols[1].z - m.cols[1].y * m.cols[0].z) * one_over_determinant,
        (m.cols[2].x * m.cols[1].z - m.cols[1].x * m.cols[2].z) * one_over_determinant,
        (m.cols[0].x * m.cols[2].z - m.cols[2].x * m.cols[0].z) * one_over_determinant,
        (m.cols[1].x * m.cols[0].z - m.cols[0].x * m.cols[1].z) * one_over_determinant,
        (m.cols[1].x * m.cols[2].y - m.cols[2].x * m.cols[1].y) * one_over_determinant,
        (m.cols[2].x * m.cols[0].y - m.cols[0].x * m.cols[2].y) * one_over_determinant,
        (m.cols[0].x * m.cols[1].y - m.cols[1].x * m.cols[0].y) * one_over_determinant);
}
inline uniform float4x4 inverse(uniform float4x4 m) {
    const uniform float coef00 = m.cols[2].z * m.cols[3].w - m.cols[3].z * m.cols[2].w;
    const uniform float coef02 = m.cols[1].z * m.cols[3].w - m.cols[3].z * m.cols[1].w;
    const uniform float coef03 = m.cols[1].z * m.cols[2].w - m.cols[2].z * m.cols[1].w;
    const uniform float coef04 = m.cols[2].y * m.cols[3].w - m.cols[3].y * m.cols[2].w;
    const uniform float coef06 = m.cols[1].y * m.cols[3].w - m.cols[3].y * m.cols[1].w;
    const uniform float coef07 = m.cols[1].y * m.cols[2].w - m.cols[2].y * m.cols[1].w;
    const uniform float coef08 = m.cols[2].y * m.cols[3].z - m.cols[3].y * m.cols[2].z;
    const uniform float coef10 = m.cols[1].y * m.cols[3].z - m.cols[3].y * m.cols[1].z;
    const uniform float coef11 = m.cols[1].y * m.cols[2].z - m.cols[2].y * m.cols[1].z;
    const uniform float coef12 = m.cols[2].x * m.cols[3].w - m.cols[3].x * m.cols[2].w;
    const uniform float coef14 = m.cols[1].x * m.cols[3].w - m.cols[3].x * m.cols[1].w;
    const uniform float coef15 = m.cols[1].x * m.cols[2].w - m.cols[2].x * m.cols[1].w;
    const uniform float coef16 = m.cols[2].x * m.cols[3].z - m.cols[3].x * m.cols[2].z;
    const uniform float coef18 = m.cols[1].x * m.cols[3].z - m.cols[3].x * m.cols[1].z;
    const uniform float coef19 = m.cols[1].x * m.cols[2].z - m.cols[2].x * m.cols[1].z;
    const uniform float coef20 = m.cols[2].x * m.cols[3].y - m.cols[3].x * m.cols[2].y;
    const uniform float coef22 = m.cols[1].x * m.cols[3].y - m.cols[3].x * m.cols[1].y;
    const uniform float coef23 = m.cols[1].x * m.cols[2].y - m.cols[2].x * m.cols[1].y;
    const uniform float4 fac0 = make_float4(coef00, coef00, coef02, coef03);
    const uniform float4 fac1 = make_float4(coef04, coef04, coef06, coef07);
    const uniform float4 fac2 = make_float4(coef08, coef08, coef10, coef11);
    const uniform float4 fac3 = make_float4(coef12, coef12, coef14, coef15);
    const uniform float4 fac4 = make_float4(coef16, coef16, coef18, coef19);
    const uniform float4 fac5 = make_float4(coef20, coef20, coef22, coef23);
    const uniform float4 Vec0 = make_float4(m.cols[1].x, m.cols[0].x, m.cols[0].x, m.cols[0].x);
    const uniform float4 Vec1 = make_float4(m.cols[1].y, m.cols[0].y, m.cols[0].y, m.cols[0].y);
    const uniform float4 Vec2 = make_float4(m.cols[1].z, m.cols[0].z, m.cols[0].z, m.cols[0].z);
    const uniform float4 Vec3 = make_float4(m.cols[1].w, m.cols[0].w, m.cols[0].w, m.cols[0].w);
    const uniform float4 inv0 = binary_add(binary_sub(binary_mul(Vec1, fac0), binary_mul(Vec2, fac1)), binary_mul(Vec3, fac2));
    const uniform float4 inv1 = binary_add(binary_sub(binary_mul(Vec0, fac0), binary_mul(Vec2, fac3)), binary_mul(Vec3, fac4));
    const uniform float4 inv2 = binary_add(binary_sub(binary_mul(Vec0, fac1), binary_mul(Vec1, fac3)), binary_mul(Vec3, fac5));
    const uniform float4 inv3 = binary_add(binary_sub(binary_mul(Vec0, fac2), binary_mul(Vec1, fac4)), binary_mul(Vec2, fac5));
    const uniform float4 sign_a = make_float4(+1.0f, -1.0f, +1.0f, -1.0f);
    const uniform float4 sign_b = make_float4(-1.0f, +1.0f, -1.0f, +1.0f);
    const uniform float4 inv_0 = binary_mul(inv0, sign_a);
    const uniform float4 inv_1 = binary_mul(inv1, sign_b);
    const uniform float4 inv_2 = binary_mul(inv2, sign_a);
    const uniform float4 inv_3 = binary_mul(inv3, sign_b);
    const uniform float4 dot0 = binary_mul(m.cols[0], make_float4(inv_0.x, inv_1.x, inv_2.x, inv_3.x));
    const uniform float dot1 = dot0.x + dot0.y + dot0.z + dot0.w;
    const uniform float one_over_determinant = 1.0f / dot1;
    return make_float4x4(binary_mul(inv_0, one_over_determinant),
                         binary_mul(inv_1, one_over_determinant),
                         binary_mul(inv_2, one_over_determinant),
                         binary_mul(inv_3, one_over_determinant));
}'''
        print(template, file=file)
        print(template.replace("uniform ", ""), file=file)
