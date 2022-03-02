from os.path import realpath, dirname
from os import makedirs

if __name__ == "__main__":
    curr_dir = dirname(realpath(__file__))
    library_name = "ispc_device_library"
    support_dir = f"{curr_dir}/ispc_support"
    makedirs(support_dir, exist_ok=True)
    with open(f"{support_dir}/{library_name}.isph", "w") as file:

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

#define _x v[0]
#define _y v[1]
#define _z v[2]
#define _w v[3]

#define LUISA_INLINE static

typedef uint8 char;
''', file=file)
        # vector types
        scalar_types = ["int", "uint", "float", "char"]
        for t in scalar_types:
            print(f'''
struct {t}2 {{ {t} v[2]; }};
struct {t}3 {{ {t} v[4]; }};
struct {t}4 {{ {t} v[4]; }};''', file=file)

        template = '''
LUISA_INLINE uniform {T}2 make_{T}2() {
    uniform {T}2 v;
    v.v[0] = v.v[1] = 0;
    return v;
}
LUISA_INLINE uniform {T}3 make_{T}3() {
    uniform {T}3 v;
    v.v[0] = v.v[1] = v.v[2] = 0;
    return v;
}
LUISA_INLINE uniform {T}4 make_{T}4() {
    uniform {T}4 v;
    v.v[0] = v.v[1] = v.v[2] = v.v[3] = 0;
    return v;
}'''
        for t in scalar_types:
            print(template.replace("{T}", t), file=file)
        # make_typeN
        template = '''
// make_{T}2 functions
LUISA_INLINE {U}2 make_{T}2({U} s) {
    {U}2 v;
    v.v[0] = v.v[1] = s;
    return v;
}
LUISA_INLINE {U}2 make_{T}2({U} x, {U} y) {
    {U}2 v;
    v.v[0] = x;
    v.v[1] = y;
    return v;
}
LUISA_INLINE {U}2 make_{T}2({U}2 v) {
    return v;
}
LUISA_INLINE {U}2 make_{T}2({U}3 v) {
    {U}2 u;
    u.v[0] = v.v[0];
    u.v[1] = v.v[1];
    return u;
}
LUISA_INLINE {U}2 make_{T}2({U}4 v) {
    {U}2 u;
    u.v[0] = v.v[0];
    u.v[1] = v.v[1];
    return u;
}

// make_{T}3 functions
LUISA_INLINE {U}3 make_{T}3({U} s) {
    {U}3 v;
    v.v[0] = v.v[1] = v.v[2] = s;
    return v;
}
LUISA_INLINE {U}3 make_{T}3({U} x, {U} y, {U} z) {
    {U}3 v;
    v.v[0] = x;
    v.v[1] = y;
    v.v[2] = z;
    return v;
}
LUISA_INLINE {U}3 make_{T}3({U} x, {U}2 yz) {
    {U}3 v;
    v.v[0] = x;
    v.v[1] = yz.v[0];
    v.v[2] = yz.v[1];
    return v;
}
LUISA_INLINE {U}3 make_{T}3({U}2 xy, {U} z) {
    {U}3 v;
    v.v[0] = xy.v[0];
    v.v[1] = xy.v[1];
    v.v[2] = z;
    return v;
}
LUISA_INLINE {U}3 make_{T}3({U}3 v) {
    return v;
}
LUISA_INLINE {U}3 make_{T}3({U}4 v) {
    {U}3 u;
    u.v[0] = v.v[0];
    u.v[1] = v.v[1];
    u.v[2] = v.v[2];
    return u;
}

// make_{T}4 functions
LUISA_INLINE {U}4 make_{T}4({U} s) {
    {U}4 v;
    v.v[0] = v.v[1] = v.v[2] = v.v[3] = s;
    return v;
}
LUISA_INLINE {U}4 make_{T}4({U} x, {U} y, {U} z, {U} w) {
    {U}4 v;
    v.v[0] = x;
    v.v[1] = y;
    v.v[2] = z;
    v.v[3] = w;
    return v;
}
LUISA_INLINE {U}4 make_{T}4({U} x, {U} y, {U}2 zw) {
    {U}4 v;
    v.v[0] = x;
    v.v[1] = y;
    v.v[2] = zw.v[0];
    v.v[3] = zw.v[1];
    return v;
}
LUISA_INLINE {U}4 make_{T}4({U} x, {U}2 yz, {U} w) {
    {U}4 v;
    v.v[0] = x;
    v.v[1] = yz.v[0];
    v.v[2] = yz.v[1];
    v.v[3] = w;
    return v;
}
LUISA_INLINE {U}4 make_{T}4({U}2 xy, {U} z, {U} w) {
    {U}4 v;
    v.v[0] = xy.v[0];
    v.v[1] = xy.v[1];
    v.v[2] = z;
    v.v[3] = w;
    return v;
}
LUISA_INLINE {U}4 make_{T}4({U}2 xy, {U}2 zw) {
    {U}4 v;
    v.v[0] = xy.v[0];
    v.v[1] = xy.v[1];
    v.v[2] = zw.v[0];
    v.v[3] = zw.v[1];
    return v;
}
LUISA_INLINE {U}4 make_{T}4({U} x, {U}3 yzw) {
    {U}4 v;
    v.v[0] = x;
    v.v[1] = yzw.v[0];
    v.v[2] = yzw.v[1];
    v.v[3] = yzw.v[2];
    return v;
}
LUISA_INLINE {U}4 make_{T}4({U}3 xyz, {U} w) {
    {U}4 v;
    v.v[0] = xyz.v[0];
    v.v[1] = xyz.v[1];
    v.v[2] = xyz.v[2];
    v.v[3] = w;
    return v;
}
LUISA_INLINE {U}4 make_{T}4({U}4 v) {
    return v;
}'''
        for t in scalar_types:
            print(template.replace("{U}", t).replace("{T}", t), file=file)
            print(template.replace("{U}", f"uniform {t}").replace("{T}", t), file=file)
        template = '''
// conversions
LUISA_INLINE {U}2 make_{T}2({other}2 v) {
    {U}2 u;
    u.v[0] = ({U})v.v[0];
    u.v[1] = ({U})v.v[1];
    return u;
}
LUISA_INLINE {U}2 make_{T}2({other}3 v) {
    {U}2 u;
    u.v[0] = ({U})v.v[0];
    u.v[1] = ({U})v.v[1];
    return u;
}
LUISA_INLINE {U}2 make_{T}2({other}4 v) {
    {U}2 u;
    u.v[0] = ({U})v.v[0];
    u.v[1] = ({U})v.v[1];
    return u;
}
LUISA_INLINE {U}3 make_{T}3({other}3 v) {
    {U}3 u;
    u.v[0] = ({U})v.v[0];
    u.v[1] = ({U})v.v[1];
    u.v[2] = ({U})v.v[2];
    return u;
}
LUISA_INLINE {U}3 make_{T}3({other}4 v) {
    {U}3 u;
    u.v[0] = ({U})v.v[0];
    u.v[1] = ({U})v.v[1];
    u.v[2] = ({U})v.v[2];
    return u;
}
LUISA_INLINE {U}4 make_{T}4({other}4 v) {
    {U}4 u;
    u.v[0] = ({U})v.v[0];
    u.v[1] = ({U})v.v[1];
    u.v[2] = ({U})v.v[2];
    u.v[3] = ({U})v.v[3];
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
                f"static char unary_not({type} s) {{ return !s; }}",
                file=file)
            print(
                f"static uniform char unary_not(uniform {type} s) {{ return !s; }}",
                file=file)
            if type != "char":
                print(
                    f"static {type} unary_plus({type} s) {{ return +s; }}",
                    file=file)
                print(
                    f"static {type} unary_minus({type} s) {{ return -s; }}",
                    file=file)
                print(
                    f"static uniform {type} unary_plus(uniform {type} s) {{ return +s; }}",
                    file=file)
                print(
                    f"static uniform {type} unary_minus(uniform {type} s) {{ return -s; }}",
                    file=file)
                if type != "float":
                    print(
                        f"static {type} unary_bit_not({type} s) {{ return ~s; }}",
                        file=file)
                    print(
                        f"static uniform {type} unary_bit_not(uniform {type} s) {{ return ~s; }}",
                        file=file)
            for i in range(2, 5):
                elements = ["v[0]", "v[1]", "v[2]", "v[3]"][:i]
                print(
                    f"static char{i} unary_not({type}{i} v) {{ return make_char{i}({', '.join(f'!v.{m}' for m in elements)}); }}",
                    file=file)
                print(
                    f"static uniform char{i} unary_not(uniform {type}{i} v) {{ return make_char{i}({', '.join(f'!v.{m}' for m in elements)}); }}",
                    file=file)
                if type != "char":
                    print(
                        f"static {type}{i} unary_plus({type}{i} v) {{ return make_{type}{i}({', '.join(f'+v.{m}' for m in elements)}); }}",
                        file=file)
                    print(
                        f"static {type}{i} unary_minus({type}{i} v) {{ return make_{type}{i}({', '.join(f'-v.{m}' for m in elements)}); }}",
                        file=file)
                    print(
                        f"static uniform {type}{i} unary_plus(uniform {type}{i} v) {{ return make_{type}{i}({', '.join(f'+v.{m}' for m in elements)}); }}",
                        file=file)
                    print(
                        f"static uniform {type}{i} unary_minus(uniform {type}{i} v) {{ return make_{type}{i}({', '.join(f'-v.{m}' for m in elements)}); }}",
                        file=file)
                    if type != "float":
                        print(
                            f"static {type}{i} unary_bit_not({type}{i} v) {{ return make_{type}{i}({', '.join(f'~v.{m}' for m in elements)}); }}",
                            file=file)
                        print(
                            f"static {type}{i} unary_bit_not(uniform {type}{i} v) {{ return make_{type}{i}({', '.join(f'~v.{m}' for m in elements)}); }}",
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
                f"static uniform {ret_t} {op2name[op]}(uniform {arg_t} lhs, uniform {arg_t} rhs) {{ return lhs {op} rhs; }}",
                file=file)
            print(
                f"static {ret_t} {op2name[op]}({arg_t} lhs, uniform {arg_t} rhs) {{ return lhs {op} rhs; }}",
                file=file)
            print(
                f"static {ret_t} {op2name[op]}(uniform {arg_t} lhs, {arg_t} rhs) {{ return lhs {op} rhs; }}",
                file=file)
            print(
                f"static {ret_t} {op2name[op]}({arg_t} lhs, {arg_t} rhs) {{ return lhs {op} rhs; }}", file=file)
            for i in range(2, 5):
                elements = ["v[0]", "v[1]", "v[2]", "v[3]"][:i]
                # vector-vector
                print(
                    f"static uniform {ret_t}{i} {op2name[op]}(uniform {arg_t}{i} lhs, uniform {arg_t}{i} rhs) {{ return make_{ret_t}{i}({', '.join(f'lhs.{m} {op} rhs.{m}' for m in elements)}); }}",
                    file=file)
                print(
                    f"static {ret_t}{i} {op2name[op]}({arg_t}{i} lhs, uniform {arg_t}{i} rhs) {{ return make_{ret_t}{i}({', '.join(f'lhs.{m} {op} rhs.{m}' for m in elements)}); }}",
                    file=file)
                print(
                    f"static {ret_t}{i} {op2name[op]}(uniform {arg_t}{i} lhs, {arg_t}{i} rhs) {{ return make_{ret_t}{i}({', '.join(f'lhs.{m} {op} rhs.{m}' for m in elements)}); }}",
                    file=file)
                print(
                    f"static {ret_t}{i} {op2name[op]}({arg_t}{i} lhs, {arg_t}{i} rhs) {{ return make_{ret_t}{i}({', '.join(f'lhs.{m} {op} rhs.{m}' for m in elements)}); }}",
                    file=file)
                # vector-scalar
                operation = ", ".join(f"lhs.{e} {op} rhs" for e in ["v[0]", "v[1]", "v[2]", "v[3]"][:i])
                print(
                    f"static uniform {ret_t}{i} {op2name[op]}(uniform {arg_t}{i} lhs, uniform {arg_t} rhs) {{ return make_{ret_t}{i}({operation}); }}",
                    file=file)
                print(
                    f"static {ret_t}{i} {op2name[op]}({arg_t}{i} lhs, uniform {arg_t} rhs) {{ return make_{ret_t}{i}({operation}); }}",
                    file=file)
                print(
                    f"static {ret_t}{i} {op2name[op]}(uniform {arg_t}{i} lhs, {arg_t} rhs) {{ return make_{ret_t}{i}({operation}); }}",
                    file=file)
                print(
                    f"static {ret_t}{i} {op2name[op]}({arg_t}{i} lhs, {arg_t} rhs) {{ return make_{ret_t}{i}({operation}); }}",
                    file=file)
                # scalar-vector
                operation = ", ".join(f"lhs {op} rhs.{e}" for e in ["v[0]", "v[1]", "v[2]", "v[3]"][:i])
                print(
                    f"static uniform {ret_t}{i} {op2name[op]}(uniform {arg_t} lhs, uniform {arg_t}{i} rhs) {{ return make_{ret_t}{i}({operation}); }}",
                    file=file)
                print(
                    f"static {ret_t}{i} {op2name[op]}({arg_t} lhs, uniform {arg_t}{i} rhs) {{ return make_{ret_t}{i}({operation}); }}",
                    file=file)
                print(
                    f"static {ret_t}{i} {op2name[op]}(uniform {arg_t} lhs, {arg_t}{i} rhs) {{ return make_{ret_t}{i}({operation}); }}",
                    file=file)
                print(
                    f"static {ret_t}{i} {op2name[op]}({arg_t} lhs, {arg_t}{i} rhs) {{ return make_{ret_t}{i}({operation}); }}",
                    file=file)


        # binary operators
        for op in ["==", "!="]:
            for type in scalar_types:
                gen_binary_op(type, "char", op)
            print(file=file)
        for op in ["<", ">", "<=", ">="]:
            for type in ["int", "uint", "float"]:
                gen_binary_op(type, "char", op)
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
            for type in ["int", "uint", "char"]:
                gen_binary_op(type, type, op)
            print(file=file)
        for op in ["||", "&&"]:
            gen_binary_op("char", "char", op)
        print(file=file)
        # any, all, none
        for f, uop, bop in [("any", "", "||"), ("all", "", "&&"), ("none", "!", "&&")]:
            for i in range(2, 5):
                elements = ["v[0]", "v[1]", "v[2]", "v[3]"][:i]
                print(
                    f"static uniform char {f}(uniform char{i} v) {{ return {f' {bop} '.join(f'{uop}v.{m}' for m in elements)}; }}",
                    file=file)
                print(
                    f"static char {f}(char{i} v) {{ return {f' {bop} '.join(f'{uop}v.{m}' for m in elements)}; }}",
                    file=file)
        print(file=file)


        def generate_vector_call(name, c, types, args):
            types = [{"i": "int",
                      "u": "uint",
                      "f": "float",
                      "b": "char"}[t] for t in types]

            def call(i):
                e = ["v[0]", "v[1]", "v[2]", "v[3]"][i]
                return f"{c}(" + ", ".join(f"{a}.{e}" for a in args) + ")"

            for t in types:
                for n in range(2, 5):
                    ret_t = f"{t if name not in ['is_nan', 'is_inf'] else 'char'}{n}"
                    print(
                        f"static {ret_t} {name}({', '.join(f'{t}{n} {a}' for a in args)}) {{ return make_{ret_t}({', '.join(call(i) for i in range(n))}); }}",
                        file=file)
                    print(
                        f"static uniform {ret_t} {name}({', '.join(f'uniform {t}{n} {a}' for a in args)}) {{ return make_{ret_t}({', '.join(call(i) for i in range(n))}); }}",
                        file=file)
            print(file=file)


        # select
        print(
            "#define select_scalar(f, t, p) ((p) ? (t) : (f))",
            file=file)
        for t in ["int", "uint", "float"]:
            for n in range(2, 5):
                print(
                    f"static {t}{n} select({t}{n} f, {t}{n} t, char{n} p) {{ return make_{t}{n}({', '.join(f'select_scalar(f.{e}, t.{e}, p.{e})' for e in ['v[0]', 'v[1]', 'v[2]', 'v[3]'][:n])}); }}",
                    file=file)
                print(
                    f"static {t}{n} select({t}{n} f, {t}{n} t, uniform char{n} p) {{ return make_{t}{n}({', '.join(f'select_scalar(f.{e}, t.{e}, p.{e})' for e in ['v[0]', 'v[1]', 'v[2]', 'v[3]'][:n])}); }}",
                    file=file)
                print(
                    f"static uniform {t}{n} select(uniform {t}{n} f, uniform {t}{n} t, uniform char{n} p) {{ return make_{t}{n}({', '.join(f'select_scalar(f.{e}, t.{e}, p.{e})' for e in ['v[0]', 'v[1]', 'v[2]', 'v[3]'][:n])}); }}",
                    file=file)

        print('''
LUISA_INLINE float fma(float a, float b, float c) { return a * b + c; }
LUISA_INLINE uniform float fma(uniform float a, uniform float b, uniform float c) { return a * b + c; }
LUISA_INLINE float copysign(float a, float b) { return floatbits((intbits(a) & 0x7fffffffu) | signbits(b)); }
LUISA_INLINE uniform float copysign(uniform float a, uniform float b) { return floatbits((intbits(a) & 0x7fffffffu) | signbits(b)); }
LUISA_INLINE float log2(float x) { return log(x) / log(2.f); }
LUISA_INLINE uniform float log2(uniform float x) { return log(x) / log(2.f); }
LUISA_INLINE float log10(float x) { return log(x) / log(10.f); }
LUISA_INLINE uniform float log10(uniform float x) { return log(x) / log(10.f); }
LUISA_INLINE float exp2(float x) { return pow(2.f, x); }
LUISA_INLINE uniform float exp2(uniform float x) { return pow(2.f, x); }
LUISA_INLINE float exp10(float x) { return pow(10.f, x); }
LUISA_INLINE uniform float exp10(uniform float x) { return pow(10.f, x); }
LUISA_INLINE char is_nan(float x) {
  uint u = intbits(x);
  return (u & 0x7F800000u) == 0x7F800000u && (u & 0x7FFFFFu);
}
LUISA_INLINE char is_inf(float x) {
  uint u = intbits(x);
  return (u & 0x7F800000u) == 0x7F800000u && !(u & 0x7FFFFFu);
}
LUISA_INLINE uniform char is_nan(uniform float x) {
  uniform uint u = intbits(x);
  return (u & 0x7F800000u) == 0x7F800000u && (u & 0x7FFFFFu);
}
LUISA_INLINE uniform char is_inf(uniform float x) {
  uniform uint u = intbits(x);
  return (u & 0x7F800000u) == 0x7F800000u && !(u & 0x7FFFFFu);
}
LUISA_INLINE float sinh(float x) { return .5f * (exp(x) - exp(-x)); }
LUISA_INLINE uniform float sinh(uniform float x) { return .5f * (exp(x) - exp(-x)); }
LUISA_INLINE float cosh(float x) { return .5f * (exp(x) + exp(-x)); }
LUISA_INLINE uniform float cosh(uniform float x) { return .5f * (exp(x) + exp(-x)); }
LUISA_INLINE float tanh(float x) { return sinh(x) / cosh(x); }
LUISA_INLINE uniform float tanh(uniform float x) { return sinh(x) / cosh(x); }
LUISA_INLINE float asinh(float x) { return log(x + sqrt(x * x + 1.f)); }
LUISA_INLINE uniform float asinh(uniform float x) { return log(x + sqrt(x * x + 1.f)); }
LUISA_INLINE float acosh(float x) { return log(x + sqrt(x * x - 1.f)); }
LUISA_INLINE uniform float acosh(uniform float x) { return log(x + sqrt(x * x - 1.f)); }
LUISA_INLINE float atanh(float x) { return .5f * log((1.f + x) / (1.f - x)); }
LUISA_INLINE uniform float atanh(uniform float x) { return .5f * log((1.f + x) / (1.f - x)); }
LUISA_INLINE float saturate(float x) { return clamp(x, 0.f, 1.f); }
LUISA_INLINE uniform float saturate(uniform float x) { return clamp(x, 0.f, 1.f); }
LUISA_INLINE float lerp(float a, float b, float t) { return fma(t, b - a, a); }
LUISA_INLINE uniform float lerp(uniform float a, uniform float b, uniform float t) { return fma(t, b - a, a); }
LUISA_INLINE float degrees(float x) { return x * (180.f * M_1_PI); }
LUISA_INLINE uniform float degrees(uniform float x) { return x * (180.f * M_1_PI); }
LUISA_INLINE float radians(float x) { return x * (M_PI / 180.f); }
LUISA_INLINE uniform float radians(uniform float x) { return x * (M_PI / 180.f); }
LUISA_INLINE float step(float edge, float x) { return x < edge ? 0.0f : 1.0f; }
LUISA_INLINE uniform float step(uniform float edge, uniform float x) { return x < edge ? 0.0f : 1.0f; }
LUISA_INLINE float smoothstep(float edge0, float edge1, float x) {
    float t = saturate((x - edge0) / (edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}
LUISA_INLINE uniform float smoothstep(uniform float edge0, uniform float edge1, uniform float x) {
    uniform float t = saturate((x - edge0) / (edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}
LUISA_INLINE uint ctz(uint x) { return count_trailing_zeros((int)x); }
LUISA_INLINE uniform uint ctz(uniform uint x) { return count_trailing_zeros((int)x); }
LUISA_INLINE uint clz(uint x) { return count_leading_zeros((int)x); }
LUISA_INLINE uniform uint clz(uniform uint x) { return count_leading_zeros((int)x); }
LUISA_INLINE uint popcount(uint x) { return popcnt((int)x); }
LUISA_INLINE uniform uint popcount(uniform uint x) { return popcnt((int)x); }
LUISA_INLINE uniform uint reverse(uniform uint n) {
    n = (n << 16u) | (n >> 16u);
    n = ((n & 0x00ff00ffu) << 8u) | ((n & 0xff00ff00u) >> 8u);
    n = ((n & 0x0f0f0f0fu) << 4u) | ((n & 0xf0f0f0f0u) >> 4u);
    n = ((n & 0x33333333u) << 2u) | ((n & 0xccccccccu) >> 2u);
    n = ((n & 0x55555555u) << 1u) | ((n & 0xaaaaaaaau) >> 1u);
    return n;
}
LUISA_INLINE uint reverse(uint n) {
    n = (n << 16u) | (n >> 16u);
    n = ((n & 0x00ff00ffu) << 8u) | ((n & 0xff00ff00u) >> 8u);
    n = ((n & 0x0f0f0f0fu) << 4u) | ((n & 0xf0f0f0f0u) >> 4u);
    n = ((n & 0x33333333u) << 2u) | ((n & 0xccccccccu) >> 2u);
    n = ((n & 0x55555555u) << 1u) | ((n & 0xaaaaaaaau) >> 1u);
    return n;
}
LUISA_INLINE uniform float fract(uniform float x) { return x - floor(x); }
LUISA_INLINE float fract(float x) { return x - floor(x); }
LUISA_INLINE float3 cross(float3 u, float3 v) {
    return make_float3(
        u.v[1] * v.v[2] - v.v[1] * u.v[2],
        u.v[2] * v.v[0] - v.v[2] * u.v[0],
        u.v[0] * v.v[1] - v.v[0] * u.v[1]);
}
LUISA_INLINE uniform float3 cross(uniform float3 u, uniform float3 v) {
    return make_float3(
        u.v[1] * v.v[2] - v.v[1] * u.v[2],
        u.v[2] * v.v[0] - v.v[2] * u.v[0],
        u.v[0] * v.v[1] - v.v[0] * u.v[1]);
}
LUISA_INLINE float dot(float2 a, float2 b) { return a.v[0] * b.v[0] + a.v[1] * b.v[1]; }
LUISA_INLINE float dot(float3 a, float3 b) { return a.v[0] * b.v[0] + a.v[1] * b.v[1] + a.v[2] * b.v[2]; }
LUISA_INLINE float dot(float4 a, float4 b) { return a.v[0] * b.v[0] + a.v[1] * b.v[1] + a.v[2] * b.v[2] + a.v[3] * b.v[3]; }
LUISA_INLINE uniform float dot(uniform float2 a, uniform float2 b) { return a.v[0] * b.v[0] + a.v[1] * b.v[1]; }
LUISA_INLINE uniform float dot(uniform float3 a, uniform float3 b) { return a.v[0] * b.v[0] + a.v[1] * b.v[1] + a.v[2] * b.v[2]; }
LUISA_INLINE uniform float dot(uniform float4 a, uniform float4 b) { return a.v[0] * b.v[0] + a.v[1] * b.v[1] + a.v[2] * b.v[2] + a.v[3] * b.v[3]; }
LUISA_INLINE float length(float2 v) { return sqrt(dot(v, v)); }
LUISA_INLINE float length(float3 v) { return sqrt(dot(v, v)); }
LUISA_INLINE float length(float4 v) { return sqrt(dot(v, v)); }
LUISA_INLINE uniform float length(uniform float2 v) { return sqrt(dot(v, v)); }
LUISA_INLINE uniform float length(uniform float3 v) { return sqrt(dot(v, v)); }
LUISA_INLINE uniform float length(uniform float4 v) { return sqrt(dot(v, v)); }
LUISA_INLINE float length_squared(float2 v) { return dot(v, v); }
LUISA_INLINE float length_squared(float3 v) { return dot(v, v); }
LUISA_INLINE float length_squared(float4 v) { return dot(v, v); }
LUISA_INLINE uniform float length_squared(uniform float2 v) { return dot(v, v); }
LUISA_INLINE uniform float length_squared(uniform float3 v) { return dot(v, v); }
LUISA_INLINE uniform float length_squared(uniform float4 v) { return dot(v, v); }
LUISA_INLINE float2 normalize(float2 v) { return binary_mul(v, rsqrt(length_squared(v))); }
LUISA_INLINE float3 normalize(float3 v) { return binary_mul(v, rsqrt(length_squared(v))); }
LUISA_INLINE float4 normalize(float4 v) { return binary_mul(v, rsqrt(length_squared(v))); }
LUISA_INLINE uniform float2 normalize(uniform float2 v) { return binary_mul(v, rsqrt(length_squared(v))); }
LUISA_INLINE uniform float3 normalize(uniform float3 v) { return binary_mul(v, rsqrt(length_squared(v))); }
LUISA_INLINE uniform float4 normalize(uniform float4 v) { return binary_mul(v, rsqrt(length_squared(v))); }
LUISA_INLINE float3 faceforward(float3 n, float3 i, float3 n_ref) { return dot(n_ref, i) < 0.f ? n : unary_minus(n); }
LUISA_INLINE uniform float3 faceforward(uniform float3 n, uniform float3 i, uniform float3 n_ref) { return dot(n_ref, i) < 0.f ? n : unary_minus(n); }
''', file=file)

        # min/max/abs/acos/asin/asinh/acosh/atan/atanh/atan2/
        # cos/cosh/sin/sinh/tan/tanh/exp/exp2/exp10/log/log2/
        # log10/sqrt/rsqrt/ceil/floor/trunc/round/fma/copysignf/
        # isinf/isnan/clamp/saturate/lerp/radians/degrees/step
        # smoothstep/clz/ctz/popcount/reverse/fract
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
        generate_vector_call("fract", "fract", "f", ["x"])

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

LUISA_INLINE uniform float2x2 make_float2x2() {
    uniform float2x2 m;
    m.cols[0] = make_float2();
    m.cols[1] = make_float2();
    return m;
}
LUISA_INLINE uniform float3x3 make_float3x3() {
    uniform float3x3 m;
    m.cols[0] = make_float3();
    m.cols[1] = make_float3();
    m.cols[2] = make_float3();
    return m;
}
LUISA_INLINE uniform float4x4 make_float4x4() {
    uniform float4x4 m;
    m.cols[0] = make_float4();
    m.cols[1] = make_float4();
    m.cols[2] = make_float4();
    m.cols[3] = make_float4();
    return m;
}''', file=file)
        template = '''
// make_float2x2 functions
LUISA_INLINE uniform float2x2 make_float2x2(uniform float s) {
    uniform float2x2 m;
    m.cols[0] = make_float2(s);
    m.cols[1] = make_float2(s);
    return m;
}
LUISA_INLINE uniform float2x2 make_float2x2(uniform float2 c0, uniform float2 c1) {
    uniform float2x2 m;
    m.cols[0] = c0;
    m.cols[1] = c1;
    return m;
}
LUISA_INLINE uniform float2x2 make_float2x2(
        uniform float m00, uniform float m01,
        uniform float m10, uniform float m11) {
    uniform float2x2 m;
    m.cols[0] = make_float2(m00, m01);
    m.cols[1] = make_float2(m10, m11);
    return m;
}
LUISA_INLINE uniform float2x2 make_float2x2(uniform float2x2 m) { return m; }

// make_float3x3 functions
LUISA_INLINE uniform float3x3 make_float3x3(uniform float s) {
    uniform float3x3 m;
    m.cols[0] = make_float3(s);
    m.cols[1] = make_float3(s);
    m.cols[2] = make_float3(s);
    return m;
}
LUISA_INLINE uniform float3x3 make_float3x3(uniform float3 c0, uniform float3 c1, uniform float3 c2) {
    uniform float3x3 m;
    m.cols[0] = c0;
    m.cols[1] = c1;
    m.cols[2] = c2;
    return m;
}
LUISA_INLINE uniform float3x3 make_float3x3(
        uniform float m00, uniform float m01, uniform float m02,
        uniform float m10, uniform float m11, uniform float m12,
        uniform float m20, uniform float m21, uniform float m22) {
    uniform float3x3 m;
    m.cols[0] = make_float3(m00, m01, m02);
    m.cols[1] = make_float3(m10, m11, m12);
    m.cols[2] = make_float3(m20, m21, m22);
    return m;
}
LUISA_INLINE uniform float3x3 make_float3x3(uniform float3x3 m) { return m; }

// make_float4x4 functions
LUISA_INLINE uniform float4x4 make_float4x4(uniform float s) {
    uniform float4x4 m;
    m.cols[0] = make_float4(s);
    m.cols[1] = make_float4(s);
    m.cols[2] = make_float4(s);
    m.cols[3] = make_float4(s);
    return m;
}
LUISA_INLINE uniform float4x4 make_float4x4(uniform float4 c0, uniform float4 c1, uniform float4 c2, uniform float4 c3) {
    uniform float4x4 m;
    m.cols[0] = c0;
    m.cols[1] = c1;
    m.cols[2] = c2;
    m.cols[3] = c3;
    return m;
}
LUISA_INLINE uniform float4x4 make_float4x4(
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
LUISA_INLINE uniform float4x4 make_float4x4(uniform float4x4 m) { return m; }

// conversions
LUISA_INLINE uniform float2x2 make_float2x2(uniform float3x3 m) {
    uniform float2x2 n;
    n.cols[0] = make_float2(m.cols[0]);
    n.cols[1] = make_float2(m.cols[1]);
    return n;
}
LUISA_INLINE uniform float2x2 make_float2x2(uniform float4x4 m) {
    uniform float2x2 n;
    n.cols[0] = make_float2(m.cols[0]);
    n.cols[1] = make_float2(m.cols[1]);
    return n;
}
LUISA_INLINE uniform float3x3 make_float3x3(uniform float2x2 m) {
    uniform float3x3 n;
    n.cols[0] = make_float3(m.cols[0], 0.f);
    n.cols[1] = make_float3(m.cols[1], 0.f);
    n.cols[2] = make_float3(0.f, 0.f, 1.f);
    return n;
}
LUISA_INLINE uniform float3x3 make_float3x3(uniform float4x4 m) {
    uniform float3x3 n;
    n.cols[0] = make_float3(m.cols[0]);
    n.cols[1] = make_float3(m.cols[1]);
    n.cols[2] = make_float3(m.cols[2]);
    return n;
}
LUISA_INLINE uniform float4x4 make_float4x4(uniform float2x2 m) {
    uniform float4x4 n;
    n.cols[0] = make_float4(m.cols[0], 0.f, 0.f);
    n.cols[1] = make_float4(m.cols[1], 0.f, 0.f);
    n.cols[2] = make_float4(0.f, 0.f, 1.f, 0.f);
    n.cols[3] = make_float4(0.f, 0.f, 0.f, 1.f);
    return n;
}
LUISA_INLINE uniform float4x4 make_float4x4(uniform float3x3 m) {
    uniform float4x4 n;
    n.cols[0] = make_float4(m.cols[0], 0.f);
    n.cols[1] = make_float4(m.cols[1], 0.f);
    n.cols[2] = make_float4(m.cols[2], 0.f);
    n.cols[3] = make_float4(0.f, 0.f, 0.f, 1.f);
    return n;
}

// unary operators
LUISA_INLINE uniform float2x2 unary_plus(uniform float2x2 m) { return m; }
LUISA_INLINE uniform float3x3 unary_plus(uniform float3x3 m) { return m; }
LUISA_INLINE uniform float4x4 unary_plus(uniform float4x4 m) { return m; }
LUISA_INLINE uniform float2x2 unary_minus(uniform float2x2 m) {
    return make_float2x2(
        unary_minus(m.cols[0]),
        unary_minus(m.cols[1]));
}
LUISA_INLINE uniform float3x3 unary_minus(uniform float3x3 m) {
    return make_float3x3(
        unary_minus(m.cols[0]),
        unary_minus(m.cols[1]),
        unary_minus(m.cols[2]));
}
LUISA_INLINE uniform float4x4 unary_minus(uniform float4x4 m) {
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
LUISA_INLINE {ret}float2x2 binary_add({lhs}float2x2 m, {rhs}float s) {
    return make_float2x2(
        binary_add(m.cols[0], s),
        binary_add(m.cols[1], s));
}
LUISA_INLINE {ret}float2x2 binary_add({lhs}float s, {rhs}float2x2 m) {
    return make_float2x2(
        binary_add(s, m.cols[0]),
        binary_add(s, m.cols[1]));
}
LUISA_INLINE {ret}float2x2 binary_sub({lhs}float2x2 m, {rhs}float s) {
    return make_float2x2(
        binary_sub(m.cols[0], s),
        binary_sub(m.cols[1], s));
}
LUISA_INLINE {ret}float2x2 binary_sub({lhs}float s, {rhs}float2x2 m) {
    return make_float2x2(
        binary_sub(s, m.cols[0]),
        binary_sub(s, m.cols[1]));
}
LUISA_INLINE {ret}float2x2 binary_mul({lhs}float2x2 m, {rhs}float s) {
    return make_float2x2(
        binary_mul(m.cols[0], s),
        binary_mul(m.cols[1], s));
}
LUISA_INLINE {ret}float2x2 binary_mul({lhs}float s, {rhs}float2x2 m) {
    return make_float2x2(
        binary_mul(s, m.cols[0]),
        binary_mul(s, m.cols[1]));
}
LUISA_INLINE {ret}float2x2 binary_div({lhs}float2x2 m, {rhs}float s) {
    return make_float2x2(
        binary_div(m.cols[0], s),
        binary_div(m.cols[1], s));
}
LUISA_INLINE {ret}float2x2 binary_div({lhs}float s, {rhs}float2x2 m) {
    return make_float2x2(
        binary_div(s, m.cols[0]),
        binary_div(s, m.cols[1]));
}
LUISA_INLINE {ret}float3x3 binary_add({lhs}float3x3 m, {rhs}float s) {
    return make_float3x3(
        binary_add(m.cols[0], s),
        binary_add(m.cols[1], s),
        binary_add(m.cols[2], s));
}
LUISA_INLINE {ret}float3x3 binary_add({lhs}float s, {rhs}float3x3 m) {
    return make_float3x3(
        binary_add(s, m.cols[0]),
        binary_add(s, m.cols[1]),
        binary_add(s, m.cols[2]));
}
LUISA_INLINE {ret}float3x3 binary_sub({lhs}float3x3 m, {rhs}float s) {
    return make_float3x3(
        binary_sub(m.cols[0], s),
        binary_sub(m.cols[1], s),
        binary_sub(m.cols[2], s));
}
LUISA_INLINE {ret}float3x3 binary_sub({lhs}float s, {rhs}float3x3 m) {
    return make_float3x3(
        binary_sub(s, m.cols[0]),
        binary_sub(s, m.cols[1]),
        binary_sub(s, m.cols[2]));
}
LUISA_INLINE {ret}float3x3 binary_mul({lhs}float3x3 m, {rhs}float s) {
    return make_float3x3(
        binary_mul(m.cols[0], s),
        binary_mul(m.cols[1], s),
        binary_mul(m.cols[2], s));
}
LUISA_INLINE {ret}float3x3 binary_mul({lhs}float s, {rhs}float3x3 m) {
    return make_float3x3(
        binary_mul(s, m.cols[0]),
        binary_mul(s, m.cols[1]),
        binary_mul(s, m.cols[2]));
}
LUISA_INLINE {ret}float3x3 binary_div({lhs}float3x3 m, {rhs}float s) {
    return make_float3x3(
        binary_div(m.cols[0], s),
        binary_div(m.cols[1], s),
        binary_div(m.cols[2], s));
}
LUISA_INLINE {ret}float3x3 binary_div({lhs}float s, {rhs}float3x3 m) {
    return make_float3x3(
        binary_div(s, m.cols[0]),
        binary_div(s, m.cols[1]),
        binary_div(s, m.cols[2]));
}
LUISA_INLINE {ret}float4x4 binary_add({lhs}float4x4 m, {rhs}float s) {
    return make_float4x4(
        binary_add(m.cols[0], s),
        binary_add(m.cols[1], s),
        binary_add(m.cols[2], s),
        binary_add(m.cols[3], s));
}
LUISA_INLINE {ret}float4x4 binary_add({lhs}float s, {rhs}float4x4 m) {
    return make_float4x4(
        binary_add(s, m.cols[0]),
        binary_add(s, m.cols[1]),
        binary_add(s, m.cols[2]),
        binary_add(s, m.cols[3]));
}
LUISA_INLINE {ret}float4x4 binary_sub({lhs}float4x4 m, {rhs}float s) {
    return make_float4x4(
        binary_sub(m.cols[0], s),
        binary_sub(m.cols[1], s),
        binary_sub(m.cols[2], s),
        binary_sub(m.cols[3], s));
}
LUISA_INLINE {ret}float4x4 binary_sub({lhs}float s, {rhs}float4x4 m) {
    return make_float4x4(
        binary_sub(s, m.cols[0]),
        binary_sub(s, m.cols[1]),
        binary_sub(s, m.cols[2]),
        binary_sub(s, m.cols[3]));
}
LUISA_INLINE {ret}float4x4 binary_mul({lhs}float4x4 m, {rhs}float s) {
    return make_float4x4(
        binary_mul(m.cols[0], s),
        binary_mul(m.cols[1], s),
        binary_mul(m.cols[2], s),
        binary_mul(m.cols[3], s));
}
LUISA_INLINE {ret}float4x4 binary_mul({lhs}float s, {rhs}float4x4 m) {
    return make_float4x4(
        binary_mul(s, m.cols[0]),
        binary_mul(s, m.cols[1]),
        binary_mul(s, m.cols[2]),
        binary_mul(s, m.cols[3]));
}
LUISA_INLINE {ret}float4x4 binary_div({lhs}float4x4 m, {rhs}float s) {
    return make_float4x4(
        binary_div(m.cols[0], s),
        binary_div(m.cols[1], s),
        binary_div(m.cols[2], s),
        binary_div(m.cols[3], s));
}
LUISA_INLINE {ret}float4x4 binary_div({lhs}float s, {rhs}float4x4 m) {
    return make_float4x4(
        binary_div(s, m.cols[0]),
        binary_div(s, m.cols[1]),
        binary_div(s, m.cols[2]),
        binary_div(s, m.cols[3]));
}

// matrix-vector binary operators
LUISA_INLINE {ret}float2 binary_mul({lhs}float2x2 m, {rhs}float2 v) {
    return binary_add(
        binary_mul(m.cols[0], v.v[0]),
        binary_mul(m.cols[1], v.v[1]));
}
LUISA_INLINE {ret}float3 binary_mul({lhs}float3x3 m, {rhs}float3 v) {
    return binary_add(
        binary_add(
            binary_mul(m.cols[0], v.v[0]),
            binary_mul(m.cols[1], v.v[1])),
        binary_mul(m.cols[2], v.v[2]));
}
LUISA_INLINE {ret}float4 binary_mul({lhs}float4x4 m, {rhs}float4 v) {
    return binary_add(
        binary_add(
            binary_mul(m.cols[0], v.v[0]),
            binary_mul(m.cols[1], v.v[1])),
        binary_add(
           binary_mul(m.cols[2], v.v[2]),
           binary_mul(m.cols[3], v.v[3])));
}

// matrix-matrix binary operators
LUISA_INLINE {ret}float2x2 binary_add({lhs}float2x2 lhs, {rhs}float2x2 rhs) {
    return make_float2x2(
        binary_add(lhs.cols[0], rhs.cols[0]),
        binary_add(lhs.cols[1], rhs.cols[1]));
}
LUISA_INLINE {ret}float2x2 binary_sub({lhs}float2x2 lhs, {rhs}float2x2 rhs) {
    return make_float2x2(
        binary_sub(lhs.cols[0], rhs.cols[0]),
        binary_sub(lhs.cols[1], rhs.cols[1]));
}
LUISA_INLINE {ret}float3x3 binary_add({lhs}float3x3 lhs, {rhs}float3x3 rhs) {
    return make_float3x3(
        binary_add(lhs.cols[0], rhs.cols[0]),
        binary_add(lhs.cols[1], rhs.cols[1]),
        binary_add(lhs.cols[2], rhs.cols[2]));
}
LUISA_INLINE {ret}float3x3 binary_sub({lhs}float3x3 lhs, {rhs}float3x3 rhs) {
    return make_float3x3(
        binary_sub(lhs.cols[0], rhs.cols[0]),
        binary_sub(lhs.cols[1], rhs.cols[1]),
        binary_sub(lhs.cols[2], rhs.cols[2]));
}
LUISA_INLINE {ret}float4x4 binary_add({lhs}float4x4 lhs, {rhs}float4x4 rhs) {
    return make_float4x4(
        binary_add(lhs.cols[0], rhs.cols[0]),
        binary_add(lhs.cols[1], rhs.cols[1]),
        binary_add(lhs.cols[2], rhs.cols[2]),
        binary_add(lhs.cols[3], rhs.cols[3]));
}
LUISA_INLINE {ret}float4x4 binary_sub({lhs}float4x4 lhs, {rhs}float4x4 rhs) {
    return make_float4x4(
        binary_sub(lhs.cols[0], rhs.cols[0]),
        binary_sub(lhs.cols[1], rhs.cols[1]),
        binary_sub(lhs.cols[2], rhs.cols[2]),
        binary_sub(lhs.cols[3], rhs.cols[3]));
}
LUISA_INLINE {ret}float2x2 binary_mul({lhs}float2x2 lhs, {rhs}float2x2 rhs) {
    return make_float2x2(
        binary_mul(lhs, rhs.cols[0]),
        binary_mul(lhs, rhs.cols[1]));
}
LUISA_INLINE {ret}float3x3 binary_mul({lhs}float3x3 lhs, {rhs}float3x3 rhs) {
    return make_float3x3(
        binary_mul(lhs, rhs.cols[0]),
        binary_mul(lhs, rhs.cols[1]),
        binary_mul(lhs, rhs.cols[2]));
}
LUISA_INLINE {ret}float4x4 binary_mul({lhs}float4x4 lhs, {rhs}float4x4 rhs) {
    return make_float4x4(
        binary_mul(lhs, rhs.cols[0]),
        binary_mul(lhs, rhs.cols[1]),
        binary_mul(lhs, rhs.cols[2]),
        binary_mul(lhs, rhs.cols[3]));
}'''
        print(template.replace("{ret}", "").replace("{lhs}", "").replace("{rhs}", ""), file=file)
        print(template.replace("{ret}", "").replace("{lhs}", "uniform ").replace("{rhs}", ""), file=file)
        print(template.replace("{ret}", "").replace("{lhs}", "").replace("{rhs}", "uniform "), file=file)
        print(template.replace("{ret}", "uniform ").replace("{lhs}", "uniform ").replace("{rhs}", "uniform "),
              file=file)

        template = '''
// transpose
LUISA_INLINE uniform float2x2 transpose(uniform float2x2 m) {
    return make_float2x2(
        make_float2(m.cols[0].v[0], m.cols[1].v[0]),
        make_float2(m.cols[0].v[1], m.cols[1].v[1]));
}
LUISA_INLINE uniform float3x3 transpose(uniform float3x3 m) {
    return make_float3x3(
        make_float3(m.cols[0].v[0], m.cols[1].v[0], m.cols[2].v[0]),
        make_float3(m.cols[0].v[1], m.cols[1].v[1], m.cols[2].v[1]),
        make_float3(m.cols[0].v[2], m.cols[1].v[2], m.cols[2].v[2]));
}
LUISA_INLINE uniform float4x4 transpose(uniform float4x4 m) {
    return make_float4x4(
        make_float4(m.cols[0].v[0], m.cols[1].v[0], m.cols[2].v[0], m.cols[3].v[0]),
        make_float4(m.cols[0].v[1], m.cols[1].v[1], m.cols[2].v[1], m.cols[3].v[1]),
        make_float4(m.cols[0].v[2], m.cols[1].v[2], m.cols[2].v[2], m.cols[3].v[2]),
        make_float4(m.cols[0].v[3], m.cols[1].v[3], m.cols[2].v[3], m.cols[3].v[3]));
}

// determinant
LUISA_INLINE uniform float determinant(uniform float2x2 m) {
    return m.cols[0].v[0] * m.cols[1].v[1] - m.cols[1].v[0] * m.cols[0].v[1];
}
LUISA_INLINE uniform float determinant(uniform float3x3 m) {
    return m.cols[0].v[0] * (m.cols[1].v[1] * m.cols[2].v[2] - m.cols[2].v[1] * m.cols[1].v[2])
         - m.cols[1].v[0] * (m.cols[0].v[1] * m.cols[2].v[2] - m.cols[2].v[1] * m.cols[0].v[2])
         + m.cols[2].v[0] * (m.cols[0].v[1] * m.cols[1].v[2] - m.cols[1].v[1] * m.cols[0].v[2]);
}
LUISA_INLINE uniform float determinant(uniform float4x4 m) {
    const uniform float coef00 = m.cols[2].v[2] * m.cols[3].v[3] - m.cols[3].v[2] * m.cols[2].v[3];
    const uniform float coef02 = m.cols[1].v[2] * m.cols[3].v[3] - m.cols[3].v[2] * m.cols[1].v[3];
    const uniform float coef03 = m.cols[1].v[2] * m.cols[2].v[3] - m.cols[2].v[2] * m.cols[1].v[3];
    const uniform float coef04 = m.cols[2].v[1] * m.cols[3].v[3] - m.cols[3].v[1] * m.cols[2].v[3];
    const uniform float coef06 = m.cols[1].v[1] * m.cols[3].v[3] - m.cols[3].v[1] * m.cols[1].v[3];
    const uniform float coef07 = m.cols[1].v[1] * m.cols[2].v[3] - m.cols[2].v[1] * m.cols[1].v[3];
    const uniform float coef08 = m.cols[2].v[1] * m.cols[3].v[2] - m.cols[3].v[1] * m.cols[2].v[2];
    const uniform float coef10 = m.cols[1].v[1] * m.cols[3].v[2] - m.cols[3].v[1] * m.cols[1].v[2];
    const uniform float coef11 = m.cols[1].v[1] * m.cols[2].v[2] - m.cols[2].v[1] * m.cols[1].v[2];
    const uniform float coef12 = m.cols[2].v[0] * m.cols[3].v[3] - m.cols[3].v[0] * m.cols[2].v[3];
    const uniform float coef14 = m.cols[1].v[0] * m.cols[3].v[3] - m.cols[3].v[0] * m.cols[1].v[3];
    const uniform float coef15 = m.cols[1].v[0] * m.cols[2].v[3] - m.cols[2].v[0] * m.cols[1].v[3];
    const uniform float coef16 = m.cols[2].v[0] * m.cols[3].v[2] - m.cols[3].v[0] * m.cols[2].v[2];
    const uniform float coef18 = m.cols[1].v[0] * m.cols[3].v[2] - m.cols[3].v[0] * m.cols[1].v[2];
    const uniform float coef19 = m.cols[1].v[0] * m.cols[2].v[2] - m.cols[2].v[0] * m.cols[1].v[2];
    const uniform float coef20 = m.cols[2].v[0] * m.cols[3].v[1] - m.cols[3].v[0] * m.cols[2].v[1];
    const uniform float coef22 = m.cols[1].v[0] * m.cols[3].v[1] - m.cols[3].v[0] * m.cols[1].v[1];
    const uniform float coef23 = m.cols[1].v[0] * m.cols[2].v[1] - m.cols[2].v[0] * m.cols[1].v[1];
    const uniform float4 fac0 = make_float4(coef00, coef00, coef02, coef03);
    const uniform float4 fac1 = make_float4(coef04, coef04, coef06, coef07);
    const uniform float4 fac2 = make_float4(coef08, coef08, coef10, coef11);
    const uniform float4 fac3 = make_float4(coef12, coef12, coef14, coef15);
    const uniform float4 fac4 = make_float4(coef16, coef16, coef18, coef19);
    const uniform float4 fac5 = make_float4(coef20, coef20, coef22, coef23);
    const uniform float4 Vec0 = make_float4(m.cols[1].v[0], m.cols[0].v[0], m.cols[0].v[0], m.cols[0].v[0]);
    const uniform float4 Vec1 = make_float4(m.cols[1].v[1], m.cols[0].v[1], m.cols[0].v[1], m.cols[0].v[1]);
    const uniform float4 Vec2 = make_float4(m.cols[1].v[2], m.cols[0].v[2], m.cols[0].v[2], m.cols[0].v[2]);
    const uniform float4 Vec3 = make_float4(m.cols[1].v[3], m.cols[0].v[3], m.cols[0].v[3], m.cols[0].v[3]);
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
    const uniform float4 dot0 = binary_mul(m.cols[0], make_float4(inv_0.v[0], inv_1.v[0], inv_2.v[0], inv_3.v[0]));
    return dot0.v[0] + dot0.v[1] + dot0.v[2] + dot0.v[3];
}

// inverse
LUISA_INLINE uniform float2x2 inverse(uniform float2x2 m) {
    const uniform float one_over_determinant = 1.f / determinant(m);
    return make_float2x2(m.cols[1].v[1] * one_over_determinant,
                        -m.cols[0].v[1] * one_over_determinant,
                        -m.cols[1].v[0] * one_over_determinant,
                        +m.cols[0].v[0] * one_over_determinant);
}
LUISA_INLINE uniform float3x3 inverse(uniform float3x3 m) {
    const uniform float one_over_determinant = 1.f / determinant(m);
    return make_float3x3(
        (m.cols[1].v[1] * m.cols[2].v[2] - m.cols[2].v[1] * m.cols[1].v[2]) * one_over_determinant,
        (m.cols[2].v[1] * m.cols[0].v[2] - m.cols[0].v[1] * m.cols[2].v[2]) * one_over_determinant,
        (m.cols[0].v[1] * m.cols[1].v[2] - m.cols[1].v[1] * m.cols[0].v[2]) * one_over_determinant,
        (m.cols[2].v[0] * m.cols[1].v[2] - m.cols[1].v[0] * m.cols[2].v[2]) * one_over_determinant,
        (m.cols[0].v[0] * m.cols[2].v[2] - m.cols[2].v[0] * m.cols[0].v[2]) * one_over_determinant,
        (m.cols[1].v[0] * m.cols[0].v[2] - m.cols[0].v[0] * m.cols[1].v[2]) * one_over_determinant,
        (m.cols[1].v[0] * m.cols[2].v[1] - m.cols[2].v[0] * m.cols[1].v[1]) * one_over_determinant,
        (m.cols[2].v[0] * m.cols[0].v[1] - m.cols[0].v[0] * m.cols[2].v[1]) * one_over_determinant,
        (m.cols[0].v[0] * m.cols[1].v[1] - m.cols[1].v[0] * m.cols[0].v[1]) * one_over_determinant);
}
LUISA_INLINE uniform float4x4 inverse(uniform float4x4 m) {
    const uniform float coef00 = m.cols[2].v[2] * m.cols[3].v[3] - m.cols[3].v[2] * m.cols[2].v[3];
    const uniform float coef02 = m.cols[1].v[2] * m.cols[3].v[3] - m.cols[3].v[2] * m.cols[1].v[3];
    const uniform float coef03 = m.cols[1].v[2] * m.cols[2].v[3] - m.cols[2].v[2] * m.cols[1].v[3];
    const uniform float coef04 = m.cols[2].v[1] * m.cols[3].v[3] - m.cols[3].v[1] * m.cols[2].v[3];
    const uniform float coef06 = m.cols[1].v[1] * m.cols[3].v[3] - m.cols[3].v[1] * m.cols[1].v[3];
    const uniform float coef07 = m.cols[1].v[1] * m.cols[2].v[3] - m.cols[2].v[1] * m.cols[1].v[3];
    const uniform float coef08 = m.cols[2].v[1] * m.cols[3].v[2] - m.cols[3].v[1] * m.cols[2].v[2];
    const uniform float coef10 = m.cols[1].v[1] * m.cols[3].v[2] - m.cols[3].v[1] * m.cols[1].v[2];
    const uniform float coef11 = m.cols[1].v[1] * m.cols[2].v[2] - m.cols[2].v[1] * m.cols[1].v[2];
    const uniform float coef12 = m.cols[2].v[0] * m.cols[3].v[3] - m.cols[3].v[0] * m.cols[2].v[3];
    const uniform float coef14 = m.cols[1].v[0] * m.cols[3].v[3] - m.cols[3].v[0] * m.cols[1].v[3];
    const uniform float coef15 = m.cols[1].v[0] * m.cols[2].v[3] - m.cols[2].v[0] * m.cols[1].v[3];
    const uniform float coef16 = m.cols[2].v[0] * m.cols[3].v[2] - m.cols[3].v[0] * m.cols[2].v[2];
    const uniform float coef18 = m.cols[1].v[0] * m.cols[3].v[2] - m.cols[3].v[0] * m.cols[1].v[2];
    const uniform float coef19 = m.cols[1].v[0] * m.cols[2].v[2] - m.cols[2].v[0] * m.cols[1].v[2];
    const uniform float coef20 = m.cols[2].v[0] * m.cols[3].v[1] - m.cols[3].v[0] * m.cols[2].v[1];
    const uniform float coef22 = m.cols[1].v[0] * m.cols[3].v[1] - m.cols[3].v[0] * m.cols[1].v[1];
    const uniform float coef23 = m.cols[1].v[0] * m.cols[2].v[1] - m.cols[2].v[0] * m.cols[1].v[1];
    const uniform float4 fac0 = make_float4(coef00, coef00, coef02, coef03);
    const uniform float4 fac1 = make_float4(coef04, coef04, coef06, coef07);
    const uniform float4 fac2 = make_float4(coef08, coef08, coef10, coef11);
    const uniform float4 fac3 = make_float4(coef12, coef12, coef14, coef15);
    const uniform float4 fac4 = make_float4(coef16, coef16, coef18, coef19);
    const uniform float4 fac5 = make_float4(coef20, coef20, coef22, coef23);
    const uniform float4 Vec0 = make_float4(m.cols[1].v[0], m.cols[0].v[0], m.cols[0].v[0], m.cols[0].v[0]);
    const uniform float4 Vec1 = make_float4(m.cols[1].v[1], m.cols[0].v[1], m.cols[0].v[1], m.cols[0].v[1]);
    const uniform float4 Vec2 = make_float4(m.cols[1].v[2], m.cols[0].v[2], m.cols[0].v[2], m.cols[0].v[2]);
    const uniform float4 Vec3 = make_float4(m.cols[1].v[3], m.cols[0].v[3], m.cols[0].v[3], m.cols[0].v[3]);
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
    const uniform float4 dot0 = binary_mul(m.cols[0], make_float4(inv_0.v[0], inv_1.v[0], inv_2.v[0], inv_3.v[0]));
    const uniform float dot1 = dot0.v[0] + dot0.v[1] + dot0.v[2] + dot0.v[3];
    const uniform float one_over_determinant = 1.0f / dot1;
    return make_float4x4(binary_mul(inv_0, one_over_determinant),
                         binary_mul(inv_1, one_over_determinant),
                         binary_mul(inv_2, one_over_determinant),
                         binary_mul(inv_3, one_over_determinant));
}'''
        print(template, file=file)
        print(template.replace("uniform ", ""), file=file)
        print('''
struct packed_float3 {
    float a[3];
};
struct LCRay {
    packed_float3 m0;
    float m1;
    packed_float3 m2;
    float m3;
};
struct LCHit {
    uint m0;
    uint m1;
    float2 m2;
};
struct LCAccel {
    void *uniform handle;
    const uniform float4x4 *uniform transforms;
};

LUISA_INLINE float4x4 accel_instance_transform(uniform LCAccel accel, uint index) {
    return accel.transforms[index];
}

// pixel format
enum PixelStorage {

    BYTE1,
    BYTE2,
    BYTE4,

    SHORT1,
    SHORT2,
    SHORT4,

    INT1,
    INT2,
    INT4,

    HALF1,
    HALF2,
    HALF4,

    FLOAT1,
    FLOAT2,
    FLOAT4
};

LUISA_INLINE uint pixel_storage_size(PixelStorage storage) {
    switch (storage) {
        case BYTE1: return sizeof(uniform uint8) * 1u;
        case BYTE2: return sizeof(uniform uint8) * 2u;
        case BYTE4: return sizeof(uniform uint8) * 4u;
        case SHORT1: return sizeof(uniform uint16) * 1u;
        case SHORT2: return sizeof(uniform uint16) * 2u;
        case SHORT4: return sizeof(uniform uint16) * 4u;
        case INT1: return sizeof(uniform int) * 1u;
        case INT2: return sizeof(uniform int) * 2u;
        case INT4: return sizeof(uniform int) * 4u;
        case HALF1: return sizeof(uniform int16) * 1u;
        case HALF2: return sizeof(uniform int16) * 2u;
        case HALF4: return sizeof(uniform int16) * 4u;
        case FLOAT1: return sizeof(uniform float) * 1u;
        case FLOAT2: return sizeof(uniform float) * 2u;
        case FLOAT4: return sizeof(uniform float) * 4u;
        default: break;
    }
    return 0u;
}

LUISA_INLINE uint pixel_storage_channel_count(PixelStorage storage) {
    switch (storage) {
        case BYTE1: return 1u;
        case BYTE2: return 2u;
        case BYTE4: return 4u;
        case SHORT1: return 1u;
        case SHORT2: return 2u;
        case SHORT4: return 4u;
        case INT1: return 1u;
        case INT2: return 2u;
        case INT4: return 4u;
        case HALF1: return 1u;
        case HALF2: return 2u;
        case HALF4: return 4u;
        case FLOAT1: return 1u;
        case FLOAT2: return 2u;
        case FLOAT4: return 4u;
        default: break;
    }
    return 0u;
}

LUISA_INLINE void pixel_write_float(PixelStorage storage, void* data, float4 value)
{
    switch (storage) {
        case BYTE1: for (int i=0; i<1; ++i) ((uint8*)data)[i] = (uint8)clamp(value.v[i]*255,0,255); break;
        case BYTE2: for (int i=0; i<2; ++i) ((uint8*)data)[i] = (uint8)clamp(value.v[i]*255,0,255); break;
        case BYTE4: for (int i=0; i<4; ++i) ((uint8*)data)[i] = (uint8)clamp(value.v[i]*255,0,255); break;
        case SHORT1: for (int i=0; i<1; ++i) ((uint16*)data)[i] = (uint16)clamp(value.v[i]*65535,0,65535); break;
        case SHORT2: for (int i=0; i<2; ++i) ((uint16*)data)[i] = (uint16)clamp(value.v[i]*65535,0,65535); break;
        case SHORT4: for (int i=0; i<4; ++i) ((uint16*)data)[i] = (uint16)clamp(value.v[i]*65535,0,65535); break;
        case HALF1: for (int i=0; i<1; ++i) ((int16*)data)[i] = float_to_half(value.v[i]); break;
        case HALF2: for (int i=0; i<2; ++i) ((int16*)data)[i] = float_to_half(value.v[i]); break;
        case HALF4: for (int i=0; i<4; ++i) ((int16*)data)[i] = float_to_half(value.v[i]); break;
        case FLOAT1: for (int i=0; i<1; ++i) ((float*)data)[i] = value.v[i]; break;
        case FLOAT2: for (int i=0; i<2; ++i) ((float*)data)[i] = value.v[i]; break;
        case FLOAT4: for (int i=0; i<4; ++i) ((float*)data)[i] = value.v[i]; break;
        default: break;
    }

}

LUISA_INLINE void pixel_write_int(PixelStorage storage, void* data, int4 value)
{
    switch (storage) {
        case BYTE1: for (int i=0; i<1; ++i) ((int8*)data)[i] = (int8)clamp(value.v[i],-128,127); break;
        case BYTE2: for (int i=0; i<2; ++i) ((int8*)data)[i] = (int8)clamp(value.v[i],-128,127); break;
        case BYTE4: for (int i=0; i<4; ++i) ((int8*)data)[i] = (int8)clamp(value.v[i],-128,127); break;
        case SHORT1: for (int i=0; i<1; ++i) ((int16*)data)[i] = (int16)clamp(value.v[i],-32768,32767); break;
        case SHORT2: for (int i=0; i<2; ++i) ((int16*)data)[i] = (int16)clamp(value.v[i],-32768,32767); break;
        case SHORT4: for (int i=0; i<4; ++i) ((int16*)data)[i] = (int16)clamp(value.v[i],-32768,32767); break;
        case INT1: for (int i=0; i<1; ++i) ((int*)data)[i] = (int)value.v[i]; break;
        case INT2: for (int i=0; i<2; ++i) ((int*)data)[i] = (int)value.v[i]; break;
        case INT4: for (int i=0; i<4; ++i) ((int*)data)[i] = (int)value.v[i]; break;
        default: break;
    }
}

LUISA_INLINE void pixel_write_uint(PixelStorage storage, void* data, uint4 value)
{
    switch (storage) {
        case BYTE1: for (int i=0; i<1; ++i) ((uint8*)data)[i] = (uint8)clamp(value.v[i],0,255); break;
        case BYTE2: for (int i=0; i<2; ++i) ((uint8*)data)[i] = (uint8)clamp(value.v[i],0,255); break;
        case BYTE4: for (int i=0; i<4; ++i) ((uint8*)data)[i] = (uint8)clamp(value.v[i],0,255); break;
        case SHORT1: for (int i=0; i<1; ++i) ((uint16*)data)[i] = (uint16)clamp(value.v[i],0,65535); break;
        case SHORT2: for (int i=0; i<2; ++i) ((uint16*)data)[i] = (uint16)clamp(value.v[i],0,65535); break;
        case SHORT4: for (int i=0; i<4; ++i) ((uint16*)data)[i] = (uint16)clamp(value.v[i],0,65535); break;
        case INT1: for (int i=0; i<1; ++i) ((uint*)data)[i] = (uint)value.v[i]; break;
        case INT2: for (int i=0; i<2; ++i) ((uint*)data)[i] = (uint)value.v[i]; break;
        case INT4: for (int i=0; i<4; ++i) ((uint*)data)[i] = (uint)value.v[i]; break;
        default: break;
    }
}

LUISA_INLINE float4 pixel_read_float(PixelStorage storage, void* data)
{
    float4 value = make_float4(0.0f);
    switch (storage) {
        case BYTE1: for (int i=0; i<1; ++i) value.v[i] = (1.0f/255) * ((uint8*)data)[i]; break;
        case BYTE2: for (int i=0; i<2; ++i) value.v[i] = (1.0f/255) * ((uint8*)data)[i]; break;
        case BYTE4: for (int i=0; i<4; ++i) value.v[i] = (1.0f/255) * ((uint8*)data)[i]; break;
        case SHORT1: for (int i=0; i<1; ++i) value.v[i] = (1.0f/65535) * ((uint16*)data)[i]; break;
        case SHORT2: for (int i=0; i<2; ++i) value.v[i] = (1.0f/65535) * ((uint16*)data)[i]; break;
        case SHORT4: for (int i=0; i<4; ++i) value.v[i] = (1.0f/65535) * ((uint16*)data)[i]; break;
        case HALF1: for (int i=0; i<1; ++i) value.v[i] = half_to_float(((int16*)data)[i]); break;
        case HALF2: for (int i=0; i<2; ++i) value.v[i] = half_to_float(((int16*)data)[i]); break;
        case HALF4: for (int i=0; i<4; ++i) value.v[i] = half_to_float(((int16*)data)[i]); break;
        case FLOAT1: for (int i=0; i<1; ++i) value.v[i] = ((float*)data)[i]; break;
        case FLOAT2: for (int i=0; i<2; ++i) value.v[i] = ((float*)data)[i]; break;
        case FLOAT4: for (int i=0; i<4; ++i) value.v[i] = ((float*)data)[i]; break;
        default: break;
    }
    return value;
}

LUISA_INLINE int4 pixel_read_int(PixelStorage storage, void* data)
{
    int4 value = make_int4(0);
    switch (storage) {
        case BYTE1: for (int i=0; i<1; ++i) value.v[i] = ((int8*)data)[i]; break;
        case BYTE2: for (int i=0; i<2; ++i) value.v[i] = ((int8*)data)[i]; break;
        case BYTE4: for (int i=0; i<4; ++i) value.v[i] = ((int8*)data)[i]; break;
        case SHORT1: for (int i=0; i<1; ++i) value.v[i] = ((int16*)data)[i]; break;
        case SHORT2: for (int i=0; i<2; ++i) value.v[i] = ((int16*)data)[i]; break;
        case SHORT4: for (int i=0; i<4; ++i) value.v[i] = ((int16*)data)[i]; break;
        case INT1: for (int i=0; i<1; ++i) value.v[i] = ((int*)data)[i]; break;
        case INT2: for (int i=0; i<2; ++i) value.v[i] = ((int*)data)[i]; break;
        case INT4: for (int i=0; i<4; ++i) value.v[i] = ((int*)data)[i]; break;
        default: break;
    }
    return value;
}

LUISA_INLINE uint4 pixel_read_uint(PixelStorage storage, void* data)
{
    uint4 value = make_uint4(0.0f);
    switch (storage) {
        case BYTE1: for (int i=0; i<1; ++i) value.v[i] = ((uint8*)data)[i]; break;
        case BYTE2: for (int i=0; i<2; ++i) value.v[i] = ((uint8*)data)[i]; break;
        case BYTE4: for (int i=0; i<4; ++i) value.v[i] = ((uint8*)data)[i]; break;
        case SHORT1: for (int i=0; i<1; ++i) value.v[i] = ((uint16*)data)[i]; break;
        case SHORT2: for (int i=0; i<2; ++i) value.v[i] = ((uint16*)data)[i]; break;
        case SHORT4: for (int i=0; i<4; ++i) value.v[i] = ((uint16*)data)[i]; break;
        case INT1: for (int i=0; i<1; ++i) value.v[i] = ((uint*)data)[i]; break;
        case INT2: for (int i=0; i<2; ++i) value.v[i] = ((uint*)data)[i]; break;
        case INT4: for (int i=0; i<4; ++i) value.v[i] = ((uint*)data)[i]; break;
        default: break;
    }
    return value;
}

struct LCTexture
{
    PixelStorage storage;
    uint dim;
    uint size[3];
    uint lodLevel;
    void* lods[20];
};

struct TextureView {
    const void* uniform ptr;
    uniform const uint level, dummy;
};

''', file=file)

        texrw_template = '''

LUISA_INLINE void texture2d_write_{T}(uniform LCTexture * tex, uint2 p, uint level, {T}4 value)
{
    uint pxsize = pixel_storage_size(tex->storage);
    uint width = max(tex->size[0] >> level, 1);
    uint height = max(tex->size[1] >> level, 1);
    if (p.v[0] >= width || p.v[1] >= height)
        print("texture write out of bound %u %u, %u %u\\n", p.v[0], p.v[1], width, height);
    void* data = (uint8*)tex->lods[level] + (p.v[1] * width + p.v[0]) * pxsize;
    pixel_write_{T}(tex->storage, data, value);
}

LUISA_INLINE {T}4 texture2d_read_{T}(uniform LCTexture * tex, uint2 p, uint level)
{
    uint pxsize = pixel_storage_size(tex->storage);
    uint width = max(tex->size[0] >> level, 1);
    uint height = max(tex->size[1] >> level, 1);
    if (p.v[0] >= width || p.v[1] >= height) {
        print("texture@%u read out of bound %u %u, %u %u\\n", level, p.v[0], p.v[1], width, height);
    }
    void* data = (uint8*)tex->lods[level] + (p.v[1] * width + p.v[0]) * pxsize;
    return pixel_read_{T}(tex->storage, data);
}

LUISA_INLINE void texture3d_write_{T}(uniform LCTexture * tex, uint3 p, uint level, {T}4 value)
{
    uint pxsize = pixel_storage_size(tex->storage);
    uint sx = max(tex->size[0] >> level, 1);
    uint sy = max(tex->size[1] >> level, 1);
    uint sz = max(tex->size[2] >> level, 1);
    if (p.v[0] >= sx || p.v[1] >= sy || p.v[2] >= sz)
        print("texture write out of bound %u %u %u, %u %u %u\\n", p.v[0], p.v[1], p.v[2], sx, sy, sz);

    void* data = (uint8*)tex->lods[level] + ((p.v[2] * sy + p.v[1]) * sx + p.v[0]) * pxsize;
    pixel_write_{T}(tex->storage, data, value);
}

LUISA_INLINE {T}4 texture3d_read_{T}(uniform LCTexture * tex, uint3 p, uint level)
{
    uint pxsize = pixel_storage_size(tex->storage);
    uint sx = max(tex->size[0] >> level, 1);
    uint sy = max(tex->size[1] >> level, 1);
    uint sz = max(tex->size[2] >> level, 1);
    if (p.v[0] >= sx || p.v[1] >= sy || p.v[2] >= sz)
        print("texture read out of bound %u %u %u, %u %u %u\\n", p.v[0], p.v[1], p.v[2], sx, sy, sz);

    void* data = (uint8*)tex->lods[level] + ((p.v[2] * sy + p.v[1]) * sx + p.v[0]) * pxsize;
    return pixel_read_{T}(tex->storage, data);
}


LUISA_INLINE {T}4 surf2d_read_{T}(uniform TextureView view, uint2 p)
{
    return texture2d_read_{T}((uniform LCTexture *uniform)view.ptr, p, view.level);
}

LUISA_INLINE void surf2d_write_{T}(uniform TextureView view, uint2 p, {T}4 value)
{
    texture2d_write_{T}((LCTexture*)view.ptr, p, view.level, value);
}

LUISA_INLINE {T}4 surf3d_read_{T}(uniform TextureView view, uint3 p)
{
    return texture3d_read_{T}((uniform LCTexture *uniform)view.ptr, p, view.level);
}

LUISA_INLINE void surf3d_write_{T}(uniform TextureView view, uint3 p, {T}4 value)
{
    texture3d_write_{T}((LCTexture*)view.ptr, p, view.level, value);
}'''
        print(texrw_template.replace('{T}', "float"), file=file)
        print(texrw_template.replace('{T}', "uint"), file=file)
        print(texrw_template.replace('{T}', "int"), file=file)

        print('''
struct LCBindlessItem {
    uniform const uint *buffer;
    uniform const LCTexture *tex2d;
    uniform const LCTexture *tex3d;
    uniform const uint sampler2d;
    uniform const uint sampler3d;
};

struct LCBindlessArray {
    uniform const LCBindlessItem *uniform items;
};

float4 bindless_texture_sample2d(uniform LCBindlessArray array, uint index, float2 uv);
float4 bindless_texture_sample2d_level(uniform LCBindlessArray array, uint index, float2 uv, float level);
float4 bindless_texture_sample2d_grad(uniform LCBindlessArray array, uint index, float2 uv, float2 ddx, float2 ddy);
float4 bindless_texture_sample3d(uniform LCBindlessArray array, uint index, float3 uv);
float4 bindless_texture_sample3d_level(uniform LCBindlessArray array, uint index, float3 uv, float level);
float4 bindless_texture_sample3d_grad(uniform LCBindlessArray array, uint index, float3 uv, float3 ddx, float3 ddy);

LUISA_INLINE float4 bindless_texture_sample2d_intlevel(uniform LCBindlessArray array, uint index, float2 uv, uint level)
{
    uniform const LCTexture * tex = array.items[index].tex2d;
    if (uv._x < 0 || uv._x > 1 || uv._y < 0 || uv._y > 1)
        return make_float4(0.0f);
    // bilinear
    uint w = max(tex->size[0]>>level, 1u);
    uint h = max(tex->size[1]>>level, 1u);
    float x = uv._x * w - 0.5f;
    float y = uv._y * h - 0.5f;
    uint x0 = (uint)max((int)0, (int)x);
    uint x1 = (uint)min((int)w-1, (int)x+1);
    uint y0 = (uint)max((int)0, (int)y);
    uint y1 = (uint)min((int)h-1, (int)y+1);
    float fx = max(0, min(1, x-x0));
    float fy = max(0, min(1, y-y0));
    return
    binary_add(binary_mul((1-fx)*(1-fy), texture2d_read_float(tex, make_uint2(x0,y0), level)),
    binary_add(binary_mul((1-fx)*(fy), texture2d_read_float(tex, make_uint2(x0,y1), level)),
    binary_add(binary_mul((fx)*(1-fy), texture2d_read_float(tex, make_uint2(x1,y0), level)),
               binary_mul((fx)*(fy), texture2d_read_float(tex, make_uint2(x1,y1), level)))));
}

LUISA_INLINE float4 bindless_texture_sample3d_intlevel(uniform LCBindlessArray array, uint index, float3 uv, uint level)
{
    uniform const LCTexture * tex = array.items[index].tex3d;
    if (uv._x < 0 || uv._x > 1 || uv._y < 0 || uv._y > 1 || uv._z < 0 || uv._z > 1)
        return make_float4(0.0f);
    // trilinear
    uint w = max(tex->size[0]>>level, 1u);
    uint h = max(tex->size[1]>>level, 1u);
    uint d = max(tex->size[2]>>level, 1u);
    float x = uv._x * w - 0.5f;
    float y = uv._y * h - 0.5f;
    float z = uv._z * d - 0.5f;
    uint x0 = (uint)max((int)0, (int)x);
    uint x1 = (uint)min((int)w-1, (int)x+1);
    uint y0 = (uint)max((int)0, (int)y);
    uint y1 = (uint)min((int)h-1, (int)y+1);
    uint z0 = (uint)max((int)0, (int)z);
    uint z1 = (uint)min((int)d-1, (int)z+1);
    float fx = max(0, min(1, x-x0));
    float fy = max(0, min(1, y-y0));
    float fz = max(0, min(1, z-z0));
    return
    binary_add(binary_mul((1-fx)*(1-fy)*(1-fz), texture3d_read_float(tex, make_uint3(x0,y0,z0), level)),
    binary_add(binary_mul((1-fx)*(1-fy)*(  fz), texture3d_read_float(tex, make_uint3(x0,y0,z1), level)),
    binary_add(binary_mul((1-fx)*(  fy)*(1-fz), texture3d_read_float(tex, make_uint3(x0,y1,z0), level)),
    binary_add(binary_mul((1-fx)*(  fy)*(  fz), texture3d_read_float(tex, make_uint3(x0,y1,z1), level)),
    binary_add(binary_mul((  fx)*(1-fy)*(1-fz), texture3d_read_float(tex, make_uint3(x1,y0,z0), level)),
    binary_add(binary_mul((  fx)*(1-fy)*(  fz), texture3d_read_float(tex, make_uint3(x1,y0,z1), level)),
    binary_add(binary_mul((  fx)*(  fy)*(1-fz), texture3d_read_float(tex, make_uint3(x1,y1,z0), level)),
               binary_mul((  fx)*(  fy)*(  fz), texture3d_read_float(tex, make_uint3(x1,y1,z1), level))
    )))))));
}

LUISA_INLINE float4 bindless_texture_sample2d(uniform LCBindlessArray array, uint index, float2 uv)
{
    return bindless_texture_sample2d_intlevel(array, index, uv, 0);
}

LUISA_INLINE float4 bindless_texture_sample3d(uniform LCBindlessArray array, uint index, float3 uv)
{
    return bindless_texture_sample3d_intlevel(array, index, uv, 0);
}

LUISA_INLINE float4 bindless_texture_sample2d_level(uniform LCBindlessArray array, uint index, float2 uv, float level)
{
    uint l0 = (int)level;
    uint l1 = (int)level+1;
    float fl = max(0, min(1, level - l0));
    return
    binary_add(binary_mul(1-fl, bindless_texture_sample2d_intlevel(array, index, uv, l0)),
               binary_mul(  fl, bindless_texture_sample2d_intlevel(array, index, uv, l1)));
}

LUISA_INLINE float4 bindless_texture_sample3d_level(uniform LCBindlessArray array, uint index, float3 uv, float level)
{
    uint l0 = (int)level;
    uint l1 = (int)level+1;
    float fl = max(0, min(1, level - l0));
    binary_add(binary_mul(1-fl, bindless_texture_sample3d_intlevel(array, index, uv, l0)),
               binary_mul(  fl, bindless_texture_sample3d_intlevel(array, index, uv, l1)));
}

LUISA_INLINE float4 bindless_texture_read2d(uniform LCBindlessArray array, uint index, uint2 coord)
{
    uniform const LCTexture * tex = array.items[index].tex2d;
    return texture2d_read_float(tex, coord, 0);
}

LUISA_INLINE float4 bindless_texture_read2d_level(uniform LCBindlessArray array, uint index, uint2 coord, uint level)
{
    uniform const LCTexture * tex = array.items[index].tex2d;
    return texture2d_read_float(tex, coord, level);
}

LUISA_INLINE float4 bindless_texture_read3d(uniform LCBindlessArray array, uint index, uint3 coord)
{
    uniform const LCTexture * tex = array.items[index].tex3d;
    return texture3d_read_float(tex, coord, 0);
}

LUISA_INLINE float4 bindless_texture_read3d_level(uniform LCBindlessArray array, uint index, uint3 coord, uint level)
{
    uniform const LCTexture * tex = array.items[index].tex3d;
    return texture3d_read_float(tex, coord, level);
}


uint2 bindless_texture_size2d(uniform LCBindlessArray array, uint index);
uint2 bindless_texture_size2d_level(uniform LCBindlessArray array, uint index, uint level);

uint3 bindless_texture_size3d(uniform LCBindlessArray array, uint index);
uint3 bindless_texture_size3d_level(uniform LCBindlessArray array, uint index, uint level);

LUISA_INLINE const void *bindless_buffer(uniform LCBindlessArray array, uint buffer_id) {
    return array.items[buffer_id].buffer;
}

LUISA_INLINE uint2 bindless_texture_size2d(uniform LCBindlessArray array, uint index) {
    uniform const LCTexture *tex = array.items[index].tex2d;
    return make_uint2(tex->size[0], tex->size[1]);
}

LUISA_INLINE uint2 bindless_texture_size2d_level(uniform LCBindlessArray array, uint index, uint level) {
    uniform const LCTexture *tex = array.items[index].tex2d;
    return make_uint2(max(tex->size[0] >> level, 1u), max(tex->size[1] >> level, 1u));
}

#ifdef LC_ISPC_RAYTRACING

#include <embree3/rtcore.isph>

LUISA_INLINE char trace_any(uniform LCAccel accel, LCRay ray) {
    uniform RTCIntersectContext ctx;
    rtcInitIntersectContext(&ctx);
    RTCRay r;
    r.org_x = ray.m0.a[0];
    r.org_y = ray.m0.a[1];
    r.org_z = ray.m0.a[2];
    r.tnear = ray.m1;
    r.dir_x = ray.m2.a[0];
    r.dir_y = ray.m2.a[1];
    r.dir_z = ray.m2.a[2];
    r.time = 0.f;
    r.tfar = ray.m3;
    r.mask = 0xffu;
    r.id = 0u;
    r.flags = 0u;
    rtcOccludedV((RTCScene)accel.handle, &ctx, &r);
    return r.tfar < 0.f;
}

LUISA_INLINE LCHit trace_closest(uniform LCAccel accel, LCRay ray) {
    uniform RTCIntersectContext ctx;
    rtcInitIntersectContext(&ctx);
    RTCRayHit rh;
    rh.ray.org_x = ray.m0.a[0];
    rh.ray.org_y = ray.m0.a[1];
    rh.ray.org_z = ray.m0.a[2];
    rh.ray.tnear = ray.m1;
    rh.ray.dir_x = ray.m2.a[0];
    rh.ray.dir_y = ray.m2.a[1];
    rh.ray.dir_z = ray.m2.a[2];
    rh.ray.time = 0.f;
    rh.ray.tfar = ray.m3;
    rh.ray.mask = 0xffu;
    rh.ray.id = 0u;
    rh.ray.flags = 0u;
    rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rh.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rh.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rtcIntersectV((RTCScene)accel.handle, &ctx, &rh);
    LCHit hit;
    hit.m0 = rh.hit.instID[0];
    hit.m1 = rh.hit.primID;
    hit.m2.v[0] = rh.hit.u;
    hit.m2.v[1] = rh.hit.v;
    return hit;
}

#endif

LUISA_INLINE void lc_assume(char) {}
LUISA_INLINE void lc_assume(uniform char pred) { assume(pred); }
LUISA_INLINE void lc_unreachable() { assert(false); }

#define make_array_type(name, T, N) struct name { T a[N]; }

#define array_access(arr, i) ((arr).a[i])
#define vector_access(vec, i) ((vec).v[i])
#define matrix_access(mat, i) ((mat).cols[i])
#define buffer_access(buf, i) ((buf)[i])
#define buffer_read(buf, i) buffer_access(buf, i)
#define buffer_write(buf, i, value) ((void)(buffer_access(buf, i) = (value)))
''', file=file)
