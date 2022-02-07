from os.path import realpath, dirname

if __name__ == "__main__":
    curr_dir = dirname(realpath(__file__))
    math_library_name = "ispc_device_math"
    surf_library_name = "ispc_device_resource"
    with open(f"{curr_dir}/{math_library_name}.isph", "w") as file:
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

#         # any, all, none
#         for f, uop, bop in [("any", "", "||"), ("all", "", "&&"), ("none", "!", "&&")]:
#             for i in range(2, 5):
#                 elements = ["x", "y", "z", "w"][:i]
#                 print(
#                     f"inline auto {f}(bool{i} v) {{ return {f' {bop} '.join(f'{uop}v.{m}' for m in elements)}; }}",
#                     file=file)
