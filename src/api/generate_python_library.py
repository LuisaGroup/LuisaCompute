from os import listdir, makedirs
from os.path import realpath, dirname

header_to_ctypes = {
    "void *": "c_void_p",
    "char *": "c_char_p",
    "void": "None",
    "int": "c_int",
    "int32_t": "c_int32",
    "uint32_t": "c_uint32",
    "int64_t": "c_int64",
    "uint64_t": "c_uint64",
    "size_t": "c_size_t"
}

api_directory = dirname(realpath(__file__))
pylib_directory = f"{dirname(api_directory)}/python/LuisaCompute/_internal"
makedirs(pylib_directory, exist_ok=True)

header_files = [f for f in listdir(api_directory) if f.endswith(".h")]
print(header_files)


def parse_argument(p):
    tokens = p.split()
    t, a = " ".join(tokens[:-1]), tokens[-1]
    if a.startswith("*"):
        t, a = f"{t} *", a[1:]
    return {
        "name": a,
        "type": header_to_ctypes[t.replace("const ", "")]
    }


def parse_function(f):
    ret, decl = f.split("luisa_compute_")
    name, args = decl.split("(")
    args = args.split(")")[0]
    args = [p.strip() for p in args.split(", ")] if args else []
    return {
        "name": f"{name}",
        "return": header_to_ctypes[ret.strip().replace("const ", "")],
        "args": [parse_argument(arg) for arg in args]
    }


def parse_file(f):
    with open(f"{api_directory}/{f}") as header:
        content = "".join(l.strip() for l in header.readlines())
    functions = [f.strip() for f in content.split("LUISA_EXPORT_API")[1:]]
    return [parse_function(f) for f in functions]


def generate_module(header, functions):
    def invoke_arg(type, name):
        return f"{name}.encode()" if type == "c_char_p" else name

    def invoke_ret(type, expr):
        if type == "None":
            return f"    {expr}"
        elif type == "c_char_p":
            return f"    return bytes({expr}).decode()"
        else:
            return f"    return {expr}"

    with open(f"{pylib_directory}/{header}.py", "w") as file:
        print(f"from ctypes import {', '.join(v for v in header_to_ctypes.values() if v != 'None')}", file=file)
        print(f"from .config import dll", file=file)
        for f in functions:
            print(f"\n\ndll.luisa_compute_{f['name']}.restype = {f['return']}", file=file)
            print(f"dll.luisa_compute_{f['name']}.argtypes = [{', '.join(a['type'] for a in f['args'])}]", file=file)
            print(f"\n\ndef {f['name']}({', '.join(a['name'] for a in f['args'])}):", file=file)
            print(invoke_ret(
                f["return"],
                f"dll.luisa_compute_{f['name']}({', '.join(invoke_arg(a['type'], a['name']) for a in f['args'])})"),
                file=file)


for header in header_files:
    functions = parse_file(header)
    generate_module(header.replace(".h", ""), functions)
