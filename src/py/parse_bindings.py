from os.path import realpath, dirname


if __name__ == "__main__":
    module_dir = dirname(realpath(__file__))
    with open(f"{module_dir}/lcapi.cpp") as file:
        lines = [line.strip() for line in file.readlines()]

    builtin_functions = ["make_float2x2", "make_float3x3", "make_float4x4",
                         "dispatch_id", "thread_id", "block_id",
                         "dispatch_size", "set_block_size"]
    builtin_classes = {"float2x2": [], "float3x3": [], "float4x4": []}
    for t in ["int", "uint", "float", "bool"]:
        for n in [2, 3, 4]:
            builtin_functions.append(f"make_{t}{n}")
            builtin_classes[f"{t}{n}"] = []
    functions = builtin_functions
    classes = builtin_classes
    enums = {}
    last_enum = ""
    last_class = ""
    for line in lines:
        print(line)
        if line.startswith("m.def("):  # module functions
            functions.append(line.split('"')[1])
        elif "py::enum_<" in line:  # enums
            last_enum = line.split('"')[1]
            enums[last_enum] = []
        elif line.startswith(".value("):  # enum values
            value = line.split('"')[1]
            enums[last_enum].append(value)
        elif "py::class_<" in line:  # classes
            last_class = line.split('"')[1]
            classes[last_class] = []
        elif line.startswith(".def(") and '"' in line:  # class methods
            classes[last_class].append(line.split('"')[1])

    print("########## RESULTS ##########")
    print(f"functions: {functions}")
    print(f"classes: {classes}")
    print(f"enums: {enums}")

    # generate stub file
    with open(f"{module_dir}/luisa/lcapi.pyi", "w") as file:
        for function in functions:
            file.write(f"def {function}(*args, **kwargs): ...\n")
        for class_name, methods in classes.items():
            file.write(f"class {class_name}:\n")
            file.write("    def __init__(self, *args, **kwargs): ...\n")
            for method in {m for m in methods}:
                file.write(f"    def {method}(self, *args, **kwargs): ...\n")
        for enum_name, values in enums.items():
            file.write(f"class {enum_name}:\n")
            for value in values:
                file.write(f"    {value} = ...\n")

    with open(f"{module_dir}/../dsl/builtin.h") as file:
        lines = [line.strip() for line in file.readlines()]

    functions = builtin_functions
    classes = builtin_classes
    for line in lines:
        if "[[nodiscard]] inline auto " in line and \
                "make" not in line and "def" not in line and \
                "soa" not in line and "thread" not in line and \
                "block" not in line and "dispatch" not in line and \
                "eval" not in line and "cast" not in line and "as" not in line and \
                "vectorize" not in line:
            functions.append(line.split("auto")[1].split("(")[0].strip())
    functions = {f for f in functions}
    with open(f"{module_dir}/luisa/mathtypes.pyi", "w") as file:
        for function in functions:
            file.write(f"def {function}(*args, **kwargs): ...\n")
        for class_name, methods in classes.items():
            file.write(f"class {class_name}:\n")
            file.write("    def __init__(self, *args, **kwargs): ...\n")
            if "2" in class_name:
                for x in "xy":
                    file.write(f"    @property\n    def {x}(self): ...\n")
                for x in "xy":
                    for y in "xy":
                        file.write(f"    @property\n    def {x}{y}(self): ...\n")
                for x in "xy":
                    for y in "xy":
                        for z in "xy":
                            file.write(f"    @property\n    def {x}{y}{z}(self): ...\n")
                for x in "xy":
                    for y in "xy":
                        for z in "xy":
                            for w in "xy":
                                file.write(f"    @property\n    def {x}{y}{z}{w}(self): ...\n")
            elif "3" in class_name:
                for x in "xyz":
                    file.write(f"    @property\n    def {x}(self): ...\n")
                for x in "xyz":
                    for y in "xyz":
                        file.write(f"    @property\n    def {x}{y}(self): ...\n")
                for x in "xyz":
                    for y in "xyz":
                        for z in "xyz":
                            file.write(f"    @property\n    def {x}{y}{z}(self): ...\n")
                for x in "xyz":
                    for y in "xyz":
                        for z in "xyz":
                            for w in "xyz":
                                file.write(f"    @property\n    def {x}{y}{z}{w}(self): ...\n")
            elif "4" in class_name:
                for x in "xyzw":
                    file.write(f"    @property\n    def {x}(self): ...\n")
                for x in "xyzw":
                    for y in "xyzw":
                        file.write(f"    @property\n    def {x}{y}(self): ...\n")
                for x in "xyzw":
                    for y in "xyzw":
                        for z in "xyzw":
                            file.write(f"    @property\n    def {x}{y}{z}(self): ...\n")
                for x in "xyzw":
                    for y in "xyzw":
                        for z in "xyzw":
                            for w in "xyzw":
                                file.write(f"    @property\n    def {x}{y}{z}{w}(self): ...\n")
