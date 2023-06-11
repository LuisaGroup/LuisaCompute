from os import listdir
from os.path import realpath, relpath, dirname, abspath, isdir, normpath


def normalize(base, path):
    return relpath(normpath(abspath(path)), base).replace("\\", "/")


def glob(source_files, header_files, folder, recursive=True):
    for f in listdir(folder):
        if f.endswith(".h") or f.endswith(".hpp"):
            header_files.append(f"{folder}/{f}")
        elif f.endswith(".cpp"):
            source_files.append(f"{folder}/{f}")
        elif isdir(f"{folder}/{f}"):
            glob(source_files, header_files, f"{folder}/{f}")


def find_binding_includes(file):
    with open(file, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    includes = []
    for line in lines:
        if line.startswith("#include"):
            if "binding" in line:
                includes.append(line)
    includes = set(includes)
    if includes:
        print(f"includes in {file}: {includes}")


if __name__ == "__main__":
    src_dir = normpath(abspath(f"{dirname(realpath(__file__))}/../src"))
    source_files = []
    header_files = []
    base_modules = [f"{src_dir}/{f}" for f in listdir(src_dir) if isdir(f"{src_dir}/{f}") and f != "ext"]
    for m in base_modules:
        glob(source_files, header_files, m)
    source_files = [normalize(src_dir, f) for f in source_files]
    header_files = [normalize(src_dir, f) for f in header_files]

    print(f"source files: {source_files}")
    print(f"header files: {header_files}")

    print()

    for h in header_files:
        find_binding_includes(f"{src_dir}/{h}")

