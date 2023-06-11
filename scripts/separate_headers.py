from os import listdir, makedirs
from os.path import realpath, relpath, dirname, abspath, isdir, normpath, basename
from shutil import move


def normalize(base, path):
    return relpath(normpath(abspath(path)), base).replace("\\", "/")


def glob(source_files, header_files, folder, recursive=True):
    for f in listdir(folder):
        if f.endswith(".h") or f.endswith(".hpp"):
            header_files.append(f"{folder}/{f}")
        elif f.endswith(".cpp"):
            source_files.append(f"{folder}/{f}")
        elif recursive and isdir(f"{folder}/{f}"):
            glob(source_files, header_files, f"{folder}/{f}")


def fix_include_line(src_dir, file, line, moved_headers):
    include = line.strip().split()[1]
    if include.startswith('"'):
        file_dir = dirname(f"{src_dir}/{file}")
        norm_inc = normalize(src_dir, f"{file_dir}/{include[1:-1]}")
        if norm_inc in moved_headers:
            include = f"<luisa/{norm_inc}>"
    elif include.startswith('<'):
        if include[1:-1] in moved_headers:
            include = f"<luisa/{include[1:-1]}>"
    return f"#include {include}\n"


def fix_include(src_dir, file, moved_headers):
    print(f"Fixing {file}...")
    fixed_lines = []
    abs_path = f"{src_dir}/{file}"
    with open(abs_path, "r", encoding="utf8") as f:
        lines = f.readlines()
    for line in lines:
        if line.strip().startswith("#include"):
            fixed = fix_include_line(src_dir, file, line, moved_headers)
            print(f"  {line.strip()} -> {fixed.strip()}")
            fixed_lines.append(fixed)
        else:
            fixed_lines.append(line)
    if not fixed_lines or fixed_lines[-1].strip():
        fixed_lines.append("\n")
    with open(abs_path, "w", encoding="utf8") as f:
        f.writelines(fixed_lines)


def move_header(src_dir, header):
    src_path = f"{src_dir}/{header}"
    dst_path = f"{src_dir}/../include/luisa/{header}"
    dst_dir = dirname(dst_path)
    makedirs(dst_dir, exist_ok=True)
    print(f"Moving {src_path} -> {dst_path}")
    move(src_path, dst_path)


if __name__ == "__main__":
    src_dir = normpath(abspath(f"{dirname(realpath(__file__))}/../src")).replace("\\", "/")
    source_files = []
    header_files = []
    base_modules = [f"{src_dir}/{f}" for f in listdir(src_dir) if isdir(f"{src_dir}/{f}") and f != "ext"]
    for m in base_modules:
        recursive = not m.endswith("rust")
        glob(source_files, header_files, m, recursive)
    source_files = [normalize(src_dir, f) for f in source_files]
    header_files = [normalize(src_dir, f) for f in header_files]

    exclude = ["backends", "py", "tests"]
    include = ["backends/ext"]
    headers_to_move = [f for f in header_files if
                       not any([f.startswith(e) for e in exclude]) or
                       any([f.startswith(e) for e in include])]

    for f in source_files + header_files:
        fix_include(src_dir, f, headers_to_move)

    for h in headers_to_move:
        move_header(src_dir, h)
