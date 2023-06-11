from os import listdir
from os.path import realpath, dirname, isdir, relpath


def glob_headers(headers, folder):
    for f in listdir(folder):
        if f.endswith(".h") or f.endswith(".hpp"):
            headers.append(f"{folder}/{f}")
        elif isdir(f"{folder}/{f}"):
            glob_headers(headers, f"{folder}/{f}")


if __name__ == "__main__":
    base = realpath(dirname(realpath(__file__)) + "/../include/luisa").replace("\\", "/")
    # glob all headers
    headers = []
    modules = [f for f in listdir(base) if isdir(f"{base}/{f}") and f not in ["api", "backends"]]
    for module in modules:
        glob_headers(headers, f"{base}/{module}")
    headers = [relpath(header, base).replace("\\", "/") for header in headers if
               not header.endswith(".inl.h")]

    header_groups = {}
    for header in headers:
        group = header.split("/")[0]
        if group not in header_groups:
            header_groups[group] = []
        header_groups[group].append(header)

    optional_modules = ["dsl", "gui", "ir", "rust"]
    with open(f"{base}/luisa-compute.h", "w", encoding="utf8") as f:
        f.write("#pragma once\n\n")
        for group, headers in header_groups.items():
            if group in optional_modules:
                f.write(f"#ifdef LUISA_ENABLE_{group.upper()}\n")
            for header in headers:
                f.write(f"#include <luisa/{header}>\n")
            if group in optional_modules:
                f.write(f"#endif\n")
            f.write("\n")
