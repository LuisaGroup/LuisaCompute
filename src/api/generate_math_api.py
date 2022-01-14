from os.path import realpath, dirname


def generate_vector_decl(file, t: str, n: int):
    # create
    print(f"LUISA_EXPORT_API void *luisa_compute_{t}{n}_create(", file=file, end="")
    tm = {
        "int": "int",
        "uint": "uint32_t",
        "float": "float",
        "bool": "int"
    }
    for i in range(n - 1):
        print(f"{tm[t]} v{i}, ", file=file, end="")
    print(f"{tm[t]} v{n - 1}) LUISA_NOEXCEPT;", file=file)
    # destroy
    print(f"LUISA_EXPORT_API void luisa_compute_{t}{n}_destroy(void *v) LUISA_NOEXCEPT;", file=file)


def generate_matrix_decl(file, n: int):
    # create
    print(f"LUISA_EXPORT_API void *luisa_compute_float{n}x{n}_create(", file=file)
    for i in range(n - 1):
        print("    ", file=file, end="")
        for j in range(n - 1):
            print(f"float m{i}{j}, ", file=file, end="")
        print(f"float m{i}{n - 1},", file=file)
    print("    ", file=file, end="")
    for j in range(n - 1):
        print(f"float m{n - 1}{j}, ", file=file, end="")
    print(f"float m{n - 1}{n - 1}) LUISA_NOEXCEPT;", file=file)
    # destroy
    print(f"LUISA_EXPORT_API void luisa_compute_float{n}x{n}_destroy(void *m) LUISA_NOEXCEPT;", file=file)


def generate_header_file(filename):
    with open(filename, "w") as file:
        print("""//
// Created by Mike on 2022/1/7.
//

#pragma once

#include <core/platform.h>
        """, file=file)
        for t in ["int", "uint", "float", "bool"]:
            for n in [2, 3, 4]:
                generate_vector_decl(file, t, n)
        for n in [2, 3, 4]:
            generate_matrix_decl(file, n)


def generate_source_file(filename):
    with open(filename, "w") as file:
        print("""//
// Created by Mike on 2022/1/7.
//

#include <luisa-compute.h>
#include <api/math.h>
        """, file=file)


if __name__ == "__main__":
    api_directory = dirname(realpath(__file__))
    generate_header_file(f"{api_directory}/math.h")
    generate_source_file(f"{api_directory}/math.cpp")
