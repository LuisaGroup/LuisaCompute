from os import listdir
from os.path import realpath, dirname


if __name__ == "__main__":
    base = dirname(realpath(__file__))
    modules = ["core", "ast", "runtime", "dsl", "rtx", "gui"]
    with open("luisa-compute.h", "w") as file:
        print("""//
// Created by Mike on 2021/12/8.
//

#pragma once""", file=file)
        for m in modules:
            if m == "gui":
                print(f"\n#ifdef LUISA_GUI_ENABLED", file=file, end="")
            headers = [h for h in listdir(f"{base}/{m}") if h.endswith(".h") and "impl" not in h and ".inl" not in h]
            print(file=file)
            for h in sorted(headers):
                print(f"#include <{m}/{h}>", file=file)
            if m == "gui":
                print(f"#endif", file=file)
