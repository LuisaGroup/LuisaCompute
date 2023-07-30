import sys
import os


def embed_file(file_name, array_name, header, source):
    with open(file_name, "rb") as f:
        data = [f"0x{ord(c):02x}" for c in f.read().decode("utf-8").replace("\r\n", "\n")]
    size = len(data)
    header.write(f'extern "C" const char {array_name}[{size}];\n')
    source.write(f'\nextern "C" const char {array_name}[{size}] = {{\n')
    wrapped = ["    " + ", ".join(data[i: i + 16]) for i in range(0, len(data), 16)]
    source.write(",\n".join(wrapped))
    source.write("\n};\n")


def main(path):
    files = sorted([f for f in os.listdir(path) if not f.startswith(".")])
    print("BUILTIN_FILES = [")
    for f in files:
        print(f'    "{f}",')
    print("]\n")
    folder_name = os.path.basename(path)
    output_name = f"{os.path.abspath(path)}/../{folder_name}_embedded"
    with open(f"{output_name}.h", "w") as header_file:
        header_file.write("#pragma once\n\n")
        with open(f"{output_name}.cpp", "w") as source_file:
            source_file.write("// clang-format off")
            for f in files:
                array_name = f"luisa_{folder_name}_{os.path.splitext(os.path.basename(f))[0]}"
                embed_file(f"{path}/{f}", array_name, header_file, source_file)
            source_file.write("// clang-format on\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <builtin-folder>")
        sys.exit(1)
    main(sys.argv[1])
