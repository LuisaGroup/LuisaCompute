def generate(file, dim):
    entries = ["x", "y", "z", "w"][:dim]
    for x in entries:
        for y in entries:
            print(f"[[nodiscard]] auto {x}{y}() const noexcept {{ return make_vector({x}, {y}); }}", file=file)
    for x in entries:
        for y in entries:
            for z in entries:
                print(f"[[nodiscard]] auto {x}{y}{z}() const noexcept {{ return make_vector({x}, {y}, {z}); }}", file=file)
    for x in entries:
        for y in entries:
            for z in entries:
                for w in entries:
                    print(f"[[nodiscard]] auto {x}{y}{z}{w}() const noexcept {{ return make_vector({x}, {y}, {z}, {w}); }}", file=file)


if __name__ == "__main__":
    for dim in range(2, 5):
        with open(f"swizzle_{dim}.inl.h", "w") as file:
            print("#pragma once", file=file)
            generate(file, dim)
