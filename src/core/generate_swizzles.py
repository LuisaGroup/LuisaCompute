from os.path import realpath, dirname


def generate(file, dim):
    entries = ["x", "y", "z", "w"][:dim]
    for x in entries:
        for y in entries:
            print(f"[[nodiscard]] constexpr auto {x}{y}() const noexcept {{ return Vector<T, 2>{{{x}, {y}}}; }}", file=file)
    for x in entries:
        for y in entries:
            for z in entries:
                print(f"[[nodiscard]] constexpr auto {x}{y}{z}() const noexcept {{ return Vector<T, 3>{{{x}, {y}, {z}}}; }}",
                      file=file)
    for x in entries:
        for y in entries:
            for z in entries:
                for w in entries:
                    print(f"[[nodiscard]] constexpr auto {x}{y}{z}{w}() const noexcept {{ " +
                          f"return Vector<T, 4>{{{x}, {y}, {z}, {w}}}; }}",
                          file=file)


if __name__ == "__main__":
    base = dirname(realpath(__file__))
    for dim in range(2, 5):
        with open(f"{base}/swizzle_{dim}.inl.h", "w") as file:
            generate(file, dim)
