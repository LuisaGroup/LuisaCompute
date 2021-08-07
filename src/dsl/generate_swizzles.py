def generate(file, dim):
    entries = ["x", "y", "z", "w"][:dim]
    for ix, x in enumerate(entries):
        for iy, y in enumerate(entries):
            print(f"[[nodiscard]] auto {x}{y}() const noexcept {{ " +
                  f"return Expr<Vector<T, 2>>{{" +
                  f"detail::FunctionBuilder::current()->swizzle(" +
                  f"Type::of<Vector<T, 2>>(), this->expression(), 2u, 0x{iy}{ix}u)}}; }}",
                  file=file)
    for ix, x in enumerate(entries):
        for iy, y in enumerate(entries):
            for iz, z in enumerate(entries):
                print(f"[[nodiscard]] auto {x}{y}{z}() const noexcept {{ " +
                      f"return Expr<Vector<T, 3>>{{" +
                      f"detail::FunctionBuilder::current()->swizzle(" +
                      f"Type::of<Vector<T, 3>>(), this->expression(), 3u, 0x{iz}{iy}{ix}u)}}; }}",
                      file=file)
    for ix, x in enumerate(entries):
        for iy, y in enumerate(entries):
            for iz, z in enumerate(entries):
                for iw, w in enumerate(entries):
                    print(f"[[nodiscard]] auto {x}{y}{z}{w}() const noexcept {{ " +
                          f"return Expr<Vector<T, 4>>{{" +
                          f"detail::FunctionBuilder::current()->swizzle(" +
                          f"Type::of<Vector<T, 4>>(), this->expression(), 4u, 0x{iw}{iz}{iy}{ix}u)}}; }}",
                          file=file)


if __name__ == "__main__":
    for dim in range(2, 5):
        with open(f"swizzle_{dim}.inl.h", "w") as file:
            generate(file, dim)
