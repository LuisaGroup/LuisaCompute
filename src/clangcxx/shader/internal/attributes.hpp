#pragma once
#define ignore clang::annotate("luisa-shader", "ignore")
#define bypass clang::annotate("luisa-shader", "bypass")
#define swizzle clang::annotate("luisa-shader", "swizzle")
#define builtin(name) clang::annotate("luisa-shader", "builtin", (name))
#define binop(name) clang::annotate("luisa-shader", "binop", (name))
#define callop(name) clang::annotate("luisa-shader", "callop", (name))
#define expr(name) clang::annotate("luisa-shader", "callop", (name))
#define kernel_1d(x) clang::annotate("luisa-shader", "kernel_1d", (x))
#define kernel_2d(x, y) clang::annotate("luisa-shader", "kernel_2d", (x), (y))
#define kernel_3d(x, y, z) clang::annotate("luisa-shader", "kernel_3d", (x), (y), (z))
// TODO: external library, deserialize AST from lib_path, do we need this?
// [[external("my_lib.bytes")]] int func(int a, int b);
#define external(lib_path) clang::annotate("luisa-shader", "external", (lib_path))

#define trait struct [[ignore]]