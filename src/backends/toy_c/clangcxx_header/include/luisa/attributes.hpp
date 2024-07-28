#pragma once

#ifdef _EXPORT
#define export [[clang::annotate("luisa-shader", "export")]]
#define kernel_1d(x)
#define kernel_2d(x, y)
#define kernel_3d(x, y, z)
#else
#define export
#define kernel_1d(x) clang::annotate("luisa-shader", "kernel_1d", (x))
#define kernel_2d(x, y) clang::annotate("luisa-shader", "kernel_2d", (x), (y))
#define kernel_3d(x, y, z) clang::annotate("luisa-shader", "kernel_3d", (x), (y), (z))
#endif
#define ignore clang::annotate("luisa-shader", "ignore")
#define noignore clang::annotate("luisa-shader", "noignore")
#define dump clang::annotate("luisa-shader", "dump")
#define bypass clang::annotate("luisa-shader", "bypass")
#define swizzle clang::annotate("luisa-shader", "swizzle")
#define access clang::annotate("luisa-shader", "access")
#define builtin(name) clang::annotate("luisa-shader", "builtin", (name))
#define unaop(name) clang::annotate("luisa-shader", "unaop", (name))
#define binop(name) clang::annotate("luisa-shader", "binop", (name))
#define callop(name) clang::annotate("luisa-shader", "callop", (name))
#define ext_call(name) clang::annotate("luisa-shader", "ext_call", (name))
#define expr(name) clang::annotate("luisa-shader", "expr", (name))

#define custom_attr(name) clang::annotate("luisa-shader", "custom", name)

#define trait struct [[ignore]]