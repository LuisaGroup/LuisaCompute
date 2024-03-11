#pragma once

#ifdef _EXPORT
#define export [[clang::annotate("luisa", "export")]]
#define kernel_1d(x)
#define kernel_2d(x, y)
#define kernel_3d(x, y, z)
#else
#define export
#define kernel_1d(x) clang::annotate("luisa", "kernel_1d", (x))
#define kernel_2d(x, y) clang::annotate("luisa", "kernel_2d", (x), (y))
#define kernel_3d(x, y, z) clang::annotate("luisa", "kernel_3d", (x), (y), (z))
#endif
#define ignore clang::annotate("luisa", "ignore")
#define noignore clang::annotate("luisa", "noignore")
#define dump clang::annotate("luisa", "dump")
#define bypass clang::annotate("luisa", "bypass")
#define swizzle clang::annotate("luisa", "swizzle")
#define access clang::annotate("luisa", "access")
#define builtin(name) clang::annotate("luisa", "builtin", (name))
#define unaop(name) clang::annotate("luisa", "unaop", (name))
#define binop(name) clang::annotate("luisa", "binop", (name))
#define callop(name) clang::annotate("luisa", "callop", (name))
#define expr(name) clang::annotate("luisa", "expr", (name))


#define trait struct [[ignore]]