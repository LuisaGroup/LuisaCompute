#pragma once
#define ignore clang::annotate("luisa-shader", "ignore")
#define type(name) clang::annotate("luisa-shader", "type", (name))
#define type_ex(name, ...) clang::annotate("luisa-shader", "type_ex", (name), __VA_ARGS__)
#define builtin(name) clang::annotate("luisa-shader", "builtin", (name), 1)
#define kernel_1d(x) clang::annotate("luisa-shader", "kernel_1d", (x))
#define kernel_2d(x, y) clang::annotate("luisa-shader", "kernel_2d", (x), (y))
#define kernel_3d(x, y, z) clang::annotate("luisa-shader", "kernel_3d", (x), (y), (z))
