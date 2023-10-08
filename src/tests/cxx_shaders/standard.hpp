#pragma once
#define _cxx_builtin clang::annotate("luisa", "_cxx_builtin")
#define builtin(name) clang::annotate("luisa", "builtin", (name), 1)
#define attribute(semantic) clang::annotate("luisa", "attribute", semantic)
#define stage_in(i) clang::annotate("luisa", "stage_input", #i)
#define stage(name) clang::annotate("luisa", "stage", (#name))

#define in clang::annotate("luisa", "input_modifier", "in")
#define out clang::annotate("luisa", "input_modifier", "out")
#define inout clang::annotate("luisa", "input_modifier", "inout")

#define block clang::annotate("luisa", "block")

#define sv(semantic, ...) clang::annotate("luisa", "sv", semantic, __VA_ARGS__)
#define sv_position sv("position") 
#define sv_target(i) out, sv("target", (i)) 

namespace luisa::shader
{

}