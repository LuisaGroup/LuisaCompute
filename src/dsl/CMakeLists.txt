set(LUISA_COMPUTE_DSL_SOURCES
        var.h
        expr.cpp expr.h
        expr_traits.h
        syntax.h
        func.h
        shared.h
        constant.h
        arg.h
        struct.h
        stmt.h
        sugar.h
        builtin.h
        ref.h
        operators.h
        printer.cpp printer.h local.h)

add_library(luisa-compute-dsl SHARED ${LUISA_COMPUTE_DSL_SOURCES})
target_link_libraries(luisa-compute-dsl PUBLIC luisa-compute-ast luisa-compute-runtime)
target_compile_definitions(luisa-compute-dsl PRIVATE LC_DSL_EXPORT_DLL=1)
set_target_properties(luisa-compute-dsl PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ON)
