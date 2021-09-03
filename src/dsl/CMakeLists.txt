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
        builtin.h ref.h)

add_library(luisa-compute-dsl SHARED ${LUISA_COMPUTE_DSL_SOURCES})
target_link_libraries(luisa-compute-dsl PUBLIC luisa-compute-ast luisa-compute-runtime)
set_target_properties(luisa-compute-dsl PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ON)
