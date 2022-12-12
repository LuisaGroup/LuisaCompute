set(LUISA_COMPUTE_DSL_SOURCES
        arg.h
        builtin.h
        constant.h
        expr.cpp
        expr.h
        expr_traits.h
        func.h
        local.h
        operators.h
        polymorphic.h
        printer.cpp printer.h
        ref.h
        shared.h
        stmt.h
        struct.h
        sugar.h
        swizzle_2.inl.h
        swizzle_3.inl.h
        swizzle_4.inl.h
        syntax.h
        var.h)

add_library(luisa-compute-dsl SHARED ${LUISA_COMPUTE_DSL_SOURCES})
target_link_libraries(luisa-compute-dsl PUBLIC luisa-compute-ast luisa-compute-runtime)
target_compile_definitions(luisa-compute-dsl PRIVATE LC_DSL_EXPORT_DLL=1)
set_target_properties(luisa-compute-dsl PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD})

luisa_compute_install(dsl SOURCES ${LUISA_COMPUTE_DSL_SOURCES})
