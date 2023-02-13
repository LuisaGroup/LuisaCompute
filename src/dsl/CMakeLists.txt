if (LUISA_COMPUTE_ENABLE_DSL)

    set(LUISA_COMPUTE_DSL_RTX_SOURCES
            rtx/aabb.h
            rtx/accel.cpp
            rtx/accel.h
            rtx/hit.cpp
            rtx/hit.h
            rtx/ray.cpp
            rtx/ray.h
            rtx/ray_query.cpp
            rtx/ray_query.h)

    set(LUISA_COMPUTE_DSL_RASTER_SOURCES
            raster/draw_indirect.h)

    set(LUISA_COMPUTE_DSL_SOURCES
            arg.h
            autodiff.cpp autodiff.h
            buffer.h
            builtin.h
            constant.h
            dispatch_indirect.cpp dispatch_indirect.h
            expr.cpp expr.h
            expr_traits.h
            func.cpp func.h
            generate_swizzles.py
            image.h
            local.cpp local.h
            operators.h
            polymorphic.cpp polymorphic.h
            printer.cpp printer.h
            ref.cpp ref.h
            shared.h
            stmt.h
            struct.h
            sugar.h
            swizzle_2.inl.h
            swizzle_3.inl.h
            swizzle_4.inl.h
            syntax.h
            var.h
            volume.h
            ${LUISA_COMPUTE_DSL_RTX_SOURCES}
            ${LUISA_COMPUTE_DSL_RASTER_SOURCES})

    add_library(luisa-compute-dsl SHARED ${LUISA_COMPUTE_DSL_SOURCES})
    target_link_libraries(luisa-compute-dsl PUBLIC luisa-compute-ast luisa-compute-runtime)
    target_compile_definitions(luisa-compute-dsl
            PRIVATE LC_DSL_EXPORT_DLL=1
            PUBLIC LUISA_DSL_ENABLED=1)
    set_target_properties(luisa-compute-dsl PROPERTIES
            WINDOWS_EXPORT_ALL_SYMBOLS ON
            UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD})
    luisa_compute_install(dsl SOURCES ${LUISA_COMPUTE_DSL_SOURCES})
else ()
    add_library(luisa-compute-dsl INTERFACE)
    luisa_compute_install(dsl)
endif ()
