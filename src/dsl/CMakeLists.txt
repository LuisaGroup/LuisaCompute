if (LUISA_COMPUTE_ENABLE_DSL)

    set(LUISA_COMPUTE_DSL_RTX_SOURCES
            rtx/accel.cpp
            rtx/hit.cpp
            rtx/ray.cpp
            rtx/ray_query.cpp)

    set(LUISA_COMPUTE_DSL_RASTER_SOURCES
            raster/raster_kernel.cpp)

    set(LUISA_COMPUTE_DSL_SOURCES
            builtin.cpp
            dispatch_indirect.cpp
            func.cpp
            local.cpp
            polymorphic.cpp
            printer.cpp
            resource.cpp
            soa.cpp
            sugar.cpp
            ${LUISA_COMPUTE_DSL_RTX_SOURCES}
            ${LUISA_COMPUTE_DSL_RASTER_SOURCES})

    add_library(luisa-compute-dsl SHARED ${LUISA_COMPUTE_DSL_SOURCES})
    target_link_libraries(luisa-compute-dsl PUBLIC luisa-compute-ast luisa-compute-runtime luisa-compute-ir)
    target_compile_definitions(luisa-compute-dsl
            PRIVATE LC_DSL_EXPORT_DLL=1
            PUBLIC LUISA_ENABLE_DSL=1)
    set_target_properties(luisa-compute-dsl PROPERTIES
            UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
            OUTPUT_NAME lc-dsl)
    luisa_compute_install(dsl SOURCES ${LUISA_COMPUTE_DSL_SOURCES})
    target_precompile_headers(luisa-compute-dsl PRIVATE pch.h)
else ()
    add_library(luisa-compute-dsl INTERFACE)
    luisa_compute_install(dsl)
endif ()
