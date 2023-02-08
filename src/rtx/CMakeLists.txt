set(LUISA_COMPUTE_RTX_SOURCES
        accel.cpp accel.h
        ray.cpp ray.h
        hit.cpp hit.h
        mesh.cpp mesh.h
        option.h
        procedural_primitive.cpp procedural_primitive.h
        ray_query.cpp ray_query.h)

add_library(luisa-compute-rtx SHARED ${LUISA_COMPUTE_RTX_SOURCES})
target_link_libraries(luisa-compute-rtx PUBLIC luisa-compute-runtime luisa-compute-dsl)
target_compile_definitions(luisa-compute-rtx PRIVATE LC_RUNTIME_EXPORT_DLL=1)
set_target_properties(luisa-compute-rtx PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD})

luisa_compute_install(rtx SOURCES ${LUISA_COMPUTE_RTX_SOURCES})
