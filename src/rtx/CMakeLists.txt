set(LUISA_COMPUTE_RTX_SOURCES
        accel.cpp accel.h
        ray.h
        hit.h
        mesh.cpp mesh.h
        procedural_primitive.cpp procedural_primitive.h)

add_library(luisa-compute-rtx SHARED ${LUISA_COMPUTE_RTX_SOURCES})
target_link_libraries(luisa-compute-rtx PUBLIC luisa-compute-runtime)
if(LUISA_COMPUTE_ENABLE_DSL)
target_link_libraries(luisa-compute-rtx PUBLIC luisa-compute-dsl)
endif()
target_compile_definitions(luisa-compute-rtx PRIVATE LC_RUNTIME_EXPORT_DLL=1)
set_target_properties(luisa-compute-rtx PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD})

luisa_compute_install(rtx SOURCES ${LUISA_COMPUTE_RTX_SOURCES})
