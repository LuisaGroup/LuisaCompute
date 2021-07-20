set(LUISA_COMPUTE_RTX_SOURCES
    geometry.cpp geometry.h
    ray.cpp ray.h
    hit.cpp hit.h)

add_library(luisa-compute-rtx SHARED ${LUISA_COMPUTE_RTX_SOURCES})
target_link_libraries(luisa-compute-rtx PUBLIC luisa-compute-runtime luisa-compute-dsl)
set_target_properties(luisa-compute-rtx PROPERTIES
                      WINDOWS_EXPORT_ALL_SYMBOLS ON
                      UNITY_BUILD ON)
