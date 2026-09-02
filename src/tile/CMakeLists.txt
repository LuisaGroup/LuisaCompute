set(LUISA_COMPUTE_TILE_SOURCES
        dimension.cpp
        ir.cpp
        layout.cpp
        target.cpp
        verifier.cpp)

add_library(luisa-compute-tile ${LUISA_COMPUTE_TILE_SOURCES})
target_link_libraries(luisa-compute-tile PUBLIC luisa-compute-core)
target_compile_definitions(luisa-compute-tile PRIVATE LUISA_TILE_EXPORT_DLL=1)

set_target_properties(luisa-compute-tile PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
        OUTPUT_NAME luisa-tile)

luisa_compute_install(tile SOURCES ${LUISA_COMPUTE_TILE_SOURCES})

if (LUISA_COMPUTE_ENABLE_TILE_TIRX_BRIDGE)
    add_subdirectory(bridge/tirx)
endif ()
