add_library(luisa-compute-tile-bridge-xir lower.cpp planner.cpp)
target_link_libraries(luisa-compute-tile-bridge-xir PUBLIC
        luisa-compute-tile luisa-compute-xir)
target_compile_definitions(luisa-compute-tile-bridge-xir PRIVATE
        LUISA_TILE_XIR_BRIDGE_EXPORT_DLL=1)
set_target_properties(luisa-compute-tile-bridge-xir PROPERTIES
        OUTPUT_NAME luisa-tile-bridge-xir)
add_library(luisa::tile-xir ALIAS luisa-compute-tile-bridge-xir)
luisa_compute_install(tile-bridge-xir SOURCES lower.cpp planner.cpp)
