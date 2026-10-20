add_library(luisa-compute-ext-stb-interface INTERFACE)
target_include_directories(luisa-compute-ext-stb-interface INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        # Fallback for <stb/stb_image_resize2.h> (upstream stb moved the header
        # out of the repo; see compat/stb/stb_image_resize2.h).
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/compat>)

add_library(luisa-compute-ext-stb OBJECT stb.c)
target_link_libraries(luisa-compute-ext-stb PRIVATE luisa-compute-ext-stb-interface)
set_target_properties(luisa-compute-ext-stb PROPERTIES POSITION_INDEPENDENT_CODE ON)
