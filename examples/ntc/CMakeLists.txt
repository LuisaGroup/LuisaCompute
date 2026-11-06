# LuisaCompute NTC (RTX Neural Texture Compression) CMake configuration

add_library(luisa-ntc STATIC
    src/core/manifest.cpp
    src/core/feature_grid.cpp
    src/core/ntex_container.cpp
    src/core/gdeflate_codec.cpp
    src/infer/infer_engine.cpp
    src/bcn/bcn_encoder.cpp
    src/train/trainer.cpp
)

target_link_libraries(luisa-ntc PUBLIC
    luisa::compute
    luisa-compute-ext-yyjson-interface
    luisa-compute-ext-stb-interface
)

target_include_directories(luisa-ntc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/libntc
    ${CMAKE_SOURCE_DIR}/examples
    ${CMAKE_SOURCE_DIR}/src/ext/yyjson/src
    ${CMAKE_SOURCE_DIR}/src/ext/stb
)

luisa_compute_add_executable(luisa_ntc_cli tools/luisa_ntc_cli.cpp)
target_link_libraries(luisa_ntc_cli PRIVATE luisa-ntc)
target_include_directories(luisa_ntc_cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/libntc
    ${CMAKE_SOURCE_DIR}/examples
    ${CMAKE_SOURCE_DIR}/src/ext/stb
)

luisa_compute_add_executable(luisa_ntc_bench tools/luisa_ntc_bench.cpp)
target_link_libraries(luisa_ntc_bench PRIVATE luisa-ntc)
target_include_directories(luisa_ntc_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/libntc
    ${CMAKE_SOURCE_DIR}/examples
)

luisa_compute_add_executable(luisa_ntc_test tools/luisa_ntc_test.cpp)
target_link_libraries(luisa_ntc_test PRIVATE luisa-ntc)
target_include_directories(luisa_ntc_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/libntc
    ${CMAKE_SOURCE_DIR}/examples
    ${CMAKE_SOURCE_DIR}/src/ext/stb
)

luisa_compute_add_executable(luisa_ntc_viewer tools/luisa_ntc_viewer.cpp)
target_link_libraries(luisa_ntc_viewer PRIVATE luisa-ntc)
if (LUISA_COMPUTE_ENABLE_GUI)
    target_link_libraries(luisa_ntc_viewer PRIVATE luisa-compute-gui)
    target_compile_definitions(luisa_ntc_viewer PRIVATE LUISA_ENABLE_GUI)
endif ()
target_include_directories(luisa_ntc_viewer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/libntc
    ${CMAKE_SOURCE_DIR}/examples
    ${CMAKE_SOURCE_DIR}/src/ext/stb
)
