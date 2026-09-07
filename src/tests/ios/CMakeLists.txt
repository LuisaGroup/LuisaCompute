if (NOT APPLE OR NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR
            "The iOS Metal4 tests require Apple's Clang toolchain.")
endif ()
if (NOT TARGET luisa-compute-metal4-air-codegen OR
        NOT TARGET luisa-compute-backend-metal4)
    message(FATAL_ERROR
            "The iOS tests require the complete Metal4 AIR backend.")
endif ()

if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
    include(${CMAKE_SOURCE_DIR}/examples/ios/metal4_path_tracing/Target.cmake)
    luisa_compute_add_ios_metal4_path_tracing_bundle(
            luisa-metal4-ios-device-air-path-tracer
            BUNDLE_ID "org.luisa-compute.tests.ios.metal4-device-air"
            BUNDLE_NAME "Luisa Metal4 Device AIR"
            FOLDER "tests/ios")
    add_custom_target(luisa-ios-device-tests
            DEPENDS luisa-metal4-ios-device-air-path-tracer)
    set_target_properties(luisa-ios-device-tests PROPERTIES
            FOLDER "tests/ios")
else ()
    # Host-side iOS AIR container/ABI oracle. It is intentionally gated by
    # LUISA_COMPUTE_BUILD_IOS_TESTS even though it executes on macOS.
    add_executable(luisa-metal4-ios-path-tracer-aot
            metal4_ios_path_tracing_aot.cpp
            ${CMAKE_SOURCE_DIR}/src/backends/metal4/metal_api.cpp)
    target_include_directories(luisa-metal4-ios-path-tracer-aot PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src/backends/metal4)
    target_compile_definitions(luisa-metal4-ios-path-tracer-aot PRIVATE
            LUISA_METAL4_BACKEND=1)
    target_link_libraries(luisa-metal4-ios-path-tracer-aot PRIVATE
            luisa-compute-metal4-air-codegen
            luisa-compute-dsl
            luisa-compute-metal-cpp
            "-framework Foundation"
            "-framework Metal"
            "-framework QuartzCore")
    set_target_properties(luisa-metal4-ios-path-tracer-aot PROPERTIES
            FOLDER tests/ios)
endif ()
