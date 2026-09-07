include(${CMAKE_CURRENT_LIST_DIR}/../cmake/LuisaIOSBundle.cmake)

string(TOLOWER "${LUISA_COMPUTE_IOS_BENCHMARK_BACKEND}"
        _LUISA_IOS_BENCHMARK_BACKEND)
set(_LUISA_IOS_BENCHMARK_TARGET
        example_ios_path_tracing_benchmark)

add_executable(${_LUISA_IOS_BENCHMARK_TARGET} MACOSX_BUNDLE
        main.mm
        ${CMAKE_CURRENT_LIST_DIR}/../common/Info.plist.in
        ${CMAKE_SOURCE_DIR}/examples/rendering/path_tracing.cpp
        ${CMAKE_SOURCE_DIR}/examples/rendering/path_tracing_test.h)
target_compile_definitions(${_LUISA_IOS_BENCHMARK_TARGET} PRIVATE
        LUISA_PATH_TRACING_LIBRARY_ONLY=1
        LUISA_IOS_BENCHMARK_BACKEND_NAME="${_LUISA_IOS_BENCHMARK_BACKEND}")

if (_LUISA_IOS_BENCHMARK_BACKEND STREQUAL "metal4")
    target_compile_definitions(${_LUISA_IOS_BENCHMARK_TARGET} PRIVATE
            LUISA_IOS_BENCHMARK_METAL4=1)
    luisa_compute_link_ios_metal4_application(
            ${_LUISA_IOS_BENCHMARK_TARGET})
    set(_LUISA_IOS_BENCHMARK_BUNDLE_ID
            org.luisa-compute.benchmark.ios.metal4)
    set(_LUISA_IOS_BENCHMARK_BUNDLE_NAME
            "Luisa Metal4 Benchmark")
else ()
    target_compile_definitions(${_LUISA_IOS_BENCHMARK_TARGET} PRIVATE
            LUISA_IOS_BENCHMARK_METAL=1)
    luisa_compute_link_ios_metal_application(
            ${_LUISA_IOS_BENCHMARK_TARGET})
    set(_LUISA_IOS_BENCHMARK_BUNDLE_ID
            org.luisa-compute.benchmark.ios.metal)
    set(_LUISA_IOS_BENCHMARK_BUNDLE_NAME
            "Luisa Metal Benchmark")
endif ()

if (LUISA_COMPUTE_IOS_BENCHMARK_BUNDLE_ID)
    set(_LUISA_IOS_BENCHMARK_BUNDLE_ID
            "${LUISA_COMPUTE_IOS_BENCHMARK_BUNDLE_ID}")
endif ()

luisa_compute_configure_ios_bundle(${_LUISA_IOS_BENCHMARK_TARGET}
        BUNDLE_ID "${_LUISA_IOS_BENCHMARK_BUNDLE_ID}"
        BUNDLE_NAME "${_LUISA_IOS_BENCHMARK_BUNDLE_NAME}"
        FOLDER "examples/ios/benchmark"
        INFO_PLIST "${CMAKE_CURRENT_LIST_DIR}/../common/Info.plist.in")
