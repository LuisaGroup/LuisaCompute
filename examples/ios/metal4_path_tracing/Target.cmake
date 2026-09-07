include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/LuisaIOSBundle.cmake)

set(LUISA_IOS_PATH_TRACING_SPP 8 CACHE STRING
        "Samples per pixel rendered by the iOS Metal4 conformance preflight")
set(LUISA_IOS_INTERACTIVE_SNAPSHOT_SPP 64 CACHE STRING
        "Progressive sample count used for iOS path-tracing evidence snapshots")

execute_process(
        COMMAND xcrun --sdk iphoneos --show-sdk-version
        OUTPUT_VARIABLE LUISA_IOS_AIR_SDK_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE LUISA_IOS_AIR_SDK_VERSION_RESULT)
if (NOT LUISA_IOS_AIR_SDK_VERSION_RESULT EQUAL 0 OR
        LUISA_IOS_AIR_SDK_VERSION STREQUAL "")
    message(FATAL_ERROR
            "Failed to determine the installed iPhoneOS SDK version.")
endif ()

function(luisa_compute_add_ios_metal4_path_tracing_bundle target)
    cmake_parse_arguments(
            IOS_PT "" "BUNDLE_ID;BUNDLE_NAME;FOLDER" "" ${ARGN})
    foreach (argument BUNDLE_ID BUNDLE_NAME FOLDER)
        if (NOT IOS_PT_${argument})
            message(FATAL_ERROR
                    "Missing ${argument} while configuring iOS path tracer '${target}'.")
        endif ()
    endforeach ()

    add_executable(${target} MACOSX_BUNDLE
            ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/main.mm
            ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/Info.plist.in
            ${CMAKE_SOURCE_DIR}/src/tests/ios/metal4_device_conformance.cpp
            ${CMAKE_SOURCE_DIR}/src/tests/ios/metal4_device_conformance.h
            ${CMAKE_SOURCE_DIR}/src/tests/ios/metal4_ios_path_tracing_kernel.h
            ${CMAKE_SOURCE_DIR}/examples/rendering/path_tracing.cpp
            ${CMAKE_SOURCE_DIR}/examples/rendering/path_tracing_test.h)
    luisa_compute_link_ios_metal4_application(${target})
    target_include_directories(${target} PRIVATE
            ${CMAKE_SOURCE_DIR}/src/tests/ios)
    target_compile_definitions(${target} PRIVATE
            LUISA_METAL4_BACKEND=1
            LUISA_METAL_AIR_SDK_VERSION="${LUISA_IOS_AIR_SDK_VERSION}"
            LUISA_IOS_ON_DEVICE_AIR=1
            LUISA_IOS_RUNTIME_DEVICE=1
            LUISA_IOS_PATH_TRACING_SPP=${LUISA_IOS_PATH_TRACING_SPP}
            LUISA_IOS_INTERACTIVE_SNAPSHOT_SPP=${LUISA_IOS_INTERACTIVE_SNAPSHOT_SPP}
            LUISA_PATH_TRACING_LIBRARY_ONLY=1
            LUISA_IOS_AIR_DEPLOYMENT_VERSION="${CMAKE_OSX_DEPLOYMENT_TARGET}"
            LUISA_IOS_AIR_SDK_VERSION="${LUISA_IOS_AIR_SDK_VERSION}")
    luisa_compute_configure_ios_bundle(${target}
            BUNDLE_ID "${IOS_PT_BUNDLE_ID}"
            BUNDLE_NAME "${IOS_PT_BUNDLE_NAME}"
            FOLDER "${IOS_PT_FOLDER}"
            INFO_PLIST "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/Info.plist.in")
endfunction()
