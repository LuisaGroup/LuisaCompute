include_guard(GLOBAL)

if (NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    message(FATAL_ERROR
            "Luisa iOS application bundles require an iOS toolchain.")
endif ()
if (NOT LUISA_COMPUTE_ENABLE_GUI)
    message(FATAL_ERROR
            "Luisa iOS application bundles require LUISA_COMPUTE_ENABLE_GUI=ON.")
endif ()

set(LUISA_IOS_DEVELOPMENT_TEAM "" CACHE STRING
        "Apple development team used to sign LuisaCompute iOS applications")

function(luisa_compute_link_ios_metal4_application target)
    if (NOT TARGET luisa-compute-backend-metal4 OR
            NOT TARGET luisa-compute-metal4-air-codegen)
        message(FATAL_ERROR
                "Target '${target}' requires the complete Metal4 AIR backend.")
    endif ()
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_options(${target} PRIVATE -fblocks)
    target_include_directories(${target} PRIVATE
            ${CMAKE_SOURCE_DIR}/examples
            ${CMAKE_SOURCE_DIR}/src/backends/metal4
            ${CMAKE_SOURCE_DIR}/src/tests
            ${CMAKE_SOURCE_DIR}/src/tests/common)
    target_link_libraries(${target} PRIVATE
            luisa-compute-metal4-air-codegen
            luisa-compute-backend-metal4
            luisa-compute-dsl
            luisa-compute-gui
            luisa-compute-metal-cpp
            "-framework Foundation"
            "-framework UIKit"
            "-framework Metal"
            "-framework QuartzCore"
            "-framework CoreGraphics"
            "-framework ImageIO")
endfunction()

function(luisa_compute_link_ios_metal_application target)
    if (NOT TARGET luisa-compute-backend-metal)
        message(FATAL_ERROR
                "Target '${target}' requires the old Metal MSL backend.")
    endif ()
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_options(${target} PRIVATE -fblocks)
    target_include_directories(${target} PRIVATE
            ${CMAKE_SOURCE_DIR}/examples
            ${CMAKE_SOURCE_DIR}/src/backends/metal
            ${CMAKE_SOURCE_DIR}/src/tests
            ${CMAKE_SOURCE_DIR}/src/tests/common)
    target_link_libraries(${target} PRIVATE
            luisa-compute-backend-metal
            luisa-compute-dsl
            luisa-compute-gui
            luisa-compute-metal-cpp
            "-framework Foundation"
            "-framework UIKit"
            "-framework Metal"
            "-framework QuartzCore"
            "-framework CoreGraphics"
            "-framework ImageIO")
endfunction()

function(luisa_compute_configure_ios_bundle target)
    cmake_parse_arguments(
            IOS_BUNDLE "" "BUNDLE_ID;BUNDLE_NAME;FOLDER;INFO_PLIST" "" ${ARGN})
    foreach (argument BUNDLE_ID BUNDLE_NAME FOLDER INFO_PLIST)
        if (NOT IOS_BUNDLE_${argument})
            message(FATAL_ERROR
                    "Missing ${argument} while configuring iOS target '${target}'.")
        endif ()
    endforeach ()
    set_target_properties(${target} PROPERTIES
            FOLDER "${IOS_BUNDLE_FOLDER}"
            MACOSX_BUNDLE_INFO_PLIST "${IOS_BUNDLE_INFO_PLIST}"
            MACOSX_BUNDLE_BUNDLE_NAME "${IOS_BUNDLE_BUNDLE_NAME}"
            MACOSX_BUNDLE_GUI_IDENTIFIER "${IOS_BUNDLE_BUNDLE_ID}"
            MACOSX_BUNDLE_BUNDLE_VERSION "1"
            MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
            XCODE_GENERATE_SCHEME TRUE
            XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "${IOS_BUNDLE_BUNDLE_ID}"
            XCODE_ATTRIBUTE_CODE_SIGN_STYLE Automatic
            XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET
            "${CMAKE_OSX_DEPLOYMENT_TARGET}"
            XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1"
            XCODE_ATTRIBUTE_SUPPORTED_PLATFORMS "iphoneos"
            XCODE_ATTRIBUTE_SUPPORTS_MACCATALYST NO
            XCODE_ATTRIBUTE_ENABLE_BITCODE NO
            XCODE_ATTRIBUTE_DEAD_CODE_STRIPPING YES)
    if (LUISA_IOS_DEVELOPMENT_TEAM)
        set_target_properties(${target} PROPERTIES
                XCODE_ATTRIBUTE_DEVELOPMENT_TEAM
                "${LUISA_IOS_DEVELOPMENT_TEAM}")
    endif ()
endfunction()
