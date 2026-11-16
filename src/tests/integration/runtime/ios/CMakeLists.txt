# Each executable pulls a backend's registration entry point into the link,
# checking its full runtime dependency closure (including AIR/LLVM for Metal4).
# These unsigned cross-compiled binaries are never executed by host CTest.
set(_luisa_ios_link_targets)
foreach (backend IN ITEMS metal metal4)
    if (TARGET luisa-compute-backend-${backend})
        set(target test_ios_${backend}_link)
        add_executable(${target} MACOSX_BUNDLE backend_link.cpp)
        target_include_directories(${target} PRIVATE
                ${PROJECT_SOURCE_DIR}/src/backends/${backend})
        target_compile_definitions(${target} PRIVATE
                LUISA_IOS_REGISTER_BACKEND=luisa_compute_${backend}_register_static_backend)
        target_link_libraries(${target} PRIVATE
                luisa::compute luisa-compute-backend-${backend})
        set_target_properties(${target} PROPERTIES
                MACOSX_BUNDLE_GUI_IDENTIFIER "org.luisa-compute.tests.ios.${backend}-link"
                MACOSX_BUNDLE_BUNDLE_VERSION "1"
                MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
                XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED NO
                FOLDER tests/ios)
        list(APPEND _luisa_ios_link_targets ${target})
    endif ()
endforeach ()
if (NOT _luisa_ios_link_targets)
    message(FATAL_ERROR "iOS link tests require a Metal or Metal4 runtime backend.")
endif ()
add_custom_target(luisa-ios-link-tests DEPENDS ${_luisa_ios_link_targets})
