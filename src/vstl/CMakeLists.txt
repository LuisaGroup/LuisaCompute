set(LUISA_COMPUTE_VSTL_SOURCES
        log.cpp
        md5.cpp
        stack_allocator.cpp
        string_utility.cpp
        v_guid.cpp
        vstl.cpp)

add_library(luisa-compute-vstl SHARED ${LUISA_COMPUTE_VSTL_SOURCES})
target_link_libraries(luisa-compute-vstl PUBLIC luisa-compute-core)
target_compile_definitions(luisa-compute-vstl PRIVATE LC_VSTL_EXPORT_DLL=1)
set_target_properties(luisa-compute-vstl PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
        OUTPUT_NAME lc-vstl)

if (WIN32)
    target_link_libraries(luisa-compute-vstl PUBLIC Shlwapi)
elseif (APPLE)
    target_link_libraries(luisa-compute-vstl PUBLIC "-framework CoreFoundation")
else ()
    find_library(LibUUID_LIBRARY NAMES uuid REQUIRED)
    find_path(LibUUID_INCLUDE_DIR NAMES uuid/uuid.h REQUIRED)
    target_include_directories(luisa-compute-vstl PUBLIC "${LibUUID_INCLUDE_DIR}")
    target_link_libraries(luisa-compute-vstl PUBLIC "${LibUUID_LIBRARY}")
endif ()

luisa_compute_install(vstl SOURCES ${LUISA_COMPUTE_VSTL_SOURCES})
target_precompile_headers(luisa-compute-vstl PRIVATE pch.h)
