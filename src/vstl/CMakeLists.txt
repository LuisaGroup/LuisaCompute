set(LUISA_COMPUTE_UTIL_SOURCES
        ObjectPtr.cpp
        StringUtility.cpp
        xxhash.cpp
        Log.cpp
        MD5.cpp
        VGuid.cpp
        ThreadPool.cpp
        ThreadTaskHandle.cpp
        file_system.cpp
        vstl.cpp)

add_library(luisa-compute-vstl SHARED ${LUISA_COMPUTE_UTIL_SOURCES})
target_link_libraries(luisa-compute-vstl PUBLIC luisa-compute-core)
set_target_properties(luisa-compute-vstl PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ON)

if (WIN32)
    target_link_libraries(luisa-compute-vstl PRIVATE Shlwapi)
elseif (APPLE)
    target_link_libraries(luisa-compute-vstl PUBLIC "-framework CoreFoundation")
else ()
    find_library(LibUUID_LIBRARY NAMES uuid)
    find_path(LibUUID_INCLUDE_DIR NAMES uuid/uuid.h)
    target_include_directories(luisa-compute-vstl PUBLIC "${LibUUID_INCLUDE_DIR}")
    target_link_libraries(luisa-compute-vstl PUBLIC "${LibUUID_LIBRARY}")
endif ()

