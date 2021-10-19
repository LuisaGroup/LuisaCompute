set(LUISA_COMPUTE_UTIL_SOURCES
        ObjectPtr.cpp
        stringUtility.cpp
        xxhash.cpp
        Log.cpp
        MD5.cpp
        VGuid.cpp
        vstl.cpp)

add_library(luisa-compute-vstl SHARED ${LUISA_COMPUTE_UTIL_SOURCES})
target_link_libraries(luisa-compute-vstl PUBLIC luisa-compute-core)
set_target_properties(luisa-compute-vstl PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ON)
