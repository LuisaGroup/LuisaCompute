set(LUISA_COMPUTE_UTIL_SOURCES
        Hash.h
        HashMap.h
        LockFreeArrayQueue.h
        LockFreeStepQueue.h
        config.h
        VAllocator.h
        Memory.h
        MetaLib.h
        Pool.h
        vstl.cpp
        Log.cpp Log.h
        MD5.cpp MD5.h
        VGuid.cpp VGuid.h
        ObjectPtr.cpp ObjectPtr.h)

add_library(luisa-compute-vstl SHARED ${LUISA_COMPUTE_UTIL_SOURCES})
target_compile_definitions(luisa-compute-vstl PUBLIC VSTL_ENABLE_MIMALLOC)
target_link_libraries(luisa-compute-vstl PUBLIC luisa-compute-core PRIVATE mimalloc-static)
set_target_properties(luisa-compute-vstl PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ON)
