set(LUISA_COMPUTE_VSTL_SOURCES
        AllocateType.h
        BinaryReader.cpp BinaryReader.h
        Common.h
        CommonIterator.h
        Compare.h
        config.h
        Executor.h
        functional.h
        Hash.h
        HashMap.h
        LockFreeArrayQueue.h
        LockFreeStepQueue.h
        Log.cpp Log.h
        MD5.cpp MD5.h
        Memory.h
        MetaLib.h
        Pool.h
        PoolAllocator.cpp PoolAllocator.h
        Serializer.h
        span.h
        spin_mutex.h
        StackAllocator.cpp StackAllocator.h
        string_hash.h
        StringUtility.cpp StringUtility.h
        ThreadPool.cpp ThreadPool.h
        ThreadTaskHandle.cpp ThreadTaskHandle.h
        TreeMap.h
        TreeMapBase.h
        unique_ptr.h
        VAllocator.h
        variant_util.h
        vector.h
        VGuid.cpp VGuid.h
        vstl.cpp vstring.h)

add_library(luisa-compute-vstl SHARED ${LUISA_COMPUTE_VSTL_SOURCES})
target_link_libraries(luisa-compute-vstl PUBLIC luisa-compute-core)
target_compile_definitions(luisa-compute-vstl PRIVATE LC_VSTL_EXPORT_DLL=1)
set_target_properties(luisa-compute-vstl PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD})

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
