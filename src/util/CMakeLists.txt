set(LUISA_COMPUTE_UTIL_SOURCES
        arena.cpp arena.h
        dynamic_module.cpp dynamic_module.h
        spin_mutex.h)

list(APPEND LUISA_COMPUTE_UTIL_SOURCES
        AllocateType.h
        Hash.h
        HashMap.h
        LockFreeArrayQueue.h
        LockFreeStepQueue.h
        vstl_config.h
        VAllocator.h
        Memory.h
        MetaLib.h
        Pool.h
        vstl.cpp
        Log.cpp Log.h
        MD5.cpp MD5.h
        VGuid.cpp VGuid.h
        ObjectPtr.cpp ObjectPtr.h
        StackAllocator.cpp StackAllocator.h)

add_library(luisa-compute-util SHARED ${LUISA_COMPUTE_UTIL_SOURCES})
target_compile_definitions(luisa-compute-util PRIVATE LUISA_COMPUTE_UTIL_INTERNAL PUBLIC LUISA_COMPUTE_UTIL_EXTERNAL)
target_link_libraries(luisa-compute-util PUBLIC luisa-compute-core)
set_target_properties(luisa-compute-util PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ON)
