set(LUISA_COMPUTE_UTIL_SOURCES
AllocateType.h
BinaryReader.h
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
Log.h
MD5.h
Memory.h
MetaLib.h
Pool.h
PoolAllocator.h
Serializer.h
span.h
spin_mutex.h
StackAllocator.h
StringUtility.h
string_hash.h
ThreadPool.h
ThreadTaskHandle.h
TreeMap.h
TreeMapBase.h
unique_ptr.h
VAllocator.h
variant_util.h
vector.h
VGuid.h
vstring.h
StringUtility.cpp
Log.cpp
MD5.cpp
VGuid.cpp
ThreadPool.cpp
ThreadTaskHandle.cpp
StackAllocator.cpp
BinaryReader.cpp
PoolAllocator.cpp
vstl.cpp)

add_library(luisa-compute-vstl SHARED ${LUISA_COMPUTE_UTIL_SOURCES})
target_link_libraries(luisa-compute-vstl PUBLIC luisa-compute-core)
set_target_properties(luisa-compute-vstl PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ON)

if (WIN32)
    target_link_libraries(luisa-compute-vstl PUBLIC Shlwapi)
elseif (APPLE)
    target_link_options(luisa-compute-vstl PUBLIC "-F${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/")
    target_link_libraries(luisa-compute-vstl PUBLIC "-framework CoreFoundation")
else ()
    find_library(LibUUID_LIBRARY NAMES uuid REQUIRED)
    find_path(LibUUID_INCLUDE_DIR NAMES uuid/uuid.h REQUIRED)
    target_include_directories(luisa-compute-vstl PUBLIC "${LibUUID_INCLUDE_DIR}")
    target_link_libraries(luisa-compute-vstl PUBLIC "${LibUUID_LIBRARY}")
endif ()
