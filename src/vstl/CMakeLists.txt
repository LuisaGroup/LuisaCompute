set(LUISA_COMPUTE_UTIL_SOURCES
allocate_type.h
binary_reader.h
common.h
common_iterator.h
compare.h
config.h
executor.h
functional.h
hash.h
hash_map.h
lockfree_array_queue.h
lockfree_step_queue.h
log.h
md5.h
memory.h
meta_lib.h
string_builder.h
pool.h
pool_allocator.h
serializer.h
span.h
spin_mutex.h
stack_allocator.h
string_utility.h
string_hash.h
tree_map.h
tree_map_base.h
unique_ptr.h
v_allocator.h
variant_util.h
vector.h
v_guid.h
vstring.h
StringUtility.cpp
Log.cpp
MD5.cpp
VGuid.cpp
StackAllocator.cpp
BinaryReader.cpp
PoolAllocator.cpp
string_builder.cpp
vstl.cpp)

add_library(luisa-compute-vstl SHARED ${LUISA_COMPUTE_UTIL_SOURCES})
target_link_libraries(luisa-compute-vstl PUBLIC luisa-compute-core)
target_compile_definitions(luisa-compute-vstl PRIVATE LC_VSTL_EXPORT_DLL=1)
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
