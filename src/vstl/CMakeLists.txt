set(LUISA_COMPUTE_VSTL_SOURCES
        allocate_type.h
        array.h
        binary_reader.cpp binary_reader.h
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
        log.cpp log.h
        md5.cpp md5.h
        memory.h
        meta_lib.h
        pdqsort.h
        pool.h
        pool_allocator.cpp pool_allocator.h
        ranges.h
        serializer.h
        small_vector.h
        span.h
        spin_mutex.h
        stack_allocator.cpp stack_allocator.h
        string_builder.cpp string_builder.h
        string_hash.h
        string_utility.cpp string_utility.h
        tree_map_base.h
        tuple.h
        unique_ptr.h
        v_allocator.h
        v_guid.cpp v_guid.h
        variant_util.h
        vector.h
        vstl.cpp
        vstring.h)

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
