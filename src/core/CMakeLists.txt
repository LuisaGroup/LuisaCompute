set(LUISA_COMPUTE_CORE_STL_SOURCES
        stl/deque.h
        stl/filesystem.cpp stl/filesystem.h
        stl/fixed_map.h
        stl/format.h
        stl/functional.h
        stl/hash.cpp stl/hash.h
        stl/hash_fwd.h
        stl/list.h
        stl/lru_cache.h
        stl/map.h
        stl/memory.cpp
        stl/memory.h
        stl/optional.h
        stl/priority_queue.h
        stl/queue.h
        stl/ring_buffer.h
        stl/stack.h
        stl/string.h
        stl/unordered_map.h
        stl/variant.h
        stl/vector.h
        stl/vector_map.h)


set(LUISA_COMPUTE_CORE_SOURCES
        basic_traits.h
        basic_types.cpp basic_types.h
        binary_buffer.cpp binary_buffer.h
        binary_io.h
        clock.h
        concepts.h
        constants.h
        dirty_range.cpp dirty_range.h
        dll_export.h
        dynamic_module.cpp dynamic_module.h
        first_fit.cpp first_fit.h
        intrin.h
        json.h
        json_fwd.h
        logging.cpp logging.h
        macro.h
        mathematics.h
        observer.cpp observer.h
        platform.cpp platform.h
        pool.h
        spin_mutex.h
        stl.h
        swizzle_2.inl.h
        swizzle_3.inl.h
        swizzle_4.inl.h
        thread_pool.cpp thread_pool.h
        thread_safety.h)

find_package(Threads REQUIRED)
add_library(luisa-compute-core SHARED ${LUISA_COMPUTE_CORE_STL_SOURCES} ${LUISA_COMPUTE_CORE_SOURCES})
target_compile_definitions(luisa-compute-core PRIVATE LC_CORE_EXPORT_DLL=1)

target_link_libraries(luisa-compute-core PUBLIC
        luisa-compute-include
        luisa-compute-ext
        Threads::Threads
        ${CMAKE_DL_LIBS}
        $<$<PLATFORM_ID:Windows>:dbghelp>)

set_target_properties(luisa-compute-core PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD})
add_dependencies(luisa-compute-core luisa-compute-ext-copy)

luisa_compute_install(core SOURCES ${LUISA_COMPUTE_CORE_SOURCES})
