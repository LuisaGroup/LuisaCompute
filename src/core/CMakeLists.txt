set(LUISA_COMPUTE_CORE_SOURCES
        basic_traits.h
        basic_types.cpp basic_types.h
        binary_buffer.h
        clock.h
        concepts.h
        constants.h
        dirty_range.cpp dirty_range.h
        dll_export.h
        dynamic_module.cpp dynamic_module.h
        first_fit.cpp first_fit.h
        hash.h
        intrin.h
        json.h
        json_fwd.h
        logging.cpp logging.h
        lru_cache.h
        macro.h
        mathematics.h
        observer.cpp observer.h
        platform.cpp platform.h
        pool.h
        spin_mutex.h
        stl.cpp stl.h
        swizzle_2.inl.h
        swizzle_3.inl.h
        swizzle_4.inl.h
        thread_pool.cpp thread_pool.h
        thread_safety.h)

find_package(Threads REQUIRED)
add_library(luisa-compute-core SHARED ${LUISA_COMPUTE_CORE_SOURCES})
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

if (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(luisa-compute-core INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-stdlib=libc++>)
    target_link_options(luisa-compute-core INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-stdlib=libc++>)
endif ()

luisa_compute_install(core SOURCES ${LUISA_COMPUTE_CORE_SOURCES})
