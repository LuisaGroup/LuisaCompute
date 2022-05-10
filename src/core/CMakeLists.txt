set(LUISA_COMPUTE_CORE_SOURCES
        logging.cpp
        logging.h
        concepts.h
        hash.h
        macro.h
        spin_mutex.h
        platform.cpp platform.h
        mathematics.h
        basic_traits.h
        basic_types.cpp basic_types.h
        dynamic_module.cpp dynamic_module.h
        intrin.h
        clock.h
        stl.h stl.cpp
        pool.h
        constants.h
        lru_cache.h
        dirty_range.cpp dirty_range.h
        first_fit.cpp first_fit.h
        observer.cpp observer.h
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

if (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(luisa-compute-core INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-stdlib=libc++>)
    target_link_options(luisa-compute-core INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-stdlib=libc++>)
endif ()
