set(LUISA_COMPUTE_CORE_SOURCES
        logging.cpp
        logging.h
        concepts.h
        hash.cpp hash.h
        macro.h
        spin_mutex.h
        platform.cpp platform.h
        mathematics.h
        atomic.h
        basic_traits.h
        basic_types.cpp basic_types.h
        dynamic_module.cpp dynamic_module.h
        intrin.h
        clock.h
        allocator.h allocator.cpp
        rc.h
        pool.h)

find_package(Threads REQUIRED)

add_library(luisa-compute-core SHARED ${LUISA_COMPUTE_CORE_SOURCES})
target_link_libraries(luisa-compute-core PUBLIC
        luisa-compute-include
        luisa-compute-ext
        Threads::Threads
        ${CMAKE_DL_LIBS}
        $<$<PLATFORM_ID:Windows>:dbghelp>)
set_target_properties(luisa-compute-core PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ON)

if (LUISA_COMPUTE_ENABLE_MIMALLOC)
    message(STATUS "Build with mimalloc")
    target_link_libraries(luisa-compute-core PRIVATE mimalloc-static)
    target_compile_definitions(luisa-compute-core PUBLIC LUISA_ENABLE_MIMALLOC=1)
endif ()
