set(LUISA_COMPUTE_CORE_SOURCES
        logging.cpp
        logging.h
        concepts.h
        hash.cpp hash.h
        macro.h
        platform.cpp platform.h
        mathematics.h
        atomic.h
        basic_types.cpp basic_types.h
        intrin.h
        clock.h
        uuid.h basic_traits.h)

find_package(Threads REQUIRED)

add_library(luisa-compute-core SHARED ${LUISA_COMPUTE_CORE_SOURCES})
target_compile_definitions(luisa-compute-core PRIVATE LUISA_COMPUTE_CORE_INTERNAL PUBLIC LUISA_COMPUTE_CORE_EXTERNAL)
target_link_libraries(luisa-compute-core PUBLIC
        luisa-compute-include
        luisa-compute-ext
        Threads::Threads
        ${CMAKE_DL_LIBS}
        $<$<PLATFORM_ID:Windows>:dbghelp>)
set_target_properties(luisa-compute-core PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ON)
