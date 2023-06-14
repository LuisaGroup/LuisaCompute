set(LUISA_COMPUTE_CORE_STL_SOURCES
        stl/filesystem.cpp
        stl/hash.cpp
        stl/memory.cpp)

set(LUISA_COMPUTE_CORE_SOURCES
        basic_types.cpp
        binary_buffer.cpp
        binary_file_stream.cpp
        dynamic_module.cpp
        first_fit.cpp
        logging.cpp
        platform.cpp
        pool.cpp
        thread_pool.cpp)

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
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
        OUTPUT_NAME lc-core)
add_dependencies(luisa-compute-core luisa-compute-ext-copy)

luisa_compute_install(core SOURCES ${LUISA_COMPUTE_CORE_SOURCES})
