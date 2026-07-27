set(LUISA_COMPUTE_CORE_STL_SOURCES
        stl/filesystem.cpp
        stl/hash.cpp
        stl/memory.cpp)

set(LUISA_COMPUTE_CORE_SOURCES
        basic_types.cpp
        binary_file_stream.cpp
        binary_io.cpp
        dynamic_module.cpp
        first_fit.cpp
        logging.cpp
        platform.cpp
        pool.cpp
        string_scratch.cpp)

find_package(Threads REQUIRED)
add_library(luisa-compute-core SHARED ${LUISA_COMPUTE_CORE_STL_SOURCES} ${LUISA_COMPUTE_CORE_SOURCES})
target_compile_definitions(luisa-compute-core PRIVATE LUISA_CORE_EXPORT_DLL=1)
if (WIN32 AND LUISA_COMPUTE_DISABLE_WIN_MESSAGE_BOX)
    target_compile_definitions(luisa-compute-core PRIVATE $<$<CONFIG:Debug>:LUISA_DISABLE_WIN_MESSAGE_BOX>)
endif ()

target_link_libraries(luisa-compute-core PUBLIC
        luisa-compute-include
        luisa-compute-ext
        Threads::Threads
        ${CMAKE_DL_LIBS}
        $<$<PLATFORM_ID:Windows>:dbghelp>)

if (NOT LUISA_COMPUTE_USE_SYSTEM_MARL)
    target_link_libraries(luisa-compute-core PRIVATE luisa-compute-ext-marl)
endif ()

if (NOT LUISA_COMPUTE_USE_SYSTEM_YYJSON)
    target_link_libraries(luisa-compute-core
            PRIVATE luisa-compute-ext-yyjson
            PUBLIC luisa-compute-ext-yyjson-interface)
endif ()

if (NOT LUISA_COMPUTE_USE_SYSTEM_STL)
    target_link_libraries(luisa-compute-core PRIVATE EASTL-object)
endif ()

set_target_properties(luisa-compute-core PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
        OUTPUT_NAME luisa-core)
add_dependencies(luisa-compute-core luisa-compute-ext-copy)

luisa_compute_install(core SOURCES ${LUISA_COMPUTE_CORE_SOURCES})
