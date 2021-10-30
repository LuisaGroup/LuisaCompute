if (WIN32)
    set(LUISA_COMPUTE_ENABLE_MIMALLOC_DEFAULT ON)
else ()
    set(LUISA_COMPUTE_ENABLE_MIMALLOC_DEFAULT OFF)
endif ()
mark_as_advanced(LUISA_COMPUTE_ENABLE_MIMALLOC_DEFAULT)
option(LUISA_COMPUTE_ENABLE_MIMALLOC "Enable mimalloc" ${LUISA_COMPUTE_ENABLE_MIMALLOC_DEFAULT})

add_subdirectory(ext)

add_library(luisa-compute-include INTERFACE)
target_include_directories(luisa-compute-include INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(luisa-compute-include INTERFACE c_std_11 cxx_std_20)
if (MSVC)
    target_compile_definitions(luisa-compute-include INTERFACE
            NOMINMAX
            _ENABLE_EXTENDED_ALIGNED_STORAGE
            _CRT_SECURE_NO_WARNINGS
            _UNICODE)
endif ()

if (MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(luisa-compute-include INTERFACE
            /wd4068 # Maxwell loves custom pragma
            /wd5105 # winbase.h: macro expansion producing 'defined' has undefined behavior
            /Zc:preprocessor)
else ()
    target_compile_options(luisa-compute-include INTERFACE -Wno-unknown-attributes)
endif ()

add_subdirectory(core)
add_subdirectory(vstl)
add_subdirectory(meta)
add_subdirectory(ast)
add_subdirectory(runtime)
#add_subdirectory(serialize)
add_subdirectory(compile)
add_subdirectory(dsl)
add_subdirectory(rtx)
add_subdirectory(api)
add_subdirectory(backends)

add_library(luisa-compute INTERFACE)
target_link_libraries(luisa-compute INTERFACE
        luisa-compute-core
        luisa-compute-vstl
        luisa-compute-meta
        luisa-compute-ast
#        luisa-compute-serialize
        luisa-compute-compile
        luisa-compute-runtime
        luisa-compute-dsl
        luisa-compute-rtx
        luisa-compute-backends)

add_dependencies(luisa-compute luisa-compute-api)
add_library(luisa::compute ALIAS luisa-compute)

if (LUISA_COMPUTE_BUILD_TESTS)
    add_subdirectory(tests)
endif ()
