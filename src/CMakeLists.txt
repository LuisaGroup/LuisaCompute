if (WIN32) # mimalloc seems not beneficial on other platforms
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
if (WIN32)
    target_compile_definitions(luisa-compute-include INTERFACE
            NOMINMAX=1
            _ENABLE_EXTENDED_ALIGNED_STORAGE=1
            _CRT_SECURE_NO_WARNINGS=1
            _UNICODE=1)
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
add_subdirectory(serde)
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
        luisa-compute-serde
        luisa-compute-backends)

add_dependencies(luisa-compute luisa-compute-api)
add_library(luisa::compute ALIAS luisa-compute)

if (LUISA_COMPUTE_BUILD_TESTS)
    add_subdirectory(tests)
endif ()
