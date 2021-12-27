option(LUISA_COMPUTE_ENABLE_GUI "Enable GUI support" ON)

if (WIN32)
    add_compile_definitions(
            NOMINMAX=1
            _ENABLE_EXTENDED_ALIGNED_STORAGE=1
            _CRT_SECURE_NO_WARNINGS=1
            _UNICODE=1)
    set(CMAKE_SHARED_MODULE_PREFIX "") # for MinGW/Cygwin etc.
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        add_compile_options(/MP)
    endif ()
endif ()

add_subdirectory(ext)

add_library(luisa-compute-include INTERFACE)
target_include_directories(luisa-compute-include INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(luisa-compute-include INTERFACE c_std_11 cxx_std_20)

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(luisa-compute-include INTERFACE
            /wd4068 # Maxwell loves custom pragmas
            /wd5105 # winbase.h: macro expansion producing 'defined' has undefined behavior
            /Zc:preprocessor)
endif ()

add_subdirectory(core)
add_subdirectory(vstl)
add_subdirectory(meta)
add_subdirectory(ast)
add_subdirectory(runtime)
add_subdirectory(serialize)
add_subdirectory(compile)
add_subdirectory(dsl)
add_subdirectory(rtx)
add_subdirectory(api)
add_subdirectory(serde)
add_subdirectory(gui)
add_subdirectory(backends)

add_library(luisa-compute INTERFACE luisa-compute.h)
target_link_libraries(luisa-compute INTERFACE
        luisa-compute-core
        luisa-compute-vstl
        luisa-compute-meta
        luisa-compute-ast
        luisa-compute-compile
        luisa-compute-runtime
        luisa-compute-dsl
        luisa-compute-rtx
        luisa-compute-gui
        luisa-compute-backends)

add_dependencies(luisa-compute
        luisa-compute-api
        luisa-compute-serialize
        luisa-compute-serde)
add_library(luisa::compute ALIAS luisa-compute)

if (LUISA_COMPUTE_BUILD_TESTS)
    add_subdirectory(tests)
endif ()
