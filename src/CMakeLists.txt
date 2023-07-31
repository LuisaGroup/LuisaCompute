if (WIN32)
    add_compile_definitions(
            NOMINMAX=1
            _ENABLE_EXTENDED_ALIGNED_STORAGE=1
            _CRT_SECURE_NO_WARNINGS=1
            _UNICODE=1
            WIN32_LEAN_AND_MEAN=1)
    set(CMAKE_SHARED_MODULE_PREFIX "") # for MinGW/Cygwin etc.
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/wd4005>)
    endif ()
endif ()

function(luisa_compute_install target)
    cmake_parse_arguments(INSTALL "" "" "SOURCES" ${ARGN})
    install(TARGETS luisa-compute-${target} EXPORT LuisaComputeTargets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction()

add_subdirectory(ext)

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(luisa-compute-include INTERFACE
            $<$<COMPILE_LANGUAGE:C,CXX>:/wd4068> # Maxwell loves custom pragmas
            $<$<COMPILE_LANGUAGE:C,CXX>:/wd5105> # winbase.h: macro expansion producing 'defined' has undefined behavior
            /Zc:preprocessor
            /Zc:__cplusplus)
elseif (MSVC AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(luisa-compute-include INTERFACE $<$<COMPILE_LANGUAGE:C,CXX>:/EHsc>)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(luisa-compute-include INTERFACE
            -Wno-invalid-offsetof
            -Wno-pragmas)
endif ()

add_subdirectory(api)
add_subdirectory(core)
add_subdirectory(vstl)
add_subdirectory(ast)
add_subdirectory(rust)
add_subdirectory(runtime)
add_subdirectory(ir)
add_subdirectory(dsl)
add_subdirectory(tensor)
add_subdirectory(osl)
add_subdirectory(gui)
add_subdirectory(backends)

add_library(compute INTERFACE)

target_link_libraries(compute INTERFACE
        luisa-compute-core
        luisa-compute-vstl
        luisa-compute-ast
        luisa-compute-dsl
        luisa-compute-osl
        luisa-compute-runtime
        luisa-compute-gui
        luisa-compute-api
        luisa-compute-backends
        luisa-compute-ir)

if (LUISA_COMPUTE_ENABLE_CUDA_EXT_LCUB)
    target_link_libraries(compute INTERFACE luisa-compute-cuda-ext-lcub)
endif ()

add_library(luisa::compute ALIAS compute)

function(luisa_compute_add_executable name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE luisa::compute)
    install(TARGETS ${name} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction()

if (LUISA_COMPUTE_BUILD_TESTS)
    add_subdirectory(tests)
endif ()

install(TARGETS compute
        EXPORT LuisaComputeTargets
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/LuisaCompute)

if (SKBUILD)
    add_subdirectory(py)
endif ()

if (NOT SKBUILD)
    install(EXPORT LuisaComputeTargets
            FILE LuisaComputeTargets.cmake
            NAMESPACE luisa::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LuisaCompute)
endif ()
