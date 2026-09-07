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

set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(luisa-compute-include INTERFACE
            $<$<COMPILE_LANGUAGE:C,CXX>:/wd4068> # Maxwell loves custom pragmas
            $<$<COMPILE_LANGUAGE:C,CXX>:/wd5105> # winbase.h: macro expansion producing 'defined' has undefined behavior
            /Zc:preprocessor
            /Zc:__cplusplus)
elseif (MSVC AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(luisa-compute-include INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(luisa-compute-include INTERFACE
            $<$<OR:$<COMPILE_LANGUAGE:OBJCXX>,$<COMPILE_LANGUAGE:CXX>>:-Wno-invalid-offsetof>
            -Wno-pragmas)
endif ()

add_subdirectory(core)
add_subdirectory(vstl)
add_subdirectory(tile)
add_subdirectory(ast)
add_subdirectory(xir)
add_subdirectory(runtime)
add_subdirectory(dsl)
add_subdirectory(coro)
# add_subdirectory(tensor)
add_subdirectory(osl)
add_subdirectory(gui)
add_subdirectory(backends)
add_subdirectory(clangcxx)

add_library(compute INTERFACE)
target_compile_features(compute INTERFACE cxx_std_20)

target_link_libraries(compute INTERFACE
        luisa-compute-core
        luisa-compute-tile
        luisa-compute-ast
        luisa-compute-xir
        luisa-compute-dsl
        luisa-compute-osl
        luisa-compute-runtime
        luisa-compute-gui
        luisa-compute-backends
        luisa-compute-coro
        luisa-compute-clangcxx)

if (LUISA_COMPUTE_ENABLE_CUDA_EXT_LCUB)
    target_link_libraries(compute INTERFACE luisa-compute-cuda-ext-lcub)
endif ()

add_library(luisa::compute ALIAS compute)

function(luisa_compute_add_executable name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE luisa::compute)
endfunction()

if (LUISA_COMPUTE_BUILD_IOS_TESTS)
    add_subdirectory(tests/ios)
endif ()

if (LUISA_COMPUTE_BUILD_TESTS)
    enable_testing()
    message(STATUS "Building tests")
    add_subdirectory(tests)
endif ()

install(TARGETS compute
        EXPORT LuisaComputeTargets
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/LuisaCompute)

if (SKBUILD OR LUISA_COMPUTE_FORCE_PYTHON_BINDINGS)
    add_subdirectory(py)
endif ()

if (NOT SKBUILD AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    install(EXPORT LuisaComputeTargets
            FILE LuisaComputeTargets.cmake
            NAMESPACE luisa::
            DESTINATION ${LUISA_COMPUTE_INSTALL_CMAKEDIR})
endif ()
