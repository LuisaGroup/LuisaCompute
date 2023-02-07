if (WIN32)
    add_compile_definitions(
            NOMINMAX=1
            _ENABLE_EXTENDED_ALIGNED_STORAGE=1
            _CRT_SECURE_NO_WARNINGS=1
            _UNICODE=1
            WIN32_LEAN_AND_MEAN=1)
    set(CMAKE_SHARED_MODULE_PREFIX "") # for MinGW/Cygwin etc.
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        add_compile_options(/wd4005)
    endif ()
endif ()

function(luisa_compute_install target)
    cmake_parse_arguments(INSTALL "" "" "SOURCES" ${ARGN})
    if (INSTALL_SOURCES)
        set(INSTALL_HEADERS ${INSTALL_SOURCES})
        list(FILTER INSTALL_HEADERS INCLUDE REGEX "^.*\\.h$")
        set_target_properties(luisa-compute-${target} PROPERTIES PUBLIC_HEADER "${INSTALL_HEADERS}")
    endif ()
    install(TARGETS luisa-compute-${target} EXPORT LuisaComputeTargets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/LuisaCompute/${target})
endfunction()

add_subdirectory(ext)

add_library(luisa-compute-include INTERFACE)
target_include_directories(luisa-compute-include INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_compile_features(luisa-compute-include INTERFACE c_std_11 cxx_std_20)

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(luisa-compute-include INTERFACE
            /wd4068 # Maxwell loves custom pragmas
            /wd5105 # winbase.h: macro expansion producing 'defined' has undefined behavior
            /Zc:preprocessor)
elseif (MSVC AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(luisa-compute-include INTERFACE /EHsc)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(luisa-compute-include INTERFACE
            -Wno-invalid-offsetof
            -Wno-pragmas)
endif ()

install(TARGETS luisa-compute-include
        EXPORT LuisaComputeTargets)

add_subdirectory(api)
add_subdirectory(core)
add_subdirectory(vstl)
add_subdirectory(ast)
add_subdirectory(rust)
add_subdirectory(ir)
add_subdirectory(runtime)
add_subdirectory(dsl)
add_subdirectory(rtx)
add_subdirectory(gui/backup)
add_subdirectory(backends)

add_library(compute INTERFACE luisa-compute.h)
set_target_properties(compute PROPERTIES PUBLIC_HEADER "luisa-compute.h")
target_link_libraries(compute INTERFACE
        luisa-compute-core
        luisa-compute-vstl
        luisa-compute-ast
        luisa-compute-runtime
        luisa-compute-dsl
        luisa-compute-rtx
        luisa-compute-gui
        luisa-compute-api
        luisa-compute-backends
        # luisa-compute-ir
        )
add_library(luisa::compute ALIAS compute)

if (LUISA_COMPUTE_ENABLE_PYTHON)
    add_subdirectory(py)
endif ()

if (LUISA_COMPUTE_BUILD_TESTS)
    add_subdirectory(tests)
endif ()

install(TARGETS compute
        EXPORT LuisaComputeTargets
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/LuisaCompute)

install(EXPORT LuisaComputeTargets
        FILE LuisaComputeTargets.cmake
        NAMESPACE luisa::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LuisaCompute)
