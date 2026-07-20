if (LUISA_COMPUTE_ENABLE_CLANG_CXX)
    message(STATUS "Build with Clang C++ support")
    file(GLOB_RECURSE LUISA_CLANGCXX_SOURCES "src/*.cpp")
    find_package(Clang REQUIRED CONFIG)
    add_library(luisa-compute-clangcxx SHARED ${LUISA_CLANGCXX_SOURCES})
    target_link_libraries(luisa-compute-clangcxx PUBLIC
            luisa-compute-ast luisa-compute-runtime)
    set_target_properties(luisa-compute-clangcxx PROPERTIES
            UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
            OUTPUT_NAME "luisa-clangcxx")
    target_compile_definitions(luisa-compute-clangcxx PRIVATE
            LUISA_CLANGCXX_EXPORT_DLL
            $<$<CONFIG:Debug>:LC_CLANGCXX_ENABLE_COMMENT=1>
            PUBLIC LUISA_ENABLE_CLANGCXX=1)
    target_include_directories(luisa-compute-clangcxx PRIVATE ${CLANG_INCLUDE_DIRS})
    if (TARGET clangTooling)
        target_link_libraries(luisa-compute-clangcxx PRIVATE clangTooling)
    else ()
        target_link_libraries(luisa-compute-clangcxx PRIVATE clang-cpp)
    endif ()
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(luisa-compute-clangcxx PRIVATE /bigobj)
    endif ()
    luisa_compute_install(clangcxx)
else ()
    add_library(luisa-compute-clangcxx INTERFACE)
    luisa_compute_install(clangcxx)
endif ()
