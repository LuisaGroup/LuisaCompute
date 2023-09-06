set(LUISA_COMPUTE_AST_SOURCES
        ast2json.cpp
        atomic_ref_node.cpp
        callable_library.cpp
        constant_data.cpp
        expression.cpp
        external_function.cpp
        function.cpp
        function_builder.cpp
        op.cpp
        statement.cpp
        type.cpp
        variable.cpp)

add_library(luisa-compute-ast SHARED ${LUISA_COMPUTE_AST_SOURCES})
target_compile_definitions(luisa-compute-ast PRIVATE LC_AST_EXPORT_DLL=1)

if (LUISA_COMPUTE_ENABLE_RUST)
    target_compile_definitions(luisa-compute-ast PRIVATE LC_AST_ENABLE_IR=1)
endif()

target_link_libraries(luisa-compute-ast PUBLIC luisa-compute-core luisa-compute-ext)

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(luisa-compute-ast PRIVATE /bigobj)
endif ()

set_target_properties(luisa-compute-ast PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
        OUTPUT_NAME lc-ast)

target_precompile_headers(luisa-compute-ast PRIVATE pch.h)

luisa_compute_install(ast SOURCES ${LUISA_COMPUTE_AST_SOURCES})
