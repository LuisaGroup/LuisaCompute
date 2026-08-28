set(LUISA_COMPUTE_AST_SOURCES
        ast2json.cpp
        atomic_ref_node.cpp
        callable_library.cpp
        constant_data.cpp
        coro_suspend.cpp
        expression.cpp
        external_function.cpp
        function.cpp
        function_builder.cpp
        function_duplicator.cpp
        op.cpp
        statement.cpp
        tensor.cpp
        tile_function_builder.cpp
        tile_to_kernel.cpp
        type.cpp
        variable.cpp)

add_library(luisa-compute-ast ${LUISA_COMPUTE_AST_SOURCES})
target_compile_definitions(luisa-compute-ast PRIVATE LUISA_AST_EXPORT_DLL=1)

target_link_libraries(luisa-compute-ast PUBLIC luisa-compute-core luisa-compute-ext)

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(luisa-compute-ast PRIVATE /bigobj)
endif ()

set_target_properties(luisa-compute-ast PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
        OUTPUT_NAME luisa-ast)

luisa_compute_install(ast SOURCES ${LUISA_COMPUTE_AST_SOURCES})
