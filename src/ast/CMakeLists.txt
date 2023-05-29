set(LUISA_COMPUTE_AST_SOURCES
        ast_evaluator.cpp ast_evaluator.h
        atomic_ref_node.cpp atomic_ref_node.h
        constant_data.cpp constant_data.h
        expression.cpp expression.h
        external_function.cpp external_function.h
        function.cpp function.h
        function_builder.cpp function_builder.h
        interface.h
        op.cpp op.h
        statement.cpp statement.h
        type.cpp type.h
        type_registry.h
        usage.h
        variable.cpp variable.h)

add_library(luisa-compute-ast SHARED ${LUISA_COMPUTE_AST_SOURCES})
target_compile_definitions(luisa-compute-ast PRIVATE LC_AST_EXPORT_DLL=1)
target_link_libraries(luisa-compute-ast PUBLIC luisa-compute-core luisa-compute-ext)

set_target_properties(luisa-compute-ast PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
        OUTPUT_NAME lc-ast)

luisa_compute_install(ast SOURCES ${LUISA_COMPUTE_AST_SOURCES})
