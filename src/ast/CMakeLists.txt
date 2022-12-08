set(LUISA_COMPUTE_AST_SOURCES
        function.h function.cpp
        function_builder.cpp function_builder.h
        function_serializer.cpp function_serializer.h
        expression.cpp expression.h
        variable.h
        statement.cpp statement.h
        type.cpp type.h
        type_registry.cpp type_registry.h
        interface.h
        constant_data.cpp constant_data.h
        usage.h
        op.cpp op.h)
add_library(luisa-compute-ast SHARED ${LUISA_COMPUTE_AST_SOURCES})
target_compile_definitions(luisa-compute-ast PRIVATE LC_AST_EXPORT_DLL=1)
target_link_libraries(luisa-compute-ast PUBLIC luisa-compute-core luisa-compute-ext)
set_target_properties(luisa-compute-ast PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD})
