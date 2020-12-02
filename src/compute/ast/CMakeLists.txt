set(LUISA_COMPUTE_AST_SOURCES
    function.cpp function.h
    module.cpp module.h
    expression.cpp expression.h
    variable.cpp variable.h
    statement.cpp statement.h)

add_library(luisa-compute-ast SHARED ${LUISA_COMPUTE_AST_SOURCES})
target_link_libraries(luisa-compute-ast PUBLIC luisa-compute-include)
