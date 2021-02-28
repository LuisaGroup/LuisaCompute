set(LUISA_COMPUTE_DSL_SOURCES
    var.h
    expr.h
    syntax.h func.h)

add_library(luisa-compute-dsl INTERFACE ${LUISA_COMPUTE_DSL_SOURCES})
target_link_libraries(luisa-compute-dsl INTERFACE luisa-compute-core)
set_target_properties(luisa-compute-dsl PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
