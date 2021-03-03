set(LUISA_COMPUTE_DSL_SOURCES
    var.h
    expr.h
    syntax.h func.h buffer.h shared.h constant.h texture.h argument.h)

add_library(luisa-compute-dsl INTERFACE ${LUISA_COMPUTE_DSL_SOURCES})
target_link_libraries(luisa-compute-dsl INTERFACE luisa-compute-core)
set_target_properties(luisa-compute-dsl PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
