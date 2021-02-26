set(LUISA_COMPUTE_DSL_SOURCES
    var.cpp var.h
    expr.cpp expr.h)

add_library(luisa-compute-dsl SHARED ${LUISA_COMPUTE_DSL_SOURCES})
target_link_libraries(luisa-compute-dsl PUBLIC luisa-compute-core)
set_target_properties(luisa-compute-dsl PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
