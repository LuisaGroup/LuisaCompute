set(LUISA_COMPUTE_SYNTAX_SOURCES
    var.cpp var.h)

add_library(luisa-compute-syntax SHARED ${LUISA_COMPUTE_SYNTAX_SOURCES})
target_link_libraries(luisa-compute-runtime PUBLIC luisa-compute-include)
