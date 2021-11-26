set(LUISA_COMPUTE_META_SOURCES
        property.h
        s_expr.cpp s_expr.h)

add_library(luisa-compute-meta SHARED ${LUISA_COMPUTE_META_SOURCES})
target_link_libraries(luisa-compute-meta PUBLIC luisa-compute-core)
set_target_properties(luisa-compute-meta PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON)
