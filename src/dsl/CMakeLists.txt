set(LUISA_COMPUTE_DSL_SOURCES
    dummy.cpp # to make msvc happy
    var.h
    expr.h
    syntax.h
    func.h
    buffer.h
    shared.h
    constant.h
    texture.h
    argument.h struct.h stmt.h)

add_library(luisa-compute-dsl SHARED ${LUISA_COMPUTE_DSL_SOURCES})
target_link_libraries(luisa-compute-dsl PUBLIC luisa-compute-ast)
set_target_properties(luisa-compute-dsl PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
