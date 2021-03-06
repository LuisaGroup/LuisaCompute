set(LUISA_COMPUTE_COMPILE_SOURCES
    codegen.cpp codegen.h
    cpp_codegen.cpp cpp_codegen.h)

add_library(luisa-compute-compile SHARED ${LUISA_COMPUTE_COMPILE_SOURCES})
target_link_libraries(luisa-compute-compile PUBLIC luisa-compute-ast)
set_target_properties(luisa-compute-compile PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
