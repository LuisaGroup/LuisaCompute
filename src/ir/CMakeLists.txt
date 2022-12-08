add_library(luisa-compute-ir-api SHARED ir.hpp ir.cpp ast2ir.cpp ast2ir.h)
target_link_libraries(luisa-compute-ir-api PUBLIC luisa-compute-ast luisa-compute-rust-meta)
target_compile_definitions(luisa-compute-ir-api PRIVATE LC_IR_EXPORT_DLL=1)
set_target_properties(luisa-compute-ir-api PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
