corrosion_import_crate(MANIFEST_PATH luisa-compute-ir/Cargo.toml)

add_library(luisa-compute-ir-api SHARED ir.h ir.hpp ir.cpp)
target_link_libraries(luisa-compute-ir-api PUBLIC luisa_compute_ir luisa-compute-ast)
target_include_directories(luisa-compute-ir-api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(luisa-compute-ir-api PUBLIC luisa-compute-ir/) 
target_compile_definitions(luisa-compute-ir-api PRIVATE LC_IR_EXPORT_DLL=1)