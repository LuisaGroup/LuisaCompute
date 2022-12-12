set(LUISA_COMPUTE_COMPILE_SOURCES
        codegen.cpp codegen.h
        definition_analysis.cpp definition_analysis.h
        spirv_translator.cpp spirv_translator.h)

add_library(luisa-compute-compile SHARED ${LUISA_COMPUTE_COMPILE_SOURCES})
target_link_libraries(luisa-compute-compile PUBLIC luisa-compute-ast)
set_target_properties(luisa-compute-compile PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD})

luisa_compute_install(compile SOURCES ${LUISA_COMPUTE_COMPILE_SOURCES})
