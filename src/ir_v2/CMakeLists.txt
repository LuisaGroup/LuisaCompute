set(LUISA_COMPUTE_IR_V2_SOURCES
            ir.cpp
            ir_v2_api.cpp
            ir_v2_defs.cpp
            ir_v2_bindings.cpp
            analysis/usedef.cpp
            transform/dce.cpp
            transform/ssa.cpp)

add_library(luisa-compute-ir-v2 SHARED ${LUISA_COMPUTE_IR_V2_SOURCES})
target_link_libraries(luisa-compute-ir-v2 PUBLIC luisa-compute-ast)
target_compile_definitions(luisa-compute-ir-v2 PRIVATE LC_IR_EXPORT_DLL=1)
luisa_compute_install(ir-v2 SOURCES ${LUISA_COMPUTE_IR_V2_SOURCES})

set_target_properties(luisa-compute-ir-v2 PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
        OUTPUT_NAME lc-ir-v2)