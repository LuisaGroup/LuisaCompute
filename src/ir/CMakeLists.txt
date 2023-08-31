if (LUISA_COMPUTE_ENABLE_RUST)
    set(LUISA_COMPUTE_IR_SOURCES
            ast2ir.cpp
            ir2ast.cpp
            ir.cpp)

    add_library(luisa-compute-ir SHARED ${LUISA_COMPUTE_IR_SOURCES})
    target_link_libraries(luisa-compute-ir PUBLIC luisa-compute-ast luisa-compute-rust-meta)
    target_compile_definitions(luisa-compute-ir PRIVATE LC_IR_EXPORT_DLL=1)
    luisa_compute_install(ir SOURCES ${LUISA_COMPUTE_IR_SOURCES})

    set_target_properties(luisa-compute-ir PROPERTIES
            UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
            OUTPUT_NAME lc-ir)
else ()
    add_library(luisa-compute-ir INTERFACE)
    luisa_compute_install(ir)
endif ()
