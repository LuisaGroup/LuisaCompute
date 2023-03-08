set(LUISA_PYTHON_SOURCES
        export_gui.cpp
        export_matrix.cpp
        export_op.cpp
        export_vector2.cpp
        export_vector3.cpp
        export_vector4.cpp
        image_util.cpp
        lcapi.cpp
        managed_accel.cpp managed_accel.h
        managed_bindless.cpp managed_bindless.h
        managed_collector.cpp managed_collector.h
        py_stream.cpp py_stream.h
        ref_counter.cpp ref_counter.h
        stb.cpp)
pybind11_add_module(lcapi ${LUISA_PYTHON_SOURCES})
if (WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(lcapi PRIVATE -Wa,-mbig-obj)
endif ()
target_link_libraries(lcapi PRIVATE luisa::compute)
add_custom_target(lcapi-copy ALL)

add_custom_command(TARGET lcapi-copy
        DEPENDS luisa
        COMMENT "Copying Python package"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/luisa"
        "$<TARGET_FILE_DIR:luisa-compute-core>/luisa"
        VERBATIM)
add_dependencies(lcapi lcapi-copy)
