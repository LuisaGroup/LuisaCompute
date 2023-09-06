set(LUISA_PYTHON_SOURCES
        ast_evaluator.cpp ast_evaluator.h
        export_commands.cpp
        export_expr.cpp
        export_gui.cpp
        export_matrix.cpp
        export_op.cpp
        export_runtime.cpp
        export_vector2.cpp
        export_vector3.cpp
        export_vector4.cpp
        image_util.cpp
        lcapi.cpp
        managed_accel.cpp managed_accel.h
        managed_bindless.cpp managed_bindless.h
        managed_collector.cpp managed_collector.h
        managed_device.h
        py_stream.cpp py_stream.h
        ref_counter.cpp ref_counter.h
        stb.cpp)

pybind11_add_module(lcapi ${LUISA_PYTHON_SOURCES})

if (WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(lcapi PRIVATE -Wa,-mbig-obj)
endif ()
target_link_libraries(lcapi PRIVATE luisa::compute)
target_precompile_headers(lcapi PRIVATE pch.h)

install(DIRECTORY luisa
        DESTINATION ${SKBUILD_PLATLIB_DIR})
install(TARGETS lcapi
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
