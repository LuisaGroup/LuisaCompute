set(LUISA_COMPUTE_VALIDATION_SOURCES
        accel.cpp accel.h
        bindless_array.h
        buffer.h
        depth_buffer.h
        device.cpp device.h
        event.h
        mesh.h
        procedural_primitives.h
        resource.cpp resource.h
        rw_resource.cpp rw_resource.h
        shader.cpp shader.h
        stream.cpp stream.h
        swap_chain.h
        texture.h)

add_library(lc-validation-layer SHARED ${LUISA_COMPUTE_VALIDATION_SOURCES})
target_link_libraries(lc-validation-layer PRIVATE luisa-compute-runtime luisa-compute-gui)
add_dependencies(luisa-compute-backends lc-validation-layer)
set_target_properties(lc-validation-layer PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
        DEBUG_POSTFIX "")
install(TARGETS lc-validation-layer
        LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
