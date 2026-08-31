set(LUISA_COMPUTE_VALIDATION_SOURCES
        accel.cpp accel.h
        bindless_array.h
        buffer.h
        depth_buffer.h
        device.cpp device.h
        dstorage_ext_impl.cpp dstorage_ext_impl.h
        dx_hdr_ext_impl.cpp dx_hdr_ext_impl.h
        native_res_ext_impl.cpp native_res_ext_impl.h
        event.h
        mesh.h
        curve.cpp curve.h
        motion_instance.cpp motion_instance.h
        pinned_mem_impl.cpp pinned_mem_impl.h
        procedural_primitives.h
        range.h
        raster_ext_impl.cpp raster_ext_impl.h
        resource.cpp resource.h
        rw_resource.cpp rw_resource.h
        shader.h
        sparse_heap.cpp sparse_heap.h
        stream.cpp stream.h
        stream_func.h
        stats.cpp        stats.h
        swap_chain.h
        vk_cuda_interop_impl.cpp vk_cuda_interop_impl.h
        dx_cuda_interop_impl.cpp dx_cuda_interop_impl.h
        texture.h)

add_library(luisa-compute-validation-layer MODULE ${LUISA_COMPUTE_VALIDATION_SOURCES})
target_link_libraries(luisa-compute-validation-layer PRIVATE
        luisa-compute-runtime luisa-compute-gui)
target_include_directories(luisa-compute-validation-layer PRIVATE ../../runtime)
add_dependencies(luisa-compute-backends luisa-compute-validation-layer)
set_target_properties(luisa-compute-validation-layer PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
        DEBUG_POSTFIX ""
        EXPORT_NAME validation_layer
        OUTPUT_NAME luisa-validation-layer)
install(TARGETS luisa-compute-validation-layer
        EXPORT LuisaComputeTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
