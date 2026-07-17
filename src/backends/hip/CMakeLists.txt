find_package(hip)
find_package(hiprtc QUIET)

if (hip_FOUND)
    message(STATUS "Build with HIP backend: ${hip_VERSION}")
    set(LUISA_HIP_SOURCES
            ../common/default_binary_io.cpp ../common/default_binary_io.h
            hip_device.cpp hip_device.h
            hip_buffer.cpp hip_buffer.h
            hip_bindless_array.cpp hip_bindless_array.h
            hip_accel.cpp hip_accel.h
            hip_callback_context.h
            hip_command_encoder.cpp hip_command_encoder.h
            hip_event.cpp hip_event.h
            hip_mesh.cpp hip_mesh.h
            hip_geometry.h
            hip_procedural_primitive.cpp hip_procedural_primitive.h
            hip_shader.h hip_shader.cpp
            hip_shader_metadata.h
            hip_shader_native.h hip_shader_native.cpp
            hip_shader_printer.h
            hip_stage_buffer_pool.cpp hip_stage_buffer_pool.h
            hip_stream.cpp hip_stream.h
            hip_swapchain.cpp hip_swapchain.h
            hip_texture.cpp hip_texture.h
    )
    luisa_compute_add_backend(hip SOURCES ${LUISA_HIP_SOURCES})

    target_link_libraries(luisa-compute-backend-hip PRIVATE
            luisa-compute-ast
            luisa-compute-runtime
            luisa-compute-xir
            hip::host
    )
    if (TARGET hiprtc::hiprtc)
        target_link_libraries(luisa-compute-backend-hip PRIVATE hiprtc::hiprtc)
    elseif (TARGET hip::hiprtc)
        target_link_libraries(luisa-compute-backend-hip PRIVATE hip::hiprtc)
    else ()
        message(WARNING "hiprtc target not found; linking against libhiprtc.so by name")
        target_link_libraries(luisa-compute-backend-hip PRIVATE hiprtc)
    endif ()
    if (TARGET luisa-compute-vulkan-swapchain)
        target_link_libraries(luisa-compute-backend-hip PRIVATE luisa-compute-vulkan-swapchain)
    endif ()

    # HIPRT integration: build from source (submodule at src/ext/HIPRT)
    # Uses system hipcc for compilation
    set(HIPRT_SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/ext/HIPRT")
    include(${CMAKE_SOURCE_DIR}/src/ext/hiprt_from_source.cmake)

    # Link against the from-source HIPRT target
    target_include_directories(luisa-compute-backend-hip PRIVATE
            ${HIPRT_FROM_SOURCE_INCLUDE_DIR}
            ${HIPRT_SOURCE_DIR})
    target_link_libraries(luisa-compute-backend-hip PRIVATE hiprt_from_source)

    set(LUISA_COMPUTE_HIPRT_SDK_INCLUDE_DIR ${HIPRT_FROM_SOURCE_INCLUDE_DIR})

    # copy HIPRT runtime files to binary output directory
    # Copy the from-source built shared library and bitcode artifacts
    add_custom_command(TARGET luisa-compute-backend-hip POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:hiprt_from_source>
            $<TARGET_FILE_DIR:luisa-compute-backend-hip>)
    # Copy bitcode/hipfb artifacts needed at runtime
    add_custom_command(TARGET luisa-compute-backend-hip POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${HIPRT_FROM_SOURCE_BITCODE_DIR}/${HIPRT_FROM_SOURCE_LIB_NAME}_${HIP_VERSION_STR}_amd_lib_linux.bc"
            "${HIPRT_FROM_SOURCE_BITCODE_DIR}/${HIPRT_FROM_SOURCE_LIB_NAME}_${HIP_VERSION_STR}_amd.hipfb"
            "${HIPRT_FROM_SOURCE_BITCODE_DIR}/oro_compiled_kernels.hipfb"
            $<TARGET_FILE_DIR:luisa-compute-backend-hip>)

    # Add LLVM codegen for HIP
    add_subdirectory(llvm_codegen)

elseif (NOT LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES)
    message(FATAL_ERROR "HIP not found. The HIP backend will not be built.")
else ()
    message(WARNING "HIP not found. The HIP backend will not be built.")
endif ()
