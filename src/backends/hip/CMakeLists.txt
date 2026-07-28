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
            hip_curve.cpp hip_curve.h
            hip_curve_bounds.cpp hip_curve_bounds.h
            hip_event.cpp hip_event.h
            hip_mesh.cpp hip_mesh.h
            hip_motion_mesh_builtin.cpp hip_motion_mesh_builtin.h
            hip_motion_instance.cpp hip_motion_instance.h
            hip_pinned_memory.cpp hip_pinned_memory.h
            hip_geometry.h
            hip_procedural_primitive.cpp hip_procedural_primitive.h
            hip_shader.h hip_shader.cpp
            hip_shader_metadata.h
            hip_shader_native.h hip_shader_native.cpp
            hip_shader_printer.h hip_shader_printer.cpp
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
    # Resolve HIPRT from this project instead of the top-level source tree so
    # the HIP backend also works when LuisaCompute is added as a subdirectory.
    set(HIPRT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../ext/HIPRT")
    include("${CMAKE_CURRENT_LIST_DIR}/../../ext/hiprt_from_source.cmake")

    # Compile the motion-mesh preprocessing kernels once for the configured
    # HIPRT offload architectures and embed the resulting fat binary in the
    # backend. Keep floating-point contraction/optimization at the HIP default;
    # finite-value validation and swept bounds must not use fast-math.
    set(_hip_motion_mesh_builtin_source
            "${CMAKE_CURRENT_LIST_DIR}/hip_motion_mesh_builtin.hip")
    set(_hip_motion_mesh_builtin_fatbin
            "${CMAKE_CURRENT_BINARY_DIR}/hip_motion_mesh_builtin.hipfb")
    set(_hip_motion_mesh_builtin_embedded_cpp
            "${CMAKE_CURRENT_BINARY_DIR}/hip_motion_mesh_builtin_embedded.cpp")
    set(_hip_motion_mesh_builtin_embedded_h
            "${CMAKE_CURRENT_BINARY_DIR}/hip_motion_mesh_builtin_embedded.h")
    add_custom_command(
            OUTPUT "${_hip_motion_mesh_builtin_fatbin}"
            COMMAND "${hip_HIPCC_EXECUTABLE}"
            -x hip "${_hip_motion_mesh_builtin_source}"
            -O3 -std=c++17 -fno-fast-math
            ${_hiprt_offload_args}
            --genco
            -o "${_hip_motion_mesh_builtin_fatbin}"
            DEPENDS "${_hip_motion_mesh_builtin_source}"
                    "${hip_HIPCC_EXECUTABLE}"
                    "${CMAKE_CURRENT_LIST_FILE}"
            COMMENT "[HIP] Compiling motion-mesh preprocessing kernels for ${_hiprt_gpu_archs}..."
            VERBATIM)
    add_custom_command(
            OUTPUT "${_hip_motion_mesh_builtin_embedded_cpp}"
                   "${_hip_motion_mesh_builtin_embedded_h}"
            COMMAND $<TARGET_FILE:luisa_embed_device_lib>
            "${_hip_motion_mesh_builtin_fatbin}"
            --unsigned --prefix luisa_compute_
            -o "${_hip_motion_mesh_builtin_embedded_cpp}"
            -h "${_hip_motion_mesh_builtin_embedded_h}"
            COMMAND ${CMAKE_COMMAND} -E touch
                    "${_hip_motion_mesh_builtin_embedded_cpp}"
                    "${_hip_motion_mesh_builtin_embedded_h}"
            DEPENDS luisa_embed_device_lib
                    "${_hip_motion_mesh_builtin_fatbin}"
            COMMENT "[HIP] Embedding motion-mesh preprocessing kernels..."
            VERBATIM)
    add_custom_target(luisa-compute-backend-hip-embed-motion-mesh-builtin
            DEPENDS "${_hip_motion_mesh_builtin_fatbin}"
                    "${_hip_motion_mesh_builtin_embedded_cpp}"
                    "${_hip_motion_mesh_builtin_embedded_h}")
    add_dependencies(luisa-compute-backend-hip
            luisa-compute-backend-hip-embed-motion-mesh-builtin)
    set_source_files_properties(
            "${_hip_motion_mesh_builtin_embedded_cpp}"
            "${_hip_motion_mesh_builtin_embedded_h}"
            PROPERTIES GENERATED TRUE SKIP_UNITY_BUILD_INCLUSION TRUE)
    target_sources(luisa-compute-backend-hip PRIVATE
            "${_hip_motion_mesh_builtin_embedded_cpp}"
            "${_hip_motion_mesh_builtin_embedded_h}")
    target_include_directories(luisa-compute-backend-hip PRIVATE
            "${CMAKE_CURRENT_BINARY_DIR}")

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
    # Keep runtime artifacts synchronized even when only an included HIPRT
    # kernel header changes and the backend shared library does not relink.
    add_custom_target(luisa-compute-hiprt-runtime-artifacts
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${HIPRT_FROM_SOURCE_BITCODE_DIR}/${HIPRT_FROM_SOURCE_LIB_NAME}_${HIP_VERSION_STR}_amd_lib_linux.bc"
            "${HIPRT_FROM_SOURCE_BITCODE_DIR}/${HIPRT_FROM_SOURCE_LIB_NAME}_${HIP_VERSION_STR}_amd.hipfb"
            "${HIPRT_FROM_SOURCE_BITCODE_DIR}/oro_compiled_kernels.hipfb"
            $<TARGET_FILE_DIR:luisa-compute-backend-hip>
            DEPENDS hiprt_compile_bitcodes)
    add_dependencies(luisa-compute-backend-hip
            luisa-compute-hiprt-runtime-artifacts)

    # Add LLVM codegen for HIP
    add_subdirectory(llvm_codegen)

elseif (NOT LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES)
    message(FATAL_ERROR "HIP not found. The HIP backend will not be built.")
else ()
    message(WARNING "HIP not found. The HIP backend will not be built.")
endif ()
