find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
    message(STATUS "Build with CUDA backend: ${CUDAToolkit_VERSION}")
    set(LUISA_COMPUTE_CUDA_SOURCES

            ../common/default_binary_io.cpp ../common/default_binary_io.h
            ../common/resource_tracker.h
            ../common/string_scratch.cpp ../common/string_scratch.h

            cuda_error.h
            cuda_bindless_array.cpp cuda_bindless_array.h
            cuda_buffer.cpp cuda_buffer.h
            cuda_stream.cpp cuda_stream.h
            cuda_device.cpp cuda_device.h
            cuda_host_buffer_pool.cpp cuda_host_buffer_pool.h
            cuda_command_encoder.cpp cuda_command_encoder.h
            cuda_texture.cpp cuda_texture.h
            cuda_codegen_ast.cpp cuda_codegen_ast.h
            cuda_compiler.cpp cuda_compiler.h
            cuda_accel.cpp cuda_accel.h
            cuda_primitive.cpp cuda_primitive.h
            cuda_procedural_primitive.cpp cuda_procedural_primitive.h
            cuda_mesh.cpp cuda_mesh.h
            cuda_shader.cpp cuda_shader.h
            cuda_shader_metadata.cpp cuda_shader_metadata.h
            cuda_shader_native.cpp cuda_shader_native.h
            cuda_shader_optix.cpp cuda_shader_optix.h
            cuda_swapchain.cpp cuda_swapchain.h
            cuda_callback_context.h
            cuda_builtin_embedded.cpp cuda_builtin_embedded.h
            optix_api.cpp optix_api.h)

    add_library(lc-backend-cuda-dependency INTERFACE)
    find_package(Vulkan)
    if (Vulkan_FOUND AND Vulkan_VERSION VERSION_GREATER_EQUAL 1.1)
        message(STATUS "Build CUDA backend with Vulkan Swapchain: ${Vulkan_VERSION}")
        list(APPEND LUISA_COMPUTE_CUDA_SOURCES
                ../common/vulkan_swapchain.cpp ../common/vulkan_swapchain.h)
        target_link_libraries(lc-backend-cuda-dependency INTERFACE Vulkan::Vulkan)
        target_compile_definitions(lc-backend-cuda-dependency INTERFACE LUISA_CUDA_ENABLE_VULKAN_SWAPCHAIN)
    endif ()

    set(LUISA_COMPUTE_CUDA_BUILTIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cuda_builtin")
    luisa_compute_add_backend(cuda SOURCES ${LUISA_COMPUTE_CUDA_SOURCES})
    target_link_libraries(lc-backend-cuda PRIVATE
            CUDA::cuda_driver
            CUDA::nvrtc
            lc-backend-cuda-dependency)
    if (WIN32)
        target_link_libraries(lc-backend-cuda PRIVATE cfgmgr32)
    endif ()
elseif (NOT LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES)
    message(FATAL_ERROR "CUDA not found. The CUDA backend will not be built.")
else ()
    message(WARNING "CUDA not found. The CUDA backend will not be built.")
endif ()
