if (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")

    message(STATUS "Build with Metal backend")

    set(METAL_SOURCES
            metal_device.mm metal_device.h
            metal_command_encoder.mm metal_command_encoder.h
            metal_compiler.mm metal_compiler.h
            metal_codegen.mm metal_codegen.h
            metal_event.h
            metal_stream.h metal_stream.mm
            metal_buffer_view.h
            metal_host_buffer_pool.mm metal_host_buffer_pool.h
            metal_bindless_array.mm metal_bindless_array.h
            metal_shader.h
            metal_mesh.mm metal_mesh.h
            metal_accel.mm metal_accel.h)

    luisa_compute_add_backend(metal SOURCES ${METAL_SOURCES})
    target_compile_options(lc-backend-metal PRIVATE -fobjc-arc)
    target_link_libraries(lc-backend-metal PRIVATE
            "-framework Foundation"
            "-framework Metal"
            "-framework QuartzCore")
elseif (NOT LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES)
    message(FATAL_ERROR "Metal backend requires clang.")
else ()
    message(WARNING "Metal backend requires clang but LuisaCompute is using ${CMAKE_CXX_COMPILER_ID}. The backend will be disabled.")
endif ()
