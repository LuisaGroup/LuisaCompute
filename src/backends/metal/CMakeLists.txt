if (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    
    message(STATUS "Build with Metal")
    
    enable_language(OBJC)
    enable_language(OBJCXX)
    
    option(LUISA_COMPUTE_ENABLE_METAL_RAYTRACING "Enable Metal raytracing functionality" ON)
    
    set(METAL_SOURCES
        metal_device.mm metal_device.h
        metal_command_encoder.mm metal_command_encoder.h
        metal_compiler.mm metal_compiler.h
        metal_codegen.mm metal_codegen.h
        metal_shared_buffer_pool.mm metal_shared_buffer_pool.h
        metal_event.h
        metal_stream.h
        metal_buffer_view.h
        metal_ring_buffer.mm metal_ring_buffer.h
        metal_heap.mm metal_heap.h
        metal_shader.h)
    
    luisa_compute_add_backend(metal SOURCES ${METAL_SOURCES})
    if (LUISA_COMPUTE_ENABLE_METAL_RAYTRACING)
        target_sources(luisa-compute-backend-metal PRIVATE
                       metal_mesh.mm metal_mesh.h
                       metal_accel.mm metal_accel.h)
        target_compile_definitions(luisa-compute-backend-metal PRIVATE LUISA_METAL_RAYTRACING_ENABLED=1)
    endif ()
    set_target_properties(luisa-compute-backend-metal PROPERTIES
                          OBJC_STANDARD 11
                          OBJCXX_STANDARD 20
                          OBJC_STANDARD_REQUIRED ON
                          OBJCXX_STANDARD_REQUIRED ON
                          OBJC_EXTENSIONS ON
                          OBJCXX_EXTENSIONS ON
                          UNITY_BUILD ON)
    target_compile_options(luisa-compute-backend-metal PRIVATE -fobjc-arc)
    target_link_libraries(luisa-compute-backend-metal PRIVATE
                          "-framework Foundation"
                          "-framework Metal")
endif ()
