if (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    
    message(STATUS "Build with Metal")
    
    enable_language(OBJC)
    enable_language(OBJCXX)
    
    set(METAL_SOURCES
        metal_device.mm metal_device.h
        metal_command_encoder.mm metal_command_encoder.h
        metal_compiler.mm metal_compiler.h
        metal_codegen.mm metal_codegen.h
        metal_argument_buffer.mm metal_argument_buffer.h
        metal_event.h
        metal_stream.h)
    
    luisa_compute_add_backend(metal SOURCES ${METAL_SOURCES})
    set_target_properties(luisa-compute-backend-metal PROPERTIES
                          OBJC_STANDARD 11
                          OBJCXX_STANDARD 20
                          OBJC_STANDARD_REQUIRED ON
                          OBJCXX_STANDARD_REQUIRED ON
                          OBJC_EXTENSIONS OFF
                          OBJCXX_EXTENSIONS OFF)
    target_compile_options(luisa-compute-backend-metal PRIVATE -fobjc-arc)
    target_link_libraries(luisa-compute-backend-metal PRIVATE
                          "-framework Foundation"
                          "-framework Metal"
                          "-framework MetalKit"
                          "-framework MetalPerformanceShaders")
endif ()
