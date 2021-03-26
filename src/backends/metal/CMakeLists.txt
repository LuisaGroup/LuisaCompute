if (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    
    message(STATUS "Build with Metal")
    
    enable_language(OBJC)
    enable_language(OBJCXX)
    
    set(METAL_SOURCES
        metal_device.mm metal_device.h
        metal_command_encoder.mm metal_command_encoder.h
        metal_compiler.mm metal_compiler.h
        metal_codegen.mm metal_codegen.h
        metal_argument_buffer.mm metal_argument_buffer.h)
    
    luisa_compute_add_backend(metal SOURCES ${METAL_SOURCES})
    
    target_compile_options(luisa-compute-backend-metal PRIVATE -fobjc-arc)
    target_link_libraries(luisa-compute-backend-metal PRIVATE
                          "-framework Foundation"
                          "-framework Metal"
                          "-framework MetalKit"
                          "-framework MetalPerformanceShaders")
endif ()
