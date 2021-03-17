if (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    
    message(STATUS "Build with Metal")
    
    enable_language(OBJC)
    enable_language(OBJCXX)
    
    set(METAL_SOURCES
        metal_device.mm metal_device.h)
    
    luisa_compute_add_backend(metal SOURCES ${METAL_SOURCES})
    
    target_compile_options(luisa-compute-backend-metal PRIVATE -fobjc-arc)
    
    target_link_libraries(luisa-compute-backend-metal PRIVATE
                          "-framework Foundation"
                          "-framework Metal"
                          "-framework MetalKit"
                          "-framework MetalPerformanceShaders")
endif ()
