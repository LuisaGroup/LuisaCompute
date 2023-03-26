if(LUISA_COMPUTE_ENABLE_CPU OR LUISA_COMPUTE_ENABLE_REMOTE OR LUISA_COMPUTE_COMPILED_BY_RUST_FRONTEND)
    set(LUISA_COMPUTE_DEVICE_COMMON_SOURCES 
        rust_device_common.h rust_device_common.cpp
        vulkan_swapchain.h vulkan_swapchain.cpp
    )
    add_library(luisa-compute-device-common SHARED ${LUISA_COMPUTE_DEVICE_COMMON_SOURCES})
    target_link_libraries(luisa-compute-device-common PRIVATE luisa-compute-runtime)
endif()