if(LUISA_COMPUTE_ENABLE_CPU OR LUISA_COMPUTE_ENABLE_REMOTE OR LUISA_COMPUTE_COMPILED_BY_RUST_FRONTEND)
    find_package(Vulkan)
    find_package(X11)
    set(LUISA_COMPUTE_VULKAN_SWAPCHAIN_SOURCES
        # rust_device_common.h rust_device_common.cpp
        vulkan_swapchain.h vulkan_swapchain.cpp
    )
    add_library(luisa-compute-vulkan-swapchain SHARED ${LUISA_COMPUTE_VULKAN_SWAPCHAIN_SOURCES})
    target_link_libraries(luisa-compute-vulkan-swapchain PRIVATE luisa-compute-runtime)
    target_link_libraries(luisa-compute-vulkan-swapchain PRIVATE Vulkan::Vulkan ${X11_LIBRARIES})
endif()
