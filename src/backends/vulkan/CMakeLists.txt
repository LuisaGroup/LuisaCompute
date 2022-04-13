if (LUISA_COMPUTE_ENABLE_VULKAN)
    message(STATUS "Build with Vulkan backend")
    set(LUISA_COMPUTE_VULKAN_SOURCES
            vulkan_device.cpp vulkan_device.h)
    luisa_compute_add_backend(vulkan SOURCES ${LUISA_COMPUTE_VULKAN_SOURCES})
    target_link_libraries(luisa-compute-backend-vulkan PRIVATE luisa-compute-ext)

    # playground
    add_executable(test_playground_vulkan vulkan_playground.cpp)
    target_link_libraries(test_playground_vulkan PRIVATE luisa-compute-runtime)
endif ()
