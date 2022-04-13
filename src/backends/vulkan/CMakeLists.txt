set(LUISA_COMPUTE_VULKAN_SOURCES
        vulkan_device.cpp vulkan_device.h)

message(STATUS "Build with Vulkan backend")
luisa_compute_add_backend(vulkan SOURCES ${LUISA_COMPUTE_VULKAN_SOURCES})
