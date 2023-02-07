set(LUISA_COMPUTE_RHI_SOURCES
        command.cpp command.h
        device_interface.cpp device_interface.h
        pixel_format.h
        resource.cpp resource.h
        stream_tag.h
        texture_sampler.h)

add_library(luisa-compute-rhi SHARED ${LUISA_COMPUTE_RHI_SOURCES})
target_link_libraries(luisa-compute-rhi PUBLIC luisa-compute-core luisa-compute-ext)
target_compile_definitions(luisa-compute-rhi PRIVATE LC_RHI_EXPORT_DLL=1)
set_target_properties(luisa-compute-rhi PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD})

luisa_compute_install(rhi SOURCES ${LUISA_COMPUTE_RHI_SOURCES})
