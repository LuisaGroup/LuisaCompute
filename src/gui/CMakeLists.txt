if (LUISA_COMPUTE_ENABLE_GUI)
    set(LUISA_COMPUTE_GUI_SOURCES
            imgui_impl_glfw.cpp imgui_impl_glfw.h
            window.cpp window.h
            framerate.cpp framerate.h)
    add_library(luisa-compute-gui SHARED ${LUISA_COMPUTE_GUI_SOURCES})
    target_link_libraries(luisa-compute-gui PUBLIC luisa-compute-core luisa-compute-ext)
    target_compile_definitions(luisa-compute-gui PUBLIC LUISA_GUI_ENABLED=1)
    set_target_properties(luisa-compute-gui PROPERTIES
            WINDOWS_EXPORT_ALL_SYMBOLS ON
            UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD})
    luisa_compute_install(gui SOURCES ${LUISA_COMPUTE_GUI_SOURCES})
else ()
    add_library(luisa-compute-gui INTERFACE)
    luisa_compute_install(gui)
endif ()
