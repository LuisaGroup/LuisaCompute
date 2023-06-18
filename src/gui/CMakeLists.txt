if (LUISA_COMPUTE_ENABLE_GUI)
    set(LUISA_COMPUTE_GUI_SOURCES
            framerate.cpp
            window.cpp)
    add_library(luisa-compute-gui SHARED ${LUISA_COMPUTE_GUI_SOURCES})
    target_link_libraries(luisa-compute-gui PUBLIC luisa-compute-runtime)
    target_compile_definitions(luisa-compute-gui
            PRIVATE LC_GUI_EXPORT_DLL
            PUBLIC LUISA_ENABLE_GUI=1)
    set_target_properties(luisa-compute-gui PROPERTIES
            UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
            OUTPUT_NAME lc-gui)
    luisa_compute_install(gui SOURCES ${LUISA_COMPUTE_GUI_SOURCES})
else ()
    add_library(luisa-compute-gui INTERFACE)
    luisa_compute_install(gui)
endif ()
