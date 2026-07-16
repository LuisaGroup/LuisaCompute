local lc_enable_gui = has_config("lc_enable_gui")

local function tutorial_proj(name, source, gui_dep, callable)
    if gui_dep and not lc_enable_gui then return end
    target(name)
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({project_kind = "binary"})
    add_files(source)
    add_includedirs("$(projectdir)/src/tests/", "$(projectdir)/src/tests/common/")
    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "stb-image")
    if lc_enable_gui then add_deps("lc-gui") end
    if gui_dep then add_defines("LUISA_ENABLE_GUI") end
    if callable then callable() end
    target_end()
end

-- Tutorials that work without GUI (pure offline rendering)
tutorial_proj("tutorial_01_mandelbrot",       "01_mandelbrot/main.cpp")
tutorial_proj("tutorial_02_path_tracing",     "02_path_tracing/main.cpp")
tutorial_proj("tutorial_06_image_processing", "06_image_processing/main.cpp")

-- Tutorials that require GUI for interactive mode
tutorial_proj("tutorial_03_mpm_fluid",          "03_mpm_fluid/main.cpp",          true)
tutorial_proj("tutorial_04_game_of_life",       "04_game_of_life/main.cpp",       true)
tutorial_proj("tutorial_05_wave_equation",      "05_wave_equation/main.cpp",      true)
tutorial_proj("tutorial_07_nbody",              "07_nbody/main.cpp",              true)
tutorial_proj("tutorial_08_fire_particles",     "08_fire_particles/main.cpp",     true)
tutorial_proj("tutorial_09_reaction_diffusion", "09_reaction_diffusion/main.cpp", true)
tutorial_proj("tutorial_10_voxel_raytracer",    "10_voxel_raytracer/main.cpp",    true)
tutorial_proj("tutorial_11_black_hole",         "11_black_hole/main.cpp",         true)
