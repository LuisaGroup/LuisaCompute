include(${CMAKE_CURRENT_LIST_DIR}/cmake/LuisaIOSBundle.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/metal4_path_tracing/Target.cmake)

# The primary iOS path-tracing example additionally runs the Metal4 device
# conformance preflight and captures an evidence snapshot while continuing to
# accumulate samples interactively.
luisa_compute_add_ios_metal4_path_tracing_bundle(
        example_ios_path_tracing
        # Keep the original bring-up identifier so locally generated iOS
        # development profiles remain reusable after the examples/ios move.
        BUNDLE_ID "org.luisa-compute.metal4-path-tracer"
        BUNDLE_NAME "Luisa Path Tracing"
        FOLDER "examples/ios")
if (TARGET luisa-metal4-ios-device-air-path-tracer)
    set_target_properties(example_ios_path_tracing PROPERTIES
            LUISA_TEST_MIRROR luisa-metal4-ios-device-air-path-tracer)
endif ()

# Preserve the early bring-up target name for existing local install scripts.
add_custom_target(example_metal4_ios_path_tracing
        DEPENDS example_ios_path_tracing)
set_target_properties(example_metal4_ios_path_tracing PROPERTIES
        FOLDER "examples/ios")

function(luisa_compute_add_ios_rendering_example name display_name source)
    set(target "example_ios_${name}")
    set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/${name}")
    file(MAKE_DIRECTORY "${generated_dir}")
    get_filename_component(
            LUISA_IOS_RENDERING_EXAMPLE_SOURCE "${source}" ABSOLUTE
            BASE_DIR "${CMAKE_SOURCE_DIR}/examples")
    set(entry "${generated_dir}/${name}_entry.cpp")
    configure_file(
            ${CMAKE_CURRENT_LIST_DIR}/common/rendering_example_entry.cpp.in
            "${entry}" @ONLY)

    add_executable(${target} MACOSX_BUNDLE
            ${CMAKE_CURRENT_LIST_DIR}/common/rendering_example_host.mm
            ${CMAKE_CURRENT_LIST_DIR}/common/Info.plist.in
            "${entry}")
    luisa_compute_link_ios_metal4_application(${target})
    target_compile_definitions(${target} PRIVATE
            LUISA_IOS_RENDERING_EXAMPLE_NAME="${display_name}")
    if (name MATCHES "^coro_")
        target_link_libraries(${target} PRIVATE luisa-compute-coro)
    endif ()
    string(REPLACE "_" "-" bundle_suffix "${name}")
    luisa_compute_configure_ios_bundle(${target}
            BUNDLE_ID "org.luisa-compute.examples.ios.${bundle_suffix}"
            BUNDLE_NAME "${display_name}"
            FOLDER "examples/ios/rendering"
            INFO_PLIST "${CMAKE_CURRENT_LIST_DIR}/common/Info.plist.in")

    if (name STREQUAL "path_tracing_spectrum")
        target_sources(${target} PRIVATE
                ${CMAKE_SOURCE_DIR}/examples/rendering/srgb_to_fourier_embedded.cpp
                ${CMAKE_SOURCE_DIR}/examples/rendering/srgb_to_fourier_embedded.h)
    endif ()
    set_property(GLOBAL APPEND PROPERTY
            LUISA_IOS_RENDERING_EXAMPLE_TARGETS ${target})
endfunction()

# Keep this list explicit: every entry is an independently signed app, so a
# new desktop rendering example is not silently claimed as iOS-compatible.
luisa_compute_add_ios_rendering_example(
        path_tracing_camera "Path Tracing Camera"
        rendering/path_tracing_camera.cpp)
luisa_compute_add_ios_rendering_example(
        path_tracing_cutout "Path Tracing Cutout"
        rendering/path_tracing_cutout.cpp)
luisa_compute_add_ios_rendering_example(
        path_tracing_hdr "Path Tracing HDR"
        rendering/path_tracing_hdr.cpp)
luisa_compute_add_ios_rendering_example(
        path_tracing_nested_callable "Path Tracing Nested Callable"
        rendering/path_tracing_nested_callable.cpp)
luisa_compute_add_ios_rendering_example(
        path_tracing_ray_masks "Path Tracing Ray Masks"
        rendering/path_tracing_ray_masks.cpp)
luisa_compute_add_ios_rendering_example(
        path_tracing_spectrum "Path Tracing Spectrum"
        rendering/path_tracing_spectrum.cpp)
luisa_compute_add_ios_rendering_example(
        path_tracing_xir2ast "Path Tracing XIR to AST"
        rendering/path_tracing_xir2ast.cpp)
luisa_compute_add_ios_rendering_example(
        photon_mapping "Photon Mapping"
        rendering/photon_mapping.cpp)
luisa_compute_add_ios_rendering_example(
        sdf_renderer "SDF Renderer"
        rendering/sdf_renderer.cpp)
luisa_compute_add_ios_rendering_example(
        sdf_renderer_xir2ast "SDF Renderer XIR to AST"
        rendering/sdf_renderer_xir2ast.cpp)
luisa_compute_add_ios_rendering_example(
        blackhole "Black Hole"
        rendering/blackhole.cpp)
luisa_compute_add_ios_rendering_example(
        voxel_raytracer "Voxel Ray Tracer"
        rendering/voxel_raytracer.cpp)
luisa_compute_add_ios_rendering_example(
        procedural "Procedural Ray Tracing"
        rendering/procedural.cpp)
luisa_compute_add_ios_rendering_example(
        shader_toy "Shader Toy"
        rendering/shader_toy.cpp)
luisa_compute_add_ios_rendering_example(
        shader_toy_spacex "Shader Toy SpaceX"
        rendering/shader_toy_spacex.cpp)
luisa_compute_add_ios_rendering_example(
        shader_visuals_present "Shader Visuals"
        rendering/shader_visuals_present.cpp)
luisa_compute_add_ios_rendering_example(
        coro_path_tracing "Coroutine Path Tracing"
        rendering/coro_path_tracing.cpp)
luisa_compute_add_ios_rendering_example(
        coro_sdf_renderer "Coroutine SDF Renderer"
        rendering/coro_sdf_renderer.cpp)

get_property(_LUISA_IOS_RENDERING_EXAMPLE_TARGETS
        GLOBAL PROPERTY LUISA_IOS_RENDERING_EXAMPLE_TARGETS)
add_custom_target(luisa-ios-rendering-examples
        DEPENDS example_ios_path_tracing
        ${_LUISA_IOS_RENDERING_EXAMPLE_TARGETS})
set_target_properties(luisa-ios-rendering-examples PROPERTIES
        FOLDER "examples/ios")
