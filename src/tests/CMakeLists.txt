add_executable(test_type test_type.cpp)
target_link_libraries(test_type PRIVATE luisa::compute)

add_executable(test_ast test_ast.cpp)
target_link_libraries(test_ast PRIVATE luisa::compute)

add_executable(test_dsl test_dsl.cpp)
target_link_libraries(test_dsl PRIVATE luisa::compute)

add_executable(test_dsl_multithread test_dsl_multithread.cpp)
target_link_libraries(test_dsl_multithread PRIVATE luisa::compute)

add_executable(test_dsl_sugar test_dsl_sugar.cpp)
target_link_libraries(test_dsl_sugar PRIVATE luisa::compute)

add_executable(test_runtime test_runtime.cpp)
target_link_libraries(test_runtime PRIVATE luisa::compute)

add_executable(test_simple test_simple.cpp)
target_link_libraries(test_simple PRIVATE luisa::compute)

add_executable(test_texture_io test_texture_io.cpp)
target_link_libraries(test_texture_io PRIVATE luisa::compute)

add_executable(test_texture test_texture.cpp)
target_link_libraries(test_texture PRIVATE luisa::compute)

add_executable(test_builtin test_builtin.cpp)
target_link_libraries(test_builtin PRIVATE luisa::compute)

add_executable(test_atomic test_atomic.cpp)
target_link_libraries(test_atomic PRIVATE luisa::compute)

add_executable(test_bindless test_bindless.cpp)
target_link_libraries(test_bindless PRIVATE luisa::compute)

add_executable(test_sampler test_sampler.cpp)
target_link_libraries(test_sampler PRIVATE luisa::compute)

add_executable(test_bindless_buffer test_bindless_buffer.cpp)
target_link_libraries(test_bindless_buffer PRIVATE luisa::compute)

add_executable(test_rtx test_rtx.cpp)
target_link_libraries(test_rtx PRIVATE luisa::compute)

add_executable(test_ispc_101 test_ispc_101.cpp)
target_link_libraries(test_ispc_101 PRIVATE luisa::compute)

add_executable(test_command_reorder test_command_reorder.cpp)
target_link_libraries(test_command_reorder PRIVATE luisa::compute)

add_executable(test_command_schedule test_command_schedule.cpp)
target_link_libraries(test_command_schedule PRIVATE luisa::compute)

if (LUISA_COMPUTE_ENABLE_GUI)
    add_executable(test_shader_toy test_shader_toy.cpp)
    target_link_libraries(test_shader_toy PRIVATE luisa::compute)
    add_executable(test_path_tracing test_path_tracing.cpp)
    target_link_libraries(test_path_tracing PRIVATE luisa::compute)
    add_executable(test_path_tracing_buffer test_path_tracing_buffer.cpp)
    target_link_libraries(test_path_tracing_buffer PRIVATE luisa::compute)
    add_executable(test_normal_encoding test_normal_encoding.cpp)
    target_link_libraries(test_normal_encoding PRIVATE luisa::compute)
    add_executable(test_mpm88 test_mpm88.cpp)
    target_link_libraries(test_mpm88 PRIVATE luisa::compute)
    add_executable(test_mpm3d test_mpm3d.cpp)
    target_link_libraries(test_mpm3d PRIVATE luisa::compute)
    if (APPLE)
        add_executable(test_shader_toy_present test_shader_toy_present.cpp)
        target_link_libraries(test_shader_toy_present PRIVATE luisa::compute)
        add_executable(test_shader_visuals_present test_shader_visuals_present.cpp)
        target_link_libraries(test_shader_visuals_present PRIVATE luisa::compute)
        add_executable(test_path_tracing_present test_path_tracing_present.cpp)
        target_link_libraries(test_path_tracing_present PRIVATE luisa::compute)
    endif ()
endif ()

add_executable(test_sdf_renderer test_sdf_renderer.cpp)
target_link_libraries(test_sdf_renderer PRIVATE luisa::compute)

find_package(OpenCV CONFIG)
if (OpenCV_FOUND)
    message(STATUS "OpenCV: ${OpenCV_LIBS}")
    target_link_libraries(test_sdf_renderer PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(test_sdf_renderer PRIVATE OPENCV_ENABLED)
    add_executable(test_game_of_life test_game_of_life.cpp)
    target_link_libraries(test_game_of_life PRIVATE luisa::compute ${OpenCV_LIBS})
    add_executable(test_sdf_renderer_multidev test_sdf_renderer_multidev.cpp)
    target_link_libraries(test_sdf_renderer_multidev PRIVATE luisa::compute ${OpenCV_LIBS})
endif ()

add_executable(test_playground test_playground.cpp)
target_link_libraries(test_playground PRIVATE luisa::compute)

add_executable(test_radiative_bp test_radiative_bp.cpp)
target_link_libraries(test_radiative_bp PRIVATE luisa::compute)

add_executable(test_image_byte4 test_image_byte4.cpp)
target_link_libraries(test_image_byte4 PRIVATE luisa::compute)
